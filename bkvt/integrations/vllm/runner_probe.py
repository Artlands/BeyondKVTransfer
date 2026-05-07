"""
vLLM GPU model-runner probes — Milestone 2.

Instrumentation sites (DESIGN.md §5.2):

  §5.2  prefill_chunk / decode   vllm/v1/worker/gpu_model_runner.py
                                  GPUModelRunner.execute_model
  §5.2  first_token               vllm/v1/engine/output_processor.py
                                  OutputProcessor (first decode output detection)

Design notes:
  * We bracket execute_model with a CudaEventTimer to capture
    kernel_start_ts_ns / kernel_end_ts_ns (§4.3).
  * Token records are sampled at BKVT_SAMPLE_TOKEN (default 0.05) with
    first_token always emitted (§9).
  * The runner sets a thread-local (threading.current_thread()._bkvt_ctx)
    so block-pool probes can read the current request_id.
  * step_id is read from the scheduler if accessible; otherwise a
    per-process monotonic counter is used.
  * apply_patches() is idempotent and BKVT_ENABLE-gated.
"""

from __future__ import annotations

import functools
import itertools
import logging
import threading
from typing import Any, Callable, Optional, Set

from bkvt import emitter as _emitter_mod
from bkvt.clock import CudaEventTimer, now_ns
from bkvt.integrations.vllm.scheduler_probe import get_tracker

logger = logging.getLogger(__name__)

_warned: Set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("bkvt[vllm]: %s", msg)


# ---------------------------------------------------------------------------
# Per-process step counter (fallback when Scheduler.step_id unavailable)
# ---------------------------------------------------------------------------

_step_counter = itertools.count(0)


def _next_step_id() -> int:
    return next(_step_counter)


# ---------------------------------------------------------------------------
# Thread-local context (used by block_pool_probe to read current request_id)
# ---------------------------------------------------------------------------

def _set_tls_request_id(request_id: Optional[str]) -> None:
    t = threading.current_thread()
    ctx = getattr(t, "_bkvt_ctx", None)
    if ctx is None:
        ctx = threading.local()
        t._bkvt_ctx = ctx  # type: ignore[attr-defined]
    ctx.request_id = request_id  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# §5.2 — GPUModelRunner.execute_model wrapper
# ---------------------------------------------------------------------------

def make_execute_model_wrapper(original: Callable) -> Callable:
    """Wrap GPUModelRunner.execute_model to emit token records.

    The method receives a ``SchedulerOutput`` (or equivalent) and returns
    a ``ModelRunnerOutput`` containing per-request sampled token IDs.

    What we extract from ``scheduler_output``:
      * ``num_scheduled_tokens``      dict[request_id → int_token_count]
      * ``scheduled_new_reqs``        newly admitted (→ prefill)
      * ``scheduled_running_reqs`` /  continuing decode requests
        ``scheduled_cached_reqs``

    What we extract from ``model_output``:
      * ``outputs``                   list of RequestOutput
        - each has ``request_id``, ``output_token_ids`` or ``new_token``

    When exact attributes are missing we fall back to estimating the
    subtype from whether each request is in its first step.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        scheduler_output = (
            args[0] if args else kwargs.get("scheduler_output",
                                            kwargs.get("model_input"))
        )

        # ── Step identity ────────────────────────────────────────────────
        step_id: int = _next_step_id()

        # ── CUDA event bracket ───────────────────────────────────────────
        cuda_timer = CudaEventTimer()
        cuda_timer.record_start()

        result = original(self, *args, **kwargs)

        cuda_timer.record_end()

        t_end = now_ns()
        kernel_start = cuda_timer.start_ns()
        kernel_end = cuda_timer.end_ns()
        cuda_event_ts = cuda_timer.cuda_event_ts_ns()

        # ── Determine which requests were prefill vs. decode ─────────────
        prefill_ids: Set[str] = set()
        if scheduler_output is not None:
            for attr in ("scheduled_new_reqs", "new_req_data", "new_reqs"):
                new_reqs = getattr(scheduler_output, attr, None)
                if new_reqs is not None:
                    for req in new_reqs:
                        rid = _req_id(req)
                        if rid:
                            prefill_ids.add(rid)
                    break

        # ── num_scheduled_tokens: dict[req_id → int] ────────────────────
        num_tokens_map: dict[str, int] = {}
        nst = getattr(scheduler_output, "num_scheduled_tokens", None)
        if isinstance(nst, dict):
            num_tokens_map = nst
        elif nst is not None:
            # May be an int (total), not per-request; skip
            pass

        # ── Tracker for per-request state ────────────────────────────────
        tracker = get_tracker()

        # ── Emit token records from model outputs ────────────────────────
        outputs = getattr(result, "outputs", None)
        if outputs is None:
            # Some vLLM versions embed outputs differently
            outputs = getattr(result, "sampled_token_ids", None)
            if outputs is not None:
                # It's a 2-D tensor; we can't extract request_ids here
                outputs = None

        sampler = getattr(em, "_sampler", None)

        if outputs is not None:
            for req_out in outputs:
                request_id = _req_id(req_out)
                if request_id is None:
                    continue

                # Determine subtype
                is_prefill = request_id in prefill_ids
                subtype = "prefill_chunk" if is_prefill else "decode"

                # Count tokens produced this step
                new_tokens_out = getattr(req_out, "output_token_ids",
                                         getattr(req_out, "new_token_ids",
                                                 getattr(req_out, "token_ids", None)))
                if new_tokens_out is not None:
                    try:
                        n_produced = len(new_tokens_out)
                    except TypeError:
                        n_produced = 1
                else:
                    n_produced = num_tokens_map.get(request_id, 1)

                # Update output token count
                prev_output = tracker.get(request_id).get("output_tokens_so_far", 0)
                tracker.update(request_id,
                               output_tokens_so_far=prev_output + n_produced)

                # ── first_token detection ────────────────────────────────
                is_first_token = tracker.set_if_absent(request_id,
                                                       "first_token_ts_ns",
                                                       t_end)
                if is_first_token:
                    # first_token is always emitted (§9)
                    em.event({
                        "ts_ns": t_end,
                        "type": "token",
                        "subtype": "first_token",
                        "request_id": request_id,
                        "step_id": step_id,
                        "token_idx": prev_output,
                        "num_tokens": n_produced,
                        "kernel_start_ts_ns": kernel_start,
                        "kernel_end_ts_ns": kernel_end,
                        "cuda_event_ts_ns": cuda_event_ts,
                    })
                    # Don't also emit a decode record for first token
                    continue

                # ── Sampled token record ─────────────────────────────────
                emit_ok, rate = (sampler.should_emit_token(subtype)
                                 if sampler else (True, None))
                if not emit_ok:
                    continue

                token_idx = (num_tokens_map.get(request_id, 0)
                             + tracker.get(request_id).get("output_tokens_so_far", 0)
                             - n_produced)
                num_prefill = num_tokens_map.get(request_id, 0) if is_prefill else 0

                rec: dict = {
                    "ts_ns": t_end,
                    "type": "token",
                    "subtype": subtype,
                    "request_id": request_id,
                    "step_id": step_id,
                    "token_idx": token_idx,
                    "num_tokens": n_produced,
                    "num_prefill_tokens": num_prefill,
                    "kernel_start_ts_ns": kernel_start,
                    "kernel_end_ts_ns": kernel_end,
                    "cuda_event_ts_ns": cuda_event_ts,
                }
                if rate is not None:
                    rec["sample_decision"] = rate
                em.event(rec)

        else:
            # ── Fallback: emit one aggregate token record per step ────────
            # when we can't decompose by request
            all_ids = list(num_tokens_map.keys())
            total_tokens = sum(num_tokens_map.values()) or 1

            emit_ok, rate = (sampler.should_emit_token("decode")
                             if sampler else (True, None))
            if emit_ok and all_ids:
                rec = {
                    "ts_ns": t_end,
                    "type": "token",
                    "subtype": "decode",
                    "step_id": step_id,
                    "token_idx": -1,
                    "num_tokens": total_tokens,
                    "kernel_start_ts_ns": kernel_start,
                    "kernel_end_ts_ns": kernel_end,
                }
                if rate is not None:
                    rec["sample_decision"] = rate
                em.event(rec)

        return result

    return wrapper


def _req_id(obj: Any) -> Optional[str]:
    """Try common vLLM attribute names for request ID."""
    for attr in ("req_id", "request_id", "id"):
        v = getattr(obj, attr, None)
        if isinstance(v, str):
            return v
    return None


# ---------------------------------------------------------------------------
# §5.2 — first_token detection in OutputProcessor (alternative hook)
# ---------------------------------------------------------------------------

def make_output_processor_first_token_wrapper(original: Callable) -> Callable:
    """Wrap the per-step output-processing call to detect first tokens.

    This is a secondary hook for first_token — the runner wrapper is the
    primary one.  This wrapper only emits if the primary hook missed it
    (i.e. ``first_token_ts_ns`` not yet set in the tracker).
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)

        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return result

        tracker = get_tracker()
        ts = now_ns()

        # result is typically a list of CompletionOutput or similar
        outputs = result if isinstance(result, (list, tuple)) else (
            getattr(result, "outputs", []) or []
        )
        for out in outputs:
            request_id = _req_id(out)
            if request_id is None:
                continue
            is_first = tracker.set_if_absent(request_id, "first_token_ts_ns", ts)
            if is_first:
                em.event({
                    "ts_ns": ts,
                    "type": "token",
                    "subtype": "first_token",
                    "request_id": request_id,
                    "step_id": -1,
                    "token_idx": 0,
                    "num_tokens": 1,
                })

        return result

    return wrapper


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

_PATCHES_APPLIED: bool = False
_PATCHES_LOCK = threading.Lock()


def apply_patches() -> bool:
    """Monkey-patch GPUModelRunner.execute_model.

    Returns True if any patches were applied.  Idempotent.
    """
    global _PATCHES_APPLIED

    with _PATCHES_LOCK:
        if _PATCHES_APPLIED:
            return True

        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return False

        applied = 0

        # ── GPUModelRunner.execute_model ─────────────────────────────────
        try:
            from vllm.v1.worker import gpu_model_runner as _gmr_mod  # type: ignore
            cls = _gmr_mod.GPUModelRunner

            for mname in ("execute_model", "forward", "run_model"):
                orig = getattr(cls, mname, None)
                if orig is not None:
                    setattr(cls, mname, make_execute_model_wrapper(orig))
                    logger.info("bkvt[vllm]: patched GPUModelRunner.%s", mname)
                    applied += 1
                    break
            else:
                _warn_once("runner_exec",
                           "GPUModelRunner: no execute_model method found")
        except Exception as exc:
            _warn_once("runner", f"could not patch GPUModelRunner: {exc}")

        # ── OutputProcessor — secondary first_token hook ─────────────────
        try:
            from vllm.v1.engine import output_processor as _op_mod  # type: ignore
            op_cls = _op_mod.OutputProcessor

            for mname in ("process_outputs", "_process_model_outputs",
                          "step"):
                orig = getattr(op_cls, mname, None)
                if orig is not None:
                    setattr(op_cls, mname,
                            make_output_processor_first_token_wrapper(orig))
                    applied += 1
                    break
        except Exception:
            pass  # secondary hook; silent failure is fine

        _PATCHES_APPLIED = applied > 0
        return _PATCHES_APPLIED
