"""
vLLM scheduler-side probes for BeyondKVTransfer — Milestone 2.

Instrumentation sites (DESIGN.md §5.1, §5.6):

  §5.1  arrival            vllm/v1/engine/processor.py
                           Processor.process_inputs
  §5.1  admit              vllm/v1/core/sched/scheduler.py
                           Scheduler.schedule  (waiting → running transition)
  §5.1  preempt            same  (preempted_req_ids in SchedulerOutput)
  §5.1  resume             same  (resumed reqs detected by state diff)
  §5.1  finish / abort     vllm/v1/engine/output_processor.py
                           OutputProcessor._free_request  (or equivalent)
  §5.6  scheduler_decision same Scheduler.schedule (post-call metadata)

Design constraints:
  * Every wrapper is a no-op (single boolean check) when BKVT_ENABLE=0.
  * All wrappers accept the original callable as an argument so they can
    be unit-tested with mock objects without a real vLLM installation.
  * API mismatches emit a one-time WARN rather than crashing the engine.
  * apply_patches() is idempotent: calling it twice does not double-patch.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable, Optional, Set

from bkvt import emitter as _emitter_mod
from bkvt.clock import now_ns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# One-time warning guard (§15.1 — API drift)
# ---------------------------------------------------------------------------

_warned: Set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("bkvt[vllm]: %s", msg)


# ---------------------------------------------------------------------------
# Per-request state tracker
# ---------------------------------------------------------------------------

class RequestStateTracker:
    """Thread-safe per-request lifecycle state (timestamps, flags).

    Shared across all probe modules so they can add/read timestamps to the
    same request record (e.g., arrival_ts_ns set by process_inputs probe,
    read later by the finish probe to compute ttft_ns).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def update(self, request_id: str, **kwargs: Any) -> None:
        """Merge keyword args into the state for request_id."""
        with self._lock:
            s = self._states.setdefault(request_id, {})
            s.update(kwargs)

    def set_if_absent(self, request_id: str, key: str, value: Any) -> bool:
        """Set key=value only if not already set.  Returns True when set."""
        with self._lock:
            s = self._states.setdefault(request_id, {})
            if key not in s:
                s[key] = value
                return True
            return False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, request_id: str) -> dict:
        """Return a shallow copy of the state for request_id."""
        with self._lock:
            return dict(self._states.get(request_id, {}))

    def pop(self, request_id: str) -> dict:
        """Remove and return the state for request_id."""
        with self._lock:
            return self._states.pop(request_id, {})

    def all_ids(self) -> Set[str]:
        """Return the set of currently tracked request IDs."""
        with self._lock:
            return set(self._states.keys())


# Module-level singleton — imported by other probe modules.
_tracker = RequestStateTracker()


def get_tracker() -> RequestStateTracker:
    """Return the shared RequestStateTracker singleton."""
    return _tracker


# ---------------------------------------------------------------------------
# Helpers for extracting IDs from vLLM objects
# ---------------------------------------------------------------------------

def _req_id(obj: Any) -> Optional[str]:
    """Try common vLLM attribute names for request ID."""
    for attr in ("req_id", "request_id", "id"):
        val = getattr(obj, attr, None)
        if isinstance(val, str):
            return val
    return None


def _req_ids_from_iterable(iterable: Any) -> list[str]:
    """Extract request IDs from an iterable of vLLM request/data objects."""
    ids = []
    if iterable is None:
        return ids
    for item in iterable:
        rid = _req_id(item) if not isinstance(item, str) else item
        if rid:
            ids.append(rid)
    return ids


def _get_running_ids(scheduler: Any) -> Set[str]:
    """Return the set of currently running request IDs from a Scheduler."""
    running = getattr(scheduler, "running", None)
    if running is None:
        return set()
    return {rid for rid in _req_ids_from_iterable(running) if rid}


def _get_waiting_ids(scheduler: Any) -> Set[str]:
    """Return the set of waiting request IDs from a Scheduler."""
    waiting = getattr(scheduler, "waiting", None)
    if waiting is None:
        return set()
    return {rid for rid in _req_ids_from_iterable(waiting) if rid}


# ---------------------------------------------------------------------------
# §5.1 — process_inputs wrapper (arrival)
# ---------------------------------------------------------------------------

def make_process_inputs_wrapper(original: Callable) -> Callable:
    """Wrap Processor.process_inputs to emit arrival records.

    Expected vLLM v1 signature (one common variant)::

        def process_inputs(self, request_id, prompt, params,
                           arrival_time=None, ...) -> EngineCoreRequest

    We accept *args/**kwargs so we tolerate signature drift.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)

        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return result

        ts = now_ns()

        # ── Extract request_id ───────────────────────────────────────────
        # It may be the first positional arg or a kwarg.
        request_id: Optional[str] = kwargs.get("request_id")
        if request_id is None and args:
            candidate = args[0]
            if isinstance(candidate, str):
                request_id = candidate
        # Fallback: the return value may carry it.
        if request_id is None:
            request_id = _req_id(result)
        if request_id is None:
            _warn_once("no_request_id_process_inputs",
                       "process_inputs: could not extract request_id; "
                       "arrival record skipped")
            return result

        # ── Extract input_len ────────────────────────────────────────────
        input_len: Optional[int] = None
        # result often has prompt_token_ids or input_tokens
        for attr in ("prompt_token_ids", "input_tokens", "token_ids"):
            tok = getattr(result, attr, None)
            if tok is not None:
                try:
                    input_len = len(tok)
                except TypeError:
                    pass
                break

        # ── Extract max_output_len ───────────────────────────────────────
        max_output_len: Optional[int] = None
        params = kwargs.get("params") or (args[2] if len(args) > 2 else None)
        if params is not None:
            for attr in ("max_tokens", "max_new_tokens", "n"):
                v = getattr(params, attr, None)
                if v is not None:
                    max_output_len = int(v)
                    break

        # ── Store state for later probes ─────────────────────────────────
        _tracker.update(request_id,
                        arrival_ts_ns=ts,
                        input_len=input_len,
                        max_output_len=max_output_len)

        # ── Emit arrival record ──────────────────────────────────────────
        em.event({
            "ts_ns": ts,
            "type": "request",
            "subtype": "arrival",
            "request_id": request_id,
            "input_len": input_len,
            "output_len_so_far": 0,
            "max_output_len": max_output_len,
            "arrival_ts_ns": ts,
        })

        return result

    return wrapper


# ---------------------------------------------------------------------------
# §5.1 / §5.6 — schedule wrapper (admit, preempt, resume, scheduler_decision)
# ---------------------------------------------------------------------------

def make_schedule_wrapper(original: Callable) -> Callable:
    """Wrap Scheduler.schedule to detect admits/preempts and emit records.

    vLLM v1 SchedulerOutput attributes we try (with fallbacks):
      - scheduled_new_reqs       → newly admitted requests
      - scheduled_cached_reqs    → admitted from prefix cache
      - preempted_req_ids        → preempted (frozenset[str])
      - resumed_req_ids          → resumed after preemption
      - finished_req_ids         → requests that finished this step
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()

        if not em.enabled:
            return original(self, *args, **kwargs)

        # ── Pre-call state snapshot ──────────────────────────────────────
        running_before = _get_running_ids(self)
        waiting_before = _get_waiting_ids(self)
        step_id: Optional[int] = getattr(self, "step_id",
                                         getattr(self, "_step_counter", None))

        # ── Forward call ─────────────────────────────────────────────────
        result = original(self, *args, **kwargs)

        ts = now_ns()

        # ── Extract admitted requests from SchedulerOutput ───────────────
        admitted_ids: list[str] = []
        for attr in ("scheduled_new_reqs", "new_req_data", "new_reqs"):
            new_reqs = getattr(result, attr, None)
            if new_reqs is not None:
                admitted_ids.extend(_req_ids_from_iterable(new_reqs))
                break

        # Also include cached requests that are admitted for the first time
        for attr in ("scheduled_cached_reqs", "cached_reqs"):
            cached = getattr(result, attr, None)
            if cached is not None:
                for rid in _req_ids_from_iterable(cached):
                    if rid not in running_before and rid not in admitted_ids:
                        admitted_ids.append(rid)
                break

        # ── Emit admit records ───────────────────────────────────────────
        for rid in admitted_ids:
            admit_ts = ts
            # Only record first_schedule_ts once per request
            _tracker.set_if_absent(rid, "first_schedule_ts_ns", admit_ts)
            state = _tracker.get(rid)
            em.event({
                "ts_ns": admit_ts,
                "type": "request",
                "subtype": "admit",
                "request_id": rid,
                "arrival_ts_ns": state.get("arrival_ts_ns"),
                "first_schedule_ts_ns": admit_ts,
                "input_len": state.get("input_len"),
            })

        # ── Emit preempt records ─────────────────────────────────────────
        preempted_ids: list[str] = []
        for attr in ("preempted_req_ids", "preempted", "preempted_reqs"):
            preempted = getattr(result, attr, None)
            if preempted is not None:
                preempted_ids.extend(_req_ids_from_iterable(preempted))
                break

        for rid in preempted_ids:
            em.event({
                "ts_ns": ts,
                "type": "request",
                "subtype": "preempt",
                "request_id": rid,
            })

        # ── Emit resume records ──────────────────────────────────────────
        resumed_ids: list[str] = []
        for attr in ("resumed_req_ids", "resumed", "resumed_reqs"):
            resumed = getattr(result, attr, None)
            if resumed is not None:
                resumed_ids.extend(_req_ids_from_iterable(resumed))
                break

        # Also detect resumes by state diff (was preempted, now running)
        running_after = _get_running_ids(self)
        for rid in running_after - running_before - set(admitted_ids):
            if rid not in resumed_ids:
                resumed_ids.append(rid)

        for rid in resumed_ids:
            em.event({
                "ts_ns": ts,
                "type": "request",
                "subtype": "resume",
                "request_id": rid,
            })

        # ── §5.6 scheduler_decision metadata record ──────────────────────
        waiting_after = _get_waiting_ids(self)

        # Try to extract block counts from kv_cache_manager / block_pool
        free_blocks: Optional[int] = None
        used_blocks: Optional[int] = None
        kv_mgr = getattr(self, "kv_cache_manager", None)
        if kv_mgr is not None:
            bp = getattr(kv_mgr, "block_pool", None)
            if bp is not None:
                fl = getattr(bp, "free_block_ids", None) or \
                     getattr(bp, "free_blocks", None)
                if fl is not None:
                    try:
                        free_blocks = len(fl)
                    except TypeError:
                        pass
        num_free = getattr(result, "num_free_gpu_blocks", free_blocks)
        num_used = getattr(result, "num_used_gpu_blocks", used_blocks)

        em.event({
            "ts_ns": ts,
            "type": "metadata",
            "subtype": "scheduler_decision",
            "step_id": step_id,
            "scheduler_inputs": {
                "running": len(running_before),
                "waiting": len(waiting_before),
            },
            "scheduler_outputs": {
                "running": len(running_after),
                "waiting": len(waiting_after),
                "admitted": admitted_ids,
                "preempted": preempted_ids,
                "resumed": resumed_ids,
                "free_blocks": num_free,
                "used_blocks": num_used,
            },
        })

        return result

    return wrapper


# ---------------------------------------------------------------------------
# §5.1 — finish / abort wrappers
# ---------------------------------------------------------------------------

def make_finish_request_wrapper(original: Callable) -> Callable:
    """Wrap the output-processor's request-finalisation call.

    Covers both normal finish and abort.  vLLM v1 has several candidate
    method names; patch.py picks the right one after probing the class.

    The wrapper detects abort vs. finish from:
      - a kwarg ``aborted=True``
      - the finish_reason on the output
      - presence of the request_id in a known-aborted set (if the engine
        sets that before calling here)
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        # Extract request_id before the call (request object may be freed)
        request_id: Optional[str] = None
        aborted: bool = kwargs.get("aborted", False)

        # Try to find request_id from first arg
        if args:
            first = args[0]
            if isinstance(first, str):
                request_id = first
            else:
                request_id = _req_id(first)
                # Check for abort signal
                finish_reason = getattr(first, "finish_reason", None)
                if finish_reason is not None:
                    aborted = str(finish_reason).lower() in ("abort", "aborted",
                                                              "cancelled", "canceled")

        result = original(self, *args, **kwargs)

        if request_id is None:
            return result

        ts = now_ns()
        state = _tracker.pop(request_id)

        arrival_ts = state.get("arrival_ts_ns")
        first_token_ts = state.get("first_token_ts_ns")
        first_schedule_ts = state.get("first_schedule_ts_ns")
        output_tokens = state.get("output_tokens_so_far", 0)

        ttft_ns: Optional[int] = None
        tpot_ns: Optional[int] = None
        if arrival_ts and first_token_ts and first_token_ts >= arrival_ts:
            ttft_ns = first_token_ts - arrival_ts
        if first_token_ts and output_tokens > 1 and ts > first_token_ts:
            tpot_ns = (ts - first_token_ts) // max(output_tokens - 1, 1)

        em.event({
            "ts_ns": ts,
            "type": "request",
            "subtype": "abort" if aborted else "finish",
            "request_id": request_id,
            "arrival_ts_ns": arrival_ts,
            "first_schedule_ts_ns": first_schedule_ts,
            "first_token_ts_ns": first_token_ts,
            "finish_ts_ns": ts,
            "ttft_ns": ttft_ns,
            "tpot_ns": tpot_ns,
            "output_len_so_far": output_tokens,
        })

        return result

    return wrapper


def make_abort_request_wrapper(original: Callable) -> Callable:
    """Wrap an explicit abort path (distinct from finish when vLLM has one)."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        request_id: Optional[str] = None
        if args:
            first = args[0]
            request_id = first if isinstance(first, str) else _req_id(first)

        result = original(self, *args, **kwargs)

        if request_id is None:
            return result

        ts = now_ns()
        state = _tracker.pop(request_id)

        em.event({
            "ts_ns": ts,
            "type": "request",
            "subtype": "abort",
            "request_id": request_id,
            "arrival_ts_ns": state.get("arrival_ts_ns"),
            "finish_ts_ns": ts,
        })

        return result

    return wrapper


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

_PATCHES_APPLIED: bool = False
_PATCHES_LOCK = threading.Lock()


def apply_patches() -> bool:
    """Monkey-patch vLLM scheduler/processor classes.

    Returns True if patches were applied, False if vLLM is not installed
    or patches were already applied.  Thread-safe and idempotent.
    """
    global _PATCHES_APPLIED

    with _PATCHES_LOCK:
        if _PATCHES_APPLIED:
            return True

        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return False

        applied = 0

        # ── Processor.process_inputs (arrival) ───────────────────────────
        try:
            from vllm.v1.engine import processor as _proc_mod  # type: ignore
            cls = _proc_mod.Processor
            orig = cls.process_inputs
            cls.process_inputs = make_process_inputs_wrapper(orig)
            logger.info("bkvt[vllm]: patched Processor.process_inputs")
            applied += 1
        except Exception as exc:
            _warn_once("proc_inputs", f"could not patch Processor.process_inputs: {exc}")

        # ── Scheduler.schedule (admit / preempt / scheduler_decision) ────
        try:
            from vllm.v1.core.sched import scheduler as _sched_mod  # type: ignore
            cls = _sched_mod.Scheduler
            orig = cls.schedule
            cls.schedule = make_schedule_wrapper(orig)
            logger.info("bkvt[vllm]: patched Scheduler.schedule")
            applied += 1
        except Exception as exc:
            _warn_once("sched_schedule", f"could not patch Scheduler.schedule: {exc}")

        # ── Output processor finish / abort ──────────────────────────────
        try:
            from vllm.v1.engine import output_processor as _op_mod  # type: ignore
            op_cls = _op_mod.OutputProcessor

            # Try the most likely method names in vLLM v1
            for mname in ("_free_request", "free_request", "_finish_request",
                          "finish_request"):
                orig = getattr(op_cls, mname, None)
                if orig is not None:
                    setattr(op_cls, mname, make_finish_request_wrapper(orig))
                    logger.info("bkvt[vllm]: patched OutputProcessor.%s", mname)
                    applied += 1
                    break
            else:
                _warn_once("op_finish",
                           "OutputProcessor: no finish/free method found; "
                           "finish/abort records will be missing")
        except Exception as exc:
            _warn_once("output_proc", f"could not patch OutputProcessor: {exc}")

        _PATCHES_APPLIED = applied > 0
        return _PATCHES_APPLIED
