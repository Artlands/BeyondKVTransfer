"""
SGLang scheduler-side probes for BeyondKVTransfer -- Milestone 4.

Instrumentation sites (DESIGN.md sec. 6.1, 6.6):

  arrival              Scheduler.handle_generate_request
  admit/preempt/resume Scheduler.run_batch
  finish/abort         Scheduler.handle_finished_requests
  scheduler_decision   Scheduler.run_batch

The wrappers are deliberately SGLang-version tolerant: they accept
*args/**kwargs, use best-effort attribute extraction, and emit a one-time
warning rather than breaking the engine when an upstream API changes.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable, Optional, Set

from bkvt import emitter as _emitter_mod
from bkvt.clock import now_ns

logger = logging.getLogger(__name__)

_warned: Set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("bkvt[sglang]: %s", msg)


class RequestStateTracker:
    """Thread-safe lifecycle state shared by SGLang scheduler probes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, dict] = {}

    def update(self, request_id: str, **kwargs: Any) -> None:
        with self._lock:
            state = self._states.setdefault(request_id, {})
            state.update(kwargs)

    def set_if_absent(self, request_id: str, key: str, value: Any) -> bool:
        with self._lock:
            state = self._states.setdefault(request_id, {})
            if key in state:
                return False
            state[key] = value
            return True

    def get(self, request_id: str) -> dict:
        with self._lock:
            return dict(self._states.get(request_id, {}))

    def pop(self, request_id: str) -> dict:
        with self._lock:
            return self._states.pop(request_id, {})


_tracker = RequestStateTracker()


def get_tracker() -> RequestStateTracker:
    return _tracker


def _req_id(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        return obj
    for attr in ("rid", "request_id", "req_id", "id"):
        val = getattr(obj, attr, None)
        if isinstance(val, str):
            return val
    return None


def _as_list(obj: Any) -> list:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return list(obj.values())
    if isinstance(obj, (str, bytes)):
        return [obj]
    try:
        return list(obj)
    except TypeError:
        return [obj]


def _req_ids_from_iterable(obj: Any) -> list[str]:
    ids: list[str] = []
    for item in _as_list(obj):
        rid = _req_id(item)
        if rid:
            ids.append(rid)
    return ids


def _batch_reqs(batch: Any) -> list:
    if batch is None:
        return []
    for attr in ("reqs", "requests", "batch_reqs"):
        val = getattr(batch, attr, None)
        if val is not None:
            return _as_list(val)
    return _as_list(batch)


def _get_running_ids(scheduler: Any) -> Set[str]:
    for attr in ("running_batch", "cur_batch", "batch"):
        batch = getattr(scheduler, attr, None)
        ids = set(_req_ids_from_iterable(_batch_reqs(batch)))
        if ids:
            return ids
    running = getattr(scheduler, "running", None)
    return set(_req_ids_from_iterable(running))


def _get_waiting_ids(scheduler: Any) -> Set[str]:
    for attr in ("waiting_queue", "req_queue", "waiting", "recv_reqs"):
        queue = getattr(scheduler, attr, None)
        ids = set(_req_ids_from_iterable(queue))
        if ids:
            return ids
    return set()


def _input_len(req: Any) -> Optional[int]:
    for attr in ("origin_input_ids", "input_ids", "prompt_token_ids", "input_tokens"):
        val = getattr(req, attr, None)
        if val is not None:
            try:
                return len(val)
            except TypeError:
                return None
    return None


def _max_output_len(req: Any) -> Optional[int]:
    params = getattr(req, "sampling_params", None)
    for obj in (req, params):
        if obj is None:
            continue
        for attr in ("max_new_tokens", "max_tokens", "max_output_len"):
            val = getattr(obj, attr, None)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return None
    return None


def _scheduler_size(obj: Any) -> Optional[int]:
    if obj is None:
        return None
    try:
        return len(obj)
    except TypeError:
        return None


def make_handle_generate_request_wrapper(original: Callable) -> Callable:
    """Wrap Scheduler.handle_generate_request to emit request/arrival."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)

        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return result

        req = args[0] if args else kwargs.get("request")
        request_id = _req_id(req) or _req_id(result)
        if request_id is None:
            _warn_once(
                "arrival_request_id",
                "handle_generate_request: could not extract request_id; arrival skipped",
            )
            return result

        ts = now_ns()
        input_len = _input_len(req) if req is not None else _input_len(result)
        max_output_len = _max_output_len(req) if req is not None else _max_output_len(result)
        priority = getattr(req, "priority", None) if req is not None else None

        _tracker.update(
            request_id,
            arrival_ts_ns=ts,
            input_len=input_len,
            max_output_len=max_output_len,
        )

        em.event({
            "ts_ns": ts,
            "type": "request",
            "subtype": "arrival",
            "request_id": request_id,
            "input_len": input_len,
            "output_len_so_far": 0,
            "max_output_len": max_output_len,
            "priority": priority,
            "arrival_ts_ns": ts,
        })
        return result

    return wrapper


def make_run_batch_wrapper(original: Callable) -> Callable:
    """Wrap Scheduler.run_batch for admits, preempts, resumes, and Q6."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        running_before = _get_running_ids(self)
        waiting_before = _get_waiting_ids(self)
        step_id = getattr(self, "step_id", getattr(self, "forward_ct", None))

        result = original(self, *args, **kwargs)
        ts = now_ns()

        running_after = _get_running_ids(self)
        waiting_after = _get_waiting_ids(self)

        admitted_ids = []
        for attr in ("new_reqs", "can_run_list", "admitted_reqs", "scheduled_reqs"):
            val = getattr(result, attr, None)
            if val is not None:
                admitted_ids.extend(_req_ids_from_iterable(val))
                break
        if not admitted_ids:
            admitted_ids.extend(sorted(running_after - running_before))

        preempted_ids = []
        for attr in ("preempted_reqs", "preempted_req_ids", "retracted_reqs"):
            val = getattr(result, attr, None)
            if val is not None:
                preempted_ids.extend(_req_ids_from_iterable(val))
                break
        if not preempted_ids:
            preempted_ids.extend(sorted(running_before - running_after - set(admitted_ids)))

        resumed_ids = []
        for attr in ("resumed_reqs", "resumed_req_ids"):
            val = getattr(result, attr, None)
            if val is not None:
                resumed_ids.extend(_req_ids_from_iterable(val))
                break

        for rid in admitted_ids:
            _tracker.set_if_absent(rid, "first_schedule_ts_ns", ts)
            state = _tracker.get(rid)
            em.event({
                "ts_ns": ts,
                "type": "request",
                "subtype": "admit",
                "request_id": rid,
                "arrival_ts_ns": state.get("arrival_ts_ns"),
                "first_schedule_ts_ns": ts,
                "input_len": state.get("input_len"),
            })

        for rid in preempted_ids:
            em.event({
                "ts_ns": ts,
                "type": "request",
                "subtype": "preempt",
                "request_id": rid,
            })

        for rid in resumed_ids:
            em.event({
                "ts_ns": ts,
                "type": "request",
                "subtype": "resume",
                "request_id": rid,
            })

        token_pool = getattr(self, "token_to_kv_pool", None)
        free_blocks = getattr(token_pool, "available_size", None)
        if callable(free_blocks):
            try:
                free_blocks = free_blocks()
            except Exception:
                free_blocks = None

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
                "free_blocks": free_blocks,
                "waiting_queue_size": _scheduler_size(getattr(self, "waiting_queue", None)),
            },
        })
        return result

    return wrapper


def make_handle_finished_requests_wrapper(original: Callable) -> Callable:
    """Wrap Scheduler.handle_finished_requests to emit finish/abort records."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        candidates = []
        if args:
            candidates.extend(_as_list(args[0]))
        for key in ("finished_reqs", "finished_requests", "reqs"):
            if key in kwargs:
                candidates.extend(_as_list(kwargs[key]))

        result = original(self, *args, **kwargs)
        if not candidates:
            for attr in ("finished_reqs", "finished_requests"):
                candidates.extend(_as_list(getattr(result, attr, None)))

        ts = now_ns()
        seen: set[str] = set()
        for req in candidates:
            rid = _req_id(req)
            if not rid or rid in seen:
                continue
            seen.add(rid)
            state = _tracker.pop(rid)

            aborted = bool(getattr(req, "aborted", False))
            finish_reason = getattr(req, "finish_reason", None)
            if finish_reason is not None:
                aborted = aborted or str(finish_reason).lower() in {
                    "abort", "aborted", "cancelled", "canceled"
                }

            first_token_ts = state.get("first_token_ts_ns")
            arrival_ts = state.get("arrival_ts_ns")
            output_tokens = getattr(req, "completion_tokens", None)
            if output_tokens is None:
                output_tokens = getattr(req, "output_len", state.get("output_tokens_so_far", 0))

            ttft_ns = None
            if arrival_ts and first_token_ts and first_token_ts >= arrival_ts:
                ttft_ns = first_token_ts - arrival_ts
            tpot_ns = None
            if first_token_ts and output_tokens and output_tokens > 1 and ts > first_token_ts:
                tpot_ns = (ts - first_token_ts) // max(int(output_tokens) - 1, 1)

            em.event({
                "ts_ns": ts,
                "type": "request",
                "subtype": "abort" if aborted else "finish",
                "request_id": rid,
                "arrival_ts_ns": arrival_ts,
                "first_schedule_ts_ns": state.get("first_schedule_ts_ns"),
                "first_token_ts_ns": first_token_ts,
                "finish_ts_ns": ts,
                "ttft_ns": ttft_ns,
                "tpot_ns": tpot_ns,
                "output_len_so_far": output_tokens,
            })

        return result

    return wrapper


_PATCHES_APPLIED = False
_PATCHES_LOCK = threading.Lock()


def _patch_method(cls: Any, method_name: str, wrapper_factory: Callable[[Callable], Callable]) -> bool:
    original = getattr(cls, method_name, None)
    if original is None:
        return False
    if getattr(original, "_bkvt_wrapped", False):
        return True
    wrapped = wrapper_factory(original)
    wrapped._bkvt_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapped)
    return True


def apply_patches() -> bool:
    """Patch SGLang scheduler methods. Returns True if any site was patched."""
    global _PATCHES_APPLIED

    with _PATCHES_LOCK:
        if _PATCHES_APPLIED:
            return True

        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return False

        try:
            from sglang.srt.managers import scheduler as _sched_mod  # type: ignore
            cls = _sched_mod.Scheduler
        except Exception as exc:
            _warn_once("scheduler_import", f"could not import SGLang Scheduler: {exc}")
            return False

        applied = 0
        if _patch_method(cls, "handle_generate_request", make_handle_generate_request_wrapper):
            applied += 1
        if _patch_method(cls, "run_batch", make_run_batch_wrapper):
            applied += 1
        if _patch_method(cls, "handle_finished_requests", make_handle_finished_requests_wrapper):
            applied += 1

        _PATCHES_APPLIED = applied > 0
        return _PATCHES_APPLIED
