"""
SGLang PD-disaggregation transfer probes for BeyondKVTransfer -- Milestone 5.

Instrumentation sites (DESIGN.md sec. 6.4):

  disaggregation/prefill.py dispatcher methods  transfer save
  disaggregation/decode.py reception methods     transfer load
  disaggregation/mooncake/conn.py send/recv      backend transfer timing
  disaggregation/nixl/* transfer calls           backend transfer timing

SGLang has changed class and method names across PD-disagg releases.  This
module therefore patches public functions and class methods with known transfer
verbs when their module is importable, while keeping missing sites non-fatal.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import threading
from types import ModuleType
from typing import Any, Callable, Optional, Set

from bkvt import emitter as _emitter_mod
from bkvt.clock import now_ns

logger = logging.getLogger(__name__)

_warned: Set[str] = set()

_PREFILL_VERBS = (
    "dispatch", "dispatch_kv", "send", "send_kv", "send_kv_cache",
    "transfer", "transfer_kv", "push", "put",
)
_DECODE_VERBS = (
    "recv", "recv_kv", "recv_kv_cache", "receive", "receive_kv",
    "fetch", "get", "pull",
)
_BACKEND_VERBS = _PREFILL_VERBS + _DECODE_VERBS + ("check_xfer_status", "poll", "wait")


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("bkvt[sglang]: %s", msg)


def _to_list(obj: Any) -> list:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return list(obj.values())
    if isinstance(obj, (str, bytes)):
        return [obj]
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            obj = tolist()
        except Exception:
            pass
    try:
        return list(obj)
    except TypeError:
        return [obj]


def _flatten(obj: Any) -> list:
    out: list = []
    for item in _to_list(obj):
        if isinstance(item, (list, tuple)):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out


def _req_id(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        return obj
    for attr in ("rid", "request_id", "req_id", "id"):
        val = getattr(obj, attr, None)
        if isinstance(val, str):
            return val
    return None


def _request_id_from(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[str]:
    for name in ("request", "req", "request_id", "req_id", "rid"):
        if name in kwargs:
            rid = _req_id(kwargs[name])
            if rid:
                return rid
    for arg in args:
        rid = _req_id(arg)
        if rid:
            return rid
    return None


def _block_id(block: Any) -> Optional[int]:
    if isinstance(block, int):
        return block
    try:
        return int(block)
    except (TypeError, ValueError):
        pass
    for attr in ("block_id", "id", "idx", "index", "page_id"):
        val = getattr(block, attr, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                return None
    return None


def _block_ids_from(value: Any) -> list[int]:
    ids: list[int] = []
    for item in _flatten(value):
        bid = _block_id(item)
        if bid is not None:
            ids.append(bid)
    return ids


def _block_ids_for(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any = None) -> list[int]:
    for name in ("block_ids", "blocks", "kv_indices", "indices", "page_indices"):
        if name in kwargs:
            ids = _block_ids_from(kwargs[name])
            if ids:
                return ids
    for arg in args:
        ids = _block_ids_from(arg)
        if ids:
            return ids
    return _block_ids_from(result)


def _bytes_from(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    for attr in ("nbytes", "num_bytes", "bytes", "size_bytes"):
        val = getattr(value, attr, None)
        if isinstance(val, int):
            return val
    return None


def _bytes_for(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any = None) -> Optional[int]:
    for name in ("bytes", "bytes_", "num_bytes", "size_bytes"):
        if name in kwargs:
            val = _bytes_from(kwargs[name])
            if val is not None:
                return val
    for arg in args:
        val = _bytes_from(arg)
        if val is not None:
            return val
    return _bytes_from(result)


def _direction_for(method_name: str, role_hint: str) -> str:
    name = method_name.lower()
    if role_hint == "decode" or any(v in name for v in _DECODE_VERBS):
        return "load"
    return "save"


def _transport_for(module_name: str, owner: Any = None) -> str:
    text = module_name.lower()
    if owner is not None:
        text += f".{owner.__class__.__module__}.{owner.__class__.__name__}".lower()
    if "nixl" in text:
        return "nixl"
    if "mooncake" in text:
        return "tcp"
    if "file" in text or "storage" in text:
        return "file"
    return "tcp"


def _endpoints(direction: str, transport: str) -> tuple[str, str]:
    if direction == "load":
        src = "HBM_PEER_RDMA" if transport == "nixl" else "DRAM_REMOTE"
        return src, "HBM_LOCAL"
    dst = "HBM_PEER_RDMA" if transport == "nixl" else "DRAM_REMOTE"
    return "HBM_LOCAL", dst


def make_disagg_transfer_wrapper(
    original: Callable,
    method_name: str,
    *,
    module_name: str,
    role_hint: str,
) -> Callable:
    """Wrap a PD-disagg transfer function or method."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(*args, **kwargs)

        owner = args[0] if args else None
        direction = _direction_for(method_name, role_hint)
        transport = _transport_for(module_name, owner)
        src_tier, dst_tier = _endpoints(direction, transport)
        request_id = _request_id_from(args, kwargs)
        block_ids_before = _block_ids_for(args, kwargs)
        bytes_before = _bytes_for(args, kwargs)
        t_start = now_ns()
        tid = em.transfer_start(
            direction,
            request_id=request_id,
            src_tier=src_tier,
            dst_tier=dst_tier,
            transport=transport,
            num_blocks=len(block_ids_before) if block_ids_before else None,
            bytes_=bytes_before,
            block_ids=block_ids_before or None,
            issued_by="connector",
            issued_at_phase="prefill" if direction == "save" else "decode",
            earliest_known_ts_ns=t_start,
            disagg_method=method_name,
            disagg_module=module_name,
        )
        ok = False
        try:
            result = original(*args, **kwargs)
            ok = True
            return result
        finally:
            t_end = now_ns()
            result_value = locals().get("result")
            block_ids = block_ids_before or _block_ids_for(args, kwargs, result_value)
            bytes_ = bytes_before if bytes_before is not None else _bytes_for(args, kwargs, result_value)
            wr_count = None
            if result_value is not None:
                descriptors = getattr(result_value, "descriptors", None)
                try:
                    wr_count = len(descriptors) if descriptors is not None else None
                except TypeError:
                    wr_count = None
            em.transfer_end(
                tid,
                bytes_=bytes_,
                wire_time_ns=t_end - t_start,
                wr_count=wr_count,
                cancelled=not ok,
                block_ids=block_ids or None,
                disagg_method=method_name,
                disagg_module=module_name,
            )
            if "nixl" in module_name.lower():
                em.event({
                    "ts_ns": t_start,
                    "type": "metadata",
                    "subtype": "nixl_call",
                    "request_id": request_id,
                    "duration_ns": t_end - t_start,
                    "n_keys": len(block_ids),
                    "disagg_method": method_name,
                })

    return wrapper


_PATCHES_APPLIED = False
_PATCHES_LOCK = threading.Lock()


def _is_patchable_function(obj: Any) -> bool:
    return inspect.isfunction(obj) or inspect.ismethod(obj)


def _patch_attr(
    owner: Any,
    attr_name: str,
    *,
    module_name: str,
    role_hint: str,
) -> bool:
    original = getattr(owner, attr_name, None)
    if original is None or not _is_patchable_function(original):
        return False
    if getattr(original, "_bkvt_wrapped", False):
        return True
    wrapped = make_disagg_transfer_wrapper(
        original,
        attr_name,
        module_name=module_name,
        role_hint=role_hint,
    )
    wrapped._bkvt_wrapped = True  # type: ignore[attr-defined]
    setattr(owner, attr_name, wrapped)
    return True


def _patch_module(module: ModuleType, module_name: str, role_hint: str, verbs: tuple[str, ...]) -> int:
    applied = 0
    for name in verbs:
        if _patch_attr(module, name, module_name=module_name, role_hint=role_hint):
            applied += 1

    for _name, obj in inspect.getmembers(module, inspect.isclass):
        obj_module = getattr(obj, "__module__", "")
        if obj_module and not obj_module.startswith(module.__name__):
            continue
        for verb in verbs:
            if _patch_attr(obj, verb, module_name=module_name, role_hint=role_hint):
                applied += 1
    return applied


def _import_and_patch(module_name: str, role_hint: str, verbs: tuple[str, ...]) -> int:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        _warn_once(module_name, f"could not patch {module_name}: {exc}")
        return 0
    return _patch_module(module, module_name, role_hint, verbs)


def apply_patches() -> bool:
    """Patch SGLang PD-disaggregation modules and backend transfer calls."""
    global _PATCHES_APPLIED

    with _PATCHES_LOCK:
        if _PATCHES_APPLIED:
            return True

        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return False

        applied = 0
        applied += _import_and_patch(
            "sglang.srt.disaggregation.prefill",
            "prefill",
            _PREFILL_VERBS,
        )
        applied += _import_and_patch(
            "sglang.srt.disaggregation.decode",
            "decode",
            _DECODE_VERBS,
        )
        applied += _import_and_patch(
            "sglang.srt.disaggregation.mooncake.conn",
            "backend",
            _BACKEND_VERBS,
        )

        for module_name in (
            "sglang.srt.disaggregation.nixl",
            "sglang.srt.disaggregation.nixl.conn",
            "sglang.srt.disaggregation.nixl.transfer",
        ):
            applied += _import_and_patch(module_name, "backend", _BACKEND_VERBS)

        _PATCHES_APPLIED = applied > 0
        return _PATCHES_APPLIED
