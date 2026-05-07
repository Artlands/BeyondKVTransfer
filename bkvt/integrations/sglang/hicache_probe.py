"""
SGLang HiCache probes for BeyondKVTransfer -- Milestone 5.

Instrumentation sites (DESIGN.md sec. 6.5):

  HiCacheController.load_to_device / load    transfer load + tier_promote
  HiCacheController.backup_to_host / backup  transfer save + tier_demote
  HiCacheController L2/L3 variants           transfer load/save + tier_*
  HiRadixCache tier helpers                  kv_block tier_* visibility

The wrappers are intentionally version-tolerant.  SGLang's HiCache method
names have changed across releases, so this module patches every known method
name that exists and extracts record fields on a best-effort basis.
"""

from __future__ import annotations

import functools
import importlib
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


def _block_id(block: Any) -> Optional[int]:
    if isinstance(block, int):
        return block
    try:
        return int(block)
    except (TypeError, ValueError):
        pass
    for attr in ("block_id", "id", "idx", "index", "token_idx"):
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


def _first_present(args: tuple[Any, ...], kwargs: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return kwargs[name]
    for arg in args:
        for name in names:
            val = getattr(arg, name, None)
            if val is not None:
                return val
    return None


def _request_id_from(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[str]:
    value = _first_present(args, kwargs, ("request", "req", "request_id", "req_id", "rid"))
    rid = _req_id(value)
    if rid:
        return rid
    for arg in args:
        rid = _req_id(arg)
        if rid:
            return rid
    return None


def _block_ids_for(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any = None) -> list[int]:
    for name in ("block_ids", "blocks", "indices", "token_indices", "page_indices"):
        if name in kwargs:
            ids = _block_ids_from(kwargs[name])
            if ids:
                return ids
    for arg in args:
        ids = _block_ids_from(arg)
        if ids:
            return ids
    ids = _block_ids_from(result)
    return ids


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


def _tier_plan(method_name: str) -> tuple[str, str, str, str, str]:
    name = method_name.lower()
    if any(s in name for s in ("l3", "storage", "file", "disk", "mooncake")):
        if any(s in name for s in ("load", "get", "restore", "promote")):
            return "load", "OBJECT_STORE", "DRAM_LOCAL", "tier_promote", "hicache_promote"
        return "save", "DRAM_LOCAL", "OBJECT_STORE", "tier_demote", "hicache_demote"
    if any(s in name for s in ("load", "get", "restore", "promote", "to_device")):
        return "load", "DRAM_LOCAL", "HBM_LOCAL", "tier_promote", "hicache_promote"
    return "save", "HBM_LOCAL", "DRAM_LOCAL", "tier_demote", "hicache_demote"


def make_hicache_transfer_wrapper(original: Callable, method_name: str) -> Callable:
    """Wrap a HiCache controller movement method."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        direction, src_tier, dst_tier, kv_subtype, meta_subtype = _tier_plan(method_name)
        request_id = _request_id_from(args, kwargs)
        block_ids_before = _block_ids_for(args, kwargs)
        bytes_before = _bytes_for(args, kwargs)
        t_start = now_ns()
        tid = em.transfer_start(
            direction,
            request_id=request_id,
            src_tier=src_tier,
            dst_tier=dst_tier,
            transport="local_memcpy" if "OBJECT_STORE" not in (src_tier, dst_tier) else "file",
            num_blocks=len(block_ids_before) if block_ids_before else None,
            bytes_=bytes_before,
            block_ids=block_ids_before or None,
            issued_by="cache_controller",
            issued_at_phase="prefetch" if direction == "load" else "spillover",
            earliest_known_ts_ns=t_start,
            hicache_method=method_name,
        )
        ok = False
        try:
            result = original(self, *args, **kwargs)
            ok = True
            return result
        finally:
            t_end = now_ns()
            result_value = locals().get("result")
            block_ids = block_ids_before or _block_ids_for(args, kwargs, result_value)
            bytes_ = bytes_before if bytes_before is not None else _bytes_for(args, kwargs, result_value)
            em.transfer_end(
                tid,
                bytes_=bytes_,
                wire_time_ns=t_end - t_start,
                cancelled=not ok,
                block_ids=block_ids or None,
                hicache_method=method_name,
            )
            em.event({
                "ts_ns": t_start,
                "type": "metadata",
                "subtype": meta_subtype,
                "request_id": request_id,
                "duration_ns": t_end - t_start,
                "n_keys": len(block_ids),
                "tier_scope": f"{src_tier}->{dst_tier}",
                "hicache_method": method_name,
            })

            sampler = getattr(em, "_sampler", None)
            for block_id in block_ids:
                emit_ok, rate = (
                    sampler.should_emit_kv_block(kv_subtype) if sampler else (True, None)
                )
                if not emit_ok:
                    continue
                rec: dict = {
                    "ts_ns": t_end,
                    "type": "kv_block",
                    "subtype": kv_subtype,
                    "block_id": block_id,
                    "tier_before": src_tier,
                    "tier_after": dst_tier,
                    "owner_request_id": request_id,
                    "reason": "controller_promote" if kv_subtype == "tier_promote" else "capacity_evict",
                }
                if rate is not None:
                    rec["sample_decision"] = rate
                em.event(rec)

    return wrapper


def make_hiradix_tier_wrapper(original: Callable, method_name: str) -> Callable:
    """Wrap HiRadixCache tier bookkeeping methods when present."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        direction, src_tier, dst_tier, kv_subtype, meta_subtype = _tier_plan(method_name)
        t_start = now_ns()
        result = original(self, *args, **kwargs)
        t_end = now_ns()
        block_ids = _block_ids_for(args, kwargs, result)

        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": meta_subtype,
            "duration_ns": t_end - t_start,
            "n_keys": len(block_ids),
            "tier_scope": f"{src_tier}->{dst_tier}",
            "structure": "radix",
            "hicache_method": method_name,
        })

        sampler = getattr(em, "_sampler", None)
        for block_id in block_ids:
            emit_ok, rate = (
                sampler.should_emit_kv_block(kv_subtype) if sampler else (True, None)
            )
            if not emit_ok:
                continue
            rec: dict = {
                "ts_ns": t_end,
                "type": "kv_block",
                "subtype": kv_subtype,
                "block_id": block_id,
                "tier_before": src_tier,
                "tier_after": dst_tier,
                "reason": "controller_promote" if kv_subtype == "tier_promote" else "capacity_evict",
            }
            if rate is not None:
                rec["sample_decision"] = rate
            em.event(rec)
        return result

    return wrapper


_PATCHES_APPLIED = False
_PATCHES_LOCK = threading.Lock()


def _patch_method(cls: Any, method_name: str, wrapper_factory: Callable[[Callable, str], Callable]) -> bool:
    original = getattr(cls, method_name, None)
    if original is None:
        return False
    if getattr(original, "_bkvt_wrapped", False):
        return True
    wrapped = wrapper_factory(original, method_name)
    wrapped._bkvt_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapped)
    return True


def _patch_class_methods(module_name: str, class_name: str, names: tuple[str, ...], wrapper_factory: Callable[[Callable, str], Callable]) -> int:
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
    except Exception as exc:
        _warn_once(f"{module_name}.{class_name}", f"could not patch {class_name}: {exc}")
        return 0
    applied = 0
    for name in names:
        if _patch_method(cls, name, wrapper_factory):
            applied += 1
    return applied


def apply_patches() -> bool:
    """Patch SGLang HiCache / HiRadix cache movement sites."""
    global _PATCHES_APPLIED

    with _PATCHES_LOCK:
        if _PATCHES_APPLIED:
            return True

        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return False

        controller_methods = (
            "load", "load_to_device", "load_to_host", "load_from_storage",
            "backup", "backup_to_host", "backup_to_storage",
            "promote", "demote", "evict", "write_through",
        )
        hiradix_methods = (
            "load", "backup", "promote", "demote", "evict",
            "load_to_device", "backup_to_host",
        )

        applied = 0
        applied += _patch_class_methods(
            "sglang.srt.managers.cache_controller",
            "HiCacheController",
            controller_methods,
            make_hicache_transfer_wrapper,
        )
        applied += _patch_class_methods(
            "sglang.srt.mem_cache.hiradix_cache",
            "HiRadixCache",
            hiradix_methods,
            make_hiradix_tier_wrapper,
        )

        _PATCHES_APPLIED = applied > 0
        return _PATCHES_APPLIED
