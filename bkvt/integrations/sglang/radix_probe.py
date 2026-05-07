"""
SGLang radix-cache and allocator probes for BeyondKVTransfer -- M4.

Instrumentation sites (DESIGN.md sec. 6.3):

  RadixCache.match_prefix       metadata/prefix_lookup, kv_block/prefix_hit
  RadixCache.insert             metadata/prefix_insert, kv_block/hash_insert
  RadixCache.evict              metadata/evict_select, kv_block/evict
  RadixCache.inc_lock_ref       metadata/refcount_inc
  RadixCache.dec_lock_ref       metadata/refcount_dec
  BaseTokenToKVPoolAllocator.*  metadata/allocator_*, kv_block allocate/free
  ReqToTokenPool.write          metadata/block_table_update
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


def _req_id(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        return obj
    for attr in ("rid", "request_id", "req_id", "id"):
        val = getattr(obj, attr, None)
        if isinstance(val, str):
            return val
    return None


def _to_list(obj: Any) -> list:
    if obj is None:
        return []
    if isinstance(obj, tuple) and obj:
        # SGLang match_prefix commonly returns (indices, last_node).
        obj = obj[0]
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


def _block_id(block: Any) -> Optional[int]:
    if isinstance(block, int):
        return block
    try:
        return int(block)
    except (TypeError, ValueError):
        pass
    for attr in ("block_id", "id", "idx", "index"):
        val = getattr(block, attr, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                return None
    return None


def _block_hash(block: Any) -> Optional[str]:
    for attr in ("block_hash", "content_hash", "hash"):
        val = getattr(block, attr, None)
        if val is not None:
            return val if isinstance(val, str) else hex(int(val))
    return None


def _block_size_tokens(cache: Any) -> int:
    for attr in ("page_size", "block_size", "chunk_size"):
        val = getattr(cache, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    return 1


def make_match_prefix_wrapper(original: Callable) -> Callable:
    """Wrap RadixCache.match_prefix."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        request = kwargs.get("req") or kwargs.get("request")
        request_id = _req_id(request)
        key = args[0] if args else kwargs.get("key", kwargs.get("token_ids"))
        n_keys = len(_to_list(key)) if key is not None else None

        t_start = now_ns()
        result = original(self, *args, **kwargs)
        t_end = now_ns()

        hits = _flatten(result)
        block_size = _block_size_tokens(self)
        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": "prefix_lookup",
            "request_id": request_id,
            "duration_ns": t_end - t_start,
            "n_keys": n_keys,
            "n_hits": len(hits),
            "hit_depth_tokens": len(hits) * block_size,
            "tier_scope": "HBM_LOCAL",
            "structure": "radix",
        })

        sampler = getattr(em, "_sampler", None)
        for block in hits:
            emit_ok, rate = (
                sampler.should_emit_kv_block("prefix_hit") if sampler else (True, None)
            )
            if not emit_ok:
                continue
            bid = _block_id(block)
            rec: dict = {
                "ts_ns": t_end,
                "type": "kv_block",
                "subtype": "prefix_hit",
                "block_id": bid if bid is not None else -1,
                "block_size_tokens": block_size,
                "tier_before": "HBM_LOCAL",
                "tier_after": "HBM_LOCAL",
                "reason": "prefix_match",
            }
            if request_id:
                rec["owner_request_id"] = request_id
            bh = _block_hash(block)
            if bh is not None:
                rec["block_hash"] = bh
            reuse_count = getattr(block, "reuse_count", None)
            if reuse_count is not None:
                rec["reuse_count"] = reuse_count
            if rate is not None:
                rec["sample_decision"] = rate
            em.event(rec)
        return result

    return wrapper


def make_insert_wrapper(original: Callable) -> Callable:
    """Wrap RadixCache.insert."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        request = kwargs.get("req") or kwargs.get("request")
        request_id = _req_id(request)
        size_before = _cache_size(self)
        key = args[0] if args else kwargs.get("key", kwargs.get("token_ids"))

        t_start = now_ns()
        result = original(self, *args, **kwargs)
        t_end = now_ns()
        size_after = _cache_size(self)

        inserted = _flatten(result)
        if not inserted and len(args) > 1:
            inserted = _flatten(args[1])

        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": "prefix_insert",
            "request_id": request_id,
            "duration_ns": t_end - t_start,
            "n_keys": len(_to_list(key)) if key is not None else len(inserted),
            "tier_scope": "HBM_LOCAL",
            "structure": "radix",
            "size_before": size_before,
            "size_after": size_after,
        })

        sampler = getattr(em, "_sampler", None)
        for block in inserted:
            emit_ok, rate = (
                sampler.should_emit_kv_block("hash_insert") if sampler else (True, None)
            )
            if not emit_ok:
                continue
            bid = _block_id(block)
            rec: dict = {
                "ts_ns": t_end,
                "type": "kv_block",
                "subtype": "hash_insert",
                "block_id": bid if bid is not None else -1,
                "tier_before": "HBM_LOCAL",
                "tier_after": "HBM_LOCAL",
                "reason": "scheduler",
            }
            if request_id:
                rec["owner_request_id"] = request_id
            bh = _block_hash(block)
            if bh is not None:
                rec["block_hash"] = bh
            if rate is not None:
                rec["sample_decision"] = rate
            em.event(rec)
        return result

    return wrapper


def make_evict_wrapper(original: Callable) -> Callable:
    """Wrap RadixCache.evict."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        t_start = now_ns()
        result = original(self, *args, **kwargs)
        t_end = now_ns()
        evicted = _flatten(result)

        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": "evict_select",
            "duration_ns": t_end - t_start,
            "n_keys": len(evicted),
            "tier_scope": "HBM_LOCAL",
            "structure": "radix",
        })

        sampler = getattr(em, "_sampler", None)
        for block in evicted:
            emit_ok, rate = (
                sampler.should_emit_kv_block("evict") if sampler else (True, None)
            )
            if not emit_ok:
                continue
            bid = _block_id(block)
            rec: dict = {
                "ts_ns": t_end,
                "type": "kv_block",
                "subtype": "evict",
                "block_id": bid if bid is not None else -1,
                "tier_before": "HBM_LOCAL",
                "tier_after": None,
                "reason": "capacity_evict",
            }
            if rate is not None:
                rec["sample_decision"] = rate
            em.event(rec)
        return result

    return wrapper


def make_refcount_wrapper(original: Callable, subtype: str) -> Callable:
    """Wrap RadixCache.inc_lock_ref / dec_lock_ref."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        result = original(self, *args, **kwargs)
        em.event({
            "ts_ns": now_ns(),
            "type": "metadata",
            "subtype": subtype,
            "n_keys": 1,
            "structure": "radix",
        })
        return result

    return wrapper


def make_allocator_alloc_wrapper(original: Callable) -> Callable:
    """Wrap BaseTokenToKVPoolAllocator.alloc."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        n_requested = _first_int(args, kwargs, ("num_tokens", "num_blocks", "size"))
        t_start = now_ns()
        result = original(self, *args, **kwargs)
        t_end = now_ns()
        blocks = _flatten(result)

        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": "allocator_alloc",
            "duration_ns": t_end - t_start,
            "n_keys": n_requested,
            "n_hits": len(blocks),
        })

        sampler = getattr(em, "_sampler", None)
        for block in blocks:
            emit_ok, rate = (
                sampler.should_emit_kv_block("allocate") if sampler else (True, None)
            )
            if not emit_ok:
                continue
            bid = _block_id(block)
            rec: dict = {
                "ts_ns": t_end,
                "type": "kv_block",
                "subtype": "allocate",
                "block_id": bid if bid is not None else -1,
                "tier_after": "HBM_LOCAL",
                "reason": "scheduler",
                "refcount_before": 0,
                "refcount_after": 1,
            }
            if rate is not None:
                rec["sample_decision"] = rate
            em.event(rec)
        return result

    return wrapper


def make_allocator_free_wrapper(original: Callable) -> Callable:
    """Wrap BaseTokenToKVPoolAllocator.free."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        blocks = _flatten(args[0] if args else kwargs.get("free_index", kwargs.get("indices")))
        t_start = now_ns()
        result = original(self, *args, **kwargs)
        t_end = now_ns()

        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": "allocator_free",
            "duration_ns": t_end - t_start,
            "n_keys": len(blocks),
        })

        sampler = getattr(em, "_sampler", None)
        for block in blocks:
            emit_ok, rate = (
                sampler.should_emit_kv_block("free") if sampler else (True, None)
            )
            if not emit_ok:
                continue
            bid = _block_id(block)
            rec: dict = {
                "ts_ns": t_end,
                "type": "kv_block",
                "subtype": "free",
                "block_id": bid if bid is not None else -1,
                "tier_before": "HBM_LOCAL",
                "reason": "finish_free",
                "refcount_before": 1,
                "refcount_after": 0,
            }
            if rate is not None:
                rec["sample_decision"] = rate
            em.event(rec)
        return result

    return wrapper


def make_req_to_token_write_wrapper(original: Callable) -> Callable:
    """Wrap ReqToTokenPool.write for block_table_update metadata."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        t_start = now_ns()
        result = original(self, *args, **kwargs)
        t_end = now_ns()
        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": "block_table_update",
            "duration_ns": t_end - t_start,
            "n_keys": len(_flatten(args[0])) if args else None,
            "structure": "treelist",
        })
        return result

    return wrapper


def _first_int(args: tuple, kwargs: dict, names: tuple[str, ...]) -> Optional[int]:
    if args:
        try:
            return int(args[0])
        except (TypeError, ValueError):
            pass
    for name in names:
        val = kwargs.get(name)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                return None
    return None


def _cache_size(cache: Any) -> Optional[int]:
    for attr in ("size", "total_size", "evictable_size_"):
        val = getattr(cache, attr, None)
        if callable(val):
            try:
                val = val()
            except Exception:
                continue
        if isinstance(val, int):
            return val
    return None


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
    """Patch SGLang radix cache, allocator, and request-token pool sites."""
    global _PATCHES_APPLIED

    with _PATCHES_LOCK:
        if _PATCHES_APPLIED:
            return True

        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return False

        applied = 0

        try:
            from sglang.srt.mem_cache import radix_cache as _radix_mod  # type: ignore
            cls = _radix_mod.RadixCache
            if _patch_method(cls, "match_prefix", make_match_prefix_wrapper):
                applied += 1
            if _patch_method(cls, "insert", make_insert_wrapper):
                applied += 1
            if _patch_method(cls, "evict", make_evict_wrapper):
                applied += 1
            if _patch_method(cls, "inc_lock_ref", lambda orig: make_refcount_wrapper(orig, "refcount_inc")):
                applied += 1
            if _patch_method(cls, "dec_lock_ref", lambda orig: make_refcount_wrapper(orig, "refcount_dec")):
                applied += 1
        except Exception as exc:
            _warn_once("radix_import", f"could not patch RadixCache: {exc}")

        try:
            from sglang.srt.mem_cache import allocator as _alloc_mod  # type: ignore
            cls = _alloc_mod.BaseTokenToKVPoolAllocator
            for name in ("alloc", "allocate"):
                if _patch_method(cls, name, make_allocator_alloc_wrapper):
                    applied += 1
                    break
            for name in ("free", "release"):
                if _patch_method(cls, name, make_allocator_free_wrapper):
                    applied += 1
                    break
        except Exception as exc:
            _warn_once("allocator_import", f"could not patch allocator: {exc}")

        try:
            from sglang.srt.mem_cache import memory_pool as _pool_mod  # type: ignore
            cls = _pool_mod.ReqToTokenPool
            if _patch_method(cls, "write", make_req_to_token_write_wrapper):
                applied += 1
        except Exception as exc:
            _warn_once("memory_pool_import", f"could not patch ReqToTokenPool: {exc}")

        _PATCHES_APPLIED = applied > 0
        return _PATCHES_APPLIED
