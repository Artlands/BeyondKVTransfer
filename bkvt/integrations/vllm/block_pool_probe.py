"""
vLLM block-pool and KV-cache-manager probes — Milestone 2.

Instrumentation sites (DESIGN.md §5.3, §5.5):

  §5.3  allocate          vllm/v1/core/block_pool.py
                          BlockPool.alloc_blocks (or allocate)
  §5.3  free              same — free_blocks / free
  §5.3  evict             same — LRU eviction path
  §5.3  prefix_hit        vllm/v1/core/kv_cache_manager.py
                          KVCacheManager.get_computed_blocks
  §5.3  hash_insert       vllm/v1/core/kv_cache_utils.py
                          block-hash insertion sites

  §5.5  prefix_lookup     KVCacheManager.get_computed_blocks
  §5.5  prefix_insert     KVCacheManager.cache_full_blocks
  §5.5  allocator_alloc   BlockPool.alloc_blocks
  §5.5  allocator_free    BlockPool.free_blocks
  §5.5  refcount_inc/dec  BlockPool._inc_ref / _dec_ref
  §5.5  evict_select      BlockPool LRU selection step
  §5.5  block_table_update KVCacheManager._update_block_table_for_request

Design constraints same as scheduler_probe.py.
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
        logger.warning("bkvt[vllm]: %s", msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_list(obj: Any) -> list:
    """Coerce block-id container to a plain list."""
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return list(obj)
    try:
        return list(obj)
    except TypeError:
        return []


def _block_size_bytes(block: Any, num_layers: int = 0) -> Optional[int]:
    """Try to infer block_size_bytes from a block or pool object."""
    for attr in ("block_size_bytes", "kv_size_bytes", "size_bytes"):
        v = getattr(block, attr, None)
        if isinstance(v, int):
            return v
    return None


def _current_request_id() -> Optional[str]:
    """Best-effort extraction of the current request_id from thread-local."""
    # runner_probe and scheduler_probe set this on threading.local
    import threading as _th
    tls = getattr(_th.current_thread(), "_bkvt_ctx", None)
    if tls is not None:
        return getattr(tls, "request_id", None)
    return None


# ---------------------------------------------------------------------------
# §5.3/§5.5 — BlockPool.alloc_blocks wrapper
# ---------------------------------------------------------------------------

def make_alloc_blocks_wrapper(original: Callable) -> Callable:
    """Wrap BlockPool.alloc_blocks (or equivalent).

    Emits:
      * kv_block/allocate  (one per block — §5.3)
      * metadata/allocator_alloc  (one per call — §5.5)
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        t_start = now_ns()
        result = original(self, *args, **kwargs)
        t_end = now_ns()

        # Try to determine number of blocks requested
        n_requested: Optional[int] = None
        if args:
            try:
                n_requested = int(args[0])
            except (TypeError, ValueError):
                pass
        n_requested = kwargs.get("num_blocks", n_requested)

        # Blocks returned may be a list of block ids or block objects
        returned = _to_list(result)
        n_allocated = len(returned)

        request_id = _current_request_id() or kwargs.get("request_id")

        # metadata/allocator_alloc
        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": "allocator_alloc",
            "request_id": request_id,
            "duration_ns": t_end - t_start,
            "n_keys": n_requested,
            "n_hits": n_allocated,
        })

        # kv_block/allocate — one per block
        sampler = getattr(em, "_sampler", None)
        for block in returned:
            emit_ok, rate = (sampler.should_emit_kv_block("allocate")
                             if sampler else (True, None))
            if not emit_ok:
                continue

            block_id = (block if isinstance(block, int)
                        else getattr(block, "block_id",
                                     getattr(block, "id", None)))
            block_hash = getattr(block, "block_hash",
                                 getattr(block, "content_hash", None))
            rec: dict = {
                "ts_ns": t_end,
                "type": "kv_block",
                "subtype": "allocate",
                "block_id": block_id if block_id is not None else -1,
                "tier_before": None,
                "tier_after": "HBM_LOCAL",
                "reason": "scheduler",
                "refcount_before": 0,
                "refcount_after": 1,
            }
            if block_hash is not None:
                rec["block_hash"] = (block_hash if isinstance(block_hash, str)
                                     else hex(block_hash))
            if request_id:
                rec["owner_request_id"] = request_id
            if rate is not None:
                rec["sample_decision"] = rate
            em.event(rec)

        return result

    return wrapper


# ---------------------------------------------------------------------------
# §5.3/§5.5 — BlockPool.free_blocks wrapper
# ---------------------------------------------------------------------------

def make_free_blocks_wrapper(original: Callable) -> Callable:
    """Wrap BlockPool.free_blocks.

    Emits:
      * kv_block/free      (one per block — §5.3)
      * metadata/allocator_free  (one per call — §5.5)
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        # Capture the block list before freeing
        block_arg = args[0] if args else kwargs.get("block_ids",
                                                    kwargs.get("blocks"))
        blocks = _to_list(block_arg)

        t_start = now_ns()
        result = original(self, *args, **kwargs)
        t_end = now_ns()

        request_id = _current_request_id() or kwargs.get("request_id")

        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": "allocator_free",
            "request_id": request_id,
            "duration_ns": t_end - t_start,
            "n_keys": len(blocks),
        })

        sampler = getattr(em, "_sampler", None)
        for block in blocks:
            emit_ok, rate = (sampler.should_emit_kv_block("free")
                             if sampler else (True, None))
            if not emit_ok:
                continue

            block_id = (block if isinstance(block, int)
                        else getattr(block, "block_id",
                                     getattr(block, "id", None)))
            rec: dict = {
                "ts_ns": t_end,
                "type": "kv_block",
                "subtype": "free",
                "block_id": block_id if block_id is not None else -1,
                "tier_before": "HBM_LOCAL",
                "tier_after": None,
                "reason": "finish_free",
                "refcount_before": 1,
                "refcount_after": 0,
            }
            if request_id:
                rec["owner_request_id"] = request_id
            if rate is not None:
                rec["sample_decision"] = rate
            em.event(rec)

        return result

    return wrapper


# ---------------------------------------------------------------------------
# §5.3/§5.5 — LRU eviction wrapper
# ---------------------------------------------------------------------------

def make_evict_wrapper(original: Callable) -> Callable:
    """Wrap the LRU eviction method in BlockPool.

    Emits:
      * metadata/evict_select  (§5.5)
      * kv_block/evict         per evicted block (§5.3)
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        t_start = now_ns()
        result = original(self, *args, **kwargs)
        t_end = now_ns()

        evicted = _to_list(result) if result is not None else []

        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": "evict_select",
            "duration_ns": t_end - t_start,
            "n_keys": len(evicted),
        })

        sampler = getattr(em, "_sampler", None)
        for block in evicted:
            emit_ok, rate = (sampler.should_emit_kv_block("evict")
                             if sampler else (True, None))
            if not emit_ok:
                continue

            block_id = (block if isinstance(block, int)
                        else getattr(block, "block_id",
                                     getattr(block, "id", None)))
            age_ns = getattr(block, "age_ns", None)
            reuse_count = getattr(block, "reuse_count",
                                  getattr(block, "num_hashed_tokens", None))
            block_hash = getattr(block, "block_hash",
                                 getattr(block, "content_hash", None))

            rec: dict = {
                "ts_ns": t_end,
                "type": "kv_block",
                "subtype": "evict",
                "block_id": block_id if block_id is not None else -1,
                "tier_before": "HBM_LOCAL",
                "tier_after": None,
                "reason": "capacity_evict",
            }
            if block_hash is not None:
                rec["block_hash"] = (block_hash if isinstance(block_hash, str)
                                     else hex(block_hash))
            if age_ns is not None:
                rec["age_ns"] = age_ns
            if reuse_count is not None:
                rec["reuse_count"] = reuse_count
            if rate is not None:
                rec["sample_decision"] = rate
            em.event(rec)

        return result

    return wrapper


# ---------------------------------------------------------------------------
# §5.3/§5.5 — refcount wrappers
# ---------------------------------------------------------------------------

def make_inc_ref_wrapper(original: Callable) -> Callable:
    """Wrap BlockPool._inc_ref → metadata/refcount_inc."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        result = original(self, *args, **kwargs)
        block_id = args[0] if args else kwargs.get("block_id")
        if not isinstance(block_id, int):
            block_id = getattr(block_id, "block_id", None)

        em.event({
            "ts_ns": now_ns(),
            "type": "metadata",
            "subtype": "refcount_inc",
            "n_keys": 1,
        })
        return result

    return wrapper


def make_dec_ref_wrapper(original: Callable) -> Callable:
    """Wrap BlockPool._dec_ref → metadata/refcount_dec."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        result = original(self, *args, **kwargs)
        em.event({
            "ts_ns": now_ns(),
            "type": "metadata",
            "subtype": "refcount_dec",
            "n_keys": 1,
        })
        return result

    return wrapper


# ---------------------------------------------------------------------------
# §5.3/§5.5 — KVCacheManager.get_computed_blocks (prefix_hit / prefix_lookup)
# ---------------------------------------------------------------------------

def make_get_computed_blocks_wrapper(original: Callable) -> Callable:
    """Wrap KVCacheManager.get_computed_blocks.

    Emits:
      * metadata/prefix_lookup  (§5.5)
      * kv_block/prefix_hit     per matched block (§5.3, Q4, Q5)

    Also captures ``earliest_known_ts_ns`` for transfers that will be
    issued for these matched blocks (Q5).
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        t_start = now_ns()

        # Extract request from args
        request = args[0] if args else kwargs.get("request")
        request_id: Optional[str] = None
        if request is not None:
            for attr in ("req_id", "request_id", "id"):
                v = getattr(request, attr, None)
                if isinstance(v, str):
                    request_id = v
                    break

        result = original(self, *args, **kwargs)

        t_end = now_ns()

        matched_blocks = _to_list(result)
        n_hits = len(matched_blocks)

        # hit_depth_tokens: each matched block covers block_size tokens
        block_size = getattr(self, "block_size",
                             getattr(self, "kv_cache_spec", None))
        if hasattr(block_size, "block_size"):
            block_size = block_size.block_size
        if not isinstance(block_size, int):
            block_size = 16  # vLLM default

        hit_depth_tokens = n_hits * block_size

        # metadata/prefix_lookup
        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": "prefix_lookup",
            "request_id": request_id,
            "duration_ns": t_end - t_start,
            "n_keys": 1,          # one lookup per request
            "n_hits": n_hits,
            "hit_depth_tokens": hit_depth_tokens,
            "tier_scope": "HBM_LOCAL",
            "structure": "radix",
        })

        # kv_block/prefix_hit — one per matched block
        sampler = getattr(em, "_sampler", None)
        for block in matched_blocks:
            emit_ok, rate = (sampler.should_emit_kv_block("prefix_hit")
                             if sampler else (True, None))
            if not emit_ok:
                continue

            block_id = (block if isinstance(block, int)
                        else getattr(block, "block_id",
                                     getattr(block, "id", None)))
            block_hash = getattr(block, "block_hash",
                                 getattr(block, "content_hash", None))
            reuse_count = getattr(block, "reuse_count", None)
            last_reuse_ts = getattr(block, "last_accessed", None)

            rec: dict = {
                "ts_ns": t_end,
                "type": "kv_block",
                "subtype": "prefix_hit",
                "block_id": block_id if block_id is not None else -1,
                "tier_before": "HBM_LOCAL",
                "tier_after": "HBM_LOCAL",
                "reason": "prefix_match",
            }
            if request_id:
                rec["owner_request_id"] = request_id
            if block_hash is not None:
                rec["block_hash"] = (block_hash if isinstance(block_hash, str)
                                     else hex(block_hash))
            if reuse_count is not None:
                rec["reuse_count"] = reuse_count
            if last_reuse_ts is not None:
                rec["last_reuse_ts_ns"] = int(last_reuse_ts)
            if rate is not None:
                rec["sample_decision"] = rate
            em.event(rec)

        return result

    return wrapper


# ---------------------------------------------------------------------------
# §5.5 — KVCacheManager.cache_full_blocks (prefix_insert)
# ---------------------------------------------------------------------------

def make_cache_full_blocks_wrapper(original: Callable) -> Callable:
    """Wrap KVCacheManager.cache_full_blocks → metadata/prefix_insert."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        request = args[0] if args else kwargs.get("request")
        request_id: Optional[str] = None
        if request is not None:
            for attr in ("req_id", "request_id", "id"):
                v = getattr(request, attr, None)
                if isinstance(v, str):
                    request_id = v
                    break

        # Count blocks being inserted
        blocks_to_cache = args[1] if len(args) > 1 else kwargs.get("blocks", [])
        n_blocks = len(_to_list(blocks_to_cache))

        t_start = now_ns()
        result = original(self, *args, **kwargs)
        t_end = now_ns()

        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": "prefix_insert",
            "request_id": request_id,
            "duration_ns": t_end - t_start,
            "n_keys": n_blocks,
            "tier_scope": "HBM_LOCAL",
            "structure": "radix",
        })

        # kv_block/hash_insert — one per block
        sampler = getattr(em, "_sampler", None)
        for block in _to_list(blocks_to_cache):
            emit_ok, rate = (sampler.should_emit_kv_block("hash_insert")
                             if sampler else (True, None))
            if not emit_ok:
                continue

            block_id = (block if isinstance(block, int)
                        else getattr(block, "block_id",
                                     getattr(block, "id", None)))
            block_hash = getattr(block, "block_hash",
                                 getattr(block, "content_hash", None))

            rec: dict = {
                "ts_ns": t_end,
                "type": "kv_block",
                "subtype": "hash_insert",
                "block_id": block_id if block_id is not None else -1,
                "tier_before": "HBM_LOCAL",
                "tier_after": "HBM_LOCAL",
                "reason": "scheduler",
            }
            if request_id:
                rec["owner_request_id"] = request_id
            if block_hash is not None:
                rec["block_hash"] = (block_hash if isinstance(block_hash, str)
                                     else hex(block_hash))
            if rate is not None:
                rec["sample_decision"] = rate
            em.event(rec)

        return result

    return wrapper


# ---------------------------------------------------------------------------
# §5.5 — block_table_update wrapper
# ---------------------------------------------------------------------------

def make_block_table_update_wrapper(original: Callable) -> Callable:
    """Wrap KVCacheManager._update_block_table_for_request → metadata/block_table_update."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        request = args[0] if args else kwargs.get("request")
        request_id: Optional[str] = None
        if request is not None:
            for attr in ("req_id", "request_id", "id"):
                v = getattr(request, attr, None)
                if isinstance(v, str):
                    request_id = v
                    break

        t_start = now_ns()
        result = original(self, *args, **kwargs)
        t_end = now_ns()

        em.event({
            "ts_ns": t_start,
            "type": "metadata",
            "subtype": "block_table_update",
            "request_id": request_id,
            "duration_ns": t_end - t_start,
        })

        return result

    return wrapper


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

_PATCHES_APPLIED: bool = False
_PATCHES_LOCK = threading.Lock()


def apply_patches() -> bool:
    """Monkey-patch vLLM block-pool and KV-cache-manager classes.

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

        # ── BlockPool ────────────────────────────────────────────────────
        try:
            from vllm.v1.core import block_pool as _bp_mod  # type: ignore
            cls = _bp_mod.BlockPool

            for alloc_name in ("alloc_blocks", "allocate", "alloc"):
                orig = getattr(cls, alloc_name, None)
                if orig is not None:
                    setattr(cls, alloc_name, make_alloc_blocks_wrapper(orig))
                    logger.info("bkvt[vllm]: patched BlockPool.%s", alloc_name)
                    applied += 1
                    break
            else:
                _warn_once("bp_alloc", "BlockPool: no alloc method found")

            for free_name in ("free_blocks", "free", "release"):
                orig = getattr(cls, free_name, None)
                if orig is not None:
                    setattr(cls, free_name, make_free_blocks_wrapper(orig))
                    logger.info("bkvt[vllm]: patched BlockPool.%s", free_name)
                    applied += 1
                    break
            else:
                _warn_once("bp_free", "BlockPool: no free method found")

            for evict_name in ("_evict_blocks", "evict_blocks", "_evict"):
                orig = getattr(cls, evict_name, None)
                if orig is not None:
                    setattr(cls, evict_name, make_evict_wrapper(orig))
                    logger.info("bkvt[vllm]: patched BlockPool.%s", evict_name)
                    applied += 1
                    break

            for inc_name in ("_inc_ref", "inc_ref"):
                orig = getattr(cls, inc_name, None)
                if orig is not None:
                    setattr(cls, inc_name, make_inc_ref_wrapper(orig))
                    applied += 1
                    break

            for dec_name in ("_dec_ref", "dec_ref"):
                orig = getattr(cls, dec_name, None)
                if orig is not None:
                    setattr(cls, dec_name, make_dec_ref_wrapper(orig))
                    applied += 1
                    break

        except Exception as exc:
            _warn_once("block_pool", f"could not patch BlockPool: {exc}")

        # ── KVCacheManager ───────────────────────────────────────────────
        try:
            from vllm.v1.core import kv_cache_manager as _kvm_mod  # type: ignore
            cls = _kvm_mod.KVCacheManager

            for lookup_name in ("get_computed_blocks", "get_prefix_cache_hit_blocks",
                                "match_prefix"):
                orig = getattr(cls, lookup_name, None)
                if orig is not None:
                    setattr(cls, lookup_name,
                            make_get_computed_blocks_wrapper(orig))
                    logger.info("bkvt[vllm]: patched KVCacheManager.%s", lookup_name)
                    applied += 1
                    break
            else:
                _warn_once("kvm_lookup",
                           "KVCacheManager: no prefix-lookup method found")

            for insert_name in ("cache_full_blocks", "insert_blocks",
                                "update_prefix_cache"):
                orig = getattr(cls, insert_name, None)
                if orig is not None:
                    setattr(cls, insert_name,
                            make_cache_full_blocks_wrapper(orig))
                    logger.info("bkvt[vllm]: patched KVCacheManager.%s",
                                insert_name)
                    applied += 1
                    break

            for bt_name in ("_update_block_table_for_request",
                            "update_block_table", "_update_block_table"):
                orig = getattr(cls, bt_name, None)
                if orig is not None:
                    setattr(cls, bt_name,
                            make_block_table_update_wrapper(orig))
                    applied += 1
                    break

        except Exception as exc:
            _warn_once("kv_cache_mgr", f"could not patch KVCacheManager: {exc}")

        _PATCHES_APPLIED = applied > 0
        return _PATCHES_APPLIED
