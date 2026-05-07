"""
vLLM KV connector probes — Milestone 3.

This module implements the wrapper surface described in DESIGN.md §5.4.
It intentionally avoids importing vLLM at module import time so unit tests
and non-vLLM environments can import bkvt cleanly.
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
from collections import defaultdict
from types import MethodType
from typing import Any, Callable, Optional

from bkvt import emitter as _emitter_mod
from bkvt.clock import now_ns

logger = logging.getLogger(__name__)


_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("bkvt[vllm]: %s", msg)


def _req_id(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    for attr in ("req_id", "request_id", "id"):
        val = getattr(obj, attr, None)
        if isinstance(val, str):
            return val
    return None


def _layer_idx(layer: Any) -> Optional[int]:
    if layer is None:
        return None
    if isinstance(layer, int):
        return layer
    if isinstance(layer, str):
        digits = "".join(ch if ch.isdigit() else " " for ch in layer).split()
        if digits:
            try:
                return int(digits[-1])
            except ValueError:
                return None
    for attr in ("layer_idx", "layer_id", "idx"):
        val = getattr(layer, attr, None)
        if isinstance(val, int):
            return val
    return None


def _as_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, (str, bytes)):
        return [obj]
    try:
        return list(obj)
    except TypeError:
        return [obj]


def _block_id(block: Any) -> Optional[int]:
    if isinstance(block, int):
        return block
    for attr in ("block_id", "id", "block_number"):
        val = getattr(block, attr, None)
        if isinstance(val, int):
            return val
    return None


def _block_ids_from(value: Any) -> list[int]:
    ids: list[int] = []
    for item in _as_list(value):
        bid = _block_id(item)
        if bid is not None:
            ids.append(bid)
    return ids


def _len_or_none(value: Any) -> Optional[int]:
    try:
        return len(value)
    except TypeError:
        return None


def _bytes_from(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    for attr in ("nbytes", "num_bytes", "bytes", "size_bytes"):
        val = getattr(value, attr, None)
        if isinstance(val, int):
            return val
    try:
        nbytes = value.nbytes  # numpy / torch-like
        if isinstance(nbytes, int):
            return nbytes
    except Exception:
        pass
    return None


def _request_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    for key in ("request", "req", "request_id", "req_id"):
        if key in kwargs:
            return kwargs[key]
    return args[0] if args else None


def _transport_for(inner: Any) -> str:
    text = f"{inner.__class__.__module__}.{inner.__class__.__name__}".lower()
    if "nixl" in text:
        return "nixl"
    if "lmcache" in text:
        return "tcp"
    if "shared_storage" in text or "file" in text or "storage" in text:
        return "file"
    return "tcp"


def _src_dst_for(direction: str, transport: str) -> tuple[str, str]:
    if direction == "load":
        if transport == "nixl":
            return "HBM_PEER_RDMA", "HBM_LOCAL"
        if transport == "file":
            return "OBJECT_STORE", "HBM_LOCAL"
        return "DRAM_REMOTE", "HBM_LOCAL"
    if transport == "nixl":
        return "HBM_LOCAL", "HBM_PEER_RDMA"
    if transport == "file":
        return "HBM_LOCAL", "OBJECT_STORE"
    return "HBM_LOCAL", "DRAM_REMOTE"


class ConnectorTransferState:
    """Thread-safe transfer bookkeeping for one wrapped connector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loads_by_layer: dict[Optional[int], list[dict[str, Any]]] = defaultdict(list)
        self._saves: list[dict[str, Any]] = []
        self._earliest_by_request: dict[str, int] = {}

    def remember_earliest(self, request_id: Optional[str], ts_ns: int) -> None:
        if not request_id:
            return
        with self._lock:
            self._earliest_by_request.setdefault(request_id, ts_ns)

    def earliest_for(self, request_id: Optional[str]) -> Optional[int]:
        if not request_id:
            return None
        with self._lock:
            return self._earliest_by_request.get(request_id)

    def add_load(self, layer_idx: Optional[int], entry: dict[str, Any]) -> None:
        with self._lock:
            self._loads_by_layer[layer_idx].append(entry)

    def pop_loads(self, layer_idx: Optional[int]) -> list[dict[str, Any]]:
        with self._lock:
            if layer_idx in self._loads_by_layer:
                return self._loads_by_layer.pop(layer_idx)
            if layer_idx is not None and None in self._loads_by_layer:
                return self._loads_by_layer.pop(None)
            return []

    def add_save(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._saves.append(entry)

    def pop_saves(self) -> list[dict[str, Any]]:
        with self._lock:
            saves, self._saves = self._saves, []
            return saves


class TracingConnectorWrapper:
    """Probe wrapper around a vLLM ``KVConnectorBase_V1`` instance."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self._state = ConnectorTransferState()
        self._transport = _transport_for(inner)
        self._patch_backend_probes()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.inner!r})"

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return self.inner.get_num_new_matched_tokens(request, num_computed_tokens, *args, **kwargs)

        request_id = _req_id(request)
        with em.metadata(
            "prefix_lookup",
            request_id=request_id,
            tier_scope="DRAM_REMOTE" if self._transport != "nixl" else "HBM_PEER_RDMA",
            structure="hashmap",
        ) as rec:
            result = self.inner.get_num_new_matched_tokens(request, num_computed_tokens, *args, **kwargs)
            rec["n_keys"] = 1
            try:
                matched_tokens = int(result)
            except (TypeError, ValueError):
                matched_tokens = 0
            rec["n_hits"] = max(matched_tokens - int(num_computed_tokens or 0), 0)
            rec["hit_depth_tokens"] = matched_tokens

        self._state.remember_earliest(request_id, now_ns())
        return result

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int = 0, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return self.inner.update_state_after_alloc(request, blocks, num_external_tokens, *args, **kwargs)

        result = self.inner.update_state_after_alloc(request, blocks, num_external_tokens, *args, **kwargs)
        request_id = _req_id(request)
        ids = _block_ids_from(blocks)
        ts = now_ns()
        for bid in ids:
            em.event({
                "ts_ns": ts,
                "type": "kv_block",
                "subtype": "tier_promote",
                "block_id": bid,
                "tier_before": "DRAM_REMOTE" if self._transport != "nixl" else "HBM_PEER_RDMA",
                "tier_after": "HBM_LOCAL",
                "owner_request_id": request_id,
                "reason": "connector_load",
            })
        return result

    def build_connector_meta(self, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return self.inner.build_connector_meta(scheduler_output, *args, **kwargs)

        with em.metadata("connector_build_meta") as rec:
            result = self.inner.build_connector_meta(scheduler_output, *args, **kwargs)
            rec["n_keys"] = _len_or_none(getattr(scheduler_output, "scheduled_new_reqs", None))
            rec["size_after"] = _len_or_none(result)
        return result

    def start_load_kv(self, forward_context: Any, *args: Any, **kwargs: Any) -> Any:
        request_id = kwargs.get("request_id") or _req_id(getattr(forward_context, "request", None))
        layer_idx = kwargs.get("layer_idx")
        block_ids = _block_ids_from(kwargs.get("block_ids") or getattr(forward_context, "block_ids", None))
        bytes_ = _bytes_from(kwargs.get("bytes") or kwargs.get("kv") or getattr(forward_context, "kv_tensors", None))
        src, dst = _src_dst_for("load", self._transport)
        em = _emitter_mod.get_emitter()

        tid: Optional[str] = None
        start_ts = now_ns()
        if em.enabled:
            tid = em.transfer_start(
                "load",
                request_id=request_id,
                layer_idx=layer_idx,
                src_tier=src,
                dst_tier=dst,
                transport=self._transport,
                num_blocks=len(block_ids) if block_ids else None,
                bytes_=bytes_,
                block_ids=block_ids or None,
                issued_by="connector",
                issued_at_phase=kwargs.get("issued_at_phase"),
                earliest_known_ts_ns=self._state.earliest_for(request_id) or start_ts,
            )

        succeeded = False
        try:
            result = self.inner.start_load_kv(forward_context, *args, **kwargs)
            succeeded = True
            return result
        except Exception:
            if tid:
                em.transfer_end(tid, cancelled=True)
            raise
        finally:
            if tid and succeeded:
                self._state.add_load(layer_idx, {
                    "transfer_id": tid,
                    "started_ts_ns": start_ts,
                    "bytes": bytes_,
                    "request_id": request_id,
                    "layer_idx": layer_idx,
                })

    def wait_for_layer_load(self, layer_name: Any, *args: Any, **kwargs: Any) -> Any:
        layer_idx = kwargs.get("layer_idx", _layer_idx(layer_name))
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return self.inner.wait_for_layer_load(layer_name, *args, **kwargs)

        loads: list[dict[str, Any]] = []
        try:
            with em.metadata("wait_for_layer_load", n_keys=1) as rec:
                result = self.inner.wait_for_layer_load(layer_name, *args, **kwargs)
                loads = self._state.pop_loads(layer_idx)
                rec["n_hits"] = len(loads)
        except Exception:
            loads = self._state.pop_loads(layer_idx)
            for entry in loads:
                em.transfer_end(entry["transfer_id"], cancelled=True)
            raise

        end_ts = now_ns()
        for entry in loads:
            start_ts = entry.get("started_ts_ns")
            wire_time_ns = end_ts - start_ts if isinstance(start_ts, int) else None
            em.transfer_end(
                entry["transfer_id"],
                bytes_=entry.get("bytes"),
                wire_time_ns=wire_time_ns,
                request_id=entry.get("request_id"),
                layer_idx=entry.get("layer_idx"),
            )
        return result

    def save_kv_layer(self, layer_name: Any, kv_layer: Any, *args: Any, **kwargs: Any) -> Any:
        request_id = kwargs.get("request_id") or _req_id(kwargs.get("request"))
        layer_idx = kwargs.get("layer_idx", _layer_idx(layer_name))
        block_ids = _block_ids_from(kwargs.get("block_ids") or kwargs.get("blocks"))
        bytes_ = _bytes_from(kwargs.get("bytes") or kv_layer)
        src, dst = _src_dst_for("save", self._transport)
        em = _emitter_mod.get_emitter()

        tid: Optional[str] = None
        start_ts = now_ns()
        if em.enabled:
            tid = em.transfer_start(
                "save",
                request_id=request_id,
                layer_idx=layer_idx,
                src_tier=src,
                dst_tier=dst,
                transport=self._transport,
                num_blocks=len(block_ids) if block_ids else None,
                bytes_=bytes_,
                block_ids=block_ids or None,
                issued_by="connector",
                issued_at_phase=kwargs.get("issued_at_phase"),
                earliest_known_ts_ns=kwargs.get("earliest_known_ts_ns", start_ts),
            )

        succeeded = False
        try:
            result = self.inner.save_kv_layer(layer_name, kv_layer, *args, **kwargs)
            succeeded = True
            return result
        except Exception:
            if tid:
                em.transfer_end(tid, cancelled=True)
            raise
        finally:
            if tid and succeeded:
                self._state.add_save({
                    "transfer_id": tid,
                    "started_ts_ns": start_ts,
                    "bytes": bytes_,
                    "request_id": request_id,
                    "layer_idx": layer_idx,
                })

    def wait_for_save(self, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return self.inner.wait_for_save(*args, **kwargs)

        saves: list[dict[str, Any]] = []
        try:
            with em.metadata("wait_for_save") as rec:
                result = self.inner.wait_for_save(*args, **kwargs)
                saves = self._state.pop_saves()
                rec["n_hits"] = len(saves)
        except Exception:
            saves = self._state.pop_saves()
            for entry in saves:
                em.transfer_end(entry["transfer_id"], cancelled=True)
            raise

        end_ts = now_ns()
        for entry in saves:
            start_ts = entry.get("started_ts_ns")
            wire_time_ns = end_ts - start_ts if isinstance(start_ts, int) else None
            em.transfer_end(
                entry["transfer_id"],
                bytes_=entry.get("bytes"),
                wire_time_ns=wire_time_ns,
                request_id=entry.get("request_id"),
                layer_idx=entry.get("layer_idx"),
            )
        return result

    def get_finished(self, finished_req_ids: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return self.inner.get_finished(finished_req_ids, *args, **kwargs)

        with em.metadata("connector_get_finished", n_keys=len(_as_list(finished_req_ids))) as rec:
            result = self.inner.get_finished(finished_req_ids, *args, **kwargs)
            rec["n_hits"] = _len_or_none(result)
        return result

    def _patch_backend_probes(self) -> None:
        if self._transport != "nixl":
            return
        agent = None
        for attr in ("nixl_agent", "agent", "_nixl_agent"):
            candidate = getattr(self.inner, attr, None)
            if candidate is not None:
                agent = candidate
                break
        if agent is None or getattr(agent, "_bkvt_patched", False):
            return
        for name in ("make_xfer", "transfer"):
            func = getattr(agent, name, None)
            if callable(func):
                try:
                    setattr(agent, name, MethodType(_make_nixl_call_wrapper(func, name), agent))
                except Exception as exc:
                    _warn_once(f"nixl_{name}", f"could not patch NIXL agent {name}: {exc}")
        setattr(agent, "_bkvt_patched", True)


def _make_nixl_call_wrapper(original: Callable, name: str) -> Callable:
    if inspect.ismethod(original):
        bound_original = original
    else:
        bound_original = None

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            if bound_original is not None:
                return bound_original(*args, **kwargs)
            return original(self, *args, **kwargs)

        start = now_ns()
        result = bound_original(*args, **kwargs) if bound_original is not None else original(self, *args, **kwargs)
        end = now_ns()
        descriptors = kwargs.get("descriptors") or (args[0] if args else None)
        wr_count = _len_or_none(descriptors)
        em.event({
            "ts_ns": start,
            "type": "metadata",
            "subtype": "nixl_call",
            "duration_ns": end - start,
            "n_keys": wr_count,
            "structure": "treelist",
        })
        return result

    return wrapper


def wrap_connector(connector: Any) -> Any:
    """Return a tracing wrapper unless ``connector`` is already wrapped."""
    if connector is None or isinstance(connector, TracingConnectorWrapper):
        return connector
    return TracingConnectorWrapper(connector)
