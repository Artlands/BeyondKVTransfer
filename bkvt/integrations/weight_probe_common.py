"""Shared helpers for vLLM/SGLang Q7 weight and adapter probes."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import re
import threading
from typing import Any, Callable, Iterator, Optional

from bkvt import emitter as _emitter_mod
from bkvt.clock import now_ns

_TLS = threading.local()


@contextlib.contextmanager
def payload_context(
    payload_kind: str,
    *,
    param_name: Optional[str] = None,
    expert_id: Optional[int] = None,
    lora_adapter_id: Optional[str] = None,
    weight_version: Optional[str] = None,
    reason: Optional[str] = None,
) -> Iterator[None]:
    prev = getattr(_TLS, "payload", None)
    _TLS.payload = {
        "payload_kind": payload_kind,
        "param_name": param_name,
        "expert_id": expert_id,
        "lora_adapter_id": lora_adapter_id,
        "weight_version": weight_version,
        "reason": reason,
    }
    try:
        yield
    finally:
        _TLS.payload = prev


def current_payload_context(default_kind: Optional[str] = None) -> Optional[dict[str, Any]]:
    ctx = getattr(_TLS, "payload", None)
    if ctx is None and default_kind is not None:
        return {"payload_kind": default_kind}
    return ctx


def make_collective_wrapper(
    original: Callable,
    method_name: str,
    *,
    transport: str = "nccl_send_recv",
    default_payload_kind: Optional[str] = None,
) -> Callable:
    """Wrap a collective and tag it from the active payload context."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        ctx = current_payload_context(default_payload_kind)
        if not em.enabled or ctx is None:
            return original(*args, **kwargs)

        payload_kind = ctx.get("payload_kind") or default_payload_kind or "activation"
        bytes_ = _bytes_for(args, kwargs)
        t_start = now_ns()
        tid = em.transfer_start(
            "load",
            payload_kind=payload_kind,
            src_tier="HBM_PEER_RDMA" if transport.startswith("nccl") else "DRAM_REMOTE",
            dst_tier="HBM_LOCAL",
            transport=transport,
            bytes_=bytes_,
            num_blocks=1 if payload_kind == "weight" else None,
            issued_by="weight_update" if ctx.get("reason") == "rlhf_update" else "moe_layer",
            issued_at_phase=kwargs.get("issued_at_phase"),
            earliest_known_ts_ns=t_start,
            param_name=ctx.get("param_name"),
            expert_id=ctx.get("expert_id"),
            lora_adapter_id=ctx.get("lora_adapter_id"),
            weight_version=ctx.get("weight_version"),
            collective_method=method_name,
        )
        ok = False
        try:
            result = original(*args, **kwargs)
            ok = True
            return result
        finally:
            t_end = now_ns()
            em.transfer_end(
                tid,
                bytes_=bytes_ if bytes_ is not None else _bytes_from(locals().get("result")),
                wire_time_ns=t_end - t_start,
                cancelled=not ok,
                collective_method=method_name,
            )

    return wrapper


def make_load_model_wrapper(original: Callable, *, backend: str, transport: str = "mmap") -> Callable:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)

        t_start = now_ns()
        with payload_context("weight", reason="startup_load"):
            result = original(self, *args, **kwargs)
        params = _params_from(result) or _params_from(self) or _params_from(kwargs.get("model"))
        if not params:
            params = [("model", result)]
        if _coarse_profile(em):
            total = sum((_tensor_info(value)["bytes"] or 0) for _, value in params)
            _emit_weight(em, t_start, "load", "model", total, None, None, None, None, "startup_load", "OBJECT_STORE", "HBM_LOCAL")
            _emit_weight_transfer(em, "load", "model", total, "OBJECT_STORE", "HBM_LOCAL", transport, t_start, backend=backend)
            return result

        for name, tensor in params:
            info = _tensor_info(tensor)
            _emit_weight(
                em,
                t_start,
                "load",
                name,
                info["bytes"],
                info["shape"],
                info["dtype"],
                _layer_idx(name),
                None,
                "startup_load",
                "OBJECT_STORE",
                "HBM_LOCAL",
                shard_role=_shard_role(self),
            )
            _emit_weight_transfer(em, "load", name, info["bytes"], "OBJECT_STORE", "HBM_LOCAL", transport, t_start, layer_idx=_layer_idx(name), backend=backend)
        return result

    return wrapper


def make_offload_wrapper(original: Callable, method_name: str) -> Callable:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)
        name = _param_name(args, kwargs, default=method_name)
        tensor = kwargs.get("tensor") or (args[0] if args else None)
        info = _tensor_info(tensor)
        promote = any(part in method_name.lower() for part in ("prefetch", "load", "to_gpu", "promote"))
        subtype = "tier_promote" if promote else "tier_demote"
        src, dst = ("DRAM_LOCAL", "HBM_LOCAL") if promote else ("HBM_LOCAL", "DRAM_LOCAL")
        t_start = now_ns()
        result = original(self, *args, **kwargs)
        _emit_weight(em, t_start, subtype, name, info["bytes"], info["shape"], info["dtype"], _layer_idx(name), None, "cpu_offload", src, dst)
        _emit_weight_transfer(em, "load" if promote else "save", name, info["bytes"], src, dst, "local_memcpy", t_start, layer_idx=_layer_idx(name))
        return result

    return wrapper


def make_moe_wrapper(original: Callable, method_name: str) -> Callable:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)
        experts = _expert_ids(args, kwargs) or [getattr(self, "expert_id", 0)]
        bytes_ = _bytes_for(args, kwargs)
        t_start = now_ns()
        with payload_context("activation", reason="expert_routing"):
            result = original(self, *args, **kwargs)
        for expert_id in experts:
            param = f"{method_name}.expert.{expert_id}"
            _emit_weight(em, t_start, "expert_dispatch", param, bytes_, None, None, _layer_idx(param), int(expert_id), "expert_routing", "HBM_LOCAL", "HBM_LOCAL")
            _emit_weight_transfer(
                em,
                "load",
                param,
                bytes_,
                "HBM_PEER_RDMA",
                "HBM_LOCAL",
                "nccl_send_recv",
                t_start,
                payload_kind="activation",
                expert_id=int(expert_id),
                issued_by="moe_layer",
            )
        return result

    return wrapper


def make_lora_wrapper(original: Callable, method_name: str) -> Callable:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)
        adapter_id = _adapter_id(args, kwargs) or method_name
        activate = "remove" not in method_name.lower() and "deactivate" not in method_name.lower() and "unload" not in method_name.lower()
        subtype = "lora_activate" if activate else "lora_deactivate"
        src, dst = ("DRAM_LOCAL", "HBM_LOCAL") if activate else ("HBM_LOCAL", "DRAM_LOCAL")
        bytes_ = _bytes_for(args, kwargs)
        request_id = _request_id(args, kwargs)
        t_start = now_ns()
        with payload_context("weight", lora_adapter_id=adapter_id, reason="lora_swap"):
            result = original(self, *args, **kwargs)
        _emit_weight(em, t_start, subtype, f"lora.{adapter_id}", bytes_, None, None, None, None, "lora_swap", src, dst, lora_adapter_id=adapter_id, owner_request_id=request_id)
        _emit_weight_transfer(em, "load" if activate else "save", f"lora.{adapter_id}", bytes_, src, dst, "local_memcpy", t_start, lora_adapter_id=adapter_id, request_id=request_id, issued_by="lora_manager")
        return result

    return wrapper


def make_update_weights_wrapper(original: Callable, method_name: str) -> Callable:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        em = _emitter_mod.get_emitter()
        if not em.enabled:
            return original(self, *args, **kwargs)
        version = str(kwargs.get("weight_version") or kwargs.get("version") or _version_id(args, kwargs))
        t_start = now_ns()
        with payload_context("weight", weight_version=version, reason="rlhf_update"):
            result = original(self, *args, **kwargs)
        params = _params_from(kwargs.get("weights")) or _params_from(args[0] if args else None) or _params_from(result)
        if not params:
            params = [("update", kwargs.get("tensor") or result)]
        for name, tensor in params:
            info = _tensor_info(tensor)
            _emit_weight(em, t_start, "update_apply", name, info["bytes"], info["shape"], info["dtype"], _layer_idx(name), None, "rlhf_update", "HBM_PEER_RDMA", "HBM_LOCAL", weight_version=version)
            _emit_weight_transfer(em, "load", name, info["bytes"], "HBM_PEER_RDMA", "HBM_LOCAL", "nccl_send_recv", t_start, layer_idx=_layer_idx(name), weight_version=version, issued_by="weight_update")
        return result

    return wrapper


def _emit_weight(
    em: Any,
    ts_ns: int,
    subtype: str,
    param_name: str,
    bytes_: Optional[int],
    shape: Optional[list[int]],
    dtype: Optional[str],
    layer_idx: Optional[int],
    expert_id: Optional[int],
    reason: str,
    tier_before: str,
    tier_after: str,
    *,
    shard_role: Optional[str] = None,
    lora_adapter_id: Optional[str] = None,
    weight_version: Optional[str] = None,
    owner_request_id: Optional[str] = None,
) -> None:
    em.event({
        "ts_ns": ts_ns,
        "type": "weight_block",
        "subtype": subtype,
        "param_name": param_name,
        "shape": shape,
        "dtype": dtype,
        "bytes": bytes_,
        "layer_idx": layer_idx,
        "expert_id": expert_id,
        "shard_role": shard_role,
        "lora_adapter_id": lora_adapter_id,
        "weight_version": weight_version,
        "tier_before": tier_before,
        "tier_after": tier_after,
        "owner_request_id": owner_request_id,
        "reason": reason,
    })


def _emit_weight_transfer(
    em: Any,
    direction: str,
    param_name: str,
    bytes_: Optional[int],
    src_tier: str,
    dst_tier: str,
    transport: str,
    earliest_known_ts_ns: int,
    *,
    payload_kind: str = "weight",
    layer_idx: Optional[int] = None,
    expert_id: Optional[int] = None,
    lora_adapter_id: Optional[str] = None,
    weight_version: Optional[str] = None,
    request_id: Optional[str] = None,
    issued_by: str = "model_loader",
    backend: Optional[str] = None,
) -> None:
    t0 = now_ns()
    tid = em.transfer_start(
        direction,
        payload_kind=payload_kind,
        request_id=request_id,
        layer_idx=layer_idx,
        src_tier=src_tier,
        dst_tier=dst_tier,
        transport=transport,
        num_blocks=1 if payload_kind == "weight" else None,
        bytes_=bytes_,
        issued_by=issued_by,
        earliest_known_ts_ns=earliest_known_ts_ns,
        param_name=param_name,
        expert_id=expert_id,
        lora_adapter_id=lora_adapter_id,
        weight_version=weight_version,
        weight_backend=backend,
    )
    em.transfer_end(tid, bytes_=bytes_, wire_time_ns=now_ns() - t0)


def _params_from(obj: Any) -> list[tuple[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return [(str(k), v) for k, v in obj.items()]
    for name in ("state_dict", "named_parameters"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                data = fn()
                return list(data.items()) if isinstance(data, dict) else [(str(k), v) for k, v in data]
            except Exception:
                pass
    return []


def _tensor_info(tensor: Any) -> dict[str, Any]:
    shape = getattr(tensor, "shape", None)
    if shape is not None:
        try:
            shape = [int(x) for x in shape]
        except Exception:
            shape = None
    dtype_obj = getattr(tensor, "dtype", None)
    dtype = str(dtype_obj).replace("torch.", "") if dtype_obj is not None else None
    return {"shape": shape, "dtype": dtype, "bytes": _bytes_from(tensor)}


def _bytes_from(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    for attr in ("nbytes", "num_bytes", "bytes", "size_bytes"):
        val = getattr(value, attr, None)
        if isinstance(val, int):
            return val
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return int(numel()) * int(element_size())
        except Exception:
            return None
    return None


def _bytes_for(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[int]:
    for key in ("bytes", "bytes_", "num_bytes", "size_bytes"):
        if key in kwargs:
            val = _bytes_from(kwargs[key])
            if val is not None:
                return val
    for arg in args:
        val = _bytes_from(arg)
        if val is not None:
            return val
    return None


def _param_name(args: tuple[Any, ...], kwargs: dict[str, Any], *, default: str) -> str:
    for key in ("param_name", "name", "weight_name"):
        val = kwargs.get(key)
        if isinstance(val, str):
            return val
    for arg in args:
        if isinstance(arg, str):
            return arg
    return default


def _layer_idx(name: str) -> Optional[int]:
    match = re.search(r"(?:layers?|layer)\.(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def _expert_ids(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[int]:
    value = kwargs.get("expert_ids") or kwargs.get("experts") or kwargs.get("topk_ids")
    if value is None:
        for arg in args:
            if hasattr(arg, "expert_ids"):
                value = getattr(arg, "expert_ids")
                break
    if value is None:
        return []
    try:
        values = value.flatten().tolist()
    except Exception:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    out: list[int] = []
    for item in values:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            pass
    return sorted(set(out))


def _adapter_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[str]:
    for key in ("lora_adapter_id", "adapter_id", "lora_id", "name"):
        val = kwargs.get(key)
        if val is not None:
            return str(val)
    for arg in args:
        for attr in ("lora_adapter_id", "adapter_id", "lora_id", "name"):
            val = getattr(arg, attr, None)
            if val is not None:
                return str(val)
        if isinstance(arg, (str, int)):
            return str(arg)
    return None


def _request_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[str]:
    for key in ("request_id", "req_id", "rid"):
        val = kwargs.get(key)
        if isinstance(val, str):
            return val
    req = kwargs.get("request") or kwargs.get("req")
    for obj in (req, *args):
        for attr in ("request_id", "req_id", "rid", "id"):
            val = getattr(obj, attr, None)
            if isinstance(val, str):
                return val
    return None


def _version_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    h = hashlib.sha1()
    h.update(str(now_ns()).encode())
    h.update(repr(args[:2]).encode(errors="ignore"))
    h.update(repr(sorted(kwargs)).encode(errors="ignore"))
    return "update-" + h.hexdigest()[:12]


def _shard_role(obj: Any) -> Optional[str]:
    parts = []
    for name in ("tp_rank", "pp_rank", "ep_rank"):
        val = getattr(obj, name, None)
        if val is not None:
            parts.append(f"{name}={val}")
    return ";".join(parts) if parts else None


def _coarse_profile(em: Any) -> bool:
    return getattr(getattr(em, "_config", None), "profile", "full") == "coarse"
