"""vLLM weight and adapter probes — Milestone 9 / Q7."""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from typing import Any, Callable

from bkvt.integrations.weight_probe_common import (
    make_collective_wrapper,
    make_load_model_wrapper,
    make_lora_wrapper,
    make_moe_wrapper,
    make_offload_wrapper,
    make_update_weights_wrapper,
    payload_context,
)

logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False
_PATCHES_LOCK = threading.Lock()


def apply_patches() -> bool:
    """Apply import-safe vLLM Q7 monkey patches."""

    global _PATCHES_APPLIED
    with _PATCHES_LOCK:
        if _PATCHES_APPLIED:
            return True

        patched = False
        patched |= _patch_methods(
            "vllm.model_executor.model_loader.loader",
            {
                "DefaultModelLoader": {"load_model": lambda f, n: make_load_model_wrapper(f, backend="vllm", transport="mmap")},
                "ShardedStateLoader": {"load_model": lambda f, n: make_load_model_wrapper(f, backend="vllm", transport="nccl_allgather")},
                "RunaiModelStreamerLoader": {"load_model": lambda f, n: make_load_model_wrapper(f, backend="vllm", transport="object_store")},
                "BitsAndBytesModelLoader": {"load_model": lambda f, n: make_load_model_wrapper(f, backend="vllm", transport="mmap")},
                "TensorizerLoader": {"load_model": lambda f, n: make_load_model_wrapper(f, backend="vllm", transport="mmap")},
            },
        )
        patched |= _patch_methods(
            "vllm.v1.worker.gpu_worker",
            {"Worker": {"load_model": lambda f, n: make_load_model_wrapper(f, backend="vllm-worker", transport="mmap")}},
        )
        patched |= _patch_methods(
            "vllm.v1.worker.gpu_model_runner",
            {"GPUModelRunner": {"prefetch_layer": make_offload_wrapper, "offload_layer": make_offload_wrapper}},
        )
        patched |= _patch_methods(
            "vllm.model_executor.layers.fused_moe.layer",
            {"FusedMoE": {"forward": make_moe_wrapper}},
        )
        patched |= _patch_methods(
            "vllm.lora.worker_manager",
            {"WorkerLoRAManager": {
                "set_active_loras": make_lora_wrapper,
                "add_dummy_lora": make_lora_wrapper,
                "remove_lora_from_cache": make_lora_wrapper,
            }},
        )
        patched |= _patch_methods(
            "vllm.lora.models",
            {"LoRAModelManager": {
                "activate_adapter": make_lora_wrapper,
                "add_adapter": make_lora_wrapper,
                "remove_adapter": make_lora_wrapper,
            }},
        )
        patched |= _patch_methods(
            "vllm.v1.worker.worker_base",
            {"Worker": {
                "update_weights_from_distributed": make_update_weights_wrapper,
                "update_weights_from_tensor": make_update_weights_wrapper,
            }},
        )
        patched |= _patch_methods(
            "vllm.worker.worker",
            {"Worker": {
                "update_weights_from_distributed": make_update_weights_wrapper,
                "update_weights_from_tensor": make_update_weights_wrapper,
            }},
        )
        patched |= _patch_module_functions(
            "vllm.distributed.parallel_state",
            {
                "broadcast_tensor_dict": lambda f, n: make_collective_wrapper(f, n, transport="nccl_broadcast"),
                "tensor_model_parallel_all_gather": lambda f, n: make_collective_wrapper(f, n, transport="nccl_allgather"),
                "all_to_all": lambda f, n: make_collective_wrapper(f, n, transport="nccl_send_recv", default_payload_kind="activation"),
            },
        )

        _PATCHES_APPLIED = patched
        if patched:
            logger.info("bkvt[vllm]: weight probes active")
        return patched


def _patch_methods(module_name: str, spec: dict[str, dict[str, Callable]]) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.debug("bkvt[vllm]: weight probe skipped %s: %s", module_name, exc)
        return False

    patched = False
    for class_name, methods in spec.items():
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        for method_name, factory in methods.items():
            original = getattr(cls, method_name, None)
            if original is None or getattr(original, "_bkvt_wrapped", False):
                continue
            wrapped = factory(original, method_name)
            wrapped._bkvt_wrapped = True  # type: ignore[attr-defined]
            setattr(cls, method_name, wrapped)
            patched = True
    return patched


def _patch_module_functions(module_name: str, spec: dict[str, Callable]) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.debug("bkvt[vllm]: collective probe skipped %s: %s", module_name, exc)
        return False

    patched = False
    for name, factory in spec.items():
        original = getattr(module, name, None)
        if original is None or not (inspect.isfunction(original) or inspect.ismethod(original)):
            continue
        if getattr(original, "_bkvt_wrapped", False):
            continue
        wrapped = factory(original, name)
        wrapped._bkvt_wrapped = True  # type: ignore[attr-defined]
        setattr(module, name, wrapped)
        patched = True
    return patched


__all__ = [
    "apply_patches",
    "make_collective_wrapper",
    "make_load_model_wrapper",
    "make_lora_wrapper",
    "make_moe_wrapper",
    "make_offload_wrapper",
    "make_update_weights_wrapper",
    "payload_context",
]
