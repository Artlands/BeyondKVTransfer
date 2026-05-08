"""SGLang weight and adapter probes — Milestone 9 / Q7."""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from typing import Callable

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
    """Apply import-safe SGLang Q7 monkey patches."""

    global _PATCHES_APPLIED
    with _PATCHES_LOCK:
        if _PATCHES_APPLIED:
            return True

        patched = False
        patched |= _patch_methods(
            "sglang.srt.model_loader.loader",
            {
                "DefaultModelLoader": {"load_model": lambda f, n: make_load_model_wrapper(f, backend="sglang", transport="mmap")},
                "ShardedStateLoader": {"load_model": lambda f, n: make_load_model_wrapper(f, backend="sglang", transport="nccl_allgather")},
            },
        )
        patched |= _patch_methods(
            "sglang.srt.model_executor.model_runner",
            {"ModelRunner": {
                "load_model": lambda f, n: make_load_model_wrapper(f, backend="sglang-runner", transport="mmap"),
                "update_weights_from_distributed": make_update_weights_wrapper,
                "update_weights_from_tensor": make_update_weights_wrapper,
                "update_weights_from_disk": make_update_weights_wrapper,
                "prefetch_layer": make_offload_wrapper,
                "offload_layer": make_offload_wrapper,
            }},
        )
        patched |= _patch_methods(
            "sglang.srt.managers.tp_worker",
            {"TpModelWorker": {
                "__init__": lambda f, n: make_load_model_wrapper(f, backend="sglang-tp-worker", transport="mmap"),
                "update_weights_from_distributed": make_update_weights_wrapper,
                "update_weights_from_tensor": make_update_weights_wrapper,
                "update_weights_from_disk": make_update_weights_wrapper,
            }},
        )
        patched |= _patch_methods(
            "sglang.srt.lora.lora_manager",
            {"LoRAManager": {
                "set_lora_module": make_lora_wrapper,
                "load_lora_adapter": make_lora_wrapper,
                "unload_lora_adapter": make_lora_wrapper,
            }},
        )
        patched |= _patch_methods(
            "sglang.srt.lora.lora",
            {"LoRAAdapter": {"activate": make_lora_wrapper, "deactivate": make_lora_wrapper}},
        )
        patched |= _patch_methods(
            "sglang.srt.layers.moe.fused_moe",
            {"FusedMoE": {"forward": make_moe_wrapper}},
        )
        patched |= _patch_methods(
            "sglang.srt.layers.moe.ep_moe.layer",
            {"EPMoE": {"forward": make_moe_wrapper}},
        )

        for module_name in (
            "sglang.srt.distributed",
            "sglang.srt.distributed.parallel_state",
            "sglang.srt.layers.moe.ep_moe.token_dispatcher",
        ):
            patched |= _patch_module_functions(
                module_name,
                {
                    "all_to_all": lambda f, n: make_collective_wrapper(f, n, transport="nccl_send_recv", default_payload_kind="activation"),
                    "all_gather": lambda f, n: make_collective_wrapper(f, n, transport="nccl_allgather"),
                    "broadcast": lambda f, n: make_collective_wrapper(f, n, transport="nccl_broadcast"),
                },
            )

        _PATCHES_APPLIED = patched
        if patched:
            logger.info("bkvt[sglang]: weight probes active")
        return patched


def _patch_methods(module_name: str, spec: dict[str, dict[str, Callable]]) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.debug("bkvt[sglang]: weight probe skipped %s: %s", module_name, exc)
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
        logger.debug("bkvt[sglang]: collective probe skipped %s: %s", module_name, exc)
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
