"""
SGLang monkey-patch entrypoint for BeyondKVTransfer -- Milestones 4 and 5.

Usage:

    BKVT_ENABLE=1 python -c "from bkvt.integrations.sglang.patch import apply; apply()"

The entrypoint is idempotent and import-safe when SGLang is not installed.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from bkvt import emitter as _emitter_mod
from bkvt.config import get_config

logger = logging.getLogger(__name__)

_apply_lock = threading.Lock()
_apply_result: Optional[dict] = None


def apply(config: Any = None) -> dict:
    """Apply all SGLang probes."""
    global _apply_result

    with _apply_lock:
        if _apply_result is not None:
            return _apply_result

        cfg = config or get_config()
        if not cfg.enabled:
            _apply_result = {
                "applied": False,
                "reason": "BKVT_ENABLE=0",
                "modules": {
                    "scheduler": False,
                    "radix": False,
                    "hicache": False,
                    "disagg": False,
                    "weight": False,
                },
            }
            return _apply_result

        _emitter_mod._get_or_create_emitter(cfg)

        from bkvt.integrations.sglang import (
            disagg_probe,
            hicache_probe,
            radix_probe,
            scheduler_probe,
            weight_probe,
        )

        scheduler_ok = scheduler_probe.apply_patches()
        radix_ok = radix_probe.apply_patches()
        hicache_ok = hicache_probe.apply_patches()
        disagg_ok = disagg_probe.apply_patches()
        weight_ok = weight_probe.apply_patches()
        any_ok = scheduler_ok or radix_ok or hicache_ok or disagg_ok or weight_ok

        if any_ok:
            logger.info(
                "bkvt[sglang]: patches active -- scheduler=%s radix=%s hicache=%s disagg=%s weight=%s",
                scheduler_ok,
                radix_ok,
                hicache_ok,
                disagg_ok,
                weight_ok,
            )
        else:
            logger.warning(
                "bkvt[sglang]: no patches applied -- SGLang may not be installed "
                "or all probe sites were unreachable"
            )

        _apply_result = {
            "applied": any_ok,
            "reason": "ok" if any_ok else "no_probe_sites_found",
            "modules": {
                "scheduler": scheduler_ok,
                "radix": radix_ok,
                "hicache": hicache_ok,
                "disagg": disagg_ok,
                "weight": weight_ok,
            },
        }
        return _apply_result


def reset() -> None:
    """Reset applied state for tests."""
    global _apply_result
    with _apply_lock:
        _apply_result = None

    import importlib
    for mod_name in (
        "bkvt.integrations.sglang.scheduler_probe",
        "bkvt.integrations.sglang.radix_probe",
        "bkvt.integrations.sglang.hicache_probe",
        "bkvt.integrations.sglang.disagg_probe",
        "bkvt.integrations.sglang.weight_probe",
    ):
        try:
            mod = importlib.import_module(mod_name)
            mod._PATCHES_APPLIED = False  # type: ignore[attr-defined]
        except Exception:
            pass


def is_applied() -> bool:
    return _apply_result is not None and _apply_result.get("applied", False)
