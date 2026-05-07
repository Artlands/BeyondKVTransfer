"""
SGLang monkey-patch entrypoint for BeyondKVTransfer -- Milestone 4.

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
    """Apply all M4 SGLang probes."""
    global _apply_result

    with _apply_lock:
        if _apply_result is not None:
            return _apply_result

        cfg = config or get_config()
        if not cfg.enabled:
            _apply_result = {
                "applied": False,
                "reason": "BKVT_ENABLE=0",
                "modules": {"scheduler": False, "radix": False},
            }
            return _apply_result

        _emitter_mod._get_or_create_emitter(cfg)

        from bkvt.integrations.sglang import radix_probe, scheduler_probe

        scheduler_ok = scheduler_probe.apply_patches()
        radix_ok = radix_probe.apply_patches()
        any_ok = scheduler_ok or radix_ok

        if any_ok:
            logger.info(
                "bkvt[sglang]: patches active -- scheduler=%s radix=%s",
                scheduler_ok,
                radix_ok,
            )
        else:
            logger.warning(
                "bkvt[sglang]: no patches applied -- SGLang may not be installed "
                "or all probe sites were unreachable"
            )

        _apply_result = {
            "applied": any_ok,
            "reason": "ok" if any_ok else "no_probe_sites_found",
            "modules": {"scheduler": scheduler_ok, "radix": radix_ok},
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
    ):
        try:
            mod = importlib.import_module(mod_name)
            mod._PATCHES_APPLIED = False  # type: ignore[attr-defined]
        except Exception:
            pass


def is_applied() -> bool:
    return _apply_result is not None and _apply_result.get("applied", False)
