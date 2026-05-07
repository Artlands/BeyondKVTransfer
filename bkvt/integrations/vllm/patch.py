"""
vLLM monkey-patch entrypoint for BeyondKVTransfer — Milestone 2.

Usage
-----
The recommended way to activate tracing is to set the env var::

    BKVT_ENABLE=1

and then, before starting the vLLM engine, import and call::

    from bkvt.integrations.vllm.patch import apply
    apply()

Alternatively, for the connector-based activation path (M3+)::

    VLLM_KV_CONNECTOR_FACTORY=bkvt.integrations.vllm.factory.tracing_factory

Design constraints
------------------
* apply() is idempotent — safe to call multiple times.
* apply() returns immediately (no-op) when BKVT_ENABLE=0.
* Each sub-module's apply_patches() is called in dependency order:
    1. emitter.init()         — start the background flusher
    2. scheduler_probe        — arrival / admit / preempt / scheduler_decision
    3. block_pool_probe       — allocate / free / evict / prefix_hit / metadata
    4. runner_probe           — token records
* API-version checking: if the wrapped function's signature has changed
  since this code was written a one-time WARN is emitted (§15.1).
* apply() returns a dict summarising what was and wasn't patched, useful
  for logging at engine startup.
"""

from __future__ import annotations

import inspect
import logging
import threading
from typing import Any, Optional

from bkvt import emitter as _emitter_mod
from bkvt.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version pin  (update together with pyproject.toml pin when upgrading vLLM)
# ---------------------------------------------------------------------------

# Expected signatures of the probe sites.  If the actual running vLLM has a
# different signature we emit a one-time WARN (§15.1).
_EXPECTED_SIGNATURES: dict[str, str] = {
    "Processor.process_inputs":
        "(self, request_id, prompt, params, arrival_time",
    "Scheduler.schedule":
        "(self)",
    "GPUModelRunner.execute_model":
        "(self, scheduler_output",
    "BlockPool.alloc_blocks":
        "(self, num_blocks",
    "BlockPool.free_blocks":
        "(self, block_ids",
    "KVCacheManager.get_computed_blocks":
        "(self, request",
}

_warned_signatures: set[str] = set()


def _check_signature(cls: Any, method_name: str, sig_key: str) -> None:
    """Emit a one-time WARN if the method signature looks different."""
    expected_fragment = _EXPECTED_SIGNATURES.get(sig_key, "")
    if not expected_fragment:
        return
    method = getattr(cls, method_name, None)
    if method is None:
        return
    try:
        sig = str(inspect.signature(method))
    except (ValueError, TypeError):
        return
    if not sig.startswith(expected_fragment):
        key = f"sig_{sig_key}"
        if key not in _warned_signatures:
            _warned_signatures.add(key)
            logger.warning(
                "bkvt[vllm]: %s signature mismatch — expected to start with "
                "%r, got %r. "
                "Probe will still be applied but may capture incomplete data. "
                "Update bkvt/integrations/vllm/patch.py if vLLM was upgraded.",
                sig_key, expected_fragment, sig,
            )


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_apply_lock = threading.Lock()
_apply_result: Optional[dict] = None   # None → not yet applied


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply(config: Any = None) -> dict:
    """Apply all M2 vLLM probes.  Idempotent.

    Parameters
    ----------
    config : BkvtConfig, optional
        If None the singleton from ``bkvt.config.get_config()`` is used.

    Returns
    -------
    dict
        ``{"applied": bool, "modules": {"scheduler": bool, "block_pool": bool,
           "runner": bool}, "reason": str}``
    """
    global _apply_result

    with _apply_lock:
        if _apply_result is not None:
            return _apply_result

        cfg = config or get_config()
        if not cfg.enabled:
            _apply_result = {
                "applied": False,
                "reason": "BKVT_ENABLE=0",
                "modules": {"scheduler": False, "block_pool": False,
                            "runner": False},
            }
            return _apply_result

        # Initialise the emitter singleton (writes manifest.json too)
        em = _emitter_mod._get_or_create_emitter(cfg)

        # ── Signature checks (advisory only) ────────────────────────────
        try:
            from vllm.v1.engine import processor as _proc_mod  # type: ignore
            _check_signature(_proc_mod.Processor,
                             "process_inputs", "Processor.process_inputs")
        except ImportError:
            pass

        try:
            from vllm.v1.core.sched import scheduler as _sched_mod  # type: ignore
            _check_signature(_sched_mod.Scheduler,
                             "schedule", "Scheduler.schedule")
        except ImportError:
            pass

        try:
            from vllm.v1.worker import gpu_model_runner as _gmr_mod  # type: ignore
            _check_signature(_gmr_mod.GPUModelRunner,
                             "execute_model", "GPUModelRunner.execute_model")
        except ImportError:
            pass

        # ── Apply probes ─────────────────────────────────────────────────
        from bkvt.integrations.vllm import scheduler_probe, block_pool_probe, runner_probe

        sched_ok = scheduler_probe.apply_patches()
        bp_ok = block_pool_probe.apply_patches()
        runner_ok = runner_probe.apply_patches()

        any_ok = sched_ok or bp_ok or runner_ok

        if any_ok:
            logger.info(
                "bkvt[vllm]: M2 patches active — scheduler=%s block_pool=%s runner=%s",
                sched_ok, bp_ok, runner_ok,
            )
        else:
            logger.warning(
                "bkvt[vllm]: no patches applied — vLLM may not be installed "
                "or all probe sites were unreachable"
            )

        _apply_result = {
            "applied": any_ok,
            "reason": "ok" if any_ok else "no_probe_sites_found",
            "modules": {
                "scheduler": sched_ok,
                "block_pool": bp_ok,
                "runner": runner_ok,
            },
        }
        return _apply_result


def reset() -> None:
    """Reset the applied state — intended for testing only."""
    global _apply_result
    with _apply_lock:
        _apply_result = None

    # Also reset sub-module flags
    import importlib
    for mod_name in (
        "bkvt.integrations.vllm.scheduler_probe",
        "bkvt.integrations.vllm.block_pool_probe",
        "bkvt.integrations.vllm.runner_probe",
    ):
        try:
            mod = importlib.import_module(mod_name)
            mod._PATCHES_APPLIED = False  # type: ignore[attr-defined]
        except Exception:
            pass


def is_applied() -> bool:
    """Return True if apply() has been called and patches are active."""
    return _apply_result is not None and _apply_result.get("applied", False)
