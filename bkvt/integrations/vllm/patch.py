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
    5. connector factory      — wrap KVConnectorBase_V1 instances
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
           "runner": bool, "connector": bool}, "reason": str}``
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
                            "runner": False, "connector": False},
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
        connector_ok = _patch_connector_factory()

        any_ok = sched_ok or bp_ok or runner_ok or connector_ok

        if any_ok:
            logger.info(
                "bkvt[vllm]: patches active — scheduler=%s block_pool=%s runner=%s connector=%s",
                sched_ok, bp_ok, runner_ok, connector_ok,
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
                "connector": connector_ok,
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

    _reset_connector_factory_patch()


def is_applied() -> bool:
    """Return True if apply() has been called and patches are active."""
    return _apply_result is not None and _apply_result.get("applied", False)


_CONNECTOR_FACTORY_PATCHED = False
_CONNECTOR_FACTORY_ORIGINAL: Any = None


def _patch_connector_factory() -> bool:
    """Monkey-patch vLLM connector creation to wrap returned connectors."""
    global _CONNECTOR_FACTORY_PATCHED, _CONNECTOR_FACTORY_ORIGINAL

    if _CONNECTOR_FACTORY_PATCHED:
        return True

    try:
        from vllm.distributed.kv_transfer.kv_connector import factory as _factory_mod  # type: ignore
        from bkvt.integrations.vllm.connector_wrapper import wrap_connector
    except Exception as exc:
        logger.debug("bkvt[vllm]: connector factory not patched: %s", exc)
        return False

    cls = getattr(_factory_mod, "KVConnectorFactory", None)
    if cls is None:
        return False

    original = getattr(cls, "create_connector_v1", None)
    if original is None:
        return False

    if getattr(original, "_bkvt_wrapped", False):
        _CONNECTOR_FACTORY_PATCHED = True
        return True

    _CONNECTOR_FACTORY_ORIGINAL = original

    def create_connector_v1_wrapper(*args: Any, **kwargs: Any) -> Any:
        connector = original(*args, **kwargs)
        return wrap_connector(connector)

    create_connector_v1_wrapper._bkvt_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, "create_connector_v1", staticmethod(create_connector_v1_wrapper))
    _CONNECTOR_FACTORY_PATCHED = True
    logger.info("bkvt[vllm]: patched KVConnectorFactory.create_connector_v1")
    return True


def _reset_connector_factory_patch() -> None:
    """Undo connector factory patch for tests."""
    global _CONNECTOR_FACTORY_PATCHED, _CONNECTOR_FACTORY_ORIGINAL
    if not _CONNECTOR_FACTORY_PATCHED or _CONNECTOR_FACTORY_ORIGINAL is None:
        _CONNECTOR_FACTORY_PATCHED = False
        _CONNECTOR_FACTORY_ORIGINAL = None
        return
    try:
        from vllm.distributed.kv_transfer.kv_connector import factory as _factory_mod  # type: ignore
        cls = getattr(_factory_mod, "KVConnectorFactory", None)
        if cls is not None:
            setattr(cls, "create_connector_v1", _CONNECTOR_FACTORY_ORIGINAL)
    except Exception:
        pass
    _CONNECTOR_FACTORY_PATCHED = False
    _CONNECTOR_FACTORY_ORIGINAL = None
