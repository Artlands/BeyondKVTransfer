"""
vLLM connector factory for BeyondKVTransfer — Milestone 3.

This module provides the ``tracing_factory`` entry point that is activated by::

    VLLM_KV_CONNECTOR_FACTORY=bkvt.integrations.vllm.factory.tracing_factory

When set, vLLM calls this entry point instead of directly creating the
configured connector.  We apply the M2 probes, delegate to vLLM's native
factory, then wrap the resulting connector with ``TracingConnectorWrapper``.

"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from bkvt.integrations.vllm.connector_wrapper import (
    TracingConnectorWrapper,
    wrap_connector,
)

logger = logging.getLogger(__name__)

_IN_TRACING_FACTORY = False


# ---------------------------------------------------------------------------
# Factory entry point (vLLM reads VLLM_KV_CONNECTOR_FACTORY and calls this)
# ---------------------------------------------------------------------------

def tracing_factory(config: Any, *args: Any, **kwargs: Any) -> Any:
    """Create the configured vLLM connector and wrap it for tracing.

    The exact vLLM factory signature has changed across releases, so this
    function preserves ``*args``/``**kwargs`` and only assumes the first
    argument is the connector configuration.
    """
    global _IN_TRACING_FACTORY

    from bkvt.integrations.vllm.patch import apply as _apply
    result = _apply()
    if not result.get("applied"):
        logger.debug(
            "bkvt[vllm]: factory called but probes not applied (%s)",
            result.get("reason"),
        )

    if isinstance(config, TracingConnectorWrapper):
        return config

    if _IN_TRACING_FACTORY:
        _original = _original_factory()
        if _original is None:
            return None
        return _original(config, *args, **kwargs)

    # Delegate to vLLM's own factory so the inner connector is created normally.
    # Some vLLM builds consult VLLM_KV_CONNECTOR_FACTORY inside the factory; we
    # temporarily remove this env var to avoid recursively calling ourselves.
    try:
        _original = _original_factory()
        if _original is None:
            return None

        old_env = os.environ.pop("VLLM_KV_CONNECTOR_FACTORY", None)
        _IN_TRACING_FACTORY = True
        try:
            connector = _original(config, *args, **kwargs)
        finally:
            _IN_TRACING_FACTORY = False
            if old_env is not None:
                os.environ["VLLM_KV_CONNECTOR_FACTORY"] = old_env

        return wrap_connector(connector)
    except Exception as exc:
        logger.warning(
            "bkvt[vllm]: could not delegate to KVConnectorFactory: %s", exc
        )
        return None
def _original_factory() -> Optional[Any]:
    """Return vLLM's native create_connector_v1 callable if importable."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (  # type: ignore
            KVConnectorFactory,
        )
    except Exception as exc:
        logger.warning("bkvt[vllm]: KVConnectorFactory unavailable: %s", exc)
        return None
    return getattr(KVConnectorFactory, "create_connector_v1", None)
