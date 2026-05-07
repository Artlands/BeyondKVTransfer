"""
vLLM connector factory stub for BeyondKVTransfer — M3 compatibility shim.

This module provides the ``tracing_factory`` entry point that is activated by::

    VLLM_KV_CONNECTOR_FACTORY=bkvt.integrations.vllm.factory.tracing_factory

When set, vLLM will call ``tracing_factory(inner_connector)`` instead of
directly using the inner connector.  In M2 this factory is a no-op pass-through
that also applies the M2 patches (so the env var alone can activate all probes).

In M3, this factory will be replaced by the full ``TracingConnectorWrapper``
(§5.4) which wraps ``KVConnectorBase_V1`` methods.

Usage (M2)
----------
Either call ``bkvt.integrations.vllm.patch.apply()`` explicitly, or set::

    BKVT_ENABLE=1
    VLLM_KV_CONNECTOR_FACTORY=bkvt.integrations.vllm.factory.tracing_factory
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory entry point (vLLM reads VLLM_KV_CONNECTOR_FACTORY and calls this)
# ---------------------------------------------------------------------------

def tracing_factory(config: Any, *args: Any, **kwargs: Any) -> Any:
    """Connector factory that activates M2 probes and returns the inner connector.

    vLLM will call this as if it were a ``KVConnectorFactory.create_connector_v1``
    replacement.  We first ensure M2 patches are applied, then delegate to the
    real factory.

    In M3 this function will be replaced by one that wraps the inner connector
    with ``TracingConnectorWrapper``.
    """
    from bkvt.integrations.vllm.patch import apply as _apply
    result = _apply()
    if not result.get("applied"):
        logger.debug(
            "bkvt[vllm]: factory called but probes not applied (%s)",
            result.get("reason"),
        )

    # Delegate to vLLM's own factory so the inner connector is created normally.
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (  # type: ignore
            KVConnectorFactory,
        )
        return KVConnectorFactory.create_connector_v1(config, *args, **kwargs)
    except Exception as exc:
        logger.warning(
            "bkvt[vllm]: could not delegate to KVConnectorFactory: %s", exc
        )
        return None


# ---------------------------------------------------------------------------
# M3 placeholder — TracingConnectorWrapper stub
# ---------------------------------------------------------------------------

class TracingConnectorWrapper:
    """Placeholder for the M3 tracing connector wrapper (§5.4).

    In M2 this class is not used.  It is defined here so that M3 can
    import and extend it without changing the module structure.

    The full implementation in M3 will subclass ``KVConnectorBase_V1`` and
    wrap all lifecycle methods (get_num_new_matched_tokens, start_load_kv,
    wait_for_layer_load, save_kv_layer, wait_for_save, get_finished, …) with
    transfer/start and transfer/end probe calls.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        # Pass through all attribute access to the inner connector.
        return getattr(self.inner, name)
