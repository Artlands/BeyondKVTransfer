"""
BeyondKVTransfer (bkvt) — Remote Memory Behavior Measurement Framework
for Distributed LLM Serving.

Enable tracing by setting BKVT_ENABLE=1 before launching the engine.
All probes are no-ops when BKVT_ENABLE is unset or 0.

Quickstart::

    import bkvt
    bkvt.init()   # reads config, writes manifest, starts flusher thread
    # ... engine runs, probes fire automatically ...
    bkvt.shutdown()
"""

from bkvt.config import BkvtConfig, get_config
from bkvt.emitter import Emitter, get_emitter

__version__ = "0.1.0"
# Git SHA is injected by CI; the fallback is the version string.
__git_sha__ = "dev"

__all__ = [
    "BkvtConfig",
    "get_config",
    "Emitter",
    "get_emitter",
    "__version__",
    "__git_sha__",
]


def init(config: "BkvtConfig | None" = None) -> "Emitter":
    """Initialise the framework: load config, write manifest, start flusher.

    Safe to call multiple times; subsequent calls are no-ops unless the
    emitter has been shut down.
    """
    from bkvt.emitter import _get_or_create_emitter
    return _get_or_create_emitter(config)


def shutdown() -> None:
    """Flush pending records and stop background threads."""
    from bkvt.emitter import _shutdown_emitter
    _shutdown_emitter()
