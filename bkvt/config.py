"""
Configuration loader for BeyondKVTransfer (§12).

All runtime settings live under the ``BKVT_*`` environment-variable
prefix.  A YAML file at ``${BKVT_OUTPUT_DIR}/config.yaml`` takes
precedence over env vars if it exists.

The ``BkvtConfig`` dataclass is the single source of truth for
configuration within the process.  It is frozen at the point
``get_config()`` is first called (i.e. at engine init).  All downstream
modules obtain the same singleton via ``get_config()``.

Environment variables (§12)
---------------------------
BKVT_ENABLE          0          master on/off switch (1 = enabled)
BKVT_OUTPUT_DIR      ./bkvt_traces  trace root directory
BKVT_TRACE_ID        <auto>     override trace UUID for reproducible runs
BKVT_PROFILE         full       full | coarse | request_only
BKVT_SAMPLE_TOKEN    0.05       per-token record sampling rate [0,1]
BKVT_SAMPLE_METADATA 1.0        metadata record sampling rate [0,1]
BKVT_SAMPLE_TRANSFER 1.0        transfer record sampling rate [0,1]
BKVT_ROTATE_BYTES    268435456  jsonl rotation threshold (bytes)
BKVT_FLUSH_BYTES     4194304    flusher batch size (bytes)
BKVT_SYS_COUNTER_HZ  10        sys_counter poll rate (Hz)
BKVT_NCCL_PROFILER   0          load native NCCL profiler plugin
BKVT_CLOCK_ANCHOR_HZ 1          how often to re-emit clock_anchor
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Profile enum constants
# ---------------------------------------------------------------------------

PROFILE_FULL = "full"
PROFILE_COARSE = "coarse"
PROFILE_REQUEST_ONLY = "request_only"
_VALID_PROFILES = {PROFILE_FULL, PROFILE_COARSE, PROFILE_REQUEST_ONLY}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class BkvtConfig:
    """Immutable configuration snapshot for one engine init."""

    # Master switch
    enabled: bool = False

    # Output
    output_dir: str = "./bkvt_traces"
    trace_id: Optional[str] = None  # None → auto-generate at init

    # Profile
    profile: str = PROFILE_FULL

    # Sampling rates
    sample_token: float = 0.05
    sample_metadata: float = 1.0
    sample_transfer: float = 1.0

    # I/O tuning
    rotate_bytes: int = 268_435_456   # 256 MiB
    flush_bytes: int = 4_194_304      # 4 MiB

    # Collectors
    sys_counter_hz: float = 10.0
    nccl_profiler: bool = False
    clock_anchor_hz: float = 1.0

    # Extra raw env snapshot (for manifest)
    _raw_env: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.profile not in _VALID_PROFILES:
            raise ValueError(
                f"Invalid BKVT_PROFILE={self.profile!r}; "
                f"must be one of {sorted(_VALID_PROFILES)}"
            )
        for name, val in [
            ("sample_token", self.sample_token),
            ("sample_metadata", self.sample_metadata),
            ("sample_transfer", self.sample_transfer),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name}={val} out of range [0, 1]")

    # ------------------------------------------------------------------
    # Helpers for downstream modules
    # ------------------------------------------------------------------

    @property
    def is_full(self) -> bool:
        return self.profile == PROFILE_FULL

    @property
    def is_coarse(self) -> bool:
        return self.profile == PROFILE_COARSE

    @property
    def is_request_only(self) -> bool:
        return self.profile == PROFILE_REQUEST_ONLY


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip() not in ("", "0", "false", "no", "off")


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("bkvt: invalid value for %s=%r, using default %s", key, raw, default)
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("bkvt: invalid value for %s=%r, using default %s", key, raw, default)
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _load_from_env() -> BkvtConfig:
    raw_env = {k: v for k, v in os.environ.items() if k.startswith("BKVT_")}
    return BkvtConfig(
        enabled=_env_bool("BKVT_ENABLE", False),
        output_dir=_env_str("BKVT_OUTPUT_DIR", "./bkvt_traces"),
        trace_id=os.environ.get("BKVT_TRACE_ID"),  # None if not set
        profile=_env_str("BKVT_PROFILE", PROFILE_FULL),
        sample_token=_env_float("BKVT_SAMPLE_TOKEN", 0.05),
        sample_metadata=_env_float("BKVT_SAMPLE_METADATA", 1.0),
        sample_transfer=_env_float("BKVT_SAMPLE_TRANSFER", 1.0),
        rotate_bytes=_env_int("BKVT_ROTATE_BYTES", 268_435_456),
        flush_bytes=_env_int("BKVT_FLUSH_BYTES", 4_194_304),
        sys_counter_hz=_env_float("BKVT_SYS_COUNTER_HZ", 10.0),
        nccl_profiler=_env_bool("BKVT_NCCL_PROFILER", False),
        clock_anchor_hz=_env_float("BKVT_CLOCK_ANCHOR_HZ", 1.0),
        _raw_env=raw_env,
    )


def _apply_yaml_overrides(cfg: BkvtConfig) -> BkvtConfig:
    """Overlay settings from ${BKVT_OUTPUT_DIR}/config.yaml if present."""
    yaml_path = os.path.join(cfg.output_dir, "config.yaml")
    if not os.path.isfile(yaml_path):
        return cfg

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("bkvt: pyyaml not installed; skipping %s", yaml_path)
        return cfg

    try:
        with open(yaml_path) as fh:
            overrides: dict = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("bkvt: could not read %s: %s", yaml_path, exc)
        return cfg

    # Map YAML keys (lower-case, no BKVT_ prefix) to field names.
    _key_map = {
        "enable": "enabled",
        "enabled": "enabled",
        "output_dir": "output_dir",
        "trace_id": "trace_id",
        "profile": "profile",
        "sample_token": "sample_token",
        "sample_metadata": "sample_metadata",
        "sample_transfer": "sample_transfer",
        "rotate_bytes": "rotate_bytes",
        "flush_bytes": "flush_bytes",
        "sys_counter_hz": "sys_counter_hz",
        "nccl_profiler": "nccl_profiler",
        "clock_anchor_hz": "clock_anchor_hz",
    }

    updates: dict = {}
    for yaml_key, value in overrides.items():
        field_name = _key_map.get(yaml_key.lower())
        if field_name:
            updates[field_name] = value
        else:
            logger.warning("bkvt: unknown config key %r in %s", yaml_key, yaml_path)

    if not updates:
        return cfg

    # Rebuild with overrides applied.
    from dataclasses import asdict
    d = asdict(cfg)
    d.update(updates)
    return BkvtConfig(**d)


def load_config() -> BkvtConfig:
    """Load, validate, and return a new ``BkvtConfig`` from the environment
    and optional YAML file.  Does *not* cache the result; use ``get_config()``
    for the singleton.
    """
    cfg = _load_from_env()
    cfg = _apply_yaml_overrides(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_lock = threading.Lock()
_config: Optional[BkvtConfig] = None


def get_config() -> BkvtConfig:
    """Return the process-wide ``BkvtConfig`` singleton.

    Loads from env/YAML on first call; subsequent calls return the cached
    instance.  Thread-safe.
    """
    global _config
    if _config is not None:
        return _config
    with _config_lock:
        if _config is None:
            _config = load_config()
            if _config.enabled:
                logger.info(
                    "bkvt: enabled — profile=%s output_dir=%s",
                    _config.profile,
                    _config.output_dir,
                )
    return _config


def reset_config(new_config: Optional[BkvtConfig] = None) -> None:
    """Replace the singleton — intended for tests only.

    If ``new_config`` is None the singleton is cleared so the next call to
    ``get_config()`` will reload from the environment.
    """
    global _config
    with _config_lock:
        _config = new_config
