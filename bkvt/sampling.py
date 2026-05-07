"""
Sampling helpers for BeyondKVTransfer (§9 / §10).

Per-probe sampling rates are read from ``BkvtConfig`` and frozen at
engine init.  The ``Sampler`` class provides fast, deterministic
decisions and correctly attaches ``sample_decision`` metadata to records
that were sampled.

Always-on probes (never sampled)
---------------------------------
* Request-level events: arrival, admit, preempt, resume, finish, abort.
* ``first_token`` token event.
* ``clock_anchor`` metadata records.
* ``scheduler_decision`` metadata records.

Sampled probes
--------------
* ``token`` (prefill_chunk, decode): rate ``sample_token``.
* ``metadata`` (fine-grained): rate ``sample_metadata``.
* ``transfer``: rate ``sample_transfer``.
* ``kv_block``: rate ``sample_metadata`` (same knob, scheduler-side).

The ``sample_decision`` field is added to records only when the probe is
sampled (i.e. rate < 1.0), so analyses can normalise counts correctly.
"""

from __future__ import annotations

import random
import threading
from typing import Optional


# ---------------------------------------------------------------------------
# Always-on guard sets
# ---------------------------------------------------------------------------

# Record types whose events are ALWAYS emitted regardless of sampling rate.
_ALWAYS_ON_TYPES: frozenset[str] = frozenset({"request", "sys_counter"})

# Subtypes within the "token" type that are always emitted.
_ALWAYS_ON_TOKEN_SUBTYPES: frozenset[str] = frozenset({"first_token"})

# Subtypes within "metadata" that are always emitted.
_ALWAYS_ON_METADATA_SUBTYPES: frozenset[str] = frozenset(
    {"clock_anchor", "scheduler_decision"}
)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class Sampler:
    """Thread-safe sampling decision engine.

    Each call to ``should_emit`` draws a single uniform random float and
    compares it to the relevant rate.  The ``_rng`` is per-thread to
    avoid lock contention on the hot path.
    """

    def __init__(
        self,
        sample_token: float = 0.05,
        sample_metadata: float = 1.0,
        sample_transfer: float = 1.0,
    ) -> None:
        self.sample_token = sample_token
        self.sample_metadata = sample_metadata
        self.sample_transfer = sample_transfer
        self._local = threading.local()

    @property
    def _rng(self) -> random.Random:
        rng = getattr(self._local, "rng", None)
        if rng is None:
            rng = random.Random()
            self._local.rng = rng
        return rng

    # ------------------------------------------------------------------
    # Core decision method
    # ------------------------------------------------------------------

    def should_emit(
        self,
        record_type: str,
        subtype: str = "",
    ) -> tuple[bool, Optional[float]]:
        """Decide whether a record should be emitted.

        Returns ``(emit: bool, rate: float | None)``.
        ``rate`` is ``None`` for always-on events (no ``sample_decision``
        field needed on those records).
        ``rate`` is the effective rate (e.g. 0.05) for sampled events,
        so the caller can attach it to the record.
        """
        # Always-on by type
        if record_type in _ALWAYS_ON_TYPES:
            return True, None

        if record_type == "token":
            if subtype in _ALWAYS_ON_TOKEN_SUBTYPES:
                return True, None
            rate = self.sample_token
        elif record_type == "metadata":
            if subtype in _ALWAYS_ON_METADATA_SUBTYPES:
                return True, None
            rate = self.sample_metadata
        elif record_type == "transfer":
            rate = self.sample_transfer
        elif record_type == "kv_block":
            rate = self.sample_metadata   # same knob
        else:
            # Unknown type — emit by default
            return True, None

        if rate >= 1.0:
            return True, None  # effectively always-on, skip field

        emit = self._rng.random() < rate
        return emit, rate

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def should_emit_token(self, subtype: str = "") -> tuple[bool, Optional[float]]:
        return self.should_emit("token", subtype)

    def should_emit_metadata(self, subtype: str = "") -> tuple[bool, Optional[float]]:
        return self.should_emit("metadata", subtype)

    def should_emit_transfer(self) -> tuple[bool, Optional[float]]:
        return self.should_emit("transfer")

    def should_emit_kv_block(self, subtype: str = "") -> tuple[bool, Optional[float]]:
        return self.should_emit("kv_block", subtype)


# ---------------------------------------------------------------------------
# Module-level singleton (initialised by emitter at engine init)
# ---------------------------------------------------------------------------

_sampler: Optional[Sampler] = None
_sampler_lock = threading.Lock()


def init_sampler(
    sample_token: float = 0.05,
    sample_metadata: float = 1.0,
    sample_transfer: float = 1.0,
) -> Sampler:
    """Create (or replace) the process-wide ``Sampler`` singleton."""
    global _sampler
    s = Sampler(
        sample_token=sample_token,
        sample_metadata=sample_metadata,
        sample_transfer=sample_transfer,
    )
    with _sampler_lock:
        _sampler = s
    return s


def get_sampler() -> Sampler:
    """Return the process-wide sampler, creating a default one if needed."""
    global _sampler
    if _sampler is not None:
        return _sampler
    with _sampler_lock:
        if _sampler is None:
            _sampler = Sampler()
    return _sampler
