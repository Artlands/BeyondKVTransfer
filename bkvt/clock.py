"""
Clock utilities for BeyondKVTransfer (§3.2).

Design rules
------------
* All timestamps are **integer nanoseconds since CLOCK_MONOTONIC_RAW**
  (Linux).  On non-Linux platforms we fall back to CLOCK_MONOTONIC
  (``time.monotonic_ns()``), which is still monotonic but may drift
  differently from CLOCK_REALTIME.  A one-time warning is emitted.
* A single ``t0_unix_ns`` is captured at module import time to anchor
  monotonic time to wall-clock for post-hoc analysis.
* CUDA event timing is optional and gated on ``torch`` / CUDA being
  available.  When unavailable, the ``CudaEventTimer`` class is a no-op
  that returns ``None`` for GPU timestamps.

Public API
----------
``now_ns() -> int``
    Current monotonic timestamp in nanoseconds.

``unix_now_ns() -> int``
    Current UNIX wall-clock time in nanoseconds (``time.time_ns()``).

``t0_unix_ns: int``
    UNIX nanoseconds at the time this module was first imported.

``t0_monotonic_ns: int``
    Monotonic nanoseconds at the time this module was first imported
    (same instant as ``t0_unix_ns``).

``CudaEventTimer``
    Context manager / helper for bracketing CUDA kernels with
    ``cudaEvent_t``-based timing.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
import warnings
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform detection — prefer CLOCK_MONOTONIC_RAW on Linux
# ---------------------------------------------------------------------------

_CLOCK_MONOTONIC_RAW: Optional[int] = None
_libc: Optional[ctypes.CDLL] = None

if os.name == "posix":
    try:
        import ctypes.util
        _libc_name = ctypes.util.find_library("c") or "libc.so.6"
        _libc = ctypes.CDLL(_libc_name, use_errno=True)

        # struct timespec { time_t tv_sec; long tv_nsec; }
        class _Timespec(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

        # CLOCK_MONOTONIC_RAW = 4 on Linux
        _CLOCK_MONOTONIC_RAW = 4
        _ts = _Timespec()
        rc = _libc.clock_gettime(_CLOCK_MONOTONIC_RAW, ctypes.byref(_ts))
        if rc != 0:
            # Not supported on this kernel
            _CLOCK_MONOTONIC_RAW = None
            _libc = None
    except Exception:
        _CLOCK_MONOTONIC_RAW = None
        _libc = None

if _CLOCK_MONOTONIC_RAW is None:
    warnings.warn(
        "bkvt.clock: CLOCK_MONOTONIC_RAW not available; "
        "falling back to time.monotonic_ns(). "
        "Cross-node clock skew correction may be less accurate.",
        stacklevel=1,
    )


# Build a fast, allocation-free path when CLOCK_MONOTONIC_RAW is available.
if _libc is not None and _CLOCK_MONOTONIC_RAW is not None:
    class _Timespec(ctypes.Structure):  # type: ignore[no-redef]
        _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

    _ts_buf = _Timespec()
    _clock_gettime = _libc.clock_gettime
    _clock_id = _CLOCK_MONOTONIC_RAW

    def now_ns() -> int:
        """Return CLOCK_MONOTONIC_RAW in nanoseconds (Linux fast path)."""
        _clock_gettime(_clock_id, ctypes.byref(_ts_buf))
        return _ts_buf.tv_sec * 1_000_000_000 + _ts_buf.tv_nsec

else:
    def now_ns() -> int:  # type: ignore[misc]
        """Return a monotonic nanosecond timestamp (portable fallback)."""
        return time.monotonic_ns()


def unix_now_ns() -> int:
    """Return the current UNIX wall-clock time in nanoseconds."""
    return time.time_ns()


# ---------------------------------------------------------------------------
# Anchor timestamps — captured once at module import
# ---------------------------------------------------------------------------

# Capture both clocks in rapid succession so the offset is tight.
t0_unix_ns: int = unix_now_ns()
t0_monotonic_ns: int = now_ns()


# ---------------------------------------------------------------------------
# CUDA event timing helpers
# ---------------------------------------------------------------------------

_CUDA_AVAILABLE: bool = False
try:
    import torch  # type: ignore[import-untyped]
    if torch.cuda.is_available():
        _CUDA_AVAILABLE = True
except ImportError:
    pass


class CudaEventTimer:
    """Bracket GPU work with CUDA event timing.

    Usage::

        timer = CudaEventTimer()
        timer.record_start()
        # ... dispatch kernel(s) ...
        timer.record_end()
        # ... later, after synchronisation ...
        elapsed_ms = timer.elapsed_ms()   # may block until done
        start_ns   = timer.start_ns()     # monotonic start ts
        end_ns     = timer.end_ns()       # approximate; start + elapsed

    When CUDA is not available every method is a no-op and ``elapsed_ms``
    / ``start_ns`` / ``end_ns`` return ``None``.
    """

    def __init__(self) -> None:
        self._start_event = None
        self._end_event = None
        self._host_start_ns: Optional[int] = None

        if _CUDA_AVAILABLE:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)

    def record_start(self) -> None:
        """Record a CUDA event at the current point in the stream."""
        self._host_start_ns = now_ns()
        if self._start_event is not None:
            self._start_event.record()

    def record_end(self) -> None:
        """Record the end CUDA event."""
        if self._end_event is not None:
            self._end_event.record()

    def elapsed_ms(self) -> Optional[float]:
        """Return elapsed GPU time in milliseconds (blocks until done).

        Returns ``None`` if CUDA is not available or events were not
        recorded.
        """
        if self._start_event is None or self._end_event is None:
            return None
        try:
            self._end_event.synchronize()
            return self._start_event.elapsed_time(self._end_event)
        except Exception as exc:
            logger.debug("CudaEventTimer.elapsed_ms failed: %s", exc)
            return None

    def start_ns(self) -> Optional[int]:
        """Return the host-side monotonic ns at which ``record_start`` was called."""
        return self._host_start_ns

    def end_ns(self) -> Optional[int]:
        """Approximate host-side end timestamp derived from CUDA elapsed time.

        Returns ``None`` if timing is unavailable.
        """
        if self._host_start_ns is None:
            return None
        elapsed = self.elapsed_ms()
        if elapsed is None:
            return None
        return self._host_start_ns + int(elapsed * 1_000_000)

    def cuda_event_ts_ns(self) -> Optional[int]:
        """Return ``elapsed_ms`` converted to nanoseconds (integer).

        This is the raw CUDA-derived duration; store it alongside
        ``start_ns()`` so post-hoc skew correction is possible (§3.2).
        """
        elapsed = self.elapsed_ms()
        if elapsed is None:
            return None
        return int(elapsed * 1_000_000)
