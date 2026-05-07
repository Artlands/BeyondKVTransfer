"""
Core probe emitter for BeyondKVTransfer (§8).

Design
------
* Per-thread ring buffer (``_RingBuffer``) of pre-allocated slots.
  When the buffer has room the hot path never allocates Python objects.
* A single background flusher thread serialises records to ``orjson``
  and writes batched output to a per-worker ``.jsonl`` file.
* Files rotate at ``config.rotate_bytes``; the previous file is gzipped
  in a separate background thread.
* When ``BKVT_ENABLE`` is 0 (the default), every probe is a no-op with a
  single boolean check and zero allocations.

Public API (module-level convenience functions)
-----------------------------------------------
``event(record_dict)``
    Enqueue a pre-built dict record.

``transfer_start(direction, ...) -> str``
    Enqueue a transfer/start record and return the ``transfer_id``.

``transfer_end(transfer_id, ...)``
    Enqueue a transfer/end record.

``metadata(subtype, **kw) -> contextmanager``
    Context manager that times its body and emits a metadata record.

``init(config=None) -> Emitter``
    Initialise (or return existing) emitter singleton.

``shutdown()``
    Flush and stop background threads.
"""

from __future__ import annotations

import collections
import contextlib
import gzip
import logging
import os
import shutil
import threading
import time
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

try:
    import orjson  # type: ignore[import-untyped]
    def _dumps(obj: Any) -> bytes:
        return orjson.dumps(obj)
except ImportError:
    import json
    def _dumps(obj: Any) -> bytes:  # type: ignore[misc]
        return json.dumps(obj, default=str).encode()

from bkvt.clock import now_ns
from bkvt.config import BkvtConfig, get_config
from bkvt.ids import get_node_id, get_worker_id, new_trace_id, new_transfer_id
from bkvt.sampling import Sampler, init_sampler, get_sampler


# ---------------------------------------------------------------------------
# Ring buffer (lock-free-ish — uses a deque with GIL protection)
# ---------------------------------------------------------------------------

_RING_CAPACITY = 4096  # records per thread


class _RingBuffer:
    """Fixed-capacity deque used as a ring buffer per thread."""

    __slots__ = ("_buf", "_capacity", "dropped")

    def __init__(self, capacity: int = _RING_CAPACITY) -> None:
        self._buf: collections.deque = collections.deque()
        self._capacity = capacity
        self.dropped: int = 0

    def put(self, item: Any) -> bool:
        if len(self._buf) >= self._capacity:
            self.dropped += 1
            return False
        self._buf.append(item)
        return True

    def drain(self) -> list:
        items, self._buf = list(self._buf), collections.deque()
        return items


# ---------------------------------------------------------------------------
# Per-worker output file (with rotation + async gzip)
# ---------------------------------------------------------------------------

class _OutputFile:
    """Manages a single JSONL output file with rotation."""

    def __init__(self, path: str, rotate_bytes: int) -> None:
        self._path = path
        self._rotate_bytes = rotate_bytes
        self._written = 0
        self._seq = 1
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(self._current_path, "ab")

    @property
    def _current_path(self) -> str:
        base, ext = os.path.splitext(self._path)
        if not ext:
            base = self._path
            ext = ".jsonl"
        return f"{base}.{self._seq:04d}{ext}"

    def write_batch(self, lines: list[bytes]) -> None:
        if not lines:
            return
        data = b"\n".join(lines) + b"\n"
        with self._lock:
            self._fh.write(data)
            self._written += len(data)
            if self._written >= self._rotate_bytes:
                self._rotate()

    def _rotate(self) -> None:
        """Close current file, gzip it in background, open new one."""
        old_path = self._current_path
        self._fh.flush()
        self._fh.close()
        self._seq += 1
        self._written = 0
        self._fh = open(self._current_path, "ab")
        # Gzip old file in background
        t = threading.Thread(
            target=_gzip_file,
            args=(old_path,),
            daemon=True,
            name=f"bkvt-gzip-{os.path.basename(old_path)}",
        )
        t.start()

    def flush(self) -> None:
        with self._lock:
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.flush()
            self._fh.close()


def _gzip_file(path: str) -> None:
    gz_path = path + ".gz"
    try:
        with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(path)
    except Exception as exc:
        logger.warning("bkvt: failed to gzip %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

class Emitter:
    """Process-wide record emitter.

    One instance is created at ``init()`` time and shared across threads.
    Each thread gets its own ring buffer; the flusher thread drains all
    buffers every ``flush_interval_s`` seconds.
    """

    def __init__(self, config: BkvtConfig) -> None:
        self._config = config
        self._enabled = config.enabled

        if not self._enabled:
            return

        # IDs
        self._trace_id = config.trace_id or new_trace_id()
        self._node_id = get_node_id()
        self._worker_id = get_worker_id(self._node_id)

        # Sampler
        self._sampler: Sampler = init_sampler(
            sample_token=config.sample_token,
            sample_metadata=config.sample_metadata,
            sample_transfer=config.sample_transfer,
        )

        # Per-thread ring buffers
        self._buffers: list[_RingBuffer] = []
        self._buf_lock = threading.Lock()
        self._tls = threading.local()

        # Track which transfer_ids had their start record actually emitted.
        # Maps transfer_id -> direction ("load"|"save").
        # transfer_end() only emits if the id is in this map (avoids orphans
        # when transfer sampling rate < 1.0), and re-uses the direction so the
        # end record also carries the required `direction` field.
        self._live_transfers: dict[str, str] = {}
        self._live_transfers_lock = threading.Lock()

        # Output file
        trace_dir = os.path.join(
            config.output_dir, self._trace_id, self._node_id
        )
        worker_safe = self._worker_id.replace("/", "_").replace(":", "_")
        out_path = os.path.join(trace_dir, worker_safe)
        self._outfile = _OutputFile(out_path, config.rotate_bytes)

        # Flush state
        self._flush_bytes = config.flush_bytes
        self._flush_interval_s = 0.1  # 100 ms
        self._pending: list[bytes] = []
        self._pending_size = 0

        # Background flusher
        self._stop_event = threading.Event()
        self._flusher = threading.Thread(
            target=self._flusher_loop,
            name="bkvt-flusher",
            daemon=True,
        )
        self._flusher.start()

        # Dropped-record counter (a live sys_counter)
        self._dropped_total = 0

        # System counters + periodic clock anchors (§7.3, §7.4, M6).
        self._counter_collector = None
        if config.sys_counter_hz > 0 or config.clock_anchor_hz > 0:
            try:
                from bkvt.collectors.sys_counters import SystemCounterCollector
                self._counter_collector = SystemCounterCollector(self, config)
                self._counter_collector.start()
            except Exception as exc:
                logger.warning("bkvt: failed to start system counter collector: %s", exc)

    # ------------------------------------------------------------------
    # Thread-local ring buffer
    # ------------------------------------------------------------------

    def _get_buf(self) -> _RingBuffer:
        buf = getattr(self._tls, "buf", None)
        if buf is None:
            buf = _RingBuffer()
            self._tls.buf = buf
            with self._buf_lock:
                self._buffers.append(buf)
        return buf

    # ------------------------------------------------------------------
    # Public probe API
    # ------------------------------------------------------------------

    def event(self, record: dict) -> None:
        """Enqueue a pre-built record dict.  No-op when disabled."""
        if not self._enabled:
            return
        # Stamp common fields if missing
        record.setdefault("trace_id", self._trace_id)
        record.setdefault("node_id", self._node_id)
        record.setdefault("worker_id", self._worker_id)
        record.setdefault("v", 1)
        if not self._get_buf().put(record):
            self._dropped_total += 1

    def transfer_start(
        self,
        direction: str,
        *,
        request_id: Optional[str] = None,
        layer_idx: Optional[int] = None,
        src_tier: Optional[str] = None,
        dst_tier: Optional[str] = None,
        transport: Optional[str] = None,
        num_blocks: Optional[int] = None,
        bytes_: Optional[int] = None,
        block_ids: Optional[list] = None,
        issued_by: Optional[str] = None,
        issued_at_phase: Optional[str] = None,
        earliest_known_ts_ns: Optional[int] = None,
        **extra: Any,
    ) -> str:
        """Emit a ``transfer/start`` record and return ``transfer_id``."""
        transfer_id = new_transfer_id()
        if not self._enabled:
            return transfer_id

        emit, rate = self._sampler.should_emit_transfer()
        if not emit:
            return transfer_id

        ts = now_ns()
        rec: dict = {
            "ts_ns": ts,
            "type": "transfer",
            "subtype": "start",
            "transfer_id": transfer_id,
            "direction": direction,
            "started_ts_ns": ts,
            "queued_ts_ns": ts,
        }
        if request_id:
            rec["request_id"] = request_id
        if layer_idx is not None:
            rec["layer_idx"] = layer_idx
        if src_tier:
            rec["src"] = {"tier": src_tier}
        if dst_tier:
            rec["dst"] = {"tier": dst_tier}
        if transport:
            rec["transport"] = transport
        if num_blocks is not None:
            rec["num_blocks"] = num_blocks
        if bytes_ is not None:
            rec["bytes"] = bytes_
        if block_ids:
            rec["block_ids"] = block_ids
        if issued_by:
            rec["issued_by"] = issued_by
        if issued_at_phase:
            rec["issued_at_phase"] = issued_at_phase
        if earliest_known_ts_ns is not None:
            rec["earliest_known_ts_ns"] = earliest_known_ts_ns
        if rate is not None:
            rec["sample_decision"] = rate
        rec.update(extra)
        self.event(rec)
        with self._live_transfers_lock:
            self._live_transfers[transfer_id] = direction
        return transfer_id

    def transfer_end(
        self,
        transfer_id: str,
        *,
        bytes_: Optional[int] = None,
        wire_time_ns: Optional[int] = None,
        achieved_bw_gbps: Optional[float] = None,
        wr_count: Optional[int] = None,
        wr_completion_ts_ns: Optional[list] = None,
        cancelled: bool = False,
        **extra: Any,
    ) -> None:
        """Emit a ``transfer/end`` (or ``transfer/cancel``) record.

        Only emits if the corresponding ``transfer_start`` was emitted
        (i.e. the transfer was not sampled away).
        """
        if not self._enabled:
            return
        with self._live_transfers_lock:
            direction = self._live_transfers.pop(transfer_id, None)
        if direction is None:
            return  # start was sampled away or already reaped; drop end too

        ts = now_ns()
        rec: dict = {
            "ts_ns": ts,
            "type": "transfer",
            "subtype": "cancel" if cancelled else "end",
            "transfer_id": transfer_id,
            "direction": direction,   # carried from start record (required by schema)
            "completed_ts_ns": ts,
        }
        if bytes_ is not None:
            rec["bytes"] = bytes_
        if wire_time_ns is not None:
            rec["wire_time_ns"] = wire_time_ns
        if achieved_bw_gbps is not None:
            rec["achieved_bw_gbps"] = achieved_bw_gbps
        if wr_count is not None:
            rec["wr_count"] = wr_count
        if wr_completion_ts_ns:
            rec["wr_completion_ts_ns"] = wr_completion_ts_ns
        rec.update(extra)
        self.event(rec)

    @contextlib.contextmanager
    def metadata(
        self,
        subtype: str,
        *,
        request_id: Optional[str] = None,
        **kw: Any,
    ) -> Generator[dict, None, None]:
        """Context manager that times its body and emits a metadata record.

        Usage::

            with emitter.metadata("prefix_lookup", request_id=req_id) as m:
                result = cache.lookup(...)
                m["n_hits"] = result.hits
        """
        if not self._enabled:
            yield {}
            return

        emit, rate = self._sampler.should_emit_metadata(subtype)
        rec: dict = {}
        t_start = now_ns()
        try:
            yield rec
        finally:
            if emit:
                duration = now_ns() - t_start
                full_rec: dict = {
                    "ts_ns": t_start,
                    "type": "metadata",
                    "subtype": subtype,
                    "duration_ns": duration,
                }
                if request_id:
                    full_rec["request_id"] = request_id
                if rate is not None:
                    full_rec["sample_decision"] = rate
                full_rec.update(rec)
                full_rec.update(kw)
                self.event(full_rec)

    # ------------------------------------------------------------------
    # Flusher internals
    # ------------------------------------------------------------------

    def _flusher_loop(self) -> None:
        """Background thread: drain ring buffers → serialize → write."""
        while not self._stop_event.is_set():
            self._flush_once()
            self._stop_event.wait(self._flush_interval_s)
        # Final drain
        self._flush_once()
        self._outfile.close()

    def _flush_once(self) -> None:
        with self._buf_lock:
            buffers = list(self._buffers)

        lines: list[bytes] = []
        size = 0
        for buf in buffers:
            for rec in buf.drain():
                try:
                    line = _dumps(rec)
                    lines.append(line)
                    size += len(line)
                except Exception as exc:
                    logger.debug("bkvt: serialisation error: %s", exc)

        if lines:
            self._outfile.write_batch(lines)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def trace_id(self) -> str:
        return getattr(self, "_trace_id", "")

    @property
    def node_id(self) -> str:
        return getattr(self, "_node_id", "")

    @property
    def worker_id(self) -> str:
        return getattr(self, "_worker_id", "")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dropped_records_total(self) -> int:
        return getattr(self, "_dropped_total", 0)

    def shutdown(self) -> None:
        """Flush pending records and stop the flusher thread."""
        if not self._enabled:
            return
        if self._counter_collector is not None:
            self._counter_collector.stop()
        self._stop_event.set()
        self._flusher.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Module-level singleton and convenience wrappers
# ---------------------------------------------------------------------------

_emitter: Optional[Emitter] = None
_emitter_lock = threading.Lock()


def _get_or_create_emitter(config: Optional[BkvtConfig] = None) -> Emitter:
    global _emitter
    if _emitter is not None:
        return _emitter
    with _emitter_lock:
        if _emitter is None:
            cfg = config or get_config()
            _emitter = Emitter(cfg)
            if cfg.enabled:
                from bkvt import manifest as _manifest
                _manifest.write_manifest(_emitter)
    return _emitter


def _shutdown_emitter() -> None:
    global _emitter
    with _emitter_lock:
        if _emitter is not None:
            _emitter.shutdown()
            _emitter = None


def get_emitter() -> Emitter:
    """Return the process-wide emitter, initialising it if needed."""
    return _get_or_create_emitter()


# Module-level shortcut functions ----------------------------------------

def event(record: dict) -> None:
    get_emitter().event(record)


def transfer_start(direction: str, **kw: Any) -> str:
    return get_emitter().transfer_start(direction, **kw)


def transfer_end(transfer_id: str, **kw: Any) -> None:
    get_emitter().transfer_end(transfer_id, **kw)


@contextlib.contextmanager
def metadata(subtype: str, **kw: Any) -> Generator[dict, None, None]:
    with get_emitter().metadata(subtype, **kw) as m:
        yield m
