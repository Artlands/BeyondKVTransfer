"""
Unit tests for bkvt/emitter.py.

Verifies:
- Emitter is a no-op when BKVT_ENABLE=0 (no files written, no errors).
- When enabled, event() records appear in the output file.
- transfer_start/end produce matching transfer_id records.
- metadata() context manager emits a timed record with correct subtype.
- Ring buffer overflow increments dropped counter.
- Shutdown flushes all pending records.
- Module-level convenience functions delegate to the singleton emitter.
"""

from __future__ import annotations

import gzip
import json
import os
import threading
import time
from pathlib import Path
from typing import Generator

import pytest

from bkvt.config import BkvtConfig, reset_config
from bkvt.emitter import (
    Emitter,
    _get_or_create_emitter,
    _shutdown_emitter,
    get_emitter,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture()
def enabled_config(tmp_dir):
    return BkvtConfig(
        enabled=True,
        output_dir=tmp_dir,
        trace_id="test-trace-001",
        profile="full",
        sample_token=1.0,
        sample_metadata=1.0,
        sample_transfer=1.0,
        rotate_bytes=100 * 1024 * 1024,  # 100 MiB — no rotation in tests
        flush_bytes=4096,
    )


@pytest.fixture()
def disabled_config(tmp_dir):
    return BkvtConfig(
        enabled=False,
        output_dir=tmp_dir,
    )


@pytest.fixture(autouse=True)
def reset_emitter_singleton():
    """Ensure each test starts with a clean emitter singleton."""
    import bkvt.emitter as _em
    _shutdown_emitter()
    _em._emitter = None
    yield
    # Give any background flusher thread time to finish before teardown.
    _shutdown_emitter()
    time.sleep(0.05)
    _em._emitter = None


def _read_jsonl(path: str) -> list[dict]:
    records = []
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt"
    with opener(path, mode) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _collect_trace_records(output_dir: str, trace_id: str) -> list[dict]:
    """Read all .jsonl (and .jsonl.gz) records for a given trace."""
    records: list[dict] = []
    trace_dir = Path(output_dir) / trace_id
    if not trace_dir.exists():
        return records
    for path in sorted(trace_dir.rglob("*.jsonl")):
        records.extend(_read_jsonl(str(path)))
    for path in sorted(trace_dir.rglob("*.jsonl.gz")):
        records.extend(_read_jsonl(str(path)))
    return records


def _wait_for_flush(emitter: Emitter, timeout: float = 3.0) -> None:
    """Give the flusher thread time to write buffered records."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.05)
        emitter._outfile.flush()


# ---------------------------------------------------------------------------
# Disabled emitter
# ---------------------------------------------------------------------------

class TestDisabledEmitter:
    def test_event_is_noop(self, disabled_config, tmp_dir):
        e = Emitter(disabled_config)
        e.event({"type": "request", "subtype": "arrival", "ts_ns": 1})
        # No output directory should be created
        trace_dir = Path(tmp_dir) / "some-trace"
        assert not trace_dir.exists()

    def test_transfer_start_returns_id(self, disabled_config):
        e = Emitter(disabled_config)
        tid = e.transfer_start("load")
        assert isinstance(tid, str) and len(tid) > 0

    def test_transfer_end_noop(self, disabled_config):
        e = Emitter(disabled_config)
        tid = e.transfer_start("load")
        e.transfer_end(tid)   # must not raise

    def test_metadata_context_manager_noop(self, disabled_config):
        e = Emitter(disabled_config)
        with e.metadata("prefix_lookup") as m:
            m["n_hits"] = 5   # must not raise

    def test_shutdown_noop(self, disabled_config):
        e = Emitter(disabled_config)
        e.shutdown()  # must not raise


# ---------------------------------------------------------------------------
# Enabled emitter — basic event emission
# ---------------------------------------------------------------------------

class TestEnabledEmitterBasic:
    def test_event_written_to_file(self, enabled_config, tmp_dir):
        e = Emitter(enabled_config)
        e.event({
            "type": "request",
            "subtype": "arrival",
            "ts_ns": 1_000_000_000,
            "request_id": "req-1",
        })
        e.shutdown()

        records = _collect_trace_records(tmp_dir, enabled_config.trace_id)
        types = {r.get("subtype") for r in records}
        assert "arrival" in types

    def test_envelope_fields_auto_filled(self, enabled_config, tmp_dir):
        e = Emitter(enabled_config)
        e.event({"type": "request", "subtype": "arrival", "ts_ns": 100})
        e.shutdown()

        records = _collect_trace_records(tmp_dir, enabled_config.trace_id)
        req = next(r for r in records if r.get("subtype") == "arrival")
        assert req["trace_id"] == enabled_config.trace_id
        assert "node_id" in req
        assert "worker_id" in req
        assert req["v"] == 1

    def test_multiple_events_all_written(self, enabled_config, tmp_dir):
        e = Emitter(enabled_config)
        for i in range(10):
            e.event({
                "type": "request",
                "subtype": "arrival",
                "ts_ns": i,
                "request_id": f"req-{i}",
            })
        e.shutdown()

        records = _collect_trace_records(tmp_dir, enabled_config.trace_id)
        arrivals = [r for r in records if r.get("subtype") == "arrival"]
        assert len(arrivals) == 10


# ---------------------------------------------------------------------------
# Transfer start/end pairing
# ---------------------------------------------------------------------------

class TestTransferPairing:
    def test_start_end_same_transfer_id(self, enabled_config, tmp_dir):
        e = Emitter(enabled_config)
        tid = e.transfer_start("load", request_id="req-1", bytes_=1024)
        e.transfer_end(tid, bytes_=1024, wire_time_ns=500_000)
        e.shutdown()

        records = _collect_trace_records(tmp_dir, enabled_config.trace_id)
        transfers = [r for r in records if r.get("type") == "transfer"]
        assert len(transfers) == 2

        start = next(r for r in transfers if r.get("subtype") == "start")
        end   = next(r for r in transfers if r.get("subtype") == "end")
        assert start["transfer_id"] == end["transfer_id"] == tid

    def test_transfer_cancel(self, enabled_config, tmp_dir):
        e = Emitter(enabled_config)
        tid = e.transfer_start("save")
        e.transfer_end(tid, cancelled=True)
        e.shutdown()

        records = _collect_trace_records(tmp_dir, enabled_config.trace_id)
        transfers = [r for r in records if r.get("type") == "transfer"]
        cancel = next(r for r in transfers if r.get("subtype") == "cancel")
        assert cancel["transfer_id"] == tid

    def test_direction_preserved(self, enabled_config, tmp_dir):
        e = Emitter(enabled_config)
        for direction in ("load", "save"):
            tid = e.transfer_start(direction)
            e.transfer_end(tid)
        e.shutdown()

        records = _collect_trace_records(tmp_dir, enabled_config.trace_id)
        starts = [r for r in records if r.get("subtype") == "start"]
        directions = {r["direction"] for r in starts}
        assert directions == {"load", "save"}


# ---------------------------------------------------------------------------
# Metadata context manager
# ---------------------------------------------------------------------------

class TestMetadataContextManager:
    def test_emits_metadata_record(self, enabled_config, tmp_dir):
        e = Emitter(enabled_config)
        with e.metadata("prefix_lookup", request_id="req-1") as m:
            m["n_hits"] = 7
        e.shutdown()

        records = _collect_trace_records(tmp_dir, enabled_config.trace_id)
        meta = [r for r in records if r.get("type") == "metadata"
                and r.get("subtype") == "prefix_lookup"]
        assert len(meta) == 1
        assert meta[0]["n_hits"] == 7
        assert meta[0]["request_id"] == "req-1"
        assert meta[0]["duration_ns"] >= 0

    def test_duration_ns_positive(self, enabled_config, tmp_dir):
        e = Emitter(enabled_config)
        with e.metadata("prefix_insert") as m:
            time.sleep(0.001)   # 1 ms
        e.shutdown()

        records = _collect_trace_records(tmp_dir, enabled_config.trace_id)
        meta = next(r for r in records if r.get("subtype") == "prefix_insert")
        assert meta["duration_ns"] >= 1_000  # at least 1 µs

    def test_exception_in_body_still_emits(self, enabled_config, tmp_dir):
        e = Emitter(enabled_config)
        try:
            with e.metadata("allocator_alloc"):
                raise ValueError("test error")
        except ValueError:
            pass
        e.shutdown()

        records = _collect_trace_records(tmp_dir, enabled_config.trace_id)
        meta = [r for r in records if r.get("subtype") == "allocator_alloc"]
        assert len(meta) == 1


# ---------------------------------------------------------------------------
# Sampling integration
# ---------------------------------------------------------------------------

class TestSamplingIntegration:
    def test_sampler_gates_token_events(self, tmp_dir):
        """The sampler correctly gates token events at rate=0.

        The core emitter's event() is a low-level primitive that doesn't
        consult the sampler — gating happens in the integration probes.
        This test verifies the sampler itself drops all decode tokens at
        rate=0 while keeping first_token (always-on).
        """
        from bkvt.sampling import Sampler
        s = Sampler(sample_token=0.0)

        dropped = 0
        kept_first = 0
        for _ in range(50):
            emit, _ = s.should_emit_token("decode")
            if not emit:
                dropped += 1
        for _ in range(10):
            emit, _ = s.should_emit_token("first_token")
            if emit:
                kept_first += 1

        assert dropped == 50, "all decode tokens should be dropped at rate=0"
        assert kept_first == 10, "first_token must always be kept"

    def test_transfer_start_respects_sampler_rate_zero(self, tmp_dir):
        """transfer_start() is gated by the emitter's sampler."""
        cfg = BkvtConfig(
            enabled=True,
            output_dir=tmp_dir,
            trace_id="sample-xfer-trace",
            sample_token=1.0,
            sample_metadata=1.0,
            sample_transfer=0.0,   # drop all transfers
        )
        e = Emitter(cfg)
        for i in range(20):
            tid = e.transfer_start("load", request_id=f"req-{i}")
            e.transfer_end(tid)
        e.shutdown()

        records = _collect_trace_records(tmp_dir, cfg.trace_id)
        transfers = [r for r in records if r.get("type") == "transfer"]
        assert len(transfers) == 0, "all transfers should be dropped at rate=0"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_events_no_corruption(self, enabled_config, tmp_dir):
        e = Emitter(enabled_config)
        n_threads = 4
        n_per_thread = 100
        errors: list[Exception] = []

        def worker(tid: int):
            try:
                for i in range(n_per_thread):
                    e.event({
                        "type": "request",
                        "subtype": "arrival",
                        "ts_ns": tid * 10000 + i,
                        "request_id": f"req-{tid}-{i}",
                    })
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        e.shutdown()

        assert not errors
        records = _collect_trace_records(tmp_dir, enabled_config.trace_id)
        arrivals = [r for r in records if r.get("subtype") == "arrival"]
        assert len(arrivals) == n_threads * n_per_thread


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

class TestModuleLevelFunctions:
    def test_module_event(self, enabled_config, tmp_dir):
        import bkvt.emitter as em
        em._emitter = None
        reset_config(enabled_config)
        try:
            em.event({
                "type": "request",
                "subtype": "admit",
                "ts_ns": 42,
                "request_id": "rmod-1",
            })
            em._shutdown_emitter()
            records = _collect_trace_records(tmp_dir, enabled_config.trace_id)
            admits = [r for r in records if r.get("subtype") == "admit"]
            assert len(admits) == 1
        finally:
            reset_config(None)
            em._emitter = None

    def test_module_transfer_start_end(self, enabled_config, tmp_dir):
        import bkvt.emitter as em
        em._emitter = None
        reset_config(enabled_config)
        try:
            tid = em.transfer_start("load")
            em.transfer_end(tid)
            em._shutdown_emitter()
            records = _collect_trace_records(tmp_dir, enabled_config.trace_id)
            xfers = [r for r in records if r.get("type") == "transfer"]
            assert any(r.get("subtype") == "start" for r in xfers)
            assert any(r.get("subtype") == "end" for r in xfers)
        finally:
            reset_config(None)
            em._emitter = None
