"""Overhead validation for Milestone 8.

The quick tests in this module are suitable for normal CI.  The lab-rig
comparison is opt-in because it is intentionally hardware and workload
sensitive:

    BKVT_RUN_OVERHEAD=1 pytest tests/overhead/test_overhead_budget.py

Budget defaults come from DESIGN.md section 9 and can be overridden with env vars
when validating a specific machine.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from bkvt.config import BkvtConfig
from bkvt.emitter import Emitter


def _budget_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _time_ns(fn, iterations: int) -> int:
    start = time.perf_counter_ns()
    for i in range(iterations):
        fn(i)
    return time.perf_counter_ns() - start


def test_disabled_event_probe_cost_under_budget(tmp_path: Path) -> None:
    """Disabled hot-path probes should be a boolean check plus return."""

    iterations = int(os.environ.get("BKVT_OVERHEAD_ITERS", "200000"))
    budget_ns = _budget_float("BKVT_NOOP_EVENT_BUDGET_NS", 1500.0)
    emitter = Emitter(BkvtConfig(enabled=False, output_dir=str(tmp_path)))
    record = {"type": "metadata", "subtype": "prefix_lookup", "ts_ns": 1}

    elapsed = _time_ns(lambda _i: emitter.event(record), iterations)
    per_call_ns = elapsed / iterations

    assert per_call_ns <= budget_ns


def test_transfer_sampling_zero_emits_no_records(tmp_path: Path) -> None:
    """Transfer sampling at 0 must avoid orphan start/end records."""

    cfg = BkvtConfig(
        enabled=True,
        output_dir=str(tmp_path),
        trace_id="overhead-sample-zero",
        sample_transfer=0.0,
        sys_counter_hz=0,
        clock_anchor_hz=0,
    )
    emitter = Emitter(cfg)
    for _ in range(1000):
        transfer_id = emitter.transfer_start("load", request_id="req-overhead")
        emitter.transfer_end(transfer_id)
    emitter.shutdown()

    trace_dir = tmp_path / "overhead-sample-zero"
    transfer_records = []
    for path in trace_dir.rglob("*.jsonl"):
        transfer_records.extend(
            line for line in path.read_text(encoding="utf-8").splitlines()
            if '"type":"transfer"' in line or '"type": "transfer"' in line
        )
    assert transfer_records == []


@pytest.mark.skipif(
    os.environ.get("BKVT_RUN_OVERHEAD") != "1",
    reason="set BKVT_RUN_OVERHEAD=1 on a pinned lab rig to run regression budgets",
)
def test_synthetic_enabled_vs_baseline_budget(tmp_path: Path) -> None:
    """Opt-in synthetic regression check for the section 9 overhead budgets.

    This does not replace the ShareGPT/vLLM lab run from DESIGN.md.  It gives a
    reproducible local signal that the emitter's enabled path has not regressed
    badly before running the full workload.
    """

    iterations = int(os.environ.get("BKVT_OVERHEAD_ITERS", "50000"))
    throughput_budget = _budget_float("BKVT_THROUGHPUT_REGRESSION_BUDGET", 0.03)
    per_record_budget_ns = _budget_float("BKVT_RECORD_CPU_BUDGET_NS", 1500.0)

    def baseline(i: int) -> None:
        _ = {
            "type": "metadata",
            "subtype": "prefix_lookup",
            "ts_ns": i,
            "request_id": "req-overhead",
        }

    baseline_ns = _time_ns(baseline, iterations)

    cfg = BkvtConfig(
        enabled=True,
        output_dir=str(tmp_path),
        trace_id="overhead-enabled",
        sample_metadata=1.0,
        sys_counter_hz=0,
        clock_anchor_hz=0,
        rotate_bytes=1024 * 1024 * 1024,
    )
    emitter = Emitter(cfg)

    def traced(i: int) -> None:
        emitter.event({
            "type": "metadata",
            "subtype": "prefix_lookup",
            "ts_ns": i,
            "request_id": "req-overhead",
        })

    traced_ns = _time_ns(traced, iterations)
    emitter.shutdown()

    added_per_record_ns = (traced_ns - baseline_ns) / iterations
    regression = (traced_ns / baseline_ns) - 1.0

    assert added_per_record_ns <= per_record_budget_ns
    assert regression <= throughput_budget
