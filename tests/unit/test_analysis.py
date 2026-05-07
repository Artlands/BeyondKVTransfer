from __future__ import annotations

import json
from pathlib import Path

from analysis.build_index import build_index
from analysis.lib.load import load_index, load_trace


def test_build_index_jsonl_smoke(tmp_path: Path) -> None:
    trace = tmp_path / "trace-1"
    node = trace / "node-0"
    node.mkdir(parents=True)
    (trace / "manifest.json").write_text(json.dumps({"trace_id": "trace-1"}) + "\n")
    records = [
        {
            "ts_ns": 100,
            "type": "request",
            "subtype": "arrival",
            "trace_id": "trace-1",
            "node_id": "node-0",
            "worker_id": "worker-0",
            "request_id": "req-1",
            "v": 1,
        },
        {
            "ts_ns": 150,
            "type": "request",
            "subtype": "admit",
            "trace_id": "trace-1",
            "node_id": "node-0",
            "worker_id": "worker-0",
            "request_id": "req-1",
            "v": 1,
        },
        {
            "ts_ns": 300,
            "type": "token",
            "subtype": "first_token",
            "trace_id": "trace-1",
            "node_id": "node-0",
            "worker_id": "worker-0",
            "request_id": "req-1",
            "token_idx": 8,
            "step_id": 1,
            "v": 1,
        },
        {
            "ts_ns": 500,
            "type": "request",
            "subtype": "finish",
            "trace_id": "trace-1",
            "node_id": "node-0",
            "worker_id": "worker-0",
            "request_id": "req-1",
            "arrival_ts_ns": 100,
            "first_schedule_ts_ns": 150,
            "first_token_ts_ns": 300,
            "finish_ts_ns": 500,
            "ttft_ns": 200,
            "tpot_ns": 25,
            "v": 1,
        },
        {
            "ts_ns": 160,
            "type": "transfer",
            "subtype": "start",
            "trace_id": "trace-1",
            "node_id": "node-0",
            "worker_id": "worker-0",
            "transfer_id": "xfer-1",
            "direction": "load",
            "request_id": "req-1",
            "transport": "nixl",
            "bytes": 1024,
            "started_ts_ns": 170,
            "earliest_known_ts_ns": 120,
            "src": {"tier": "DRAM_REMOTE", "node_id": "peer"},
            "dst": {"tier": "HBM_LOCAL", "device_id": 0},
            "v": 1,
        },
        {
            "ts_ns": 260,
            "type": "transfer",
            "subtype": "end",
            "trace_id": "trace-1",
            "node_id": "node-0",
            "worker_id": "worker-0",
            "transfer_id": "xfer-1",
            "direction": "load",
            "request_id": "req-1",
            "completed_ts_ns": 260,
            "wire_time_ns": 90,
            "queue_wait_ns": 10,
            "v": 1,
        },
        {
            "ts_ns": 180,
            "type": "metadata",
            "subtype": "prefix_lookup",
            "trace_id": "trace-1",
            "node_id": "node-0",
            "worker_id": "worker-0",
            "request_id": "req-1",
            "duration_ns": 7,
            "v": 1,
        },
        {
            "ts_ns": 190,
            "type": "kv_block",
            "subtype": "prefix_hit",
            "trace_id": "trace-1",
            "node_id": "node-0",
            "worker_id": "worker-0",
            "block_id": 1,
            "tier_before": "HBM_LOCAL",
            "tier_after": "HBM_LOCAL",
            "v": 1,
        },
    ]
    with open(node / "worker.0001.jsonl", "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    raw = load_trace(trace)
    assert len(raw["request"]) == 3
    assert raw["transfer"].iloc[0]["src_tier"] == "DRAM_REMOTE"

    out = tmp_path / "index"
    counts = build_index(trace, out, output_format="jsonl")
    assert counts["request_lifecycle"] == 1
    assert counts["transfer_pairs"] == 1
    assert counts["prefetch_slack"] == 1

    indexed = load_index(out)
    assert indexed["request_lifecycle"].iloc[0]["ttft_ns"] == 200
    assert indexed["prefetch_slack"].iloc[0]["prefetch_slack_ns"] == 50
