from __future__ import annotations

import json
import time
from pathlib import Path

from bkvt.collectors.ib_counters import iter_ib_counter_samples
from bkvt.collectors.nccl_log import NcclLogTailer
from bkvt.config import BkvtConfig
from bkvt.emitter import Emitter


def test_ib_counter_samples_convert_words_to_bytes(tmp_path):
    counters = tmp_path / "mlx5_0" / "ports" / "1" / "counters"
    counters.mkdir(parents=True)
    (counters / "port_xmit_data").write_text("10\n")
    (counters / "port_rcv_data").write_text("7\n")
    (counters / "port_xmit_packets").write_text("3\n")
    (counters / "port_rcv_packets").write_text("4\n")
    (counters / "port_xmit_constraint_errors").write_text("1\n")
    (counters / "port_rcv_constraint_errors").write_text("2\n")

    samples = iter_ib_counter_samples(str(tmp_path))

    byte_values = sorted(s.value for s in samples if s.subtype == "nic_bytes")
    packet_values = sorted(s.value for s in samples if s.subtype == "nic_packets")
    pkey = next(s for s in samples if s.subtype == "ib_pkey_violation")
    assert byte_values == [28, 40]
    assert packet_values == [3, 4]
    assert pkey.value == 3
    assert pkey.scope == "nic:mlx5_0/port:1"


def test_nccl_log_tailer_accumulates_p2p_bytes(tmp_path):
    log_path = tmp_path / "nccl.log"
    log_path.write_text("noise\nNCCL INFO Send bytes=1024 peer=1\n")

    tailer = NcclLogTailer(str(log_path))
    first = tailer.collect()
    assert first[0].subtype == "nccl_p2p_bytes"
    assert first[0].value == 1024

    with log_path.open("a") as fh:
        fh.write("NCCL INFO Recv count=2048 peer=0\n")
    second = tailer.collect()
    assert second[0].value == 3072


def test_emitter_starts_system_counter_and_clock_anchor(tmp_path):
    cfg = BkvtConfig(
        enabled=True,
        output_dir=str(tmp_path),
        trace_id="m6-test",
        sys_counter_hz=20.0,
        clock_anchor_hz=20.0,
        sample_metadata=1.0,
    )
    emitter = Emitter(cfg)
    time.sleep(0.18)
    emitter.shutdown()

    records: list[dict] = []
    for path in Path(tmp_path).rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            records.append(json.loads(line))

    assert any(r.get("type") == "sys_counter" for r in records)
    assert any(
        r.get("type") == "metadata" and r.get("subtype") == "clock_anchor"
        for r in records
    )
    assert any(r.get("subtype") == "process_rss" for r in records)
