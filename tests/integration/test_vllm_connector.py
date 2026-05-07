"""
M3 integration smoke test — vLLM connector wrapper.

These tests use mock connector objects instead of a real vLLM installation.
They verify that the wrapper emits paired transfer.start / transfer.end
records for load and save paths, plus connector metadata records.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import time
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture()
def bkvt_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BKVT_ENABLE", "1")
    monkeypatch.setenv("BKVT_OUTPUT_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("BKVT_SAMPLE_TOKEN", "1.0")
    monkeypatch.setenv("BKVT_SAMPLE_METADATA", "1.0")
    monkeypatch.setenv("BKVT_SAMPLE_TRANSFER", "1.0")
    monkeypatch.setenv("BKVT_TRACE_ID", "test-trace-m3")

    import bkvt.config as _cfg_mod
    import bkvt.emitter as _em_mod
    import bkvt.sampling as _samp_mod

    _cfg_mod.reset_config()
    _em_mod._shutdown_emitter()
    _em_mod._emitter = None
    _samp_mod._sampler = None

    yield _em_mod.get_emitter()

    em = _em_mod.get_emitter()
    em.shutdown()
    _em_mod._emitter = None
    _cfg_mod.reset_config()


def _collect(em: Any) -> list[dict]:
    time.sleep(0.25)
    em.shutdown()

    records = []
    for path in glob.glob(os.path.join(em._config.output_dir, "**", "*.jsonl*"), recursive=True):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


class MockNixlConnector:
    __module__ = "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> int:
        self.calls.append("match")
        return num_computed_tokens + 16

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int = 0) -> None:
        self.calls.append("alloc")

    def build_connector_meta(self, scheduler_output: Any) -> dict:
        self.calls.append("meta")
        return {"ok": True}

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        self.calls.append("load")

    def wait_for_layer_load(self, layer_name: str) -> None:
        self.calls.append(f"wait:{layer_name}")

    def save_kv_layer(self, layer_name: str, kv_layer: Any, **kwargs: Any) -> None:
        self.calls.append(f"save:{layer_name}")

    def wait_for_save(self) -> None:
        self.calls.append("wait_save")

    def get_finished(self, finished_req_ids: Any) -> list[str]:
        self.calls.append("finished")
        return list(finished_req_ids)


def test_connector_wrapper_emits_paired_load_and_save_transfers(bkvt_env):
    from bkvt.integrations.vllm.connector_wrapper import TracingConnectorWrapper

    inner = MockNixlConnector()
    wrapper = TracingConnectorWrapper(inner)

    req = SimpleNamespace(req_id="req-m3")
    blocks = [SimpleNamespace(block_id=7), SimpleNamespace(block_id=8)]
    wrapper.get_num_new_matched_tokens(req, 32)
    wrapper.update_state_after_alloc(req, blocks, num_external_tokens=16)
    wrapper.build_connector_meta(SimpleNamespace(scheduled_new_reqs=[req]))

    fctx = SimpleNamespace(request=req, block_ids=blocks)
    wrapper.start_load_kv(fctx, bytes=4096)
    wrapper.wait_for_layer_load("model.layers.12")

    kv_layer = SimpleNamespace(nbytes=8192)
    wrapper.save_kv_layer("model.layers.12", kv_layer, request=req, block_ids=blocks)
    wrapper.wait_for_save()
    wrapper.get_finished(["req-m3"])

    records = _collect(bkvt_env)
    transfers = [r for r in records if r.get("type") == "transfer"]
    starts = [r for r in transfers if r.get("subtype") == "start"]
    ends = [r for r in transfers if r.get("subtype") == "end"]

    assert len(starts) == 2
    assert len(ends) == 2
    assert {r["transfer_id"] for r in starts} == {r["transfer_id"] for r in ends}
    assert {r["direction"] for r in starts} == {"load", "save"}
    assert all(r.get("transport") == "nixl" for r in starts)

    metadata_subtypes = {
        r.get("subtype") for r in records if r.get("type") == "metadata"
    }
    assert "prefix_lookup" in metadata_subtypes
    assert "connector_build_meta" in metadata_subtypes
    assert "wait_for_layer_load" in metadata_subtypes
    assert "wait_for_save" in metadata_subtypes
    assert "connector_get_finished" in metadata_subtypes

    tier_events = [r for r in records if r.get("type") == "kv_block" and r.get("subtype") == "tier_promote"]
    assert {r["block_id"] for r in tier_events} == {7, 8}
