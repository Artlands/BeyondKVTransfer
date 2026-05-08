from __future__ import annotations

import glob
import gzip
import json
import os
import time
from types import SimpleNamespace
from typing import Any

import pytest


def _ns() -> SimpleNamespace:
    return SimpleNamespace()


def make_req(rid: str, input_len: int = 16, max_new_tokens: int = 8) -> Any:
    req = _ns()
    req.rid = rid
    req.origin_input_ids = list(range(input_len))
    req.sampling_params = _ns()
    req.sampling_params.max_new_tokens = max_new_tokens
    req.completion_tokens = 2
    return req


@pytest.fixture()
def bkvt_env(tmp_path, monkeypatch):
    trace_dir = str(tmp_path / "traces")
    monkeypatch.setenv("BKVT_ENABLE", "1")
    monkeypatch.setenv("BKVT_OUTPUT_DIR", trace_dir)
    monkeypatch.setenv("BKVT_TRACE_ID", "sglang-m4-test")
    monkeypatch.setenv("BKVT_SAMPLE_TOKEN", "1.0")
    monkeypatch.setenv("BKVT_SAMPLE_METADATA", "1.0")
    monkeypatch.setenv("BKVT_SAMPLE_TRANSFER", "1.0")
    monkeypatch.setenv("BKVT_SYS_COUNTER_HZ", "0")
    monkeypatch.setenv("BKVT_CLOCK_ANCHOR_HZ", "0")

    import bkvt.config as _cfg_mod
    import bkvt.emitter as _em_mod
    import bkvt.integrations.sglang.patch as patch
    import bkvt.integrations.sglang.disagg_probe as dp
    import bkvt.integrations.sglang.hicache_probe as hp
    import bkvt.integrations.sglang.scheduler_probe as sp

    _cfg_mod.reset_config()
    _em_mod._shutdown_emitter()
    _em_mod._emitter = None
    patch.reset()
    sp._tracker = sp.RequestStateTracker()
    hp._PATCHES_APPLIED = False
    dp._PATCHES_APPLIED = False

    yield _em_mod.get_emitter()

    _em_mod._shutdown_emitter()
    _em_mod._emitter = None
    _cfg_mod.reset_config()


def _flush_and_collect(em) -> list[dict]:
    time.sleep(0.25)
    em.shutdown()

    import bkvt.emitter as _em_mod
    _em_mod._emitter = None

    records: list[dict] = []
    for path in glob.glob(os.path.join(em._config.output_dir, "**", "*.jsonl"), recursive=True):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    return records


def test_scheduler_lifecycle_and_q6_records(bkvt_env):
    from bkvt.integrations.sglang.scheduler_probe import (
        get_tracker,
        make_handle_finished_requests_wrapper,
        make_handle_generate_request_wrapper,
        make_run_batch_wrapper,
    )

    req = make_req("sg-req-1")

    def generate_orig(self, request):
        return request

    generate = make_handle_generate_request_wrapper(generate_orig)
    scheduler = _ns()
    scheduler.waiting_queue = [req]
    scheduler.running_batch = _ns()
    scheduler.running_batch.reqs = []
    scheduler.step_id = 7
    generate(scheduler, req)

    out = _ns()
    out.new_reqs = [req]

    def run_orig(self):
        self.waiting_queue = []
        self.running_batch.reqs = [req]
        return out

    run_batch = make_run_batch_wrapper(run_orig)
    run_batch(scheduler)

    # Set first_token_ts_ns to 10 ms *after* arrival so the guard condition
    # (first_token_ts >= arrival_ts) in scheduler_probe is always satisfied,
    # regardless of how fast the test machine is.
    arrival_ts = get_tracker().get("sg-req-1").get("arrival_ts_ns", 0)
    get_tracker().update("sg-req-1", first_token_ts_ns=arrival_ts + 10_000_000)

    def finish_orig(self, finished_reqs):
        return None

    finish = make_handle_finished_requests_wrapper(finish_orig)
    finish(scheduler, [req])

    records = _flush_and_collect(bkvt_env)
    assert any(r.get("type") == "request" and r.get("subtype") == "arrival" for r in records)
    assert any(r.get("type") == "request" and r.get("subtype") == "admit" for r in records)
    assert any(r.get("type") == "request" and r.get("subtype") == "finish" for r in records)
    decision = next(
        r for r in records
        if r.get("type") == "metadata" and r.get("subtype") == "scheduler_decision"
    )
    assert decision["scheduler_outputs"]["admitted"] == ["sg-req-1"]
    finish_rec = next(r for r in records if r.get("subtype") == "finish")
    assert finish_rec["request_id"] == "sg-req-1"
    assert finish_rec["ttft_ns"] is not None


def test_radix_allocator_records_make_q4_inputs(bkvt_env):
    from bkvt.integrations.sglang.radix_probe import (
        make_allocator_alloc_wrapper,
        make_allocator_free_wrapper,
        make_insert_wrapper,
        make_match_prefix_wrapper,
    )

    cache = _ns()
    cache.page_size = 4
    cache.size = lambda: 2

    def match_orig(self, key):
        return [10, 11]

    match = make_match_prefix_wrapper(match_orig)
    assert match(cache, [1, 2, 3]) == [10, 11]

    def insert_orig(self, key, values):
        return values

    insert = make_insert_wrapper(insert_orig)
    insert(cache, [1, 2, 3, 4], [10, 11])

    allocator = _ns()

    def alloc_orig(self, n):
        return [20, 21, 22]

    alloc = make_allocator_alloc_wrapper(alloc_orig)
    alloc(allocator, 3)

    def free_orig(self, indices):
        return None

    free = make_allocator_free_wrapper(free_orig)
    free(allocator, [20, 21])

    records = _flush_and_collect(bkvt_env)
    assert any(r.get("subtype") == "prefix_lookup" and r.get("n_hits") == 2 for r in records)
    assert len([r for r in records if r.get("type") == "kv_block" and r.get("subtype") == "prefix_hit"]) == 2
    assert len([r for r in records if r.get("type") == "kv_block" and r.get("subtype") == "hash_insert"]) == 2
    assert len([r for r in records if r.get("type") == "kv_block" and r.get("subtype") == "allocate"]) == 3
    assert len([r for r in records if r.get("type") == "kv_block" and r.get("subtype") == "free"]) == 2


def test_hicache_probe_emits_transfer_pairs_and_tier_transitions(bkvt_env):
    from bkvt.integrations.sglang.hicache_probe import make_hicache_transfer_wrapper

    controller = _ns()

    def load_orig(self, blocks, request=None, bytes=None):
        return blocks

    load = make_hicache_transfer_wrapper(load_orig, "load_to_device")
    req = make_req("sg-hicache-1")
    load(controller, [30, 31], request=req, bytes=8192)

    def backup_orig(self, blocks, request_id=None, bytes=None):
        return blocks

    backup = make_hicache_transfer_wrapper(backup_orig, "backup_to_storage")
    backup(controller, [32], request_id="sg-hicache-1", bytes=4096)

    records = _flush_and_collect(bkvt_env)
    transfers = [r for r in records if r.get("type") == "transfer"]
    starts = [r for r in transfers if r.get("subtype") == "start"]
    ends = [r for r in transfers if r.get("subtype") == "end"]
    assert len(starts) == 2
    assert {r["transfer_id"] for r in starts} == {r["transfer_id"] for r in ends}
    assert {r["direction"] for r in starts} == {"load", "save"}

    tier_events = [r for r in records if r.get("type") == "kv_block"]
    assert any(
        r.get("subtype") == "tier_promote"
        and r.get("tier_before") == "DRAM_LOCAL"
        and r.get("tier_after") == "HBM_LOCAL"
        and r.get("block_id") == 30
        for r in tier_events
    )
    assert any(
        r.get("subtype") == "tier_demote"
        and r.get("tier_before") == "DRAM_LOCAL"
        and r.get("tier_after") == "OBJECT_STORE"
        and r.get("block_id") == 32
        for r in tier_events
    )
    metadata_subtypes = {r.get("subtype") for r in records if r.get("type") == "metadata"}
    assert "hicache_promote" in metadata_subtypes
    assert "hicache_demote" in metadata_subtypes


def test_disagg_probe_emits_paired_pd_transfer_records(bkvt_env):
    from bkvt.integrations.sglang.disagg_probe import make_disagg_transfer_wrapper

    def send_orig(request, block_ids, bytes=None):
        return _ns()

    send = make_disagg_transfer_wrapper(
        send_orig,
        "send_kv",
        module_name="sglang.srt.disaggregation.prefill",
        role_hint="prefill",
    )
    req = make_req("sg-pd-1")
    send(req, [40, 41], bytes=16384)

    def recv_orig(request_id=None, block_ids=None, bytes=None):
        out = _ns()
        out.descriptors = [1, 2]
        return out

    recv = make_disagg_transfer_wrapper(
        recv_orig,
        "recv_kv",
        module_name="sglang.srt.disaggregation.nixl.conn",
        role_hint="decode",
    )
    recv(request_id="sg-pd-1", block_ids=[40, 41], bytes=16384)

    records = _flush_and_collect(bkvt_env)
    transfers = [r for r in records if r.get("type") == "transfer"]
    starts = [r for r in transfers if r.get("subtype") == "start"]
    ends = [r for r in transfers if r.get("subtype") == "end"]
    assert len(starts) == 2
    assert {r["transfer_id"] for r in starts} == {r["transfer_id"] for r in ends}
    assert any(r.get("direction") == "save" and r.get("issued_at_phase") == "prefill" for r in starts)
    assert any(
        r.get("direction") == "load"
        and r.get("transport") == "nixl"
        and r.get("issued_at_phase") == "decode"
        for r in starts
    )
    assert any(r.get("wr_count") == 2 for r in ends)
    assert any(
        r.get("type") == "metadata" and r.get("subtype") == "nixl_call"
        for r in records
    )
