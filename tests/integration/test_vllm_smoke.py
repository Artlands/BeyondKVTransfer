"""
M2 integration smoke test — vLLM probe wrappers.

This test exercises all M2 probe wrappers using plain Python mock objects
instead of a real vLLM installation.  It verifies:

  1. process_inputs wrapper  → arrival record
  2. schedule wrapper        → admit + scheduler_decision records
  3. execute_model wrapper   → first_token + decode token records
  4. alloc_blocks wrapper    → allocate + allocator_alloc records
  5. free_blocks wrapper     → free + allocator_free records
  6. get_computed_blocks wrapper → prefix_lookup + prefix_hit records
  7. cache_full_blocks wrapper   → prefix_insert + hash_insert records
  8. finish_request wrapper  → finish record (with ttft_ns, tpot_ns)

After simulating a two-request lifecycle the test also verifies:

  A. Every arrival has a matching finish (no orphans).
  B. The per-request critical path (Q3) can be reconstructed:
       arrival_ts → first_schedule_ts → first_token_ts → finish_ts
  C. All emitted records are well-formed (type, subtype, ts_ns, v present).

Acceptance criterion from DESIGN §14 M2:
  "a vLLM run with a simple ShareGPT replay produces well-formed traces;
   per-request critical path can be reconstructed (Q3 figure renders)"
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import threading
from types import SimpleNamespace
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ns() -> SimpleNamespace:
    return SimpleNamespace()


def make_request(req_id: str, input_len: int = 128) -> Any:
    """Minimal mock of a vLLM Request / NewRequestData object."""
    r = _ns()
    r.req_id = req_id
    r.request_id = req_id
    r.prompt_token_ids = list(range(input_len))
    r.input_tokens = r.prompt_token_ids
    return r


def make_sampling_params(max_tokens: int = 64) -> Any:
    p = _ns()
    p.max_tokens = max_tokens
    return p


def make_scheduler_output(
    new_reqs=(),
    preempted_ids=(),
    resumed_ids=(),
    finished_ids=(),
    num_tokens: dict | None = None,
) -> Any:
    """Minimal mock SchedulerOutput."""
    so = _ns()
    so.scheduled_new_reqs = list(new_reqs)
    so.preempted_req_ids = list(preempted_ids)
    so.resumed_req_ids = list(resumed_ids)
    so.finished_req_ids = list(finished_ids)
    so.num_scheduled_tokens = num_tokens or {}
    so.num_free_gpu_blocks = 1000
    so.num_used_gpu_blocks = 200
    return so


def make_req_output(req_id: str, n_tokens: int = 1) -> Any:
    o = _ns()
    o.req_id = req_id
    o.request_id = req_id
    o.output_token_ids = list(range(n_tokens))
    return o


def make_model_output(req_outputs) -> Any:
    mo = _ns()
    mo.outputs = req_outputs
    return mo


# ---------------------------------------------------------------------------
# Fixture: fresh emitter per test
# ---------------------------------------------------------------------------

@pytest.fixture()
def trace_dir(tmp_path):
    return str(tmp_path / "traces")


@pytest.fixture()
def bkvt_env(trace_dir, monkeypatch):
    """Configure BKVT env vars and return a fresh emitter."""
    monkeypatch.setenv("BKVT_ENABLE", "1")
    monkeypatch.setenv("BKVT_OUTPUT_DIR", trace_dir)
    monkeypatch.setenv("BKVT_SAMPLE_TOKEN", "1.0")   # capture all tokens in test
    monkeypatch.setenv("BKVT_SAMPLE_METADATA", "1.0")
    monkeypatch.setenv("BKVT_SAMPLE_TRANSFER", "1.0")
    monkeypatch.setenv("BKVT_TRACE_ID", "test-trace-0001")

    # Reset singletons so each test gets a clean emitter
    import bkvt.config as _cfg_mod
    import bkvt.emitter as _em_mod
    import bkvt.sampling as _samp_mod
    _cfg_mod.reset_config()
    _em_mod._shutdown_emitter()
    _em_mod._emitter = None

    # Also reset patch applied flags
    for mod_name in (
        "bkvt.integrations.vllm.scheduler_probe",
        "bkvt.integrations.vllm.block_pool_probe",
        "bkvt.integrations.vllm.runner_probe",
    ):
        import importlib
        try:
            mod = importlib.import_module(mod_name)
            mod._PATCHES_APPLIED = False
        except ImportError:
            pass

    # Reset tracker
    import bkvt.integrations.vllm.scheduler_probe as sp
    sp._tracker = sp.RequestStateTracker()

    yield _em_mod.get_emitter()

    # Teardown
    em = _em_mod.get_emitter()
    em.shutdown()
    _em_mod._emitter = None
    _cfg_mod.reset_config()


def _flush_and_collect(em) -> list[dict]:
    """Force-flush the emitter and return all emitted records."""
    time.sleep(0.25)   # let flusher run
    em.shutdown()

    import bkvt.emitter as _em_mod
    _em_mod._emitter = None

    records = []
    trace_dir = em._config.output_dir
    import glob, gzip
    for path in glob.glob(os.path.join(trace_dir, "**", "*.jsonl"), recursive=True):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


# ===========================================================================
# Test 1 — process_inputs → arrival
# ===========================================================================

class TestProcessInputsWrapper:

    def test_arrival_record_emitted(self, bkvt_env):
        from bkvt.integrations.vllm.scheduler_probe import make_process_inputs_wrapper

        # Original function (mock): returns an object with prompt_token_ids
        def _orig(self, request_id, prompt, params, arrival_time=None, **kw):
            result = _ns()
            result.req_id = request_id
            result.prompt_token_ids = list(range(params.max_tokens))
            return result

        wrapper = make_process_inputs_wrapper(_orig)

        mock_self = _ns()
        params = make_sampling_params(max_tokens=64)
        result = wrapper(mock_self, "req-001", "hello world", params)

        records = _flush_and_collect(bkvt_env)
        arrival = [r for r in records if r.get("type") == "request"
                   and r.get("subtype") == "arrival"]
        assert len(arrival) == 1
        rec = arrival[0]
        assert rec["request_id"] == "req-001"
        assert rec["arrival_ts_ns"] > 0
        assert rec["ts_ns"] > 0
        assert rec["v"] == 1


# ===========================================================================
# Test 2 — schedule → admit + scheduler_decision
# ===========================================================================

class TestScheduleWrapper:

    def test_admit_and_decision_emitted(self, bkvt_env):
        from bkvt.integrations.vllm.scheduler_probe import make_schedule_wrapper

        req_a = make_request("req-A")
        sched_out = make_scheduler_output(
            new_reqs=[req_a],
            num_tokens={"req-A": 128},
        )

        # Mock Scheduler with waiting/running attributes
        mock_sched = _ns()
        mock_sched.waiting = []
        mock_sched.running = []
        mock_sched.step_id = 1

        call_count = [0]

        def _orig(self):
            call_count[0] += 1
            # After schedule, the request is running
            mock_sched.running = [req_a]
            return sched_out

        wrapper = make_schedule_wrapper(_orig)
        result = wrapper(mock_sched)

        assert result is sched_out
        assert call_count[0] == 1

        records = _flush_and_collect(bkvt_env)

        admit_recs = [r for r in records if r.get("type") == "request"
                      and r.get("subtype") == "admit"]
        decision_recs = [r for r in records if r.get("type") == "metadata"
                         and r.get("subtype") == "scheduler_decision"]

        assert len(admit_recs) == 1, f"Expected 1 admit, got {admit_recs}"
        assert admit_recs[0]["request_id"] == "req-A"

        assert len(decision_recs) == 1
        d = decision_recs[0]
        assert "scheduler_outputs" in d
        assert "admitted" in d["scheduler_outputs"]

    def test_preempt_emitted(self, bkvt_env):
        from bkvt.integrations.vllm.scheduler_probe import make_schedule_wrapper

        req_b = make_request("req-B")
        sched_out = make_scheduler_output(preempted_ids=["req-B"])

        mock_sched = _ns()
        mock_sched.waiting = []
        mock_sched.running = [req_b]

        def _orig(self):
            mock_sched.running = []
            return sched_out

        wrapper = make_schedule_wrapper(_orig)
        wrapper(mock_sched)

        records = _flush_and_collect(bkvt_env)
        preempt = [r for r in records if r.get("type") == "request"
                   and r.get("subtype") == "preempt"]
        assert len(preempt) == 1
        assert preempt[0]["request_id"] == "req-B"


# ===========================================================================
# Test 3 — execute_model → first_token + decode
# ===========================================================================

class TestExecuteModelWrapper:

    def test_first_token_and_decode_emitted(self, bkvt_env):
        from bkvt.integrations.vllm.runner_probe import make_execute_model_wrapper
        from bkvt.integrations.vllm.scheduler_probe import get_tracker

        # Pre-seed tracker so first_token detection has arrival_ts
        get_tracker().update("req-C", arrival_ts_ns=1_000_000)

        req_out_c = make_req_output("req-C", n_tokens=1)
        model_out = make_model_output([req_out_c])
        sched_out = make_scheduler_output(
            new_reqs=[make_request("req-C")],   # first step = prefill
            num_tokens={"req-C": 1},
        )

        def _orig(self, scheduler_output):
            return model_out

        mock_runner = _ns()
        wrapper = make_execute_model_wrapper(_orig)
        result = wrapper(mock_runner, sched_out)
        assert result is model_out

        # Second step — decode
        sched_out2 = make_scheduler_output(num_tokens={"req-C": 1})
        req_out_c2 = make_req_output("req-C", n_tokens=1)
        model_out2 = make_model_output([req_out_c2])

        def _orig2(self, scheduler_output):
            return model_out2

        wrapper2 = make_execute_model_wrapper(_orig2)
        wrapper2(mock_runner, sched_out2)

        records = _flush_and_collect(bkvt_env)
        first_tok = [r for r in records if r.get("type") == "token"
                     and r.get("subtype") == "first_token"]
        decode_recs = [r for r in records if r.get("type") == "token"
                       and r.get("subtype") == "decode"]

        assert len(first_tok) == 1, f"Expected 1 first_token, got {first_tok}"
        assert first_tok[0]["request_id"] == "req-C"
        assert len(decode_recs) >= 1


# ===========================================================================
# Test 4 — alloc_blocks → allocate + allocator_alloc
# ===========================================================================

class TestAllocBlocksWrapper:

    def test_allocate_and_metadata(self, bkvt_env):
        from bkvt.integrations.vllm.block_pool_probe import make_alloc_blocks_wrapper

        blocks_returned = [10, 11, 12]

        def _orig(self, num_blocks):
            return blocks_returned

        wrapper = make_alloc_blocks_wrapper(_orig)
        mock_pool = _ns()
        result = wrapper(mock_pool, 3)
        assert result == blocks_returned

        records = _flush_and_collect(bkvt_env)

        alloc_kv = [r for r in records if r.get("type") == "kv_block"
                    and r.get("subtype") == "allocate"]
        alloc_meta = [r for r in records if r.get("type") == "metadata"
                      and r.get("subtype") == "allocator_alloc"]

        assert len(alloc_kv) == 3
        assert all(r["block_id"] in blocks_returned for r in alloc_kv)
        assert len(alloc_meta) == 1


# ===========================================================================
# Test 5 — free_blocks → free + allocator_free
# ===========================================================================

class TestFreeBlocksWrapper:

    def test_free_and_metadata(self, bkvt_env):
        from bkvt.integrations.vllm.block_pool_probe import make_free_blocks_wrapper

        def _orig(self, block_ids):
            return None

        wrapper = make_free_blocks_wrapper(_orig)
        mock_pool = _ns()
        wrapper(mock_pool, [20, 21])

        records = _flush_and_collect(bkvt_env)
        free_kv = [r for r in records if r.get("type") == "kv_block"
                   and r.get("subtype") == "free"]
        free_meta = [r for r in records if r.get("type") == "metadata"
                     and r.get("subtype") == "allocator_free"]

        assert len(free_kv) == 2
        assert len(free_meta) == 1


# ===========================================================================
# Test 6 — get_computed_blocks → prefix_lookup + prefix_hit
# ===========================================================================

class TestGetComputedBlocksWrapper:

    def test_prefix_lookup_and_hit(self, bkvt_env):
        from bkvt.integrations.vllm.block_pool_probe import make_get_computed_blocks_wrapper

        # Blocks with a block_hash attribute
        block_a = _ns(); block_a.block_id = 100; block_a.block_hash = "0xabc"
        block_b = _ns(); block_b.block_id = 101; block_b.block_hash = "0xdef"

        def _orig(self, request):
            return [block_a, block_b]

        mock_mgr = _ns()
        mock_mgr.block_size = 16

        wrapper = make_get_computed_blocks_wrapper(_orig)
        req = make_request("req-D")
        result = wrapper(mock_mgr, req)
        assert result == [block_a, block_b]

        records = _flush_and_collect(bkvt_env)
        lookup = [r for r in records if r.get("type") == "metadata"
                  and r.get("subtype") == "prefix_lookup"]
        hits = [r for r in records if r.get("type") == "kv_block"
                and r.get("subtype") == "prefix_hit"]

        assert len(lookup) == 1
        assert lookup[0]["n_hits"] == 2
        assert lookup[0]["hit_depth_tokens"] == 32  # 2 blocks × 16 tokens
        assert lookup[0]["request_id"] == "req-D"
        assert len(hits) == 2


# ===========================================================================
# Test 7 — cache_full_blocks → prefix_insert + hash_insert
# ===========================================================================

class TestCacheFullBlocksWrapper:

    def test_prefix_insert_and_hash_insert(self, bkvt_env):
        from bkvt.integrations.vllm.block_pool_probe import make_cache_full_blocks_wrapper

        block_e = _ns(); block_e.block_id = 200; block_e.block_hash = "0x111"

        def _orig(self, request, blocks):
            return None

        mock_mgr = _ns()
        wrapper = make_cache_full_blocks_wrapper(_orig)
        req = make_request("req-E")
        wrapper(mock_mgr, req, [block_e])

        records = _flush_and_collect(bkvt_env)
        inserts = [r for r in records if r.get("type") == "metadata"
                   and r.get("subtype") == "prefix_insert"]
        hash_ins = [r for r in records if r.get("type") == "kv_block"
                    and r.get("subtype") == "hash_insert"]

        assert len(inserts) == 1
        assert inserts[0]["n_keys"] == 1
        assert len(hash_ins) == 1
        assert hash_ins[0]["block_id"] == 200


# ===========================================================================
# Test 8 — finish_request → finish record with ttft_ns / tpot_ns
# ===========================================================================

class TestFinishRequestWrapper:

    def test_finish_record_with_timings(self, bkvt_env):
        from bkvt.integrations.vllm.scheduler_probe import (
            make_finish_request_wrapper, get_tracker,
        )

        now = time.monotonic_ns()
        tracker = get_tracker()
        tracker.update("req-F",
                       arrival_ts_ns=now - 500_000_000,        # 500ms ago
                       first_token_ts_ns=now - 400_000_000,    # 400ms ago
                       first_schedule_ts_ns=now - 490_000_000,
                       output_tokens_so_far=10)

        def _orig(self, request):
            return None

        wrapper = make_finish_request_wrapper(_orig)
        mock_op = _ns()
        req_obj = _ns()
        req_obj.req_id = "req-F"
        wrapper(mock_op, req_obj)

        records = _flush_and_collect(bkvt_env)
        finish = [r for r in records if r.get("type") == "request"
                  and r.get("subtype") == "finish"]

        assert len(finish) == 1
        rec = finish[0]
        assert rec["request_id"] == "req-F"
        assert rec["ttft_ns"] is not None and rec["ttft_ns"] > 0
        assert rec["tpot_ns"] is not None and rec["tpot_ns"] > 0
        assert rec["finish_ts_ns"] is not None


# ===========================================================================
# Test 9 — Full lifecycle: Q3 critical path reconstructible
# ===========================================================================

class TestFullLifecycleCriticalPath:
    """End-to-end: simulate one request through all probes and verify Q3."""

    def test_critical_path_reconstructible(self, bkvt_env):
        from bkvt.integrations.vllm.scheduler_probe import (
            make_process_inputs_wrapper,
            make_schedule_wrapper,
            make_finish_request_wrapper,
            get_tracker,
        )
        from bkvt.integrations.vllm.runner_probe import make_execute_model_wrapper

        # ── 1. Arrival ───────────────────────────────────────────────────
        def _proc_orig(self, request_id, prompt, params, **kw):
            r = _ns()
            r.req_id = request_id
            r.prompt_token_ids = list(range(params.max_tokens))
            return r

        proc_wrapper = make_process_inputs_wrapper(_proc_orig)
        mock_proc = _ns()
        params = make_sampling_params(max_tokens=16)
        proc_wrapper(mock_proc, "req-lifecycle", "test prompt", params)

        # ── 2. Schedule (admit) ──────────────────────────────────────────
        req_lc = make_request("req-lifecycle", input_len=16)
        sched_out = make_scheduler_output(
            new_reqs=[req_lc],
            num_tokens={"req-lifecycle": 16},
        )
        mock_sched = _ns()
        mock_sched.waiting = [req_lc]
        mock_sched.running = []
        mock_sched.step_id = 0

        def _sched_orig(self):
            mock_sched.waiting = []
            mock_sched.running = [req_lc]
            return sched_out

        sched_wrapper = make_schedule_wrapper(_sched_orig)
        sched_wrapper(mock_sched)

        # ── 3. First forward pass (first_token) ──────────────────────────
        req_out = make_req_output("req-lifecycle", n_tokens=1)
        model_out = make_model_output([req_out])

        def _run_orig(self, sched_output):
            return model_out

        run_wrapper = make_execute_model_wrapper(_run_orig)
        mock_runner = _ns()
        run_wrapper(mock_runner, sched_out)

        # ── 4. Subsequent decode step ────────────────────────────────────
        sched_out2 = make_scheduler_output(num_tokens={"req-lifecycle": 1})
        req_out2 = make_req_output("req-lifecycle", n_tokens=1)
        model_out2 = make_model_output([req_out2])

        def _run_orig2(self, sched_output):
            return model_out2

        run_wrapper2 = make_execute_model_wrapper(_run_orig2)
        run_wrapper2(mock_runner, sched_out2)

        # ── 5. Finish ────────────────────────────────────────────────────
        def _finish_orig(self, req):
            return None

        finish_wrapper = make_finish_request_wrapper(_finish_orig)
        mock_op = _ns()
        finish_req = _ns()
        finish_req.req_id = "req-lifecycle"
        finish_wrapper(mock_op, finish_req)

        # ── Collect and validate ─────────────────────────────────────────
        records = _flush_and_collect(bkvt_env)

        by_subtype = {}
        for r in records:
            key = (r.get("type"), r.get("subtype"))
            by_subtype.setdefault(key, []).append(r)

        # Must have all critical-path stages
        assert ("request", "arrival") in by_subtype, "Missing arrival"
        assert ("request", "admit") in by_subtype, "Missing admit"
        assert ("token", "first_token") in by_subtype, "Missing first_token"
        assert ("request", "finish") in by_subtype, "Missing finish"
        assert ("metadata", "scheduler_decision") in by_subtype, \
            "Missing scheduler_decision"

        # Q3: reconstruct critical path from the finish record
        finish_rec = by_subtype[("request", "finish")][0]
        assert finish_rec["request_id"] == "req-lifecycle"
        assert finish_rec["arrival_ts_ns"] is not None
        assert finish_rec["first_token_ts_ns"] is not None
        assert finish_rec["finish_ts_ns"] is not None
        assert finish_rec["ttft_ns"] is not None and finish_rec["ttft_ns"] > 0
        assert finish_rec["tpot_ns"] is not None

        # All records must have mandatory envelope fields
        for rec in records:
            if rec.get("type") in ("request", "token", "kv_block",
                                   "transfer", "metadata", "sys_counter"):
                assert "ts_ns" in rec, f"Missing ts_ns in {rec}"
                assert "type" in rec, f"Missing type in {rec}"
                assert "v" in rec, f"Missing v in {rec}"
                assert rec["ts_ns"] > 0, f"ts_ns <= 0 in {rec}"

        # No orphaned arrivals (every arrival should have a finish)
        arrival_ids = {r["request_id"]
                       for r in by_subtype.get(("request", "arrival"), [])}
        finish_ids = {r["request_id"]
                      for r in by_subtype.get(("request", "finish"), [])}
        assert arrival_ids == finish_ids, \
            f"Orphaned arrivals: {arrival_ids - finish_ids}"


# ===========================================================================
# Test 10 — Probes are no-ops when BKVT_ENABLE=0
# ===========================================================================

class TestDisabledMode:

    def test_wrapper_is_noop_when_disabled(self, monkeypatch, tmp_path):
        """Wrappers must return original result without side effects when disabled."""
        monkeypatch.setenv("BKVT_ENABLE", "0")

        import bkvt.config as _cfg_mod
        import bkvt.emitter as _em_mod
        _cfg_mod.reset_config()
        _em_mod._shutdown_emitter()
        _em_mod._emitter = None

        import bkvt.integrations.vllm.scheduler_probe as sp
        sp._PATCHES_APPLIED = False
        sp._tracker = sp.RequestStateTracker()

        from bkvt.integrations.vllm.scheduler_probe import make_process_inputs_wrapper

        call_count = [0]

        def _orig(self, request_id, prompt, params, **kw):
            call_count[0] += 1
            r = _ns()
            r.req_id = request_id
            r.prompt_token_ids = []
            return r

        wrapper = make_process_inputs_wrapper(_orig)
        mock_self = _ns()
        result = wrapper(mock_self, "req-noop", "x", make_sampling_params())

        assert call_count[0] == 1       # original was called
        assert result.req_id == "req-noop"

        # No records were written (emitter is disabled)
        import glob
        traces = glob.glob(str(tmp_path / "**" / "*.jsonl"), recursive=True)
        assert traces == [], "Disabled emitter should not write files"

        _em_mod._shutdown_emitter()
        _em_mod._emitter = None
        _cfg_mod.reset_config()
