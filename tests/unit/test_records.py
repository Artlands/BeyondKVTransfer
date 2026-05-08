"""
Unit tests for bkvt/records.py.

Verifies:
- All record types can be instantiated with required fields.
- to_dict() produces the correct "type" field.
- None-valued optional fields are stripped.
- SCHEMA_VERSION is present and consistent.
- Tier enum values are consistent with §2.3.
"""

from __future__ import annotations

import pytest

from bkvt.records import (
    SCHEMA_VERSION,
    KVBlockRecord,
    MetadataRecord,
    RequestRecord,
    SysCounterRecord,
    Tier,
    TokenRecord,
    TransferEndpoint,
    TransferRecord,
    WeightBlockRecord,
    _strip_none,
)


# ---------------------------------------------------------------------------
# _strip_none helper
# ---------------------------------------------------------------------------

class TestStripNone:
    def test_removes_none_values(self):
        assert _strip_none({"a": 1, "b": None, "c": "x"}) == {"a": 1, "c": "x"}

    def test_empty_dict(self):
        assert _strip_none({}) == {}

    def test_keeps_zero_and_false(self):
        d = {"a": 0, "b": False, "c": "", "d": None}
        result = _strip_none(d)
        assert "a" in result and result["a"] == 0
        assert "b" in result and result["b"] is False
        assert "c" in result and result["c"] == ""
        assert "d" not in result


# ---------------------------------------------------------------------------
# Common envelope fields
# ---------------------------------------------------------------------------

_ENVELOPE = dict(
    ts_ns=1_000_000_000,
    trace_id="trace-001",
    node_id="host-0",
    worker_id="host-0/tp0/pp0",
)


class TestSchemaVersion:
    def test_is_integer(self):
        assert isinstance(SCHEMA_VERSION, int)

    def test_is_positive(self):
        assert SCHEMA_VERSION >= 1

    def test_record_carries_version(self):
        r = RequestRecord(
            **_ENVELOPE,
            request_id="req-1",
            subtype="arrival",
        )
        d = r.to_dict()
        assert d["v"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# RequestRecord
# ---------------------------------------------------------------------------

class TestRequestRecord:
    def _make(self, subtype="arrival", **kw):
        return RequestRecord(
            **_ENVELOPE,
            request_id="req-1",
            subtype=subtype,
            **kw,
        )

    def test_type_field(self):
        assert self._make().to_dict()["type"] == "request"

    def test_subtype_preserved(self):
        for sub in ("arrival", "admit", "preempt", "resume", "finish", "abort"):
            assert self._make(subtype=sub).to_dict()["subtype"] == sub

    def test_none_fields_stripped(self):
        d = self._make().to_dict()
        assert "ttft_ns" not in d
        assert "tpot_ns" not in d
        assert "seq_id" not in d

    def test_optional_fields_included_when_set(self):
        r = self._make(
            input_len=512,
            ttft_ns=12_000_000,
            scheduler_state_snapshot={"running": 3, "waiting": 1, "swapped": 0,
                                      "free_blocks": 100, "used_blocks": 400},
        )
        d = r.to_dict()
        assert d["input_len"] == 512
        assert d["ttft_ns"] == 12_000_000
        assert d["scheduler_state_snapshot"]["running"] == 3

    def test_finish_record_full_lifecycle(self):
        r = self._make(
            subtype="finish",  # type: ignore[call-arg]
            arrival_ts_ns=100,
            first_token_ts_ns=500,
            finish_ts_ns=900,
            ttft_ns=400,
            tpot_ns=100,
        )
        d = r.to_dict()
        assert d["ttft_ns"] == 400
        assert d["tpot_ns"] == 100


# ---------------------------------------------------------------------------
# TokenRecord
# ---------------------------------------------------------------------------

class TestTokenRecord:
    def _make(self, subtype="decode", **kw):
        return TokenRecord(
            **_ENVELOPE,
            subtype=subtype,
            request_id="req-1",
            token_idx=10,
            step_id=42,
            **kw,
        )

    def test_type_field(self):
        assert self._make().to_dict()["type"] == "token"

    def test_default_num_tokens(self):
        assert self._make().to_dict()["num_tokens"] == 1

    def test_sample_decision_stripped_when_none(self):
        assert "sample_decision" not in self._make().to_dict()

    def test_sample_decision_included_when_set(self):
        d = self._make(sample_decision=0.05).to_dict()
        assert d["sample_decision"] == pytest.approx(0.05)

    def test_first_token_subtype(self):
        d = self._make(subtype="first_token").to_dict()  # type: ignore[call-arg]
        assert d["subtype"] == "first_token"


# ---------------------------------------------------------------------------
# KVBlockRecord
# ---------------------------------------------------------------------------

class TestKVBlockRecord:
    def _make(self, subtype="allocate", **kw):
        return KVBlockRecord(
            **_ENVELOPE,
            subtype=subtype,
            block_id=12345,
            **kw,
        )

    def test_type_field(self):
        assert self._make().to_dict()["type"] == "kv_block"

    def test_tier_fields(self):
        r = self._make(
            subtype="tier_promote",
            tier_before=Tier.DRAM_LOCAL,
            tier_after=Tier.HBM_LOCAL,
        )
        d = r.to_dict()
        assert d["tier_before"] == "DRAM_LOCAL"
        assert d["tier_after"] == "HBM_LOCAL"

    def test_tier_same_allowed(self):
        r = self._make(
            subtype="prefix_hit",
            tier_before=Tier.HBM_LOCAL,
            tier_after=Tier.HBM_LOCAL,
        )
        d = r.to_dict()
        assert d["tier_before"] == d["tier_after"] == "HBM_LOCAL"

    def test_block_hash_hex(self):
        r = self._make(block_hash="0xdeadbeef")
        assert r.to_dict()["block_hash"] == "0xdeadbeef"


# ---------------------------------------------------------------------------
# TransferRecord
# ---------------------------------------------------------------------------

class TestTransferRecord:
    def _make(self, subtype="start", direction="load", **kw):
        return TransferRecord(
            **_ENVELOPE,
            subtype=subtype,
            transfer_id="xfer-001",
            direction=direction,
            **kw,
        )

    def test_type_field(self):
        assert self._make().to_dict()["type"] == "transfer"

    def test_direction_values(self):
        for d in ("load", "save"):
            rec = self._make(direction=d).to_dict()
            assert rec["direction"] == d

    def test_src_dst_serialised(self):
        r = self._make(
            src=TransferEndpoint(tier=Tier.DRAM_REMOTE, node_id="peer-0"),
            dst=TransferEndpoint(tier=Tier.HBM_LOCAL, device_id=0),
        )
        d = r.to_dict()
        assert d["src"]["tier"] == "DRAM_REMOTE"
        assert d["src"]["node_id"] == "peer-0"
        assert d["dst"]["tier"] == "HBM_LOCAL"
        assert d["dst"]["device_id"] == 0
        assert "node_id" not in d["dst"]  # stripped because None

    def test_end_record_subtype(self):
        r = self._make(subtype="end", completed_ts_ns=999)  # type: ignore[call-arg]
        d = r.to_dict()
        assert d["subtype"] == "end"
        assert d["completed_ts_ns"] == 999

    def test_earliest_known_ts_ns(self):
        r = self._make(earliest_known_ts_ns=500, started_ts_ns=700)
        d = r.to_dict()
        assert d["earliest_known_ts_ns"] == 500

    def test_payload_kind_defaults_to_kv(self):
        assert self._make().to_dict()["payload_kind"] == "kv"

    def test_weight_identity_fields(self):
        r = self._make(
            payload_kind="weight",
            param_name="model.layers.0.mlp.weight",
            lora_adapter_id="adapter-a",
            weight_version="v1",
        )
        d = r.to_dict()
        assert d["payload_kind"] == "weight"
        assert d["param_name"] == "model.layers.0.mlp.weight"
        assert d["lora_adapter_id"] == "adapter-a"


# ---------------------------------------------------------------------------
# WeightBlockRecord
# ---------------------------------------------------------------------------

class TestWeightBlockRecord:
    def _make(self, subtype="load", **kw):
        return WeightBlockRecord(
            **_ENVELOPE,
            subtype=subtype,
            param_name="model.layers.0.mlp.weight",
            **kw,
        )

    def test_type_field(self):
        assert self._make().to_dict()["type"] == "weight_block"

    def test_lora_fields(self):
        d = self._make(
            subtype="lora_activate",
            lora_adapter_id="adapter-a",
            tier_before=Tier.DRAM_LOCAL,
            tier_after=Tier.HBM_LOCAL,
            reason="lora_swap",
        ).to_dict()
        assert d["lora_adapter_id"] == "adapter-a"
        assert d["reason"] == "lora_swap"

    def test_update_version(self):
        d = self._make(subtype="update_apply", weight_version="update-1").to_dict()
        assert d["weight_version"] == "update-1"


# ---------------------------------------------------------------------------
# MetadataRecord
# ---------------------------------------------------------------------------

class TestMetadataRecord:
    def _make(self, subtype="prefix_lookup", **kw):
        return MetadataRecord(
            **_ENVELOPE,
            subtype=subtype,
            **kw,
        )

    def test_type_field(self):
        assert self._make().to_dict()["type"] == "metadata"

    def test_duration_ns_included(self):
        r = self._make(duration_ns=12_500)
        assert r.to_dict()["duration_ns"] == 12_500

    def test_clock_anchor_fields(self):
        r = self._make(
            subtype="clock_anchor",
            t0_unix_ns=1_700_000_000_000_000_000,
            t0_monotonic_ns=1_000_000_000,
        )
        d = r.to_dict()
        assert d["subtype"] == "clock_anchor"
        assert d["t0_unix_ns"] == 1_700_000_000_000_000_000


# ---------------------------------------------------------------------------
# SysCounterRecord
# ---------------------------------------------------------------------------

class TestSysCounterRecord:
    def _make(self, **kw):
        return SysCounterRecord(
            **_ENVELOPE,
            subtype="nic_bytes",
            scope="nic:mlx5_0",
            value=1_234_567_890,
            unit="bytes",
            **kw,
        )

    def test_type_field(self):
        assert self._make().to_dict()["type"] == "sys_counter"

    def test_value_and_unit(self):
        d = self._make().to_dict()
        assert d["value"] == 1_234_567_890
        assert d["unit"] == "bytes"

    def test_interval_stripped_when_none(self):
        assert "interval_ns" not in self._make().to_dict()

    def test_interval_included_when_set(self):
        d = self._make(interval_ns=100_000_000).to_dict()
        assert d["interval_ns"] == 100_000_000


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------

class TestTier:
    def test_all_values_defined(self):
        expected = {
            "HBM_LOCAL", "HBM_PEER_NVLINK", "HBM_PEER_RDMA",
            "DRAM_LOCAL", "DRAM_REMOTE",
            "SSD_LOCAL", "SSD_REMOTE",
            "OBJECT_STORE",
        }
        assert Tier.ALL == expected

    def test_constants_match_all(self):
        for val in Tier.ALL:
            assert hasattr(Tier, val), f"Tier.{val} missing"
