"""M9 smoke tests for weight/adaptor probes and payload-kind tagging."""

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
    monkeypatch.setenv("BKVT_TRACE_ID", "test-trace-m9")
    monkeypatch.setenv("BKVT_SAMPLE_TOKEN", "1.0")
    monkeypatch.setenv("BKVT_SAMPLE_METADATA", "1.0")
    monkeypatch.setenv("BKVT_SAMPLE_TRANSFER", "1.0")
    monkeypatch.setenv("BKVT_SYS_COUNTER_HZ", "0")
    monkeypatch.setenv("BKVT_CLOCK_ANCHOR_HZ", "0")

    import bkvt.config as _cfg_mod
    import bkvt.emitter as _em_mod
    import bkvt.sampling as _samp_mod

    _cfg_mod.reset_config()
    _em_mod._shutdown_emitter()
    _em_mod._emitter = None
    _samp_mod._sampler = None
    yield _em_mod.get_emitter()
    _em_mod._shutdown_emitter()
    _em_mod._emitter = None
    _cfg_mod.reset_config()


def _collect(em: Any) -> list[dict]:
    time.sleep(0.25)
    em.shutdown()
    records: list[dict] = []
    for path in glob.glob(os.path.join(em._config.output_dir, "**", "*.jsonl*"), recursive=True):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    return records


class Tensor:
    def __init__(self, nbytes: int, shape=(2, 4), dtype="bf16") -> None:
        self.nbytes = nbytes
        self.shape = shape
        self.dtype = dtype


class Model:
    def state_dict(self) -> dict[str, Tensor]:
        return {"model.layers.0.mlp.weight": Tensor(64)}


def test_vllm_weight_probe_families_and_payload_disambiguation(bkvt_env):
    from bkvt.integrations.vllm.weight_probe import (
        make_collective_wrapper,
        make_load_model_wrapper,
        make_lora_wrapper,
        make_moe_wrapper,
        make_update_weights_wrapper,
        payload_context,
    )

    def load_orig(self):
        return Model()

    load = make_load_model_wrapper(load_orig, backend="vllm", transport="mmap")
    assert isinstance(load(SimpleNamespace(tp_rank=0)), Model)

    def collective_orig(tensor=None, bytes=None):
        return tensor

    collective = make_collective_wrapper(collective_orig, "broadcast_tensor_dict", transport="nccl_broadcast")
    with payload_context("weight", param_name="model.layers.0.mlp.weight", reason="startup_load"):
        collective(bytes=64)
    collective_activation = make_collective_wrapper(
        collective_orig,
        "all_to_all",
        transport="nccl_send_recv",
        default_payload_kind="activation",
    )
    collective_activation(bytes=32)

    def moe_orig(self, hidden_states=None, expert_ids=None, bytes=None):
        return hidden_states

    moe = make_moe_wrapper(moe_orig, "FusedMoE.forward")
    moe(SimpleNamespace(), expert_ids=[1, 2, 1], bytes=128)

    def lora_orig(self, adapter_id=None, bytes=None, request_id=None):
        return None

    lora = make_lora_wrapper(lora_orig, "set_active_loras")
    lora(SimpleNamespace(), adapter_id="adapter-a", bytes=256, request_id="req-lora")

    def update_orig(self, weights=None, weight_version=None):
        return None

    update = make_update_weights_wrapper(update_orig, "update_weights_from_distributed")
    update(SimpleNamespace(), weights={"model.layers.1.attn.weight": Tensor(96)}, weight_version="update-1")

    records = _collect(bkvt_env)
    weights = [r for r in records if r.get("type") == "weight_block"]
    subtypes = {r["subtype"] for r in weights}
    assert {"load", "expert_dispatch", "lora_activate", "update_apply"}.issubset(subtypes)

    transfers = [r for r in records if r.get("type") == "transfer" and r.get("subtype") == "start"]
    assert any(r.get("payload_kind") == "weight" and r.get("param_name") == "model.layers.0.mlp.weight" for r in transfers)
    assert any(r.get("payload_kind") == "activation" and r.get("transport") == "nccl_send_recv" for r in transfers)
    assert any(r.get("payload_kind") == "weight" and r.get("lora_adapter_id") == "adapter-a" for r in transfers)
    assert any(r.get("payload_kind") == "weight" and r.get("weight_version") == "update-1" for r in transfers)


def test_sglang_weight_probe_wrappers_emit_q7_records(bkvt_env):
    from bkvt.integrations.sglang.weight_probe import make_lora_wrapper, make_update_weights_wrapper

    def load_adapter(self, adapter_id=None, bytes=None):
        return None

    lora = make_lora_wrapper(load_adapter, "load_lora_adapter")
    lora(SimpleNamespace(), adapter_id="sg-adapter", bytes=128)

    def update_from_disk(self, weights=None, version=None):
        return None

    update = make_update_weights_wrapper(update_from_disk, "update_weights_from_disk")
    update(SimpleNamespace(), weights={"model.layers.2.weight": Tensor(80)}, version="sg-update")

    records = _collect(bkvt_env)
    assert any(r.get("type") == "weight_block" and r.get("subtype") == "lora_activate" for r in records)
    assert any(r.get("type") == "weight_block" and r.get("subtype") == "update_apply" for r in records)
    assert any(r.get("type") == "transfer" and r.get("payload_kind") == "weight" for r in records)
