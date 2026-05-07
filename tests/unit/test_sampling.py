"""
Unit tests for bkvt/sampling.py.

Verifies:
- Always-on record types are never dropped.
- Sampled types are dropped at the expected rate (within tolerances).
- sample_decision field is None for always-on events.
- sample_decision equals the configured rate for sampled events.
- Thread-local RNGs don't interfere with each other.
"""

from __future__ import annotations

import threading
from typing import Optional

import pytest

from bkvt.sampling import (
    Sampler,
    _ALWAYS_ON_METADATA_SUBTYPES,
    _ALWAYS_ON_TOKEN_SUBTYPES,
    _ALWAYS_ON_TYPES,
    get_sampler,
    init_sampler,
)


class TestAlwaysOnTypes:
    """request and sys_counter are always emitted regardless of rate."""

    def test_request_always_emits(self):
        s = Sampler(sample_token=0.0, sample_metadata=0.0, sample_transfer=0.0)
        for sub in ("arrival", "admit", "preempt", "resume", "finish", "abort"):
            emit, rate = s.should_emit("request", sub)
            assert emit, f"request/{sub} should always emit"
            assert rate is None

    def test_sys_counter_always_emits(self):
        s = Sampler(sample_token=0.0, sample_metadata=0.0, sample_transfer=0.0)
        emit, rate = s.should_emit("sys_counter", "nic_bytes")
        assert emit
        assert rate is None


class TestAlwaysOnSubtypes:
    """first_token, clock_anchor, scheduler_decision are always emitted."""

    def test_first_token_always_emits(self):
        s = Sampler(sample_token=0.0)
        emit, rate = s.should_emit_token("first_token")
        assert emit
        assert rate is None

    def test_clock_anchor_always_emits(self):
        s = Sampler(sample_metadata=0.0)
        emit, rate = s.should_emit_metadata("clock_anchor")
        assert emit
        assert rate is None

    def test_scheduler_decision_always_emits(self):
        s = Sampler(sample_metadata=0.0)
        emit, rate = s.should_emit_metadata("scheduler_decision")
        assert emit
        assert rate is None


class TestSampledTypes:
    """Verify sampling gate at rate=0 and rate=1."""

    def test_token_rate_zero_drops_all(self):
        s = Sampler(sample_token=0.0)
        for _ in range(100):
            emit, _ = s.should_emit_token("decode")
            assert not emit

    def test_token_rate_one_emits_all(self):
        s = Sampler(sample_token=1.0)
        for _ in range(100):
            emit, rate = s.should_emit_token("decode")
            assert emit
            assert rate is None  # rate=1.0 → treated as always-on

    def test_transfer_rate_zero_drops_all(self):
        s = Sampler(sample_transfer=0.0)
        for _ in range(100):
            emit, _ = s.should_emit_transfer()
            assert not emit

    def test_kv_block_rate_zero_drops_all(self):
        s = Sampler(sample_metadata=0.0)
        for _ in range(100):
            emit, _ = s.should_emit_kv_block("allocate")
            assert not emit

    def test_metadata_rate_zero_drops_non_always_on(self):
        s = Sampler(sample_metadata=0.0)
        for sub in ("prefix_lookup", "prefix_insert", "block_table_update"):
            emit, _ = s.should_emit_metadata(sub)
            assert not emit, f"metadata/{sub} should be dropped at rate=0"


class TestSampleDecisionField:
    """sample_decision is set to the rate when < 1.0 and record is emitted."""

    def test_rate_attached_when_sampled(self):
        # Use rate=1.0 on token so all pass; rate is then None (always-on)
        s = Sampler(sample_token=1.0)
        emit, rate = s.should_emit_token("decode")
        assert emit
        assert rate is None

    def test_rate_attached_when_partially_sampled(self):
        # Force a specific rate and check that the returned rate matches.
        rate_value = 0.5
        s = Sampler(sample_token=rate_value)
        # Collect results over many trials; those that emit should report rate.
        emitted_rates = set()
        for _ in range(200):
            emit, rate = s.should_emit_token("decode")
            if emit:
                emitted_rates.add(rate)
        # All emitted records should carry the same rate
        assert len(emitted_rates) == 1
        assert next(iter(emitted_rates)) == pytest.approx(rate_value)

    def test_rate_none_for_always_on_request(self):
        s = Sampler(sample_transfer=0.5)
        emit, rate = s.should_emit("request", "arrival")
        assert emit
        assert rate is None


class TestSamplingRate:
    """Statistical check: actual emission rate ≈ configured rate."""

    @pytest.mark.parametrize("configured_rate", [0.1, 0.5, 0.9])
    def test_emission_rate_approx(self, configured_rate: float):
        s = Sampler(sample_token=configured_rate)
        n = 10_000
        hits = sum(1 for _ in range(n) if s.should_emit_token("decode")[0])
        actual = hits / n
        # Allow ±5 percentage points margin
        assert abs(actual - configured_rate) < 0.05, (
            f"rate={configured_rate}: got {actual:.3f}"
        )


class TestThreadSafety:
    """RNGs are per-thread; concurrent sampling does not raise."""

    def test_concurrent_sampling(self):
        s = Sampler(sample_token=0.5)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(1000):
                    s.should_emit_token("decode")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"


class TestSingleton:
    """init_sampler / get_sampler singleton contract."""

    def test_init_returns_sampler_instance(self):
        s = init_sampler(sample_token=0.1, sample_metadata=0.9, sample_transfer=0.5)
        assert isinstance(s, Sampler)

    def test_get_sampler_returns_same_after_init(self):
        s1 = init_sampler(sample_token=0.2)
        s2 = get_sampler()
        assert s1 is s2

    def test_get_sampler_creates_default(self):
        # Reset singleton to test auto-creation
        import bkvt.sampling as _sm
        original = _sm._sampler
        _sm._sampler = None
        try:
            s = get_sampler()
            assert isinstance(s, Sampler)
        finally:
            _sm._sampler = original
