# BeyondKVTransfer

BeyondKVTransfer is a measurement framework for characterizing remote memory
behavior in distributed LLM serving.  It emits newline-delimited JSON traces
that connect request latency, KV cache block lifecycle, remote transfer
activity, metadata operations, and system counters.

The project is driven by `DESIGN.md`, which is the authoritative specification
for record schemas, hook locations, milestones, and acceptance criteria.

## What It Measures

The framework is built to answer six questions:

- How much remote KV data moves, at what granularity, and when?
- How often do prefix-cache, allocator, refcount, eviction, and block-table
  metadata operations occur?
- Which events delay TTFT, TPOT, and tail latency?
- How long do KV blocks live, where do they reside, and how often are they
  reused?
- How early could the system know that a remote block will be needed?
- How do scheduler placement decisions affect remote traffic and latency?

Records are emitted at six levels: `request`, `token`, `kv_block`, `transfer`,
`metadata`, and `sys_counter`.

## Current Status

Implemented:

- Core emitter, config, clock, IDs, sampling, manifest, and record dataclasses.
- JSON Schemas under `schemas/`.
- Trace validator: `scripts/validate_traces.py`.
- vLLM integration modules for scheduler, block pool, runner, and connector
  wrapping.
- SGLang scheduler/radix/allocator probes plus HiCache and PD-disaggregation
  transfer wrappers.
- System counter collectors and clock anchor support.
- Analysis loader/index utilities and Q1-Q6 helper libraries.
- Overhead validation scaffolding under `tests/overhead/`.
- How-to docs under `docs/`.

Deferred:

- Native NCCL profiler plugin and hardware-backed long-run acceptance on a
  multi-node SGLang deployment.

## Installation

Use Python 3.10 or newer.

```bash
python -m pip install -e ".[dev,analysis]"
```

For vLLM tracing, install vLLM in the same environment.  Optional analysis
dependencies include `pandas`, `pyarrow`, `duckdb`, and Jupyter.

## Quick Start: vLLM Trace

Run a vLLM server with BKVT tracing enabled:

```bash
BKVT_OUTPUT_DIR=./bkvt_traces \
BKVT_PROFILE=coarse \
MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
./scripts/run_vllm_with_bkvt.sh --tensor-parallel-size 1
```

The launcher sets `BKVT_ENABLE=1` by default and applies the vLLM patch before
engine initialization.

Common environment variables:

```bash
BKVT_ENABLE=1
BKVT_OUTPUT_DIR=./bkvt_traces
BKVT_TRACE_ID=my-run-id
BKVT_PROFILE=full            # full | coarse | request_only
BKVT_SAMPLE_TOKEN=0.05
BKVT_SAMPLE_METADATA=1.0
BKVT_SAMPLE_TRANSFER=1.0
BKVT_SYS_COUNTER_HZ=10
```

## Trace Layout

Traces are written as JSONL files under:

```text
${BKVT_OUTPUT_DIR}/
  ${trace_id}/
    manifest.json
    ${node_id}/
      ${worker_id}.0001.jsonl
      ${worker_id}.0002.jsonl
      sys_counters.jsonl
```

Files rotate by size and rotated files may be gzipped in the background.

## Validate A Trace

```bash
python scripts/validate_traces.py ./bkvt_traces --strict
```

Validation checks JSON parsing, schema conformance, schema version, required
envelope fields, transfer pairing, request pairing, and nearby manifest
presence.

## Build Analysis Tables

For a lightweight local smoke test:

```bash
python analysis/build_index.py ./bkvt_traces/<trace_id> \
  --format jsonl \
  --output-dir analysis/index \
  --overwrite
```

For production analysis, use the default Parquet output with `pyarrow` or
`duckdb` installed.

## Tests

Run the unit and integration tests:

```bash
python -m pytest tests
```

Run only the overhead tests:

```bash
python -m pytest tests/overhead
```

The lab-rig synthetic overhead comparison is opt-in:

```bash
BKVT_RUN_OVERHEAD=1 python -m pytest tests/overhead/test_overhead_budget.py
```

The full overhead acceptance run should compare a pinned vLLM workload with
`BKVT_ENABLE=0` and `BKVT_ENABLE=1` on the same hardware.

## Documentation

- `DESIGN.md`: authoritative design and milestone contract.
- `docs/how_to_run.md`: reproduce and validate a vLLM trace.
- `docs/how_to_add_a_probe.md`: add a new probe safely.
- `docs/how_to_add_a_backend.md`: add a transport or cache backend wrapper.

## Repository Map

```text
bkvt/                  Runtime package
bkvt/integrations/     Engine-specific probes
bkvt/collectors/       System, NCCL, IB, and NVML collectors
schemas/               JSON Schemas for trace records
scripts/               Launch and validation scripts
analysis/              Trace loading, indexing, and derived metrics
tests/                 Unit, integration, and overhead tests
docs/                  Operator and developer how-to guides
```

## Notes

- Traces contain identifiers, sizes, timings, tiers, and hashes.  They must not
  contain raw token IDs or KV tensor contents.
- `DESIGN.md` and `schemas/` must be updated together when adding record
  subtypes or changing schema fields.
- BKVT is disabled by default.  With `BKVT_ENABLE=0`, probes are expected to be
  no-ops on the hot path.
