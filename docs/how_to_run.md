# How To Reproduce A Trace

This guide covers the current vLLM path.  SGLang scheduler/radix and long
HiCache/disaggregation integration are tracked separately, so do not use this
as proof that M5 is complete.

## Install

From the repository root:

```bash
python -m pip install -e ".[dev,analysis]"
```

Install vLLM in the same environment, then confirm BKVT imports:

```bash
python -c "import bkvt; print(bkvt.__version__)"
```

## Run vLLM With Tracing

Use the launcher so the vLLM patch is applied before engine init:

```bash
BKVT_OUTPUT_DIR=./bkvt_traces \
BKVT_PROFILE=coarse \
MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
./scripts/run_vllm_with_bkvt.sh --tensor-parallel-size 1
```

For connector experiments, also configure the vLLM KV connector backend as
usual.  BKVT wraps connector instances through the vLLM patch layer and emits
`transfer.start`, `transfer.end`, and connector metadata records when the
backend exposes the relevant calls.

Useful environment variables:

```bash
BKVT_ENABLE=1
BKVT_OUTPUT_DIR=./bkvt_traces
BKVT_TRACE_ID=my-repeatable-run
BKVT_PROFILE=full            # full | coarse | request_only
BKVT_SAMPLE_TOKEN=0.05
BKVT_SAMPLE_METADATA=1.0
BKVT_SAMPLE_TRANSFER=1.0
BKVT_SYS_COUNTER_HZ=10
```

## Validate

After stopping the server, validate the trace directory:

```bash
python scripts/validate_traces.py ./bkvt_traces --strict
```

A valid trace has a `manifest.json`, schema-valid JSONL records, matching
transfer start/end or cancel pairs, and matching request arrival/finish or
abort pairs.

## Build Analysis Tables

Use JSONL output for a lightweight smoke test:

```bash
python analysis/build_index.py ./bkvt_traces/<trace_id> \
  --format jsonl \
  --output-dir analysis/index \
  --overwrite
```

Use the default Parquet output on analysis machines with `pyarrow` or `duckdb`
installed.

## Run Overhead Validation

Normal CI runs the fast disabled-path and sampling checks:

```bash
pytest tests/overhead
```

On the lab rig, enable the synthetic budget comparison:

```bash
BKVT_RUN_OVERHEAD=1 pytest tests/overhead/test_overhead_budget.py
```

For the final M8 acceptance run, compare a pinned vLLM ShareGPT workload with
`BKVT_ENABLE=0` and `BKVT_ENABLE=1` on the same machine.  The section 9 budgets are:
TTFT <= 2%, TPOT <= 1%, throughput <= 3%, per-record CPU cost <= 1.5 us, and
default trace size <= 200 KiB per request.
