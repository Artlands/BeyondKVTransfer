# How To Add A Probe

Start from `DESIGN.md` section 4.  Every emitted record must match one of the six
schema types: `request`, `token`, `kv_block`, `transfer`, `metadata`, or
`sys_counter`.  If a new subtype is needed, update both `DESIGN.md` and the
matching file under `schemas/` in the same change.

## Probe Rules

Keep probe code importable without the target engine installed.  Integration
modules should import vLLM or SGLang only inside patch functions.

Use `BKVT_ENABLE` as the master gate.  Disabled probes must return after one
cheap boolean check and must not allocate hot-path records.

Prefer wrappers around stable public methods.  If a probe wraps an evolving
engine function, record the expected signature and log a one-time warning when
the live signature drifts.

Keep raw payloads out of traces.  Token ids and KV tensor contents are not
allowed; use request ids, block ids, hashes, sizes, tiers, and timestamps.

## Metadata Probe Pattern

```python
from bkvt import emitter

def wrapped_lookup(inner, request, *args, **kwargs):
    em = emitter.get_emitter()
    if not em.enabled:
        return inner(request, *args, **kwargs)

    with em.metadata("prefix_lookup", request_id=request.request_id) as rec:
        result = inner(request, *args, **kwargs)
        rec["n_hits"] = len(result)
        rec["structure"] = "radix"
        return result
```

## Transfer Probe Pattern

```python
tid = em.transfer_start(
    "load",
    request_id=request_id,
    src_tier="DRAM_REMOTE",
    dst_tier="HBM_LOCAL",
    transport="nixl",
    bytes_=num_bytes,
    issued_by="connector",
    issued_at_phase="prefill",
    earliest_known_ts_ns=earliest_known_ts_ns,
)
try:
    return inner.start_load_kv(*args, **kwargs)
finally:
    em.transfer_end(tid, bytes_=num_bytes)
```

If transfer sampling drops the start record, `transfer_end()` also drops the
end record so validation does not see orphans.

## Tests

Add or update tests at the narrowest useful level:

```bash
pytest tests/unit
pytest tests/integration
python scripts/validate_traces.py <sample-trace> --strict
```

For a new record field, add a unit test for the dataclass or emitter path and
a schema validation test.  For a new integration hook, add a fake-engine smoke
test when the real engine is too heavy for CI.
