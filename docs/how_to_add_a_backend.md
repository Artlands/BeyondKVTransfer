# How To Add A Backend

Backends are transport or cache implementations that move KV blocks, such as
NIXL, LMCache, shared storage, Mooncake, or a HiCache L2/L3 store.  The BKVT
integration should wrap the backend boundary instead of modifying backend
internals unless lower-level timing is explicitly required.

## Backend Checklist

1. Identify the stable API that starts a logical KV movement.
2. Identify the API or polling site that confirms completion.
3. Map source and destination locations to the tier enum in `DESIGN.md` section 2.3.
4. Emit one `transfer.start` and one `transfer.end` or `transfer.cancel` for
   each logical transfer id.
5. Emit `kv_block.tier_promote` or `kv_block.tier_demote` when block residency
   changes.
6. Emit metadata records for expensive control-plane work such as descriptor
   construction, prefix lookup, backend polling, and completion reaping.
7. Add validation coverage with a fake backend and, where possible, a real
   smoke run.

## Required Transfer Fields

At start time, fill as much of this as the backend knows:

```python
transfer_id = em.transfer_start(
    "load",                         # or "save"
    request_id=request_id,
    layer_idx=layer_idx,
    src_tier="DRAM_REMOTE",
    dst_tier="HBM_LOCAL",
    transport="nixl",
    num_blocks=num_blocks,
    bytes_=num_bytes,
    block_ids=block_ids,
    issued_by="connector",
    issued_at_phase="decode",
    earliest_known_ts_ns=earliest_known_ts_ns,
)
```

At completion time, include observed timing and backend detail:

```python
em.transfer_end(
    transfer_id,
    bytes_=num_bytes,
    wire_time_ns=wire_time_ns,
    achieved_bw_gbps=achieved_bw_gbps,
    wr_count=descriptor_count,
    wr_completion_ts_ns=completion_timestamps,
)
```

## vLLM Backends

Use `bkvt.integrations.vllm.connector_wrapper.TracingConnectorWrapper` for
connector-level coverage.  Add backend-specific wrapping only when the backend
exposes extra timing, such as NIXL descriptor completion data.

Keep the factory patch in `bkvt/integrations/vllm/patch.py` idempotent.  A
fresh connector instance should be wrapped once, and already wrapped instances
should pass through unchanged.

## SGLang Backends

Use `bkvt.integrations.sglang.hicache_probe` for HiCache controller movement
methods and `bkvt.integrations.sglang.disagg_probe` for PD-disaggregation
backends.  Both modules patch known method names only when the corresponding
SGLang modules exist, so add new backend verbs there instead of creating a
parallel patch entrypoint.

Use scheduler/radix/allocator probes for request, prefix, and allocation state.
Use HiCache controller probes for L1/L2/L3 tier transitions.  Use PD
disaggregation backend probes for prefill-to-decode transfers.  Add
Mooncake/NIXL-specific extraction only when the backend exposes extra bytes,
descriptor counts, or completion timestamps.

## Validation

Run schema validation and orphan checks after every backend smoke run:

```bash
python scripts/validate_traces.py <trace-dir> --strict
```

For RDMA backends, compare emitted transfer bytes against
`/sys/class/infiniband/*/counters` over the same interval.  The M3 acceptance
tolerance is within 5%.
