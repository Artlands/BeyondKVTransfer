# BeyondKVTransfer — Design Document

**Project:** Characterizing Remote Memory Behavior in Distributed LLM Serving
**Status:** Draft v0.1 (specification for downstream implementation agents)
**Targets:** vLLM (v1 engine) and SGLang
**Output format:** JSON traces, six event levels, post-hoc analyzable

---

## 0. How to read this document

This document is the single authoritative specification for the measurement framework. Downstream coding agents should treat it as a contract:

- **Sections 1–3** define what we measure and why.
- **Section 4** defines the on-disk JSON record schemas. Every probe emits one of these record types; agents must not invent new top-level types without updating this section.
- **Sections 5–7** are the *hook map*: where, in vLLM and SGLang, each probe is inserted. File paths are given relative to each project's repo root.
- **Sections 8–10** describe the trace pipeline, overhead budget, and sampling.
- **Section 11** maps events back to the six research questions so an agent can confirm coverage.
- **Sections 12–15** cover configuration, repository layout, milestones, and risks.

When a section says **MUST**, an implementation that violates it is wrong. **SHOULD** is a strong recommendation; deviations require a written justification in the PR description.

---

## 1. Project goals and research questions

We want to characterize how distributed LLM-serving systems use *remote* memory — KV blocks that do not live in the local HBM of the GPU executing the current attention kernel — and quantify the impact on serving latency and throughput. "Remote" is intentionally broad and includes:

- KV blocks that live on a peer GPU within the same node (NVLink/NVSwitch via NCCL or P2P copy).
- KV blocks that live on a peer GPU on a different node (RDMA via NIXL, IB Verbs, GDRCopy).
- KV blocks that have been offloaded to host DRAM, SSD, or an external KV cache pool (LMCache, Mooncake Store, HiCache L2/L3).

The framework MUST produce evidence to answer the following six questions.-

### Q1. Remote KV data path
*How much KV data moves remotely, at what granularity, and when?*
We need to know byte volume, number of transfers, page/block size, layer-fan-out, and the time within the request lifecycle (prefill vs. decode, layer index, token index) at which each transfer happens.

### Q2. Remote metadata path
*How often do block-table, prefix-cache, allocation, refcount, and eviction operations occur?*
Metadata traffic — block-table updates, prefix-cache lookups and inserts, allocator calls, ref-count bumps, evictions — is often invisible in profiles but can dominate latency in disaggregated setups. We measure each operation's frequency, latency, and triggering request.

### Q3. Critical path
*Which events delay TTFT, TPOT, or P99 latency?*
For each request we attribute end-to-end latency to a sequence of named stages (queue → schedule → block-allocate → remote-pull → prefill-forward → first-token → decode-step-i → ...). The framework MUST emit enough information to reconstruct a per-request critical path and to compute attribution histograms across a workload.

### Q4. Reuse / locality
*How long do KV blocks live, how often are they reused, and where?*
For each KV block we record its birth (allocate), all reuse hits (prefix-cache match, remote pull), tier transitions (HBM↔DRAM↔SSD↔remote), and death (evict / free). From this we can compute reuse-distance distributions, per-tier residency CDFs, and hit-rate vs. cache-size curves.

### Q5. Prefetchability
*How early can the system know that a remote KV block will be needed?*
For each "use" of a remote block, we need both the *first instant the system could have known* (request-arrival, scheduler-decision, prefix-match) and the *instant the block was actually requested*. The gap is the prefetch budget. The framework MUST emit timestamps for both.

### Q6. Scheduling impact
*How do placement decisions affect remote traffic, cache hit rate, and tail latency?*
For each scheduling decision (admit, preempt, route to prefill/decode worker, choose KV-pool tier) we record the inputs the scheduler saw and the resulting placement. Cross-referenced against Q1/Q4 events, this lets us isolate the effect of scheduling policy from workload variance.

These six questions drive every probe in §5–§7. Section 11 is the back-pointer table: for each question, which probes feed it.

---

## 2. Background on the target systems

### 2.1 vLLM v1

vLLM's v1 engine separates the API server, the engine core, the scheduler, and the model executor into asyncio-friendly components. The KV cache is paged. Relevant subsystems:

- **`vllm/v1/core/kv_cache_manager.py`** — `KVCacheManager` allocates/free/reuses blocks at scheduling time and is the single owner of the prefix-cache tree.
- **`vllm/v1/core/block_pool.py`** — `BlockPool` owns the free-list and the block hash → block-id map.
- **`vllm/v1/core/single_type_kv_cache_manager.py`** — per-layer-type managers (attention layers, MLA, sliding-window, etc.).
- **`vllm/v1/core/kv_cache_utils.py`** — block-hash construction, used by prefix caching.
- **`vllm/v1/core/sched/scheduler.py`** — `Scheduler.schedule()` chooses which requests to run each step.
- **`vllm/v1/worker/gpu_model_runner.py`** — forward pass, attention layers, KV writes/reads.
- **`vllm/distributed/kv_transfer/kv_connector/v1/base.py`** — `KVConnectorBase_V1` abstract API (the canonical hook surface for remote KV).
- **`vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`** — NIXL backend (RDMA path).
- **`vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`** — LMCache backend (DRAM/SSD/remote pool).
- **`vllm/distributed/kv_transfer/kv_connector/v1/shared_storage_connector.py`** — shared file/object storage backend.
- **`vllm/distributed/kv_transfer/kv_connector/factory.py`** — connector registry.

The connector lifecycle methods we instrument:

| Method | Where it runs | What it tells us |
|---|---|---|
| `get_num_new_matched_tokens(request, num_computed_tokens)` | scheduler | "remote prefix can contribute N more tokens beyond local hit" |
| `update_state_after_alloc(request, blocks, num_external_tokens)` | scheduler | which blocks were earmarked for remote-load this step |
| `build_connector_meta(scheduler_output)` | scheduler | per-step transfer plan handed to workers |
| `start_load_kv(forward_context, ...)` | worker (pre-forward) | issue remote pull |
| `wait_for_layer_load(layer_name)` | worker (per layer) | block until layer's KV has arrived |
| `save_kv_layer(layer_name, kv_layer, ...)` | worker (per layer) | issue remote save (push) |
| `wait_for_save()` | worker (post-forward) | block until pushes drain |
| `get_finished(finished_req_ids)` | scheduler | reap async transfer completions |

Every connector method is a probe site (§5.4).

### 2.2 SGLang

SGLang is RadixAttention-centered. Relevant subsystems:

- **`python/sglang/srt/managers/scheduler.py`** — `Scheduler.run_batch()` and `Scheduler.event_loop()`.
- **`python/sglang/srt/mem_cache/radix_cache.py`** — `RadixCache` (prefix tree over token sequences).
- **`python/sglang/srt/mem_cache/hiradix_cache.py`** — `HiRadixCache`, the hierarchical extension that knows about L1 (HBM) / L2 (host) / L3 (storage).
- **`python/sglang/srt/mem_cache/memory_pool.py`** — `ReqToTokenPool` and host-side KV pool variants (`MHATokenToKVPoolHost`, etc.).
- **`python/sglang/srt/mem_cache/allocator.py`** — `BaseTokenToKVPoolAllocator` and concrete subclasses (paged, slab).
- **`python/sglang/srt/managers/cache_controller.py`** — `HiCacheController`, the L1↔L2↔L3 mover.
- **`python/sglang/srt/managers/tp_worker.py`** — TP worker forward path.
- **`python/sglang/srt/disaggregation/`** — PD disaggregation:
  - `disaggregation/prefill.py`, `disaggregation/decode.py` — event loops on each side.
  - `disaggregation/mooncake/conn.py` — Mooncake transfer backend.
  - `disaggregation/nixl/` — NIXL transfer backend.

In SGLang, "remote" can mean: a peer worker (PD-disagg), an L2/L3 tier in HiCache, or a Mooncake/NIXL/3FS storage backend.

### 2.3 What counts as "remote"

For trace classification we use a single `tier` enum on every memory location:

```
tier ∈ { HBM_LOCAL, HBM_PEER_NVLINK, HBM_PEER_RDMA, DRAM_LOCAL, DRAM_REMOTE, SSD_LOCAL, SSD_REMOTE, OBJECT_STORE }
```

A transfer is "remote" iff `src.tier != HBM_LOCAL` or `dst.tier != HBM_LOCAL` from the perspective of the GPU executing the dependent attention kernel. The tier is recorded on every block and every transfer (§4.4–§4.5) so analysis can re-bucket as needed.

---

## 3. Conceptual model

### 3.1 Identifiers (cross-record join keys)

Every record carries a subset of these IDs. Agents MUST emit them when known and MUST use the same string format throughout the trace.

| ID | Type | Meaning |
|---|---|---|
| `trace_id` | string (UUID v4) | one trace = one launch of the framework |
| `node_id` | string | hostname or `${HOSTNAME}-${LOCAL_RANK}` |
| `worker_id` | string | `${node_id}/tp${TP}/pp${PP}` for vLLM; `${role}-${rank}` for SGLang PD |
| `request_id` | string | engine-assigned request id |
| `seq_id` | string | for multi-sequence requests (n>1, beam) |
| `step_id` | int64 | monotonic per-worker scheduler step |
| `token_idx` | int32 | absolute token position in the sequence (0-indexed) |
| `block_id` | int64 | physical block id within its pool |
| `block_hash` | hex string | content hash used by prefix caching |
| `layer_idx` | int16 | 0-indexed layer |
| `transfer_id` | string | UUID for one logical transfer (may span multiple RDMA WRs) |

### 3.2 Time

All timestamps are **integer nanoseconds since `CLOCK_MONOTONIC_RAW`**, captured via `clock_gettime` (Linux). On GPU events we additionally capture a `cuda_event_ts_ns` derived from `cudaEventElapsedTime` against a shared start event; agents MUST record both when both are available so post-hoc clock skew correction is possible. A single `t0_unix_ns` is recorded once at trace start to anchor monotonic time to wall-clock.

### 3.3 Event taxonomy (six levels)

| Level | Cardinality (per request, typical) | Examples |
|---|---|---|
| Request | 1 | arrival, schedule-admit, finish |
| Token | input_len + output_len | token-decode, first-token |
| KV-block | 10²–10³ | block-allocate, block-evict, block-tier-transition |
| Transfer | 10¹–10³ | start_load_kv, wait_for_layer_load, save_kv_layer, RDMA WR done |
| Metadata | 10²–10⁴ | prefix lookup, refcount inc/dec, hash insert, block-table update |
| System counter | sampled (1–100 Hz) | NIC bytes, NCCL p2p, GPU SM occupancy, HBM BW, page-fault rate |

The framework writes one JSON record per event. Schemas in §4.

---

## 4. Trace record schemas

### 4.1 Common envelope

Every record is a single JSON object with these required fields:

```json
{
  "ts_ns": 1700000000000,
  "type": "<one of: request|token|kv_block|transfer|metadata|sys_counter>",
  "subtype": "<event-specific, see below>",
  "trace_id": "...",
  "node_id": "...",
  "worker_id": "...",
  "v": 1
}
```

`v` is the schema version. Bump it in this document and in code together.

Records are emitted as **newline-delimited JSON** (`*.jsonl`). One file per `(worker_id, trace_id)` per rotation (§9). UTF-8.

### 4.2 Request records

```json
{
  "ts_ns": ..., "type": "request", "subtype": "arrival|admit|preempt|resume|finish|abort",
  "trace_id": "...", "node_id": "...", "worker_id": "...",
  "request_id": "...", "seq_id": "...",
  "input_len": 1024, "output_len_so_far": 0, "max_output_len": 512,
  "priority": 0,
  "arrival_ts_ns": ...,
  "first_schedule_ts_ns": ...,
  "first_token_ts_ns": ...,
  "finish_ts_ns": ...,
  "ttft_ns": null, "tpot_ns": null,
  "scheduler_state_snapshot": {
    "running": 14, "waiting": 8, "swapped": 0,
    "free_blocks": 1832, "used_blocks": 6168
  },
  "v": 1
}
```

Only the fields known at the time of the event need to be filled; the rest are `null`. The `finish` record SHOULD carry the full lifecycle (`arrival_ts_ns`, `first_token_ts_ns`, `finish_ts_ns`, computed `ttft_ns` and `tpot_ns`) so that question Q3 analyses can be done from request records alone if needed.

### 4.3 Token records

```json
{
  "ts_ns": ..., "type": "token", "subtype": "prefill_chunk|decode|first_token",
  "request_id": "...", "seq_id": "...",
  "token_idx": 1024, "step_id": 9421,
  "num_tokens": 1, "num_prefill_tokens": 0,
  "kernel_start_ts_ns": ..., "kernel_end_ts_ns": ...,
  "blocks_used_local": 24, "blocks_used_remote": 4,
  "v": 1
}
```

`num_tokens > 1` is allowed for chunked prefill or speculative-decode acceptance batches.

### 4.4 KV-block records

```json
{
  "ts_ns": ..., "type": "kv_block",
  "subtype": "allocate|free|evict|prefix_hit|tier_promote|tier_demote|hash_insert|hash_collide",
  "block_id": 78213, "block_hash": "0x...", "layer_idx": null,
  "block_size_tokens": 16, "block_size_bytes": 524288,
  "tier_before": "DRAM_LOCAL", "tier_after": "HBM_LOCAL",
  "owner_request_id": "...",
  "refcount_before": 0, "refcount_after": 1,
  "reason": "scheduler|prefix_match|connector_load|controller_promote|capacity_evict|finish_free",
  "age_ns": 1843000000,
  "reuse_count": 3, "last_reuse_ts_ns": ...,
  "v": 1
}
```

Notes:
- `layer_idx` is `null` when the block is whole-layer-stack (e.g., a DRAM L2 block in HiCache that holds all layers); set when the event is per-layer.
- `tier_before == tier_after` is allowed for in-tier events such as `prefix_hit` or `hash_insert`.
- `reuse_count` is the cumulative count over the block's life and updated on every prefix_hit/promote.

### 4.5 Transfer records

A *transfer* is the unit of remote KV movement that the system schedules as one operation. One `start` and one `end` record MUST be emitted per `transfer_id`.

```json
{
  "ts_ns": ..., "type": "transfer",
  "subtype": "start|end|cancel",
  "transfer_id": "uuid",
  "direction": "load|save",
  "src": {"tier": "DRAM_REMOTE", "node_id": "...", "device_id": null},
  "dst": {"tier": "HBM_LOCAL",   "node_id": "...", "device_id": 0},
  "transport": "nccl_p2p|nccl_send_recv|nixl|gdrcopy|ib_verbs|tcp|local_memcpy|file",
  "request_id": "...", "layer_idx": 12,
  "num_blocks": 64, "bytes": 33554432,
  "block_ids": [78213, 78214, ...],
  "queued_ts_ns": ..., "started_ts_ns": ..., "completed_ts_ns": ...,
  "queue_wait_ns": 1200000, "wire_time_ns": 4300000,
  "achieved_bw_gbps": 49.7,
  "wr_count": 8, "wr_completion_ts_ns": [...],
  "issued_by": "scheduler|connector|cache_controller|attention_layer",
  "issued_at_phase": "prefill|decode|prefetch|spillover",
  "earliest_known_ts_ns": ...,
  "v": 1
}
```

`earliest_known_ts_ns` is the timestamp at which the system first had enough information to know this transfer would be needed. It powers Q5. For prefix-cache-driven loads it is the time of the prefix match; for decode-driven pulls it is the scheduler decision time; for evictions it is the time the block became cold.

The `start` record carries `queued_ts_ns` and `started_ts_ns` (and `earliest_known_ts_ns`); the `end` record carries `completed_ts_ns`, the durations, achieved bandwidth, and per-WR completion times if the underlying API exposes them (NIXL does; NCCL does not).

### 4.6 Metadata records

```json
{
  "ts_ns": ..., "type": "metadata",
  "subtype": "prefix_lookup|prefix_insert|block_table_update|allocator_alloc|allocator_free|refcount_inc|refcount_dec|evict_select|hicache_promote|hicache_demote",
  "request_id": "...",
  "duration_ns": 12500,
  "n_keys": 24, "n_hits": 19, "hit_depth_tokens": 304,
  "tier_scope": "HBM_LOCAL|DRAM_LOCAL|...",
  "structure": "radix|hashmap|treelist",
  "size_before": 18234, "size_after": 18258,
  "v": 1
}
```

Used both for fine-grained metadata-op tracing (Q2) and for periodic snapshots of cache structure size.

### 4.7 System-counter records

Sampled, not event-driven. One record per (counter, sample).

```json
{
  "ts_ns": ..., "type": "sys_counter",
  "subtype": "nic_bytes|nic_packets|ib_pkey_violation|nccl_p2p_bytes|gpu_sm_active|gpu_hbm_bw_used|gpu_pcie_bw_used|cpu_pagefault|host_dram_used|nvlink_bytes|process_rss",
  "scope": "node|nic:mlx5_0|gpu:0|process:scheduler",
  "value": 1234567890, "unit": "bytes|count|percent|bytes_per_s",
  "interval_ns": 100000000,
  "v": 1
}
```

Counter sources (§7.3): `nvidia-smi dmon` / DCGM, `/sys/class/infiniband/*/counters/`, `/proc/net/dev`, NCCL `NCCL_DEBUG=INFO` parser, `nvidia-nsight` per-kernel exports (offline), and the GPU PMU via DCGM Field IDs.

### 4.8 JSON Schema files (machine-readable)

Each of §4.2–§4.7 MUST also be expressed as a JSON Schema file under `schemas/` in the repo (§13). Validators run in CI on a sample of records (`scripts/validate_traces.py --sample 1%`). Adding a new `subtype` requires updating the corresponding schema file *and* this document in the same PR.

---

## 5. Instrumentation hook map — vLLM

This section names, for each event type, the file and function in vLLM where the probe lives. Probes MUST be implemented as importable wrappers (decorators, monkey-patches at engine init, or a thin patch file that registers callbacks) and MUST be controllable by a single env var `BKVT_ENABLE=1`. When disabled, the probe MUST become a no-op with zero allocations on the hot path.

### 5.1 Request-level probes

| Subtype | File | Function | Notes |
|---|---|---|---|
| `arrival` | `vllm/v1/engine/processor.py` | `Processor.process_inputs` | record arrival_ts and input_len |
| `admit` | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | when a `Request` moves from waiting → running |
| `preempt` | same | `Scheduler._preempt` (or v1 equivalent) | record reason (capacity / priority) |
| `resume` | same | counterpart of preempt |
| `finish` | `vllm/v1/engine/output_processor.py` | output finalization | compute ttft/tpot from cached arrival_ts |
| `abort` | same | aborted path |

### 5.2 Token-level probes

| Subtype | File | Function | Notes |
|---|---|---|---|
| `prefill_chunk` / `decode` | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model` | wrap the call; capture cuda events around forward |
| `first_token` | `vllm/v1/engine/output_processor.py` | first-token detection | set `first_token_ts_ns` on the request |

### 5.3 KV-block probes

| Subtype | File | Function | Notes |
|---|---|---|---|
| `allocate` | `vllm/v1/core/block_pool.py` | `BlockPool.alloc_blocks` (or v1 equivalent) | one record per block |
| `free` | same | `free_blocks` | |
| `evict` | same | LRU eviction in `BlockPool` | record `reason="capacity_evict"` |
| `prefix_hit` | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.get_computed_blocks` | one per matched block, capture `hit_depth_tokens` |
| `hash_insert` | `vllm/v1/core/kv_cache_utils.py` | block-hash insertion sites | |
| `tier_promote` / `tier_demote` | connector-side, see §5.4 | | only relevant when a connector tiers blocks |

### 5.4 Transfer probes (the heart of Q1, Q3, Q5)

`KVConnectorBase_V1` is the canonical wrapping surface. Implementation strategy:

```
class TracingConnectorWrapper(KVConnectorBase_V1):
    def __init__(self, inner): self.inner = inner
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        with probe.metadata("prefix_lookup", request_id=request.request_id):
            return self.inner.get_num_new_matched_tokens(request, num_computed_tokens)
    def start_load_kv(self, ctx, **kw):
        tid = probe.transfer_start("load", ...)
        try: return self.inner.start_load_kv(ctx, **kw)
        finally: probe.transfer_end_async(tid)
    # ... same pattern for save_kv_layer / wait_for_save / wait_for_layer_load
```

Wrapper registration goes in a patch file `bkvt/integrations/vllm/patch.py` and is activated by setting `VLLM_KV_CONNECTOR_FACTORY=bkvt.integrations.vllm.factory.tracing_factory` (vLLM already reads a factory env var) or by monkey-patching `vllm.distributed.kv_transfer.kv_connector.factory.KVConnectorFactory.create_connector_v1` at engine init.

Per-method coverage:

| Method | Emits |
|---|---|
| `get_num_new_matched_tokens` | metadata `prefix_lookup` (Q2, Q5: sets `earliest_known_ts_ns` for the resulting load) |
| `update_state_after_alloc` | kv_block `tier_promote` records for blocks that become slated for remote-load |
| `build_connector_meta` | scheduler-step metadata snapshot |
| `start_load_kv` | transfer `start` (`direction="load"`) |
| `wait_for_layer_load(layer)` | transfer `end` for the load(s) covering that layer; emit *one* `wait_for_layer_load` metadata record per call |
| `save_kv_layer(layer, kv)` | transfer `start` (`direction="save"`) per layer, with `bytes` derived from the layer tensor stride |
| `wait_for_save` | transfer `end` records for all in-flight saves drained here |
| `get_finished` | reaper: emits `transfer.end` records for async transfers that completed since last call |

For the **NIXL** backend (`vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`) we additionally probe at the lower level so we can fill `wr_count` and `wr_completion_ts_ns`:

- Wrap the NIXL `nixl_agent` `make_xfer` / `transfer` calls (the public NIXL Python API) and the polling site (often called `check_xfer_status` or via descriptor handles). These give us per-request-list (per logical xfer) completion times.
- Where the connector batches into one descriptor list per layer, the per-WR breakdown is approximated by the descriptor count.

For **LMCache** (`lmcache_connector.py`), the inner `LMCacheEngine` invokes its own RPC; we wrap at the connector boundary because LMCache's internals are out of scope and may change. Bytes/timing observed at the connector are accurate enough.

### 5.5 Metadata probes

| Subtype | File | Function |
|---|---|---|
| `prefix_lookup` / `prefix_insert` | `vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks`, `cache_full_blocks` |
| `block_table_update` | same | `_update_block_table_for_request` (or v1 equivalent) |
| `allocator_alloc` / `allocator_free` | `vllm/v1/core/block_pool.py` | `alloc_blocks` / `free_blocks` |
| `refcount_inc` / `refcount_dec` | same | `_inc_ref` / `_dec_ref` |
| `evict_select` | same | LRU selection step |

### 5.6 Scheduler-decision probe (Q6)

In `vllm/v1/core/sched/scheduler.py::Scheduler.schedule`, after the scheduling round, emit a single `metadata` record of subtype `scheduler_decision` with the inputs the scheduler saw (queue depths, free blocks, KV pressure) and the outputs (admitted request ids, preempted ids, kv-load plans). One per step, regardless of how many requests it touched. This is the pivot record for question Q6.

---

## 6. Instrumentation hook map — SGLang

### 6.1 Request-level

| Subtype | File | Function |
|---|---|---|
| `arrival` | `python/sglang/srt/managers/scheduler.py` | `Scheduler.handle_generate_request` |
| `admit` | same | inside `Scheduler.run_batch` when a request enters the running batch |
| `preempt` / `resume` | same | preemption sites (different wording in SGLang; usually `retract_decode`) |
| `finish` / `abort` | same | finish path, `Scheduler.handle_finished_requests` |

### 6.2 Token-level

| Subtype | File | Function |
|---|---|---|
| `prefill_chunk` / `decode` | `python/sglang/srt/managers/tp_worker.py` | `TpModelWorker.forward_batch` (or current name) |
| `first_token` | `python/sglang/srt/managers/scheduler.py` | sender side after first decode |

### 6.3 KV-block + metadata

| Subtype | File | Function |
|---|---|---|
| `allocate` / `free` | `python/sglang/srt/mem_cache/allocator.py` | `BaseTokenToKVPoolAllocator.alloc` / `free` (and paged subclasses) |
| `prefix_hit` | `python/sglang/srt/mem_cache/radix_cache.py` | `RadixCache.match_prefix` |
| `prefix_insert` | same | `RadixCache.insert` |
| `evict` (HBM) | same | `RadixCache.evict` |
| `tier_promote` / `tier_demote` | `python/sglang/srt/mem_cache/hiradix_cache.py` + `python/sglang/srt/managers/cache_controller.py` | `HiCacheController.load` / `backup` |
| `block_table_update` | `python/sglang/srt/mem_cache/memory_pool.py` | `ReqToTokenPool.write` |
| `refcount_inc/dec` | `python/sglang/srt/mem_cache/radix_cache.py` | `inc_lock_ref` / `dec_lock_ref` |

### 6.4 Transfer probes — PD disaggregation

In `python/sglang/srt/disaggregation/`:

| Probe | File | Function |
|---|---|---|
| transfer `start` (prefill→decode) | `disaggregation/prefill.py` | the dispatcher that hands a finished prefill's KV to the chosen decode worker |
| transfer `end` | `disaggregation/decode.py` | reception side |
| Mooncake-specific | `disaggregation/mooncake/conn.py` | wrap `MooncakeConn.send` / `recv` (or current names) and capture per-RDMA-WR completions when the engine exposes them |
| NIXL-specific | `disaggregation/nixl/` | wrap the NIXL agent calls as in §5.4 |

### 6.5 Transfer probes — HiCache

| Probe | File | Function |
|---|---|---|
| L1↔L2 (HBM↔host DRAM) | `python/sglang/srt/managers/cache_controller.py` | `HiCacheController.load_to_device`, `HiCacheController.backup_to_host` |
| L2↔L3 (host↔Mooncake/3FS/NIXL/file) | same | the storage-backend dispatch sites |
| Backend-specific (Mooncake / 3FS / NIXL / file) | corresponding subdirs under `python/sglang/srt/mem_cache/` and `python/sglang/srt/disaggregation/` | wrap the backend's `put` / `get` / `transfer` |

The same wrapper-pattern from §5.4 applies. SGLang exposes its storage backends via abstract interfaces, so a single wrapper class covers all backends.

### 6.6 Scheduler-decision probe (Q6)

In `Scheduler.run_batch` (and the PD variants), emit a per-step `metadata` record `scheduler_decision` with the same payload as §5.6.

---

## 7. Transport- and system-level instrumentation

### 7.1 NCCL / NVLink

NCCL itself does not expose per-call metrics through a stable public API. Strategy:

1. Set `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=COLL,P2P` and parse the structured log with `bkvt/collectors/nccl_log.py`. Each `[Send]`/`[Recv]` line yields a transfer record with `transport="nccl_send_recv"` or `nccl_p2p`.
2. Optionally use **NCCL's profiler plugin** (the v2.19+ profiler interface) for CUDA-event-accurate timings. The plugin is loaded via `NCCL_PROFILER_PLUGIN`; we ship a minimal one in `bkvt/native/nccl_profiler/` that emits transfer records directly.
3. NVLink raw byte counters per link via `nvidia-smi nvlink --counters` (sampled at sys_counter rate).

### 7.2 RDMA (NIXL / IB Verbs / GDRCopy)

1. **NIXL Python API** — the canonical wrap site (§5.4 / §6.4). NIXL gives per-descriptor-list completion times.
2. **`/sys/class/infiniband/<dev>/ports/<port>/counters/`** — port_xmit_data, port_rcv_data, port_xmit_packets, port_rcv_packets, etc. Polled at sys_counter rate. These give an independent ground truth that the framework's transfer-bytes accounting can be checked against.
3. **GDRCopy / `cuMemcpy*`** — when used directly (not via NIXL), wrap the call sites.
4. Optionally **eBPF** probes on the NIC driver (`mlx5_ib_post_send`) for unmodified workload validation. Not required for v1, listed in §15.

### 7.3 GPU and host counters

Polled by `bkvt/collectors/sys_counters.py`:

- DCGM / NVML field IDs: SM_ACTIVE, DRAM_ACTIVE, PCIE_TX_BYTES, PCIE_RX_BYTES, NVLINK_TX_BYTES, NVLINK_RX_BYTES.
- `/proc/self/stat`, `/proc/<pid>/io` for the engine and worker processes.
- `/proc/net/dev` for ethernet fallback.
- Optional: `perf` for PEBS-based load-latency on the host DRAM tier.

Default poll period 100 ms (10 Hz). All sys_counter records share a `scope` field so analysis can join with transfer/request records by node/gpu.

### 7.4 Clock alignment

At trace start the framework emits one `metadata` record of subtype `clock_anchor` per worker, containing `(t0_unix_ns, t0_monotonic_ns, t0_cuda_event_ns)`. Across nodes we use **`chrony`** as the time source and capture its current offset in the same record. Post-hoc analysis applies the offset before computing cross-node latencies.

---

## 8. Trace pipeline

### 8.1 In-process emitter

`bkvt/emitter.py` exposes:

```python
probe.event(record_dict)            # generic
probe.transfer_start(...)           # returns transfer_id, defers the start record to a ring buffer
probe.transfer_end(transfer_id, ...)
probe.metadata(subtype, **kw)       # contextmanager that times its body
```

Implementation:
- Per-thread ring buffer (lock-free, fixed capacity) of pre-allocated `dict`-shaped slots → flusher thread serializes to `orjson` → writes to a per-worker `jsonl` file.
- Hot path emits **never allocate Python objects** when buffer has room; on overflow we increment `dropped_records_total` (a sys_counter) and continue.
- The flusher batches writes (default 4 MiB or 100 ms, whichever first). On Linux the file is opened with `O_APPEND`; multiple processes per node may share a file only if they each write atomic <4 KiB lines (PIPE_BUF guarantee), but the default is one file per worker.

### 8.2 File layout

```
${BKVT_OUTPUT_DIR}/
  ${trace_id}/
    manifest.json                 # config snapshot, env, versions, t0
    ${node_id}/
      ${worker_id}.0001.jsonl
      ${worker_id}.0002.jsonl     # rotated at BKVT_ROTATE_BYTES (default 256 MiB)
      sys_counters.jsonl
      nccl_log.jsonl              # parsed NCCL log records
```

### 8.3 Compression and rotation

Files rotate by size (default 256 MiB). After rotation the previous file is gzipped in place by a background thread. Live readers (e.g., a tailing dashboard) MUST handle both `.jsonl` and `.jsonl.gz`.

### 8.4 Aggregation

A separate offline tool `bkvt/analysis/build_index.py` produces a Parquet index from the jsonl files (using DuckDB or `pyarrow`) for fast analysis. The Parquet schemas mirror §4 1:1.

---

## 9. Overhead budget

The framework MUST stay within these budgets, measured against a connector-disabled vLLM/SGLang baseline on the same hardware:

| Metric | Budget |
|---|---|
| TTFT regression | ≤ 2 % |
| TPOT regression | ≤ 1 % |
| End-to-end throughput regression | ≤ 3 % |
| Per-record CPU cost (hot path) | ≤ 1.5 µs |
| Bytes per request (full trace) | ≤ 200 KiB at default sampling |

Strategies:
1. **No-op when disabled.** A single `if not _ENABLED: return` at every probe entry. `_ENABLED` is read once at engine init and frozen.
2. **Sampling.** `BKVT_SAMPLE_TRANSFER`, `BKVT_SAMPLE_METADATA`, `BKVT_SAMPLE_TOKEN` ∈ [0,1]. Default: 1.0 for transfer, 1.0 for kv_block and metadata in scheduler, 0.05 for per-token records (we keep first-token always). Request-level always 1.0.
3. **Coarse mode.** `BKVT_PROFILE=coarse` disables per-WR records and per-prefix-lookup metadata; keeps request, scheduler-decision, and per-transfer summaries.
4. **Async serialization.** No JSON encoding on the hot path. The flusher thread is pinned to a non-engine CPU.
5. **Batched cuda events.** A pool of pre-allocated `cudaEvent_t`s; events are stamped on the hot path and the elapsed time is computed off-path.

Overhead validation is part of CI (§14, M3): a smoke benchmark on ShareGPT v3 with BKVT off vs. on must stay within budget.

---

## 10. Sampling and reproducibility

- Every record carries a `sample_decision` field when emitted by a sampled probe (omitted for always-on probes), so analyses can correctly normalize.
- The framework MUST also emit a `manifest.json` at trace start including: BKVT version (git SHA), vLLM/SGLang versions, model name, GPU model, NCCL version, NIXL version (if loaded), kernel version, sampling configuration, and `BKVT_*` env var values. Without manifest, traces are not reproducible and analysis tools MUST refuse to run.
- A trace is considered well-formed if every `transfer.start` has a matching `transfer.end` or `transfer.cancel`, and every `request.arrival` has a matching `request.finish` or `abort`. The validator (`scripts/validate_traces.py`) checks this and reports orphans.

---

## 11. Question → probe coverage matrix

| Question | Primary records | Secondary / cross-ref |
|---|---|---|
| **Q1. Remote KV data path** | `transfer.{start,end}` (§4.5), `kv_block.tier_*` (§4.4) | `sys_counter` NIC/NVLink bytes (§4.7) for ground-truth check |
| **Q2. Remote metadata path** | `metadata.*` (§4.6) | `kv_block.{allocate,free,evict,hash_insert}` (§4.4) |
| **Q3. Critical path** | `request.*` with full ts fields (§4.2), `token.*` with kernel ts (§4.3), `transfer.*` with `queued_ts_ns/started_ts_ns/completed_ts_ns` | `metadata.scheduler_decision` for queue waits |
| **Q4. Reuse / locality** | `kv_block.{allocate,free,prefix_hit,tier_*}` with `age_ns`, `reuse_count` | `transfer` records to attribute reuse to load events |
| **Q5. Prefetchability** | `transfer.*` with `earliest_known_ts_ns` and `started_ts_ns` | `metadata.prefix_lookup` to pin earliest-known time |
| **Q6. Scheduling impact** | `metadata.scheduler_decision` (§5.6, §6.6), `request.{admit,preempt,resume}` | aggregated `transfer` and `kv_block` over scheduler-decision windows |

A reference notebook `analysis/notebooks/Q1_to_Q6.ipynb` MUST exist that loads a sample trace and produces one canonical figure per question. The notebook is the acceptance test for "the framework can answer the question."

Suggested figures (one per question):

- Q1: stacked-area of remote KV bytes over time, faceted by transport.
- Q2: per-metadata-subtype CDF of duration, plus rate (ops/s) over time.
- Q3: per-request critical-path waterfall, plus attribution histogram across the workload.
- Q4: reuse-distance CDF and per-tier residency CDF.
- Q5: histogram of `(started_ts_ns − earliest_known_ts_ns)` aka prefetch slack.
- Q6: scatter of scheduler-decision queue depth vs. resulting tail latency, colored by placement.

---

## 12. Configuration

Single env-var prefix `BKVT_*`. All flags read once at engine init; runtime hot-reload is out of scope.

| Env var | Default | Meaning |
|---|---|---|
| `BKVT_ENABLE` | `0` | master switch |
| `BKVT_OUTPUT_DIR` | `./bkvt_traces` | trace root |
| `BKVT_TRACE_ID` | auto-uuid | override for repeatable runs |
| `BKVT_PROFILE` | `full` | `full` \| `coarse` \| `request_only` |
| `BKVT_SAMPLE_TOKEN` | `0.05` | probability for token records |
| `BKVT_SAMPLE_METADATA` | `1.0` | for fine metadata records |
| `BKVT_SAMPLE_TRANSFER` | `1.0` | for transfer records |
| `BKVT_ROTATE_BYTES` | `268435456` | jsonl rotation threshold |
| `BKVT_FLUSH_BYTES` | `4194304` | flusher batch size |
| `BKVT_SYS_COUNTER_HZ` | `10` | sys_counter poll rate |
| `BKVT_NCCL_PROFILER` | `0` | load native NCCL profiler plugin |
| `BKVT_CLOCK_ANCHOR_HZ` | `1` | how often to re-emit clock_anchor |

A YAML config at `${BKVT_OUTPUT_DIR}/config.yaml` (if present) takes precedence over env vars.

---

## 13. Repository layout

```
BeyondKVTransfer/
├── DESIGN.md                      # this document
├── README.md
├── pyproject.toml
├── bkvt/                          # importable package
│   ├── __init__.py
│   ├── emitter.py                 # §8.1
│   ├── records.py                 # dataclasses for §4 records
│   ├── ids.py                     # ID generation (§3.1)
│   ├── clock.py                   # CLOCK_MONOTONIC_RAW + cuda_event helpers (§3.2)
│   ├── sampling.py
│   ├── config.py                  # env/YAML loader
│   ├── manifest.py
│   ├── integrations/
│   │   ├── vllm/
│   │   │   ├── __init__.py
│   │   │   ├── patch.py           # monkey-patch entrypoint
│   │   │   ├── factory.py         # tracing connector factory (§5.4)
│   │   │   ├── connector_wrapper.py
│   │   │   ├── scheduler_probe.py # §5.1, §5.6
│   │   │   ├── block_pool_probe.py# §5.3, §5.5
│   │   │   └── runner_probe.py    # §5.2
│   │   └── sglang/
│   │       ├── __init__.py
│   │       ├── patch.py
│   │       ├── scheduler_probe.py
│   │       ├── radix_probe.py
│   │       ├── hicache_probe.py
│   │       └── disagg_probe.py
│   ├── collectors/
│   │   ├── sys_counters.py        # DCGM / sysfs / proc
│   │   ├── nccl_log.py            # NCCL log parser
│   │   ├── ib_counters.py         # /sys/class/infiniband
│   │   └── nvml.py
│   └── native/
│       └── nccl_profiler/         # optional NCCL profiler plugin
├── schemas/                       # JSON Schemas for §4
│   ├── envelope.schema.json
│   ├── request.schema.json
│   ├── token.schema.json
│   ├── kv_block.schema.json
│   ├── transfer.schema.json
│   ├── metadata.schema.json
│   └── sys_counter.schema.json
├── scripts/
│   ├── validate_traces.py
│   ├── run_vllm_with_bkvt.sh
│   ├── run_sglang_with_bkvt.sh
│   └── replay_smoketest.py
├── analysis/
│   ├── build_index.py             # jsonl → parquet
│   ├── notebooks/
│   │   └── Q1_to_Q6.ipynb
│   └── lib/
│       ├── load.py
│       ├── critical_path.py
│       ├── reuse.py
│       └── prefetch.py
├── tests/
│   ├── unit/
│   │   ├── test_emitter.py
│   │   ├── test_records.py
│   │   └── test_sampling.py
│   ├── integration/
│   │   ├── test_vllm_smoke.py
│   │   └── test_sglang_smoke.py
│   └── overhead/
│       └── test_overhead_budget.py
└── docs/
    ├── how_to_run.md
    ├── how_to_add_a_probe.md
    └── how_to_add_a_backend.md
```

---

## 14. Milestones

Each milestone is a coherent unit of work that a downstream agent can take on independently. Milestones list their prerequisites.

**M1. Core emitter + schema (no integration).**
Deliver `bkvt/{emitter,records,ids,clock,sampling,config,manifest}.py` and `schemas/*.schema.json`. Write `tests/unit/*` and `scripts/validate_traces.py`. Synthetic traces validate.
Acceptance: unit tests pass; `validate_traces.py` accepts a hand-written sample and rejects malformed records.

**M2. vLLM integration — request/token/scheduler/block.** *(prereq: M1)*
Implement `bkvt/integrations/vllm/{patch,scheduler_probe,block_pool_probe,runner_probe}.py`. No connector hooks yet.
Acceptance: a vLLM run with a simple ShareGPT replay produces well-formed traces; per-request critical path can be reconstructed (Q3 figure renders).

**M3. vLLM integration — connectors.** *(prereq: M2)*
Implement the tracing connector wrapper for `KVConnectorBase_V1`, plus NIXL- and LMCache-specific probes.
Acceptance: a 2-worker disaggregated run (prefill+decode) produces matching transfer.start / transfer.end pairs; bytes accounting agrees with `/sys/class/infiniband/*/counters` to within 5 %; overhead budget (§9) holds.

**M4. SGLang integration — scheduler/radix/allocator.** *(prereq: M1, can run in parallel with M2)*
Implement `bkvt/integrations/sglang/{patch,scheduler_probe,radix_probe}.py`.
Acceptance: same as M2 for SGLang; Q4 figure renders.

**M5. SGLang integration — HiCache + disagg.** *(prereq: M4)*
Implement `hicache_probe.py` and `disagg_probe.py`. Cover Mooncake and NIXL backends.
Acceptance: PD-disagg smoke test produces matched transfer pairs; HiCache L1↔L2↔L3 transitions visible in `kv_block.tier_*` records.

**M6. System counters + clock alignment.** *(prereq: M1)*
Implement `bkvt/collectors/*.py`. Wire to the emitter and to manifest.
Acceptance: NIC/NVLink/SM counters appear at the configured rate; clock_anchor records emitted; cross-node skew documented.

**M7. Analysis pipeline + notebook.** *(prereq: M3 or M5)*
Implement `analysis/build_index.py` and `analysis/notebooks/Q1_to_Q6.ipynb`.
Acceptance: the notebook produces all six figures from a recorded trace.

**M8. Overhead validation + docs.** *(prereq: M3, M5)*
Implement `tests/overhead/test_overhead_budget.py`. Write `docs/how_to_*.md`.
Acceptance: overhead test passes on the lab rig; how-to docs let a fresh engineer reproduce a trace from scratch.

---

## 15. Open questions and risks

1. **API drift.** vLLM v1 and SGLang are evolving fast. The probe layer MUST log a one-time WARN at init if the wrapped function's signature differs from the version pinned in `pyproject.toml`. CI pins the upstream commits.
2. **NCCL opacity.** Until the v2.19+ profiler plugin is widely available, NCCL transfer timings are coarser than RDMA timings. Document this asymmetry in figures.
3. **Per-WR RDMA timing.** NIXL exposes per-descriptor-list timing but not per-WR. eBPF on `mlx5_ib_post_send` is the fallback if Q5 / Q3 needs finer detail; not in v1 scope.
4. **Speculative decoding & beam search.** `seq_id` handling is non-trivial; first implementation pass MUST clearly mark unsupported scenarios in `manifest.json` rather than silently dropping their records.
5. **Tiering semantics differ across engines.** `tier_promote` in vLLM's connector world means "the connector loaded this block from remote into HBM"; in SGLang's HiCache world it can mean L3→L2 or L2→L1. The `tier_before/tier_after` fields disambiguate, so analyses MUST always read both — never rely on `subtype` alone.
6. **Privacy of payloads.** Records carry `block_hash` (a content hash) but never raw token ids or KV tensor contents. Reaffirmed here so future probe additions cannot regress.
7. **Storage size at scale.** A 24-hour cluster trace can produce TBs of jsonl. Coarse profile (`BKVT_PROFILE=coarse`) is the default for long-running captures; full mode is for targeted experiments.

---

## 16. Glossary

- **Block / KV block** — the paged unit of KV cache. vLLM's default is 16 tokens; SGLang's varies.
- **Tier** — physical location of a block; one of the enum values in §2.3.
- **Connector** — vLLM's pluggable abstraction for moving KV between the engine and an external store (NIXL, LMCache, shared storage, etc.).
- **HiCache / HiRadixCache** — SGLang's hierarchical KV cache spanning HBM (L1), host DRAM (L2), and external storage (L3).
- **NIXL** — NVIDIA Inference XLink, an RDMA library used by both vLLM and SGLang for cross-node KV transport.
- **PD-disagg** — prefill–decode disaggregation; separate workers run prefill and decode and exchange KV.
- **Prefetch slack** — `started_ts_ns − earliest_known_ts_ns`, the question-Q5 metric.
- **TTFT / TPOT** — time-to-first-token and time-per-output-token; standard LLM-serving latency metrics.
- **Critical path** — the per-request sequence of stages whose durations sum to end-to-end latency.

---

## 17. References

- vLLM v1 KV cache manager — `vllm/v1/core/kv_cache_manager.py` (`docs.vllm.ai/en/latest/api/vllm/v1/core/kv_cache_manager/`).
- vLLM `KVConnectorBase_V1` — `vllm/distributed/kv_transfer/kv_connector/v1/base.py`.
- vLLM NIXL connector — `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`; usage: `docs.vllm.ai/en/stable/features/nixl_connector_usage/`.
- vLLM LMCache connector — `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`.
- SGLang PD disaggregation — `python/sglang/srt/disaggregation/`; docs: `docs.sglang.ai/advanced_features/pd_disaggregation.html`.
- SGLang HiCache — `python/sglang/srt/mem_cache/hiradix_cache.py`, `python/sglang/srt/managers/cache_controller.py`; design: `docs.sglang.io/advanced_features/hicache_design.html`.
- SGLang RadixCache & memory pools — `python/sglang/srt/mem_cache/radix_cache.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py`.
- DCGM field IDs — `developer.nvidia.com/dcgm`.
- NIXL — `github.com/ai-dynamo/nixl`.
