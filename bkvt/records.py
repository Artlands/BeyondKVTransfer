"""
Dataclasses for all BeyondKVTransfer trace record types (§4).

Each class mirrors one JSON record schema from §4.2–§4.7.
``to_dict()`` returns a plain ``dict`` with ``None`` values omitted,
ready for ``orjson.dumps``.

Common envelope fields (§4.1) are embedded directly rather than nested
so that every record is flat and directly queryable by DuckDB/Parquet.

Schema version
--------------
``SCHEMA_VERSION = 1`` — bump this and the JSON Schema files together
whenever a breaking change is made.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

SCHEMA_VERSION: int = 1

# ---------------------------------------------------------------------------
# Tier enum (§2.3)
# ---------------------------------------------------------------------------

class Tier:
    HBM_LOCAL         = "HBM_LOCAL"
    HBM_PEER_NVLINK   = "HBM_PEER_NVLINK"
    HBM_PEER_RDMA     = "HBM_PEER_RDMA"
    DRAM_LOCAL        = "DRAM_LOCAL"
    DRAM_REMOTE       = "DRAM_REMOTE"
    SSD_LOCAL         = "SSD_LOCAL"
    SSD_REMOTE        = "SSD_REMOTE"
    OBJECT_STORE      = "OBJECT_STORE"

    ALL: frozenset[str] = frozenset({
        "HBM_LOCAL", "HBM_PEER_NVLINK", "HBM_PEER_RDMA",
        "DRAM_LOCAL", "DRAM_REMOTE",
        "SSD_LOCAL", "SSD_REMOTE",
        "OBJECT_STORE",
    })


# ---------------------------------------------------------------------------
# Helper: dict with None-valued keys stripped
# ---------------------------------------------------------------------------

def _strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# §4.2 Request record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RequestRecord:
    """One record per request lifecycle event (arrival, admit, finish, …)."""

    # Envelope (§4.1)
    ts_ns: int
    trace_id: str
    node_id: str
    worker_id: str

    # Request identity
    request_id: str
    subtype: str   # arrival|admit|preempt|resume|finish|abort

    # Optional fields — filled in as they become known
    seq_id: Optional[str] = None
    input_len: Optional[int] = None
    output_len_so_far: Optional[int] = None
    max_output_len: Optional[int] = None
    priority: Optional[int] = None

    # Lifecycle timestamps (§4.2 — filled on finish)
    arrival_ts_ns: Optional[int] = None
    first_schedule_ts_ns: Optional[int] = None
    first_token_ts_ns: Optional[int] = None
    finish_ts_ns: Optional[int] = None
    ttft_ns: Optional[int] = None
    tpot_ns: Optional[int] = None

    # Scheduler state snapshot
    scheduler_state_snapshot: Optional[dict] = None

    v: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["type"] = "request"
        return _strip_none(d)


# ---------------------------------------------------------------------------
# §4.3 Token record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TokenRecord:
    """One record per token (or token batch) produced or processed."""

    # Envelope
    ts_ns: int
    trace_id: str
    node_id: str
    worker_id: str

    subtype: str         # prefill_chunk|decode|first_token
    request_id: str
    token_idx: int
    step_id: int

    seq_id: Optional[str] = None
    num_tokens: int = 1
    num_prefill_tokens: int = 0

    kernel_start_ts_ns: Optional[int] = None
    kernel_end_ts_ns: Optional[int] = None
    cuda_event_ts_ns: Optional[int] = None

    blocks_used_local: Optional[int] = None
    blocks_used_remote: Optional[int] = None

    # Set when emitted by a sampled probe
    sample_decision: Optional[float] = None

    v: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["type"] = "token"
        return _strip_none(d)


# ---------------------------------------------------------------------------
# §4.4 KV-block record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class KVBlockRecord:
    """One record per KV-block lifecycle event (allocate, evict, tier move, …)."""

    # Envelope
    ts_ns: int
    trace_id: str
    node_id: str
    worker_id: str

    subtype: str   # allocate|free|evict|prefix_hit|tier_promote|tier_demote|hash_insert|hash_collide
    block_id: int

    block_hash: Optional[str] = None
    layer_idx: Optional[int] = None
    block_size_tokens: Optional[int] = None
    block_size_bytes: Optional[int] = None

    tier_before: Optional[str] = None
    tier_after: Optional[str] = None

    owner_request_id: Optional[str] = None
    refcount_before: Optional[int] = None
    refcount_after: Optional[int] = None

    reason: Optional[str] = None    # scheduler|prefix_match|connector_load|…
    age_ns: Optional[int] = None
    reuse_count: Optional[int] = None
    last_reuse_ts_ns: Optional[int] = None

    sample_decision: Optional[float] = None

    v: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["type"] = "kv_block"
        return _strip_none(d)


# ---------------------------------------------------------------------------
# §4.5 Transfer record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TransferEndpoint:
    """Source or destination of a transfer (embedded in TransferRecord)."""
    tier: str
    node_id: Optional[str] = None
    device_id: Optional[int] = None

    def to_dict(self) -> dict:
        return _strip_none(dataclasses.asdict(self))


@dataclasses.dataclass
class TransferRecord:
    """One record per transfer start, end, or cancel."""

    # Envelope
    ts_ns: int
    trace_id: str
    node_id: str
    worker_id: str

    subtype: str        # start|end|cancel
    transfer_id: str
    direction: str      # load|save

    src: Optional[TransferEndpoint] = None
    dst: Optional[TransferEndpoint] = None

    transport: Optional[str] = None  # nccl_p2p|nccl_send_recv|nixl|gdrcopy|…
    request_id: Optional[str] = None
    layer_idx: Optional[int] = None

    num_blocks: Optional[int] = None
    bytes: Optional[int] = None
    block_ids: Optional[list[int]] = None

    queued_ts_ns: Optional[int] = None
    started_ts_ns: Optional[int] = None
    completed_ts_ns: Optional[int] = None
    queue_wait_ns: Optional[int] = None
    wire_time_ns: Optional[int] = None
    achieved_bw_gbps: Optional[float] = None

    wr_count: Optional[int] = None
    wr_completion_ts_ns: Optional[list[int]] = None

    issued_by: Optional[str] = None          # scheduler|connector|cache_controller|…
    issued_at_phase: Optional[str] = None    # prefill|decode|prefetch|spillover

    # Q5 — earliest time the system could have known this transfer was needed
    earliest_known_ts_ns: Optional[int] = None

    sample_decision: Optional[float] = None

    v: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["type"] = "transfer"
        # Serialise nested endpoint dicts
        if self.src is not None:
            d["src"] = self.src.to_dict()
        if self.dst is not None:
            d["dst"] = self.dst.to_dict()
        return _strip_none(d)


# ---------------------------------------------------------------------------
# §4.6 Metadata record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MetadataRecord:
    """One record per metadata operation (prefix lookup, block-table update, …)."""

    # Envelope
    ts_ns: int
    trace_id: str
    node_id: str
    worker_id: str

    subtype: str   # prefix_lookup|prefix_insert|block_table_update|connector_*|…

    request_id: Optional[str] = None
    duration_ns: Optional[int] = None

    n_keys: Optional[int] = None
    n_hits: Optional[int] = None
    hit_depth_tokens: Optional[int] = None

    tier_scope: Optional[str] = None
    structure: Optional[str] = None    # radix|hashmap|treelist

    size_before: Optional[int] = None
    size_after: Optional[int] = None

    # For scheduler_decision subtype
    scheduler_inputs: Optional[dict] = None
    scheduler_outputs: Optional[dict] = None

    # For clock_anchor subtype
    t0_unix_ns: Optional[int] = None
    t0_monotonic_ns: Optional[int] = None
    t0_cuda_event_ns: Optional[int] = None
    chrony_offset_ns: Optional[int] = None

    sample_decision: Optional[float] = None

    v: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["type"] = "metadata"
        return _strip_none(d)


# ---------------------------------------------------------------------------
# §4.7 System-counter record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SysCounterRecord:
    """One sample of a system-level counter (NIC, GPU, CPU, …)."""

    # Envelope
    ts_ns: int
    trace_id: str
    node_id: str
    worker_id: str

    subtype: str   # nic_bytes|nic_packets|gpu_sm_active|…
    scope: str     # node|nic:mlx5_0|gpu:0|process:scheduler

    value: int | float = 0
    unit: str = "count"         # bytes|count|percent|bytes_per_s
    interval_ns: Optional[int] = None

    v: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["type"] = "sys_counter"
        return _strip_none(d)


# ---------------------------------------------------------------------------
# Union type alias for type hints
# ---------------------------------------------------------------------------

AnyRecord = (
    RequestRecord
    | TokenRecord
    | KVBlockRecord
    | TransferRecord
    | MetadataRecord
    | SysCounterRecord
)

RECORD_TYPES: tuple[type, ...] = (
    RequestRecord,
    TokenRecord,
    KVBlockRecord,
    TransferRecord,
    MetadataRecord,
    SysCounterRecord,
)
