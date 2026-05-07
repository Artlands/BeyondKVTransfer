"""
ID generation for BeyondKVTransfer trace records.

All IDs described in §3.1 of DESIGN.md are produced here so that every
module uses a single, consistent format.

Format rules (§3.1):
  trace_id    — UUID v4 string, e.g. "3f2a1b4c-..."
  node_id     — hostname, or "${HOSTNAME}-${LOCAL_RANK}" when LOCAL_RANK is set
  worker_id   — "${node_id}/tp${TP}/pp${PP}" for vLLM,
                "${role}-${rank}" for SGLang PD disaggregation
  transfer_id — UUID v4 string (one per logical transfer)
  seq_id      — caller-supplied; no generation helper needed
"""

from __future__ import annotations

import os
import socket
import uuid


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def new_uuid() -> str:
    """Return a fresh UUID v4 as a lower-case hyphenated string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Trace-level ID (one per framework launch)
# ---------------------------------------------------------------------------

def new_trace_id() -> str:
    """Generate a unique trace_id for one launch of the framework."""
    return new_uuid()


# ---------------------------------------------------------------------------
# Node / worker IDs
# ---------------------------------------------------------------------------

def get_node_id() -> str:
    """Return the node_id for this process.

    Uses the ``BKVT_NODE_ID`` env var if set; otherwise builds
    ``${HOSTNAME}-${LOCAL_RANK}`` when ``LOCAL_RANK`` is set, or plain
    ``${HOSTNAME}`` otherwise.
    """
    override = os.environ.get("BKVT_NODE_ID")
    if override:
        return override

    hostname = socket.gethostname()
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return f"{hostname}-{local_rank}"
    return hostname


def make_vllm_worker_id(
    node_id: str,
    tp_rank: int,
    pp_rank: int,
) -> str:
    """``${node_id}/tp${TP}/pp${PP}`` — vLLM tensor/pipeline parallel worker."""
    return f"{node_id}/tp{tp_rank}/pp{pp_rank}"


def make_sglang_worker_id(role: str, rank: int) -> str:
    """``${role}-${rank}`` — SGLang PD disaggregation worker."""
    return f"{role}-{rank}"


def get_worker_id(
    node_id: str | None = None,
    *,
    tp_rank: int | None = None,
    pp_rank: int | None = None,
    role: str | None = None,
    rank: int | None = None,
) -> str:
    """Convenience factory that picks the right format based on which
    keyword args are supplied.

    For vLLM: pass ``tp_rank`` and ``pp_rank`` (both default to 0).
    For SGLang PD: pass ``role`` and ``rank``.
    Falls back to plain ``node_id`` when neither set is provided.
    """
    if node_id is None:
        node_id = get_node_id()

    if role is not None and rank is not None:
        return make_sglang_worker_id(role, rank)

    if tp_rank is not None or pp_rank is not None:
        return make_vllm_worker_id(
            node_id,
            tp_rank if tp_rank is not None else 0,
            pp_rank if pp_rank is not None else 0,
        )

    return node_id


# ---------------------------------------------------------------------------
# Transfer ID (one per logical transfer operation)
# ---------------------------------------------------------------------------

def new_transfer_id() -> str:
    """Generate a unique transfer_id for one logical transfer operation."""
    return new_uuid()
