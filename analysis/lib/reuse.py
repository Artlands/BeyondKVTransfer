"""Reuse and tier-residency metrics for Q4."""

from __future__ import annotations

import pandas as pd


def reuse_distance(kv_blocks: pd.DataFrame) -> pd.DataFrame:
    """Estimate reuse distance between prefix hits for each block."""

    if kv_blocks.empty or "block_id" not in kv_blocks:
        return pd.DataFrame()
    hits = kv_blocks[kv_blocks["subtype"] == "prefix_hit"].copy()
    if hits.empty:
        return pd.DataFrame()
    hits = hits.sort_values(["block_id", "ts_ns"])
    hits["prev_reuse_ts_ns"] = hits.groupby("block_id")["ts_ns"].shift(1)
    hits["reuse_distance_ns"] = hits["ts_ns"] - hits["prev_reuse_ts_ns"]
    return hits.dropna(subset=["reuse_distance_ns"])


def tier_residency(kv_blocks: pd.DataFrame) -> pd.DataFrame:
    """Estimate per-block residency intervals by tier."""

    if kv_blocks.empty or "block_id" not in kv_blocks or "tier_after" not in kv_blocks:
        return pd.DataFrame()

    events = kv_blocks.sort_values(["block_id", "ts_ns"]).copy()
    events["next_ts_ns"] = events.groupby("block_id")["ts_ns"].shift(-1)
    events["residency_ns"] = events["next_ts_ns"] - events["ts_ns"]
    events = events.dropna(subset=["tier_after", "residency_ns"])
    return events[events["residency_ns"] >= 0][
        ["trace_id", "worker_id", "block_id", "tier_after", "ts_ns", "next_ts_ns", "residency_ns"]
    ]

