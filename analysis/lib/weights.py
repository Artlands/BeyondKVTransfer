"""Q7 reductions for remote weight and adapter traffic."""

from __future__ import annotations

import pandas as pd


_FAMILY_BY_SUBTYPE = {
    "load": "startup",
    "tier_promote": "offload",
    "tier_demote": "offload",
    "expert_dispatch": "ep",
    "expert_release": "ep",
    "lora_activate": "lora",
    "lora_deactivate": "lora",
    "update_apply": "rlhf",
}


def classify_weight_family(subtype: object) -> str:
    """Map a weight_block subtype to the Q7 family name."""

    return _FAMILY_BY_SUBTYPE.get(str(subtype), "other")


def per_family_byte_rollup(weight_blocks: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """Return bytes over time by Q7 family and transport.

    Weight-block records provide the family labels; transfer records provide
    transport. The function is intentionally tolerant of sparse traces and
    works with either source.
    """

    rows: list[pd.DataFrame] = []
    if not weight_blocks.empty and {"subtype", "ts_ns"}.issubset(weight_blocks.columns):
        wb = weight_blocks.copy()
        wb["family"] = wb["subtype"].map(classify_weight_family)
        if "bytes" not in wb:
            wb["bytes"] = 0
        wb["transport"] = "recorded_weight_block"
        rows.append(wb[["ts_ns", "family", "transport", "bytes"]])

    if not transfers.empty and "payload_kind" in transfers and "subtype" in transfers:
        tr = transfers[(transfers["payload_kind"] == "weight") & (transfers["subtype"] == "start")].copy()
        if not tr.empty:
            if "bytes" not in tr:
                tr["bytes"] = 0
            tr["family"] = tr.apply(_family_for_transfer, axis=1)
            if "transport" not in tr:
                tr["transport"] = None
            rows.append(tr[["ts_ns", "family", "transport", "bytes"]])

    if not rows:
        return pd.DataFrame(columns=["ts_ns", "family", "transport", "bytes"])
    data = pd.concat(rows, ignore_index=True)
    return (
        data.groupby(["ts_ns", "family", "transport"], dropna=False)["bytes"]
        .sum()
        .reset_index()
        .sort_values("ts_ns")
    )


def lora_swap_latency(weight_blocks: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """Return LoRA activation/deactivation rows with measured latency when known."""

    if weight_blocks.empty or "subtype" not in weight_blocks:
        return pd.DataFrame()
    lora = weight_blocks[weight_blocks["subtype"].isin(["lora_activate", "lora_deactivate"])].copy()
    if lora.empty:
        return lora

    if not transfers.empty and {"payload_kind", "lora_adapter_id", "subtype"}.issubset(transfers.columns):
        starts = transfers[
            (transfers["payload_kind"] == "weight")
            & (transfers["subtype"] == "start")
            & transfers["lora_adapter_id"].notna()
        ].copy()
        ends = transfers[transfers["subtype"].isin(["end", "cancel"])].copy()
        if not starts.empty:
            ends_for_merge = ends[["transfer_id", "completed_ts_ns"]].rename(
                columns={"completed_ts_ns": "completed_ts_ns_end"}
            )
            paired = starts.merge(
                ends_for_merge,
                on="transfer_id",
                how="left",
            )
            paired["lora_latency_ns"] = paired["completed_ts_ns_end"] - paired["started_ts_ns"]
            latency = paired.groupby("lora_adapter_id", as_index=False)["lora_latency_ns"].max()
            lora = lora.merge(latency, on="lora_adapter_id", how="left")
    return lora.sort_values("ts_ns")


def weight_update_windows(weight_blocks: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """Return one row per weight_version with update byte and time bounds."""

    if weight_blocks.empty or "subtype" not in weight_blocks:
        return pd.DataFrame()
    updates = weight_blocks[weight_blocks["subtype"] == "update_apply"].copy()
    if updates.empty or "weight_version" not in updates:
        return pd.DataFrame()
    if "bytes" not in updates:
        updates["bytes"] = 0

    grouped = updates.groupby("weight_version", dropna=False).agg(
        start_ts_ns=("ts_ns", "min"),
        end_ts_ns=("ts_ns", "max"),
        params=("param_name", "nunique"),
        weight_block_bytes=("bytes", "sum"),
    )

    if not transfers.empty and {"payload_kind", "weight_version"}.issubset(transfers.columns):
        tr = transfers[(transfers["payload_kind"] == "weight") & (transfers["subtype"] == "start")].copy()
        if not tr.empty:
            if "bytes" not in tr:
                tr["bytes"] = 0
            transfer_bytes = tr.groupby("weight_version", dropna=False)["bytes"].sum()
            grouped = grouped.join(transfer_bytes.rename("transfer_bytes"), how="left")

    return grouped.reset_index()


def _family_for_transfer(row: pd.Series) -> str:
    if pd.notna(row.get("lora_adapter_id")):
        return "lora"
    if pd.notna(row.get("weight_version")):
        return "rlhf"
    if pd.notna(row.get("expert_id")):
        return "ep"
    return "startup"
