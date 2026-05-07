"""Prefetchability metrics for Q5."""

from __future__ import annotations

import pandas as pd


def prefetch_slack(transfers: pd.DataFrame) -> pd.DataFrame:
    """Return transfer rows with ``prefetch_slack_ns``.

    Slack is ``started_ts_ns - earliest_known_ts_ns``. Only rows with both
    timestamps are included.
    """

    if transfers.empty or "earliest_known_ts_ns" not in transfers or "started_ts_ns" not in transfers:
        return pd.DataFrame()
    rows = transfers.dropna(subset=["earliest_known_ts_ns", "started_ts_ns"]).copy()
    if rows.empty:
        return rows
    rows["prefetch_slack_ns"] = rows["started_ts_ns"] - rows["earliest_known_ts_ns"]
    return rows

