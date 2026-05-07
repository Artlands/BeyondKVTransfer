"""Critical-path and scheduler-impact derived tables."""

from __future__ import annotations

import json

import pandas as pd


def request_lifecycle(requests: pd.DataFrame, tokens: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one row per request with lifecycle timestamps and latencies."""

    if requests.empty:
        return pd.DataFrame()

    rows = []
    for request_id, group in requests.groupby("request_id", dropna=False):
        group = group.sort_values("ts_ns")
        arrival = _coalesce(_first_value(group, "arrival_ts_ns", subtype="arrival"), _first_ts(group, "arrival"))
        first_schedule = _coalesce(
            _first_value(group, "first_schedule_ts_ns", subtype="admit"), _first_ts(group, "admit")
        )
        first_token = _coalesce(
            _first_value(group, "first_token_ts_ns", subtype="finish"), _first_ts(group, "first_token")
        )
        finish = _coalesce(_first_value(group, "finish_ts_ns", subtype="finish"), _first_ts(group, "finish"))
        abort = _first_ts(group, "abort")

        if first_token is None and tokens is not None and not tokens.empty and "request_id" in tokens:
            token_rows = tokens[(tokens["request_id"] == request_id) & (tokens["subtype"] == "first_token")]
            if not token_rows.empty:
                first_token = int(token_rows["ts_ns"].min())

        terminal = finish or abort
        ttft = _first_value(group, "ttft_ns", subtype="finish")
        if ttft is None and arrival is not None and first_token is not None:
            ttft = first_token - arrival

        tpot = _first_value(group, "tpot_ns", subtype="finish")
        end_to_end = terminal - arrival if terminal is not None and arrival is not None else None

        rows.append(
            {
                "request_id": request_id,
                "trace_id": _first_non_null(group, "trace_id"),
                "arrival_ts_ns": arrival,
                "first_schedule_ts_ns": first_schedule,
                "first_token_ts_ns": first_token,
                "finish_ts_ns": finish,
                "abort_ts_ns": abort,
                "ttft_ns": ttft,
                "tpot_ns": tpot,
                "end_to_end_ns": end_to_end,
                "status": "abort" if abort is not None else "finish" if finish is not None else "open",
            }
        )
    return pd.DataFrame(rows)


def critical_path_attribution(
    lifecycle: pd.DataFrame,
    transfers: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Compute coarse per-request latency attribution for Q3.

    Attribution uses request lifecycle fields where available and joins
    transfer/metadata durations by request_id. It intentionally stays coarse:
    exact dependency ordering is engine-specific, while this table is stable
    across vLLM and SGLang traces.
    """

    if lifecycle.empty:
        return pd.DataFrame(columns=["request_id", "stage", "duration_ns"])

    transfer_wait = _sum_by_request(transfers, "queue_wait_ns") + _sum_by_request(transfers, "wire_time_ns")
    metadata_time = _sum_by_request(metadata, "duration_ns")

    rows = []
    for row in lifecycle.itertuples(index=False):
        request_id = getattr(row, "request_id")
        arrival = getattr(row, "arrival_ts_ns", None)
        first_schedule = getattr(row, "first_schedule_ts_ns", None)
        first_token = getattr(row, "first_token_ts_ns", None)
        finish = getattr(row, "finish_ts_ns", None)

        queue_ns = _nonnegative_diff(first_schedule, arrival)
        prefill_ns = _nonnegative_diff(first_token, first_schedule)
        decode_ns = _nonnegative_diff(finish, first_token)
        rows.extend(
            [
                {"request_id": request_id, "stage": "queue", "duration_ns": queue_ns},
                {"request_id": request_id, "stage": "prefill_to_first_token", "duration_ns": prefill_ns},
                {"request_id": request_id, "stage": "decode_after_first_token", "duration_ns": decode_ns},
                {"request_id": request_id, "stage": "remote_transfer", "duration_ns": int(transfer_wait.get(request_id, 0))},
                {"request_id": request_id, "stage": "metadata", "duration_ns": int(metadata_time.get(request_id, 0))},
            ]
        )
    return pd.DataFrame(rows)


def scheduler_tail_latency(metadata: pd.DataFrame, lifecycle: pd.DataFrame) -> pd.DataFrame:
    """Join scheduler decision records with following request tail latency."""

    if metadata.empty or lifecycle.empty:
        return pd.DataFrame()

    decisions = metadata[metadata["subtype"] == "scheduler_decision"].copy()
    if decisions.empty:
        return pd.DataFrame()

    p99_ns = lifecycle["end_to_end_ns"].dropna().quantile(0.99) if "end_to_end_ns" in lifecycle else None
    decisions["workload_p99_ns"] = p99_ns
    decisions["queue_depth"] = decisions.apply(_queue_depth_from_row, axis=1)
    return decisions[["ts_ns", "trace_id", "worker_id", "queue_depth", "workload_p99_ns"]]


def _sum_by_request(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or "request_id" not in df or column not in df:
        return pd.Series(dtype="int64")
    values = df.dropna(subset=["request_id"]).copy()
    if values.empty:
        return pd.Series(dtype="int64")
    return values.groupby("request_id")[column].sum(numeric_only=True)


def _queue_depth_from_row(row: pd.Series) -> int | None:
    inputs = row.get("scheduler_inputs")
    if isinstance(inputs, str):
        try:
            inputs = json.loads(inputs)
        except json.JSONDecodeError:
            return None
    if isinstance(inputs, dict):
        waiting = inputs.get("waiting", 0) or inputs.get("waiting_queue", 0) or 0
        running = inputs.get("running", 0) or 0
        return int(waiting) + int(running)
    return None


def _first_ts(df: pd.DataFrame, subtype: str) -> int | None:
    rows = df[df["subtype"] == subtype]
    if rows.empty:
        return None
    return int(rows["ts_ns"].min())


def _first_value(df: pd.DataFrame, column: str, *, subtype: str | None = None) -> int | None:
    if column not in df:
        return None
    rows = df[df["subtype"] == subtype] if subtype is not None else df
    values = rows[column].dropna()
    if values.empty:
        return None
    return int(values.iloc[0])


def _first_non_null(df: pd.DataFrame, column: str) -> object:
    if column not in df:
        return None
    values = df[column].dropna()
    return values.iloc[0] if not values.empty else None


def _nonnegative_diff(end: int | None, start: int | None) -> int | None:
    if end is None or start is None:
        return None
    return max(0, int(end) - int(start))


def _coalesce(*values: int | None) -> int | None:
    for value in values:
        if value is not None:
            return value
    return None
