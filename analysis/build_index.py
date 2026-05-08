#!/usr/bin/env python3
"""Build a tabular index from BeyondKVTransfer JSONL traces.

The index contains one table per record type plus derived tables used by the
Q1-Q6 notebook. Parquet output requires either ``pyarrow`` or ``duckdb``.
Use ``--format jsonl`` for lightweight local smoke tests without those extras.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lib.critical_path import (  # noqa: E402
    critical_path_attribution,
    request_lifecycle,
    scheduler_tail_latency,
)
from analysis.lib.load import RECORD_TYPES, load_trace, require_manifest  # noqa: E402
from analysis.lib.prefetch import prefetch_slack  # noqa: E402
from analysis.lib.reuse import reuse_distance, tier_residency  # noqa: E402
from analysis.lib.weights import (  # noqa: E402
    lora_swap_latency,
    per_family_byte_rollup,
    weight_update_windows,
)


def build_index(
    trace_path: str | Path,
    output_dir: str | Path,
    *,
    output_format: str = "parquet",
    overwrite: bool = False,
) -> dict[str, int]:
    """Build an index and return row counts by table."""

    manifest = require_manifest(trace_path)
    out = Path(output_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    tables = load_trace(trace_path, require_valid_manifest=False)
    tables["transfer_pairs"] = _transfer_pairs(tables["transfer"])
    tables["request_lifecycle"] = request_lifecycle(tables["request"], tables["token"])
    tables["critical_path"] = critical_path_attribution(
        tables["request_lifecycle"], tables["transfer"], tables["metadata"]
    )
    tables["prefetch_slack"] = prefetch_slack(tables["transfer"])
    tables["reuse_distance"] = reuse_distance(tables["kv_block"])
    tables["tier_residency"] = tier_residency(tables["kv_block"])
    tables["scheduler_tail_latency"] = scheduler_tail_latency(
        tables["metadata"], tables["request_lifecycle"]
    )
    tables["weight_bytes"] = per_family_byte_rollup(tables["weight_block"], tables["transfer"])
    tables["lora_swap_latency"] = lora_swap_latency(tables["weight_block"], tables["transfer"])
    tables["weight_update_windows"] = weight_update_windows(tables["weight_block"], tables["transfer"])

    for name, table in tables.items():
        _write_table(table, out / name, output_format)

    with open(manifest, encoding="utf-8") as fh:
        manifest_data = json.load(fh)
    with open(out / "index_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "source_trace": str(Path(trace_path).resolve()),
                "source_manifest": str(manifest),
                "output_format": output_format,
                "tables": {name: int(len(table)) for name, table in tables.items()},
                "trace_manifest": manifest_data,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
        fh.write("\n")

    return {name: int(len(table)) for name, table in tables.items()}


def _transfer_pairs(transfers: pd.DataFrame) -> pd.DataFrame:
    if transfers.empty or "transfer_id" not in transfers:
        return pd.DataFrame()

    starts = transfers[transfers["subtype"] == "start"].copy()
    ends = transfers[transfers["subtype"].isin(["end", "cancel"])].copy()
    if starts.empty:
        return pd.DataFrame()

    start_cols = [col for col in starts.columns if col != "subtype"]
    end_cols = [col for col in ends.columns if col not in {"subtype", "trace_id", "node_id", "worker_id"}]
    paired = starts[start_cols].merge(
        ends[end_cols],
        on="transfer_id",
        how="left",
        suffixes=("_start", "_end"),
    )
    if "ts_ns_start" in paired and "ts_ns_end" in paired:
        paired["observed_duration_ns"] = paired["ts_ns_end"] - paired["ts_ns_start"]
    if "started_ts_ns_start" in paired and "completed_ts_ns_end" in paired:
        paired["wire_time_observed_ns"] = paired["completed_ts_ns_end"] - paired["started_ts_ns_start"]
    return paired


def _write_table(table: pd.DataFrame, stem: Path, output_format: str) -> None:
    if output_format == "jsonl":
        table.to_json(stem.with_suffix(".jsonl"), orient="records", lines=True)
        return

    table = _prepare_parquet_table(table)
    try:
        import pyarrow  # noqa: F401

        table.to_parquet(stem.with_suffix(".parquet"), index=False)
        return
    except ImportError:
        pass

    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow or duckdb. Install the analysis "
            "extra or rerun with --format jsonl."
        ) from exc

    con = duckdb.connect(database=":memory:")
    con.register("bkvt_table", table)
    con.sql("SELECT * FROM bkvt_table").write_parquet(str(stem.with_suffix(".parquet")))
    con.close()


def _prepare_parquet_table(table: pd.DataFrame) -> pd.DataFrame:
    """Serialize mixed object columns that Parquet engines cannot infer."""

    if table.empty:
        return table
    out = table.copy()
    for column in out.columns:
        if out[column].dtype != "object":
            continue
        if out[column].map(lambda value: isinstance(value, (dict, list))).any():
            out[column] = out[column].map(
                lambda value: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", help="Raw trace directory or JSONL file")
    parser.add_argument("-o", "--output-dir", default="analysis/index", help="Directory for output tables")
    parser.add_argument("--format", choices=["parquet", "jsonl"], default="parquet")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it exists")
    args = parser.parse_args(argv)

    counts = build_index(
        args.trace_path,
        args.output_dir,
        output_format=args.format,
        overwrite=args.overwrite,
    )
    for table in sorted(counts):
        print(f"{table}: {counts[table]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
