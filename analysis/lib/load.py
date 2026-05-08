"""Trace discovery and loading utilities.

The analysis layer accepts either a raw trace directory containing
``manifest.json`` plus ``*.jsonl``/``*.jsonl.gz`` files, or an indexed
directory produced by ``analysis/build_index.py``.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

RECORD_TYPES = ("request", "token", "kv_block", "weight_block", "transfer", "metadata", "sys_counter")


def require_manifest(trace_path: str | os.PathLike[str]) -> Path:
    """Return the manifest path, or raise if none exists.

    DESIGN.md §10 requires analysis tools to refuse traces without a manifest.
    ``trace_path`` may be the trace root, a nested node directory, or an index
    directory with a sibling raw trace.
    """

    start = Path(trace_path).resolve()
    candidates = [start]
    if start.is_file():
        candidates = [start.parent]
    candidates.extend(candidates[0].parents)

    for directory in candidates:
        manifest = directory / "manifest.json"
        if manifest.is_file():
            return manifest
    raise FileNotFoundError(f"no manifest.json found at or above {start}")


def discover_trace_files(trace_path: str | os.PathLike[str]) -> list[Path]:
    """Find trace JSONL files below ``trace_path``."""

    root = Path(trace_path)
    if root.is_file():
        return [root] if _is_trace_file(root) else []
    return sorted(path for path in root.rglob("*") if path.is_file() and _is_trace_file(path))


def iter_records(files: Iterable[Path]) -> Iterator[dict]:
    """Yield decoded JSON records from JSONL and JSONL.GZ files."""

    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        mode = "rt" if path.suffix == ".gz" else "r"
        with opener(path, mode, encoding="utf-8") as fh:  # type: ignore[call-overload]
            for raw in fh:
                raw = raw.strip()
                if raw:
                    yield json.loads(raw)


def load_trace(trace_path: str | os.PathLike[str], *, require_valid_manifest: bool = True) -> dict[str, pd.DataFrame]:
    """Load raw trace JSONL files into one DataFrame per record type."""

    if require_valid_manifest:
        require_manifest(trace_path)

    by_type: dict[str, list[dict]] = {rtype: [] for rtype in RECORD_TYPES}
    for record in iter_records(discover_trace_files(trace_path)):
        rtype = record.get("type")
        if rtype in by_type:
            by_type[rtype].append(_flatten_record(record))

    return {rtype: pd.DataFrame(rows) for rtype, rows in by_type.items()}


def load_index(index_path: str | os.PathLike[str]) -> dict[str, pd.DataFrame]:
    """Load an index produced by ``analysis/build_index.py``.

    Parquet is preferred. JSONL table files are accepted as a lightweight
    fallback for developer environments without Parquet dependencies.
    """

    root = Path(index_path)
    tables: dict[str, pd.DataFrame] = {}
    for name in (
        *RECORD_TYPES,
        "transfer_pairs",
        "request_lifecycle",
        "prefetch_slack",
        "tier_residency",
        "weight_bytes",
        "lora_swap_latency",
        "weight_update_windows",
    ):
        parquet = root / f"{name}.parquet"
        jsonl = root / f"{name}.jsonl"
        if parquet.exists():
            tables[name] = pd.read_parquet(parquet)
        elif jsonl.exists():
            tables[name] = pd.read_json(jsonl, lines=True)
        else:
            tables[name] = pd.DataFrame()
    return tables


def _is_trace_file(path: Path) -> bool:
    name = path.name
    return name.endswith(".jsonl") or name.endswith(".jsonl.gz")


def _flatten_record(record: dict) -> dict:
    """Flatten nested endpoint objects enough for tabular analysis."""

    flat = dict(record)
    for endpoint in ("src", "dst"):
        value = flat.pop(endpoint, None)
        if isinstance(value, dict):
            for key, item in value.items():
                flat[f"{endpoint}_{key}"] = item
    return flat
