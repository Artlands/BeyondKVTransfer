#!/usr/bin/env python3
"""
validate_traces.py — BeyondKVTransfer trace validator (§10, §4.8).

Usage
-----
    python scripts/validate_traces.py <path-or-glob> [options]

Arguments
---------
  path          Path to a .jsonl file, a .jsonl.gz file, a directory
                containing trace files, or a glob pattern.

Options
-------
  --sample FRAC   Only validate a random sample of records (e.g. 0.01 = 1 %).
                  Default: validate all records.
  --strict        Exit with non-zero status if any orphan transfer or request
                  is found (default: warn only).
  --no-schema     Skip JSON Schema validation (faster).
  --schema-dir    Path to schemas/ directory. Default: auto-detect from the
                  script's location or the repo root.
  --quiet         Suppress per-error messages; print only the summary.

Exit codes
----------
  0   All records valid (and no orphans in strict mode).
  1   Validation errors found.
  2   File/argument error.

Checks performed
----------------
1. Every line is valid JSON.
2. Every record passes the JSON Schema for its ``type``.
3. Every ``transfer.start`` has a matching ``transfer.end`` or
   ``transfer.cancel`` (same ``transfer_id``).
4. Every ``request.arrival`` has a matching ``request.finish`` or
   ``request.abort`` (same ``request_id``).
5. Schema version ``v`` == 1.

Missing manifest
----------------
Analysis tools MUST refuse to run without a valid manifest (§10).
This script warns if no ``manifest.json`` is found in the trace directory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Schema loading (optional — skipped when --no-schema is given or jsonschema
# is not installed)
# ---------------------------------------------------------------------------

_SCHEMA_TYPE_MAP: dict[str, str] = {
    "request":     "request.schema.json",
    "token":       "token.schema.json",
    "kv_block":    "kv_block.schema.json",
    "weight_block": "weight_block.schema.json",
    "transfer":    "transfer.schema.json",
    "metadata":    "metadata.schema.json",
    "sys_counter": "sys_counter.schema.json",
}

_validators: dict[str, object] = {}  # type -> jsonschema.Draft202012Validator


def _load_validators(schema_dir: str) -> bool:
    """Load JSON Schema validators.  Returns True on success."""
    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError:
        print("WARNING: jsonschema not installed; schema validation disabled.", file=sys.stderr)
        return False

    ok = True
    for rtype, fname in _SCHEMA_TYPE_MAP.items():
        path = os.path.join(schema_dir, fname)
        if not os.path.isfile(path):
            print(f"WARNING: schema file not found: {path}", file=sys.stderr)
            ok = False
            continue
        with open(path) as fh:
            schema = json.load(fh)
        _validators[rtype] = jsonschema.Draft202012Validator(schema)

    return ok


def _validate_schema(record: dict) -> list[str]:
    """Return list of schema-violation messages for this record."""
    rtype = record.get("type", "")
    validator = _validators.get(rtype)
    if validator is None:
        return []
    errors = []
    for err in validator.iter_errors(record):
        errors.append(f"{err.json_path}: {err.message}")
    return errors


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _iter_files(paths: list[str]) -> Iterator[str]:
    for p in paths:
        if os.path.isfile(p):
            yield p
        elif os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith(".jsonl") or f.endswith(".jsonl.gz"):
                        yield os.path.join(root, f)
        else:
            # Glob
            for match in sorted(glob(p, recursive=True)):
                if os.path.isfile(match):
                    yield match


def _iter_records(path: str, sample: float) -> Iterator[tuple[int, dict | None, str | None]]:
    """Yield (lineno, record_dict_or_None, raw_line).

    record_dict is None when JSON parsing fails.
    """
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    try:
        with opener(path, mode, encoding="utf-8") as fh:  # type: ignore[call-overload]
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.rstrip("\n")
                if not raw:
                    continue
                if sample < 1.0 and random.random() > sample:
                    continue
                try:
                    yield lineno, json.loads(raw), raw
                except json.JSONDecodeError as exc:
                    yield lineno, None, f"JSON parse error: {exc}"
    except (OSError, EOFError) as exc:
        print(f"ERROR: cannot open {path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Orphan detection
# ---------------------------------------------------------------------------

def _check_manifest(trace_dir: str) -> bool:
    """Return True if manifest.json exists in trace_dir or a parent."""
    for d in [trace_dir, os.path.dirname(trace_dir)]:
        if os.path.isfile(os.path.join(d, "manifest.json")):
            return True
    return False


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------

def validate_files(
    paths: list[str],
    sample: float = 1.0,
    use_schema: bool = True,
    strict: bool = False,
    quiet: bool = False,
) -> int:
    """Validate a collection of JSONL files.

    Returns 0 (no errors) or 1 (errors found).
    """
    total_records = 0
    total_errors = 0
    total_files = 0

    # Per-trace tracking for orphan detection (keyed by trace_id)
    # transfer: {transfer_id -> {"start": bool, "end": bool}}
    # request:  {request_id -> {"arrival": bool, "finish": bool}}
    transfer_state: dict[str, dict[str, dict[str, bool]]] = defaultdict(dict)
    request_state:  dict[str, dict[str, dict[str, bool]]] = defaultdict(dict)

    for fpath in _iter_files(paths):
        total_files += 1
        file_errors = 0

        # Check for manifest
        tdir = os.path.dirname(fpath)
        if not _check_manifest(tdir) and not quiet:
            print(f"WARNING: no manifest.json found near {fpath}")

        for lineno, record, raw in _iter_records(fpath, sample):
            total_records += 1

            if record is None:
                # JSON parse failure
                if not quiet:
                    print(f"  {fpath}:{lineno}: {raw}")
                file_errors += 1
                continue

            # Schema version
            if record.get("v") != 1:
                if not quiet:
                    print(f"  {fpath}:{lineno}: unexpected schema version v={record.get('v')!r}")
                file_errors += 1

            # JSON Schema
            if use_schema and _validators:
                schema_errs = _validate_schema(record)
                for e in schema_errs:
                    if not quiet:
                        print(f"  {fpath}:{lineno}: schema error: {e}")
                    file_errors += 1

            # Envelope required fields
            for req in ("ts_ns", "type", "node_id", "worker_id", "trace_id"):
                if req not in record:
                    if not quiet:
                        print(f"  {fpath}:{lineno}: missing required field '{req}'")
                    file_errors += 1

            # Track transfer pairing
            rtype = record.get("type", "")
            subtype = record.get("subtype", "")
            trace_id = record.get("trace_id", "__unknown__")

            if rtype == "transfer":
                tid = record.get("transfer_id", "")
                if tid:
                    ts = transfer_state[trace_id]
                    entry = ts.setdefault(tid, {"start": False, "end": False})
                    if subtype == "start":
                        entry["start"] = True
                    elif subtype in ("end", "cancel"):
                        entry["end"] = True

            # Track request pairing
            if rtype == "request":
                rid = record.get("request_id", "")
                if rid:
                    rs = request_state[trace_id]
                    entry = rs.setdefault(rid, {"arrival": False, "finish": False})
                    if subtype == "arrival":
                        entry["arrival"] = True
                    elif subtype in ("finish", "abort"):
                        entry["finish"] = True

        total_errors += file_errors

    # ------------------------------------------------------------------
    # Orphan report
    # ------------------------------------------------------------------
    orphan_transfers = 0
    for trace_id, tmap in transfer_state.items():
        for tid, st in tmap.items():
            if st["start"] and not st["end"]:
                orphan_transfers += 1
                if not quiet:
                    print(f"ORPHAN transfer (no end): trace={trace_id} transfer_id={tid}")

    orphan_requests = 0
    for trace_id, rmap in request_state.items():
        for rid, st in rmap.items():
            if st["arrival"] and not st["finish"]:
                orphan_requests += 1
                if not quiet:
                    print(f"ORPHAN request (no finish): trace={trace_id} request_id={rid}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    status_str = "OK" if total_errors == 0 else "FAIL"
    print(
        f"\n[{status_str}] files={total_files} records={total_records} "
        f"errors={total_errors} "
        f"orphan_transfers={orphan_transfers} orphan_requests={orphan_requests}"
    )

    if total_errors > 0:
        return 1
    if strict and (orphan_transfers > 0 or orphan_requests > 0):
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_schema_dir(given: str | None) -> str:
    if given and os.path.isdir(given):
        return given
    # Try relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(script_dir, "..", "schemas"),
        os.path.join(os.getcwd(), "schemas"),
    ]:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return "schemas"  # fallback; will warn if not found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate BeyondKVTransfer trace files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths", nargs="+",
        help="JSONL file(s), directories, or glob patterns to validate.",
    )
    parser.add_argument(
        "--sample", type=float, default=1.0, metavar="FRAC",
        help="Fraction of records to validate (e.g. 0.01 = 1%%). Default: all.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero if orphan transfers or requests are found.",
    )
    parser.add_argument(
        "--no-schema", action="store_true",
        help="Skip JSON Schema validation.",
    )
    parser.add_argument(
        "--schema-dir", default=None, metavar="DIR",
        help="Path to schemas/ directory.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-error messages; print summary only.",
    )

    args = parser.parse_args(argv)

    if not (0.0 < args.sample <= 1.0):
        print("ERROR: --sample must be in (0, 1]", file=sys.stderr)
        return 2

    use_schema = not args.no_schema
    if use_schema:
        schema_dir = _find_schema_dir(args.schema_dir)
        _load_validators(schema_dir)
        if not _validators and not args.no_schema:
            print("WARNING: no validators loaded; schema checks skipped.", file=sys.stderr)

    return validate_files(
        paths=args.paths,
        sample=args.sample,
        use_schema=use_schema,
        strict=args.strict,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
