"""
Manifest writer for BeyondKVTransfer (§10).

A ``manifest.json`` is written to
``${BKVT_OUTPUT_DIR}/${trace_id}/manifest.json`` at trace start.

The manifest MUST include (§10):
  - BKVT version (git SHA)
  - vLLM / SGLang versions (if installed)
  - model name (from env ``BKVT_MODEL_NAME`` or inference)
  - GPU model (from nvidia-smi / NVML / torch)
  - NCCL version (if available)
  - NIXL version (if available)
  - kernel version
  - sampling configuration
  - all ``BKVT_*`` env var values

Without a well-formed manifest, analysis tools MUST refuse to run (§10).
The ``"manifest_version"`` field lets tools detect schema changes.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Optional

from bkvt.clock import t0_monotonic_ns, t0_unix_ns
from bkvt import __version__, __git_sha__

if TYPE_CHECKING:
    from bkvt.emitter import Emitter

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


# ---------------------------------------------------------------------------
# Version sniffers
# ---------------------------------------------------------------------------

def _try_version(package: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return None


def _gpu_info() -> list[dict]:
    """Return a list of GPU dicts via torch.cuda or nvidia-smi."""
    gpus: list[dict] = []

    # Try torch first (most reliable)
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
            })
        if gpus:
            return gpus
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_memory_bytes": int(parts[2]) * 1024 * 1024,
                })
    except Exception:
        pass

    return gpus


def _nccl_version() -> Optional[str]:
    try:
        import torch
        return torch.cuda.nccl.version().__str__()
    except Exception:
        pass
    return _try_version("nccl")


def _nixl_version() -> Optional[str]:
    return _try_version("nixl")


def _chrony_offset_ns() -> Optional[int]:
    """Read chrony's current tracking offset in nanoseconds (best-effort)."""
    try:
        out = subprocess.check_output(
            ["chronyc", "tracking"], text=True, timeout=3
        )
        for line in out.splitlines():
            if "System time" in line:
                # e.g. "System time     :   0.000012345 seconds fast of NTP time"
                parts = line.split(":")
                if len(parts) >= 2:
                    val_str = parts[1].strip().split()[0]
                    return int(float(val_str) * 1e9)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(emitter: "Emitter") -> dict:
    cfg = emitter._config

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "bkvt_version": __version__,
        "bkvt_git_sha": __git_sha__,

        # Anchor timestamps
        "t0_unix_ns": t0_unix_ns,
        "t0_monotonic_ns": t0_monotonic_ns,

        # Identity
        "trace_id": emitter.trace_id,
        "node_id": emitter.node_id,
        "worker_id": emitter.worker_id,

        # Runtime
        "python_version": sys.version,
        "platform": platform.platform(),
        "kernel": platform.uname().release,

        # Target engine versions
        "vllm_version": _try_version("vllm"),
        "sglang_version": _try_version("sglang"),
        "torch_version": _try_version("torch"),
        "nccl_version": _nccl_version(),
        "nixl_version": _nixl_version(),

        # Hardware
        "gpus": _gpu_info(),

        # Model (user-supplied or left null)
        "model_name": os.environ.get("BKVT_MODEL_NAME"),

        # Sampling config
        "profile": cfg.profile,
        "sample_token": cfg.sample_token,
        "sample_metadata": cfg.sample_metadata,
        "sample_transfer": cfg.sample_transfer,

        # Full BKVT_* env snapshot
        "bkvt_env": cfg._raw_env,

        # Unsupported scenarios (§15 item 4)
        "unsupported": [],
    }

    # Chrony offset for cross-node skew correction (§7.4)
    manifest["chrony_offset_ns"] = _chrony_offset_ns()

    return manifest


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_manifest(emitter: "Emitter") -> str:
    """Write ``manifest.json`` and emit a ``clock_anchor`` metadata record.

    Returns the path to the written file.
    """
    manifest = build_manifest(emitter)
    cfg = emitter._config

    out_dir = os.path.join(
        cfg.output_dir, emitter.trace_id
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.json")

    try:
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        logger.info("bkvt: manifest written to %s", path)
    except Exception as exc:
        logger.error("bkvt: failed to write manifest: %s", exc)

    # Also emit a clock_anchor metadata record into the trace stream
    emitter.event({
        "type": "metadata",
        "subtype": "clock_anchor",
        "ts_ns": t0_monotonic_ns,
        "t0_unix_ns": t0_unix_ns,
        "t0_monotonic_ns": t0_monotonic_ns,
        "t0_cuda_event_ns": None,
        "chrony_offset_ns": manifest.get("chrony_offset_ns"),
    })

    return path
