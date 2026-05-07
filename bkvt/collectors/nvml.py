"""Best-effort GPU/NVLink counter reader.

The preferred path uses ``pynvml`` when installed.  If it is unavailable, the
module falls back to ``nvidia-smi`` for coarse GPU utilization and memory
readings.  Missing NVIDIA tooling is normal on development laptops and simply
returns no samples.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class CounterSample:
    subtype: str
    scope: str
    value: int | float
    unit: str


def collect_gpu_counter_samples() -> list[CounterSample]:
    samples = _collect_with_pynvml()
    if samples:
        return samples
    return _collect_with_nvidia_smi()


def _collect_with_pynvml() -> list[CounterSample]:
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception:
        return []

    samples: list[CounterSample] = []
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            scope = f"gpu:{idx}"
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                samples.append(CounterSample("gpu_sm_active", scope, float(util.gpu), "percent"))
                samples.append(CounterSample("gpu_hbm_bw_used", scope, float(util.memory), "percent"))
            except Exception:
                pass
    except Exception:
        return []
    return samples


def _collect_with_nvidia_smi() -> list[CounterSample]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        )
    except Exception:
        return []

    samples: list[CounterSample] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            gpu_util = float(parts[1])
            mem_util = float(parts[2])
        except ValueError:
            continue
        scope = f"gpu:{idx}"
        samples.append(CounterSample("gpu_sm_active", scope, gpu_util, "percent"))
        samples.append(CounterSample("gpu_hbm_bw_used", scope, mem_util, "percent"))
    return samples
