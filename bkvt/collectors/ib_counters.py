"""InfiniBand counter reader for BKVT system-counter sampling.

The Linux IB sysfs ABI exposes monotonically increasing per-port counters
under ``/sys/class/infiniband/<dev>/ports/<port>/counters``.  This module is
best-effort and dependency-free; missing files simply produce no samples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SYSFS_ROOT = "/sys/class/infiniband"


@dataclass(frozen=True)
class CounterSample:
    subtype: str
    scope: str
    value: int | float
    unit: str


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def iter_ib_counter_samples(sysfs_root: str = DEFAULT_SYSFS_ROOT) -> list[CounterSample]:
    """Return IB byte, packet, and constraint-error counters.

    ``port_xmit_data`` and ``port_rcv_data`` are reported by the kernel in
    32-bit words, so BKVT converts them to bytes for schema consistency.
    """
    root = Path(sysfs_root)
    if not root.is_dir():
        return []

    samples: list[CounterSample] = []
    for dev_dir in sorted(root.iterdir()):
        ports_dir = dev_dir / "ports"
        if not ports_dir.is_dir():
            continue
        for port_dir in sorted(ports_dir.iterdir()):
            counters_dir = port_dir / "counters"
            if not counters_dir.is_dir():
                continue
            scope = f"nic:{dev_dir.name}/port:{port_dir.name}"

            xmit_words = _read_int(counters_dir / "port_xmit_data")
            rcv_words = _read_int(counters_dir / "port_rcv_data")
            if xmit_words is not None:
                samples.append(CounterSample("nic_bytes", scope, xmit_words * 4, "bytes"))
            if rcv_words is not None:
                samples.append(CounterSample("nic_bytes", scope, rcv_words * 4, "bytes"))

            for name in ("port_xmit_packets", "port_rcv_packets"):
                value = _read_int(counters_dir / name)
                if value is not None:
                    samples.append(CounterSample("nic_packets", scope, value, "count"))

            pkey_total = 0
            found_pkey = False
            for name in ("port_xmit_constraint_errors", "port_rcv_constraint_errors"):
                value = _read_int(counters_dir / name)
                if value is not None:
                    pkey_total += value
                    found_pkey = True
            if found_pkey:
                samples.append(CounterSample("ib_pkey_violation", scope, pkey_total, "count"))

    return samples


def has_infiniband(sysfs_root: str = DEFAULT_SYSFS_ROOT) -> bool:
    return os.path.isdir(sysfs_root)
