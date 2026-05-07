"""System-counter and periodic clock-anchor collector for BKVT."""

from __future__ import annotations

import logging
import os
import platform
import resource
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bkvt.clock import now_ns, t0_monotonic_ns, t0_unix_ns
from bkvt.collectors.ib_counters import iter_ib_counter_samples
from bkvt.collectors.nccl_log import NcclLogTailer
from bkvt.collectors.nvml import collect_gpu_counter_samples

if TYPE_CHECKING:
    from bkvt.config import BkvtConfig
    from bkvt.emitter import Emitter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CounterSample:
    subtype: str
    scope: str
    value: int | float
    unit: str


class SystemCounterCollector:
    """Background sampler for NIC/GPU/process counters and clock anchors."""

    def __init__(self, emitter: "Emitter", config: "BkvtConfig") -> None:
        self._emitter = emitter
        self._config = config
        self._period_s = 1.0 / config.sys_counter_hz if config.sys_counter_hz > 0 else 0.0
        self._clock_period_s = (
            1.0 / config.clock_anchor_hz if config.clock_anchor_hz > 0 else 0.0
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_counter_ts_ns: int | None = None
        self._last_clock_anchor_s = 0.0
        self._nccl_tailer = (
            NcclLogTailer(os.environ["BKVT_NCCL_LOG"])
            if os.environ.get("BKVT_NCCL_LOG")
            else None
        )

    def start(self) -> None:
        if self._period_s <= 0 and self._clock_period_s <= 0:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="bkvt-sys-counters",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = now_ns()
            interval_ns = None
            if self._last_counter_ts_ns is not None:
                interval_ns = now - self._last_counter_ts_ns
            self._last_counter_ts_ns = now

            if self._period_s > 0:
                self.collect_once(interval_ns=interval_ns)

            if self._clock_period_s > 0:
                monotonic_s = now / 1_000_000_000
                if monotonic_s - self._last_clock_anchor_s >= self._clock_period_s:
                    self.emit_clock_anchor()
                    self._last_clock_anchor_s = monotonic_s

            sleep_s = min(
                p for p in (self._period_s, self._clock_period_s) if p > 0
            )
            self._stop.wait(sleep_s)

    def collect_once(self, interval_ns: int | None = None) -> None:
        ts = now_ns()
        for sample in self._collect_samples():
            self._emitter.event({
                "ts_ns": ts,
                "type": "sys_counter",
                "subtype": sample.subtype,
                "scope": sample.scope,
                "value": sample.value,
                "unit": sample.unit,
                "interval_ns": interval_ns,
            })

    def emit_clock_anchor(self) -> None:
        try:
            from bkvt.manifest import chrony_offset_ns
            offset = chrony_offset_ns()
        except Exception:
            offset = None
        self._emitter.event({
            "ts_ns": now_ns(),
            "type": "metadata",
            "subtype": "clock_anchor",
            "t0_unix_ns": t0_unix_ns,
            "t0_monotonic_ns": t0_monotonic_ns,
            "t0_cuda_event_ns": None,
            "chrony_offset_ns": offset,
        })

    def _collect_samples(self) -> list[CounterSample]:
        samples: list[CounterSample] = []
        samples.extend(_process_samples())
        samples.extend(_netdev_samples())
        samples.extend(_convert_samples(iter_ib_counter_samples()))
        samples.extend(_convert_samples(collect_gpu_counter_samples()))
        if self._nccl_tailer is not None:
            samples.extend(_convert_samples(self._nccl_tailer.collect()))

        dropped = getattr(self._emitter, "dropped_records_total", 0)
        samples.append(CounterSample("dropped_records", "process:bkvt", dropped, "count"))
        return samples


def _convert_samples(samples: list[object]) -> list[CounterSample]:
    converted: list[CounterSample] = []
    for sample in samples:
        converted.append(CounterSample(
            subtype=getattr(sample, "subtype"),
            scope=getattr(sample, "scope"),
            value=getattr(sample, "value"),
            unit=getattr(sample, "unit"),
        ))
    return converted


def _process_samples() -> list[CounterSample]:
    samples: list[CounterSample] = []
    host_dram = _read_host_dram_used_bytes()
    if host_dram is not None:
        samples.append(CounterSample("host_dram_used", "node", host_dram, "bytes"))

    rss = _read_process_rss_bytes()
    if rss is not None:
        samples.append(CounterSample("process_rss", "process:self", rss, "bytes"))

    usage = resource.getrusage(resource.RUSAGE_SELF)
    samples.append(
        CounterSample("cpu_pagefault", "process:self", usage.ru_minflt + usage.ru_majflt, "count")
    )
    return samples


def _read_host_dram_used_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    values: dict[str, int] = {}
    try:
        for line in meminfo.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].rstrip(":") in {"MemTotal", "MemAvailable"}:
                values[parts[0].rstrip(":")] = int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None
    return max(0, total - available)


def _read_process_rss_bytes() -> int | None:
    status = Path("/proc/self/status")
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except (OSError, ValueError):
        pass

    try:
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if maxrss <= 0:
        return None
    if platform.system() == "Darwin":
        return int(maxrss)
    return int(maxrss) * 1024


def _netdev_samples() -> list[CounterSample]:
    path = Path("/proc/net/dev")
    try:
        lines = path.read_text().splitlines()[2:]
    except OSError:
        return []

    samples: list[CounterSample] = []
    for line in lines:
        if ":" not in line:
            continue
        iface, rest = line.split(":", 1)
        fields = rest.split()
        if len(fields) < 16:
            continue
        scope = f"nic:{iface.strip()}"
        try:
            rx_bytes = int(fields[0])
            rx_packets = int(fields[1])
            tx_bytes = int(fields[8])
            tx_packets = int(fields[9])
        except ValueError:
            continue
        samples.append(CounterSample("nic_bytes", scope, rx_bytes + tx_bytes, "bytes"))
        samples.append(CounterSample("nic_packets", scope, rx_packets + tx_packets, "count"))
    return samples
