"""Parser for coarse NCCL P2P log-derived counters.

NCCL debug output is not a stable metrics API, so this parser intentionally
extracts only conservative byte totals from lines that include Send/Recv-like
events and a recognizable byte/count field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_P2P_RE = re.compile(r"\b(Send|Recv|P2P)\b", re.IGNORECASE)
_BYTES_RE = re.compile(r"\b(?:bytes|nbytes|size|count)\s*[:=]\s*(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class CounterSample:
    subtype: str
    scope: str
    value: int | float
    unit: str


class NcclLogTailer:
    """Tail one NCCL debug log and return cumulative P2P byte samples."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._offset = 0
        self._total_bytes = 0

    def collect(self) -> list[CounterSample]:
        if not self._path.is_file():
            return []
        try:
            with self._path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._offset)
                for line in fh:
                    if _P2P_RE.search(line) is None:
                        continue
                    match = _BYTES_RE.search(line)
                    if match is not None:
                        self._total_bytes += int(match.group(1))
                self._offset = fh.tell()
        except OSError:
            return []

        if self._total_bytes <= 0:
            return []
        return [CounterSample("nccl_p2p_bytes", "node", self._total_bytes, "bytes")]
