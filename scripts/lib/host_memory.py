"""Available and total host memory in KiB, on Linux and macOS.

Linux: /proc/meminfo MemAvailable / MemTotal. macOS: (free + inactive +
purgeable) vm_stat pages, the definition 06_memwatch_macos.sh and
07_memory_guard_macos.py gate on, and sysctl hw.memsize. Plain stdlib.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

MEMINFO = Path("/proc/meminfo")
VM_STAT_COUNTERS = ("Pages free", "Pages inactive", "Pages purgeable")


def _meminfo(path: Path, key: str) -> int:
    for line in path.read_text().splitlines():
        name, _, rest = line.partition(":")
        if name == key:
            return int(rest.split()[0])
    raise ValueError(f"{key} not found in {path}")


def _vm_stat() -> str:
    return subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout


def parse_vm_stat_kib(text: str) -> int:
    page = re.search(r"page size of (\d+) bytes", text)
    counts = [re.search(rf"^{name}:\s+(\d+)\.", text, re.M) for name in VM_STAT_COUNTERS]
    if not page or not all(counts):
        raise ValueError("vm_stat output has no recognizable page counters")
    return sum(int(match.group(1)) for match in counts) * int(page.group(1)) // 1024


def available_kib(meminfo: Path = MEMINFO) -> int:
    if meminfo.exists():
        return _meminfo(meminfo, "MemAvailable")
    return parse_vm_stat_kib(_vm_stat())


def total_kib(meminfo: Path = MEMINFO) -> int:
    if meminfo.exists():
        return _meminfo(meminfo, "MemTotal")
    memsize = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, check=True).stdout
    return int(memsize) // 1024
