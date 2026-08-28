#!/usr/bin/env python3
"""Describe this host for the profile resolver. Describe, never gate.

Emits host-facts JSON: OS, arch, RAM, GPUs, toolchains, available backends,
and a best-guess host class/tier from configs/hardware-matrix.json. Every
probe fails soft to null - unknown hardware still gets a complete document
with exit status 0. Fit decisions belong to the resolver
(scripts/92_resolve_profile.py); Spark lock-equality assertions stay in
scripts/00_preflight.sh and apply only when the facts match the lock host.

Usage: scripts/04_host_facts.py [--out FILE]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def ram_bytes(system: str) -> int | None:
    if system == "linux":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    if system == "darwin":
        value = run(["sysctl", "-n", "hw.memsize"])
        return int(value) if value and value.isdigit() else None
    return None


def nvidia_gpus() -> list[dict]:
    output = run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    for line in (output or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            try:
                vram = int(parts[1]) * 1024 * 1024
            except ValueError:
                vram = None
            gpus.append({
                "vendor": "nvidia",
                "name": parts[0],
                "vram_bytes": vram,
                "driver": parts[2],
            })
    return gpus


def amd_gfx() -> str | None:
    output = run(["rocminfo"])
    if not output:
        return None
    match = re.search(r"gfx[0-9a-f]+", output)
    return match.group(0) if match else None


def toolchains() -> dict:
    facts: dict[str, str | None] = {}
    nvcc = run(["nvcc", "--version"])
    match = re.search(r"release ([0-9.]+)", nvcc or "")
    facts["nvcc"] = match.group(1) if match else None
    facts["hipcc"] = "present" if shutil.which("hipcc") else None
    facts["cmake"] = (run(["cmake", "--version"]) or "").split("\n")[0] or None
    if platform.system() == "Darwin":
        facts["metal_sdk"] = run(["xcrun", "--find", "metal"])
    else:
        facts["metal_sdk"] = None
    return facts


def collect() -> dict:
    system = platform.system().lower()
    arch = platform.machine().lower()
    ram = ram_bytes(system)
    gpus = nvidia_gpus()
    gfx = amd_gfx()
    tools = toolchains()

    # GB10/Thor report memory.total as N/A through nvidia-smi; the name is
    # the reliable UMA signal there.
    uma = None
    if gpus and ram:
        vram = gpus[0].get("vram_bytes")
        if vram is not None:
            uma = vram >= 0.9 * ram
        elif any(marker in gpus[0]["name"] for marker in ("GB10", "Thor")):
            uma = True
    if gfx == "gfx1151":
        uma = True

    backends = ["cpu"]
    if gpus and tools.get("nvcc"):
        backends.insert(0, "cuda")
    if gfx and tools.get("hipcc"):
        backends.insert(0, "rocm")
    if system == "darwin" and arch == "arm64":
        backends.insert(0, "apple-silicon")

    ram_gib = round(ram / 2**30) if ram else None
    host_class = None
    tier = None
    if system == "darwin" and arch == "arm64":
        host_class = "mac"
    elif gpus and uma:
        host_class = "spark" if "GB10" in gpus[0]["name"] else "uma"
    elif gpus:
        host_class = "dgpu"
    elif gfx == "gfx1151":
        host_class = "strix"
    elif ram_gib:
        host_class = "any"
    try:
        with open(REPO_ROOT / "configs/hardware-matrix.json", encoding="utf-8") as stream:
            matrix = json.load(stream)
        tiers = matrix["host_classes"].get(host_class, {}).get("tiers_gib", [])
        if host_class == "dgpu" and gpus and gpus[0].get("vram_bytes"):
            reference = gpus[0]["vram_bytes"] / 2**30
        else:
            reference = ram_gib
        if reference:
            # MemTotal runs ~6% under the nominal marketing size (119.7 GiB
            # on a 128 GB machine); match tiers with 10% headroom.
            eligible = [value for value in tiers if value <= reference * 1.1]
            tier = max(eligible) if eligible else None
    except (OSError, ValueError, KeyError):
        tier = None

    facts = {
        "schema_version": 1,
        "collected_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "os": system,
        "os_version": platform.version(),
        "arch": arch,
        "ram_bytes": ram,
        "ram_gib": ram_gib,
        "cpu_logical": __import__("os").cpu_count(),
        "uma": uma,
        "gpus": gpus,
        "amd_gfx": gfx,
        "toolchains": tools,
        "backends_available": backends,
        "host_class_guess": host_class,
        "tier_guess_gib": tier,
    }
    canonical = json.dumps(
        {key: value for key, value in facts.items() if key != "collected_at"},
        sort_keys=True, separators=(",", ":"),
    )
    facts["fingerprint_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write JSON here instead of stdout")
    args = parser.parse_args()
    document = json.dumps(collect(), indent=1)
    if args.out:
        Path(args.out).write_text(document + "\n", encoding="utf-8")
    else:
        print(document)
    return 0


if __name__ == "__main__":
    sys.exit(main())
