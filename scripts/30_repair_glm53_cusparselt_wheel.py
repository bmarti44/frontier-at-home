#!/usr/bin/env python3
"""Repair one pinned NVIDIA wheel's platform metadata; preserve every payload byte.

This is a local linux_aarch64 wheel, not a claim of manylinux compatibility.
The original upstream archive is retained. Installation is a separate step.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from wheel._commands.tags import tags
from wheel.wheelfile import WheelFile

UPSTREAM_SHA256 = json.loads((Path(__file__).resolve().parents[1] /
                              "configs/build-manifests/glm53-cusparselt-wheel.json").read_text())["sha256"]
UPSTREAM_NAME = "nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def members(path):
    result = {}
    native = []
    # WheelFile verifies each member against RECORD when it is read.
    with WheelFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive member")
        for name in names:
            if name.endswith("/"):
                continue
            data = archive.read(name)
            result[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            if data.startswith(b"\x7fELF"):
                if data[4:6] != b"\x02\x01" or int.from_bytes(data[18:20], "little") != 183:
                    raise ValueError(f"not a 64-bit little-endian AArch64 ELF: {name}")
                native.append(name)
    if not native:
        raise ValueError("missing native library")
    return result, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    original = args.original.resolve()
    if original.name != UPSTREAM_NAME or digest(original) != UPSTREAM_SHA256:
        raise ValueError("original is not the pinned upstream wheel")
    before, native = members(original)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # wheel tags requires the input filename to agree with its internal tag.
    # This renamed copy is byte-identical to the original published archive.
    alias = output / UPSTREAM_NAME.replace("manylinux2014_aarch64", "manylinux2014_sbsa")
    shutil.copyfile(original, alias)
    if digest(alias) != UPSTREAM_SHA256:
        raise ValueError("copy changed archive bytes")
    repaired = output / tags(str(alias), platform_tags="linux_aarch64")
    after, after_native = members(repaired)
    if before.keys() != after.keys() or native != after_native:
        raise ValueError("repair changed member inventory")
    changed = sorted(name for name in before if before[name] != after[name])
    expected = sorted("nvidia_cusparselt_cu13-0.8.1.dist-info/" + name for name in ("WHEEL", "RECORD"))
    if changed != expected:
        raise ValueError(f"repair changed payload: {changed}")
    manifest = {"schema_version": 1, "status": "metadata_repair_only",
                "original": {"path": original.name, "sha256": UPSTREAM_SHA256},
                "repaired": {"path": repaired.name, "sha256": digest(repaired)},
                "scorer_sha256": digest(Path(__file__)),
                "changed_members": changed, "aarch64_elf_members": native,
                "members": after, "payload_byte_identical": True}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(repaired)


if __name__ == "__main__":
    main()
