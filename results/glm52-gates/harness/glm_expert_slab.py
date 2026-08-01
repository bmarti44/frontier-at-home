#!/usr/bin/env python3
"""Build or verify a byte-identical contiguous GLM routed-expert sidecar.

The source GGUF is never modified. Each sidecar record is gate|up|down for one
layer/expert, 4096-aligned and SHA-256 protected. The fixed binary index is
consumed directly by the default-off CUDA loader.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import sys
import tempfile
from typing import BinaryIO, NamedTuple


MAGIC = b"GLM52SLB"
VERSION = 1
ALIGNMENT = 4096
HEADER = struct.Struct("<8sIIQ32s32sIIQ")
RECORD = struct.Struct("<IIQQQQ32s8x")
PROBE_BYTES = 1 << 20
MAX_TENSORS = 200_000
MAX_KV = 100_000
MAX_ARRAY = 1_000_000
MAX_LAYERS = 80
MAX_EXPERTS = 256
MAX_RECORDS = MAX_LAYERS * MAX_EXPERTS
MAX_ARTIFACT_BYTES = 256 << 30
LARGE_BUILD_BYTES = 1 << 30
FREE_SPACE_FLOOR = 20 << 30
TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
# Accept only types the engine permits for routed experts. Keeping an
# independent copy small makes drift obvious and testable.
GGML = {
    8: (34, 32),       # Q8_0
    10: (84, 256),     # Q2_K
    12: (144, 256),    # Q4_K
    13: (176, 256),    # Q5_K
    14: (210, 256),    # Q6_K
    16: (66, 256),     # IQ2_XXS
}
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
          10: 8, 11: 8, 12: 8}


class Tensor(NamedTuple):
    offset: int
    nbytes: int
    experts: int


class PlannedRecord(NamedTuple):
    layer: int
    expert: int
    offset: int
    gate_offset: int
    gate_bytes: int
    up_offset: int
    down_offset: int
    down_bytes: int

    @property
    def payload_bytes(self) -> int:
        return 2 * self.gate_bytes + self.down_bytes


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("invalid alignment geometry")
    return (value + alignment - 1) // alignment * alignment


def regular_identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"not a regular file: {path}")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"short read at {stream.tell()}: wanted {size}, got {len(value)}")
    return value


def read_u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(stream, 4))[0]


def read_u64(stream: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(stream, 8))[0]


def read_string(stream: BinaryIO) -> str:
    size = read_u64(stream)
    if size > 1 << 20:
        raise ValueError(f"implausible GGUF string length {size}")
    return read_exact(stream, size).decode("utf-8")


def skip_value(stream: BinaryIO, value_type: int) -> None:
    if value_type == 8:
        read_exact(stream, read_u64(stream))
    elif value_type == 9:
        element_type, count = read_u32(stream), read_u64(stream)
        if count > MAX_ARRAY:
            raise ValueError(f"implausible GGUF array count {count}")
        for _ in range(count):
            skip_value(stream, element_type)
    elif value_type in SCALAR:
        read_exact(stream, SCALAR[value_type])
    else:
        raise ValueError(f"unsupported GGUF metadata type {value_type}")


def parse_expert_tensors(path: Path) -> tuple[int, dict[int, dict[str, Tensor]]]:
    file_size = regular_identity(path)[2]
    with path.open("rb") as stream:
        if read_exact(stream, 4) != b"GGUF":
            raise ValueError("source is not GGUF")
        version, tensor_count, kv_count = read_u32(stream), read_u64(stream), read_u64(stream)
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")
        if tensor_count == 0 or tensor_count > MAX_TENSORS:
            raise ValueError(f"implausible GGUF tensor count {tensor_count}")
        if kv_count > MAX_KV:
            raise ValueError(f"implausible GGUF metadata count {kv_count}")
        alignment = 32
        for _ in range(kv_count):
            key, value_type = read_string(stream), read_u32(stream)
            if key == "general.alignment" and value_type == 4:
                alignment = read_u32(stream)
            else:
                skip_value(stream, value_type)
        raw: list[tuple[str, list[int], int, int]] = []
        for _ in range(tensor_count):
            name, dimensions = read_string(stream), read_u32(stream)
            dims = [read_u64(stream) for _ in range(dimensions)]
            raw.append((name, dims, read_u32(stream), read_u64(stream)))
        if alignment > ALIGNMENT:
            raise ValueError(f"unsupported GGUF alignment {alignment}")
        data_start = align_up(stream.tell(), alignment)

    tensors: dict[int, dict[str, Tensor]] = {}
    for name, dims, data_type, relative_offset in raw:
        match = TENSOR_RE.match(name)
        if not match:
            continue
        if data_type not in GGML or len(dims) != 3:
            raise ValueError(f"unsupported routed tensor {name}")
        if int(match.group(1)) >= MAX_LAYERS or not dims[2] or dims[2] > MAX_EXPERTS:
            raise ValueError(f"unsupported routed tensor geometry {name}")
        block_bytes, block_elements = GGML[data_type]
        elements = 1
        for dimension in dims:
            if dimension == 0 or dimension > (1 << 32):
                raise ValueError(f"invalid dimension for {name}")
            elements *= dimension
            if elements > (1 << 63):
                raise ValueError(f"routed tensor too large: {name}")
        if dims[0] % block_elements or elements % block_elements:
            raise ValueError(f"invalid quantized dimensions for {name}")
        nbytes = elements // block_elements * block_bytes
        offset = data_start + relative_offset
        if offset > file_size or nbytes > file_size - offset:
            raise ValueError(f"routed tensor outside source file: {name}")
        layer, part = int(match.group(1)), match.group(2)
        if part in tensors.setdefault(layer, {}):
            raise ValueError(f"duplicate routed tensor {name}")
        tensors[layer][part] = Tensor(offset, nbytes, dims[2])
    if not tensors:
        raise ValueError("GGUF has no routed expert tensors")
    return file_size, tensors


def model_probe(path: Path) -> bytes:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", size))
    with path.open("rb") as stream:
        digest.update(read_exact(stream, min(PROBE_BYTES, size)))
        if size > PROBE_BYTES:
            stream.seek(max(0, size - PROBE_BYTES))
            digest.update(read_exact(stream, min(PROBE_BYTES, size)))
    return digest.digest()


def file_sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.digest()


def plan_records(path: Path) -> tuple[int, bytes, list[PlannedRecord], int]:
    model_size, tensors = parse_expert_tensors(path)
    count = sum(next(iter(parts.values())).experts for parts in tensors.values())
    if count == 0 or count > MAX_RECORDS:
        raise ValueError(f"invalid expert slab record count {count}")
    data_offset = align_up(HEADER.size + count * RECORD.size)
    output_offset = data_offset
    records: list[PlannedRecord] = []
    for layer, parts in sorted(tensors.items()):
        if set(parts) != {"gate", "up", "down"}:
            raise ValueError(f"layer {layer} lacks gate/up/down routed tensors")
        gate, up, down = parts["gate"], parts["up"], parts["down"]
        if gate.experts != up.experts or gate.experts != down.experts:
            raise ValueError(f"layer {layer} expert counts differ")
        if gate.nbytes != up.nbytes:
            raise ValueError(f"layer {layer} gate/up sizes differ")
        if gate.nbytes % gate.experts or down.nbytes % down.experts:
            raise ValueError(f"layer {layer} tensor is not expert-contiguous")
        gate_bytes, down_bytes = gate.nbytes // gate.experts, down.nbytes // down.experts
        for expert in range(gate.experts):
            records.append(PlannedRecord(
                layer, expert, output_offset,
                gate.offset + expert * gate_bytes, gate_bytes,
                up.offset + expert * gate_bytes,
                down.offset + expert * down_bytes, down_bytes,
            ))
            output_offset = align_up(output_offset + records[-1].payload_bytes)
            if output_offset > MAX_ARTIFACT_BYTES:
                raise ValueError("expert slab exceeds maximum artifact size")
    return model_size, model_probe(path), records, output_offset


def copy_range(source: BinaryIO, output: BinaryIO, offset: int, size: int,
               digest: hashlib._Hash, chunk_size: int = 8 << 20) -> None:
    source.seek(offset)
    remaining = size
    while remaining:
        data = read_exact(source, min(chunk_size, remaining))
        output.write(data)
        digest.update(data)
        remaining -= len(data)


def build(source_path: Path, output_path: Path, *, allow_large: bool = False) -> dict[str, object]:
    source_identity = regular_identity(source_path)
    model_size, probe, plans, final_size = plan_records(source_path)
    model_digest = file_sha256(source_path)
    if regular_identity(source_path) != source_identity:
        raise ValueError("source model changed during planning")
    if final_size > LARGE_BUILD_BYTES and not allow_large:
        raise ValueError("large build requires --owner-approved-large-build")
    output_path.parent.mkdir(parents=False, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("refusing to overwrite an existing slab")
    free = shutil.disk_usage(output_path.parent).free
    if free < final_size + FREE_SPACE_FLOOR:
        raise ValueError("insufficient disk space for slab plus safety floor")
    lock_path = output_path.with_name(output_path.name + ".lock")
    lock_flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock_fd = os.open(lock_path, lock_flags, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    records: list[bytes] = []
    published = False
    try:
        if output_path.exists() or output_path.is_symlink():
            raise ValueError("refusing to overwrite an existing slab")
        with source_path.open("rb") as source, os.fdopen(temporary_fd, "w+b") as output:
            temporary_fd = -1
            if not hasattr(os, "posix_fallocate"):
                raise ValueError("filesystem allocation preflight is unavailable")
            os.posix_fallocate(output.fileno(), 0, final_size)
            for plan in plans:
                output.seek(plan.offset)
                digest = hashlib.sha256()
                copy_range(source, output, plan.gate_offset, plan.gate_bytes, digest)
                copy_range(source, output, plan.up_offset, plan.gate_bytes, digest)
                copy_range(source, output, plan.down_offset, plan.down_bytes, digest)
                records.append(RECORD.pack(
                    plan.layer, plan.expert, plan.offset, plan.payload_bytes,
                    plan.gate_bytes, plan.down_bytes, digest.digest(),
                ))
            output.seek(0)
            output.write(HEADER.pack(
                MAGIC, VERSION, ALIGNMENT, model_size, probe, model_digest,
                len(records), RECORD.size, align_up(HEADER.size + len(records) * RECORD.size),
            ))
            output.seek(HEADER.size)
            for record in records:
                output.write(record)
            output.flush()
            os.fsync(output.fileno())
        if regular_identity(source_path) != source_identity:
            raise ValueError("source model changed during build")
        verify(source_path, temporary)
        os.link(temporary, output_path, follow_symlinks=False)
        published = True
        verify(source_path, output_path)
        temporary.unlink()
        directory_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary.exists():
            if published and output_path.exists():
                temporary_stat = temporary.stat(follow_symlinks=False)
                output_stat = output_path.stat(follow_symlinks=False)
                if (temporary_stat.st_dev, temporary_stat.st_ino) == (
                    output_stat.st_dev, output_stat.st_ino
                ):
                    output_path.unlink()
            temporary.unlink()
        raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return {"records": len(records), "bytes": final_size, "model_bytes": model_size}


def load_index(path: Path) -> tuple[tuple[object, ...], list[tuple[object, ...]]]:
    file_size = regular_identity(path)[2]
    with path.open("rb") as stream:
        header = HEADER.unpack(read_exact(stream, HEADER.size))
        magic, version, alignment, _, _, _, count, record_size, data_offset = header
        if magic != MAGIC or version != VERSION or alignment != ALIGNMENT:
            raise ValueError("invalid expert slab header")
        if record_size != RECORD.size or data_offset != align_up(HEADER.size + count * RECORD.size):
            raise ValueError("invalid expert slab index geometry")
        if count == 0 or count > MAX_RECORDS or HEADER.size + count * RECORD.size > file_size:
            raise ValueError("invalid expert slab record count")
        records = [RECORD.unpack(read_exact(stream, RECORD.size)) for _ in range(count)]
    return header, records


def verify(source_path: Path, slab_path: Path) -> dict[str, object]:
    source_identity = regular_identity(source_path)
    slab_identity = regular_identity(slab_path)
    model_size, probe, plans, final_size = plan_records(source_path)
    header, records = load_index(slab_path)
    if header[3] != model_size or header[4] != probe or header[5] != file_sha256(source_path):
        raise ValueError("expert slab model identity mismatch")
    if slab_path.stat().st_size != final_size or len(records) != len(plans):
        raise ValueError("expert slab size or record count mismatch")
    with source_path.open("rb") as source, slab_path.open("rb") as slab:
        for plan, record in zip(plans, records, strict=True):
            layer, expert, offset, payload_bytes, gate_bytes, down_bytes, expected = record
            identity = (layer, expert, offset, payload_bytes, gate_bytes, down_bytes)
            wanted = (plan.layer, plan.expert, plan.offset, plan.payload_bytes,
                      plan.gate_bytes, plan.down_bytes)
            if identity != wanted:
                raise ValueError("expert slab index record mismatch")
            slab.seek(offset)
            payload = read_exact(slab, payload_bytes)
            if hashlib.sha256(payload).digest() != expected:
                raise ValueError("expert slab record checksum mismatch")
            source_bytes = bytearray()
            for part_offset, part_bytes in (
                (plan.gate_offset, plan.gate_bytes),
                (plan.up_offset, plan.gate_bytes),
                (plan.down_offset, plan.down_bytes),
            ):
                source.seek(part_offset)
                source_bytes.extend(read_exact(source, part_bytes))
            if payload != source_bytes:
                raise ValueError("expert slab payload differs from GGUF")
    if regular_identity(source_path) != source_identity or regular_identity(slab_path) != slab_identity:
        raise ValueError("source or slab changed during verification")
    return {"records": len(records), "bytes": final_size, "verified": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "build", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("source", type=Path)
        if command != "plan":
            child.add_argument("slab", type=Path)
        if command == "build":
            child.add_argument("--owner-approved-large-build", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "plan":
            model_size, _, records, final_size = plan_records(args.source)
            result = {"records": len(records), "bytes": final_size,
                      "model_bytes": model_size}
        elif args.command == "build":
            result = build(
                args.source, args.slab,
                allow_large=args.owner_approved_large_build,
            )
        else:
            result = verify(args.source, args.slab)
    except (MemoryError, OSError, OverflowError, ValueError, struct.error) as error:
        print(json.dumps({"verdict": "FAIL", "error": str(error)}))
        return 1
    print(json.dumps({"verdict": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
