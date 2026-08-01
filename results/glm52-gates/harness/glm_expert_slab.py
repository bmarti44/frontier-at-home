#!/usr/bin/env python3
"""Build or verify a byte-identical contiguous GLM routed-expert sidecar.

The source GGUF is never modified. Each sidecar record is gate|up|down for one
layer/expert, 4096-aligned and SHA-256 protected. The fixed binary index is
consumed directly by the default-off CUDA loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
from typing import BinaryIO, NamedTuple


MAGIC = b"GLM52SLB"
VERSION = 1
ALIGNMENT = 4096
HEADER = struct.Struct("<8sIIQ32sIIQ")
RECORD = struct.Struct("<IIQQQQ32s8x")
PROBE_BYTES = 1 << 20
TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
GGML = {
    0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32), 6: (22, 32),
    7: (24, 32), 8: (34, 32), 9: (36, 32), 10: (84, 256),
    11: (110, 256), 12: (144, 256), 13: (176, 256),
    14: (210, 256), 15: (292, 256), 16: (66, 256),
    17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (18, 32),
    21: (110, 256), 22: (82, 256), 23: (136, 256), 24: (1, 1),
    25: (2, 1), 26: (4, 1), 27: (8, 1), 28: (8, 1),
    29: (56, 256), 30: (2, 1),
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
    return (value + alignment - 1) // alignment * alignment


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
        for _ in range(count):
            skip_value(stream, element_type)
    elif value_type in SCALAR:
        read_exact(stream, SCALAR[value_type])
    else:
        raise ValueError(f"unsupported GGUF metadata type {value_type}")


def parse_expert_tensors(path: Path) -> tuple[int, dict[int, dict[str, Tensor]]]:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        if read_exact(stream, 4) != b"GGUF":
            raise ValueError("source is not GGUF")
        version, tensor_count, kv_count = read_u32(stream), read_u64(stream), read_u64(stream)
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")
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
        data_start = align_up(stream.tell(), alignment)

    tensors: dict[int, dict[str, Tensor]] = {}
    for name, dims, data_type, relative_offset in raw:
        match = TENSOR_RE.match(name)
        if not match:
            continue
        if data_type not in GGML or len(dims) != 3:
            raise ValueError(f"unsupported routed tensor {name}")
        block_bytes, block_elements = GGML[data_type]
        elements = dims[0] * dims[1] * dims[2]
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


def plan_records(path: Path) -> tuple[int, bytes, list[PlannedRecord], int]:
    model_size, tensors = parse_expert_tensors(path)
    count = sum(next(iter(parts.values())).experts for parts in tensors.values())
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


def build(source_path: Path, output_path: Path) -> dict[str, object]:
    model_size, probe, plans, final_size = plan_records(source_path)
    temporary = output_path.with_name(output_path.name + ".partial")
    if temporary.exists() or output_path.exists():
        raise ValueError("refusing to overwrite an existing slab or partial file")
    records: list[bytes] = []
    try:
        with source_path.open("rb") as source, temporary.open("xb+") as output:
            output.truncate(final_size)
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
                MAGIC, VERSION, ALIGNMENT, model_size, probe,
                len(records), RECORD.size, align_up(HEADER.size + len(records) * RECORD.size),
            ))
            output.seek(HEADER.size)
            for record in records:
                output.write(record)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"records": len(records), "bytes": final_size, "model_bytes": model_size}


def load_index(path: Path) -> tuple[tuple[object, ...], list[tuple[object, ...]]]:
    with path.open("rb") as stream:
        header = HEADER.unpack(read_exact(stream, HEADER.size))
        magic, version, alignment, _, _, count, record_size, data_offset = header
        if magic != MAGIC or version != VERSION or alignment != ALIGNMENT:
            raise ValueError("invalid expert slab header")
        if record_size != RECORD.size or data_offset != align_up(HEADER.size + count * RECORD.size):
            raise ValueError("invalid expert slab index geometry")
        records = [RECORD.unpack(read_exact(stream, RECORD.size)) for _ in range(count)]
    return header, records


def verify(source_path: Path, slab_path: Path) -> dict[str, object]:
    model_size, probe, plans, final_size = plan_records(source_path)
    header, records = load_index(slab_path)
    if header[3] != model_size or header[4] != probe:
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
    return {"records": len(records), "bytes": final_size, "verified": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "build", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("source", type=Path)
        if command != "plan":
            child.add_argument("slab", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            model_size, _, records, final_size = plan_records(args.source)
            result = {"records": len(records), "bytes": final_size,
                      "model_bytes": model_size}
        elif args.command == "build":
            result = build(args.source, args.slab)
        else:
            result = verify(args.source, args.slab)
    except (OSError, ValueError, struct.error) as error:
        print(json.dumps({"verdict": "FAIL", "error": str(error)}))
        return 1
    print(json.dumps({"verdict": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
