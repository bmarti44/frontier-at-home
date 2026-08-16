#!/usr/bin/env python3
"""Offline, streaming requantization of selected dense tensors in a GGUF v3 file.

The default plan selects GLM attention, dense/shared FFN, embedding, and output
weights stored as Q8_0 and targets Q4_K. Routed experts, routers, norms, and
biases are hard exclusions regardless of user patterns.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile
from typing import BinaryIO

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = ROOT / "results" / "glm52-gates" / "harness" / "gguf-py"
if os.fspath(GGUF_PY) not in sys.path:
    sys.path.insert(0, os.fspath(GGUF_PY))

try:
    from gguf import quants
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
except ImportError as error:  # pragma: no cover - installation failure path
    raise SystemExit(f"cannot import vendored gguf-py from {GGUF_PY}: {error}")


MAX_TENSORS = 200_000
MAX_KV = 100_000
MAX_DIMS = 4
MAX_STRING_BYTES = 128 << 20
MAX_ARRAY_ELEMENTS = 2_000_000
MAX_METADATA_DEPTH = 8
COPY_CHUNK = 8 << 20
QUANT_CHUNK = 64 << 20
SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
                6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
INTEGER_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I",
                   5: "<i", 10: "<Q", 11: "<q"}
DEFAULT_INCLUDE = re.compile(
    r"^(?:token_embd|output)\.weight$|^blk\.\d+\.(?:attn_|ffn_)[^.]+\.weight$"
)


@dataclass(frozen=True)
class Metadata:
    key: str
    value_type: int
    raw: bytes
    value: object | None


@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    relative_offset: int
    offset: int
    nbytes: int


@dataclass(frozen=True)
class GGUF:
    path: Path
    file_size: int
    alignment: int
    data_start: int
    metadata: tuple[Metadata, ...]
    tensors: tuple[Tensor, ...]


@dataclass(frozen=True)
class Plan:
    source: GGUF
    converted: frozenset[str]
    source_types: frozenset[int]
    target_type: int
    new_offsets: dict[str, int]
    data_start: int
    final_size: int
    source_identity: tuple[int, int, int, int]


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("invalid GGUF alignment")
    return (value + alignment - 1) // alignment * alignment


def read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(
            f"short GGUF read at {stream.tell()}: wanted {size}, got {len(data)}"
        )
    return data


def read_u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(stream, 4))[0]


def read_u64(stream: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(stream, 8))[0]


def read_string(stream: BinaryIO, budget: dict[str, int]) -> str:
    size = read_u64(stream)
    if size > (1 << 20):
        raise ValueError(f"implausible GGUF string length {size}")
    budget["strings"] += size
    if budget["strings"] > MAX_STRING_BYTES:
        raise ValueError("GGUF aggregate string budget exceeded")
    try:
        return read_exact(stream, size).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("GGUF contains invalid UTF-8") from error


def read_value(stream: BinaryIO, value_type: int, budget: dict[str, int],
               depth: int = 0) -> object | None:
    if depth > MAX_METADATA_DEPTH:
        raise ValueError("GGUF metadata nesting budget exceeded")
    if value_type == 8:
        return read_string(stream, budget)
    if value_type == 9:
        element_type, count = read_u32(stream), read_u64(stream)
        budget["arrays"] += count
        if count > 1_000_000 or budget["arrays"] > MAX_ARRAY_ELEMENTS:
            raise ValueError("GGUF metadata array budget exceeded")
        for _ in range(count):
            read_value(stream, element_type, budget, depth + 1)
        return None
    size = SCALAR_BYTES.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF metadata type {value_type}")
    raw = read_exact(stream, size)
    fmt = INTEGER_FORMATS.get(value_type)
    return struct.unpack(fmt, raw)[0] if fmt else None


def qtype_from_name(name: str) -> GGMLQuantizationType:
    try:
        return GGMLQuantizationType[name.upper()]
    except KeyError as error:
        choices = ", ".join(item.name for item in GGMLQuantizationType)
        raise ValueError(f"unknown GGML type {name!r}; known types: {choices}") from error


def geometry(ggml_type: int) -> tuple[int, int]:
    try:
        return GGML_QUANT_SIZES[GGMLQuantizationType(ggml_type)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unsupported GGML tensor type {ggml_type}") from error


def tensor_elements(name: str, dims: tuple[int, ...]) -> int:
    elements = 1
    for dim in dims:
        if dim <= 0 or dim > (1 << 40):
            raise ValueError(f"invalid dimension for {name}")
        elements *= dim
        if elements > (1 << 63):
            raise ValueError(f"tensor too large: {name}")
    return elements


def tensor_nbytes(name: str, dims: tuple[int, ...], ggml_type: int) -> int:
    block_elements, block_bytes = geometry(ggml_type)
    elements = tensor_elements(name, dims)
    if dims[0] % block_elements or elements % block_elements:
        raise ValueError(f"tensor {name} crosses a quantization block boundary")
    return elements // block_elements * block_bytes


def stream_identity(stream: BinaryIO) -> tuple[int, int, int, int]:
    value = os.fstat(stream.fileno())
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("source descriptor is not a regular file")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


@contextmanager
def open_regular(path: Path):
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"source is not a regular file: {path}") from error
    try:
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        stream_identity(stream)
        try:
            yield stream
        finally:
            stream.close()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def parse_gguf_stream(stream: BinaryIO, path: Path) -> GGUF:
    file_size = stream_identity(stream)[2]
    budget = {"strings": 0, "arrays": 0}
    stream.seek(0)
    if read_exact(stream, 4) != b"GGUF":
        raise ValueError("source is not a little-endian GGUF")
    version = read_u32(stream)
    tensor_count, kv_count = read_u64(stream), read_u64(stream)
    if version != 3:
        raise ValueError(f"unsupported GGUF version {version}")
    if tensor_count == 0 or tensor_count > MAX_TENSORS:
        raise ValueError(f"implausible GGUF tensor count {tensor_count}")
    if kv_count == 0 or kv_count > MAX_KV:
        raise ValueError(f"implausible GGUF metadata count {kv_count}")

    metadata: list[Metadata] = []
    keys: set[str] = set()
    alignment = 32
    for _ in range(kv_count):
        start = stream.tell()
        key = read_string(stream, budget)
        if key in keys:
            raise ValueError(f"duplicate metadata key {key}")
        keys.add(key)
        value_type = read_u32(stream)
        value = read_value(stream, value_type, budget)
        end = stream.tell()
        stream.seek(start)
        raw = read_exact(stream, end - start)
        metadata.append(Metadata(key, value_type, raw, value))
        if key == "general.alignment":
            if value_type != 4 or not isinstance(value, int):
                raise ValueError("general.alignment must be UINT32")
            alignment = value

    infos: list[tuple[str, tuple[int, ...], int, int]] = []
    names: set[str] = set()
    for _ in range(tensor_count):
        name = read_string(stream, budget)
        if name in names:
            raise ValueError(f"duplicate tensor {name}")
        names.add(name)
        n_dims = read_u32(stream)
        if n_dims == 0 or n_dims > MAX_DIMS:
            raise ValueError(f"implausible dimension count for {name}")
        dims = tuple(read_u64(stream) for _ in range(n_dims))
        infos.append((name, dims, read_u32(stream), read_u64(stream)))

    data_start = align_up(stream.tell(), alignment)
    tensors: list[Tensor] = []
    intervals: list[tuple[int, int, str]] = []
    for name, dims, ggml_type, relative_offset in infos:
        nbytes = tensor_nbytes(name, dims, ggml_type)
        offset = data_start + relative_offset
        if relative_offset % alignment:
            raise ValueError(f"unaligned tensor offset for {name}")
        if offset > file_size or nbytes > file_size - offset:
            raise ValueError(f"tensor outside GGUF: {name}")
        intervals.append((offset, offset + nbytes, name))
        tensors.append(Tensor(name, dims, ggml_type, relative_offset, offset, nbytes))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] > current[0]:
            raise ValueError(f"overlapping tensors {previous[2]} and {current[2]}")
    return GGUF(path, file_size, alignment, data_start, tuple(metadata), tuple(tensors))


def parse_gguf(path: Path) -> GGUF:
    with open_regular(path) as stream:
        return parse_gguf_stream(stream, path)


def hard_excluded(name: str) -> bool:
    lower = name.lower()
    return (
        "_exps" in lower
        or "ffn_gate_inp" in lower
        or "exp_probs_b" in lower
        or "norm" in lower
        or lower.endswith(".bias")
    )


def encode_tensor_info(tensor: Tensor, ggml_type: int, offset: int) -> bytes:
    name = tensor.name.encode("utf-8")
    return b"".join((
        struct.pack("<Q", len(name)), name, struct.pack("<I", len(tensor.dims)),
        b"".join(struct.pack("<Q", dim) for dim in tensor.dims),
        struct.pack("<IQ", ggml_type, offset),
    ))


def compile_pattern(value: str | None, default: re.Pattern[str] | None,
                    option: str) -> re.Pattern[str] | None:
    if value is None:
        return default
    try:
        return re.compile(value)
    except re.error as error:
        raise ValueError(f"invalid {option} regex: {error}") from error


def require_glm_architecture(source: GGUF) -> None:
    values = [item.value for item in source.metadata
              if item.key == "general.architecture"]
    if values != ["glm-dsa"]:
        raise ValueError("GGUF general.architecture must be exactly glm-dsa")


def make_plan(source_stream: BinaryIO, source_path: Path,
              type_names: list[str], target_name: str,
              include_pattern: str | None = None,
              exclude_pattern: str | None = None) -> Plan:
    identity = stream_identity(source_stream)
    source = parse_gguf_stream(source_stream, source_path)
    require_glm_architecture(source)
    source_types = frozenset(int(qtype_from_name(name)) for name in type_names)
    target_type = int(qtype_from_name(target_name))
    include = compile_pattern(include_pattern, DEFAULT_INCLUDE, "--include-pattern")
    exclude = compile_pattern(exclude_pattern, None, "--exclude-pattern")
    converted: set[str] = set()
    for tensor in source.tensors:
        selected = (
            tensor.ggml_type in source_types
            and include is not None and include.search(tensor.name) is not None
            and not hard_excluded(tensor.name)
            and (exclude is None or exclude.search(tensor.name) is None)
        )
        if selected:
            # Validate both row geometries before any output is opened. A
            # tensor whose rows do not fit the target's block geometry (e.g.
            # 128-wide MLA projections vs Q4_K's 256-element superblocks) is
            # skipped and copied unchanged rather than failing the run.
            try:
                tensor_nbytes(tensor.name, tensor.dims, target_type)
            except ValueError:
                print(f"skip (incompatible rows for target): {tensor.name}",
                      file=sys.stderr)
                continue
            converted.add(tensor.name)

    info_size = sum(len(encode_tensor_info(
        tensor, target_type if tensor.name in converted else tensor.ggml_type, 0
    )) for tensor in source.tensors)
    metadata_size = sum(len(item.raw) for item in source.metadata)
    data_start = align_up(24 + metadata_size + info_size, source.alignment)
    cursor = 0
    offsets: dict[str, int] = {}
    for tensor in source.tensors:
        cursor = align_up(cursor, source.alignment)
        offsets[tensor.name] = cursor
        output_type = target_type if tensor.name in converted else tensor.ggml_type
        cursor += tensor_nbytes(tensor.name, tensor.dims, output_type)
    return Plan(source, frozenset(converted), source_types, target_type, offsets,
                data_start, data_start + cursor, identity)


def quantizer_inventory() -> tuple[list[str], list[str]]:
    available: list[str] = []
    unavailable: list[str] = []
    for qtype in GGMLQuantizationType:
        if qtype not in GGML_QUANT_SIZES:
            continue
        block, _ = GGML_QUANT_SIZES[qtype]
        try:
            quants.quantize(np.zeros((1, block), dtype=np.float32), qtype)
        except (NotImplementedError, TypeError, ValueError, AssertionError):
            unavailable.append(qtype.name)
        else:
            available.append(qtype.name)
    return available, unavailable


def require_quantizer(ggml_type: int) -> None:
    qtype = GGMLQuantizationType(ggml_type)
    available, _ = quantizer_inventory()
    if qtype.name not in available:
        raise ValueError(
            f"vendored gguf-py has no {qtype.name} quantizer; available quantizers: "
            + ", ".join(available)
        )


def plan_result(plan: Plan) -> dict[str, object]:
    rows = []
    for tensor in plan.source.tensors:
        convert = tensor.name in plan.converted
        output_type = plan.target_type if convert else tensor.ggml_type
        rows.append({
            "name": tensor.name,
            "action": "convert" if convert else "copy",
            "from": GGMLQuantizationType(tensor.ggml_type).name,
            "to": GGMLQuantizationType(output_type).name,
            "source_bytes": tensor.nbytes,
            "projected_bytes": tensor_nbytes(tensor.name, tensor.dims, output_type),
        })
    available, unavailable = quantizer_inventory()
    return {
        "source_bytes": plan.source.file_size,
        "projected_file_size": plan.final_size,
        "converted_tensors": len(plan.converted),
        "tensors": rows,
        "available_quantizers": available,
        "unavailable_quantizers": unavailable,
    }


def copy_range(source: BinaryIO, output: BinaryIO, offset: int, size: int) -> None:
    source.seek(offset)
    remaining = size
    while remaining:
        chunk = read_exact(source, min(COPY_CHUNK, remaining))
        output.write(chunk)
        remaining -= len(chunk)


def iter_dequantized(source: BinaryIO, tensor: Tensor):
    qtype = GGMLQuantizationType(tensor.ggml_type)
    block, block_bytes = geometry(tensor.ggml_type)
    row_elements = tensor.dims[0]
    row_bytes = row_elements // block * block_bytes
    rows = tensor_elements(tensor.name, tensor.dims) // row_elements
    rows_per_chunk = max(1, QUANT_CHUNK // max(row_bytes, row_elements * 4))
    source.seek(tensor.offset)
    for first in range(0, rows, rows_per_chunk):
        count = min(rows_per_chunk, rows - first)
        raw = read_exact(source, count * row_bytes)
        packed = np.frombuffer(raw, dtype=np.uint8).reshape((count, row_bytes))
        yield quants.dequantize(packed, qtype)


def requantize_tensor(source: BinaryIO, output: BinaryIO, tensor: Tensor,
                      target_type: int) -> None:
    qtype = GGMLQuantizationType(target_type)
    for values in iter_dequantized(source, tensor):
        packed = quants.quantize(values, qtype)
        output.write(packed.tobytes(order="C"))


def write_requantized(plan: Plan, source: BinaryIO, destination: Path) -> None:
    if not destination.parent.is_dir():
        raise ValueError(f"destination directory does not exist: {destination.parent}")
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing destination: {destination}")
    if not plan.converted:
        raise ValueError("no tensors match the requantization selection")
    require_quantizer(plan.target_type)
    if stream_identity(source) != plan.source_identity:
        raise ValueError("source GGUF identity changed before requantization")
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
        )
        with os.fdopen(descriptor, "w+b") as output:
            output.write(b"GGUF")
            output.write(struct.pack("<IQQ", 3, len(plan.source.tensors),
                                     len(plan.source.metadata)))
            for item in plan.source.metadata:
                output.write(item.raw)
            for tensor in plan.source.tensors:
                output_type = (plan.target_type if tensor.name in plan.converted
                               else tensor.ggml_type)
                output.write(encode_tensor_info(
                    tensor, output_type, plan.new_offsets[tensor.name]
                ))
            output.write(b"\0" * (plan.data_start - output.tell()))
            for tensor in plan.source.tensors:
                wanted = plan.data_start + plan.new_offsets[tensor.name]
                if output.tell() > wanted:
                    raise ValueError("internal GGUF tensor offset overlap")
                output.write(b"\0" * (wanted - output.tell()))
                if tensor.name in plan.converted:
                    requantize_tensor(source, output, tensor, plan.target_type)
                else:
                    copy_range(source, output, tensor.offset, tensor.nbytes)
            if output.tell() != plan.final_size:
                raise ValueError("internal GGUF final size mismatch")
            output.flush()
            os.fsync(output.fileno())
        if stream_identity(source) != plan.source_identity:
            raise ValueError("source GGUF changed while requantizing")
        try:
            os.link(temporary_name, destination)
        except FileExistsError as error:
            raise ValueError(f"refusing to overwrite existing destination: {destination}") from error
        os.unlink(temporary_name)
        temporary_name = None
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def sha256_range(stream: BinaryIO, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    stream.seek(offset)
    remaining = size
    while remaining:
        chunk = read_exact(stream, min(COPY_CHUNK, remaining))
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def verify(plan: Plan, source: BinaryIO, destination: Path) -> dict[str, object]:
    target = parse_gguf(destination)
    if target.file_size != plan.final_size or target.data_start != plan.data_start:
        raise ValueError("destination header/file size differs from plan")
    if target.alignment != plan.source.alignment:
        raise ValueError("destination alignment differs from source")
    if [item.raw for item in target.metadata] != [item.raw for item in plan.source.metadata]:
        raise ValueError("destination metadata differs from source")
    if [item.name for item in target.tensors] != [item.name for item in plan.source.tensors]:
        raise ValueError("destination tensor names/order differ from source")
    target_by_name = {item.name: item for item in target.tensors}
    mse: dict[str, float] = {}
    copied = 0
    with destination.open("rb") as output:
        for original in plan.source.tensors:
            actual = target_by_name[original.name]
            expected_type = plan.target_type if original.name in plan.converted else original.ggml_type
            if (actual.dims != original.dims or actual.ggml_type != expected_type
                    or actual.relative_offset != plan.new_offsets[original.name]):
                raise ValueError(f"destination tensor info incorrect: {original.name}")
            if original.name not in plan.converted:
                if actual.nbytes != original.nbytes or sha256_range(
                    source, original.offset, original.nbytes
                ) != sha256_range(output, actual.offset, actual.nbytes):
                    raise ValueError(f"non-converted tensor bytes changed: {original.name}")
                copied += 1
                continue
            output.seek(actual.offset)
            squared_error = 0.0
            elements = 0
            target_block, target_bytes = geometry(actual.ggml_type)
            row_target_bytes = actual.dims[0] // target_block * target_bytes
            for source_values in iter_dequantized(source, original):
                count = source_values.shape[0]
                raw = read_exact(output, count * row_target_bytes)
                packed = np.frombuffer(raw, dtype=np.uint8).reshape((count, row_target_bytes))
                target_values = quants.dequantize(
                    packed, GGMLQuantizationType(actual.ggml_type)
                )
                delta = source_values.astype(np.float64) - target_values.astype(np.float64)
                squared_error += float(np.sum(delta * delta))
                elements += delta.size
            mse[original.name] = squared_error / elements
    return {"converted_mse": mse, "non_converted_byte_identical": copied,
            "bytes": target.file_size, "header_consistent": True}


def split_types(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("--types must name at least one GGML type")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--types", type=split_types, default=["Q8_0"])
    parser.add_argument("--to", default="Q4_K")
    parser.add_argument("--include-pattern")
    parser.add_argument("--exclude-pattern")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="verify an existing destination instead of writing")
    args = parser.parse_args(argv)
    if args.verify and args.dry_run:
        parser.error("--verify and --dry-run are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with open_regular(args.source) as source:
            plan = make_plan(source, args.source, args.types, args.to,
                             args.include_pattern, args.exclude_pattern)
            if args.verify:
                result = verify(plan, source, args.destination)
            else:
                result = plan_result(plan)
                if not args.dry_run:
                    write_requantized(plan, source, args.destination)
        print(json.dumps({"verdict": "PASS", **result}, sort_keys=True))
        return 0
    except (MemoryError, OSError, OverflowError, ValueError, struct.error) as error:
        print(json.dumps({"verdict": "FAIL", "error": str(error)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
