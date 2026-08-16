#!/usr/bin/env python3
"""Prune whole routed experts from a GLM-5.2 GGUF without decoding weights.

The parser and offset arithmetic deliberately follow
results/glm52-gates/harness/glm_expert_slab.py: GGUF v3, little endian,
alignment-relative tensor offsets, and the final tensor dimension as the
outermost (expert) dimension.
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


MAX_TENSORS = 200_000
MAX_KV = 100_000
MAX_DIMS = 4
MAX_STRING_BYTES = 128 << 20
MAX_ARRAY_ELEMENTS = 2_000_000
MAX_METADATA_DEPTH = 8
COPY_CHUNK = 8 << 20

# Copied from the vendored gguf-py GGML_QUANT_SIZES table. Entries are
# ggml_type: (elements per block, bytes per block).
GGML_TYPES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 40),
    10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98),
    19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 24: (1, 1),
    25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 34: (256, 54),
    35: (256, 66), 39: (32, 17), 40: (64, 36),
    41: (128, 18), 42: (64, 18),
}
SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
                6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
INTEGER_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    10: "<Q", 11: "<q",
}

ROUTED_RE = re.compile(
    r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$"
)
ROUTER_WEIGHT_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_inp(?:\.weight)?$")
ROUTER_BIAS_RE = re.compile(
    r"^blk\.(\d+)\.(?:exp_probs_b|ffn_gate_inp)(?:\.bias)$"
)
LEGACY_EXPERT_COUNT_RE = re.compile(r"(?:^|[._])n_experts?(?:$|[._])")


@dataclass(frozen=True)
class Metadata:
    key: str
    value_type: int
    raw: bytes
    value_offset: int
    value_end: int
    value: object | None


@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    relative_offset: int
    offset: int
    nbytes: int
    info_raw: bytes


@dataclass(frozen=True)
class GGUF:
    path: Path
    file_size: int
    alignment: int
    data_start: int
    metadata: tuple[Metadata, ...]
    tensors: tuple[Tensor, ...]


@dataclass(frozen=True)
class Slice:
    layer: int
    kind: str


@dataclass(frozen=True)
class Plan:
    source: GGUF
    keep: dict[int, tuple[int, ...]]
    expert_count: int
    new_expert_count: int
    slices: dict[str, Slice]
    new_dims: dict[str, tuple[int, ...]]
    new_offsets: dict[str, int]
    metadata_bytes: bytes
    data_start: int
    final_size: int
    layer_bytes: dict[int, tuple[int, int]]
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


def read_value(
    stream: BinaryIO, value_type: int, budget: dict[str, int], depth: int = 0
) -> object | None:
    if depth > MAX_METADATA_DEPTH:
        raise ValueError("GGUF metadata nesting budget exceeded")
    if value_type == 8:
        return read_string(stream, budget)
    if value_type == 9:
        element_type = read_u32(stream)
        count = read_u64(stream)
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
    if value_type in INTEGER_FORMATS:
        return struct.unpack(INTEGER_FORMATS[value_type], raw)[0]
    return None


def tensor_nbytes(name: str, dims: tuple[int, ...], ggml_type: int) -> int:
    geometry = GGML_TYPES.get(ggml_type)
    if geometry is None:
        raise ValueError(f"unsupported GGML tensor type {ggml_type} for {name}")
    elements = 1
    for dim in dims:
        if dim <= 0 or dim > (1 << 40):
            raise ValueError(f"invalid dimension for {name}")
        elements *= dim
        if elements > (1 << 63):
            raise ValueError(f"tensor too large: {name}")
    block_elements, block_bytes = geometry
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
    try:
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
        metadata_keys: set[str] = set()
        alignment = 32
        for _ in range(kv_count):
            start = stream.tell()
            key = read_string(stream, budget)
            if key in metadata_keys:
                raise ValueError(f"duplicate metadata key {key}")
            metadata_keys.add(key)
            value_type = read_u32(stream)
            value_offset = stream.tell() - start
            value = read_value(stream, value_type, budget)
            value_end = stream.tell() - start
            end = stream.tell()
            stream.seek(start)
            raw = read_exact(stream, end - start)
            metadata.append(
                Metadata(key, value_type, raw, value_offset, value_end, value)
            )
            if key == "general.alignment":
                if value_type != 4 or not isinstance(value, int):
                    raise ValueError("general.alignment must be UINT32")
                alignment = value

        raw_tensors: list[tuple[str, tuple[int, ...], int, int, bytes]] = []
        names: set[str] = set()
        for _ in range(tensor_count):
            start = stream.tell()
            name = read_string(stream, budget)
            n_dims = read_u32(stream)
            if n_dims == 0 or n_dims > MAX_DIMS:
                raise ValueError(f"implausible dimension count for {name}")
            dims = tuple(read_u64(stream) for _ in range(n_dims))
            ggml_type, relative_offset = read_u32(stream), read_u64(stream)
            end = stream.tell()
            stream.seek(start)
            info_raw = read_exact(stream, end - start)
            if name in names:
                raise ValueError(f"duplicate tensor {name}")
            names.add(name)
            raw_tensors.append((name, dims, ggml_type, relative_offset, info_raw))

        data_start = align_up(stream.tell(), alignment)
        tensors: list[Tensor] = []
        intervals: list[tuple[int, int, str]] = []
        for name, dims, ggml_type, relative_offset, info_raw in raw_tensors:
            nbytes = tensor_nbytes(name, dims, ggml_type)
            offset = data_start + relative_offset
            if relative_offset % alignment:
                raise ValueError(f"unaligned tensor offset for {name}")
            if offset > file_size or nbytes > file_size - offset:
                raise ValueError(f"tensor outside GGUF: {name}")
            intervals.append((offset, offset + nbytes, name))
            tensors.append(
                Tensor(name, dims, ggml_type, relative_offset, offset, nbytes, info_raw)
            )
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if previous[1] > current[0]:
                raise ValueError(
                    f"overlapping tensors {previous[2]} and {current[2]}"
                )
    except MemoryError as error:
        raise ValueError("GGUF parsing exceeded memory budget") from error
    return GGUF(
        path, file_size, alignment, data_start, tuple(metadata), tuple(tensors)
    )


def parse_gguf(path: Path) -> GGUF:
    with open_regular(path) as stream:
        return parse_gguf_stream(stream, path)


def architecture(model: GGUF) -> str:
    values = [m.value for m in model.metadata if m.key == "general.architecture"]
    if values != ["glm-dsa"]:
        raise ValueError("GGUF general.architecture must be exactly glm-dsa")
    return "glm-dsa"


def classify_tensors(
    model: GGUF,
) -> tuple[int, dict[int, dict[str, Tensor]], dict[str, Slice]]:
    routed: dict[int, dict[str, Tensor]] = {}
    routers: dict[int, dict[str, Tensor]] = {}
    slices: dict[str, Slice] = {}
    for tensor in model.tensors:
        match = ROUTED_RE.match(tensor.name)
        kind = ""
        if match:
            layer, kind = int(match.group(1)), match.group(2)
            if kind in routed.setdefault(layer, {}):
                raise ValueError(f"duplicate layer {layer} routed {kind} tensor")
            routed[layer][kind] = tensor
        else:
            match = ROUTER_WEIGHT_RE.match(tensor.name)
            if match:
                layer, kind = int(match.group(1)), "router_weight"
            else:
                match = ROUTER_BIAS_RE.match(tensor.name)
                if match:
                    layer, kind = int(match.group(1)), "router_bias"
        if match and kind:
            if kind.startswith("router"):
                if kind in routers.setdefault(layer, {}):
                    raise ValueError(f"duplicate layer {layer} {kind} tensor")
                routers[layer][kind] = tensor
            slices[tensor.name] = Slice(layer, kind)

    if not routed:
        raise ValueError("GGUF has no routed expert tensors")
    layers = set(routed)
    if set(routers) - layers:
        raise ValueError("router tensor exists outside the routed layer set")
    expert_counts: set[int] = set()
    for layer in sorted(layers):
        if set(routed[layer]) != {"gate", "up", "down"}:
            raise ValueError(f"layer {layer} lacks gate/up/down routed tensors")
        if "router_weight" not in routers.get(layer, {}):
            raise ValueError(f"layer {layer} lacks ffn_gate_inp router weight")
        for tensor in (*routed[layer].values(), *routers[layer].values()):
            if tensor.dims[-1] <= 0:
                raise ValueError(f"invalid expert dimension for {tensor.name}")
            expert_counts.add(tensor.dims[-1])
    if len(expert_counts) != 1:
        raise ValueError("routed/router tensors have nonuniform expert counts")
    expert_count = expert_counts.pop()
    if expert_count > 256:
        raise ValueError(f"source expert count {expert_count} exceeds 256")
    return expert_count, routed, slices


def load_keep(
    path: Path, layers: set[int], expert_count: int
) -> dict[int, tuple[int, ...]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read keep-list: {error}") from error
    if isinstance(value, list):
        raw = {layer: value for layer in layers}
    elif isinstance(value, dict) and set(value) == {"layers"} and isinstance(value["layers"], dict):
        raw = {}
        for key, ids in value["layers"].items():
            if not isinstance(key, str) or not key.isdigit() or str(int(key)) != key:
                raise ValueError(f"invalid layer key {key!r}")
            layer = int(key)
            if layer in raw:
                raise ValueError(f"duplicate layer {layer}")
            raw[layer] = ids
        if set(raw) != layers:
            missing = sorted(layers - set(raw))
            extra = sorted(set(raw) - layers)
            raise ValueError(f"keep-list layer mismatch: missing={missing}, extra={extra}")
    else:
        raise ValueError("keep-list must be a global list or {'layers': {...}}")

    result: dict[int, tuple[int, ...]] = {}
    counts: set[int] = set()
    for layer in sorted(layers):
        ids = raw[layer]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"layer {layer} keep-list must be a nonempty list")
        if any(type(item) is not int for item in ids):
            raise ValueError(f"layer {layer} expert ids must be integers")
        if ids != sorted(ids):
            raise ValueError(f"layer {layer} expert ids must be sorted")
        if len(ids) != len(set(ids)):
            raise ValueError(f"layer {layer} expert ids must be unique")
        if ids[0] < 0 or ids[-1] >= expert_count:
            raise ValueError(
                f"layer {layer} expert ids must be in [0,{expert_count - 1}]"
            )
        result[layer] = tuple(ids)
        counts.add(len(ids))
    if len(counts) != 1:
        raise ValueError("keep-list counts must be identical for every layer")
    return result


def encode_tensor_info(tensor: Tensor, dims: tuple[int, ...], offset: int) -> bytes:
    name = tensor.name.encode("utf-8")
    return b"".join(
        (
            struct.pack("<Q", len(name)), name, struct.pack("<I", len(dims)),
            b"".join(struct.pack("<Q", dim) for dim in dims),
            struct.pack("<IQ", tensor.ggml_type, offset),
        )
    )


def patched_metadata(model: GGUF, old_count: int, new_count: int) -> bytes:
    arch = architecture(model)
    allowed_key = f"{arch}.expert_count"
    output = bytearray()
    found = False
    for item in model.metadata:
        raw = bytearray(item.raw)
        if item.key != allowed_key and LEGACY_EXPERT_COUNT_RE.search(item.key):
            raise ValueError(
                f"ambiguous expert-count metadata key {item.key}; "
                f"only {allowed_key} is allowed"
            )
        if item.key == allowed_key:
            fmt = INTEGER_FORMATS.get(item.value_type)
            if fmt is None or item.value_end - item.value_offset != struct.calcsize(fmt):
                raise ValueError(f"expert-count metadata {item.key} is not an integer scalar")
            if item.value != old_count:
                raise ValueError(
                    f"expert-count metadata {item.key} is {item.value}, "
                    f"but tensor geometry is {old_count}"
                )
            try:
                raw[item.value_offset:item.value_end] = struct.pack(fmt, new_count)
            except struct.error as error:
                raise ValueError(f"new expert count does not fit {item.key}") from error
            found = True
        output.extend(raw)
    if not found:
        raise ValueError(f"GGUF must contain expert-count metadata {allowed_key}")
    return bytes(output)


def make_plan(
    source_stream: BinaryIO,
    source_path: Path,
    keep_path: Path,
    expect_layers: int | None = None,
) -> Plan:
    source_identity = stream_identity(source_stream)
    source = parse_gguf_stream(source_stream, source_path)
    expert_count, routed, slices = classify_tensors(source)
    layers = set(routed)
    if expect_layers is not None:
        if expect_layers <= 0:
            raise ValueError("--expect-layers must be a positive integer")
        first = min(layers)
        expected = set(range(first, first + expect_layers))
        if layers != expected:
            missing = sorted(expected - layers)
            extra = sorted(layers - expected)
            raise ValueError(
                f"routed layer topology mismatch: missing={missing}, extra={extra}"
            )
    keep = load_keep(keep_path, layers, expert_count)
    new_count = len(next(iter(keep.values())))
    metadata_bytes = patched_metadata(source, expert_count, new_count)
    new_dims: dict[str, tuple[int, ...]] = {}
    for tensor in source.tensors:
        dims = tensor.dims
        if tensor.name in slices:
            if tensor.nbytes % expert_count:
                raise ValueError(f"tensor {tensor.name} is not expert-contiguous")
            dims = (*dims[:-1], new_count)
            # Recompute to prove whole-expert surgery preserves block geometry.
            expected = tensor.nbytes // expert_count * new_count
            if tensor_nbytes(tensor.name, dims, tensor.ggml_type) != expected:
                raise ValueError(f"pruning {tensor.name} changes quant-block geometry")
        new_dims[tensor.name] = dims

    info_size = sum(
        len(encode_tensor_info(t, new_dims[t.name], 0)) for t in source.tensors
    )
    header_size = 24 + len(metadata_bytes) + info_size
    data_start = align_up(header_size, source.alignment)
    cursor = 0
    new_offsets: dict[str, int] = {}
    layer_bytes = {layer: [0, 0] for layer in layers}
    for tensor in source.tensors:
        cursor = align_up(cursor, source.alignment)
        new_offsets[tensor.name] = cursor
        sliced = slices.get(tensor.name)
        if sliced:
            per_expert = tensor.nbytes // expert_count
            kept = per_expert * new_count
            layer_bytes[sliced.layer][0] += kept
            layer_bytes[sliced.layer][1] += tensor.nbytes - kept
            cursor += kept
        else:
            cursor += tensor.nbytes
    return Plan(
        source, keep, expert_count, new_count, slices, new_dims, new_offsets,
        metadata_bytes, data_start, data_start + cursor,
        {layer: (values[0], values[1]) for layer, values in layer_bytes.items()},
        source_identity,
    )


def plan_result(plan: Plan) -> dict[str, object]:
    return {
        "source_bytes": plan.source.file_size,
        "final_file_size": plan.final_size,
        "source_experts": plan.expert_count,
        "kept_experts": plan.new_expert_count,
        "layers": {
            str(layer): {"bytes_kept": kept, "bytes_dropped": dropped}
            for layer, (kept, dropped) in sorted(plan.layer_bytes.items())
        },
    }


def copy_range(source: BinaryIO, output: BinaryIO, offset: int, size: int) -> None:
    source.seek(offset)
    remaining = size
    while remaining:
        chunk = read_exact(source, min(COPY_CHUNK, remaining))
        output.write(chunk)
        remaining -= len(chunk)


def write_pruned(plan: Plan, source: BinaryIO, destination: Path) -> None:
    if not destination.parent.is_dir():
        raise ValueError(f"destination directory does not exist: {destination.parent}")
    if stream_identity(source) != plan.source_identity:
        raise ValueError("source GGUF identity changed before pruning")
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
        )
        with os.fdopen(descriptor, "w+b") as output:
            output.write(b"GGUF")
            output.write(struct.pack("<IQQ", 3, len(plan.source.tensors), len(plan.source.metadata)))
            output.write(plan.metadata_bytes)
            for tensor in plan.source.tensors:
                output.write(encode_tensor_info(
                    tensor, plan.new_dims[tensor.name], plan.new_offsets[tensor.name]
                ))
            padding = plan.data_start - output.tell()
            if padding < 0:
                raise ValueError("internal GGUF header size mismatch")
            output.write(b"\0" * padding)
            for tensor in plan.source.tensors:
                wanted = plan.data_start + plan.new_offsets[tensor.name]
                if output.tell() > wanted:
                    raise ValueError("internal GGUF tensor offset overlap")
                output.write(b"\0" * (wanted - output.tell()))
                sliced = plan.slices.get(tensor.name)
                if not sliced:
                    copy_range(source, output, tensor.offset, tensor.nbytes)
                    continue
                per_expert = tensor.nbytes // plan.expert_count
                for expert in plan.keep[sliced.layer]:
                    copy_range(source, output, tensor.offset + expert * per_expert, per_expert)
            if output.tell() != plan.final_size:
                raise ValueError("internal GGUF final size mismatch")
            output.flush()
            os.fsync(output.fileno())
        if stream_identity(source) != plan.source_identity:
            raise ValueError("source GGUF changed while pruning")
        try:
            os.link(temporary_name, destination)
        except FileExistsError as error:
            raise ValueError(
                f"refusing to overwrite existing destination: {destination}"
            ) from error
        os.unlink(temporary_name)
        temporary_name = None
        try:
            directory_fd = os.open(
                destination.parent, os.O_RDONLY | os.O_DIRECTORY
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as error:
            raise ValueError(
                f"destination exists and is complete, but directory fsync failed; "
                f"publish durability is unknown: {destination}: {error}"
            ) from error
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
    if target.file_size != plan.final_size:
        raise ValueError("destination file size differs from plan")
    if target.data_start != plan.data_start:
        raise ValueError("destination data start differs from plan")
    if target.alignment != plan.source.alignment:
        raise ValueError("destination alignment differs from source")
    if len(target.tensors) != len(plan.source.tensors):
        raise ValueError("destination tensor count differs from source")
    target_by_name = {tensor.name: tensor for tensor in target.tensors}
    if list(target_by_name) != [tensor.name for tensor in plan.source.tensors]:
        raise ValueError("destination tensor names/order differ from source")

    source_meta = {item.key: item for item in plan.source.metadata}
    target_meta = {item.key: item for item in target.metadata}
    if list(target_meta) != [item.key for item in plan.source.metadata]:
        raise ValueError("destination metadata keys/order differ from source")
    arch = architecture(plan.source)
    allowed_key = f"{arch}.expert_count"
    updated_keys = 0
    for key, source_item in source_meta.items():
        target_item = target_meta[key]
        if source_item.value_type != target_item.value_type:
            raise ValueError(f"metadata type changed for {key}")
        if key == allowed_key:
            if target_item.value != plan.new_expert_count:
                raise ValueError(f"expert count not updated for {key}")
            updated_keys += 1
        elif source_item.raw != target_item.raw:
            raise ValueError(f"metadata changed unexpectedly for {key}")
    if updated_keys == 0:
        raise ValueError("no expert-count metadata was verified")

    expert_hashes = 0
    nonexpert_tensors = 0
    with destination.open("rb") as output:
        for original in plan.source.tensors:
            pruned = target_by_name[original.name]
            if pruned.relative_offset != plan.new_offsets[original.name]:
                raise ValueError(f"tensor offset incorrect for {original.name}")
            if pruned.ggml_type != original.ggml_type:
                raise ValueError(f"tensor type changed for {original.name}")
            if pruned.dims != plan.new_dims[original.name]:
                raise ValueError(f"tensor dimensions incorrect for {original.name}")
            sliced = plan.slices.get(original.name)
            if not sliced:
                if pruned.nbytes != original.nbytes or sha256_range(
                    source, original.offset, original.nbytes
                ) != sha256_range(output, pruned.offset, pruned.nbytes):
                    raise ValueError(f"non-expert tensor bytes changed: {original.name}")
                nonexpert_tensors += 1
                continue
            source_expert_bytes = original.nbytes // plan.expert_count
            target_expert_bytes = pruned.nbytes // plan.new_expert_count
            if source_expert_bytes != target_expert_bytes:
                raise ValueError(f"expert byte size changed for {original.name}")
            for output_expert, source_expert in enumerate(plan.keep[sliced.layer]):
                source_hash = sha256_range(
                    source,
                    original.offset + source_expert * source_expert_bytes,
                    source_expert_bytes,
                )
                output_hash = sha256_range(
                    output,
                    pruned.offset + output_expert * target_expert_bytes,
                    target_expert_bytes,
                )
                if source_hash != output_hash:
                    raise ValueError(
                        f"expert bytes changed: {original.name} expert {source_expert}"
                    )
                expert_hashes += 1
    return {
        "expert_sha256_checks": expert_hashes,
        "nonexpert_tensors": nonexpert_tensors,
        "metadata_expert_count_keys": updated_keys,
        "bytes": target.file_size,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--keep", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expect-layers", type=int)
    parser.add_argument(
        "--verify", action="store_true",
        help="interpret paths as SRC DST and verify an existing pruned GGUF",
    )
    args = parser.parse_args(argv)
    if len(args.paths) != 2:
        parser.error("exactly SRC.gguf and DST.gguf are required")
    if args.verify and args.dry_run:
        parser.error("--verify and --dry-run are mutually exclusive")
    args.source, args.destination = args.paths
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with open_regular(args.source) as source:
            plan = make_plan(
                source, args.source, args.keep, expect_layers=args.expect_layers
            )
            if args.verify:
                result = verify(plan, source, args.destination)
            else:
                result = plan_result(plan)
                if not args.dry_run:
                    write_pruned(plan, source, args.destination)
        print(json.dumps({"verdict": "PASS", **result}, sort_keys=True))
        return 0
    except (MemoryError, OSError, OverflowError, ValueError, struct.error) as error:
        print(json.dumps({"verdict": "FAIL", "error": str(error)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
