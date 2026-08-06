#!/usr/bin/env python3
"""Build or verify the exact 255-tensor GLM MTP proxy pack.

The pack is a compact, mmap-friendly copy of the frozen allowed tensor ranges.
It contains no target routed-expert tensor. Publication is exclusive and the
receipt is committed last after the pack and containing directory are fsynced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import BinaryIO


MAGIC = b"DS4MTPK1"
VERSION = 1
ALIGNMENT = 4096
HEADER = struct.Struct("<8sIIIIQQQ32s32s32s")
RECORD = struct.Struct("<64sIIII4Q4Q32s16s")
TYPE_IDS = {"F32": 1, "Q8_0": 2, "Q2_K": 3}
GROUPS = ("global_allowed", "mtp_allowed", "target_router_allowed")
EXPECTED_COUNTS = {
    "global_allowed": 3,
    "mtp_allowed": 27,
    "target_router_allowed": 225,
    "forbidden_target_experts": 225,
}


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_path(path: Path, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_fd(fd: int, size: int, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        block = os.pread(fd, min(chunk, size - offset), offset)
        if not block:
            raise OSError(f"short file read at {offset}")
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


def load_inventory(path: Path) -> tuple[dict, bytes, list[dict], list[dict]]:
    raw = path.read_bytes()
    inventory = json.loads(raw)
    if inventory.get("schema_version") != 1:
        raise ValueError("unexpected inventory schema")
    counts = inventory.get("counts")
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"inventory counts differ: {counts!r}")
    rows = [row for group in GROUPS for row in inventory[group]]
    forbidden = inventory["forbidden_target_experts"]
    if len(rows) != 255 or len(forbidden) != 225:
        raise ValueError("inventory cardinality differs")
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate allowed tensor name")
    return inventory, raw, rows, forbidden


def validate_source_ranges(rows: list[dict], forbidden: list[dict], model_size: int) -> None:
    def checked(row: dict) -> tuple[int, int]:
        start = int(row["offset"])
        length = int(row["bytes"])
        end = start + length
        if start < 0 or length <= 0 or end <= start or end > model_size:
            raise ValueError(f"invalid model range for {row['name']}")
        return start, end

    allowed_ranges = []
    for row in rows:
        if row["type"] not in TYPE_IDS or not 1 <= len(row["shape"]) <= 4:
            raise ValueError(f"unsupported tensor metadata for {row['name']}")
        allowed_ranges.append((*checked(row), row["name"]))
    forbidden_ranges = [(*checked(row), row["name"]) for row in forbidden]
    for ranges, label in ((allowed_ranges, "allowed"), (forbidden_ranges, "forbidden")):
        by_start = sorted(ranges)
        for left, right in zip(by_start, by_start[1:]):
            if left[1] > right[0]:
                raise ValueError(f"overlapping {label} ranges: {left[2]} / {right[2]}")
    forbidden_by_start = sorted(forbidden_ranges)
    for a0, a1, name in allowed_ranges:
        for f0, f1, forbidden_name in forbidden_by_start:
            if f0 >= a1:
                break
            if f1 > a0:
                raise ValueError(f"allowed {name} intersects forbidden {forbidden_name}")


def copy_and_hash(source_fd: int, output_fd: int, source_offset: int,
                  output_offset: int, length: int) -> bytes:
    digest = hashlib.sha256()
    copied = 0
    while copied < length:
        want = min(16 << 20, length - copied)
        block = os.pread(source_fd, want, source_offset + copied)
        if len(block) != want:
            raise OSError(f"short model read at {source_offset + copied}")
        digest.update(block)
        written = 0
        while written < len(block):
            count = os.pwrite(output_fd, block[written:], output_offset + copied + written)
            if count <= 0:
                raise OSError("short pack write")
            written += count
        copied += len(block)
    return digest.digest()


def encode_record(row: dict, blob_offset: int, blob_digest: bytes) -> bytes:
    name = row["name"].encode()
    if not name or len(name) >= 64 or b"\0" in name:
        raise ValueError(f"invalid tensor name {row['name']!r}")
    dims = [int(value) for value in row["shape"]] + [0] * (4 - len(row["shape"]))
    return RECORD.pack(
        name.ljust(64, b"\0"), len(name), TYPE_IDS[row["type"]],
        len(row["shape"]), 0, *dims,
        int(row["offset"]), int(row["bytes"]), blob_offset, int(row["bytes"]),
        blob_digest, b"\0" * 16,
    )


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def build(model: Path, inventory_path: Path, output: Path, receipt: Path) -> None:
    inventory, inventory_raw, rows, forbidden = load_inventory(inventory_path)
    source_fd = os.open(model, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    model_stat = os.fstat(source_fd)
    try:
        if not os.path.isfile(f"/proc/self/fd/{source_fd}") or model_stat.st_size != int(inventory["model_bytes"]):
            raise ValueError("model size differs from frozen inventory")
        if sha256_fd(source_fd, model_stat.st_size) != inventory["model_sha256"]:
            raise ValueError("model SHA-256 differs from frozen inventory")
        validate_source_ranges(rows, forbidden, model_stat.st_size)
    except BaseException:
        os.close(source_fd)
        raise
    header_bytes = align_up(HEADER.size + len(rows) * RECORD.size)
    offsets = []
    cursor = header_bytes
    for row in rows:
        cursor = align_up(cursor)
        offsets.append(cursor)
        cursor += int(row["bytes"])
    file_bytes = align_up(cursor)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    output.parent.mkdir(parents=True, exist_ok=True)
    out_fd = os.open(output, flags, 0o600)
    try:
        os.ftruncate(out_fd, file_bytes)
        encoded = []
        for index, (row, blob_offset) in enumerate(zip(rows, offsets), 1):
            digest = copy_and_hash(source_fd, out_fd, int(row["offset"]),
                                   blob_offset, int(row["bytes"]))
            encoded.append(encode_record(row, blob_offset, digest))
            print(f"[{index:03d}/255] {row['name']}", file=sys.stderr, flush=True)
        table = b"".join(encoded)
        table_sha = hashlib.sha256(table).digest()
        inventory_sha = hashlib.sha256(inventory_raw).digest()
        model_sha = bytes.fromhex(inventory["model_sha256"])
        header = HEADER.pack(
            MAGIC, VERSION, header_bytes, RECORD.size, len(rows), file_bytes,
            model_stat.st_size, file_bytes - header_bytes,
            model_sha, inventory_sha, table_sha,
        )
        os.pwrite(out_fd, table, HEADER.size)
        os.pwrite(out_fd, header, 0)
        os.fsync(out_fd)
    except BaseException:
        os.close(source_fd)
        os.close(out_fd)
        output.unlink(missing_ok=True)
        raise
    else:
        os.close(source_fd)
        os.close(out_fd)
    fsync_directory(output.parent)
    pack_sha = sha256_path(output)
    receipt_payload = {
        "schema": "glm52-mtp-proxy-pack-receipt-v1",
        "pack": str(output),
        "pack_bytes": file_bytes,
        "pack_sha256": pack_sha,
        "model_bytes": model_stat.st_size,
        "model_sha256": inventory["model_sha256"],
        "inventory_sha256": hashlib.sha256(inventory_raw).hexdigest(),
        "record_count": len(rows),
        "forbidden_record_count": len(forbidden),
        "header_bytes": header_bytes,
        "record_bytes": RECORD.size,
    }
    receipt_bytes = canonical_json(receipt_payload)
    receipt_fd = os.open(receipt, flags, 0o600)
    try:
        if os.write(receipt_fd, receipt_bytes) != len(receipt_bytes):
            raise OSError("short receipt write")
        os.fsync(receipt_fd)
    finally:
        os.close(receipt_fd)
    fsync_directory(receipt.parent)


def read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("short pack metadata")
    return data


def verify(pack: Path, inventory_path: Path, receipt: Path) -> dict:
    inventory, inventory_raw, rows, forbidden = load_inventory(inventory_path)
    expected_receipt = json.loads(receipt.read_bytes())
    with pack.open("rb", buffering=0) as handle:
        header = HEADER.unpack(read_exact(handle, HEADER.size))
        (magic, version, header_bytes, record_bytes, count, file_bytes,
         model_bytes, payload_bytes, model_sha, inventory_sha, table_sha) = header
        if (magic, version, record_bytes, count) != (MAGIC, VERSION, RECORD.size, 255):
            raise ValueError("pack header identity differs")
        if file_bytes != pack.stat().st_size or payload_bytes != file_bytes - header_bytes:
            raise ValueError("pack size fields differ")
        if model_bytes != inventory["model_bytes"] or model_sha.hex() != inventory["model_sha256"]:
            raise ValueError("pack model identity differs")
        if inventory_sha != hashlib.sha256(inventory_raw).digest():
            raise ValueError("pack inventory identity differs")
        table = read_exact(handle, count * record_bytes)
        if hashlib.sha256(table).digest() != table_sha:
            raise ValueError("record table hash differs")
        prior_end = header_bytes
        for index, (row, encoded) in enumerate(zip(rows, [table[i:i+record_bytes] for i in range(0, len(table), record_bytes)])):
            values = RECORD.unpack(encoded)
            name_raw, name_len, dtype, ndim, flags, *tail = values
            dims = tail[:4]
            original_offset, original_length, blob_offset, blob_length = tail[4:8]
            blob_sha, reserved = tail[8:]
            name = name_raw[:name_len].decode()
            expected_dims = tuple(row["shape"] + [0] * (4 - len(row["shape"])))
            if (name, dtype, ndim, flags, tuple(dims), original_offset, original_length, blob_length, reserved) != (
                row["name"], TYPE_IDS[row["type"]], len(row["shape"]), 0,
                expected_dims, row["offset"], row["bytes"], row["bytes"], b"\0" * 16,
            ):
                raise ValueError(f"record {index} metadata differs")
            if blob_offset % ALIGNMENT or blob_offset < prior_end:
                raise ValueError(f"record {index} blob order/alignment differs")
            handle.seek(prior_end)
            if any(read_exact(handle, blob_offset - prior_end)):
                raise ValueError(f"record {index} padding is nonzero")
            handle.seek(blob_offset)
            digest = hashlib.sha256()
            remaining = blob_length
            while remaining:
                block = read_exact(handle, min(16 << 20, remaining))
                digest.update(block)
                remaining -= len(block)
            if digest.digest() != blob_sha:
                raise ValueError(f"record {index} payload hash differs")
            prior_end = blob_offset + blob_length
        handle.seek(prior_end)
        if any(handle.read()):
            raise ValueError("trailing pack padding is nonzero")
    validate_source_ranges(rows, forbidden, inventory["model_bytes"])
    actual_sha = sha256_path(pack)
    recomputed_receipt = {
        "schema": "glm52-mtp-proxy-pack-receipt-v1",
        "pack": str(pack),
        "pack_bytes": file_bytes,
        "pack_sha256": actual_sha,
        "model_bytes": model_bytes,
        "model_sha256": model_sha.hex(),
        "inventory_sha256": inventory_sha.hex(),
        "record_count": count,
        "forbidden_record_count": len(forbidden),
        "header_bytes": header_bytes,
        "record_bytes": record_bytes,
    }
    if expected_receipt != recomputed_receipt:
        raise ValueError("pack receipt fields differ from independently recomputed values")
    return recomputed_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.output, args.inventory, args.receipt), sort_keys=True))
        return 0
    if args.model is None:
        parser.error("--model is required when building")
    build(args.model, args.inventory, args.output, args.receipt)
    print(json.dumps(verify(args.output, args.inventory, args.receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
