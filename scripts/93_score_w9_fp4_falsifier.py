#!/usr/bin/env python3
"""Score continuous-scale FP4 candidates on frozen real GLM-5.2 tensors."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import subprocess
import sys
from typing import Any

import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_RELATIVE = "scripts/93_score_w9_fp4_falsifier.py"
TEST_RELATIVE = "scripts/tests/test_w9_fp4_falsifier.py"
PLAN_RELATIVE = "results/glm52-gates/W9-fp4-falsifier-plan-v1.json"
LAUNCHER_RELATIVE = "results/glm52-gates/harness/w9_fp4_falsifier_v1.sh"
REVIEW_RELATIVE = "results/glm52-gates/W9-fp4-falsifier-review-r252.json"
FREEZE_RELATIVE = "results/glm52-gates/W9-fp4-falsifier-candidate2-freeze.json"
DRAND_VERIFIER = REPO / "scripts/89_verify_drand_receipt.mjs"
NODE = pathlib.Path("/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node")
GIT = pathlib.Path("/usr/bin/git")
MINIMUM_REVIEW_FLOOR = 6357189
RELAY_URLS = (
    "https://api.drand.sh",
    "https://api2.drand.sh",
    "https://api3.drand.sh",
)

LAYERS = (0, 2, 10, 26, 42, 58, 74, 77)
KV_ROWS = 8192
QUERY_ROWS = 128
QUERY_HEADS = 64
WIDTH = 512
SELECTED_CAPACITY = 2048
BLOCK_WIDTH = 32
MAXIMUM_RELATIVE_RMSE = 0.05
CANDIDATES = (
    "plain_e2m1_multistart_f32_scale",
    "hadamard_e2m1_multistart_f32_scale",
    "hadamard_e2m1_multistart_f32_scale_channel_correction",
)
CAPTURE_HASHES = {
    "kv.f32": "805b30d0c4ac89bd5cd361c455c3c6eb49d69d32b49d8afb935b5a75a7de33ed",
    "query.f32": "a9346a4d3a8fc8fd6915905cc8c7f2a595816b957713a1fbfa8a1cfe182d0a9b",
    "selected.u32": "fe9edd824633783800ff16f428c1b42bcebb15276a386b951fb0727897b0eba0",
    "selected-count.u32": "99b56dbff1dd5899c41016ba76257216a809143ccf13fa0bb119343b07c42cdc",
    "metadata.json": "ddf1d406b4c1c3164d292f2aed94e740d29c538e884139ed00eaecb517adef27",
    "W9_CAPTURE_COMPLETE": "228a01a07809971faed980c97cc0d0ddee97c421905e7f82da5855250db81b52",
}
CAPTURE_SIZES = {
    "kv.f32": 134217728,
    "query.f32": 134217728,
    "selected.u32": 8388608,
    "selected-count.u32": 4096,
    "metadata.json": 479,
    "W9_CAPTURE_COMPLETE": 20,
}
E2M1_LEVELS = np.array(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32,
)
E2M1_MIDPOINTS = (E2M1_LEVELS[:-1] + E2M1_LEVELS[1:]) / 2.0
E2M1_POSITIVE_LEVELS = np.array(
    [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32,
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            value.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value in {label}: {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON root must be an object: {label}")
    return parsed


class BoundInput:
    """Hash and consume one stable, no-follow regular-file generation."""

    def __init__(self, path: pathlib.Path, expected_size: int | None,
                 expected_sha256: str | None) -> None:
        self.path = pathlib.Path(path)
        self.expected_size = expected_size
        self.expected_sha256 = expected_sha256
        self.fd = -1
        self.initial: tuple[int, int, int, int, int] | None = None
        self.sha256 = ""

    @staticmethod
    def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def _hash(self) -> str:
        digest = hashlib.sha256()
        offset = 0
        while True:
            chunk = os.pread(self.fd, 8 << 20, offset)
            if not chunk:
                break
            digest.update(chunk)
            offset += len(chunk)
        return digest.hexdigest()

    def __enter__(self) -> "BoundInput":
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.fd = os.open(self.path, flags)
        try:
            info = os.fstat(self.fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"bound input must be a single-link regular file: {self.path}")
            if self.expected_size is not None and info.st_size != self.expected_size:
                raise ValueError(f"bound input size mismatch: {self.path}")
            self.initial = self._identity(info)
            self.sha256 = self._hash()
            if self.expected_sha256 is not None and self.sha256 != self.expected_sha256:
                raise ValueError(f"bound input digest mismatch: {self.path}")
            return self
        except BaseException:
            os.close(self.fd)
            self.fd = -1
            raise

    def read_bytes(self) -> bytes:
        if self.fd < 0 or self.initial is None:
            raise ValueError("bound input is not open")
        size = self.initial[2]
        result = bytearray()
        offset = 0
        while offset < size:
            chunk = os.pread(self.fd, min(8 << 20, size - offset), offset)
            if not chunk:
                raise ValueError(f"short bound input read: {self.path}")
            result.extend(chunk)
            offset += len(chunk)
        return bytes(result)

    def memmap(self, dtype: str, shape: tuple[int, ...]) -> np.memmap:
        if self.fd < 0:
            raise ValueError("bound input is not open")
        return np.memmap(f"/proc/self/fd/{self.fd}", dtype=dtype, mode="r", shape=shape)

    def verify_final(self) -> None:
        if self.fd < 0 or self.initial is None:
            raise ValueError("bound input is not open")
        descriptor = os.fstat(self.fd)
        try:
            pathname = os.stat(self.path, follow_symlinks=False)
        except FileNotFoundError as error:
            raise ValueError(f"bound input generation disappeared: {self.path}") from error
        if (not stat.S_ISREG(pathname.st_mode) or descriptor.st_nlink != 1 or
                pathname.st_nlink != 1 or self._identity(descriptor) != self.initial or
                self._identity(pathname) != self.initial or self._hash() != self.sha256):
            raise ValueError(f"bound input generation changed: {self.path}")

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if exc_type is None:
                self.verify_final()
        finally:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1


def json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    else:
        text = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    return text.encode("utf-8")


def exclusive_write(path: pathlib.Path, value: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(value):
            offset += os.write(fd, value[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def e2m1_quantize(rows: np.ndarray, block_width: int = BLOCK_WIDTH) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] % block_width:
        raise ValueError("row width must be divisible by block width")
    if block_width < 1:
        raise ValueError("block width must be positive")
    if not np.isfinite(array).all():
        raise ValueError("quantizer input must be finite")
    blocks = array.reshape(-1, block_width)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    best = np.zeros_like(blocks)
    best_sse = np.full((blocks.shape[0],), np.inf, dtype=np.float64)

    # Each amax-to-positive-code mapping is a legal continuous-scale start;
    # amax/6 is therefore included exactly. Scale 1 closes the concrete
    # round-251 counterexample and is useful for already normalized blocks.
    starts = [np.where(amax > 0, amax / level, 1.0).astype(np.float32)
              for level in E2M1_POSITIVE_LEVELS]
    starts.append(np.ones_like(amax, dtype=np.float32))
    for initial in starts:
        scale = initial
        for _ in range(8):
            normalized = blocks / scale
            codes = E2M1_LEVELS[
                np.searchsorted(E2M1_MIDPOINTS, normalized, side="left")]
            candidate = codes * scale
            sse = np.sum(
                np.square(candidate - blocks, dtype=np.float64), axis=1,
                dtype=np.float64,
            )
            improved = sse < best_sse
            if np.any(improved):
                best_sse[improved] = sse[improved]
                best[improved] = candidate[improved]
            numerator = np.sum(
                blocks.astype(np.float64) * codes.astype(np.float64), axis=1,
                keepdims=True,
            )
            denominator = np.sum(
                np.square(codes, dtype=np.float64), axis=1, keepdims=True,
                dtype=np.float64,
            )
            refined = scale.astype(np.float64)
            np.divide(numerator, denominator, out=refined, where=denominator > 0)
            scale = np.maximum(refined, np.finfo(np.float32).tiny).astype(np.float32)
    if not np.isfinite(best).all():
        raise ValueError("non-finite quantizer output")
    return best.reshape(array.shape)


def hadamard_rotate(rows: np.ndarray, signs: np.ndarray) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    sign_array = np.asarray(signs, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Hadamard input must be two-dimensional")
    width = array.shape[1]
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard width must be a power of two")
    if sign_array.shape != (width,) or not np.isin(sign_array, (-1.0, 1.0)).all():
        raise ValueError("Hadamard signs must be exactly +/-1")
    if not np.isfinite(array).all():
        raise ValueError("Hadamard input must be finite")
    output = np.ascontiguousarray(array * sign_array)
    step = 1
    while step < width:
        for start in range(0, width, step * 2):
            left = output[:, start:start + step].copy()
            right = output[:, start + step:start + 2 * step].copy()
            output[:, start:start + step] = left + right
            output[:, start + step:start + 2 * step] = left - right
        step *= 2
    output *= np.float32(1.0 / math.sqrt(width))
    return output


def _rank(seed: bytes, domain: bytes, index: int) -> bytes:
    return hashlib.sha256(
        b"GLM52-W9-FP4-RANK-V1\0" + seed + b"\0" + domain + b"\0"
        + index.to_bytes(8, "big")
    ).digest()


def split_indices(count: int, seed: bytes, domain: bytes) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if count < 2 or count % 2 or len(seed) != 32 or not domain:
        raise ValueError("split requires an even count, 32-byte seed, and domain")
    ordered = sorted(range(count), key=lambda index: (_rank(seed, domain, index), index))
    midpoint = count // 2
    return tuple(sorted(ordered[:midpoint])), tuple(sorted(ordered[midpoint:]))


def layer_signs(seed: bytes, layer: int, width: int = WIDTH) -> np.ndarray:
    if len(seed) != 32 or layer < 0 or width < 1:
        raise ValueError("invalid sign seed domain")
    values = []
    domain = b"signs/" + str(layer).encode("ascii")
    for index in range(width):
        values.append(1.0 if _rank(seed, domain, index)[0] & 1 else -1.0)
    return np.asarray(values, dtype=np.float32)


def fit_channel_correction(reference: np.ndarray, quantized: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(quantized, dtype=np.float32)
    if ref.ndim != 2 or ref.shape != candidate.shape or ref.shape[0] < 1:
        raise ValueError("correction arrays must have equal nonempty 2D shapes")
    if not np.isfinite(ref).all() or not np.isfinite(candidate).all():
        raise ValueError("correction arrays must be finite")
    numerator = np.sum(ref.astype(np.float64) * candidate.astype(np.float64), axis=0)
    denominator = np.sum(candidate.astype(np.float64) ** 2, axis=0)
    alpha = np.ones(ref.shape[1], dtype=np.float64)
    np.divide(numerator, denominator, out=alpha, where=denominator > 0)
    if not np.isfinite(alpha).all():
        raise ValueError("non-finite channel correction")
    return alpha.astype(np.float32)


def query_weighted_error(
    queries: np.ndarray,
    reference_keys: np.ndarray,
    candidate_keys: np.ndarray,
    selected: np.ndarray,
    selected_sentinel: int,
    heldout_queries: np.ndarray,
    heldout_keys: np.ndarray,
) -> dict[str, float | int]:
    query_array = np.asarray(queries, dtype=np.float32)
    reference = np.asarray(reference_keys, dtype=np.float32)
    candidate = np.asarray(candidate_keys, dtype=np.float32)
    selection = np.asarray(selected, dtype=np.uint32)
    query_mask = np.asarray(heldout_queries, dtype=np.bool_)
    key_mask = np.asarray(heldout_keys, dtype=np.bool_)
    if (query_array.ndim != 3 or reference.ndim != 2 or
            candidate.shape != reference.shape or query_array.shape[2] != reference.shape[1] or
            selection.ndim != 2 or selection.shape[0] != query_array.shape[0] or
            query_mask.shape != (query_array.shape[0],) or
            key_mask.shape != (reference.shape[0],)):
        raise ValueError("metric input shape mismatch")
    if not (np.isfinite(query_array).all() and np.isfinite(reference).all()
            and np.isfinite(candidate).all()):
        raise ValueError("metric inputs must be finite")
    numerator = 0.0
    denominator = 0.0
    pairs = 0
    query_rows_used = 0
    key_references = 0
    for query_index in np.flatnonzero(query_mask):
        raw_ids = selection[query_index]
        invalid = raw_ids[(raw_ids != selected_sentinel) & (raw_ids >= reference.shape[0])]
        if invalid.size:
            raise ValueError("selected key ID is out of range")
        ids = raw_ids[raw_ids != selected_sentinel].astype(np.int64, copy=False)
        if np.unique(ids).size != ids.size:
            raise ValueError("duplicate selected key ID")
        ids = ids[key_mask[ids]]
        if not ids.size:
            continue
        query = query_array[query_index]
        base_logits = query @ reference[ids].T
        error_logits = query @ (candidate[ids] - reference[ids]).T
        numerator += float(np.sum(np.square(error_logits, dtype=np.float64), dtype=np.float64))
        denominator += float(np.sum(np.square(base_logits, dtype=np.float64), dtype=np.float64))
        pairs += int(ids.size * query.shape[0])
        key_references += int(ids.size)
        query_rows_used += 1
    if pairs < 1 or denominator <= 0 or not math.isfinite(numerator + denominator):
        raise ValueError("held-out metric has no finite positive-denominator pairs")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "relative_rmse": math.sqrt(numerator / denominator),
        "pairs": pairs,
        "query_rows": query_rows_used,
        "key_references": key_references,
    }


def _verify_capture(inputs: dict[str, BoundInput]) -> dict[str, Any]:
    if inputs["W9_CAPTURE_COMPLETE"].read_bytes() != b"W9_CAPTURE_COMPLETE\n":
        raise ValueError("capture completion marker mismatch")
    metadata = strict_json_bytes(
        inputs["metadata.json"].read_bytes(), "capture metadata")
    expected = {
        "schema": "glm52-w9-real-capture-v1",
        "layers": list(LAYERS),
        "kv_rows_per_layer": KV_ROWS,
        "kv_width": WIDTH,
        "query_rows_per_layer": QUERY_ROWS,
        "query_heads": QUERY_HEADS,
        "query_width": WIDTH,
        "selected_capacity": SELECTED_CAPACITY,
        "sample_position_start": 0,
        "sample_position_stride": 64,
        "selected_padding_sentinel": 8193,
        "storage_padding_sentinel": 4294967295,
        "artifacts": {name: CAPTURE_SIZES[name] for name in (
            "kv.f32", "query.f32", "selected.u32", "selected-count.u32")},
        "dtype": {"kv": "f32", "query": "f32", "selected": "u32"},
    }
    if metadata != expected:
        raise ValueError("capture metadata mismatch")
    counts = inputs["selected-count.u32"].memmap(
        dtype="<u4",
        shape=(len(LAYERS), QUERY_ROWS),
    )
    if not np.all(counts == SELECTED_CAPACITY):
        raise ValueError("selected counts are incomplete")
    return metadata


def validate_review_receipt(receipt: dict[str, Any]) -> tuple[str, int]:
    required = {"schema", "candidate_hash", "review_round", "critical", "high",
                "verdict", "drand_min_round"}
    if not required.issubset(receipt):
        raise ValueError("review receipt fields are incomplete")
    candidate = receipt.get("candidate_hash")
    floor = receipt.get("drand_min_round")
    if (receipt.get("schema") != "glm52-w9-fp4-falsifier-review-v1" or
            receipt.get("review_round") != 252 or receipt.get("critical") != [] or
            receipt.get("high") != [] or
            receipt.get("verdict") != "PASS_RUNTIME_ALLOWED" or
            not isinstance(candidate, str) or not re.fullmatch(r"[0-9a-f]{40}", candidate) or
            type(floor) is not int or floor < MINIMUM_REVIEW_FLOOR):
        raise ValueError("review receipt does not authorize candidate 2")
    return candidate, floor


def _git_show(revision: str, relative: str) -> bytes:
    return subprocess.run(
        [str(GIT), "show", f"{revision}:{relative}"], cwd=REPO,
        check=True, capture_output=True,
    ).stdout


def _verify_runtime(runtime: dict[str, Any]) -> dict[str, str]:
    expected_paths = {
        "python_sha256": pathlib.Path(sys.executable),
        "numpy_init_sha256": pathlib.Path(np.__file__),
        "numpy_multiarray_sha256": pathlib.Path(np._core._multiarray_umath.__file__),
        "node_sha256": NODE,
        "git_sha256": GIT,
    }
    if runtime.get("python_path") != "/usr/bin/python3" or sys.executable != "/usr/bin/python3":
        raise ValueError("frozen Python runtime path mismatch")
    if runtime.get("numpy_version") != np.__version__:
        raise ValueError("frozen NumPy version mismatch")
    observed: dict[str, str] = {}
    for key, path in expected_paths.items():
        expected = runtime.get(key)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"missing runtime digest: {key}")
        observed[key] = sha256_file(path)
        if observed[key] != expected:
            raise ValueError(f"frozen runtime digest mismatch: {key}")
    return observed


def _load_authorization(review_input: BoundInput) -> tuple[str, int, dict[str, str], dict[str, str]]:
    status = subprocess.run(
        [str(GIT), "status", "--porcelain"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout
    if status:
        raise ValueError("repository must be clean")
    review_bytes = review_input.read_bytes()
    if review_bytes != _git_show("HEAD", REVIEW_RELATIVE):
        raise ValueError("review receipt is not the tracked HEAD generation")
    candidate, floor = validate_review_receipt(
        strict_json_bytes(review_bytes, "review receipt"))
    subprocess.run(
        [str(GIT), "merge-base", "--is-ancestor", candidate, "HEAD"],
        cwd=REPO, check=True,
    )
    frozen_bytes = _git_show(candidate, FREEZE_RELATIVE)
    freeze_path = REPO / FREEZE_RELATIVE
    if freeze_path.read_bytes() != frozen_bytes:
        raise ValueError("candidate freeze differs from reviewed commit")
    freeze = strict_json_bytes(frozen_bytes, "candidate freeze")
    if freeze.get("schema") != "glm52-w9-fp4-falsifier-freeze-v2":
        raise ValueError("candidate freeze schema mismatch")
    expected_components = freeze.get("component_sha256")
    if not isinstance(expected_components, dict):
        raise ValueError("candidate component bindings missing")
    relatives = (
        SCRIPT_RELATIVE, TEST_RELATIVE, PLAN_RELATIVE, LAUNCHER_RELATIVE,
        "scripts/89_verify_drand_receipt.mjs",
    )
    observed: dict[str, str] = {}
    for relative in relatives:
        current = (REPO / relative).read_bytes()
        if current != _git_show(candidate, relative):
            raise ValueError(f"component differs from reviewed candidate: {relative}")
        observed[relative] = sha256_bytes(current)
        if expected_components.get(relative) != observed[relative]:
            raise ValueError(f"component freeze mismatch: {relative}")
    runtime = freeze.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("runtime freeze is missing")
    runtime_observed = _verify_runtime(runtime)
    return candidate, floor, observed, runtime_observed


def _verify_randomness(receipt_input: BoundInput, minimum_round: int) -> dict[str, Any]:
    receipt = strict_json_bytes(receipt_input.read_bytes(), "randomness receipt")
    if set(receipt) != {"schema", "relay_urls", "relay_records"}:
        raise ValueError("unexpected drand receipt schema")
    if receipt.get("schema") != "glm52-drand-three-relay-v1" or receipt.get("relay_urls") != list(RELAY_URLS):
        raise ValueError("drand relay receipt identity mismatch")
    records = receipt.get("relay_records")
    if not isinstance(records, list) or len(records) != 3 or not records[0] == records[1] == records[2]:
        raise ValueError("drand relays do not agree")
    record = records[0]
    if not isinstance(record, dict) or set(record) != {
            "round", "randomness", "signature", "previous_signature"}:
        raise ValueError("agreed drand record schema mismatch")
    round_number = record.get("round")
    randomness = record.get("randomness")
    signature = record.get("signature")
    previous = record.get("previous_signature")
    if (type(round_number) is not int or round_number <= minimum_round or
            not isinstance(randomness, str) or not re.fullmatch(r"[0-9a-f]{64}", randomness) or
            not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{192}", signature) or
            not isinstance(previous, str) or not re.fullmatch(r"[0-9a-f]{192}", previous)):
        raise ValueError("invalid or stale drand receipt")
    subprocess.run(
        [str(NODE), str(DRAND_VERIFIER), str(round_number), randomness, signature, previous],
        cwd=REPO, env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"},
        check=True, capture_output=True, text=True,
    )
    return record


def _require_idle_host() -> None:
    for process_name in ("ds4-server", "ds4", "fio"):
        result = subprocess.run(
            ["/usr/bin/pgrep", "-x", process_name], check=False,
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            raise ValueError(f"competing process is active: {process_name}")
        if result.returncode != 1:
            raise ValueError(f"could not inspect process state: {process_name}")


def _validate_selected(selection: np.ndarray, sentinel: int) -> None:
    for row_index, row in enumerate(selection):
        invalid = row[(row != sentinel) & (row >= KV_ROWS)]
        if invalid.size:
            raise ValueError("selected key ID is out of range")
        valid = row[row != sentinel]
        if np.unique(valid).size != valid.size:
            raise ValueError("duplicate selected key ID")
        position = row_index * 64
        if valid.size and int(valid.max()) > position:
            raise ValueError("selected key ID violates causal boundary")


def _evaluate(
    capture: pathlib.Path,
    inputs: dict[str, BoundInput],
    metadata: dict[str, Any],
    receipt: dict[str, Any],
    receipt_sha256: str,
    candidate_commit: str,
    minimum_drand_round: int,
    source_bindings: dict[str, str],
    runtime_bindings: dict[str, str],
    review_receipt_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    master_seed = hashlib.sha256(
        b"GLM52-W9-FP4-SPLIT-V1\0" + bytes.fromhex(receipt["randomness"])
        + bytes.fromhex(CAPTURE_HASHES["kv.f32"])
        + bytes.fromhex(CAPTURE_HASHES["query.f32"])
        + bytes.fromhex(candidate_commit)
    ).digest()

    kv = inputs["kv.f32"].memmap(
        dtype="<f4",
        shape=(len(LAYERS), KV_ROWS, WIDTH),
    )
    queries = inputs["query.f32"].memmap(
        dtype="<f4",
        shape=(len(LAYERS), QUERY_ROWS, QUERY_HEADS, WIDTH),
    )
    selected = inputs["selected.u32"].memmap(
        dtype="<u4",
        shape=(len(LAYERS), QUERY_ROWS, SELECTED_CAPACITY),
    )
    raw_rows: list[dict[str, Any]] = []
    totals = {
        candidate: {"numerator": 0.0, "denominator": 0.0, "pairs": 0,
                    "query_rows": 0, "key_references": 0}
        for candidate in CANDIDATES
    }
    split_hashes: dict[str, dict[str, str]] = {}

    for layer_index, layer in enumerate(LAYERS):
        reference = np.array(kv[layer_index], dtype=np.float32, copy=True)
        query = np.array(queries[layer_index], dtype=np.float32, copy=True)
        selection = np.array(selected[layer_index], dtype=np.uint32, copy=True)
        if not np.isfinite(reference).all() or not np.isfinite(query).all():
            raise ValueError(f"non-finite capture values at layer {layer}")
        _validate_selected(selection, int(metadata["selected_padding_sentinel"]))
        calibration_keys, heldout_keys_tuple = split_indices(
            KV_ROWS, master_seed, f"keys/{layer}".encode("ascii"),
        )
        _, heldout_queries_tuple = split_indices(
            QUERY_ROWS, master_seed, f"queries/{layer}".encode("ascii"),
        )
        key_mask = np.zeros(KV_ROWS, dtype=np.bool_)
        key_mask[list(heldout_keys_tuple)] = True
        query_mask = np.zeros(QUERY_ROWS, dtype=np.bool_)
        query_mask[list(heldout_queries_tuple)] = True
        split_hashes[str(layer)] = {
            "calibration_keys_sha256": hashlib.sha256(
                np.asarray(calibration_keys, dtype="<u4").tobytes()).hexdigest(),
            "heldout_keys_sha256": hashlib.sha256(
                np.asarray(heldout_keys_tuple, dtype="<u4").tobytes()).hexdigest(),
            "heldout_queries_sha256": hashlib.sha256(
                np.asarray(heldout_queries_tuple, dtype="<u4").tobytes()).hexdigest(),
        }

        plain = e2m1_quantize(reference)
        metrics = query_weighted_error(
            query, reference, plain, selection,
            int(metadata["selected_padding_sentinel"]), query_mask, key_mask,
        )
        raw_rows.append({"record_type": "w9_fp4_layer", "layer": layer,
                         "candidate": CANDIDATES[0], **metrics})
        for key in totals[CANDIDATES[0]]:
            totals[CANDIDATES[0]][key] += metrics[key]
        del plain

        signs = layer_signs(master_seed, layer, WIDTH)
        rotated_reference = hadamard_rotate(reference, signs)
        rotated_queries = hadamard_rotate(query.reshape(-1, WIDTH), signs).reshape(query.shape)
        rotated_quantized = e2m1_quantize(rotated_reference)
        metrics = query_weighted_error(
            rotated_queries, rotated_reference, rotated_quantized, selection,
            int(metadata["selected_padding_sentinel"]), query_mask, key_mask,
        )
        raw_rows.append({"record_type": "w9_fp4_layer", "layer": layer,
                         "candidate": CANDIDATES[1], **metrics})
        for key in totals[CANDIDATES[1]]:
            totals[CANDIDATES[1]][key] += metrics[key]

        alpha = fit_channel_correction(
            rotated_reference[list(calibration_keys)],
            rotated_quantized[list(calibration_keys)],
        )
        corrected = rotated_quantized * alpha
        metrics = query_weighted_error(
            rotated_queries, rotated_reference, corrected, selection,
            int(metadata["selected_padding_sentinel"]), query_mask, key_mask,
        )
        raw_rows.append({"record_type": "w9_fp4_layer", "layer": layer,
                         "candidate": CANDIDATES[2],
                         "alpha_min": float(alpha.min()),
                         "alpha_max": float(alpha.max()), **metrics})
        for key in totals[CANDIDATES[2]]:
            totals[CANDIDATES[2]][key] += metrics[key]

    aggregate: dict[str, dict[str, float | int]] = {}
    for candidate, values in totals.items():
        numerator = float(values["numerator"])
        denominator = float(values["denominator"])
        if denominator <= 0:
            raise ValueError(f"candidate has nonpositive denominator: {candidate}")
        aggregate[candidate] = {
            **values,
            "relative_rmse": math.sqrt(numerator / denominator),
        }
    winner = min(CANDIDATES, key=lambda name: aggregate[name]["relative_rmse"])
    best_error = float(aggregate[winner]["relative_rmse"])
    verdict = "PASS" if best_error <= MAXIMUM_RELATIVE_RMSE else "NO_RESULT"
    manifest = {
        "schema": "glm52-w9-fp4-falsifier-manifest-v2",
        "candidate_commit": candidate_commit,
        "source_bindings": source_bindings,
        "runtime_bindings": runtime_bindings,
        "review_receipt_sha256": review_receipt_sha256,
        "capture_root": str(capture),
        "capture_hashes": CAPTURE_HASHES,
        "capture_identity": {
            name: list(bound.initial or ()) for name, bound in sorted(inputs.items())
        },
        "randomness_receipt_sha256": receipt_sha256,
        "drand_round": receipt["round"],
        "minimum_drand_round": minimum_drand_round,
        "relay_agreement": list(RELAY_URLS),
        "master_seed_sha256": hashlib.sha256(master_seed).hexdigest(),
        "split_hashes": split_hashes,
        "layers": list(LAYERS),
        "candidates": list(CANDIDATES),
        "block_width": BLOCK_WIDTH,
        "scale": "continuous_f32_multistart_lloyd_raw_sse",
        "numpy_version": np.__version__,
    }
    summary = {
        "schema": "glm52-w9-fp4-falsifier-summary-v1",
        "gate": "W9",
        "checks": {
            "real_capture": True,
            "capture_width_512": True,
            "source_hashes_and_shapes": True,
            "finite_data": True,
            "calibration_heldout_disjoint": True,
            "all_layers_contributed": all(
                aggregate[name]["query_rows"] > 0 and aggregate[name]["pairs"] > 0
                for name in CANDIDATES),
        },
        "formula": "PASS iff every check is true and the minimum preregistered held-out query-weighted relative RMSE is <= 0.05",
        "maximum_allowed_error": MAXIMUM_RELATIVE_RMSE,
        "candidates": aggregate,
        "winner": winner,
        "query_weighted_error": best_error,
        "verdict": verdict,
        "interpretation": "PASS only authorizes an exact packed-format plus fixed 100-case NLL/top-1 gate; NO_RESULT stops this W9 branch without claiming every possible FP4 scale optimizer fails.",
    }
    if not all(summary["checks"].values()):
        summary["verdict"] = "FAIL"
    del kv, queries, selected
    gc.collect()
    return manifest, raw_rows, summary


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    capture = pathlib.Path(args.capture_root).resolve()
    receipt_path = pathlib.Path(args.randomness_receipt).resolve()
    if not capture.is_dir() or capture.is_symlink():
        raise ValueError("capture root must be a real directory")
    _require_idle_host()
    with contextlib.ExitStack() as stack:
        review_input = stack.enter_context(BoundInput(
            REPO / REVIEW_RELATIVE, None, None))
        candidate, floor, source_bindings, runtime_bindings = _load_authorization(
            review_input)
        receipt_input = stack.enter_context(BoundInput(receipt_path, None, None))
        inputs: dict[str, BoundInput] = {}
        for name in CAPTURE_HASHES:
            inputs[name] = stack.enter_context(BoundInput(
                capture / name, CAPTURE_SIZES[name], CAPTURE_HASHES[name]))
        metadata = _verify_capture(inputs)
        receipt = _verify_randomness(receipt_input, floor)
        return _evaluate(
            capture, inputs, metadata, receipt, receipt_input.sha256,
            candidate, floor, source_bindings, runtime_bindings,
            review_input.sha256,
        )


def publish_evidence(
    root: pathlib.Path,
    manifest: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    root = pathlib.Path(root)
    if not root.is_dir() or root.is_symlink() or any(root.iterdir()):
        raise ValueError("evidence root must be a new empty real directory")
    payloads = {
        "manifest.json": json_bytes(manifest),
        "raw.jsonl": b"".join(json_bytes(row, pretty=False) for row in raw_rows),
        "summary.json": json_bytes(summary),
    }
    artifacts = []
    for name in ("manifest.json", "raw.jsonl", "summary.json"):
        value = payloads[name]
        exclusive_write(root / name, value)
        artifacts.append({"path": name, "bytes": len(value), "sha256": sha256_bytes(value)})
    terminal = {
        "schema": "glm52-w9-fp4-falsifier-terminal-v2",
        "artifacts": artifacts,
        "verdict": summary.get("verdict"),
    }
    exclusive_write(root / "terminal-receipt.json", json_bytes(terminal))
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return terminal


def verify_terminal(root: pathlib.Path) -> dict[str, Any]:
    root = pathlib.Path(root)
    expected_names = {"manifest.json", "raw.jsonl", "summary.json", "terminal-receipt.json"}
    if (not root.is_dir() or root.is_symlink() or
            {path.name for path in root.iterdir()} != expected_names):
        raise ValueError("terminal artifact path set mismatch")
    with BoundInput(root / "terminal-receipt.json", None, None) as terminal_input:
        terminal = strict_json_bytes(terminal_input.read_bytes(), "terminal receipt")
        if (set(terminal) != {"schema", "artifacts", "verdict"} or
                terminal.get("schema") != "glm52-w9-fp4-falsifier-terminal-v2" or
                terminal.get("verdict") not in {"PASS", "NO_RESULT", "FAIL"}):
            raise ValueError("terminal receipt schema mismatch")
        artifacts = terminal.get("artifacts")
        if not isinstance(artifacts, list) or [row.get("path") for row in artifacts] != [
                "manifest.json", "raw.jsonl", "summary.json"]:
            raise ValueError("terminal artifact inventory mismatch")
        for row in artifacts:
            if (not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"} or
                    type(row.get("bytes")) is not int or row["bytes"] < 0 or
                    not isinstance(row.get("sha256"), str) or
                    not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
                raise ValueError("terminal artifact binding is malformed")
            try:
                with BoundInput(root / row["path"], row["bytes"], row["sha256"]) as bound:
                    if row["path"] == "summary.json":
                        summary = strict_json_bytes(bound.read_bytes(), "terminal summary")
                        if summary.get("verdict") != terminal["verdict"]:
                            raise ValueError("terminal verdict mismatch")
            except ValueError as error:
                raise ValueError(f"terminal artifact verification failed: {row['path']}") from error
        return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root")
    parser.add_argument("--randomness-receipt")
    parser.add_argument("--output")
    parser.add_argument("--verify-terminal")
    args = parser.parse_args()
    if args.verify_terminal:
        if args.capture_root or args.randomness_receipt or args.output:
            parser.error("--verify-terminal cannot be combined with a run")
        terminal = verify_terminal(pathlib.Path(args.verify_terminal).resolve())
        print(f"W9_FP4_TERMINAL_OK {terminal['verdict']}")
        return 0
    if not args.capture_root or not args.randomness_receipt or not args.output:
        parser.error("a run requires --capture-root, --randomness-receipt, and --output")
    output = pathlib.Path(args.output).resolve()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise SystemExit("output path must not exist beneath an existing directory")
    output.mkdir(mode=0o700)
    try:
        manifest, raw_rows, summary = run(args)
    except BaseException as error:
        manifest = {
            "schema": "glm52-w9-fp4-falsifier-failure-manifest-v2",
            "review_receipt": REVIEW_RELATIVE,
            "capture_root": str(pathlib.Path(args.capture_root).resolve()),
            "randomness_receipt": str(pathlib.Path(args.randomness_receipt).resolve()),
        }
        raw_rows = [{
            "record_type": "w9_fp4_failure",
            "error_type": type(error).__name__,
            "error": str(error),
        }]
        summary = {
            "schema": "glm52-w9-fp4-falsifier-failure-summary-v2",
            "gate": "W9",
            "checks": {},
            "verdict": "FAIL",
        }
    publish_evidence(output, manifest, raw_rows, summary)
    verify_terminal(output)
    print(f"W9_FP4_FALSIFIER_{summary['verdict']} {output}")
    if summary["verdict"] == "PASS":
        return 0
    return 3 if summary["verdict"] == "NO_RESULT" else 2


if __name__ == "__main__":
    sys.exit(main())
