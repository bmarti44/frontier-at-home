#!/usr/bin/env python3
"""Fail-closed equivalence scorer for W7 production-server request traces."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import re
import struct
import sys
import zlib


REQUEST_MARKER = b"\n===== request "
RAW_MARKER = b"\n--- raw request json ---\n"
RENDERED_MARKER = b"\n--- rendered prompt ---\n"
GENERATED_MARKER = b"\n--- generated text ---\n"
TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
TOKENIZER_INIT_SHA256 = "eff4eff4386074cbbd5e34e009bdfccf5879a7e5c5f0da6f4b6babc0597c09e4"
TOKENIZER_NATIVE_SHA256 = "fa049ce975669d8a90fb48960f412e626fa54cf596c2f75d6820949f4888e910"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_dependency(path: pathlib.Path, expected: str) -> None:
    if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
        raise ValueError(f"frozen dependency mismatch: {path}")


def _unframe(section: bytes, expected: bytes) -> bytes:
    framed = expected if expected.endswith(b"\n") else expected + b"\n"
    if section != framed:
        raise ValueError("trace section differs from expected bytes")
    return expected


def _request_blocks(trace: bytes) -> list[bytes]:
    pieces = trace.split(REQUEST_MARKER)
    if pieces[0] != b"":
        raise ValueError("unexpected bytes before first trace request")
    blocks = [REQUEST_MARKER + piece for piece in pieces[1:]]
    if len(blocks) != 2:
        raise ValueError(f"expected exactly two request blocks, got {len(blocks)}")
    return blocks


def _sections(block: bytes) -> tuple[int, bytes, bytes]:
    header = re.match(rb"\n===== request ([0-9]+) [^\n]* =====\n", block)
    if header is None:
        raise ValueError("malformed trace request header")
    expected_end = b"\n===== end request " + header.group(1) + b" =====\n"
    if not block.endswith(expected_end) or block.count(expected_end) != 1:
        raise ValueError("missing, mismatched, or non-terminal trace request end")
    if any(block.count(marker) != 1 for marker in (RAW_MARKER, RENDERED_MARKER, GENERATED_MARKER)):
        raise ValueError("missing, duplicate, or ambiguous trace section marker")
    raw_at = block.index(RAW_MARKER) + len(RAW_MARKER)
    rendered_marker_at = block.index(RENDERED_MARKER, raw_at)
    rendered_at = rendered_marker_at + len(RENDERED_MARKER)
    generated_at = block.index(GENERATED_MARKER, rendered_at)
    return int(header.group(1)), block[raw_at:rendered_marker_at], block[rendered_at:generated_at]


def score_trace(
    trace: bytes,
    pool: dict,
    live_request: bytes,
    primary_request: bytes,
    tokenizer_path: pathlib.Path,
    tokenizer_runtime: pathlib.Path,
) -> dict:
    checks = {
        "trace_exactly_two_requests": False,
        "trace_request_ids_exact": False,
        "trace_request_bytes_exact": False,
        "trace_rendered_bytes_exact": False,
        "trace_token_vectors_exact": False,
    }
    observed: list[dict] = []
    error = None
    try:
        _require_dependency(tokenizer_path, TOKENIZER_SHA256)
        _require_dependency(tokenizer_runtime / "tokenizers/__init__.py", TOKENIZER_INIT_SHA256)
        _require_dependency(tokenizer_runtime / "tokenizers/tokenizers.abi3.so", TOKENIZER_NATIVE_SHA256)
        sys.path.insert(0, str(tokenizer_runtime))
        from tokenizers import Tokenizer

        primary = next(item for item in pool["variants"] if item["variant"] == "primary-fixed")
        expected = (
            (
                live_request,
                base64.b64decode(pool["live"]["rendered_wire_utf8_b64"], validate=True),
                tuple(struct.unpack(
                    f"<{pool['live']['token_count']}i",
                    zlib.decompress(base64.b64decode(pool["live"]["token_ids_zlib_b64"], validate=True)),
                )),
            ),
            (
                primary_request,
                base64.b64decode(primary["rendered_wire_utf8_b64"], validate=True),
                tuple(struct.unpack(
                    f"<{primary['prompt_tokens']}i",
                    zlib.decompress(base64.b64decode(primary["canonical_token_ids_zlib_b64"], validate=True)),
                )),
            ),
        )
        blocks = _request_blocks(trace)
        checks["trace_exactly_two_requests"] = True
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        requests_ok = rendered_ok = tokens_ok = True
        request_ids = []
        for block, (want_request, want_rendered, want_tokens) in zip(blocks, expected, strict=True):
            request_id, raw_section, rendered_section = _sections(block)
            request_ids.append(request_id)
            actual_request = _unframe(raw_section, want_request)
            actual_rendered = _unframe(rendered_section, want_rendered)
            requests_ok &= actual_request == want_request
            rendered_ok &= actual_rendered == want_rendered
            actual_tokens = tuple(tokenizer.encode(actual_rendered.decode("utf-8"), add_special_tokens=False).ids)
            tokens_ok &= actual_tokens == want_tokens
            observed.append({
                "request_sha256": hashlib.sha256(actual_request).hexdigest(),
                "rendered_sha256": hashlib.sha256(actual_rendered).hexdigest(),
                "token_count": len(actual_tokens),
                "token_ids_sha256": hashlib.sha256(struct.pack(f"<{len(actual_tokens)}i", *actual_tokens)).hexdigest(),
            })
        checks["trace_request_bytes_exact"] = requests_ok
        checks["trace_rendered_bytes_exact"] = rendered_ok
        checks["trace_token_vectors_exact"] = tokens_ok
        checks["trace_request_ids_exact"] = request_ids == [1, 2]
    except Exception as exc:  # fail closed while retaining a useful diagnostic
        error = f"{type(exc).__name__}: {exc}"
    return {
        "schema_version": 1,
        "checks": checks,
        "observed": observed,
        "error": error,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=pathlib.Path)
    parser.add_argument("--pool", required=True, type=pathlib.Path)
    parser.add_argument("--live-request", required=True, type=pathlib.Path)
    parser.add_argument("--primary-request", required=True, type=pathlib.Path)
    parser.add_argument("--tokenizer", required=True, type=pathlib.Path)
    parser.add_argument("--tokenizer-runtime", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = score_trace(
        args.trace.read_bytes(),
        json.loads(args.pool.read_text(encoding="utf-8")),
        args.live_request.read_bytes(),
        args.primary_request.read_bytes(),
        args.tokenizer,
        args.tokenizer_runtime,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
