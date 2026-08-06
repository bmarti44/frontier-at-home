#!/usr/bin/env python3
"""Build the frozen W7 production-tokenizer fixture pool deterministically."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import struct
import sys
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
RUNTIME_INIT_SHA256 = "eff4eff4386074cbbd5e34e009bdfccf5879a7e5c5f0da6f4b6babc0597c09e4"
RUNTIME_NATIVE_SHA256 = "fa049ce975669d8a90fb48960f412e626fa54cf596c2f75d6820949f4888e910"
LIVE_SUFFIX = "\n\nOne two three four five six seven."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoded_i32(values: list[int]) -> str:
    raw = struct.pack(f"<{len(values)}i", *values)
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def load_goal_module():
    path = ROOT / "scripts/glm52_goal.py"
    spec = importlib.util.spec_from_file_location("glm52_goal_pool", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--runtime-init", type=Path, required=True)
    parser.add_argument("--runtime-native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = (
        (args.tokenizer, TOKENIZER_SHA256),
        (args.runtime_init, RUNTIME_INIT_SHA256),
        (args.runtime_native, RUNTIME_NATIVE_SHA256),
    )
    for path, digest in expected:
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise RuntimeError(f"frozen dependency mismatch: {path}")

    from tokenizers import Tokenizer

    goal = load_goal_module()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    stem_document = json.loads(goal._W7_STEM_PATH.read_text(encoding="utf-8"))
    stem = stem_document["prompt"]
    live_text = stem + LIVE_SUFFIX
    live_encoding = tokenizer.encode(live_text, add_special_tokens=False)
    live_tokens = list(live_encoding.ids)

    variants = []
    identities: list[str | int] = ["primary-fixed", *range(16)]
    for variant in identities:
        wire = goal._w7_frozen_wire(variant).decode("utf-8")
        encoding = tokenizer.encode(wire, add_special_tokens=False)
        canonical = list(encoding.ids)
        byte_ends = [len(wire[:end].encode("utf-8")) for _, end in encoding.offsets]
        if not byte_ends or byte_ends[-1] != len(wire.encode("utf-8")):
            raise RuntimeError(f"token offsets do not cover variant {variant}")
        common = 0
        while (
            common < len(canonical) and common < len(live_tokens) and
            canonical[common] == live_tokens[common]
        ):
            common += 1
        selected = common - (common % 4)
        if not (selected <= common < len(live_tokens) < len(canonical)):
            raise RuntimeError(f"invalid resume geometry for variant {variant}")
        variants.append({
            "variant": variant,
            "wire_sha256": hashlib.sha256(wire.encode("utf-8")).hexdigest(),
            "canonical_token_ids_zlib_b64": encoded_i32(canonical),
            "wire_token_end_offsets_zlib_b64": encoded_i32(byte_ends),
            "prompt_tokens": len(canonical),
            "common_tokens": common,
            "live_tokens": len(live_tokens),
            "selected_tokens": selected,
        })

    document = {
        "schema": "glm52-w7-production-fixture-pool-v1",
        "tokenizer": {
            "tokenizer_sha256": TOKENIZER_SHA256,
            "runtime_init_sha256": RUNTIME_INIT_SHA256,
            "runtime_native_sha256": RUNTIME_NATIVE_SHA256,
            "add_special_tokens": False,
        },
        "live": {
            "suffix_utf8": LIVE_SUFFIX,
            "wire_sha256": hashlib.sha256(live_text.encode("utf-8")).hexdigest(),
            "token_ids_zlib_b64": encoded_i32(live_tokens),
            "token_count": len(live_tokens),
        },
        "inventory_recipe": {
            "alignment_tokens": 4,
            "older_delta_tokens": -4,
            "wrong_lineage_delta_tokens": 1,
            "malformed_delta_tokens": 2,
        },
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.write_bytes(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
