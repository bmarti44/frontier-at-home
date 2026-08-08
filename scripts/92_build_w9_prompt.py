#!/usr/bin/env python3
"""Build one exact 8,192-token neutral W9 raw prompt after candidate freeze."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import re
import sys


BASES = (
    "Measure the system carefully, preserve raw evidence, and report uncertainty. ",
    "A bounded experiment compares identical inputs while changing one mechanism. ",
    "Stable services need explicit memory limits, clean rollback, and honest failures. ",
    "The capture records numerical tensors for an offline falsifier without changing output. ",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--randomness", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.randomness):
        raise SystemExit("randomness must be 32 lowercase hex bytes")
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("output must be new")

    # The model tokenizer is the bound authority; this import supplies only the
    # generic parser/runtime and never contributes prompt text or token IDs.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    base = BASES[int(args.randomness[:2], 16) % len(BASES)]
    source = base * 12000
    token_ids = tokenizer.encode(source, add_special_tokens=False).ids
    if len(token_ids) < 8192:
        raise SystemExit("source did not produce enough tokens")
    selected = token_ids[:8192]
    prompt = tokenizer.decode(selected, skip_special_tokens=False)
    if tokenizer.encode(prompt, add_special_tokens=False).ids != selected:
        raise SystemExit("tokenizer decode/re-encode did not preserve the fixture")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    payload = prompt.encode("utf-8")
    try:
        done = 0
        while done < len(payload):
            wrote = os.write(descriptor, payload[done:])
            if wrote <= 0:
                raise OSError("short prompt write")
            done += wrote
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(f"tokens=8192 sha256={hashlib.sha256(payload).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
