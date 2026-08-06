#!/usr/bin/python3
"""Count visible GLM output with a path- and digest-bound tokenizer runtime."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import types
from typing import Any


ALLOWED_ENVIRONMENT = {
    "HOME": "/nonexistent",
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}


def fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    raise SystemExit(message)


def validate_sha256(value: str, label: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        fail(f"invalid expected digest: {label}")


def read_bound_descriptor(descriptor: int, path: Path) -> tuple[bytes, str, str]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        fail(f"bound input is not regular: {path}")
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 8 * 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
    after = os.fstat(descriptor)
    identity_before = (
        before.st_dev, before.st_ino, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    )
    if identity_before != identity_after:
        fail(f"bound input changed while read: {path}")
    identity = ":".join(str(value) for value in identity_after)
    return b"".join(chunks), digest.hexdigest(), identity


def open_bound(path: Path) -> tuple[int, bytes, str, str]:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        raw, digest, identity = read_bound_descriptor(descriptor, path)
        return descriptor, raw, digest, identity
    except BaseException:
        os.close(descriptor)
        raise


def bound_bytes(path: Path) -> tuple[bytes, str, str]:
    descriptor, raw, digest, identity = open_bound(path)
    os.close(descriptor)
    return raw, digest, identity


def strict_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                fail(f"duplicate JSON key in {label}: {key}")
            value[key] = item
        return value

    value = json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda item: fail(f"non-finite JSON value in {label}: {item}"),
    )
    if not isinstance(value, dict):
        fail(f"{label} is not an object")
    return value


def main() -> int:
    if len(sys.argv) != 8:
        fail(
            "usage: 84_count_glm_output_tokens.py RESPONSE TOKENIZER "
            "RUNTIME_ROOT TOKENIZER_SHA256 INIT_SHA256 SO_SHA256 LABEL"
        )
    if sys.flags.isolated != 1 or os.environ != ALLOWED_ENVIRONMENT:
        fail("token scorer is not running in its closed isolated environment")

    response_path = Path(sys.argv[1]).resolve(strict=True)
    tokenizer_path = Path(sys.argv[2]).resolve(strict=True)
    runtime_root = Path(sys.argv[3]).resolve(strict=True)
    expected_tokenizer, expected_init, expected_so, label = sys.argv[4:]
    for value, name in (
        (expected_tokenizer, "tokenizer"),
        (expected_init, "runtime init"),
        (expected_so, "runtime native extension"),
    ):
        validate_sha256(value, name)
    if re.fullmatch(r"(off|on)-(warm|measured)", label) is None:
        fail("invalid response label")
    if not str(runtime_root).startswith(
        "/home/bmarti44/.cache/glm52-w3-tokenizer-runtime-"
    ):
        fail("tokenizer runtime is outside the frozen cache root")

    package = runtime_root / "tokenizers"
    init_path = package / "__init__.py"
    native_path = package / "tokenizers.abi3.so"
    inventory = sorted(str(path.relative_to(runtime_root))
                       for path in runtime_root.rglob("*"))
    if inventory != [
        "tokenizers",
        "tokenizers/__init__.py",
        "tokenizers/tokenizers.abi3.so",
    ]:
        fail("frozen tokenizer runtime inventory changed")
    for path in (runtime_root, package, init_path, native_path):
        if path.is_symlink() or path.stat().st_mode & 0o022:
            fail(f"tokenizer runtime file is unsafe: {path}")

    tokenizer_raw, tokenizer_sha, tokenizer_identity = bound_bytes(tokenizer_path)
    init_raw, init_sha, init_identity = bound_bytes(init_path)
    native_descriptor, _, native_sha, native_identity = open_bound(native_path)
    if (tokenizer_sha != expected_tokenizer or init_sha != expected_init or
            native_sha != expected_so):
        os.close(native_descriptor)
        fail("frozen tokenizer dependency digest mismatch")
    # Validate the tokenizer JSON before native code consumes it.
    strict_object(tokenizer_raw, "tokenizer")

    # Load the native extension through the already authenticated descriptor.
    # This neither executes package bytecode nor reopens a replaceable pathname.
    descriptor_path = f"/proc/self/fd/{native_descriptor}"
    package_module = types.ModuleType("tokenizers")
    package_module.__path__ = []  # type: ignore[attr-defined]
    sys.modules["tokenizers"] = package_module
    loader = importlib.machinery.ExtensionFileLoader(
        "tokenizers.tokenizers", descriptor_path
    )
    spec = importlib.util.spec_from_loader("tokenizers.tokenizers", loader)
    if spec is None:
        os.close(native_descriptor)
        fail("could not construct the frozen native tokenizer module")
    native = importlib.util.module_from_spec(spec)
    sys.modules["tokenizers.tokenizers"] = native
    loader.exec_module(native)
    if Path(native.__file__).as_posix() != descriptor_path:
        os.close(native_descriptor)
        fail("native tokenizer did not load through its bound descriptor")
    if read_bound_descriptor(native_descriptor, native_path)[1] != expected_so:
        os.close(native_descriptor)
        fail("native tokenizer changed during import")
    os.close(native_descriptor)

    tokenizer = native.Tokenizer.from_str(
        tokenizer_raw.decode("utf-8", errors="strict")
    )
    response_raw, response_sha, response_identity = bound_bytes(response_path)
    response = strict_object(response_raw, "response")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        fail("response choices are malformed")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        fail("response message is malformed")
    message = choice["message"]
    content = message.get("content", "")
    reasoning = message.get("reasoning_content", "")
    if not isinstance(content, str) or not isinstance(reasoning, str):
        fail("response generated fields are malformed")
    token_count = len(tokenizer.encode(content, add_special_tokens=False).ids)

    # Recheck every writable dependency immediately after scoring.
    if (bound_bytes(tokenizer_path)[1] != expected_tokenizer or
            bound_bytes(init_path)[1] != expected_init or
            bound_bytes(native_path)[1] != expected_so):
        fail("tokenizer dependency changed during scoring")
    record = {
        "schema_version": 1,
        "label": label,
        "reference_token_count": token_count,
        "content_bytes": len(content.encode("utf-8")),
        "reasoning_bytes": len(reasoning.encode("utf-8")),
        "response_sha256": response_sha,
        "response_identity": response_identity,
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_identity": tokenizer_identity,
        "runtime_init_sha256": init_sha,
        "runtime_init_identity": init_identity,
        "runtime_native_sha256": native_sha,
        "runtime_native_identity": native_identity,
        "runtime_init_path": str(init_path),
        "runtime_native_path": str(native_path),
        "runtime_native_loaded_path": descriptor_path,
    }
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
