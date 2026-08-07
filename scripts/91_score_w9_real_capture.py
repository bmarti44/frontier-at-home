#!/usr/bin/env python3
"""Fixed scorer for the matched GLM-5.2 W9 real-capture smoke.

This is a correctness/capture gate, not a performance, fidelity, or context
capability result. PASS requires byte-identical final F32 logits from matched
capture-OFF and capture-ON arms plus the exact preregistered real tensor corpus.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import sys

ARMS = ("off", "on")
LAYERS = (0, 2, 10, 26, 42, 58, 74, 77)
PROMPT_TOKENS = 8192
LOGIT_COUNT = 154880
CAPTURE_SIZES = {
    "kv.f32": 134217728,
    "query.f32": 134217728,
    "selected.u32": 8388608,
    "selected-count.u32": 4096,
}
CAPTURE_NAMES = set(CAPTURE_SIZES) | {"metadata.json", "W9_CAPTURE_COMPLETE"}
LOGIT_RE = re.compile(r"logits\.sync\d+\.start(\d+)\.prompt(\d+)\.suffix(\d+)")
SAMPLE_RE = re.compile(
    r"mem_avail_kb=(\d+).*?eng_rss_kb=(\d+).*?"
    r"cgroup_current_bytes=(\d+).*?cgroup_peak_bytes=(\d+).*?"
    r"cgroup_swap_current_bytes=(\d+)"
)
CGROUP_RE = re.compile(
    r"cgroup_final current_bytes=(\d+) peak_bytes=(\d+) "
    r"swap_current_bytes=(\d+) events=low (\d+),high (\d+),max (\d+),"
    r"oom (\d+),oom_kill (\d+),oom_group_kill (\d+),"
)
TERMINAL = {"raw.jsonl", "summary.json", "terminal-receipt.json"}


def read_regular(path: pathlib.Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"unsafe artifact: {path}")
        pieces = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            pieces.append(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns) != \
                (after.st_dev, after.st_ino, after.st_size,
                 after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"artifact changed while read: {path}")
        return b"".join(pieces)
    finally:
        os.close(fd)


def scan_regular(path: pathlib.Path, consume=None) -> tuple[str, int]:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"unsafe artifact: {path}")
        while True:
            chunk = os.read(fd, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if consume is not None:
                consume(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns) != \
                (after.st_dev, after.st_ino, after.st_size,
                 after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"artifact changed while hashed: {path}")
        return digest.hexdigest(), before.st_size
    finally:
        os.close(fd)


def sha_regular(path: pathlib.Path) -> tuple[str, int]:
    return scan_regular(path)


def parse_strict_json(payload: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=nonfinite)


def strict_json(path: pathlib.Path):
    return parse_strict_json(read_regular(path))


def inventory(root: pathlib.Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in TERMINAL:
            continue
        if path.is_dir() and not path.is_symlink():
            rows.append({"path": relative, "directory": True})
            continue
        digest, size = sha_regular(path)
        rows.append({"path": relative, "bytes": size, "sha256": digest})
    return rows


def verify_inventory(root: pathlib.Path, rows: list[dict]) -> None:
    if not isinstance(rows, list):
        raise ValueError("artifact inventory missing")
    seen = set()
    for row in rows:
        relative = row.get("path") if isinstance(row, dict) else None
        pure = pathlib.PurePosixPath(relative) if isinstance(relative, str) else None
        if pure is None or pure.is_absolute() or ".." in pure.parts or relative in seen:
            raise ValueError("invalid or duplicate inventory path")
        seen.add(relative)
        path = root.joinpath(*pure.parts)
        if row.get("directory") is True:
            if not path.is_dir() or path.is_symlink() or set(row) != {"path", "directory"}:
                raise ValueError(f"artifact directory mutation: {relative}")
            continue
        digest, size = sha_regular(path)
        if digest != row.get("sha256") or size != row.get("bytes"):
            raise ValueError(f"artifact mutation: {relative}")
    current = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.relative_to(root).as_posix() not in TERMINAL
    }
    if current != seen:
        raise ValueError("terminal artifact path set changed")


def validate_f32(path: pathlib.Path, expected_bytes: int) -> str:
    def consume(chunk: bytes) -> None:
        if len(chunk) % 4:
            raise ValueError(f"unaligned F32 artifact: {path.name}")
        values = array.array("f")
        values.frombytes(chunk)
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"non-finite F32 artifact: {path.name}")

    digest, size = scan_regular(path, consume)
    if size != expected_bytes:
        raise ValueError(f"wrong F32 artifact size: {path.name}")
    return digest


def validate_selected(capture: pathlib.Path, sentinel: int) -> dict:
    counts_bytes = read_regular(capture / "selected-count.u32")
    if len(counts_bytes) != CAPTURE_SIZES["selected-count.u32"]:
        raise ValueError("selected-count size mismatch")
    counts = array.array("I")
    counts.frombytes(counts_bytes)
    if len(counts) != len(LAYERS) * 128 or any(count != 2048 for count in counts):
        raise ValueError("selected counts are malformed")
    selected = capture / "selected.u32"
    row_index = 0

    def consume(chunk: bytes) -> None:
        nonlocal row_index
        if len(chunk) % (2048 * 4):
            raise ValueError("short selected row")
        for offset in range(0, len(chunk), 2048 * 4):
            if row_index >= len(counts):
                raise ValueError("extra selected row")
            count = counts[row_index]
            ids = array.array("I")
            ids.frombytes(chunk[offset:offset + 2048 * 4])
            visible = (row_index % 128) * 64 + 1
            if any(identifier >= visible and identifier != sentinel
                   for identifier in ids[:count]):
                raise ValueError("noncausal selected ID")
            if any(identifier != 0xFFFFFFFF for identifier in ids[count:]):
                raise ValueError("selected row padding mismatch")
            row_index += 1

    digest, size = scan_regular(selected, consume)
    if size != CAPTURE_SIZES["selected.u32"] or row_index != len(counts):
        raise ValueError("selected size mismatch")
    return {"sha256": digest, "rows": len(counts),
            "count_sha256": hashlib.sha256(counts_bytes).hexdigest(),
            "minimum_count": min(counts), "maximum_count": max(counts)}


def validate_capture(capture: pathlib.Path) -> dict:
    if not capture.is_dir() or capture.is_symlink():
        raise ValueError("capture directory missing or unsafe")
    names = {path.name for path in capture.iterdir()}
    if names != CAPTURE_NAMES:
        raise ValueError("capture inventory is incomplete or has extras")
    marker = read_regular(capture / "W9_CAPTURE_COMPLETE")
    if marker != b"W9_CAPTURE_COMPLETE\n":
        raise ValueError("capture completion marker mismatch")
    metadata_payload = read_regular(capture / "metadata.json")
    metadata = parse_strict_json(metadata_payload)
    expected = {
        "schema": "glm52-w9-real-capture-v1", "layers": list(LAYERS),
        "kv_rows_per_layer": 8192, "kv_width": 512,
        "query_rows_per_layer": 128, "query_heads": 64, "query_width": 512,
        "selected_capacity": 2048, "sample_position_start": 0,
        "sample_position_stride": 64, "storage_padding_sentinel": 0xFFFFFFFF,
        "artifacts": CAPTURE_SIZES,
        "dtype": {"kv": "f32", "query": "f32", "selected": "u32"},
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"capture metadata mismatch: {key}")
    sentinel = metadata.get("selected_padding_sentinel")
    if not isinstance(sentinel, int) or sentinel < PROMPT_TOKENS:
        raise ValueError("capture padding sentinel is invalid")
    selected = validate_selected(capture, sentinel)
    return {
        "kv_sha256": validate_f32(capture / "kv.f32", CAPTURE_SIZES["kv.f32"]),
        "query_sha256": validate_f32(capture / "query.f32", CAPTURE_SIZES["query.f32"]),
        "selected": selected,
        "selected_count_sha256": selected["count_sha256"],
        "metadata_sha256": hashlib.sha256(metadata_payload).hexdigest(),
        "marker_sha256": hashlib.sha256(marker).hexdigest(),
        "sentinel": sentinel,
    }


def final_logits(arm: pathlib.Path) -> tuple[pathlib.Path, bytes, str]:
    candidates = []
    for path in arm.glob("logits.sync*.start*.prompt*.suffix*"):
        match = LOGIT_RE.fullmatch(path.name)
        if match and path.is_file() and not path.is_symlink():
            candidates.append((path, *map(int, match.groups())))
    winners = [row for row in candidates if row[2] == PROMPT_TOKENS]
    if len(winners) != 1:
        raise ValueError(f"{arm.name}: expected exactly one final logit dump")
    path, start, prompt, suffix = winners[0]
    if start + suffix != prompt:
        raise ValueError(f"{arm.name}: invalid logit geometry")
    payload = read_regular(path)
    if len(payload) != LOGIT_COUNT * 4:
        raise ValueError(f"{arm.name}: logit size mismatch")
    values = array.array("f")
    values.frombytes(payload)
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{arm.name}: non-finite logits")
    return path, payload, hashlib.sha256(payload).hexdigest()


def validate_logit_publication(stderr_text: str, logit_path: pathlib.Path) -> None:
    if "prefill logits dump failed" in stderr_text:
        raise ValueError("logit publication failure diagnostic")
    publications = re.findall(r"^ds4: prefill logits dumped to (.+)$",
                              stderr_text, re.M)
    if publications != [str(logit_path)]:
        raise ValueError("logit publication success binding mismatch")


def safety(arm: pathlib.Path, expected_binary: str) -> dict:
    if read_regular(arm / "containment.rc").strip() != b"0":
        raise ValueError(f"{arm.name}: containment failed")
    wrapper = read_regular(arm / "containment.stdout").decode(errors="replace")
    if not re.search(r"^SAFE_RUN_DONE rc=0 killed=no dir=", wrapper, re.M):
        raise ValueError(f"{arm.name}: clean wrapper exit missing")
    main = read_regular(arm / "safety/main.log").decode(errors="replace")
    kernel = read_regular(arm / "safety/kernel.log").decode(errors="replace")
    samples_text = read_regular(arm / "safety/samples.log").decode(errors="replace")
    samples = [tuple(map(int, match.groups())) for match in SAMPLE_RE.finditer(samples_text)]
    if not samples or any(sample[4] != 0 for sample in samples):
        raise ValueError(f"{arm.name}: missing safety samples or swap")
    if min(sample[0] for sample in samples) < 24 * 1024 * 1024:
        raise ValueError(f"{arm.name}: memory floor violated")
    if re.search(r"out of memory|oom-kill|killed process|NVRM: Xid", main + kernel, re.I):
        raise ValueError(f"{arm.name}: OOM/Xid evidence")
    events = CGROUP_RE.findall(main)
    if len(events) != 1 or any(int(value) != 0 for value in events[0][2:]):
        raise ValueError(f"{arm.name}: cgroup events missing or nonzero")
    identities = re.findall(
        r"executed_candidate_verified .*executed_binary_sha256=([0-9a-f]{64}) ", main)
    if identities != [expected_binary] or main.count("wrapper and descendant checks clean") != 1:
        raise ValueError(f"{arm.name}: candidate identity/exit invalid")
    return {"minimum_mem_available_gib": min(row[0] for row in samples) / 1024 / 1024,
            "maximum_engine_rss_gib": max(row[1] for row in samples) / 1024 / 1024,
            "samples": len(samples)}


def score(root: pathlib.Path, manifest_path: pathlib.Path) -> tuple[list[dict], dict]:
    manifest = strict_json(manifest_path)
    if manifest.get("schema") != "glm52-w9-real-capture-manifest-v1":
        raise ValueError("manifest schema mismatch")
    if sorted(manifest.get("arm_order", [])) != list(ARMS):
        raise ValueError("arm order is incomplete")
    arms = manifest.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("arm manifests are incomplete")
    common = ("binary_sha256", "model_sha256", "tokenizer_sha256", "prompt_sha256",
              "configuration_sha256")
    for field in common:
        values = {arms[arm].get(field) for arm in ARMS}
        if len(values) != 1 or values != {manifest.get(field)} or \
                not re.fullmatch(r"[0-9a-f]{64}", next(iter(values), "")):
            raise ValueError(f"arms do not share frozen {field}")
    if arms["off"].get("capture") is not False or arms["on"].get("capture") is not True:
        raise ValueError("arm identities are invalid")
    logits, outputs, rows = {}, {}, []
    for arm_name in ARMS:
        arm = root / arm_name
        prompt_hash, _ = sha_regular(arm / "prompt.txt")
        if prompt_hash != manifest["prompt_sha256"]:
            raise ValueError(f"{arm_name}: prompt differs from frozen fixture")
        logit_path, logits[arm_name], logit_hash = final_logits(arm)
        outputs[arm_name] = read_regular(arm / "cli.stdout")
        stderr_payload = read_regular(arm / "cli.stderr")
        server_text = stderr_payload.decode(errors="replace")
        validate_logit_publication(server_text, logit_path)
        capture_path = arm / "capture"
        capture_result = validate_capture(capture_path) if arm_name == "on" else None
        if arm_name == "on" and server_text.count("W9 real capture complete") != 1:
            raise ValueError("ON arm lacks one production capture marker")
        if arm_name == "off" and (capture_path.exists() or "W9 real capture" in server_text):
            raise ValueError("OFF arm executed the W9 path")
        arm_inventory = inventory(arm)
        inventory_hashes = {row["path"]: row["sha256"] for row in arm_inventory
                            if "sha256" in row}
        semantic_arm_hashes = {
            "prompt.txt": prompt_hash,
            "cli.stdout": hashlib.sha256(outputs[arm_name]).hexdigest(),
            "cli.stderr": hashlib.sha256(stderr_payload).hexdigest(),
            logit_path.relative_to(arm).as_posix(): logit_hash,
        }
        if any(inventory_hashes.get(path) != digest
               for path, digest in semantic_arm_hashes.items()):
            raise ValueError("arm semantic/inventory digest mismatch")
        if capture_result is not None:
            semantic_hashes = {
                "capture/kv.f32": capture_result["kv_sha256"],
                "capture/query.f32": capture_result["query_sha256"],
                "capture/selected.u32": capture_result["selected"]["sha256"],
                "capture/selected-count.u32": capture_result["selected_count_sha256"],
                "capture/metadata.json": capture_result["metadata_sha256"],
                "capture/W9_CAPTURE_COMPLETE": capture_result["marker_sha256"],
            }
            if any(inventory_hashes.get(path) != digest
                   for path, digest in semantic_hashes.items()):
                raise ValueError("capture semantic/inventory digest mismatch")
        rows.append({"record_type": "w9_capture_arm", "arm": arm_name,
                     "prompt_tokens": PROMPT_TOKENS, "logit_sha256": logit_hash,
                     "stdout_sha256": hashlib.sha256(outputs[arm_name]).hexdigest(),
                     "safety": safety(arm, manifest["binary_sha256"]),
                     "capture": capture_result,
                     "artifacts": arm_inventory})
    checks = {
        "matched_inputs_and_configuration": True,
        "off_path_clean": rows[0]["capture"] is None,
        "on_capture_exact_and_finite": rows[1]["capture"] is not None,
        "final_logits_byte_identical": logits["off"] == logits["on"],
        "zero_token_stdout_byte_identical": outputs["off"] == outputs["on"],
        "containment_and_memory_clean": True,
    }
    summary = {
        "schema": "glm52-w9-real-capture-summary-v1", "gate": "W9-real-capture",
        "formula": "PASS iff every check is true; zero-token stdout is corroboration only, final F32 logit identity is authoritative",
        "checks": checks, "metrics": {"prompt_tokens": PROMPT_TOKENS,
            "compared_f32_logits": LOGIT_COUNT, "captured_layers": len(LAYERS),
            "captured_kv_rows": len(LAYERS) * 8192,
            "captured_query_rows": len(LAYERS) * 128},
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "artifacts": inventory(root),
    }
    if summary["verdict"] != "PASS":
        raise ValueError("one or more W9 acceptance checks failed")
    return rows, summary


def write_terminal(root: pathlib.Path, manifest: pathlib.Path,
                   raw: pathlib.Path, summary_path: pathlib.Path,
                   rows: list[dict], summary: dict) -> None:
    raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                   encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    receipt = {"schema": "glm52-w9-terminal-receipt-v1",
               "manifest_sha256": sha_regular(manifest)[0],
               "raw_sha256": sha_regular(raw)[0],
               "summary_sha256": sha_regular(summary_path)[0],
               "verdict": summary["verdict"], "artifacts": summary["artifacts"]}
    (root / "terminal-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_terminal(root: pathlib.Path, manifest: pathlib.Path,
                    raw: pathlib.Path, summary_path: pathlib.Path) -> None:
    receipt = strict_json(root / "terminal-receipt.json")
    if receipt.get("schema") != "glm52-w9-terminal-receipt-v1" or \
            receipt.get("manifest_sha256") != sha_regular(manifest)[0] or \
            receipt.get("raw_sha256") != sha_regular(raw)[0] or \
            receipt.get("summary_sha256") != sha_regular(summary_path)[0]:
        raise ValueError("terminal receipt binding mismatch")
    verify_inventory(root, receipt.get("artifacts"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--raw", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--verify-terminal", action="store_true")
    parser.add_argument("--failure-reason")
    args = parser.parse_args()
    try:
        if args.verify_terminal:
            verify_terminal(args.root, args.manifest, args.raw, args.summary)
            return 0
        if args.failure_reason:
            rows = [{"record_type": "w9_failure", "reason": args.failure_reason,
                     "artifacts": inventory(args.root)}]
            summary = {"schema": "glm52-w9-real-capture-summary-v1",
                       "gate": "W9-real-capture", "checks": {},
                       "verdict": "FAIL", "failure_reason": args.failure_reason,
                       "artifacts": inventory(args.root)}
            write_terminal(args.root, args.manifest, args.raw, args.summary, rows, summary)
            return 2
        rows, summary = score(args.root, args.manifest)
        write_terminal(args.root, args.manifest, args.raw, args.summary, rows, summary)
        return 0
    except Exception as error:
        print(f"W9 scorer failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
