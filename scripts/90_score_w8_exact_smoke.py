#!/usr/bin/env python3
"""Fixed scorer for the model-backed exact-W8 production smoke.

This gate is correctness-only.  PASS requires the resident and exact arms to
use the same frozen binary, model and 5,066-token request; produce byte-equal
final F32 logits and completion text; exercise the exact cKV/direct-slot path;
and exit cleanly under the standard host-safety wrapper.  It is not a 1M
capability or performance result.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import pathlib
import re
import sys
import tempfile

ARMS = ("resident", "exact")
PROMPT_TOKENS = 5066
LOGIT_COUNT = 154880
LOGIT_RE = re.compile(r"logits\.sync\d+\.start(\d+)\.prompt(\d+)\.suffix(\d+)$")
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
W8_COUNTER_RE = re.compile(
    r"W8 exact request complete append_calls=(\d+) appended_rows=(\d+) "
    r"records=(\d+) selected_read_calls=(\d+) selected_rows=(\d+) "
    r"checksum_validations=(\d+) checksum_failures=(\d+) "
    r"cache_hits=(\d+) cache_misses=(\d+) direct_slot_calls=(\d+)"
)
TERMINAL_NAMES = {"raw.jsonl", "summary.json", "terminal-receipt.json"}


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular(path: pathlib.Path) -> pathlib.Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe artifact: {path}")
    return path


def strict_json(path: pathlib.Path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out
    def reject_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )


def final_logits(arm: pathlib.Path) -> tuple[pathlib.Path, int, int]:
    candidates = []
    for path in arm.glob("logits.sync*.start*.prompt*.suffix*"):
        match = LOGIT_RE.fullmatch(path.name)
        if match and path.is_file() and not path.is_symlink():
            start, prompt, suffix = map(int, match.groups())
            candidates.append((prompt, start, suffix, path))
    winners = [row for row in candidates if row[0] == PROMPT_TOKENS]
    if len(winners) != 1:
        raise ValueError(f"{arm.name}: expected one final {PROMPT_TOKENS}-token dump")
    prompt, start, suffix, path = winners[0]
    if path.stat().st_size != LOGIT_COUNT * 4 or start + suffix != prompt:
        raise ValueError(f"{arm.name}: invalid final logit geometry")
    return path, start, suffix


def response(path: pathlib.Path) -> tuple[str, dict]:
    doc = strict_json(path)
    choices = doc.get("choices") if isinstance(doc, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response must contain exactly one choice")
    text = choices[0].get("text") if isinstance(choices[0], dict) else None
    if not isinstance(text, str):
        raise ValueError("response text is missing")
    usage = doc.get("usage")
    if not isinstance(usage, dict) or usage.get("prompt_tokens") != PROMPT_TOKENS:
        raise ValueError("response prompt-token usage is invalid")
    if usage.get("completion_tokens") != 0 or usage.get("total_tokens") != PROMPT_TOKENS:
        raise ValueError("response completion/total usage is invalid")
    return text, usage


def artifact_inventory(root: pathlib.Path, strict: bool = True) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in TERMINAL_NAMES or relative == "manifest.json" or path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            if strict:
                raise ValueError(f"unsafe artifact in inventory: {relative}")
            rows.append({"path": relative, "unsafe": True})
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size,
                     "sha256": sha(path)})
    return rows


def verify_inventory(base: pathlib.Path, inventory: list[dict]) -> None:
    if not isinstance(inventory, list):
        raise ValueError("artifact inventory is missing")
    seen = set()
    for artifact in inventory:
        if not isinstance(artifact, dict) or artifact.get("unsafe"):
            raise ValueError("unsafe artifact inventory entry")
        relative = artifact.get("path")
        pure = pathlib.PurePosixPath(relative) if isinstance(relative, str) else None
        if (pure is None or pure.is_absolute() or ".." in pure.parts or
                relative in seen):
            raise ValueError("invalid artifact inventory path")
        seen.add(relative)
        path = base.joinpath(*pure.parts)
        if (artifact.get("bytes") != regular(path).stat().st_size or
                artifact.get("sha256") != sha(path)):
            raise ValueError(f"post-score artifact mutation: {relative}")


def safety(arm: pathlib.Path) -> dict:
    if (arm / "containment.rc").read_text().strip() != "0":
        raise ValueError(f"{arm.name}: containment failed")
    done = (arm / "containment.stdout").read_text(encoding="utf-8")
    if not re.search(r"^SAFE_RUN_DONE rc=0 killed=no dir=", done, re.M):
        raise ValueError(f"{arm.name}: clean wrapper exit is absent")
    samples = []
    for line in (arm / "safety" / "samples.log").read_text().splitlines():
        match = SAMPLE_RE.search(line)
        if match:
            samples.append(tuple(map(int, match.groups())))
    if not samples or any(row[4] != 0 for row in samples):
        raise ValueError(f"{arm.name}: missing samples or cgroup swap")
    minimum = min(row[0] for row in samples) / 1024 / 1024
    if minimum < 24.0:
        raise ValueError(f"{arm.name}: memory floor violated")
    main = regular(arm / "safety" / "main.log").read_text(
        encoding="utf-8", errors="replace")
    kernel = regular(arm / "safety" / "kernel.log").read_text(
        encoding="utf-8", errors="replace")
    fault_text = main + "\n" + kernel
    if re.search(r"out of memory|oom-kill|killed process|NVRM: Xid", fault_text, re.I):
        raise ValueError(f"{arm.name}: OOM/Xid evidence")
    events = CGROUP_RE.findall(main)
    if len(events) != 1 or any(int(value) != 0 for value in events[0][2:]):
        raise ValueError(f"{arm.name}: cgroup event counters are missing or nonzero")
    identity = re.findall(
        r"executed_candidate_verified .*executed_binary_sha256=([0-9a-f]{64}) ",
        main,
    )
    if len(identity) != 1:
        raise ValueError(f"{arm.name}: executed binary identity is absent")
    if main.count("wrapper and descendant checks clean") != 1:
        raise ValueError(f"{arm.name}: descendant verification is absent")
    if len(re.findall(r"SAFE_RUN end rc=0 killed=no", main)) != 1:
        raise ValueError(f"{arm.name}: clean terminal wrapper record is absent")
    return {
        "minimum_mem_available_gib": minimum,
        "maximum_engine_rss_gib": max(row[1] for row in samples) / 1024 / 1024,
        "maximum_cgroup_bytes": max(row[3] for row in samples),
        "samples": len(samples),
        "executed_binary_sha256": identity[0],
        "cgroup_events": [int(value) for value in events[0][3:]],
    }


def exact_counters(server: str) -> dict:
    matches = W8_COUNTER_RE.findall(server)
    if len(matches) != 1:
        raise ValueError("exact arm lacks one terminal W8 counter record")
    values = list(map(int, matches[0]))
    names = ("append_calls", "appended_rows", "records", "selected_read_calls",
             "selected_rows", "checksum_validations", "checksum_failures",
             "cache_hits", "cache_misses", "direct_slot_calls")
    counters = dict(zip(names, values))
    positive = ("append_calls", "appended_rows", "records", "selected_read_calls",
                "selected_rows", "checksum_validations", "cache_misses",
                "direct_slot_calls")
    if any(counters[name] <= 0 for name in positive):
        raise ValueError("exact W8 counters are not positive")
    if counters["checksum_failures"] != 0:
        raise ValueError("exact W8 checksum failure observed")
    if counters["checksum_validations"] != counters["cache_misses"]:
        raise ValueError("exact W8 checksum/cache accounting mismatch")
    if counters["selected_read_calls"] != counters["direct_slot_calls"]:
        raise ValueError("exact W8 read/direct-slot accounting mismatch")
    return counters


def score(root: pathlib.Path, manifest_path: pathlib.Path) -> tuple[list[dict], dict]:
    manifest = strict_json(manifest_path)
    if manifest.get("schema") != "glm52-w8-exact-smoke-manifest-v1":
        raise ValueError("manifest schema mismatch")
    receipt_hash = manifest.get("randomness_receipt_sha256")
    if (not isinstance(receipt_hash, str) or
            not re.fullmatch(r"[0-9a-f]{64}", receipt_hash) or
            sha(regular(root / "randomness-receipt.json")) != receipt_hash):
        raise ValueError("preserved randomness receipt does not match manifest")
    if sorted(manifest.get("arm_order", [])) != sorted(ARMS):
        raise ValueError("arm order is incomplete")
    configs = manifest.get("arms")
    if not isinstance(configs, dict) or set(configs) != set(ARMS):
        raise ValueError("arm configurations are incomplete")
    common = ("binary_sha256", "model_sha256", "request_sha256")
    for field in common:
        values = {configs[arm].get(field) for arm in ARMS}
        value = next(iter(values)) if len(values) == 1 else None
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"arms do not share {field}")
        if manifest.get(field) != value:
            raise ValueError(f"top-level {field} does not match arms")
    if any(configs[arm].get("context") != 8192 for arm in ARMS):
        raise ValueError("smoke arms must use context 8192")
    if configs["resident"].get("ckv_mode") != "resident" or \
       configs["exact"].get("ckv_mode") != "exact-f32-nvme":
        raise ValueError("arm identities are invalid")

    rows, logits, texts, safety_rows = [], {}, {}, {}
    expected_request = manifest["request_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected_request):
        raise ValueError("invalid frozen request hash")
    for arm_name in ARMS:
        arm = root / arm_name
        if sha(regular(arm / "request.json")) != expected_request:
            raise ValueError(f"{arm_name}: actual request differs from frozen fixture")
        if regular(arm / "http-status").read_text().strip() != "200":
            raise ValueError(f"{arm_name}: HTTP completion did not return 200")
        logit, start, suffix = final_logits(arm)
        logits[arm_name] = logit.read_bytes()
        texts[arm_name], _ = response(regular(arm / "response.json"))
        safety_rows[arm_name] = safety(arm)
        if safety_rows[arm_name]["executed_binary_sha256"] != manifest["binary_sha256"]:
            raise ValueError(f"{arm_name}: executed binary differs from manifest")
        server = (arm / "server.log").read_text(encoding="utf-8", errors="replace")
        exact_marker = "W8 request cKV store=" in server and "+nvme-direct-slot" in server
        counters = exact_counters(server) if arm_name == "exact" else None
        failures = re.findall(r"W8 (?:cKV append|selected-row read) failed", server)
        if arm_name == "exact" and (not exact_marker or failures):
            raise ValueError("exact arm did not execute the clean W8 path")
        if arm_name == "resident" and (exact_marker or W8_COUNTER_RE.search(server)):
            raise ValueError("resident arm executed W8")
        rows.append({
            "record_type": "w8_smoke_arm",
            "arm": arm_name,
            "prompt_tokens": PROMPT_TOKENS,
            "logit_start": start,
            "logit_suffix": suffix,
            "logit_sha256": hashlib.sha256(logits[arm_name]).hexdigest(),
            "response_text_sha256": hashlib.sha256(texts[arm_name].encode()).hexdigest(),
            "server_log_sha256": sha(arm / "server.log"),
            "trace_sha256": sha(arm / "request.trace"),
            "safety": safety_rows[arm_name],
            "exact_marker": exact_marker,
            "w8_counters": counters,
            "artifacts": artifact_inventory(arm),
        })

    byte_equal = logits["resident"] == logits["exact"]
    floats = array.array("f")
    floats.frombytes(logits["exact"])
    if len(floats) != LOGIT_COUNT or any(not math.isfinite(x) for x in floats):
        raise ValueError("exact logits are malformed or non-finite")
    argmax = max(range(len(floats)), key=floats.__getitem__)
    checks = {
        "matched_inputs": all(sha(root / arm / "request.json") == expected_request for arm in ARMS),
        "resident_path_clean": not rows[0]["exact_marker"],
        "exact_path_observed": rows[1]["exact_marker"],
        "final_logits_byte_identical": byte_equal,
        "completion_text_identical": texts["resident"] == texts["exact"],
        "containment_clean": True,
        "memory_floor_24gib": all(v["minimum_mem_available_gib"] >= 24 for v in safety_rows.values()),
        "no_swap_oom_xid": True,
    }
    summary = {
        "schema": "glm52-w8-exact-smoke-summary-v1",
        "gate": "W8-model-backed-smoke",
        "formula": "PASS iff all checks are true; this is not a 1M capability/performance result",
        "checks": checks,
        "metrics": {
            "prompt_tokens": PROMPT_TOKENS,
            "compared_f32_logits": LOGIT_COUNT,
            "argmax": argmax,
            "resident_min_mem_gib": safety_rows["resident"]["minimum_mem_available_gib"],
            "exact_min_mem_gib": safety_rows["exact"]["minimum_mem_available_gib"],
        },
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Model-backed exact-W8 correctness smoke only; not direct-1M capability, warm disk resume, or performance evidence.",
    }
    if summary["verdict"] != "PASS":
        raise ValueError("W8 smoke acceptance formula failed")
    return rows, summary


def write_terminal(root: pathlib.Path, manifest_path: pathlib.Path,
                   raw_path: pathlib.Path, summary_path: pathlib.Path,
                   rows: list[dict], summary: dict) -> None:
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_path = root / "terminal-receipt.json"
    receipt = {
        "schema": "glm52-w8-exact-smoke-terminal-v1",
        "verdict": summary.get("verdict"),
        "manifest_sha256": sha(regular(manifest_path)),
        "raw_sha256": sha(regular(raw_path)),
        "summary_sha256": sha(regular(summary_path)),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_terminal(root, manifest_path, raw_path, summary_path)


def verify_terminal(root: pathlib.Path, manifest_path: pathlib.Path,
                    raw_path: pathlib.Path, summary_path: pathlib.Path) -> None:
    receipt = strict_json(regular(root / "terminal-receipt.json"))
    expected = {
        "manifest_sha256": sha(regular(manifest_path)),
        "raw_sha256": sha(regular(raw_path)),
        "summary_sha256": sha(regular(summary_path)),
    }
    if receipt.get("schema") != "glm52-w8-exact-smoke-terminal-v1" or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("terminal receipt hash mismatch")
    manifest = strict_json(manifest_path)
    if manifest.get("schema") == "glm52-w8-exact-smoke-manifest-v1":
        receipt_hash = manifest.get("randomness_receipt_sha256")
        if (not isinstance(receipt_hash, str) or
                sha(regular(root / "randomness-receipt.json")) != receipt_hash):
            raise ValueError("terminal randomness receipt mismatch")
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    for row in raw_rows:
        arm = row.get("arm")
        inventory = row.get("artifacts")
        if arm in ARMS:
            verify_inventory(root / arm, inventory)
        elif row.get("record_type") == "w8_smoke_failure":
            verify_inventory(root, inventory)


def write_failure(root: pathlib.Path, manifest_path: pathlib.Path,
                  raw_path: pathlib.Path, summary_path: pathlib.Path,
                  reason: str) -> None:
    row = {
        "record_type": "w8_smoke_failure",
        "failure_reason": reason,
        "artifacts": artifact_inventory(root, strict=False),
    }
    summary = {
        "schema": "glm52-w8-exact-smoke-summary-v1",
        "gate": "W8-model-backed-smoke",
        "formula": "Any failed, interrupted, missing, unsafe, or unscorable arm is FAIL",
        "checks": {"terminal_success": False},
        "verdict": "FAIL",
        "failure_reason": reason,
        "scope": "Preserved terminal failure evidence; not a capability or performance result.",
    }
    write_terminal(root, manifest_path, raw_path, summary_path, [row], summary)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        parent = pathlib.Path(tmp)

        def fixture(name: str) -> pathlib.Path:
            root = parent / name
            root.mkdir()
            request_blob = b'{"model":"default","prompt":"fixture","max_tokens":0}'
            request_sha = hashlib.sha256(request_blob).hexdigest()
            receipt_blob = b'{"round":1,"randomness":"fixture"}'
            receipt_sha = hashlib.sha256(receipt_blob).hexdigest()
            manifest = {
                "schema": "glm52-w8-exact-smoke-manifest-v1",
                "arm_order": list(ARMS),
                "binary_sha256": "a" * 64,
                "model_sha256": "b" * 64,
                "request_sha256": request_sha,
                "randomness_receipt_sha256": receipt_sha,
                "arms": {
                    arm: {
                        "binary_sha256": "a" * 64,
                        "model_sha256": "b" * 64,
                        "request_sha256": request_sha,
                        "context": 8192,
                        "ckv_mode": (
                            "resident" if arm == "resident" else "exact-f32-nvme"
                        ),
                    }
                    for arm in ARMS
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "randomness-receipt.json").write_bytes(receipt_blob)
            blob = array.array("f", [0.0] * LOGIT_COUNT).tobytes()
            for arm in ARMS:
                path = root / arm
                (path / "safety").mkdir(parents=True)
                (path / f"logits.sync1.start0.prompt{PROMPT_TOKENS}.suffix{PROMPT_TOKENS}").write_bytes(blob)
                (path / "request.json").write_bytes(request_blob)
                (path / "http-status").write_text("200\n")
                (path / "response.json").write_text(json.dumps({
                    "choices": [{"text": "ok"}],
                    "usage": {"prompt_tokens": PROMPT_TOKENS,
                              "completion_tokens": 0,
                              "total_tokens": PROMPT_TOKENS},
                }))
                (path / "containment.rc").write_text("0\n")
                (path / "containment.stdout").write_text("SAFE_RUN_DONE rc=0 killed=no dir=/x\n")
                (path / "request.trace").write_text("trace\n")
                (path / "server.log").write_text(
                    "W8 request cKV store=x\n+nvme-direct-slot\n"
                    "ds4: W8 exact request complete append_calls=10 "
                    "appended_rows=100 records=2 selected_read_calls=3 "
                    "selected_rows=12 checksum_validations=2 "
                    "checksum_failures=0 cache_hits=1 cache_misses=2 "
                    "direct_slot_calls=3\n"
                    if arm == "exact" else "resident\n"
                )
                (path / "safety" / "samples.log").write_text(
                    "mem_avail_kb=52428800 eng_rss_kb=1 "
                    "cgroup_current_bytes=1 cgroup_peak_bytes=1 "
                    "cgroup_swap_current_bytes=0\n"
                )
                (path / "safety" / "main.log").write_text(
                    "executed_candidate_verified pid=1 start_ticks=1 path=/x "
                    f"executed_binary_sha256={'a' * 64} device_inode=1:1\n"
                    "cgroup_final current_bytes=0 peak_bytes=1 "
                    "swap_current_bytes=0 events=low 0,high 0,max 0,oom 0,"
                    "oom_kill 0,oom_group_kill 0,\n"
                    "executed candidate was verified alive at least once; "
                    "no identity contradiction observed by the periodic sampler; "
                    "actual cadence is recorded in samples.log; wrapper and "
                    "descendant checks clean\n"
                    "SAFE_RUN end rc=0 killed=no\n"
                )
                (path / "safety" / "kernel.log").write_text("clean\n")
            return root

        valid = fixture("valid")
        score(valid, valid / "manifest.json")

        mutations = []
        root = fixture("missing-marker")
        (root / "exact" / "server.log").write_text("resident\n")
        mutations.append(("missing W8 marker", root))

        root = fixture("unequal-fixture")
        manifest = strict_json(root / "manifest.json")
        manifest["arms"]["exact"]["request_sha256"] = "d" * 64
        (root / "manifest.json").write_text(json.dumps(manifest))
        mutations.append(("unequal fixture", root))

        root = fixture("unequal-logits")
        logit, _, _ = final_logits(root / "exact")
        changed = bytearray(logit.read_bytes())
        changed[0:4] = array.array("f", [1.0]).tobytes()
        logit.write_bytes(changed)
        mutations.append(("unequal logits", root))

        root = fixture("swap")
        (root / "exact" / "safety" / "samples.log").write_text(
            "mem_avail_kb=52428800 eng_rss_kb=1 cgroup_current_bytes=1 "
            "cgroup_peak_bytes=1 cgroup_swap_current_bytes=4096\n"
        )
        mutations.append(("swap use", root))

        root = fixture("actual-unequal-request")
        (root / "exact" / "request.json").write_bytes(b"different")
        mutations.append(("actual unequal request", root))

        root = fixture("bad-http-status")
        (root / "exact" / "http-status").write_text("500\n")
        mutations.append(("bad HTTP status", root))

        root = fixture("bad-usage")
        response = strict_json(root / "exact" / "response.json")
        response["usage"]["prompt_tokens"] = PROMPT_TOKENS - 1
        (root / "exact" / "response.json").write_text(json.dumps(response))
        mutations.append(("bad prompt-token usage", root))

        root = fixture("cgroup-oom")
        main = root / "exact" / "safety" / "main.log"
        main.write_text(main.read_text().replace("oom 0,", "oom 1,"))
        mutations.append(("positive cgroup OOM counter", root))

        root = fixture("marker-only")
        (root / "exact" / "server.log").write_text(
            "W8 request cKV store=x\n+nvme-direct-slot\n"
        )
        mutations.append(("marker-only exact path", root))

        root = fixture("padding-only-selected-rows")
        server = root / "exact" / "server.log"
        server.write_text(server.read_text().replace(
            "selected_rows=12", "selected_rows=0"
        ))
        mutations.append(("padding-only selected-row telemetry", root))

        for label, root in mutations:
            try:
                score(root, root / "manifest.json")
            except ValueError:
                continue
            raise AssertionError(f"{label} mutation passed")

        root = fixture("post-score-mutation")
        rows, summary = score(root, root / "manifest.json")
        write_terminal(root, root / "manifest.json", root / "raw.jsonl",
                       root / "summary.json", rows, summary)
        main = root / "exact" / "safety" / "main.log"
        main.write_text(main.read_text() + "mutated\n")
        try:
            verify_terminal(root, root / "manifest.json", root / "raw.jsonl",
                            root / "summary.json")
        except ValueError:
            pass
        else:
            raise AssertionError("post-score artifact mutation passed")

        root = fixture("post-score-randomness-mutation")
        rows, summary = score(root, root / "manifest.json")
        write_terminal(root, root / "manifest.json", root / "raw.jsonl",
                       root / "summary.json", rows, summary)
        (root / "randomness-receipt.json").write_text("replaced\n")
        try:
            verify_terminal(root, root / "manifest.json", root / "raw.jsonl",
                            root / "summary.json")
        except ValueError:
            pass
        else:
            raise AssertionError("post-score randomness mutation passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--raw", type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--failure-reason")
    parser.add_argument("--verify-terminal", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); print("W8_SMOKE_SCORER_SELFTEST_OK"); return 0
    if not all((args.root, args.manifest, args.raw, args.summary)):
        parser.error("--root, --manifest, --raw and --summary are required")
    if args.verify_terminal:
        verify_terminal(args.root, args.manifest, args.raw, args.summary)
        print("W8_SMOKE_TERMINAL_OK")
        return 0
    if args.failure_reason is not None:
        write_failure(args.root, args.manifest, args.raw, args.summary,
                      args.failure_reason)
        print("FAIL")
        return 2
    rows, summary = score(args.root, args.manifest)
    write_terminal(args.root, args.manifest, args.raw, args.summary, rows, summary)
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, AssertionError, json.JSONDecodeError) as exc:
        print(f"W8 smoke scorer FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
