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


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def response_text(path: pathlib.Path) -> str:
    doc = strict_json(path)
    choices = doc.get("choices") if isinstance(doc, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response must contain exactly one choice")
    text = choices[0].get("text") if isinstance(choices[0], dict) else None
    if not isinstance(text, str):
        raise ValueError("response text is missing")
    return text


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
    fault_text = "\n".join(
        (arm / "safety" / name).read_text(encoding="utf-8", errors="replace")
        for name in ("main.log", "kernel.log")
    )
    if re.search(r"out of memory|oom-kill|killed process|NVRM: Xid", fault_text, re.I):
        raise ValueError(f"{arm.name}: OOM/Xid evidence")
    return {
        "minimum_mem_available_gib": minimum,
        "maximum_engine_rss_gib": max(row[1] for row in samples) / 1024 / 1024,
        "maximum_cgroup_bytes": max(row[3] for row in samples),
        "samples": len(samples),
    }


def score(root: pathlib.Path, manifest_path: pathlib.Path) -> tuple[list[dict], dict]:
    manifest = strict_json(manifest_path)
    if manifest.get("schema") != "glm52-w8-exact-smoke-manifest-v1":
        raise ValueError("manifest schema mismatch")
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
    for arm_name in ARMS:
        arm = root / arm_name
        logit, start, suffix = final_logits(arm)
        logits[arm_name] = logit.read_bytes()
        texts[arm_name] = response_text(arm / "response.json")
        safety_rows[arm_name] = safety(arm)
        server = (arm / "server.log").read_text(encoding="utf-8", errors="replace")
        exact_marker = "W8 request cKV store=" in server and "+nvme-direct-slot" in server
        failures = re.findall(r"W8 (?:cKV append|selected-row read) failed", server)
        if arm_name == "exact" and (not exact_marker or failures):
            raise ValueError("exact arm did not execute the clean W8 path")
        if arm_name == "resident" and exact_marker:
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
        })

    byte_equal = logits["resident"] == logits["exact"]
    floats = array.array("f")
    floats.frombytes(logits["exact"])
    if len(floats) != LOGIT_COUNT or any(not math.isfinite(x) for x in floats):
        raise ValueError("exact logits are malformed or non-finite")
    argmax = max(range(len(floats)), key=floats.__getitem__)
    checks = {
        "matched_inputs": True,
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


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        parent = pathlib.Path(tmp)

        def fixture(name: str) -> pathlib.Path:
            root = parent / name
            root.mkdir()
            manifest = {
                "schema": "glm52-w8-exact-smoke-manifest-v1",
                "arm_order": list(ARMS),
                "binary_sha256": "a" * 64,
                "model_sha256": "b" * 64,
                "request_sha256": "c" * 64,
                "arms": {
                    arm: {
                        "binary_sha256": "a" * 64,
                        "model_sha256": "b" * 64,
                        "request_sha256": "c" * 64,
                        "context": 8192,
                        "ckv_mode": (
                            "resident" if arm == "resident" else "exact-f32-nvme"
                        ),
                    }
                    for arm in ARMS
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            blob = array.array("f", [0.0] * LOGIT_COUNT).tobytes()
            for arm in ARMS:
                path = root / arm
                (path / "safety").mkdir(parents=True)
                (path / f"logits.sync1.start0.prompt{PROMPT_TOKENS}.suffix{PROMPT_TOKENS}").write_bytes(blob)
                (path / "response.json").write_text(json.dumps({"choices": [{"text": "ok"}]}))
                (path / "containment.rc").write_text("0\n")
                (path / "containment.stdout").write_text("SAFE_RUN_DONE rc=0 killed=no dir=/x\n")
                (path / "request.trace").write_text("trace\n")
                (path / "server.log").write_text(
                    "W8 request cKV store=x\n+nvme-direct-slot\n"
                    if arm == "exact" else "resident\n"
                )
                (path / "safety" / "samples.log").write_text(
                    "mem_avail_kb=52428800 eng_rss_kb=1 "
                    "cgroup_current_bytes=1 cgroup_peak_bytes=1 "
                    "cgroup_swap_current_bytes=0\n"
                )
                for log_name in ("main.log", "kernel.log"):
                    (path / "safety" / log_name).write_text("clean\n")
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

        for label, root in mutations:
            try:
                score(root, root / "manifest.json")
            except ValueError:
                continue
            raise AssertionError(f"{label} mutation passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--raw", type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); print("W8_SMOKE_SCORER_SELFTEST_OK"); return 0
    if not all((args.root, args.manifest, args.raw, args.summary)):
        parser.error("--root, --manifest, --raw and --summary are required")
    rows, summary = score(args.root, args.manifest)
    args.raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, AssertionError, json.JSONDecodeError) as exc:
        print(f"W8 smoke scorer FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
