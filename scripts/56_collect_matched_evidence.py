#!/usr/bin/env python3
"""Convert one complete matched campaign into strict controller raw records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm52_goal import score_registered_gate  # noqa: E402


LABEL = re.compile(r"block([0-4])-seq([0-3])-arm([AB])\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
BOOT_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
FAULT = re.compile(
    r"NV_ERR_NO_MEMORY|NVRM.*Xid|oom-kill|Out of memory: Killed process|"
    r"Killed process .*total-vm",
    re.IGNORECASE,
)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
            object_pairs_hook=_pairs,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} is not finite and positive")
    return result


def _memory_min(path: Path, pattern: re.Pattern[str], scale: float) -> float:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"missing or unreadable memory samples {path}") from exc
    values = []
    for match in pattern.finditer(text):
        values.append(float(match.group(1)) / scale)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"memory samples are missing or invalid in {path}")
    return min(values)


def _server_identity(directory: Path, profile: str) -> tuple[str, str, str]:
    if profile == "glm52":
        try:
            parts = (directory / "process.identity").read_text(
                encoding="ascii"
            ).split()
            boot_id = (directory / "host.boot_id").read_text(
                encoding="ascii"
            ).strip()
        except OSError as exc:
            raise ValueError(f"GLM process identity is missing in {directory}") from exc
        if (
            len(parts) != 3
            or not all(part.isdigit() for part in parts[:2])
            or not SHA256.fullmatch(parts[2])
            or not BOOT_ID.fullmatch(boot_id)
        ):
            raise ValueError(f"GLM process identity is invalid in {directory}")
        runtime = directory / "runtime.config"
        if not runtime.is_file() or runtime.is_symlink():
            raise ValueError(f"GLM runtime configuration is missing in {directory}")
        identity = hashlib.sha256(
            f"{boot_id}:{parts[0]}:{parts[1]}".encode()
        ).hexdigest()
        return identity, parts[2], _sha256(runtime)

    identity = _read_json(directory / "process.identity.json")
    if not isinstance(identity, dict):
        raise ValueError(f"DeepSeek process identity is invalid in {directory}")
    required_true = ("server_alive", "memwatch_alive", "watchdog_armed", "healthy")
    if any(identity.get(field) is not True for field in required_true):
        raise ValueError(f"DeepSeek supervision is incomplete in {directory}")
    boot_id = identity.get("boot_id")
    pid = identity.get("server_pid")
    ticks = identity.get("server_start_ticks")
    if (
        not isinstance(boot_id, str)
        or not BOOT_ID.fullmatch(boot_id)
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 1
        or not isinstance(ticks, int)
        or isinstance(ticks, bool)
        or ticks <= 0
    ):
        raise ValueError(f"DeepSeek process identity is malformed in {directory}")
    server_id = hashlib.sha256(f"{boot_id}:{pid}:{ticks}".encode()).hexdigest()
    return server_id, "", ""


def _load_result(
    directory: Path, fixture: Path
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    result = _read_json(directory / "result.json")
    if not isinstance(result, dict) or result.get("suite_valid") is not True:
        raise ValueError(f"benchmark suite is invalid in {directory}")
    metadata = result.get("metadata")
    cells = result.get("cells")
    if (
        not isinstance(metadata, dict)
        or metadata.get("reps") != 2
        or not isinstance(cells, list)
        or len(cells) != 1
        or not isinstance(cells[0], dict)
        or cells[0].get("ctx_tokens") != 0
        or cells[0].get("valid") is not True
    ):
        raise ValueError(f"benchmark shape is invalid in {directory}")
    fixture_value = metadata.get("fixture_path")
    if not isinstance(fixture_value, str) or not fixture_value:
        raise ValueError(f"benchmark fixture path is invalid in {directory}")
    observed_fixture = Path(fixture_value)
    if not observed_fixture.is_absolute():
        observed_fixture = ROOT / observed_fixture
    if observed_fixture.resolve() != fixture.resolve():
        raise ValueError(f"benchmark fixture does not match in {directory}")
    model = metadata.get("model")
    if model == "glm-5.2":
        profile = "glm52"
    elif model == "deepseek-v4-flash":
        profile = "dsv4"
    else:
        raise ValueError(f"benchmark model identity is invalid in {directory}")
    reps = cells[0].get("reps")
    if (
        not isinstance(reps, list)
        or len(reps) != 2
        or any(not isinstance(rep, dict) or rep.get("valid") is not True for rep in reps)
    ):
        raise ValueError(f"cold/warm benchmark reps are incomplete in {directory}")
    return profile, reps[0], reps[1]


SERVING_WEIGHTS_MANIFEST = ROOT / "weights" / "unsloth-ud-q2_k_xl" / "manifest.json"


def collect_records(
    campaign: Path,
    fixture: Path,
    dsv4_profile_path: Path,
    serving_manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    campaign = campaign.resolve()
    fixture = fixture.resolve()
    if (
        not campaign.is_dir()
        or campaign.is_symlink()
        or not fixture.is_file()
        or fixture.is_symlink()
    ):
        raise ValueError("campaign or fixture path is unsafe")
    dsv4_profile = _read_json(dsv4_profile_path)
    if (
        not isinstance(dsv4_profile, dict)
        or dsv4_profile.get("schema_version") != 3
        or dsv4_profile.get("profile") != "dsv4"
        or not SHA256.fullmatch(str(dsv4_profile.get("binary_sha256", "")))
        or not SHA256.fullmatch(
            str(dsv4_profile.get("configuration_sha256", ""))
        )
        or not SHA256.fullmatch(
            str(dsv4_profile.get("serving_weights_manifest_sha256", ""))
        )
    ):
        raise ValueError("approved DeepSeek profile is invalid")
    # The DeepSeek arm is llama.cpp serving UD-Q2_K_XL. binary_sha256 and
    # configuration_sha256 identify the engine and its unit, and neither moves when
    # the GGUF generation underneath them is replaced -- the 0731 swap changed no
    # value this collector previously recorded. Without this check a GLM candidate
    # measured against pre-0731 and one measured against 0731 both claim the same
    # DeepSeek baseline. Recording the profile's own digest is not sufficient
    # either: a profile edit alone would relabel the baseline, so the manifest on
    # disk is hashed and compared.
    serving_manifest = (
        SERVING_WEIGHTS_MANIFEST
        if serving_manifest_path is None
        else serving_manifest_path
    ).resolve()
    if serving_manifest.is_symlink() or not serving_manifest.is_file():
        raise ValueError("serving weights manifest is missing or unsafe")
    serving_manifest_sha256 = _sha256(serving_manifest)
    if serving_manifest_sha256 != dsv4_profile["serving_weights_manifest_sha256"]:
        raise ValueError(
            "served GGUF generation does not match the approved DeepSeek profile: "
            f"manifest {serving_manifest_sha256} != profile "
            f"{dsv4_profile['serving_weights_manifest_sha256']}"
        )
    fixture_sha256 = _sha256(fixture)
    directories: dict[tuple[int, int], tuple[str, Path]] = {}
    for path in campaign.iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        match = LABEL.fullmatch(path.name)
        if match is None:
            continue
        block, sequence, arm = int(match.group(1)), int(match.group(2)), match.group(3)
        key = (block, sequence)
        if key in directories:
            raise ValueError("matched campaign contains duplicate block/sequence")
        directories[key] = (arm, path)
    expected = {(block, sequence) for block in range(5) for sequence in range(4)}
    if set(directories) != expected:
        raise ValueError("matched campaign does not contain exactly 20 arms")

    records: list[dict[str, Any]] = []
    seeds: set[int] = set()
    for block, sequence in sorted(directories):
        arm, directory = directories[(block, sequence)]
        profile, cold, warm = _load_result(directory, fixture)
        result = _read_json(directory / "result.json")
        seed = result["metadata"].get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"benchmark seed is invalid in {directory}")
        seeds.add(seed)
        server_id, binary_sha256, configuration_sha256 = _server_identity(
            directory, profile
        )
        if profile == "dsv4":
            binary_sha256 = dsv4_profile["binary_sha256"]
            configuration_sha256 = dsv4_profile["configuration_sha256"]
            available_memory = _memory_min(
                directory / "memwatch.segment.log",
                re.compile(r"\bmem_available_gib=([0-9]+(?:\.[0-9]+)?)\b"),
                1.0,
            )
        else:
            available_memory = _memory_min(
                directory / "samples.log",
                re.compile(r"\bmem_avail_kb=([0-9]+)\b"),
                1_048_576.0,
            )
            try:
                safety = (directory / "safety.main.log").read_text(
                    encoding="utf-8"
                )
            except OSError as exc:
                raise ValueError(f"GLM safety log is missing in {directory}") from exc
            if "SAFE_RUN_DONE rc=0" not in safety or re.search(
                r"\bFATAL\b|KILL_FLOOR|oom_kill", safety, re.IGNORECASE
            ):
                raise ValueError(f"GLM safety wrapper failed in {directory}")
        try:
            kernel = (directory / "kernel.log").read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"kernel evidence is missing in {directory}") from exc
        if FAULT.search(kernel):
            raise ValueError(f"kernel GPU/OOM fault invalidates {directory}")

        timestamps_ns = warm.get("token_timestamps_ns")
        if (
            not isinstance(timestamps_ns, list)
            or len(timestamps_ns) < 128
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in timestamps_ns
            )
            or any(
                right <= left
                for left, right in zip(timestamps_ns, timestamps_ns[1:])
            )
        ):
            raise ValueError(f"warm token timestamps are invalid in {directory}")
        prompt_tokens = warm.get("prompt_tokens")
        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens <= 0
        ):
            raise ValueError(f"evaluated prompt tokens are invalid in {directory}")
        warm_ttft = _finite(warm.get("ttft_s"), "warm TTFT")
        cold_ttft = _finite(cold.get("ttft_s"), "cold TTFT")
        _finite(warm.get("prefill_tok_s"), "warm prefill rate")
        records.append(
            {
                "record_type": "matched_arm",
                "block": block,
                "sequence": sequence,
                "arm": arm,
                "profile": profile,
                "server_boot_id": server_id,
                "fixture_sha256": fixture_sha256,
                "binary_sha256": binary_sha256,
                "configuration_sha256": configuration_sha256,
                "token_timestamps": [
                    value / 1_000_000_000 for value in timestamps_ns
                ],
                "evaluated_tokens": prompt_tokens,
                "prefill_seconds": warm_ttft,
                "warm_ttft_seconds": warm_ttft,
                "cold_ttft_seconds": cold_ttft,
                "available_memory_gib": available_memory,
                "truncated": False,
                "oom": False,
                "xid": False,
                "failures": [],
            }
        )
    if len(seeds) != 1:
        raise ValueError("matched campaign uses unequal seeds")
    score_registered_gate("parity", "parity.performance.v1", records)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--dsv4-profile",
        type=Path,
        default=ROOT / "configs" / "dsv4-profile.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = collect_records(args.campaign, args.fixture, args.dsv4_profile)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        # The served DeepSeek generation goes in a sidecar rather than in each
        # matched_arm record: glm52_goal._score_parity pins the record key set and
        # requires exactly 20 records, and widening a GLM scorer's schema from a
        # DeepSeek change is precisely the cross-campaign coupling this repository
        # keeps splitting apart. The value is constant across arms anyway, and
        # collect_records has already refused to produce these records at all if
        # the live manifest disagreed with the approved profile.
        profile = _read_json(args.dsv4_profile)
        identity = args.out.with_suffix(args.out.suffix + ".identity.json")
        with identity.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "record_type": "matched_campaign_identity",
                    "dsv4_binary_sha256": profile["binary_sha256"],
                    "dsv4_configuration_sha256": profile["configuration_sha256"],
                    "dsv4_serving_weights_manifest_sha256": profile[
                        "serving_weights_manifest_sha256"
                    ],
                    "dsv4_serving_weights_release": profile.get(
                        "serving_weights_release"
                    ),
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError) as exc:
        print(f"56_collect_matched_evidence.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
