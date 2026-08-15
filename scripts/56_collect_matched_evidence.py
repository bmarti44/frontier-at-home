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


def _verify_campaign_artifacts(profile: dict[str, Any]) -> None:
    bindings = profile.get("artifact_sha256")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("campaign artifact bindings are missing")
    for relative, expected in bindings.items():
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or not SHA256.fullmatch(str(expected))
        ):
            raise ValueError("campaign artifact binding is malformed")
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError("campaign artifact escapes repository") from exc
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"campaign artifact digest mismatch: {relative}")


def _environment_digest(environment: dict[str, str]) -> str:
    canonical = "".join(
        f"{name}={environment[name]}\n" for name in sorted(environment)
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _execution_binding(
    directory: Path,
    expected_environment: dict[str, str],
    expected_binary_sha256: str,
    expected_model_bytes: int,
    expected_model_path: str,
    expected_launch_arguments: list[str],
) -> str:
    environment = _read_json(directory / "process.environment")
    command = _read_json(directory / "process.command")
    try:
        model_identity = (directory / "model.device-inode-size").read_text(
            encoding="ascii"
        ).strip()
    except OSError as exc:
        raise ValueError(f"live model identity is missing in {directory}") from exc
    if (
        not isinstance(environment, dict)
        or environment.get("environment") != expected_environment
        or environment.get("sha256") != _environment_digest(expected_environment)
    ):
        raise ValueError(f"executed environment does not match in {directory}")
    if not isinstance(command, dict):
        raise ValueError(f"executed command is invalid in {directory}")
    argv = command.get("argv")
    expected_arguments = [
        expected_model_path if value == "{model}" else value
        for value in expected_launch_arguments
    ]
    if "{port}" not in expected_arguments:
        raise ValueError("approved launch arguments have no port placeholder")
    port_index = expected_arguments.index("{port}")
    if (
        not isinstance(argv, list)
        or len(argv) != len(expected_arguments) + 1
        or not argv[port_index + 1].isdigit()
        or not 1024 <= int(argv[port_index + 1]) <= 65535
    ):
        raise ValueError(f"executed command does not match in {directory}")
    expected_arguments[port_index] = argv[port_index + 1]
    if (
        argv[1:] != expected_arguments
        or any(not isinstance(value, str) or "\x00" in value for value in argv)
        or command.get("context_cap") != 32_768
        or command.get("model_device_inode_size") != model_identity
    ):
        raise ValueError(f"executed command does not match in {directory}")
    parts = model_identity.split(":")
    if (
        len(parts) != 3
        or any(not value.isdigit() for value in parts)
        or int(parts[2]) != expected_model_bytes
    ):
        raise ValueError(f"live model identity is invalid in {directory}")
    identity = (directory / "process.identity").read_text(encoding="ascii").split()
    if len(identity) != 3 or identity[2] != expected_binary_sha256:
        raise ValueError(f"executed binary does not match in {directory}")
    return model_identity


def _dsv_execution_binding(
    directory: Path,
    expected_binary_sha256: str,
    expected_model_bytes: int,
    expected_runtime_closure: dict[str, str],
    expected_launch_arguments: list[str],
    expected_model_path: str,
) -> str:
    command = _read_json(directory / "process.command")
    try:
        model_identity = (directory / "model.device-inode-size").read_text(
            encoding="ascii"
        ).strip()
    except OSError as exc:
        raise ValueError(f"DeepSeek live model identity is missing in {directory}") from exc
    argv = command.get("argv") if isinstance(command, dict) else None
    runtime_closure = _read_json(directory / "process.runtime-closure.json")
    expected_arguments = [
        expected_model_path if value == "{model}" else value
        for value in expected_launch_arguments
    ]
    if "{port}" not in expected_arguments:
        raise ValueError("approved DeepSeek launch arguments have no port")
    port_index = expected_arguments.index("{port}")
    if (
        not isinstance(argv, list)
        or len(argv) != len(expected_arguments) + 1
        or not argv[port_index + 1].isdigit()
        or not 1024 <= int(argv[port_index + 1]) <= 65535
    ):
        raise ValueError(f"DeepSeek executed command does not match in {directory}")
    expected_arguments[port_index] = argv[port_index + 1]
    parts = model_identity.split(":")
    if (
        argv[1:] != expected_arguments
        or any(not isinstance(value, str) or "\x00" in value for value in argv)
        or command.get("binary_sha256") != expected_binary_sha256
        or command.get("context_cap") != 32_768
        or command.get("model_device_inode_size") != model_identity
        or len(parts) != 3
        or any(not value.isdigit() for value in parts)
        or int(parts[2]) != expected_model_bytes
        or runtime_closure != expected_runtime_closure
    ):
        raise ValueError(f"DeepSeek executed command does not match in {directory}")
    return model_identity


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
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    result = _read_json(directory / "result.json")
    if not isinstance(result, dict) or result.get("suite_valid") is not True:
        raise ValueError(f"benchmark suite is invalid in {directory}")
    metadata = result.get("metadata")
    cells = result.get("cells")
    if (
        not isinstance(metadata, dict)
        or metadata.get("reps") != 2
        or not isinstance(cells, list)
        or len(cells) != 2
    ):
        raise ValueError(f"32K-class benchmark shape is invalid in {directory}")
    cells_by_context: dict[int, dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError(f"32K-class benchmark shape is invalid in {directory}")
        context = cell.get("ctx_tokens")
        if (
            not isinstance(context, int)
            or isinstance(context, bool)
            or context in cells_by_context
            or cell.get("valid") is not True
        ):
            raise ValueError(f"32K-class benchmark shape is invalid in {directory}")
        cells_by_context[context] = cell
    if set(cells_by_context) != {0, 28_672}:
        raise ValueError(f"32K-class benchmark shape is invalid in {directory}")
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
    result_reps: list[list[dict[str, Any]]] = []
    for context in (0, 28_672):
        reps = cells_by_context[context].get("reps")
        if (
            not isinstance(reps, list)
            or len(reps) != 2
            or any(
                not isinstance(rep, dict) or rep.get("valid") is not True
                for rep in reps
            )
        ):
            raise ValueError(
                f"32K-class benchmark reps are incomplete in {directory}"
            )
        result_reps.append(reps)
    return profile, result_reps[0], result_reps[1]


SERVING_WEIGHTS_MANIFEST = ROOT / "weights" / "unsloth-ud-q2_k_xl" / "manifest.json"


def collect_records(
    campaign: Path,
    fixture: Path,
    dsv4_profile_path: Path,
    serving_manifest_path: Path | None = None,
    glm_profile_path: Path | None = None,
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
        or dsv4_profile.get("measured_server_context_cap") != 32_768
        or not isinstance(dsv4_profile.get("matched_model_first_shard_bytes"), int)
        or not isinstance(dsv4_profile.get("runtime_closure_sha256"), dict)
        or not dsv4_profile.get("runtime_closure_sha256")
        or not isinstance(dsv4_profile.get("launch_arguments"), list)
        or not isinstance(dsv4_profile.get("model_path"), str)
    ):
        raise ValueError("approved DeepSeek profile is invalid")
    _verify_campaign_artifacts(dsv4_profile)
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
    glm_profile_path = (
        ROOT / "configs" / "glm52-lossless-plateau-profile.json"
        if glm_profile_path is None
        else glm_profile_path
    ).resolve()
    if glm_profile_path.is_symlink() or not glm_profile_path.is_file():
        raise ValueError("approved GLM profile is missing or unsafe")
    glm_profile = _read_json(glm_profile_path)
    expected_glm_environment = {
        "DS4_CUDA_EXPERT_CACHE_GB": "0",
        "DS4_CUDA_EXPERT_CACHE_PIN": "1",
        "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
        "DS4_CUDA_FETCH_THREADS": "6",
        "DS4_CUDA_IQ2_DOWN_REFERENCE": "1",
        "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
        "DS4_CUDA_STABLE_MODEL_REMAP": "1",
        "DS4_TOKEN_TIMING_LOG": "1",
    }
    runtime = glm_profile.get("runtime") if isinstance(glm_profile, dict) else None
    if (
        not isinstance(glm_profile, dict)
        or glm_profile.get("schema_version") != 3
        or glm_profile.get("profile") != "glm52"
        or not SHA256.fullmatch(str(glm_profile.get("binary_sha256", "")))
        or not SHA256.fullmatch(str(glm_profile.get("model_sha256", "")))
        or glm_profile.get("model_supported_context_cap") != 1_048_576
        or glm_profile.get("measured_server_context_cap") != 32_768
        or "context_cap" in glm_profile
        or not isinstance(glm_profile.get("model_path"), str)
        or glm_profile.get("model_bytes") != 211_075_856_448
        or not isinstance(runtime, dict)
        or runtime.get("engine_environment") != expected_glm_environment
        or runtime.get("launch_arguments")
        != [
            "--cuda", "-m", "{model}", "-c", "32768", "--host",
            "127.0.0.1", "--port", "{port}", "--ssd-streaming",
            "--ssd-streaming-cache-experts", "40GB",
        ]
        or runtime.get("benchmark")
        != {
            "fixture_context_tokens": [0, 28_672],
            "max_completion_tokens": 160,
            "minimum_completion_tokens": 128,
            "raw_token_timing_required": True,
            "request_timeout_seconds": 2700,
            "prefill_timing": "external_request_to_first_token_wall",
        }
        or runtime.get("safety")
        != {
            "kill_floor_gib": 40,
            "minimum_start_gib": 110,
            "sample_hz": 4,
            "swap_max_bytes": 0,
            "timeout_seconds": 5400,
            "virtual_memory_limit_kib": 419_430_400,
        }
    ):
        raise ValueError("approved GLM profile is invalid")
    _verify_campaign_artifacts(glm_profile)
    glm_configuration_sha256 = _sha256(glm_profile_path)
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
    prompt_hashes: dict[tuple[int, int], str] = {}
    glm_model_identity: str | None = None
    dsv_model_identity: str | None = None
    for block, sequence in sorted(directories):
        arm, directory = directories[(block, sequence)]
        profile, short_reps, long_reps = _load_result(directory, fixture)
        cold, warm = short_reps
        result = _read_json(directory / "result.json")
        seed = result["metadata"].get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"benchmark seed is invalid in {directory}")
        seeds.add(seed)
        server_id, binary_sha256, configuration_sha256 = _server_identity(
            directory, profile
        )
        if profile == "dsv4":
            observed_dsv_model = _dsv_execution_binding(
                directory,
                dsv4_profile["binary_sha256"],
                dsv4_profile["matched_model_first_shard_bytes"],
                dsv4_profile["runtime_closure_sha256"],
                dsv4_profile["launch_arguments"],
                dsv4_profile["model_path"],
            )
            if dsv_model_identity is None:
                dsv_model_identity = observed_dsv_model
            elif observed_dsv_model != dsv_model_identity:
                raise ValueError("DeepSeek model device/inode/size changed between arms")
            binary_sha256 = dsv4_profile["binary_sha256"]
            configuration_sha256 = dsv4_profile["configuration_sha256"]
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
                raise ValueError(f"DeepSeek safety log is missing in {directory}") from exc
            if "SAFE_RUN_DONE rc=0" not in safety or re.search(
                r"\bFATAL\b|KILL_FLOOR|oom_kill", safety, re.IGNORECASE
            ):
                raise ValueError(f"DeepSeek safety wrapper failed in {directory}")
        else:
            if binary_sha256 != glm_profile["binary_sha256"]:
                raise ValueError(f"GLM binary does not match approved profile in {directory}")
            expected_runtime = (
                "context_cap=32768\n"
                "expert_cache_gib=0\n"
                "iq2_reference=1\n"
                "no_expert_tiles=0\n"
                "stable_model_remap=1\n"
                f"model_sha256={glm_profile['model_sha256']}\n"
            )
            try:
                observed_runtime = (directory / "runtime.config").read_text(
                    encoding="ascii"
                )
            except OSError as exc:
                raise ValueError(
                    f"GLM runtime configuration is missing in {directory}"
                ) from exc
            if observed_runtime != expected_runtime:
                raise ValueError(
                    f"GLM runtime configuration does not match approved profile in {directory}"
                )
            observed_model_identity = _execution_binding(
                directory,
                expected_glm_environment,
                glm_profile["binary_sha256"],
                glm_profile["model_bytes"],
                glm_profile["model_path"],
                runtime["launch_arguments"],
            )
            if glm_model_identity is None:
                glm_model_identity = observed_model_identity
            elif observed_model_identity != glm_model_identity:
                raise ValueError("GLM model device/inode/size changed between arms")
            configuration_sha256 = glm_configuration_sha256
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
            expected_environment_sha256 = _environment_digest(
                expected_glm_environment
            )
            if (
                f"executed_environment_sha256={expected_environment_sha256}"
                not in safety
            ):
                raise ValueError(
                    f"GLM safety environment binding is missing in {directory}"
                )
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
        for context, reps in ((0, short_reps), (28_672, long_reps)):
            for rep_index, rep in enumerate(reps):
                prompt_hash = rep.get("prompt_sha256")
                if not isinstance(prompt_hash, str) or not SHA256.fullmatch(prompt_hash):
                    raise ValueError(f"prompt bytes are unbound in {directory}")
                key = (context, rep_index)
                prior = prompt_hashes.setdefault(key, prompt_hash)
                if prior != prompt_hash:
                    raise ValueError("matched arms use unequal prompt bytes")
                prompt_tokens = rep.get("prompt_tokens")
                production_prompt_tokens = rep.get("production_prompt_tokens")
                completion_tokens = rep.get("server_completion_tokens")
                if (
                    not isinstance(prompt_tokens, int)
                    or isinstance(prompt_tokens, bool)
                    or prompt_tokens <= 0
                    or production_prompt_tokens != prompt_tokens
                    or not isinstance(completion_tokens, int)
                    or isinstance(completion_tokens, bool)
                    or completion_tokens < 128
                    or prompt_tokens + completion_tokens > 32_768
                ):
                    raise ValueError(
                        f"production prompt accounting is invalid in {directory}"
                    )
        evaluated_tokens = 0
        prefill_seconds = 0.0
        for rep in long_reps:
            prompt_tokens = rep.get("prompt_tokens")
            if (
                not isinstance(prompt_tokens, int)
                or isinstance(prompt_tokens, bool)
                or prompt_tokens < 28_672
            ):
                raise ValueError(
                    f"32K-class production prompt tokens are invalid in {directory}"
                )
            evaluated_tokens += prompt_tokens
            prefill_seconds += _finite(
                rep.get("ttft_s"), "32K-class external prefill wall time"
            )
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
                "evaluated_tokens": evaluated_tokens,
                "prefill_seconds": prefill_seconds,
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
        default=ROOT / "configs" / "dsv4-matched-32k-profile.json",
    )
    parser.add_argument(
        "--glm-profile",
        type=Path,
        default=ROOT / "configs" / "glm52-lossless-plateau-profile.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = collect_records(
            args.campaign,
            args.fixture,
            args.dsv4_profile,
            None,
            args.glm_profile,
        )
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
        glm_profile = _read_json(args.glm_profile)
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
                    "glm_binary_sha256": glm_profile["binary_sha256"],
                    "glm_model_sha256": glm_profile["model_sha256"],
                    "glm_profile_sha256": _sha256(args.glm_profile),
                    "prefill_timing": "external_request_to_first_token_wall",
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
