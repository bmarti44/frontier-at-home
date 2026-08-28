#!/usr/bin/env python3
"""Resolve declarative serving profiles into launch snapshots.

The profile system (docs/PROFILE-SCHEMA.md) keys serving configuration by
(model, backend, RAM tier). Each model directory under configs/profiles/
holds one shared model.json (artifacts, engines, backend support) and one
JSON file per profile. This module loads a profile, applies its one-level
`extends` overlay, validates the closed schema fail-closed, substitutes
host placeholders, and emits a fully-resolved snapshot: the exact binary,
argv, env, containment, and safety parameters a launcher must use.

Merge rules for `extends` are deliberately crude: scalars override, objects
merge one level, arrays replace whole, and a base (marked "partial": true)
may not itself extend. The resolver always emits the resolved snapshot, so
inheritance mistakes cannot hide from fixture comparison.

Invoked via scripts/92_resolve_profile.py; importable for tests and for
scripts/52_engine_switch.sh's render step.
"""

from __future__ import annotations

import json
import os
import re
import socket
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_ROOT = REPO_ROOT / "configs/profiles"
HOSTS_ROOT = REPO_ROOT / "configs/hosts"

PLACEHOLDERS = (
    "model",
    "mmproj",
    "draft_model",
    "binary",
    "port",
    "verb",
    "repo",
    "model_root",
    "cache_root",
    "state_root",
)
PLACEHOLDER_RE = re.compile(r"\{(" + "|".join(PLACEHOLDERS) + r")\}")
UNKNOWN_PLACEHOLDER_RE = re.compile(r"^\{[a-z_]+\}$")

MECHANISMS = (
    "systemd-run",
    "setsid-memwatch",
    "delegated-launcher",
    "setsid-watchdog-portable",
)

PROFILE_KEYS = {
    "schema_version", "profile_id", "extends", "partial", "model", "backend",
    "hardware_class", "ram_tier_gib", "min_system_ram_gib", "purpose",
    "status", "engine", "artifact_roles", "launch", "containment", "safety",
    "memory_model", "offload", "context_cap", "port_role", "bench",
    "switch_alias", "verify_on_hardware",
}
LAUNCH_KEYS = {
    "mechanism", "user", "runuser", "delegate", "log_name", "env",
    "diagnostics_unset", "args",
}
CONTAINMENT_KEYS = {
    "unit", "memory_high", "memory_max", "memory_swap_max", "oom_policy",
    "kill_mode", "extra_properties",
}
SAFETY_KEYS = {
    "kill_floor_gib", "minimum_start_gib", "sample_hz",
    "startup_timeout_seconds", "note",
}
MEMORY_MODEL_KEYS = {
    "kv_bytes_per_token", "overhead_gib", "extra_gib", "floor_gib", "basis",
    "resident_weights_gib",
}
STATUS_KEYS = {
    "state", "decision", "qualified_at", "evidence", "basis", "reason",
    "feasibility",
}
BENCH_KEYS = {"stack_label", "historical_stack_labels"}
STATUS_STATES = {"qualified", "estimated", "unsupported"}


class ProfileError(ValueError):
    """Raised for any schema, reference, or rendering failure (fail closed)."""


def _load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
    except OSError as error:
        raise ProfileError(f"cannot read {path}: {error}") from error
    except ValueError as error:
        raise ProfileError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProfileError(f"{path} must contain a JSON object")
    return value


def _require_keys(value: dict, allowed: set, required: set, label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ProfileError(f"{label}: unknown keys {sorted(unknown)}")
    missing = required - set(value)
    if missing:
        raise ProfileError(f"{label}: missing keys {sorted(missing)}")


def load_host(explicit: str | os.PathLike | None = None) -> dict:
    """Select and load the host file, fail closed when none matches."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        named = os.environ.get("FRONTIER_HOST")
        if named:
            candidates.append(HOSTS_ROOT / f"{named}.json")
        else:
            hostname = socket.gethostname().split(".")[0]
            committed = HOSTS_ROOT / f"{hostname}.json"
            if committed.exists():
                candidates.append(committed)
            candidates.append(
                Path.home() / ".config/frontier/host.json"
            )
    for candidate in candidates:
        try:
            present = candidate.exists()
        except OSError:
            present = False
        if present:
            host = _load_json(candidate)
            _require_keys(
                host,
                {"schema_version", "host_id", "os", "arch", "hardware_class",
                 "backends", "ram_gib", "paths", "ports", "versions_lock"},
                {"host_id", "os", "arch", "hardware_class", "backends",
                 "ram_gib", "paths", "ports"},
                f"host file {candidate}",
            )
            return host
    raise ProfileError(
        "no host file found: set FRONTIER_HOST, add configs/hosts/<hostname>.json, "
        "or create ~/.config/frontier/host.json"
    )


def load_model(model_slug: str) -> dict:
    model = _load_json(PROFILES_ROOT / model_slug / "model.json")
    _require_keys(
        model,
        {"schema_version", "model", "identity_manifest", "artifacts",
         "engines", "backend_support"},
        {"model", "artifacts", "engines", "backend_support"},
        f"model.json for {model_slug}",
    )
    if model["model"] != model_slug:
        raise ProfileError(
            f"model.json slug {model['model']!r} does not match directory {model_slug!r}"
        )
    return model


def _merge(base: dict, leaf: dict) -> dict:
    merged = dict(base)
    merged.pop("partial", None)
    for key, value in leaf.items():
        if key == "extends":
            continue
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            inner = dict(merged[key])
            inner.update(value)
            merged[key] = inner
        else:
            merged[key] = value
    return merged


def load_profile(model_slug: str, profile_file: str) -> dict:
    """Load a profile file, apply its one-level extends, validate the schema."""
    directory = PROFILES_ROOT / model_slug
    leaf = _load_json(directory / profile_file)
    if leaf.get("partial"):
        raise ProfileError(f"{profile_file} is a partial base, not servable")
    if "extends" in leaf:
        base_name = leaf["extends"]
        if not (isinstance(base_name, str) and base_name.startswith("_")):
            raise ProfileError(
                f"{profile_file}: extends must name an underscore-prefixed sibling"
            )
        base = _load_json(directory / base_name)
        if base.get("partial") is not True:
            raise ProfileError(f"{base_name} must set \"partial\": true")
        if "extends" in base:
            raise ProfileError(f"{base_name}: a base may not extend (one level only)")
        profile = _merge(base, leaf)
    else:
        profile = dict(leaf)

    _require_keys(
        profile, PROFILE_KEYS,
        {"schema_version", "profile_id", "model", "backend", "hardware_class",
         "ram_tier_gib", "status", "launch", "memory_model", "context_cap",
         "port_role"},
        profile_file,
    )
    if profile["schema_version"] != 4:
        raise ProfileError(f"{profile_file}: schema_version must be 4")
    if profile["model"] != model_slug:
        raise ProfileError(f"{profile_file}: model does not match directory")

    launch = profile["launch"]
    _require_keys(launch, LAUNCH_KEYS, {"mechanism"}, f"{profile_file} launch")
    if launch["mechanism"] not in MECHANISMS:
        raise ProfileError(
            f"{profile_file}: unknown mechanism {launch['mechanism']!r}"
        )
    if "containment" in profile:
        _require_keys(profile["containment"], CONTAINMENT_KEYS, set(),
                      f"{profile_file} containment")
    if "safety" in profile:
        _require_keys(profile["safety"], SAFETY_KEYS,
                      {"kill_floor_gib", "minimum_start_gib"},
                      f"{profile_file} safety")
    _require_keys(profile["memory_model"], MEMORY_MODEL_KEYS,
                  {"kv_bytes_per_token", "overhead_gib", "floor_gib"},
                  f"{profile_file} memory_model")
    _require_keys(profile["status"], STATUS_KEYS, {"state"},
                  f"{profile_file} status")
    if profile["status"]["state"] not in STATUS_STATES:
        raise ProfileError(f"{profile_file}: unknown status state")
    if "bench" in profile:
        _require_keys(profile["bench"], BENCH_KEYS, {"stack_label"},
                      f"{profile_file} bench")
    return profile


def _substitute(value: str, mapping: dict, label: str) -> str:
    rendered = PLACEHOLDER_RE.sub(
        lambda match: str(mapping.get(match.group(1), match.group(0))), value
    )
    leftover = PLACEHOLDER_RE.search(rendered)
    if leftover:
        raise ProfileError(
            f"{label}: placeholder {leftover.group(0)} has no value on this host"
        )
    if UNKNOWN_PLACEHOLDER_RE.match(rendered):
        raise ProfileError(f"{label}: unknown placeholder {rendered}")
    return rendered


def resolve(profile: dict, model: dict, host: dict, verb: str = "start") -> dict:
    """Render a validated profile against a host into a launch snapshot."""
    backend = profile["backend"]
    support = model["backend_support"].get(backend)
    if not (isinstance(support, dict) and support.get("supported")):
        reason = (support or {}).get("reason", "backend not supported")
        raise ProfileError(f"{profile['profile_id']}: {backend}: {reason}")
    if profile["status"]["state"] == "unsupported":
        raise ProfileError(
            f"{profile['profile_id']}: unsupported: "
            f"{profile['status'].get('reason', 'no reason recorded')}"
        )

    paths = host["paths"]
    mapping: dict[str, object] = {
        "repo": paths["repo"],
        "model_root": paths["model_root"],
        "cache_root": paths["cache_root"],
        "state_root": paths["state_root"],
        "verb": verb,
    }
    port_role = profile["port_role"]
    port = host["ports"].get(port_role)
    if port is None:
        raise ProfileError(f"host has no port for role {port_role!r}")
    mapping["port"] = port

    digest_checks = []
    weights_gib = 0.0
    engine = None
    if "engine" in profile:
        engine = model["engines"].get(profile["engine"])
        if engine is None:
            raise ProfileError(f"unknown engine {profile['engine']!r}")
        mapping["binary"] = _substitute(
            engine["binary_path"], mapping, "engine binary_path"
        )
        if engine.get("binary_sha256"):
            digest_checks.append(
                {"path": mapping["binary"], "sha256": engine["binary_sha256"]}
            )
    for role, artifact_name in (profile.get("artifact_roles") or {}).items():
        artifact = model["artifacts"].get(artifact_name)
        if artifact is None:
            raise ProfileError(f"unknown artifact {artifact_name!r} for role {role}")
        if "path" in artifact:
            rendered = _substitute(artifact["path"], mapping, f"artifact {artifact_name}")
            mapping[role] = rendered
            for shard in artifact.get("shards", []):
                digest_checks.append({
                    "path": _substitute(shard["path"], mapping, "shard"),
                    "sha256": shard["sha256"],
                })
            if not artifact.get("shards"):
                if artifact.get("sha256"):
                    digest_checks.append(
                        {"path": rendered, "sha256": artifact["sha256"]}
                    )
                elif artifact.get("identity"):
                    digest_checks.append(
                        {"path": rendered, "identity": artifact["identity"]}
                    )
        weights_gib += float(artifact.get("weights_gib", 0.0))

    launch = profile["launch"]
    label = profile["profile_id"]
    argv = [
        _substitute(arg, mapping, f"{label} args") for arg in launch.get("args", [])
    ]
    env = {
        key: _substitute(value, mapping, f"{label} env {key}")
        for key, value in (launch.get("env") or {}).items()
    }

    memory_model = profile["memory_model"]
    snapshot: dict[str, object] = {
        "profile_id": label,
        "mechanism": launch["mechanism"],
        "status": profile["status"]["state"],
        "port": port,
        "context_cap": profile["context_cap"],
        "stack_label": (profile.get("bench") or {}).get("stack_label"),
        "switch_alias": profile.get("switch_alias"),
        "env": env,
        "diagnostics_unset": list(launch.get("diagnostics_unset", [])),
        "digest_checks": digest_checks,
        "membudget": {
            "weights_gib": round(
                float(memory_model.get("resident_weights_gib", weights_gib)), 2
            ),
            "ctx": profile["context_cap"],
            "kv_bytes_per_token": memory_model["kv_bytes_per_token"],
            "overhead_gib": memory_model["overhead_gib"],
            "extra_gib": memory_model.get("extra_gib", 0),
            "floor_gib": memory_model["floor_gib"],
        },
    }

    mechanism = launch["mechanism"]
    if mechanism == "delegated-launcher":
        snapshot["runuser"] = launch["runuser"]
        snapshot["delegate"] = _substitute(launch["delegate"], mapping, "delegate")
    else:
        snapshot["binary"] = mapping.get("binary")
        snapshot["argv"] = argv
        if snapshot["binary"] is None:
            raise ProfileError(f"{label}: mechanism {mechanism} requires an engine")

    if mechanism == "systemd-run":
        containment = profile["containment"]
        properties = {
            "Type": "exec",
            "User": launch["user"],
            "MemoryHigh": containment["memory_high"],
            "MemoryMax": containment["memory_max"],
            "MemorySwapMax": containment["memory_swap_max"],
            "OOMPolicy": containment["oom_policy"],
            "KillMode": containment["kill_mode"],
            "Delegate": "no",
        }
        properties.update(containment.get("extra_properties", {}))
        snapshot["systemd"] = {
            "unit": containment["unit"],
            "properties": properties,
            "server_log": f"{paths['state_root']}/{launch['log_name']}.server.log",
            "flock": paths["inference_lock"],
        }
    elif mechanism in {"setsid-memwatch", "setsid-watchdog-portable"}:
        safety = profile["safety"]
        snapshot["memory_guard"] = {
            "required_gib": safety["minimum_start_gib"],
            "stable_samples": 3,
            "interval_seconds": 1,
            "timeout_seconds": 180,
        }
        snapshot["memwatch"] = {
            "threshold_gib": safety["kill_floor_gib"],
            "interval_sec": safety.get("sample_hz", 1),
        }
        snapshot["server_log"] = (
            f"{paths['state_root']}/{launch['log_name']}.server.log"
        )
    return snapshot


def usable_gib(hardware_class: str, ram_gib: float, floor_gib: float) -> float | None:
    """Usable-memory model per host class (configs/hardware-matrix.json)."""
    if hardware_class == "mac":
        return min(0.75 * ram_gib, ram_gib - 8)
    if hardware_class in {"spark", "strix", "uma"}:
        return ram_gib - floor_gib - 6
    if hardware_class == "any":
        return ram_gib - 8
    return None  # dgpu tiers use VRAM split policy, not a single number


def feasibility(profile: dict, model: dict, ram_gib: float) -> dict:
    memory_model = profile["memory_model"]
    if "resident_weights_gib" in memory_model:
        # Streaming engines (GLM-5.2 ds4 --ssd-streaming) keep weights on
        # disk; only the declared resident footprint charges against RAM.
        weights = float(memory_model["resident_weights_gib"])
    else:
        weights = 0.0
        for artifact_name in (profile.get("artifact_roles") or {}).values():
            weights += float(
                model["artifacts"].get(artifact_name, {}).get("weights_gib", 0.0)
            )
    kv_gib = profile["context_cap"] * memory_model["kv_bytes_per_token"] / 2**30
    fit = (
        weights + kv_gib + memory_model["overhead_gib"]
        + memory_model.get("extra_gib", 0)
    )
    usable = usable_gib(
        profile["hardware_class"], ram_gib, memory_model["floor_gib"]
    )
    verdict = "unknown"
    if usable is not None:
        if fit > usable:
            verdict = "infeasible"
        elif fit > 0.95 * usable:
            verdict = "estimated-tight"
        else:
            verdict = "feasible"
    return {
        "weights_gib": round(weights, 2),
        "kv_gib": round(kv_gib, 2),
        "fit_gib": round(fit, 2),
        "usable_gib": None if usable is None else round(usable, 2),
        "verdict": verdict,
    }


def list_profiles(model_slug: str | None = None) -> list[tuple[str, str]]:
    """(model_slug, profile_file) pairs for every servable committed profile."""
    pairs = []
    directories = (
        [PROFILES_ROOT / model_slug] if model_slug
        else sorted(p for p in PROFILES_ROOT.iterdir() if p.is_dir())
    )
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            if path.name == "model.json" or path.name.startswith("_"):
                continue
            pairs.append((directory.name, path.name))
    return pairs
