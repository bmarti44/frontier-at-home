#!/usr/bin/env python3
"""Generic profile path for scripts/52_engine_switch.sh.

A new switch alias needs only a profile under configs/profiles/ that carries
``switch_alias`` and a ``"switch": {"mode": "generic", ...}`` block (see
docs/PROFILE-SCHEMA.md). The six legacy aliases keep their dedicated switch
code and can never be claimed here.

Every command fails closed: any schema, discovery, host, or artifact problem
exits non-zero with the reason on stderr and nothing on stdout.

  plan ALIAS                  NUL-delimited key=value launch/readiness plan
  verify ALIAS                refuse non-qualified profiles, walk the digest
                              inventory (full sha256 or sampled identity), and
                              print the verified stat identities as JSON
  revalidate ALIAS JSON       re-check those stat identities right before launch
  approved-sha ALIAS          the engine binary sha256 the profile approves
  check-models ID BODY        /v1/models body must list exactly ID
  check-context SPEC BODY     context topology check (spec from the plan)

The switch runs this through its clean_python allowlisted environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import profile_resolver  # noqa: E402

LEGACY_ALIASES = frozenset(
    {"dsv4", "glm52", "qwen38", "qwen38-1m", "laguna", "glm53-1m"}
)
# Switch verbs share the alias namespace on the command line.
RESERVED_ALIASES = frozenset({"status", "stop", "restore", "render"})
ALIAS_RE = re.compile(r"[a-z0-9][a-z0-9.-]{0,31}")
# Names the legacy code paths own (units, state-file stems).
LEGACY_UNITS = frozenset(
    {"qwen38-engine", "laguna-engine", "glm53-engine",
     "deepseek-v4-flash-llamacpp", "dsv4-engine-restore"}
)
LEGACY_LOG_NAMES = frozenset({"dsv4", "glm52", "qwen38", "laguna", "glm53"})
NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
USER_RE = re.compile(r"[a-z_][a-z0-9_-]{0,31}")
# Lowercase: the switch compares against lowercased /v1/models ids.
SERVED_ID_RE = re.compile(r"[a-z0-9][a-z0-9._:/-]{0,127}")
HEALTH_PATH_RE = re.compile(r"/[A-Za-z0-9._/-]{0,127}")
SWITCH_KEYS = frozenset({"mode", "served_model_id", "health_path", "context_check"})
CONTEXT_KINDS = {
    "slots": ({"kind", "slots", "n_ctx"}, "/slots"),
    "model_card": ({"kind", "max_model_len"}, "/v1/models"),
}
ARTIFACT_ROLES = ("model", "mmproj", "draft_model")
# Placeholders the bash launcher substitutes (launch_systemd_profile).
ARG_PLACEHOLDERS = frozenset({"model", "port", "mmproj", "draft_model"})
ENV_PLACEHOLDERS = frozenset({"port", "repo", "cache_root"})
MAX_SAMPLE_BYTES = 64 * 1024 * 1024
INFERENCE_LOCK = "/run/lock/frontier-at-home/inference.lock"


class SwitchError(ValueError):
    """Any generic-switch contract violation (fail closed)."""


def check_alias(alias: object) -> str:
    if not isinstance(alias, str) or not ALIAS_RE.fullmatch(alias):
        raise SwitchError(f"invalid switch alias {alias!r}")
    if alias in LEGACY_ALIASES:
        raise SwitchError(f"{alias} is a legacy alias with dedicated switch code")
    if alias in RESERVED_ALIASES:
        raise SwitchError(f"{alias} is a reserved switch verb")
    return alias


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SwitchError(f"{label} must be a positive integer")
    return value


def validate_context_check(spec: object) -> dict | None:
    if spec is None:
        return None
    if not isinstance(spec, dict) or spec.get("kind") not in CONTEXT_KINDS:
        raise SwitchError(f"switch.context_check has an unknown kind: {spec!r}")
    keys, _path = CONTEXT_KINDS[spec["kind"]]
    if set(spec) != keys:
        raise SwitchError(
            f"switch.context_check {spec['kind']} needs exactly {sorted(keys)}"
        )
    for key in keys - {"kind"}:
        _positive_int(spec[key], f"switch.context_check.{key}")
    return dict(spec)


def validate_switch_block(block: object) -> dict:
    if not isinstance(block, dict):
        raise SwitchError("switch must be an object")
    unknown = set(block) - SWITCH_KEYS
    if unknown:
        raise SwitchError(f"switch: unknown keys {sorted(unknown)}")
    if block.get("mode") != "generic":
        raise SwitchError("switch.mode must be \"generic\"")
    served = block.get("served_model_id")
    if not isinstance(served, str) or not SERVED_ID_RE.fullmatch(served):
        raise SwitchError("switch.served_model_id must be a model id string")
    health = block.get("health_path")
    if health is not None and (
        not isinstance(health, str) or not HEALTH_PATH_RE.fullmatch(health)
        or ".." in health.split("/")
    ):
        raise SwitchError("switch.health_path must be null or a local /path")
    return {
        "mode": "generic",
        "served_model_id": served,
        "health_path": health,
        "context_check": validate_context_check(block.get("context_check")),
    }


def committed_entries() -> list[tuple[str, str, dict]]:
    """(model_slug, profile_file, loaded profile) for every committed profile.

    Any unreadable or invalid profile fails discovery: a broken file might be
    the one that carries the alias.
    """
    entries = []
    for model_slug, profile_file in profile_resolver.list_profiles():
        try:
            profile = profile_resolver.load_profile(model_slug, profile_file)
        except profile_resolver.ProfileError as error:
            raise SwitchError(f"profile discovery failed: {error}") from error
        entries.append((model_slug, profile_file, profile))
    return entries


def discover(alias: str, entries) -> tuple[str, dict]:
    check_alias(alias)
    matches = [
        (f"configs/profiles/{slug}/{name}", profile)
        for slug, name, profile in entries
        if profile.get("switch_alias") == alias
    ]
    if not matches:
        raise SwitchError(f"no profile carries switch_alias {alias}")
    if len(matches) > 1:
        raise SwitchError(
            f"switch_alias {alias} is claimed by "
            + " and ".join(relpath for relpath, _ in matches)
        )
    relpath, profile = matches[0]
    if "switch" not in profile:
        raise SwitchError(f"{relpath} lacks a generic switch block")
    validate_switch_block(profile["switch"])
    return relpath, profile


def check_launch_placeholders(profile: dict) -> None:
    launch = profile.get("launch") or {}
    for label, values, allowed in (
        ("args", launch.get("args") or [], ARG_PLACEHOLDERS),
        ("env", list((launch.get("env") or {}).values()), ENV_PLACEHOLDERS),
    ):
        for value in values:
            for name in profile_resolver.PLACEHOLDER_RE.findall(str(value)):
                if name not in allowed:
                    raise SwitchError(
                        f"launch.{label} placeholder {{{name}}} is not "
                        f"substituted by the switch launcher"
                    )


def build_plan(alias: str, *, repo: str, state: str, port: int,
               host_path: str | os.PathLike, test_paths: dict | None = None,
               entries=None) -> dict:
    """Resolve one generic alias against the Spark host into a switch plan."""
    relpath, profile = discover(
        alias, committed_entries() if entries is None else entries)
    switch = validate_switch_block(profile["switch"])
    try:
        host = profile_resolver.load_host(host_path)
    except profile_resolver.ProfileError as error:
        raise SwitchError(str(error)) from error
    host = json.loads(json.dumps(host))
    paths = host["paths"]
    if test_paths is not None:
        paths.update(test_paths)
        paths["repo"] = repo
        paths["state_root"] = state
    else:
        if paths.get("repo") != repo:
            raise SwitchError(f"host repo {paths.get('repo')} is not the switch repo {repo}")
        if paths.get("state_root") != state:
            raise SwitchError(
                f"host state_root {paths.get('state_root')} is not the switch state {state}")
    if paths.get("inference_lock") != INFERENCE_LOCK:
        raise SwitchError("host inference_lock is not the switch's lock")
    try:
        model = profile_resolver.load_model(profile["model"])
        snapshot = profile_resolver.resolve(profile, model, host)
    except (profile_resolver.ProfileError, KeyError, TypeError) as error:
        raise SwitchError(f"{relpath}: {error}") from error
    if snapshot["mechanism"] != "systemd-run":
        raise SwitchError(f"{relpath}: generic aliases launch via systemd-run only")
    if snapshot["port"] != port:
        raise SwitchError(f"{relpath}: port {snapshot['port']} is not switch port {port}")
    check_launch_placeholders(profile)

    roles = profile.get("artifact_roles") or {}
    unknown_roles = set(roles) - set(ARTIFACT_ROLES)
    if "model" not in roles or unknown_roles:
        raise SwitchError(
            f"{relpath}: artifact_roles must name model (and only {ARTIFACT_ROLES})")
    role_paths = {}
    for role in ARTIFACT_ROLES:
        role_paths[role] = ""
        if role in roles:
            # Same substitution the resolver applied to the digest paths.
            try:
                role_paths[role] = profile_resolver._substitute(
                    model["artifacts"][roles[role]]["path"], dict(paths), role)
            except profile_resolver.ProfileError as error:
                raise SwitchError(f"{relpath}: {error}") from error

    engine = model["engines"][profile["engine"]]
    binary_sha = engine.get("binary_sha256")
    if not isinstance(binary_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", binary_sha):
        raise SwitchError(f"{relpath}: engine binary_sha256 is not pinned")

    launch = profile["launch"]
    user = launch.get("user")
    if not isinstance(user, str) or not USER_RE.fullmatch(user) or user == "root":
        raise SwitchError(f"{relpath}: launch.user must be a non-root account")
    log_name = launch.get("log_name")
    if (not isinstance(log_name, str) or not NAME_RE.fullmatch(log_name)
            or log_name in LEGACY_LOG_NAMES):
        raise SwitchError(f"{relpath}: launch.log_name is invalid or legacy-owned")
    unit_name = snapshot["systemd"]["unit"]
    if (not isinstance(unit_name, str) or not NAME_RE.fullmatch(unit_name)
            or unit_name in LEGACY_UNITS):
        raise SwitchError(f"{relpath}: containment.unit is invalid or legacy-owned")
    safety = profile.get("safety") or {}
    required = _positive_int(safety.get("minimum_start_gib"), "safety.minimum_start_gib")
    floor = _positive_int(safety.get("kill_floor_gib"), "safety.kill_floor_gib")
    timeout = _positive_int(
        safety.get("startup_timeout_seconds"), "safety.startup_timeout_seconds")

    context = switch["context_check"]
    context_spec = ""
    context_path = ""
    if context is not None:
        spec = dict(context)
        if spec["kind"] == "model_card":
            spec["served_model_id"] = switch["served_model_id"]
        context_spec = json.dumps(spec, separators=(",", ":"), sort_keys=True)
        context_path = CONTEXT_KINDS[context["kind"]][1]

    return {
        "alias": alias,
        "relpath": relpath,
        "status": snapshot["status"],
        "binary": snapshot["binary"],
        "binary_sha256": binary_sha,
        "model": role_paths["model"],
        "mmproj": role_paths["mmproj"],
        "draft_model": role_paths["draft_model"],
        "unit": f"{unit_name}.service",
        "unit_name": unit_name,
        "user": user,
        "log_name": log_name,
        "required_gib": str(required),
        "floor_gib": str(floor),
        "timeout": str(timeout),
        "served_model_id": switch["served_model_id"],
        "health_path": switch["health_path"] or "",
        "context_check": context_spec,
        "context_path": context_path,
        "digest_checks": snapshot["digest_checks"],
    }


def _stat_fields(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _open_regular(path: str):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise SwitchError(f"cannot open artifact {path}: {error}") from error
    stream = os.fdopen(descriptor, "rb")
    before = os.fstat(stream.fileno())
    if not stat.S_ISREG(before.st_mode):
        stream.close()
        raise SwitchError(f"artifact is not a regular file: {path}")
    return stream, before


def _verify_full(path: str, expected: str) -> list[int]:
    stream, before = _open_regular(path)
    digest = hashlib.sha256()
    with stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if _stat_fields(before) != _stat_fields(after):
        raise SwitchError(f"artifact changed during verification: {path}")
    if digest.hexdigest() != expected:
        raise SwitchError(f"artifact hash is not approved: {path}")
    return _stat_fields(after)


def _verify_sampled(path: str, identity: object) -> list[int]:
    if not isinstance(identity, dict):
        raise SwitchError(f"sampled identity for {path} is not an object")
    first = identity.get("first_bytes")
    if (isinstance(first, bool) or not isinstance(first, int)
            or not 0 < first <= MAX_SAMPLE_BYTES):
        raise SwitchError(f"sampled identity for {path} has an invalid first_bytes")
    for key in ("size_bytes", "device", "inode"):
        if isinstance(identity.get(key), bool) or not isinstance(identity.get(key), int):
            raise SwitchError(f"sampled identity for {path} lacks {key}")
    prefix_sha = identity.get("first_bytes_sha256")
    if not isinstance(prefix_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", prefix_sha):
        raise SwitchError(f"sampled identity for {path} lacks first_bytes_sha256")
    stream, before = _open_regular(path)
    with stream:
        prefix = stream.read(first)
        after = os.fstat(stream.fileno())
    if _stat_fields(before) != _stat_fields(after):
        raise SwitchError(f"artifact changed during verification: {path}")
    if (after.st_size, after.st_dev, after.st_ino) != (
            identity["size_bytes"], identity["device"], identity["inode"]):
        raise SwitchError(f"artifact stat identity is not approved: {path}")
    if len(prefix) != first:
        raise SwitchError(f"artifact prefix is short: {path}")
    if hashlib.sha256(prefix).hexdigest() != prefix_sha:
        raise SwitchError(f"artifact prefix hash is not approved: {path}")
    return _stat_fields(after)


def _launch_paths(plan: dict) -> list[str]:
    return [plan[key] for key in ("binary", "model", "mmproj", "draft_model")
            if plan.get(key)]


def _approved_target(plan: dict, path: str) -> bool:
    return any(path == root or os.path.dirname(path) == root
               for root in _launch_paths(plan))


def verify_artifacts(plan: dict) -> dict[str, list[int]]:
    """Refuse non-qualified profiles, then verify every digest check."""
    if plan.get("status") != "qualified":
        raise SwitchError(
            f"{plan.get('alias')} profile status is {plan.get('status')}, not "
            "qualified; refusing to switch (DeepSeek V4 stays the default "
            "until the profile passes its gates)"
        )
    checks = plan.get("digest_checks")
    if not isinstance(checks, list) or not checks:
        raise SwitchError("digest inventory is empty")
    identities: dict[str, list[int]] = {}
    for check in checks:
        path = check.get("path") if isinstance(check, dict) else None
        if not isinstance(path, str) or not os.path.isabs(path):
            raise SwitchError(f"digest check lacks an absolute path: {check!r}")
        if not _approved_target(plan, path):
            raise SwitchError(f"digest target escapes the launch artifacts: {path}")
        if path in identities:
            raise SwitchError(f"digest inventory repeats {path}")
        if "sha256" in check and "identity" not in check:
            expected = check["sha256"]
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise SwitchError(f"digest check for {path} has a malformed sha256")
            identities[path] = _verify_full(path, expected)
        elif "identity" in check and "sha256" not in check:
            identities[path] = _verify_sampled(path, check["identity"])
        else:
            raise SwitchError(f"digest check for {path} needs sha256 or identity")
    if plan["binary"] not in identities:
        raise SwitchError("engine binary digest is not pinned")
    for role in ("model", "mmproj", "draft_model"):
        root = plan.get(role)
        if root and not any(path == root or os.path.dirname(path) == root
                            for path in identities):
            raise SwitchError(f"{role} artifact digest is not pinned")
    return identities


def revalidate(plan: dict, identities: object) -> None:
    """Re-check approved stat identities immediately before launch."""
    if not isinstance(identities, dict) or plan["binary"] not in identities:
        raise SwitchError("verified inventory lacks the engine binary")
    if plan["model"] and not any(
            path == plan["model"] or os.path.dirname(path) == plan["model"]
            for path in identities):
        raise SwitchError("verified inventory lacks the model")
    for path, fields in sorted(identities.items()):
        if not _approved_target(plan, path):
            raise SwitchError(f"verified inventory escapes the launch artifacts: {path}")
        try:
            info = os.lstat(path)
        except OSError as error:
            raise SwitchError(f"artifact vanished after hash approval: {path}") from error
        if not stat.S_ISREG(info.st_mode) or _stat_fields(info) != fields:
            raise SwitchError(f"artifact identity changed after hash approval: {path}")


def check_models(expected: str, raw: str) -> None:
    try:
        value = json.loads(raw)
        ids = [item["id"].lower() for item in value["data"]]
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise SwitchError(f"/v1/models body is malformed: {error}") from error
    if expected not in ids:
        raise SwitchError(f"exact model identity mismatch: want {expected}, got {ids}")


def check_context(spec_raw: str, raw: str) -> None:
    try:
        spec = json.loads(spec_raw)
        value = json.loads(raw)
    except ValueError as error:
        raise SwitchError(f"context check input is malformed: {error}") from error
    try:
        if spec["kind"] == "slots":
            if not isinstance(value, list) or len(value) != spec["slots"]:
                raise SwitchError("slot topology is not the profile's")
            if any(slot["n_ctx"] != spec["n_ctx"] for slot in value):
                raise SwitchError("per-slot context is not the profile's")
        elif spec["kind"] == "model_card":
            cards = [item for item in value["data"]
                     if item.get("id") == spec["served_model_id"]]
            if len(cards) != 1:
                raise SwitchError("model card is missing")
            if cards[0].get("max_model_len") != spec["max_model_len"]:
                raise SwitchError("per-request context is not the profile's")
        else:
            raise SwitchError(f"unknown context check kind {spec['kind']!r}")
    except (KeyError, TypeError, AttributeError) as error:
        raise SwitchError(f"context check body is malformed: {error}") from error


def _plan_from_args(args) -> dict:
    test_paths = None
    if args.test_cache_root or args.test_model_root:
        if not (args.test_cache_root and args.test_model_root):
            raise SwitchError("test path overrides must be given together")
        test_paths = {"cache_root": args.test_cache_root,
                      "model_root": args.test_model_root}
    for name in ("repo", "state", "port", "host"):
        if getattr(args, name) is None:
            raise SwitchError(f"--{name} is required for {args.command}")
    return build_plan(args.alias, repo=args.repo, state=args.state,
                      port=args.port, host_path=args.host, test_paths=test_paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo")
    parser.add_argument("--state")
    parser.add_argument("--port", type=int)
    parser.add_argument("--host")
    parser.add_argument("--test-cache-root")
    parser.add_argument("--test-model-root")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "verify", "approved-sha"):
        commands.add_parser(name).add_argument("alias")
    revalidate_parser = commands.add_parser("revalidate")
    revalidate_parser.add_argument("alias")
    revalidate_parser.add_argument("identities")
    models = commands.add_parser("check-models")
    models.add_argument("expected")
    models.add_argument("body")
    context = commands.add_parser("check-context")
    context.add_argument("spec")
    context.add_argument("body")
    args = parser.parse_args(argv)
    try:
        if args.command == "check-models":
            check_models(args.expected, args.body)
            return 0
        if args.command == "check-context":
            check_context(args.spec, args.body)
            return 0
        plan = _plan_from_args(args)
        if args.command == "plan":
            out = []
            for key, value in plan.items():
                if key == "digest_checks":
                    continue
                text = str(value)
                if "\0" in text or "=" in key:
                    raise SwitchError(f"plan value for {key} is not NUL-safe")
                out.append(f"{key}={text}\0")
            sys.stdout.write("".join(out))
        elif args.command == "verify":
            identities = verify_artifacts(plan)
            print(json.dumps(identities, separators=(",", ":"), sort_keys=True))
        elif args.command == "revalidate":
            try:
                identities = json.loads(args.identities)
            except ValueError as error:
                raise SwitchError(f"verified identities are malformed: {error}") from error
            revalidate(plan, identities)
        elif args.command == "approved-sha":
            print(plan["binary_sha256"])
    except SwitchError as error:
        print(f"switch_generic: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
