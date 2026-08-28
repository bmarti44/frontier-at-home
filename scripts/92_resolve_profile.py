#!/usr/bin/env python3
"""Read-only CLI over the profile resolver (docs/PROFILE-SCHEMA.md).

Verbs:
  render --profile <model>/<file> [--host FILE] [--verb start|stop|status]
      Print the fully-resolved launch snapshot as JSON.
  check  --profile <model>/<file> [--host FILE]
      Validate schema, host match, feasibility, and digest-target existence.
      Exits non-zero with the reason on any failure (fail closed).
  list   [--host FILE] [--model SLUG]
      One line per committed profile: id, status, and this host's verdict.

Never launches anything; scripts/93_profile_serve.sh and
scripts/52_engine_switch.sh consume the same resolver to launch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import profile_resolver as resolver  # noqa: E402


def split_profile(value: str) -> tuple[str, str]:
    model, _, name = value.partition("/")
    if not model or not name:
        raise SystemExit(f"ERROR: --profile must be <model>/<file>, got {value!r}")
    if not name.endswith(".json"):
        name += ".json"
    return model, name


def host_matches(profile: dict, host: dict) -> str | None:
    if profile["backend"] not in host["backends"]:
        return f"backend {profile['backend']} not available on this host"
    if profile["hardware_class"] not in {host["hardware_class"], "any"}:
        return (
            f"hardware_class {profile['hardware_class']} does not match "
            f"host class {host['hardware_class']}"
        )
    if host["ram_gib"] * 1.1 < profile["ram_tier_gib"]:
        return (
            f"tier requires {profile['ram_tier_gib']} GiB, host has "
            f"{host['ram_gib']} GiB"
        )
    return None


def cmd_render(args: argparse.Namespace) -> int:
    model_slug, profile_file = split_profile(args.profile)
    host = resolver.load_host(args.host)
    model = resolver.load_model(model_slug)
    profile = resolver.load_profile(model_slug, profile_file)
    snapshot = resolver.resolve(profile, model, host, verb=args.verb)
    print(json.dumps(snapshot, indent=1))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    model_slug, profile_file = split_profile(args.profile)
    host = resolver.load_host(args.host)
    model = resolver.load_model(model_slug)
    profile = resolver.load_profile(model_slug, profile_file)
    mismatch = host_matches(profile, host)
    if mismatch:
        print(f"FAIL {profile['profile_id']}: {mismatch}", file=sys.stderr)
        return 3
    snapshot = resolver.resolve(profile, model, host)
    fit = resolver.feasibility(profile, model, host["ram_gib"])
    missing = [
        check["path"]
        for check in snapshot["digest_checks"]
        if not os.path.exists(check["path"])
    ]
    report = {
        "profile_id": profile["profile_id"],
        "status": profile["status"]["state"],
        "feasibility": fit,
        "missing_artifacts": missing,
    }
    print(json.dumps(report, indent=1))
    if fit["verdict"] == "infeasible":
        print(f"FAIL {profile['profile_id']}: infeasible on this host", file=sys.stderr)
        return 4
    if missing:
        print(
            f"FAIL {profile['profile_id']}: missing artifacts (fetch them first)",
            file=sys.stderr,
        )
        return 5
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    host = resolver.load_host(args.host)
    for model_slug, profile_file in resolver.list_profiles(args.model):
        try:
            model = resolver.load_model(model_slug)
            profile = resolver.load_profile(model_slug, profile_file)
        except resolver.ProfileError as error:
            print(f"{model_slug}/{profile_file}\tINVALID\t{error}")
            continue
        mismatch = host_matches(profile, host)
        if mismatch:
            verdict = f"not-this-host ({mismatch})"
        else:
            verdict = resolver.feasibility(profile, model, host["ram_gib"])["verdict"]
        print(
            f"{profile['profile_id']}\t{profile['status']['state']}\t{verdict}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb_name", choices=["render", "check", "list"],
                        metavar="render|check|list")
    parser.add_argument("--profile", help="<model>/<profile-file>")
    parser.add_argument("--host", help="explicit host file (default: auto-select)")
    parser.add_argument("--model", help="list: restrict to one model slug")
    parser.add_argument("--verb", default="start",
                        choices=["start", "stop", "status"],
                        help="render: lifecycle verb for {verb} substitution")
    args = parser.parse_args()
    try:
        if args.verb_name == "render":
            if not args.profile:
                parser.error("render requires --profile")
            return cmd_render(args)
        if args.verb_name == "check":
            if not args.profile:
                parser.error("check requires --profile")
            return cmd_check(args)
        return cmd_list(args)
    except resolver.ProfileError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
