#!/usr/bin/env bash
# Dispatch a model lifecycle command to its serving backend.
#
# Resolution order for --model <slug> --backend <name>:
#   1. a legacy dev-port serve script registered in configs/backends.json
#      serve_scripts (takes precedence: those lifecycles own the dev ports);
#   2. a declarative profile under configs/profiles/<slug>/ matching the
#      backend and this host, launched via scripts/93_profile_serve.sh.
# Fails closed: unknown backend (2), registered-but-unimplemented (3), no
# serve script or matching profile (4).
set -Eeuo pipefail

readonly SELF_NAME=${0##*/}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || {
    printf 'ERROR: cannot resolve script directory\n' >&2
    exit 1
}
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P) || {
    printf 'ERROR: cannot resolve repository root\n' >&2
    exit 1
}
readonly REGISTRY=$REPO_ROOT/configs/backends.json

usage() {
    printf 'Usage: %s --model <slug> --backend <name> [--profile <file>] [--print-command] start|stop|status\n' "$SELF_NAME"
}

die_registry() {
    printf 'ERROR: invalid backend registry configs/backends.json: %s\n' "$1" >&2
    exit 1
}

command -v python3 >/dev/null 2>&1 || die_registry 'python3 is required'

model=
backend=
verb=
profile_file=
print_command=false

while (( $# > 0 )); do
    case $1 in
        --model|--backend|--profile)
            (( $# >= 2 )) || { usage >&2; exit 2; }
            case $1 in
                --model) model=$2 ;;
                --backend) backend=$2 ;;
                --profile) profile_file=$2 ;;
            esac
            shift 2
            ;;
        --print-command)
            print_command=true
            shift
            ;;
        start|stop|status)
            [[ -z $verb ]] || { usage >&2; exit 2; }
            verb=$1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n $model && -n $backend && -n $verb ]] || { usage >&2; exit 2; }

registry_error=$(mktemp) || die_registry 'cannot create parser error file'
trap 'rm -f -- "$registry_error"' EXIT
if ! resolution=$(python3 - "$REGISTRY" "$backend" "$model" "$profile_file" \
        2>"$registry_error" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(sys.argv[1]).resolve().parents[1] / "scripts/lib"))

registry_path = pathlib.Path(sys.argv[1])
requested_backend = sys.argv[2]
requested_model = sys.argv[3]
requested_profile = sys.argv[4]

try:
    with registry_path.open(encoding="utf-8") as stream:
        registry = json.load(stream)
    if type(registry) is not dict:
        raise ValueError("top level must be an object")
    if registry.get("schema_version") != 2:
        raise ValueError("schema_version must be 2")
    backends = registry.get("backends")
    if type(backends) is not dict or not backends:
        raise ValueError("backends must be a non-empty object")
    for name, entry in backends.items():
        if type(name) is not str or not name:
            raise ValueError("backend names must be non-empty strings")
        if type(entry) is not dict:
            raise ValueError(f"backend {name!r} must be an object")
        if set(entry) - {"implemented", "platform", "serve_scripts", "notes"}:
            raise ValueError(f"backend {name!r} has unknown keys")
        if type(entry.get("implemented")) is not bool:
            raise ValueError(f"backend {name!r} implemented must be boolean")
        if type(entry.get("platform")) is not str or not entry["platform"]:
            raise ValueError(f"backend {name!r} platform must be a non-empty string")
        scripts = entry.get("serve_scripts", {})
        if type(scripts) is not dict:
            raise ValueError(f"backend {name!r} serve_scripts must be an object")
        if not entry["implemented"] and scripts:
            raise ValueError(f"unimplemented backend {name!r} has serve scripts")
        for slug, script in scripts.items():
            if type(slug) is not str or not slug:
                raise ValueError(f"backend {name!r} has an invalid model slug")
            if type(script) is not str or not script:
                raise ValueError(f"backend {name!r} model {slug!r} has an invalid script")
            path = pathlib.PurePosixPath(script)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"backend {name!r} model {slug!r} script is not repo-relative")
except (OSError, json.JSONDecodeError, ValueError) as error:
    print(error, file=sys.stderr)
    raise SystemExit(1)

valid_names = " ".join(backends)
if requested_backend not in backends:
    print(f"unknown\t{valid_names}")
    raise SystemExit(0)
if not backends[requested_backend]["implemented"]:
    print("unimplemented")
    raise SystemExit(0)

scripts = backends[requested_backend].get("serve_scripts", {})
if not requested_profile and requested_model in scripts:
    print(f"okscript\t{scripts[requested_model]}")
    raise SystemExit(0)

# Profile dispatch: pick the committed profile matching backend + host.
import profile_resolver as resolver  # noqa: E402

try:
    host = resolver.load_host()
except resolver.ProfileError as error:
    print(f"ERROR: {error}", file=sys.stderr)
    raise SystemExit(1)

candidates = []
for model_slug, profile_name in resolver.list_profiles():
    if model_slug != requested_model:
        continue
    if requested_profile and profile_name not in {
        requested_profile, requested_profile + ".json"
    }:
        continue
    try:
        profile = resolver.load_profile(model_slug, profile_name)
    except resolver.ProfileError:
        continue
    if profile["backend"] != requested_backend:
        continue
    if profile["hardware_class"] not in {host["hardware_class"], "any"}:
        continue
    if profile["status"]["state"] == "unsupported":
        continue
    candidates.append((profile["ram_tier_gib"], profile_name))

if not candidates:
    print("missing-model")
else:
    # Largest tier that the host satisfies wins.
    eligible = [
        (tier, name) for tier, name in candidates
        if tier <= host["ram_gib"] * 1.1
    ]
    if not eligible:
        print("missing-model")
    else:
        _, chosen = max(eligible)
        dev_port = host["ports"].get("dev", {}).get(requested_model, "")
        print(f"okprofile\t{requested_model}/{chosen}\t{dev_port}")
PY
); then
    error_text=$(<"$registry_error")
    die_registry "${error_text:-parser failure}"
fi

case $resolution in
    $'unknown\t'*)
        printf 'ERROR: unknown backend %s; valid backends: %s\n' \
            "$backend" "${resolution#*$'\t'}" >&2
        exit 2
        ;;
    unimplemented)
        printf 'backend %s is registered but not implemented on this host; see docs/BACKEND-CONTRACT.md\n' \
            "$backend" >&2
        exit 3
        ;;
    missing-model)
        printf 'ERROR: model %s has no serve script and no matching profile for backend %s on this host; see configs/profiles/ and scripts/92_resolve_profile.py list\n' \
            "$model" "$backend" >&2
        exit 4
        ;;
    $'okscript\t'*)
        relative_script=${resolution#*$'\t'}
        serve_script=$REPO_ROOT/$relative_script
        [[ -f $serve_script && -x $serve_script ]] \
            || die_registry "registered serve script is missing or not executable: $relative_script"
        if "$print_command"; then
            printf '%s %s\n' "$relative_script" "$verb"
            exit 0
        fi
        exec "$serve_script" "$verb"
        ;;
    $'okprofile\t'*)
        rest=${resolution#*$'\t'}
        chosen_profile=${rest%%$'\t'*}
        dev_port=${rest#*$'\t'}
        launcher_args=(--profile "$chosen_profile")
        [[ -z $dev_port ]] || launcher_args+=(--port "$dev_port")
        if "$print_command"; then
            printf 'scripts/93_profile_serve.sh %s %s\n' \
                "${launcher_args[*]}" "$verb"
            exit 0
        fi
        exec "$REPO_ROOT/scripts/93_profile_serve.sh" "${launcher_args[@]}" "$verb"
        ;;
    *)
        die_registry 'parser returned an unknown result'
        ;;
esac
