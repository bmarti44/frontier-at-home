#!/usr/bin/env bash
# Dispatch a model lifecycle command to its registered serving backend.
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
    printf 'Usage: %s --model <slug> --backend <name> [--print-command] start|stop|status\n' "$SELF_NAME"
}

die_registry() {
    printf 'ERROR: invalid backend registry configs/backends.json: %s\n' "$1" >&2
    exit 1
}

command -v python3 >/dev/null 2>&1 || die_registry 'python3 is required'

model=
backend=
verb=
print_command=false

while (( $# > 0 )); do
    case $1 in
        --model|--backend)
            (( $# >= 2 )) || { usage >&2; exit 2; }
            if [[ $1 == --model ]]; then
                model=$2
            else
                backend=$2
            fi
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
if ! resolution=$(python3 - "$REGISTRY" "$backend" "$model" 2>"$registry_error" <<'PY'
import json
import pathlib
import sys

registry_path = pathlib.Path(sys.argv[1])
requested_backend = sys.argv[2]
requested_model = sys.argv[3]

try:
    with registry_path.open(encoding="utf-8") as stream:
        registry = json.load(stream)
    if type(registry) is not dict:
        raise ValueError("top level must be an object")
    if registry.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    backends = registry.get("backends")
    if type(backends) is not dict or not backends:
        raise ValueError("backends must be a non-empty object")
    for name, entry in backends.items():
        if type(name) is not str or not name:
            raise ValueError("backend names must be non-empty strings")
        if type(entry) is not dict:
            raise ValueError(f"backend {name!r} must be an object")
        if type(entry.get("implemented")) is not bool:
            raise ValueError(f"backend {name!r} implemented must be boolean")
        if type(entry.get("platform")) is not str or not entry["platform"]:
            raise ValueError(f"backend {name!r} platform must be a non-empty string")
        scripts = entry.get("serve_scripts")
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
elif not backends[requested_backend]["implemented"]:
    print("unimplemented")
elif requested_model not in backends[requested_backend]["serve_scripts"]:
    print("missing-model")
else:
    print(f"ok\t{backends[requested_backend]['serve_scripts'][requested_model]}")
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
        printf 'ERROR: model %s is not registered for backend %s in configs/backends.json; scaffold it with scripts/90_scaffold_model.sh\n' \
            "$model" "$backend" >&2
        exit 4
        ;;
    $'ok\t'*)
        relative_script=${resolution#*$'\t'}
        ;;
    *)
        die_registry 'parser returned an unknown result'
        ;;
esac

serve_script=$REPO_ROOT/$relative_script
[[ -f $serve_script && -x $serve_script ]] \
    || die_registry "registered serve script is missing or not executable: $relative_script"

if "$print_command"; then
    printf '%s %s\n' "$relative_script" "$verb"
    exit 0
fi

exec "$serve_script" "$verb"
