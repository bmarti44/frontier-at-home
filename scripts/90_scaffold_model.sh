#!/usr/bin/env bash
# Scaffold the mechanical files for a new llama.cpp model integration.
set -Eeuo pipefail
umask 077

readonly SELF_NAME=${0##*/}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || {
    printf 'ERROR: cannot resolve script directory\n' >&2
    exit 1
}
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P) || {
    printf 'ERROR: cannot resolve repository root\n' >&2
    exit 1
}
readonly REFERENCE_BUILD=$REPO_ROOT/scripts/13_build_laguna_llamacpp.sh
readonly REFERENCE_SERVE=$REPO_ROOT/scripts/25_serve_laguna.sh
readonly REFERENCE_ENCODER=$REPO_ROOT/vendor/official-encoding/encoding/encoding_laguna.py

usage() {
    cat <<EOF
Usage: $SELF_NAME --slug <catalog-slug> --port <dev-port> \\
  --engine-repo <git-url> --engine-ref <branch-or-tag> \\
  --engine-commit <sha> --weights-root <abs-path> \\
  [--quant-env-prefix <PREFIX>]
       $SELF_NAME --self-test

Generate non-overwriting model-integration scaffolding from the Laguna S 2.1
reference files. --output-root is an internal testing option used by --self-test.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_anchor() {
    local file=$1 anchor=$2
    grep -Fq -- "$anchor" "$file" \
        || die "reference template drift: missing anchor in $file: $anchor"
}

sed_replacement() {
    # Escape a value for the replacement side of a sed expression delimited by |.
    printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'
}

shell_word() {
    printf '%q' "$1"
}

next_slot_after() {
    local decade=$1 reference=$2 stem=$3 number
    for ((number=reference + 1; number <= decade + 9; number++)); do
        if ! compgen -G "$REPO_ROOT/scripts/${number}_*" >/dev/null; then
            printf '%s\n' "$number"
            return 0
        fi
    done
    die "no free ${decade}N script slot remains after $reference for $stem"
}

validate_inputs() {
    [[ $slug =~ ^[a-z0-9]+([.-][a-z0-9]+)*$ ]] \
        || die '--slug must be a lowercase catalog slug using letters, digits, dots, or hyphens'
    [[ $port =~ ^[0-9]+$ ]] && (( port >= 1024 && port <= 65535 )) \
        || die '--port must be an integer from 1024 through 65535'
    [[ -n $engine_repo && $engine_repo != *$'\n'* ]] \
        || die '--engine-repo must be a non-empty, single-line Git URL'
    [[ -n $engine_ref && $engine_ref != -* && $engine_ref != *$'\n'* ]] \
        || die '--engine-ref must be a non-empty, single-line branch or tag'
    [[ $engine_commit =~ ^[0-9a-fA-F]{40}$ ]] \
        || die '--engine-commit must be a full 40-character hexadecimal SHA'
    engine_commit=${engine_commit,,}
    [[ $weights_root == /* && $weights_root != *$'\n'* ]] \
        || die '--weights-root must be an absolute, single-line path'
    [[ $weights_root != / && $weights_root != */../* && $weights_root != */.. ]] \
        || die '--weights-root must not be root or contain a parent-directory component'
    [[ $quant_prefix =~ ^[A-Z_][A-Z0-9_]*$ ]] \
        || die '--quant-env-prefix must be a valid uppercase environment-variable prefix'
    [[ $output_root == /* ]] || die '--output-root must be absolute'
}

check_reference_contract() {
    [[ -r $REFERENCE_BUILD ]] || die "missing reference build script: $REFERENCE_BUILD"
    [[ -r $REFERENCE_SERVE ]] || die "missing reference serve script: $REFERENCE_SERVE"
    [[ -r $REFERENCE_ENCODER ]] || die "missing reference encoder: $REFERENCE_ENCODER"

    require_anchor "$REFERENCE_BUILD" 'readonly LLAMACPP_REPOSITORY=https://github.com/poolsideai/llama.cpp'
    require_anchor "$REFERENCE_BUILD" 'readonly LLAMACPP_BRANCH=laguna'
    require_anchor "$REFERENCE_BUILD" 'readonly LLAMACPP_COMMIT=06f8cebd7fe728687be3d19f8bdedb70d75883af'
    require_anchor "$REFERENCE_BUILD" 'readonly CACHE_ROOT=${HOME:?HOME must be set}/.cache/llamacpp-laguna-06f8cebd'
    require_anchor "$REFERENCE_BUILD" 'MANIFEST=$REPO_ROOT/configs/build-manifests/llamacpp-laguna-06f8cebd.json'
    require_anchor "$REFERENCE_BUILD" '"refs/heads/$LLAMACPP_BRANCH:refs/remotes/origin/$LLAMACPP_BRANCH"'
    require_anchor "$REFERENCE_BUILD" '"$LLAMACPP_COMMIT" "refs/remotes/origin/$LLAMACPP_BRANCH"'

    require_anchor "$REFERENCE_SERVE" 'readonly STACK=laguna'
    require_anchor "$REFERENCE_SERVE" 'readonly PORT=8016'
    require_anchor "$REFERENCE_SERVE" 'readonly MODEL_ROOT=/home/bmarti44/models/laguna-s-2.1'
    require_anchor "$REFERENCE_SERVE" 'readonly DFLASH_MODEL=$MODEL_ROOT/poolside/laguna-s-2.1-DFlash-BF16.gguf'
    require_anchor "$REFERENCE_SERVE" 'readonly BINARY=/home/bmarti44/.cache/llamacpp-laguna-06f8cebd/src/build/bin/llama-server'
    require_anchor "$REFERENCE_SERVE" '  LAGUNA_QUANT     ud-q4 | ud-q5 | ud-q3 (default: ud-q4)'
    require_anchor "$REFERENCE_SERVE" '    quant=${LAGUNA_QUANT:-ud-q4}'
    require_anchor "$REFERENCE_SERVE" '    MODEL=${model_files[0]}'
    require_anchor "$REFERENCE_SERVE" '    BUILD_MANIFEST=$REPO_ROOT/configs/build-manifests/llamacpp-laguna-06f8cebd.json'
    require_anchor "$REFERENCE_SERVE" '    WEIGHTS_MANIFEST=$REPO_ROOT/weights/laguna-s-2.1/manifest.json'

    require_anchor "$REFERENCE_ENCODER" 'def encode_messages('
    require_anchor "$REFERENCE_ENCODER" '    messages: List[Dict[str, Any]],'
    require_anchor "$REFERENCE_ENCODER" '    thinking_mode: str,'
    require_anchor "$REFERENCE_ENCODER" '    context: Optional[List[Dict[str, Any]]] = None,'
    require_anchor "$REFERENCE_ENCODER" '    drop_thinking: bool = True,'
    require_anchor "$REFERENCE_ENCODER" '    add_default_bos_token: bool = True,'
    require_anchor "$REFERENCE_ENCODER" '    reasoning_effort: Optional[str] = None,'
    require_anchor "$REFERENCE_ENCODER" ') -> str:'
}

write_encoder_stub() {
    local destination=$1
    cat >"$destination" <<EOF
"""Stub text encoder for ${slug}.

Implement this contract from the model's official template, then validate it
with ``scripts/tests/template_fidelity.py`` before enabling the integration.
"""

from typing import Any, Dict, List, Optional


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Render messages according to the model's verified official template."""
    raise NotImplementedError(
        "Implement the ${slug} encoder and verify it with "
        "scripts/tests/template_fidelity.py"
    )
EOF
}

write_test_stub() {
    local destination=$1
    cat >"$destination" <<EOF
"""Template-fidelity test stub for ${slug}.

Replace this RED stub with cases driven by ``scripts/tests/template_fidelity.py``.
The harness is referenced intentionally and is not imported while scaffolding.
"""


def test_${python_slug}_template_fidelity() -> None:
    """Require an official-template comparison before this encoder is enabled."""
    raise NotImplementedError(
        "Add ${slug} cases using scripts/tests/template_fidelity.py"
    )
EOF
}

print_checklist() {
    cat <<EOF

Scaffold complete. Remaining integration checklist:
  [ ] Add models/catalog.json entry for "$slug" and its catalog test expectation.
  [ ] Add the README claim badge row; include claim%3A${slug} in both badge URLs
      (claim%3A${slug} ... claim%3A${slug}).
  [ ] Force-add the ignored vendor encoder:
      git add -f vendor/official-encoding/encoding/encoding_${slug}.py
  [ ] If harness files change, run the verification suite and refresh
      verification/MANIFEST.sha256.
  [ ] Add narrowly scoped scripts/lint_secrets.sh allowlist entries likely needed
      for results/${slug}-gates/ paths and weights/${slug}/manifest.json.
  [ ] Add the ${slug} case to scripts/52_engine_switch.sh.
  [ ] Add and validate the ${slug} profile JSON.
  [ ] Download politeness: push branches BEFORE starting weight downloads; fetch
      only the primary quant, and defer ladder quants until a gate needs them.
EOF
}

generate() {
    local build_number serve_number short_commit
    local build_name serve_name encoder_name test_name
    local build_target serve_target encoder_target test_target staging
    local escaped_repo escaped_ref escaped_weights escaped_prefix escaped_slug
    local escaped_port escaped_commit escaped_short

    validate_inputs
    check_reference_contract
    build_number=$(next_slot_after 10 13 build_)
    serve_number=$(next_slot_after 20 25 serve_)
    short_commit=${engine_commit:0:8}
    build_name=${build_number}_build_${slug}_llamacpp.sh
    serve_name=${serve_number}_serve_${slug}.sh
    encoder_name=encoding_${slug}.py
    test_name=test_encoding_${slug}.py
    build_target=$output_root/scripts/$build_name
    serve_target=$output_root/scripts/$serve_name
    encoder_target=$output_root/vendor/official-encoding/encoding/$encoder_name
    test_target=$output_root/scripts/tests/$test_name

    for target in "$build_target" "$serve_target" "$encoder_target" "$test_target"; do
        [[ ! -e $target && ! -L $target ]] || die "refusing to overwrite existing path: $target"
    done

    mkdir -p -- "$output_root/scripts/tests" \
        "$output_root/vendor/official-encoding/encoding"
    staging=$(mktemp -d "$output_root/.scaffold-${slug}.XXXXXX") \
        || die 'cannot create staging directory'
    trap 'rm -rf -- "$staging"' RETURN

    escaped_repo=$(sed_replacement "$(shell_word "$engine_repo")")
    escaped_ref=$(sed_replacement "$(shell_word "$engine_ref")")
    escaped_weights=$(sed_replacement "$(shell_word "$weights_root")")
    escaped_prefix=$(sed_replacement "$quant_prefix")
    escaped_slug=$(sed_replacement "$slug")
    escaped_port=$(sed_replacement "$port")
    escaped_commit=$(sed_replacement "$engine_commit")
    escaped_short=$(sed_replacement "$short_commit")

    sed \
        -e "s|13_build_laguna_llamacpp.sh|$build_name|g" \
        -e "s|https://github.com/poolsideai/llama.cpp|$escaped_repo|g" \
        -e "s|LLAMACPP_BRANCH=laguna|LLAMACPP_BRANCH=$escaped_ref|g" \
        -e "s|06f8cebd7fe728687be3d19f8bdedb70d75883af|$escaped_commit|g" \
        -e "s|llamacpp-laguna-06f8cebd|llamacpp-$escaped_slug-$escaped_short|g" \
        -e 's|"refs/heads/$LLAMACPP_BRANCH:refs/remotes/origin/$LLAMACPP_BRANCH"|"$LLAMACPP_BRANCH"|g' \
        -e 's|"$LLAMACPP_COMMIT" "refs/remotes/origin/$LLAMACPP_BRANCH"|"$LLAMACPP_COMMIT" FETCH_HEAD|g' \
        -e "s|poolsideai Laguna|$escaped_slug|g" \
        -e "s|Laguna|$escaped_slug|g" \
        -e "s|laguna|$escaped_slug|g" \
        "$REFERENCE_BUILD" >"$staging/$build_name"

    sed \
        -e "s|25_serve_laguna.sh|$serve_name|g" \
        -e "s|readonly MODEL_ROOT=/home/bmarti44/models/laguna-s-2.1|readonly MODEL_ROOT=$escaped_weights|g" \
        -e 's|readonly DFLASH_MODEL=$MODEL_ROOT/poolside/laguna-s-2.1-DFlash-BF16.gguf|readonly DFLASH_MODEL=$MODEL_ROOT/TODO-dflash-model.gguf|g' \
        -e "s|readonly BINARY=/home/bmarti44/.cache/llamacpp-laguna-06f8cebd/src/build/bin/llama-server|readonly BINARY=\${HOME:?HOME must be set}/.cache/llamacpp-$escaped_slug-$escaped_short/src/build/bin/llama-server|g" \
        -e "s|06f8cebd7fe728687be3d19f8bdedb70d75883af|$escaped_commit|g" \
        -e "s|llamacpp-laguna-06f8cebd|llamacpp-$escaped_slug-$escaped_short|g" \
        -e "s|weights/laguna-s-2.1/manifest.json|weights/$escaped_slug/manifest.json|g" \
        -e "s@  LAGUNA_QUANT     ud-q4 | ud-q5 | ud-q3 (default: ud-q4)@  ${escaped_prefix}_QUANT     TODO: define model-specific selector values@g" \
        -e "s|Laguna S 2.1|$escaped_slug|g" \
        -e "s|Laguna|$escaped_slug|g" \
        -e "s|laguna|$escaped_slug|g" \
        -e "s|LAGUNA|$escaped_prefix|g" \
        -e "s|8016|$escaped_port|g" \
        "$REFERENCE_SERVE" |
    awk -v prefix="$quant_prefix" '
        $0 == "    quant=${" prefix "_QUANT:-ud-q4}" {
            print "    # TODO(model-selection): replace these placeholder quant cases and paths"
            print "    # with the primary layout for this model before attempting a start."
            print "    quant=${" prefix "_QUANT:-TODO-primary}"
            print "    case $quant in"
            print "        TODO-primary) quant_dir=TODO/replace-with-primary-quant-directory ;;"
            print "        *) die \047TODO: define supported " prefix "_QUANT values and GGUF layouts\047 ;;"
            print "    esac"
            print "    shopt -s nullglob"
            print "    model_files=(\"$MODEL_ROOT/$quant_dir\"/*.gguf)"
            print "    shopt -u nullglob"
            print "    (( ${#model_files[@]} > 0 )) || die \"no GGUF files found below $MODEL_ROOT/$quant_dir\""
            print "    MODEL=${model_files[0]}"
            replacing = 1
            next
        }
        replacing && $0 == "    MODEL=${model_files[0]}" {
            replacing = 0
            next
        }
        !replacing { print }
        END { if (replacing) exit 91 }
    ' >"$staging/$serve_name" \
        || die 'failed to replace the Laguna quant-selection block'

    write_encoder_stub "$staging/$encoder_name"
    write_test_stub "$staging/$test_name"
    chmod 700 -- "$staging/$build_name" "$staging/$serve_name"
    bash -n "$staging/$build_name" || die "generated script fails bash -n: $build_name"
    bash -n "$staging/$serve_name" || die "generated script fails bash -n: $serve_name"
    python3 -m py_compile "$staging/$encoder_name" "$staging/$test_name" \
        || die 'generated Python stubs fail py_compile'

    # Hard-link publication is atomic and fails rather than overwriting if
    # another process creates a target during generation.
    for pair in \
        "$staging/$build_name:$build_target" \
        "$staging/$serve_name:$serve_target" \
        "$staging/$encoder_name:$encoder_target" \
        "$staging/$test_name:$test_target"; do
        source_name=${pair%%:*}
        target_name=${pair#*:}
        ln -- "$source_name" "$target_name" 2>/dev/null \
            || die "refusing to overwrite existing path: $target_name"
    done
    trap - RETURN
    rm -rf -- "$staging"

    printf 'Generated:\n  %s\n  %s\n  %s\n  %s\n' \
        "$build_target" "$serve_target" "$encoder_target" "$test_target"
    print_checklist
}

self_test() {
    local test_root output
    test_root=$(mktemp -d) || die 'cannot create self-test directory'
    trap 'rm -rf -- "$test_root"' RETURN
    output=$("${BASH_SOURCE[0]}" \
        --slug example-model \
        --port 8099 \
        --engine-repo https://example.invalid/llama.cpp \
        --engine-ref example-branch \
        --engine-commit 0123456789abcdef0123456789abcdef01234567 \
        --weights-root /models/example-model \
        --output-root "$test_root") || die 'self-test scaffold invocation failed'

    mapfile -t build_scripts < <(find "$test_root/scripts" -maxdepth 1 \
        -type f -name '1?_build_example-model_llamacpp.sh' -print)
    mapfile -t serve_scripts < <(find "$test_root/scripts" -maxdepth 1 \
        -type f -name '2?_serve_example-model.sh' -print)
    (( ${#build_scripts[@]} == 1 )) || die 'self-test did not generate exactly one build script'
    (( ${#serve_scripts[@]} == 1 )) || die 'self-test did not generate exactly one serve script'
    local encoder=$test_root/vendor/official-encoding/encoding/encoding_example-model.py
    local test_file=$test_root/scripts/tests/test_encoding_example-model.py
    [[ -f $encoder && -f $test_file ]] || die 'self-test Python stubs are missing'
    bash -n "${BASH_SOURCE[0]}"
    bash -n "${build_scripts[0]}" "${serve_scripts[0]}"
    python3 -m py_compile "$encoder" "$test_file"
    grep -Fq 'TODO(model-selection)' "${serve_scripts[0]}" \
        || die 'self-test serve scaffold lacks quant TODO block'
    grep -Fq 'llamacpp-example-model-01234567' "${build_scripts[0]}" \
        || die 'self-test build scaffold has the wrong cache identity'
    grep -Fq 'readonly LLAMACPP_BRANCH=example-branch' "${build_scripts[0]}" \
        || die 'self-test build scaffold has the wrong engine ref'
    grep -Fq '"$LLAMACPP_COMMIT" FETCH_HEAD' "${build_scripts[0]}" \
        || die 'self-test build scaffold does not support branch-or-tag fetches'
    grep -Fq 'readonly PORT=8099' "${serve_scripts[0]}" \
        || die 'self-test serve scaffold has the wrong port'
    grep -Fq 'EXAMPLE_MODEL_QUANT' "${serve_scripts[0]}" \
        || die 'self-test serve scaffold has the wrong derived environment prefix'
    grep -Fq 'claim%3Aexample-model' <<<"$output" \
        || die 'self-test checklist lacks the encoded claim label'
    if "${BASH_SOURCE[0]}" \
        --slug example-model \
        --port 8099 \
        --engine-repo https://example.invalid/llama.cpp \
        --engine-ref example-branch \
        --engine-commit 0123456789abcdef0123456789abcdef01234567 \
        --weights-root /models/example-model \
        --output-root "$test_root" >/dev/null 2>&1; then
        die 'self-test generator overwrote an existing scaffold'
    fi
    printf 'SELF-TEST PASS: generated scripts pass bash -n and Python stubs pass py_compile.\n'
    trap - RETURN
    rm -rf -- "$test_root"
}

for command_name in awk bash find grep ln mktemp python3 sed; do
    require_command "$command_name"
done

slug=
port=
engine_repo=
engine_ref=
engine_commit=
weights_root=
quant_prefix=
output_root=$REPO_ROOT
self_test_requested=false

while (( $# > 0 )); do
    case $1 in
        --slug|--port|--engine-repo|--engine-ref|--engine-commit|--weights-root|--quant-env-prefix|--output-root)
            (( $# >= 2 )) || die "missing value for $1"
            case $1 in
                --slug) slug=$2 ;;
                --port) port=$2 ;;
                --engine-repo) engine_repo=$2 ;;
                --engine-ref) engine_ref=$2 ;;
                --engine-commit) engine_commit=$2 ;;
                --weights-root) weights_root=$2 ;;
                --quant-env-prefix) quant_prefix=$2 ;;
                --output-root) output_root=$2 ;;
            esac
            shift 2
            ;;
        --self-test) self_test_requested=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done

if "$self_test_requested"; then
    [[ -z $slug && -z $port && -z $engine_repo && -z $engine_ref &&
       -z $engine_commit && -z $weights_root && -z $quant_prefix ]] \
        || die '--self-test cannot be combined with generation arguments'
    self_test
    exit 0
fi

[[ -n $slug && -n $port && -n $engine_repo && -n $engine_ref &&
   -n $engine_commit && -n $weights_root ]] || { usage >&2; exit 2; }
if [[ -z $quant_prefix ]]; then
    quant_prefix=${slug^^}
    quant_prefix=${quant_prefix//[^A-Z0-9]/_}
fi
python_slug=${slug//[^a-zA-Z0-9]/_}
generate
