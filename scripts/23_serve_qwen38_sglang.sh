#!/usr/bin/env bash
# Start, stop, or inspect the Qwen3.8 SGLang Docker development server.
set -Eeuo pipefail
umask 077

readonly STACK=qwen38-sglang
readonly CONTAINER_NAME=qwen38-sglang
readonly IMAGE=lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1
readonly HOST=127.0.0.1
readonly PORT=30000
readonly RUNTIME_DIR=/run/user/1000
readonly STATE_FILE=$RUNTIME_DIR/qwen38-sglang.state.json
readonly MODEL_HOST_ROOT=/home/bmarti44/models
readonly CACHE_DIR=/home/bmarti44/.cache/sglang-qwen38
readonly HF_CACHE_DIR=/home/bmarti44/.cache/huggingface
readonly STOP_TIMEOUT=30

container_id=
image_id=
arm=
model_path=
draft_path=
ssm_dtype=
torch_compile=
startup_armed=false

usage() {
    cat <<'EOF'
Usage: 23_serve_qwen38_sglang.sh [start|stop|status] [--help]

Start (the default), stop, or inspect the Track B Qwen3.8-27B SGLang
development server on 127.0.0.1:30000.

Environment:
  QWEN_SGLANG_ARM
      fp8         FP8 target without speculative decoding (default)
      nvfp4       NVFP4 target without speculative decoding
      fp8-spec    FP8 target with the DSpark draft
      nvfp4-spec  NVFP4 target with the DSpark draft
  QWEN_SGLANG_SSM_DTYPE
      float32 (default, fidelity-first) | bfloat16
  QWEN_SGLANG_TORCH_COMPILE
      Set to 1 to enable torch compile (default: 0)
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

need_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

configure_arm() {
    arm=${QWEN_SGLANG_ARM:-fp8}
    case $arm in
        fp8)
            model_path=/models/qwen3.8-27b-fp8
            draft_path=
            ;;
        nvfp4)
            model_path=/models/qwen3.8-27b-nvfp4
            draft_path=
            ;;
        fp8-spec)
            model_path=/models/qwen3.8-27b-fp8
            draft_path=/models/qwen3.8-27b-dspark
            ;;
        nvfp4-spec)
            model_path=/models/qwen3.8-27b-nvfp4
            draft_path=/models/qwen3.8-27b-dspark
            ;;
        *) die 'QWEN_SGLANG_ARM must be fp8, nvfp4, fp8-spec, or nvfp4-spec' ;;
    esac

    ssm_dtype=${QWEN_SGLANG_SSM_DTYPE:-float32}
    [[ $ssm_dtype == float32 || $ssm_dtype == bfloat16 ]] \
        || die 'QWEN_SGLANG_SSM_DTYPE must be float32 or bfloat16'
    torch_compile=${QWEN_SGLANG_TORCH_COMPILE:-0}
    [[ $torch_compile == 0 || $torch_compile == 1 ]] \
        || die 'QWEN_SGLANG_TORCH_COMPILE must be 0 or 1'
}

host_model_dir() {
    local container_path=$1
    [[ $container_path == /models/* ]] \
        || die "unsafe container model path: $container_path"
    printf '%s/%s\n' "$MODEL_HOST_ROOT" "${container_path#/models/}"
}

verify_model_complete() {
    local container_path=$1 host_path incomplete=
    host_path=$(host_model_dir "$container_path")
    [[ -d $host_path && ! -L $host_path ]] \
        || die "model directory is absent or unsafe: $host_path"
    [[ -f $host_path/config.json && -r $host_path/config.json ]] \
        || die "model config is missing or unreadable: $host_path/config.json"
    if [[ -d $host_path/.cache/huggingface/download ]]; then
        incomplete=$(find "$host_path/.cache/huggingface/download" -type f \
            -name '*.incomplete' -print -quit) \
            || die "cannot inspect download state below: $host_path"
        [[ -z $incomplete ]] \
            || die "model download is incomplete: $incomplete"
    fi
}

read_state() {
    local output
    output=$(python3 - "$STATE_FILE" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        state = json.load(stream)
    expected = {
        "stack": "qwen38-sglang",
        "container_name": "qwen38-sglang",
        "image": "lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1",
        "host": "127.0.0.1",
        "port": 30000,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"invalid {key}")
    for key in ("container_id", "image_id", "arm", "model_path", "ssm_dtype"):
        value = state[key]
        if not isinstance(value, str) or not value or any(c.isspace() for c in value):
            raise ValueError(f"invalid {key}")
    if not isinstance(state.get("draft_path"), str):
        raise ValueError("invalid draft_path")
    if not isinstance(state.get("torch_compile"), bool):
        raise ValueError("invalid torch_compile")
    for key in ("container_id", "image_id", "arm", "model_path", "draft_path",
                "ssm_dtype"):
        print(state[key])
    print("1" if state["torch_compile"] else "0")
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
    print(f"invalid state file: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
    ) || die "cannot read state file: $STATE_FILE"
    mapfile -t values <<<"$output"
    container_id=${values[0]}
    image_id=${values[1]}
    arm=${values[2]}
    model_path=${values[3]}
    draft_path=${values[4]}
    ssm_dtype=${values[5]}
    torch_compile=${values[6]}
}

write_state() {
    local temporary=$STATE_FILE.tmp.$$
    python3 - "$temporary" "$STATE_FILE" "$container_id" "$image_id" "$arm" \
        "$model_path" "$draft_path" "$ssm_dtype" "$torch_compile" <<'PY'
import json
import os
import sys

(temporary, output, container_id, image_id, arm, model_path, draft_path,
 ssm_dtype, torch_compile) = sys.argv[1:]
state = {
    "stack": "qwen38-sglang",
    "container_name": "qwen38-sglang",
    "container_id": container_id,
    "image": "lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1",
    "image_id": image_id,
    "host": "127.0.0.1",
    "port": 30000,
    "arm": arm,
    "model_path": model_path,
    "draft_path": draft_path,
    "ssm_dtype": ssm_dtype,
    "torch_compile": torch_compile == "1",
}
with open(temporary, "x", encoding="utf-8") as stream:
    json.dump(state, stream, separators=(",", ":"), allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output)
PY
}

container_identity() {
    docker container inspect --format '{{.Id}}' "$CONTAINER_NAME" 2>/dev/null
}

cleanup_failed_start() {
    local rc=$?
    "$startup_armed" || return "$rc"
    startup_armed=false
    trap - ERR EXIT
    if [[ -n $container_id && $(container_identity || true) == "$container_id" ]]; then
        docker stop --time "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    rm -f -- "$STATE_FILE"
    exit "$rc"
}

do_start() {
    local attempt current_id
    local -a command
    for command_name in curl docker find mkdir python3; do
        need_command "$command_name"
    done
    [[ -d $RUNTIME_DIR && ! -L $RUNTIME_DIR ]] \
        || die "runtime directory is absent or unsafe: $RUNTIME_DIR"

    configure_arm
    verify_model_complete "$model_path"
    [[ -z $draft_path ]] || verify_model_complete "$draft_path"
    mkdir -p -- "$CACHE_DIR" "$HF_CACHE_DIR"

    if current_id=$(container_identity); then
        if [[ $(docker container inspect --format '{{.State.Running}}' \
                "$CONTAINER_NAME" 2>/dev/null) == true ]]; then
            die "$STACK is already running with container ID $current_id"
        fi
        # An exited container is retained only for postmortem logs;
        # clear it before a fresh start.
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    if [[ -e $STATE_FILE ]]; then
        printf 'WARNING: removing stale %s state; no container was signaled.\n' \
            "$STACK" >&2
        rm -f -- "$STATE_FILE"
    fi

    image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE") \
        || die "required local Docker image is absent: $IMAGE"
    [[ -n $image_id && $image_id != *$'\n'* ]] \
        || die "Docker returned an invalid image ID for: $IMAGE"

    # Keep the exited container for postmortem logs; remove any leftover
    # from a prior run before starting a new one.
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    command=(
        docker run --detach --pull never
        --name "$CONTAINER_NAME"
        --gpus all
        --memory 100g --memory-swap 100g
        --shm-size 16g
        --network host --ipc=host
        --env TORCHINDUCTOR_CACHE_DIR=/cache/inductor
        --volume "$CACHE_DIR:/cache"
        --volume "$HF_CACHE_DIR:/root/.cache/huggingface"
        --volume "$MODEL_HOST_ROOT:/models:ro"
        "$IMAGE"
        python3 -m sglang.launch_server
        --trust-remote-code
        --model-path "$model_path"
        --tp-size 1
        --served-model-name qwen3.8-27b
        --mem-fraction-static 0.50
        --attention-backend flashinfer
        --chunked-prefill-size 8192
        --disable-prefill-cuda-graph
        --cuda-graph-max-bs 8
        --disable-flashinfer-autotune
        --mamba-radix-cache-strategy extra_buffer
        --mamba-ssm-dtype "$ssm_dtype"
        --max-mamba-cache-size 96
        --max-running-requests 8
        --num-continuous-decode-steps 2
        --reasoning-parser qwen3
        --tool-call-parser qwen3_coder
        --host "$HOST"
        --port "$PORT"
    )
    if [[ -n $draft_path ]]; then
        command+=(
            --speculative-algorithm DSPARK
            --speculative-draft-model-path "$draft_path"
            --speculative-num-draft-tokens 8
            --speculative-draft-model-quantization unquant
        )
    fi
    if [[ $torch_compile == 1 ]]; then
        command+=(--enable-torch-compile --torch-compile-max-bs 4)
    fi

    startup_armed=true
    trap cleanup_failed_start ERR EXIT
    container_id=$("${command[@]}") || die "failed to launch $STACK"
    [[ -n $container_id && $container_id != *$'\n'* ]] \
        || die 'Docker returned an invalid container ID'
    current_id=$(container_identity) || die 'launched container disappeared'
    [[ $current_id == "$container_id" ]] \
        || die "launched container identity mismatch: $current_id"
    write_state

    for ((attempt=0; attempt < 300; attempt++)); do
        [[ $(container_identity || true) == "$container_id" ]] \
            || die "SGLang exited during startup; inspect: docker logs $CONTAINER_NAME"
        if curl --silent --show-error --fail --max-time 3 \
                "http://$HOST:$PORT/health" >/dev/null 2>&1 &&
           curl --silent --show-error --fail --max-time 3 \
                "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
            startup_armed=false
            trap - ERR EXIT
            printf '{"ok":true,"stack":"qwen38-sglang","container_id":"%s","image_id":"%s","arm":"%s","port":30000}\n' \
                "$container_id" "$image_id" "$arm"
            return 0
        fi
        sleep 2
    done
    die "SGLang readiness timed out; inspect: docker logs $CONTAINER_NAME"
}

do_stop() {
    local attempt current_id
    [[ -r $STATE_FILE ]] || die "$STACK is not running (state file absent)"
    read_state
    current_id=$(container_identity) || {
        rm -f -- "$STATE_FILE"
        die "stale $STACK state removed; container is absent"
    }
    [[ $current_id == "$container_id" ]] \
        || die "refusing to stop container with unexpected identity: $current_id"
    docker stop --time "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null \
        || die "Docker could not stop $STACK"
    for ((attempt=0; attempt < 100; attempt++)); do
        container_identity >/dev/null 2>&1 || break
        sleep 0.1
    done
    container_identity >/dev/null 2>&1 \
        && die "$STACK container still exists after stop"
    rm -f -- "$STATE_FILE"
    printf '{"ok":true,"stack":"qwen38-sglang","stopped":true}\n'
}

do_status() {
    local inspect_json='[]' health_body= models_body= state_json=null
    local container_present=false health_ok=false models_ok=false
    if inspect_json=$(docker inspect "$CONTAINER_NAME" 2>/dev/null); then
        container_present=true
    fi
    if [[ -r $STATE_FILE ]]; then
        state_json=$(<"$STATE_FILE")
        python3 -c 'import json,sys; json.loads(sys.argv[1])' "$state_json" \
            || die "cannot parse state file: $STATE_FILE"
    fi
    if health_body=$(curl --silent --show-error --fail --max-time 3 \
            "http://$HOST:$PORT/health" 2>/dev/null); then
        health_ok=true
    fi
    if models_body=$(curl --silent --show-error --fail --max-time 3 \
            "http://$HOST:$PORT/v1/models" 2>/dev/null); then
        models_ok=true
    fi
    python3 - "$state_json" "$inspect_json" "$container_present" \
        "$health_ok" "$health_body" "$models_ok" "$models_body" <<'PY'
import json
import sys


def response(value):
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


state = json.loads(sys.argv[1])
inspection = json.loads(sys.argv[2])
document = {
    "ok": sys.argv[3] == "true" and sys.argv[4] == "true" and sys.argv[6] == "true",
    "stack": "qwen38-sglang",
    "state": state,
    "container_present": sys.argv[3] == "true",
    "docker_inspect": inspection,
    "health": {"ok": sys.argv[4] == "true", "body": response(sys.argv[5])},
    "models": {"ok": sys.argv[6] == "true", "body": response(sys.argv[7])},
}
print(json.dumps(document, separators=(",", ":"), allow_nan=False))
PY
    "$container_present" && "$health_ok" && "$models_ok"
}

action=start
action_seen=false
while (( $# > 0 )); do
    case $1 in
        start|stop|status)
            "$action_seen" && { usage >&2; exit 2; }
            action=$1
            action_seen=true
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

case $action in
    start) do_start ;;
    stop)
        for command_name in docker python3; do need_command "$command_name"; done
        do_stop
        ;;
    status)
        for command_name in curl docker python3; do need_command "$command_name"; done
        do_status
        ;;
esac
