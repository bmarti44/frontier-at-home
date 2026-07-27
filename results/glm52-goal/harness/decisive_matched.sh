#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

REPO=/home/bmarti44/spark-deepseek-v4-flash
TAG=${MATCHED_TAG:?MATCHED_TAG is required}
OUT=/home/dsv4/ds4-project/glm52-decisive-$TAG
CGROUP=$REPO/results/glm52-gates/harness/glm_cgroup_run.sh
GLM_ARM=/tmp/glm_decisive_arm_$TAG.sh
DSV4_MEMWATCH_LOG=/home/dsv4/logs/memwatch-llamacpp.log
GLM_CRASHLOG=/home/dsv4/ds4-project/glm52-crashlog
SEED=${MATCHED_SEED:?MATCHED_SEED is required}
BLOCKS=${MATCHED_BLOCKS:-5}
[[ $BLOCKS =~ ^[1-5]$ ]] || { echo "MATCHED_BLOCKS must be 1-5" >&2; exit 2; }
PORT=${MATCHED_PORT:-8021}
[[ $PORT =~ ^[0-9]{4,5}$ ]] || { echo "MATCHED_PORT must be 1024-65535" >&2; exit 2; }
PORT=$((10#$PORT))
(( PORT >= 1024 && PORT <= 65535 )) || { echo "MATCHED_PORT must be 1024-65535" >&2; exit 2; }
ACTIVE=
DSV4_HOLD_OVERRIDE=/home/dsv4/.glm52-matched-start-allow
[[ ! -e $DSV4_HOLD_OVERRIDE ]] || {
    echo "refusing unsafe matched hold override path: $DSV4_HOLD_OVERRIDE" >&2
    exit 2
}
DSV4_ENV=(
    HOME=/home/dsv4
    DSV4_PORT="$PORT"
    DSV4_START_HOLD_FILE="$DSV4_HOLD_OVERRIDE"
    DSV4_SERVER_BINARY=/home/dsv4/llamacpp-project/src/llama.cpp-fusion/build/bin/llama-server
    DSV4_BUILD_MANIFEST=$REPO/configs/build-manifests/llamacpp-fusion.json
    DSV4_MEM_FLOOR_GIB=18
    DSV4_WATCHDOG_FLOOR_GIB=18
    DSV4_UBATCH=512
    DSV4_BATCH=2048
    DSV4_UBATCH_LARGE=0
    CTX=8192
    DSV4_PARALLEL=1
    DSV4_NO_MMAP=1
    DSV4_SPEC_TYPE=ngram-map-k4v
)

wait_full_release() {
    sudo -n -u dsv4 python3 "$REPO/scripts/03_memory_guard.py" \
        --required-gib 110 --stable-samples 3 --timeout-seconds 180
}

kernel_cursor() {
    journalctl -k -n 0 --show-cursor --no-pager |
        sed -n 's/^-- cursor: //p'
}

assert_no_kernel_faults_since() {
    local cursor=$1 label=$2 output=$3 log
    [[ -n $cursor ]] || {
        echo "$label: could not freeze kernel journal cursor" >&2
        return 1
    }
    log=$(journalctl -k --after-cursor "$cursor" --no-pager) || {
        echo "$label: could not read kernel journal after cursor" >&2
        return 1
    }
    printf '%s\n' "$log" | sudo -n -u dsv4 tee "$output" >/dev/null ||
        return 1
    if grep -Eiq \
        'NV_ERR_NO_MEMORY|NVRM.*Xid|oom-kill|Out of memory: Killed process|Killed process .*total-vm' \
        <<<"$log"; then
        printf '%s\n' "$log" >&2
        echo "$label: kernel GPU/OOM fault invalidates arm" >&2
        return 1
    fi
}

cleanup_active() {
    if [[ $ACTIVE == dsv4 ]]; then
        sudo -n -u dsv4 env "${DSV4_ENV[@]}" \
            "$REPO/scripts/21_serve_llamacpp.sh" stop || true
    fi
    ACTIVE=
    wait_full_release || true
}

restore_dsv4() {
    local status models
    if [[ $ACTIVE == glm52 ]]; then
        ACTIVE=
    fi
    wait_full_release || return 1
    sudo -n -u dsv4 env "${DSV4_ENV[@]}" \
        "$REPO/scripts/21_serve_llamacpp.sh" start
    status=$(sudo -n -u dsv4 env "${DSV4_ENV[@]}" \
        "$REPO/scripts/21_serve_llamacpp.sh" status) || return 1
    python3 - "$status" <<'PY' || return 1
import json, sys
value = json.loads(sys.argv[1])
required = ("server_alive", "memwatch_alive", "watchdog_armed", "healthy")
if not all(value.get(field) is True for field in required):
    raise SystemExit("DeepSeek supervision or health is not verified")
PY
    models=$(curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models") ||
        return 1
    python3 - "$models" <<'PY' || return 1
import json, sys
value = json.loads(sys.argv[1])
if not any(
    "deepseek-v4-flash" == item["id"].lower() for item in value["data"]
):
    raise SystemExit("exact DeepSeek model identity mismatch")
PY
    ACTIVE=dsv4
}
trap cleanup_active EXIT

if sudo -n -u dsv4 test -e "$OUT"; then
    echo "refusing to overwrite $OUT" >&2
    exit 1
fi
sudo -n -u dsv4 mkdir -p -- "$OUT"
sudo -n -u dsv4 cp -- "$REPO/results/glm52-goal/harness/glm_decisive_arm.sh" "$GLM_ARM"
sudo -n -u dsv4 chmod 0500 "$GLM_ARM"

sudo -n -u dsv4 env "${DSV4_ENV[@]}" \
    "$REPO/scripts/21_serve_llamacpp.sh" stop
wait_full_release

run_dsv4() {
    local label=$1 arm_out=$OUT/$label cursor rc=0 bench_rc=0 status
    local before_identity= before_size=0 after_identity= after_size=0 offset
    sudo -n -u dsv4 mkdir -p -- "$arm_out"
    wait_full_release
    if read -r before_identity before_size < <(
        sudo -n -u dsv4 stat -Lc '%d:%i %s' "$DSV4_MEMWATCH_LOG" 2>/dev/null
    ); then
        :
    else
        before_identity=
        before_size=0
    fi
    cursor=$(kernel_cursor)
    set +e
    sudo -n -u dsv4 env "${DSV4_ENV[@]}" \
        "$REPO/scripts/21_serve_llamacpp.sh" start
    rc=$?
    set -e
    if (( rc != 0 )); then
        assert_no_kernel_faults_since "$cursor" "$label" \
            "$arm_out/kernel.log" || true
        return "$rc"
    fi
    ACTIVE=dsv4
    status=$(sudo -n -u dsv4 env "${DSV4_ENV[@]}" \
        "$REPO/scripts/21_serve_llamacpp.sh" status) || rc=1
    printf '%s\n' "$status" | sudo -n -u dsv4 tee \
        "$arm_out/process.identity.json" >/dev/null || rc=1
    set +e
    sudo -n -u dsv4 env "${DSV4_ENV[@]}" \
        "$REPO/.venv-harness/bin/python" "$REPO/scripts/30_bench_speed.py" \
        --base-url "http://127.0.0.1:$PORT" \
        --out "$arm_out/result.json" --stack-label "$label" \
        --reps 2 --context-levels 0 --max-tokens 160 \
        --min-completion-tokens 128 --seed "$SEED" --ignore-eos-supported
    bench_rc=$?
    (( bench_rc == 0 )) || rc=$bench_rc
    read -r after_identity after_size < <(
        sudo -n -u dsv4 stat -Lc '%d:%i %s' "$DSV4_MEMWATCH_LOG"
    ) || rc=1
    if [[ -n $before_identity && $after_identity != "$before_identity" ]] ||
            (( after_size < before_size )); then
        echo "$label: DeepSeek memwatch log identity changed" >&2
        rc=1
    else
        offset=$((before_size + 1))
        sudo -n -u dsv4 bash -c \
            'tail -c "+$1" -- "$2" >"$3"' _ "$offset" \
            "$DSV4_MEMWATCH_LOG" "$arm_out/memwatch.segment.log" || rc=1
        sudo -n -u dsv4 test -s "$arm_out/memwatch.segment.log" || rc=1
    fi
    sudo -n -u dsv4 env "${DSV4_ENV[@]}" \
        "$REPO/scripts/21_serve_llamacpp.sh" stop
    (( $? == 0 )) || rc=1
    set -e
    ACTIVE=
    wait_full_release || rc=1
    assert_no_kernel_faults_since "$cursor" "$label" \
        "$arm_out/kernel.log" || rc=1
    return "$rc"
}

run_glm() {
    local label=$1 arm_out=$OUT/$label cursor rc=0 safety_marker
    local -a safety_dirs=()
    wait_full_release
    cursor=$(kernel_cursor)
    safety_marker=$(mktemp "/tmp/glm52-safety-$TAG-$label.XXXXXX")
    ACTIVE=glm52
    set +e
    env GLM_CANDIDATE_SRC="${GLM_CANDIDATE_SRC:-}" \
        GLM_SAFE_LOG_CANDIDATE_PROVENANCE=1 \
        GLM_SAFE_EXPECTED_BINARY_SHA256="${GLM_SAFE_EXPECTED_BINARY_SHA256:-}" \
        GLM_SAFE_KILL_FLOOR_GIB=40 GLM_SAFE_MIN_START_GIB=110 \
        GLM_SAFE_EVIDENCE_DIR="$arm_out" \
        GLM_SAFE_TIMEOUT_S=2400 \
        GLM_PORT="$PORT" \
        "$CGROUP" --tag "$label" -- \
        bash "$GLM_ARM" "$arm_out" "$label" "$SEED"
    rc=$?
    set -e
    mapfile -t safety_dirs < <(
        sudo -n -u dsv4 find "$GLM_CRASHLOG" -mindepth 1 -maxdepth 1 \
            -type d -newer "$safety_marker" -name "*-$label" -print
    )
    rm -f -- "$safety_marker"
    if (( ${#safety_dirs[@]} != 1 )); then
        echo "$label: expected exactly one GLM safety evidence directory" >&2
        rc=1
    else
        sudo -n -u dsv4 cp -- "${safety_dirs[0]}/samples.log" \
            "$arm_out/samples.log" || rc=1
        sudo -n -u dsv4 cp -- "${safety_dirs[0]}/main.log" \
            "$arm_out/safety.main.log" || rc=1
    fi
    ACTIVE=
    wait_full_release || rc=1
    assert_no_kernel_faults_since "$cursor" "$label" \
        "$arm_out/kernel.log" || rc=1
    return "$rc"
}

# Random mapping: odd seed => A is DSV4, B is GLM.
run_arm() {
    local arm=$1 label=$2
    if (( SEED % 2 == 1 )); then
        if [[ $arm == A ]]; then run_dsv4 "$label"; else run_glm "$label"; fi
    else
        if [[ $arm == A ]]; then run_glm "$label"; else run_dsv4 "$label"; fi
    fi
}

for ((block=0; block<BLOCKS; block++)); do
    if (( block % 2 == 0 )); then order=ABBA; else order=BAAB; fi
    for sequence in 0 1 2 3; do
        arm=${order:sequence:1}
        label="block${block}-seq${sequence}-arm${arm}"
        run_arm "$arm" "$label"
    done
done

if ! restore_dsv4; then
    trap - EXIT
    echo "FATAL: matched campaign finished without verified DeepSeek restoration" >&2
    exit 1
fi
trap - EXIT
sudo -n -u dsv4 chmod -R a+rX "$OUT"
echo "DECISIVE_MATCHED_DONE out=$OUT"
