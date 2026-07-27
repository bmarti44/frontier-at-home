#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

REPO=/home/bmarti44/spark-deepseek-v4-flash
TAG=${MATCHED_TAG:?MATCHED_TAG is required}
OUT=/home/dsv4/ds4-project/glm52-decisive-$TAG
SAFE=$REPO/results/glm52-gates/harness/glm_safe_run.sh
GLM_ARM=/tmp/glm_decisive_arm_$TAG.sh
SEED=${MATCHED_SEED:?MATCHED_SEED is required}
BLOCKS=${MATCHED_BLOCKS:-5}
[[ $BLOCKS =~ ^[1-5]$ ]] || { echo "MATCHED_BLOCKS must be 1-5" >&2; exit 2; }
PORT=8011
ACTIVE=

wait_full_release() {
    sudo -n -u dsv4 python3 "$REPO/scripts/03_memory_guard.py" \
        --required-gib 110 --stable-samples 3 --timeout-seconds 180
}

restore_dsv4() {
    local status models
    if [[ $ACTIVE == glm52 ]]; then
        ACTIVE=
    fi
    wait_full_release || return 1
    sudo -n -u dsv4 env HOME=/home/dsv4 \
        "$REPO/scripts/21_serve_llamacpp.sh" start
    status=$(sudo -n -u dsv4 env HOME=/home/dsv4 \
        "$REPO/scripts/21_serve_llamacpp.sh" status) || return 1
    python3 - "$status" <<'PY' || return 1
import json, sys
value = json.loads(sys.argv[1])
assert value["server_alive"] is True
assert value["memwatch_alive"] is True
assert value["watchdog_armed"] is True
assert value["healthy"] is True
PY
    models=$(curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models") ||
        return 1
    python3 - "$models" <<'PY' || return 1
import json, sys
value = json.loads(sys.argv[1])
assert any("deepseek" in item["id"].lower() for item in value["data"])
PY
    ACTIVE=dsv4
}
trap restore_dsv4 EXIT

if sudo -n -u dsv4 test -e "$OUT"; then
    echo "refusing to overwrite $OUT" >&2
    exit 1
fi
sudo -n -u dsv4 mkdir -p -- "$OUT"
sudo -n -u dsv4 cp -- "$REPO/results/glm52-goal/harness/glm_decisive_arm.sh" "$GLM_ARM"
sudo -n -u dsv4 chmod 0500 "$GLM_ARM"

sudo -n -u dsv4 env HOME=/home/dsv4 \
    "$REPO/scripts/21_serve_llamacpp.sh" stop
wait_full_release

run_dsv4() {
    local label=$1 arm_out=$OUT/$label
    sudo -n -u dsv4 mkdir -p -- "$arm_out"
    wait_full_release
    sudo -n -u dsv4 env HOME=/home/dsv4 \
        "$REPO/scripts/21_serve_llamacpp.sh" start
    ACTIVE=dsv4
    sudo -n -u dsv4 env HOME=/home/dsv4 \
        "$REPO/.venv-harness/bin/python" "$REPO/scripts/30_bench_speed.py" \
        --base-url "http://127.0.0.1:$PORT" \
        --out "$arm_out/result.json" --stack-label "$label" \
        --reps 1 --context-levels 0 --max-tokens 160 \
        --min-completion-tokens 128 --seed "$SEED" --ignore-eos-supported
    sudo -n -u dsv4 env HOME=/home/dsv4 \
        "$REPO/scripts/21_serve_llamacpp.sh" stop
    ACTIVE=
    wait_full_release
}

run_glm() {
    local label=$1 arm_out=$OUT/$label
    wait_full_release
    ACTIVE=glm52
    sudo -n -u dsv4 env HOME=/home/dsv4 \
        GLM_CANDIDATE_SRC="${GLM_CANDIDATE_SRC:-}" \
        GLM_SAFE_KILL_FLOOR_GIB=18 GLM_SAFE_MIN_START_GIB=110 \
        GLM_SAFE_TIMEOUT_S=2400 \
        flock -n -E 75 /run/dsv4/inference.lock \
        bash "$SAFE" --tag "$label" -- bash "$GLM_ARM" "$arm_out" "$label" "$SEED"
    ACTIVE=
    wait_full_release
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
