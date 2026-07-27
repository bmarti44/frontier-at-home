#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

REPO=/home/bmarti44/spark-deepseek-v4-flash
TAG=${MATCHED_TAG:?MATCHED_TAG is required}
OUT=/home/dsv4/ds4-project/glm52-decisive-$TAG
SAFE=$REPO/results/glm52-gates/harness/glm_safe_run.sh
GLM_ARM=/tmp/glm_decisive_arm_$TAG.sh
SEED=${MATCHED_SEED:?MATCHED_SEED is required}
PORT=8011
ACTIVE=

restore_dsv4() {
    if [[ $ACTIVE == glm52 ]]; then
        ACTIVE=
    fi
    sudo -n -u dsv4 env HOME=/home/dsv4 \
        "$REPO/scripts/21_serve_llamacpp.sh" start >/dev/null 2>&1 || true
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

run_dsv4() {
    local label=$1 arm_out=$OUT/$label
    sudo -n -u dsv4 mkdir -p -- "$arm_out"
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
}

run_glm() {
    local label=$1 arm_out=$OUT/$label
    ACTIVE=glm52
    sudo -n -u dsv4 env HOME=/home/dsv4 \
        GLM_SAFE_KILL_FLOOR_GIB=10 GLM_SAFE_TIMEOUT_S=2400 \
        bash "$SAFE" --tag "$label" -- bash "$GLM_ARM" "$arm_out" "$label" "$SEED"
    ACTIVE=
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

for block in 0 1 2 3 4; do
    if (( block % 2 == 0 )); then order=ABBA; else order=BAAB; fi
    for sequence in 0 1 2 3; do
        arm=${order:sequence:1}
        label="block${block}-seq${sequence}-arm${arm}"
        run_arm "$arm" "$label"
    done
done

restore_dsv4
trap - EXIT
sudo -n -u dsv4 chmod -R a+rX "$OUT"
echo "DECISIVE_MATCHED_DONE out=$OUT"
