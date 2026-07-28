#!/usr/bin/env bash
# Root-owned, narrowly delegated DeepSeek context qualification submitter.
set -Eeuo pipefail
umask 077
export PATH=/usr/bin:/bin

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly UNIT=dsv4-context-graduation.service
readonly STATE_ROOT=/var/lib/dsv4-context
readonly CANDIDATE_ROOT=$STATE_ROOT/candidates
readonly ATTEMPT_ROOT=$STATE_ROOT/attempts
readonly MODEL_ROOT=$STATE_ROOT/models/deepseek-v4-flash
readonly MODEL_PARENT=$STATE_ROOT/models
readonly MODEL_SOURCE=$REPO/weights/unsloth-ud-q2_k_xl
readonly MODEL_FILES=(
    DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf
    DeepSeek-V4-Flash-UD-Q2_K_XL-00002-of-00003.gguf
    DeepSeek-V4-Flash-UD-Q2_K_XL-00003-of-00003.gguf
)

die() { printf 'dsv4-context-submit: %s\n' "$*" >&2; exit 1; }

git_as_user() {
    /usr/sbin/runuser -u bmarti44 -- /usr/bin/env -i \
        HOME=/nonexistent PATH=/usr/bin:/bin LANG=C.UTF-8 \
        GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
        /usr/bin/git -c core.fsmonitor=false -c core.hooksPath=/dev/null "$@"
}

[[ $# == 4 ]] || die "usage: $0 TAG auto CANDIDATE_HASH MODE"
TAG=$1
SEED_REQUEST=$2
CANDIDATE_HASH=$3
MODE=$4
(( EUID == 0 )) || die "must run as root"
[[ $TAG =~ ^[a-z0-9][a-z0-9.-]{0,63}$ ]] || die "invalid TAG"
[[ $SEED_REQUEST == auto ]] || die "seed request must be auto"
[[ $CANDIDATE_HASH =~ ^[0-9a-f]{40}$ ]] || die "invalid candidate hash"
[[ $MODE == one-million || $MODE == graduated ]] || die "invalid mode"

actual=$(
    git_as_user -C "$REPO" rev-parse HEAD
) || die "cannot resolve candidate"
[[ $actual == "$CANDIDATE_HASH" ]] || die "candidate hash changed"
[[ -z $(
    git_as_user -C "$REPO" status --porcelain
) ]] || die "repository is not clean"
/usr/bin/systemctl is-active --quiet display-manager.service &&
    die "display manager must remain inactive"
/usr/bin/systemctl is-active --quiet "$UNIT" &&
    die "$UNIT is already active"
[[ ! -e /home/dsv4/ds4-project/engine-switch/glm52.process.json ]] ||
    die "GLM process record exists"

/usr/bin/install -d -o root -g root -m 0711 \
    "$STATE_ROOT" "$CANDIDATE_ROOT" "$ATTEMPT_ROOT" "$MODEL_PARENT"
model_valid=true
for name in "${MODEL_FILES[@]}"; do
    protected_path=$MODEL_ROOT/$name
    if [[ ! -f $protected_path || -L $protected_path ]] ||
        [[ $(/usr/bin/stat -c %u "$protected_path" 2>/dev/null) != 0 ]] ||
        [[ $(/usr/bin/stat -c %h "$protected_path" 2>/dev/null) != 1 ]]; then
        model_valid=false
    fi
done
if ! "$model_valid"; then
    model_temporary=$MODEL_PARENT/.deepseek-v4-flash-copy-$CANDIDATE_HASH
    [[ ! -e $model_temporary ]] ||
        die "protected model temporary path already exists"
    /usr/bin/install -d -o root -g root -m 0700 "$model_temporary"
    for name in "${MODEL_FILES[@]}"; do
        source_path=$MODEL_SOURCE/$name
        [[ -f $source_path && ! -L $source_path ]] ||
            die "registered model source is invalid: $name"
        printf 'Protecting model shard %s...\n' "$name"
        /usr/bin/cp --reflink=never --sparse=always -- \
            "$source_path" "$model_temporary/$name"
        /usr/bin/chown root:root "$model_temporary/$name"
        /usr/bin/chmod 0444 "$model_temporary/$name"
        [[ $(/usr/bin/stat -c %h "$model_temporary/$name") == 1 ]] ||
            die "protected model shard is not an independent inode"
    done
    /usr/bin/chmod 0555 "$model_temporary"
    if [[ -e $MODEL_ROOT ]]; then
        replaced=$MODEL_PARENT/deepseek-v4-flash-replaced-$CANDIDATE_HASH
        [[ ! -e $replaced ]] || die "protected model replacement already exists"
        /usr/bin/mv -- "$MODEL_ROOT" "$replaced"
    fi
    /usr/bin/mv -- "$model_temporary" "$MODEL_ROOT"
fi
OUT=$ATTEMPT_ROOT/dsv4-context-$TAG
[[ ! -e $OUT ]] || die "refusing to overwrite $OUT"
/usr/bin/install -d -o root -g root -m 0700 "$OUT"

FROZEN=$CANDIDATE_ROOT/$CANDIDATE_HASH
if [[ ! -d $FROZEN ]]; then
    /usr/bin/install -d -o root -g root -m 0700 "$FROZEN"
    git_as_user -C "$REPO" archive "$CANDIDATE_HASH" |
        /usr/bin/tar -x -C "$FROZEN"
    /usr/bin/install -D -o root -g root -m 0444 \
        "$REPO/vendor/official-encoding/tokenizer.json" \
        "$FROZEN/vendor/official-encoding/tokenizer.json"
    /usr/bin/python3 - "$FROZEN" "$CANDIDATE_HASH" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
candidate = sys.argv[2]
artifacts = {
    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file())
}
(root / "freeze-manifest.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "candidate_hash": candidate,
            "artifacts": artifacts,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n",
    encoding="utf-8",
)
PY
    /usr/bin/chown -R root:root "$FROZEN"
    /usr/bin/chmod -R a-w,go-rwx "$FROZEN"
fi
worker_path=$FROZEN/scripts/58_dsv4_context_worker.sh
worker_uid=$(/usr/bin/stat -c %u "$worker_path")
worker_mode=$(/usr/bin/stat -c %a "$worker_path")
[[ $worker_uid == 0 && ! -L $FROZEN && -r $worker_path ]] &&
    (( (8#$worker_mode & 8#222) == 0 )) ||
    die "root-owned candidate snapshot is invalid"
/usr/bin/python3 -I "$FROZEN/scripts/62_score_dsv4_context.py" \
    --candidate-hash "$CANDIDATE_HASH" --verify-only
/usr/bin/env -i \
    HOME=/home/dsv4 USER=root LOGNAME=root LANG=C.UTF-8 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    DSV4_PORT=8013 "$FROZEN/scripts/21_serve_llamacpp.sh" status \
    >/dev/null 2>&1 || die "verified 8K DeepSeek engine is not healthy"

/usr/bin/systemctl reset-failed "$UNIT" 2>/dev/null || true
/usr/bin/systemd-run \
    --unit="$UNIT" \
    --collect \
    --no-block \
    --property=Type=exec \
    --property=KillMode=control-group \
    --property=OOMPolicy=kill \
    --property=RuntimeMaxSec=43200 \
    --property=TimeoutStopSec=600 \
    --property=MemorySwapMax=0 \
    --working-directory="$FROZEN" \
    --property="StandardOutput=append:$OUT/worker.log" \
    --property="StandardError=append:$OUT/worker.log" \
    "$FROZEN/scripts/58_dsv4_context_worker.sh" \
    "$OUT" "$TAG" auto "$CANDIDATE_HASH" "$MODE"
printf 'Scheduled system %s. Evidence: %s\n' "$UNIT" "$OUT"
