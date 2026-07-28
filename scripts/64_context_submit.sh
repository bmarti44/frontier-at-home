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

die() { printf 'dsv4-context-submit: %s\n' "$*" >&2; exit 1; }

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
    /usr/bin/git -c safe.directory="$REPO" -C "$REPO" rev-parse HEAD
) || die "cannot resolve candidate"
[[ $actual == "$CANDIDATE_HASH" ]] || die "candidate hash changed"
[[ -z $(
    /usr/bin/git -c safe.directory="$REPO" -C "$REPO" status --porcelain
) ]] || die "repository is not clean"
/usr/bin/systemctl is-active --quiet display-manager.service &&
    die "display manager must remain inactive"
/usr/bin/systemctl is-active --quiet "$UNIT" &&
    die "$UNIT is already active"
[[ ! -e /home/dsv4/ds4-project/engine-switch/glm52.process.json ]] ||
    die "GLM process record exists"
/usr/sbin/runuser -u dsv4 -- /usr/bin/env -i \
    HOME=/home/dsv4 USER=dsv4 LOGNAME=dsv4 LANG=C.UTF-8 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    DSV4_PORT=8013 "$REPO/scripts/21_serve_llamacpp.sh" status \
    >/dev/null 2>&1 || die "verified 8K DeepSeek engine is not healthy"

/usr/bin/install -d -o root -g root -m 0711 \
    "$STATE_ROOT" "$CANDIDATE_ROOT" "$ATTEMPT_ROOT"
OUT=$ATTEMPT_ROOT/dsv4-context-$TAG
[[ ! -e $OUT ]] || die "refusing to overwrite $OUT"
/usr/bin/install -d -o root -g root -m 0700 "$OUT"

FROZEN=$CANDIDATE_ROOT/$CANDIDATE_HASH
if [[ ! -d $FROZEN ]]; then
    /usr/bin/install -d -o root -g root -m 0700 "$FROZEN"
    /usr/bin/git -c safe.directory="$REPO" -C "$REPO" \
        archive "$CANDIDATE_HASH" |
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
[[ -O $FROZEN && ! -L $FROZEN &&
    -r $FROZEN/scripts/58_dsv4_context_worker.sh &&
    ! -w $FROZEN/scripts/58_dsv4_context_worker.sh ]] ||
    die "root-owned candidate snapshot is invalid"
/usr/bin/python3 -I "$FROZEN/scripts/62_score_dsv4_context.py" \
    --candidate-hash "$CANDIDATE_HASH" --verify-only

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
