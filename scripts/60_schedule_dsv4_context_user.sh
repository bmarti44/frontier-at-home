#!/usr/bin/env bash
# Schedule a detached DeepSeek context run without root privileges.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly UNIT=dsv4-context-graduation.service
readonly STATE_ROOT=/home/bmarti44/.local/state/dsv4-context

die() { printf '60_schedule_dsv4_context_user.sh: %s\n' "$*" >&2; exit 1; }

[[ $# == 4 ]] || die "usage: $0 TAG SEED_SHA256 CANDIDATE_HASH MODE"
TAG=$1
SEED_SHA256=$2
CANDIDATE_HASH=$3
MODE=$4
[[ $EUID == 1000 && $(id -un) == bmarti44 ]] || die "must run as bmarti44"
[[ $TAG =~ ^[a-z0-9][a-z0-9.-]{0,63}$ ]] || die "invalid TAG"
[[ $SEED_SHA256 =~ ^[0-9a-f]{64}$ ]] || die "invalid SEED_SHA256"
[[ $CANDIDATE_HASH =~ ^[0-9a-f]{40}$ ]] || die "invalid CANDIDATE_HASH"
[[ $MODE == one-million || $MODE == graduated ]] || die "invalid MODE"

actual=$(/usr/bin/git -C "$REPO" rev-parse HEAD) || die "cannot resolve candidate"
[[ $actual == "$CANDIDATE_HASH" ]] || die "candidate hash changed"
[[ -z $(/usr/bin/git -C "$REPO" status --porcelain) ]] ||
    die "repository is not clean"
[[ $(loginctl show-user bmarti44 -p Linger --value) == yes ]] ||
    die "user-systemd linger is disabled"
sudo -n -u dsv4 true || die "passwordless dsv4 delegation is unavailable"
systemctl is-active --quiet display-manager.service &&
    die "display manager must remain inactive"
systemctl is-active --quiet dsv4-context-graduation.service &&
    die "root context unit is already active"
systemctl --user is-active --quiet "$UNIT" &&
    die "user context unit is already active"
sudo -n -u dsv4 test ! -e /home/dsv4/.dsv4-start-hold ||
    die "an existing maintenance hold blocks the run"
sudo -n -u dsv4 test ! -e \
    /home/dsv4/ds4-project/engine-switch/glm52.process.json ||
    die "GLM process record exists"
sudo -n -u dsv4 env -i \
    HOME=/home/dsv4 USER=dsv4 LOGNAME=dsv4 LANG=C.UTF-8 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    DSV4_PORT=8013 "$REPO/scripts/21_serve_llamacpp.sh" status \
    >/dev/null 2>&1 || die "verified 8K DeepSeek engine is not healthy"

install -d -m 0700 "$STATE_ROOT"
[[ -d $STATE_ROOT && ! -L $STATE_ROOT && -O $STATE_ROOT ]] ||
    die "invalid state root"
OUT=$STATE_ROOT/dsv4-context-$TAG
[[ ! -e $OUT ]] || die "refusing to overwrite $OUT"
install -d -m 0700 "$OUT"
FROZEN=$OUT/frozen-candidate
install -d -m 0700 "$FROZEN"
/usr/bin/git -C "$REPO" archive "$CANDIDATE_HASH" |
    /usr/bin/tar -x -C "$FROZEN"
install -D -m 0400 "$REPO/vendor/official-encoding/tokenizer.json" \
    "$FROZEN/vendor/official-encoding/tokenizer.json"
/usr/bin/python3 - "$FROZEN" "$CANDIDATE_HASH" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
candidate = sys.argv[2]
artifacts = {}
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    artifacts[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
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
chmod -R a-w "$FROZEN"
readonly WORKER=$FROZEN/scripts/58_dsv4_context_worker.sh

systemctl --user reset-failed "$UNIT" 2>/dev/null || true
systemd-run --user \
    --unit="$UNIT" \
    --collect \
    --no-block \
    --property=Type=exec \
    --property=KillMode=control-group \
    --property=OOMPolicy=kill \
    --property=RuntimeMaxSec=43200 \
    --property=TimeoutStopSec=600 \
    --property=MemorySwapMax=0 \
    --directory="$FROZEN" \
    --property="StandardOutput=append:$OUT/worker.log" \
    --property="StandardError=append:$OUT/worker.log" \
    "$WORKER" "$OUT" "$TAG" "$SEED_SHA256" "$CANDIDATE_HASH" "$MODE"
printf 'Scheduled user %s. Evidence: %s\n' "$UNIT" "$OUT"
