#!/usr/bin/env bash
# Schedule a detached, display-free DeepSeek Foundation measurement.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly HOLD=/home/dsv4/.dsv4-start-hold
readonly UNIT=dsv4-headless-smoke.service

die() {
    printf '54_schedule_headless_foundation.sh: %s\n' "$*" >&2
    exit 1
}

[[ $# == 3 ]] || die "usage: $0 TAG SEED CANDIDATE_HASH"
TAG=$1
SEED=$2
CANDIDATE_HASH=$3
[[ $TAG =~ ^[a-z0-9][a-z0-9.-]{0,63}$ ]] || die "invalid TAG"
[[ $SEED =~ ^[0-9]{1,10}$ ]] && (( 10#$SEED <= 4294967295 )) ||
    die "SEED must be an unsigned 32-bit integer"
[[ $CANDIDATE_HASH =~ ^[0-9a-f]{40}$ ]] || die "invalid CANDIDATE_HASH"
(( EUID == 0 )) || die "must run as root"
[[ -e $HOLD && -f $HOLD && ! -L $HOLD ]] ||
    die "persistent maintenance hold is required"
[[ ! -e /home/dsv4/.dsv4-headless-start-allow ]] ||
    die "headless hold-override sentinel unexpectedly exists"

actual=$(
    /usr/bin/git -c safe.directory="$REPO" -C "$REPO" rev-parse HEAD
) || die "cannot resolve repository candidate"
[[ $actual == "$CANDIDATE_HASH" ]] || die "candidate hash changed"
[[ -z $(
    /usr/bin/git -c safe.directory="$REPO" -C "$REPO" status --porcelain
) ]] || die "repository is not clean"

if systemctl is-active --quiet "$UNIT"; then
    die "$UNIT is already active"
fi
if /usr/sbin/runuser -u dsv4 -- env -i \
        HOME=/home/dsv4 PATH=/usr/bin:/bin LANG=C.UTF-8 \
        "$REPO/scripts/21_serve_llamacpp.sh" status >/dev/null 2>&1; then
    die "DeepSeek is already running"
fi
[[ ! -e /home/dsv4/ds4-project/engine-switch/glm52.process.json ]] ||
    die "GLM process record exists"
ss -H -ltn 'sport = :8013' | grep -q . &&
    die "internal engine port 8013 is occupied"

OUT=/home/dsv4/ds4-project/headless-foundation-$TAG
[[ ! -e $OUT ]] || die "refusing to overwrite $OUT"
install -d -o dsv4 -g dsv4 -m 0700 "$OUT"
systemctl reset-failed "$UNIT" 2>/dev/null || true

systemd-run \
    --unit="$UNIT" \
    --collect \
    --no-block \
    --property=Type=exec \
    --property=KillMode=control-group \
    --property=OOMPolicy=kill \
    --property=OnFailure=display-manager.service \
    --property=RuntimeMaxSec=1800 \
    "$REPO/scripts/55_headless_foundation_worker.sh" \
    "$OUT" "$TAG" "$SEED" "$CANDIDATE_HASH"

printf 'Scheduled %s; the display will return automatically. Evidence: %s\n' \
    "$UNIT" "$OUT"
