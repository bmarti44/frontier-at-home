#!/usr/bin/env bash
# Schedule one detached DeepSeek context qualification run.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly UNIT=dsv4-context-graduation.service
readonly ENGINE_LOCK=/run/dsv4/inference.lock

die() { printf '59_schedule_dsv4_context.sh: %s\n' "$*" >&2; exit 1; }

[[ $# == 4 ]] || die "usage: $0 TAG SEED_SHA256 CANDIDATE_HASH MODE"
TAG=$1
SEED_SHA256=$2
CANDIDATE_HASH=$3
MODE=$4
(( EUID == 0 )) || die "must run as root"
[[ $TAG =~ ^[a-z0-9][a-z0-9.-]{0,63}$ ]] || die "invalid TAG"
[[ $SEED_SHA256 =~ ^[0-9a-f]{64}$ ]] || die "invalid SEED_SHA256"
[[ $CANDIDATE_HASH =~ ^[0-9a-f]{40}$ ]] || die "invalid CANDIDATE_HASH"
[[ $MODE == one-million || $MODE == graduated ]] || die "invalid MODE"
actual=$(
    /usr/bin/git -c safe.directory="$REPO" -C "$REPO" rev-parse HEAD
) || die "cannot resolve repository candidate"
[[ $actual == "$CANDIDATE_HASH" ]] || die "candidate hash changed"
[[ -z $(
    /usr/bin/git -c safe.directory="$REPO" -C "$REPO" status --porcelain
) ]] || die "repository is not clean"
[[ ! -e /home/dsv4/ds4-project/engine-switch/glm52.process.json ]] ||
    die "GLM process record exists"
systemctl is-active --quiet "$UNIT" && die "$UNIT is already active"
install -d -o dsv4 -g dsv4 -m 0700 /run/dsv4
guard_handoff=false
restore_guard_on_error() {
    "$guard_handoff" || systemctl start dsv4-guard.timer >/dev/null 2>&1 || true
}
trap restore_guard_on_error EXIT
# Serialize against the health supervisor before inspecting/stopping the live
# engine. Stopping both closes a timer-expiry race already reproduced in a
# failed 1M attempt.
systemctl stop dsv4-guard.timer
systemctl stop dsv4-guard.service
# Fail if an unrelated process has the inference lock. The worker stops the
# identity-verified current DSV4 process before taking over the same launcher.
if /usr/sbin/runuser -u dsv4 -- env -i \
        HOME=/home/dsv4 PATH=/usr/bin:/bin LANG=C.UTF-8 \
        DSV4_PORT=8013 "$REPO/scripts/21_serve_llamacpp.sh" status \
        >/dev/null 2>&1; then
    : # The expected live DeepSeek launcher owns the lock.
else
    flock -n "$ENGINE_LOCK" -c true ||
        die "$ENGINE_LOCK is occupied by an unverified process"
fi

OUT=/home/dsv4/ds4-project/dsv4-context-$TAG
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
    --property=RuntimeMaxSec=14400 \
    "$REPO/scripts/58_dsv4_context_worker.sh" \
    "$OUT" "$TAG" "$SEED_SHA256" "$CANDIDATE_HASH" "$MODE"
guard_handoff=true
trap - EXIT
printf 'Scheduled %s. Evidence: %s\n' "$UNIT" "$OUT"
