#!/usr/bin/env bash
# Bring up ds4-server on the staged DeepSeek-V4-Flash-0731 production-lineage
# weights, for pre-cutover requalification only.
#
# This deliberately does NOT use scripts/20_serve_ds4.sh: that launcher is bound
# to the production runtime (/run/dsv4 state, /run/lock inflight lock, DS4_HOME
# weights tree, and the committed MANIFEST-frozen weight hashes). Running 0731
# through it would require mutating the frozen production manifest, which must
# not happen before the cutover window. This script mirrors 20_serve_ds4.sh's
# exact server invocation for the chosen profile and nothing else.
#
# Guarantees:
#   * never binds the production port (8011) or the speed-suite port (8012)
#   * never reads or writes /home/dsv4
#   * verifies every weight file's sha256 against configs/pins/antirez-imatrix-0731.json
#   * refuses to start if another engine is running or memory headroom is short
set -Eeuo pipefail
umask 077

PORT=${PORT:-8021}
CTX=${CTX:-32768}
PROFILE=${PROFILE:-mtp}
STAGING=${STAGING:-/home/bmarti44/models/dsv4-flash-0731-production}
BINARY=${BINARY:-/tmp/ds4-prod-baa8890/ds4-server}
PIN=${PIN:-configs/pins/antirez-imatrix-0731.json}
MIN_START_GIB=${MIN_START_GIB:-100}
LOG_DIR=${LOG_DIR:-/home/bmarti44/.local/state/dsv4-0731}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

(( PORT != 8011 && PORT != 8012 )) || die "refusing to bind reserved port $PORT"
[[ -x $BINARY ]] || die "engine binary missing or not executable: $BINARY"
[[ -r $PIN ]] || die "pin file missing: $PIN"

# Exclusivity: this box runs other agents' campaigns; a second engine would both
# corrupt their evidence and risk the documented UMA out-of-memory hang.
if pgrep -x ds4-server >/dev/null 2>&1 || pgrep -x llama-server >/dev/null 2>&1; then
    die 'another engine process is already running; refusing to start'
fi
if pgrep -x fio >/dev/null 2>&1; then
    die 'fio is running; refusing to contend for device bandwidth'
fi

avail_gib=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
(( avail_gib >= MIN_START_GIB )) || die "only ${avail_gib} GiB available, need ${MIN_START_GIB}"

resolve() {  # role -> "path sha256"
    python3 - "$PIN" "$1" <<'PY'
import json, sys
pin = json.load(open(sys.argv[1]))
for f in pin["files"]:
    if f["role"] == sys.argv[2]:
        print(f["path"], f["sha256"])
        break
else:
    raise SystemExit(f"role not in pin: {sys.argv[2]}")
PY
}

verify() {  # role -> echoes verified absolute path
    local role=$1 path sha actual
    read -r path sha < <(resolve "$role")
    local abs="$STAGING/$path"
    [[ -r $abs ]] || die "staged weight missing: $abs"
    actual=$(sha256sum "$abs" | cut -d' ' -f1)
    [[ $actual == "$sha" ]] || die "sha256 mismatch for $role: expected $sha got $actual"
    printf '%s\n' "$abs"
}

printf 'Verifying staged 0731 weights (sha256, this reads ~90 GB)...\n' >&2
BASE=$(verify base)
MTP=$(verify mtp)
case $PROFILE in
    plain) MTP_MODE=0 ;;
    mtp)   MTP_MODE=2 ;;
    dspark)
        MTP_MODE=2
        DRAFTER=$(verify dspark_support)
        ;;
    *) die "invalid profile: $PROFILE (plain|mtp|dspark)" ;;
esac
printf 'All staged weights verified against %s\n' "$PIN" >&2

mkdir -p -- "$LOG_DIR"
chmod 700 -- "$LOG_DIR"
LOG=$LOG_DIR/ds4-0731-$PROFILE-$PORT.log

# Mirrors scripts/20_serve_ds4.sh's server_command for the selected profile.
# Security baseline preserved: no --cors, --trace, --kv-disk-dir, --role,
# --listen, --coordinator. Loopback bind only.
cmd=(env -u DS4_CUDA_WEIGHT_IPC_MANIFEST -u DS4_CONT_DSPARK -u DS4_DSPARK_MODEL
     DS4_CUDA_BUILD_ARTIFACTS=1 "DS4_CONT_MTP_MODE=$MTP_MODE")
if [[ $PROFILE == dspark ]]; then
    cmd+=(DS4_CONT_DSPARK=1 "DS4_DSPARK_MODEL=$DRAFTER")
fi
cmd+=("$BINARY" --cuda -m "$BASE")
[[ $PROFILE == plain ]] || cmd+=(--mtp "$MTP")
cmd+=(--host 127.0.0.1 --port "$PORT" -c "$CTX")

printf '\n===== ds4 0731 session start %s profile=%s ctx=%s port=%s =====\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PROFILE" "$CTX" "$PORT" >>"$LOG"
printf 'Starting: %s\n' "${cmd[*]}" >&2
exec "${cmd[@]}" >>"$LOG" 2>&1
