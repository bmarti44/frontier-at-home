#!/usr/bin/env bash
# Install switch/restore control-plane changes without loading either model.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly HOLD=/home/dsv4/.dsv4-start-hold
readonly ENV_FILE=/etc/deepseek-v4-flash/env
readonly UPSTREAM_PORT=8013

die() {
    printf '53_install_switch_control.sh: %s\n' "$*" >&2
    exit 1
}

(( EUID == 0 )) || die "must be run as root"
[[ -e $HOLD && ! -L $HOLD && -f $HOLD ]] ||
    die "maintenance hold is required: $HOLD"
[[ -r $ENV_FILE && ! -L $ENV_FILE && -f $ENV_FILE ]] ||
    die "installed auth environment is missing or unsafe: $ENV_FILE"

if /usr/sbin/runuser -u dsv4 -- env -i \
        HOME=/home/dsv4 PATH=/usr/bin:/bin LANG=C.UTF-8 \
        "$REPO/scripts/21_serve_llamacpp.sh" status >/dev/null 2>&1; then
    die "DeepSeek is running; refusing a control-plane-only install"
fi
[[ ! -e /home/dsv4/ds4-project/engine-switch/glm52.process.json ]] ||
    die "GLM process record exists; refusing a control-plane-only install"

install_unit() {
    local source=$1 destination=/etc/systemd/system/${1##*/} temporary
    temporary=$(mktemp)
    trap 'rm -f -- "$temporary"' RETURN
    sed "s|@DSV4_REPO@|$REPO|g" "$source" >"$temporary"
    grep -F '@DSV4_REPO@' "$temporary" >/dev/null &&
        die "unexpanded repository placeholder in $source"
    install -o root -g root -m 0644 "$temporary" "$destination"
    rm -f -- "$temporary"
    trap - RETURN
}

env_tmp=$(mktemp /etc/deepseek-v4-flash/.env.switch.XXXXXX)
trap 'rm -f -- "$env_tmp"' EXIT
/usr/bin/python3 - "$ENV_FILE" "$env_tmp" "$UPSTREAM_PORT" <<'PY'
import os
import sys

source, destination, port = sys.argv[1:]
allowed = {
    "API_KEY_FILE",
    "STACK",
    "UPSTREAM_HOST",
    "UPSTREAM_PORT",
    "LISTEN_PORT",
}
values = {}
with open(source, encoding="utf-8") as stream:
    for number, raw in enumerate(stream, 1):
        line = raw.rstrip("\n")
        if not line or "=" not in line:
            raise SystemExit(f"invalid auth environment line {number}")
        key, value = line.split("=", 1)
        if key not in allowed or key in values or not value:
            raise SystemExit(f"unsafe auth environment key {key!r}")
        values[key] = value
if set(values) != allowed:
    raise SystemExit("auth environment key set is incomplete")
if values["UPSTREAM_HOST"] != "127.0.0.1" or values["LISTEN_PORT"] != "8014":
    raise SystemExit("auth proxy topology is not the approved loopback chain")
values["STACK"] = "llamacpp"
values["UPSTREAM_PORT"] = port
with open(destination, "w", encoding="utf-8") as stream:
    for key in (
        "API_KEY_FILE",
        "STACK",
        "UPSTREAM_HOST",
        "UPSTREAM_PORT",
        "LISTEN_PORT",
    ):
        stream.write(f"{key}={values[key]}\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
chown root:dsv4auth "$env_tmp"
chmod 0640 "$env_tmp"

install_unit \
    "$REPO/configs/systemd/deepseek-v4-flash-llamacpp.service"
install_unit "$REPO/configs/systemd/dsv4-engine-restore.service"
mv -f -- "$env_tmp" "$ENV_FILE"
trap - EXIT

systemctl daemon-reload
systemctl disable deepseek-v4-flash-llamacpp.service
systemctl enable dsv4-engine-restore.service
systemctl restart dsv4-authhelper.service
systemctl reset-failed deepseek-v4-flash-llamacpp.service

[[ -e $HOLD ]] || die "maintenance hold disappeared during install"
[[ $(systemctl is-active deepseek-v4-flash-llamacpp.service || true) != active ]] ||
    die "engine service unexpectedly became active"
[[ $(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 \
    http://127.0.0.1:8010/health || true) == 401 ]] ||
    die "unchanged authenticated endpoint did not reject an unauthenticated probe"

printf '{"ok":true,"engine_started":false,"internal_port":8013,"auth_port":8010}\n'
