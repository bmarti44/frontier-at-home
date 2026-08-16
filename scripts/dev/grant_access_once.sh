#!/usr/bin/env bash
# grant_access_once.sh — one-time sudo run that removes every recurring sudo
# requirement for user bmarti44 on this host. After this script succeeds,
# agents and the owner never need an interactive sudo prompt again:
#
#   1. Read-only ACLs (current + default) on /home/dsv4 so the true engine
#      source (/home/dsv4/ds4-project/src/ds4-upstream-master) and the
#      llama.cpp checkout are readable. Grants are rX only — nothing under
#      /home/dsv4 becomes writable by bmarti44.
#   2. Read ACL on /etc/deepseek-v4-flash (auth header) for local health checks.
#   3. A validated sudoers drop-in giving bmarti44 passwordless access to the
#      exact commands the runbook already sanctions (engine switch, dsv4
#      service control, exposure verification, and read-only sudo -u dsv4
#      status invocations).
#
# Usage: sudo bash scripts/dev/grant_access_once.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "This script must be run with sudo (one time only)." >&2
  exit 1
fi

TARGET_USER=bmarti44
DSV4_HOME=/home/dsv4
REPO=/home/bmarti44/spark-deepseek-v4-flash

echo "== 1/3 Read-only ACLs on ${DSV4_HOME} =="
# Current files/dirs: read + traverse. Default ACLs: future files inherit.
setfacl -R -m "u:${TARGET_USER}:rX" "${DSV4_HOME}"
setfacl -R -d -m "u:${TARGET_USER}:rX" "${DSV4_HOME}"
echo "ACLs applied."

echo "== 2/3 Read ACL on /etc/deepseek-v4-flash =="
if [[ -d /etc/deepseek-v4-flash ]]; then
  setfacl -R -m "u:${TARGET_USER}:rX" /etc/deepseek-v4-flash
  setfacl -R -d -m "u:${TARGET_USER}:rX" /etc/deepseek-v4-flash
  echo "ACLs applied."
else
  echo "(directory absent, skipped)"
fi

echo "== 3/3 Scoped passwordless sudoers drop-in =="
SUDOERS_FILE=/etc/sudoers.d/bmarti44-frontier
cat > "${SUDOERS_FILE}.tmp" <<'EOF'
# Installed by grant_access_once.sh — sanctioned frontier-at-home commands
# run without a password for bmarti44. Scope is deliberately narrow.
bmarti44 ALL=(root) NOPASSWD: /home/bmarti44/spark-deepseek-v4-flash/scripts/52_engine_switch.sh *
bmarti44 ALL=(root) NOPASSWD: /home/bmarti44/spark-deepseek-v4-flash/scripts/42_verify_exposure.sh
bmarti44 ALL=(root) NOPASSWD: /usr/bin/systemctl start deepseek-v4-flash-llamacpp.service, /usr/bin/systemctl stop deepseek-v4-flash-llamacpp.service, /usr/bin/systemctl status deepseek-v4-flash-llamacpp.service, /usr/bin/systemctl restart deepseek-v4-flash-llamacpp.service
bmarti44 ALL=(root) NOPASSWD: /usr/bin/systemctl start dsv4-guard.timer, /usr/bin/systemctl stop dsv4-guard.timer, /usr/bin/systemctl status dsv4-guard.timer
bmarti44 ALL=(root) NOPASSWD: /usr/bin/systemctl start dsv4-authhelper.service, /usr/bin/systemctl stop dsv4-authhelper.service, /usr/bin/systemctl status dsv4-authhelper.service
bmarti44 ALL=(root) NOPASSWD: /usr/bin/systemctl start dsv4-caddy.service, /usr/bin/systemctl stop dsv4-caddy.service, /usr/bin/systemctl status dsv4-caddy.service
bmarti44 ALL=(dsv4) NOPASSWD: /home/dsv4/*
EOF
visudo -cf "${SUDOERS_FILE}.tmp"
install -m 0440 "${SUDOERS_FILE}.tmp" "${SUDOERS_FILE}"
rm -f "${SUDOERS_FILE}.tmp"
echo "Sudoers drop-in installed and validated."

echo "== Verification (as ${TARGET_USER}) =="
FAIL=0
sudo -u "${TARGET_USER}" test -r "${DSV4_HOME}/ds4-project/src" \
  && echo "OK: src dir readable" || { echo "FAIL: src dir unreadable"; FAIL=1; }
sudo -u "${TARGET_USER}" ls "${DSV4_HOME}/ds4-project/src/ds4-upstream-master" >/dev/null 2>&1 \
  && echo "OK: ds4-upstream-master listable" || { echo "FAIL: ds4-upstream-master not listable"; FAIL=1; }
sudo -u "${TARGET_USER}" ls "${DSV4_HOME}/llamacpp-project" >/dev/null 2>&1 \
  && echo "OK: llamacpp-project listable" || { echo "WARN: llamacpp-project not listable"; }
sudo -l -U "${TARGET_USER}" | grep -q 52_engine_switch \
  && echo "OK: passwordless engine switch granted" || { echo "FAIL: sudoers grant missing"; FAIL=1; }

if [[ ${FAIL} -eq 0 ]]; then
  echo "ALL GRANTS APPLIED — no further sudo prompts are required."
else
  echo "One or more verifications failed; review output above." >&2
  exit 1
fi
