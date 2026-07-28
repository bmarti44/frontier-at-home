#!/usr/bin/env bash
# One-time install of the root-owned, narrowly delegated context attestor.
set -Eeuo pipefail
umask 077
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SUBMIT=/usr/local/sbin/dsv4-context-submit
readonly RULE=/etc/sudoers.d/dsv4-context-attestor
readonly OLD_RULE=/etc/sudoers.d/dsv4-delegate

die() { printf '63_install_context_attestor.sh: %s\n' "$*" >&2; exit 1; }
(( EUID == 0 )) || die "must run as root"
[[ -f $REPO/scripts/64_context_submit.sh ]] || die "submitter is missing"

/usr/bin/install -o root -g root -m 0755 \
    "$REPO/scripts/64_context_submit.sh" "$SUBMIT"
temporary=$(/usr/bin/mktemp /etc/sudoers.d/.dsv4-context-attestor.XXXXXX)
cleanup() {
    /usr/bin/rm -f -- "$temporary"
}
trap cleanup EXIT
/usr/bin/printf '%s\n' \
    'bmarti44 ALL=(root) NOPASSWD: /usr/local/sbin/dsv4-context-submit *' \
    >"$temporary"
/usr/bin/chown root:root "$temporary"
/usr/bin/chmod 0440 "$temporary"
/usr/sbin/visudo -cf "$temporary"
/usr/bin/install -o root -g root -m 0440 "$temporary" "$RULE"
if [[ -e $OLD_RULE ]]; then
    /usr/bin/rm -f -- "$OLD_RULE"
fi
/usr/sbin/visudo -c
trap - EXIT
cleanup
printf 'Installed %s and replaced unrestricted dsv4 delegation.\n' "$SUBMIT"
