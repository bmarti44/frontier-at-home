#!/usr/bin/env bash
# One-time install of the root-owned, narrowly delegated context attestor.
set -Eeuo pipefail
umask 077
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SUBMIT=/usr/local/sbin/dsv4-context-submit
readonly RULE=/etc/sudoers.d/dsv4-context-attestor
readonly OLD_RULE=/etc/sudoers.d/dsv4-delegate
readonly SUBMIT_SHA256='8297bef99d15b26732a8a9a739c07ae9''24e77112cddb051821742e41576c3f5b'

die() { printf '63_install_context_attestor.sh: %s\n' "$*" >&2; exit 1; }
(( EUID == 0 )) || die "must run as root"
[[ $# == 1 ]] || die "usage: $0 CANDIDATE_HASH"
CANDIDATE_HASH=$1
[[ $CANDIDATE_HASH =~ ^[0-9a-f]{40}$ ]] || die "invalid candidate hash"
actual=$(
    /usr/bin/git -c safe.directory="$REPO" -C "$REPO" \
        rev-parse --verify "$CANDIDATE_HASH^{commit}"
) || die "candidate does not resolve"
[[ $actual == "$CANDIDATE_HASH" ]] || die "candidate is not exact"
[[ $(
    /usr/bin/git -c safe.directory="$REPO" -C "$REPO" rev-parse HEAD
) == "$CANDIDATE_HASH" ]] || die "candidate is not HEAD"
[[ -z $(
    /usr/bin/git -c safe.directory="$REPO" -C "$REPO" status --porcelain
) ]] || die "repository is not clean"

submit_temporary=$(/usr/bin/mktemp /run/dsv4-context-submit.XXXXXX)
temporary=$(/usr/bin/mktemp /etc/sudoers.d/.dsv4-context-attestor.XXXXXX)
cleanup() {
    /usr/bin/rm -f -- "$temporary"
    /usr/bin/rm -f -- "$submit_temporary"
}
trap cleanup EXIT
/usr/bin/git -c safe.directory="$REPO" -C "$REPO" \
    show "$CANDIDATE_HASH:scripts/64_context_submit.sh" >"$submit_temporary"
actual_submit_sha=$(/usr/bin/sha256sum "$submit_temporary")
[[ ${actual_submit_sha%% *} == "$SUBMIT_SHA256" ]] ||
    die "reviewed submitter digest differs"
/usr/bin/install -o root -g root -m 0755 "$submit_temporary" "$SUBMIT"
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
