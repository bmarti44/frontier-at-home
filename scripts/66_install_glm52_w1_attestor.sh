#!/usr/bin/env bash
# One-time install of the root-owned, narrowly delegated GLM W1 authority.
set -Eeuo pipefail
umask 077
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SOURCE=scripts/65_glm52_w1_submit.py
readonly SUBMITTER=/usr/local/sbin/glm52-w1-submit
readonly STATE_ROOT=/var/lib/glm52-w1
readonly RULE=/etc/sudoers.d/glm52-w1-attestor
readonly SUBMITTER_SHA256='ab599a42117b7f718eeaa548b6e34351''276dccf8b36604bbe3abf8fa9acb7a88'

die() { printf '66_install_glm52_w1_attestor.sh: %s\n' "$*" >&2; exit 1; }
git_as_user() {
    /usr/sbin/runuser -u bmarti44 -- /usr/bin/env -i \
        HOME=/nonexistent PATH=/usr/bin:/bin LANG=C.UTF-8 \
        GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
        /usr/bin/git -c core.fsmonitor=false -c core.hooksPath=/dev/null "$@"
}

(( EUID == 0 )) || die "must run as root"
[[ $# == 1 ]] || die "usage: $0 CANDIDATE_HASH"
CANDIDATE_HASH=$1
[[ $CANDIDATE_HASH =~ ^[0-9a-f]{40}$ ]] || die "invalid candidate hash"
actual=$(git_as_user -C "$REPO" rev-parse --verify "$CANDIDATE_HASH^{commit}") ||
    die "candidate does not resolve"
[[ $actual == "$CANDIDATE_HASH" ]] || die "candidate is not exact"
[[ $(git_as_user -C "$REPO" rev-parse HEAD) == "$CANDIDATE_HASH" ]] ||
    die "candidate is not HEAD"
[[ -z $(git_as_user -C "$REPO" status --porcelain --untracked-files=all) ]] ||
    die "repository is not clean"

submitter_temporary=$(/usr/bin/mktemp /run/glm52-w1-submit.XXXXXX)
sudoers_temporary=$(/usr/bin/mktemp /etc/sudoers.d/.glm52-w1-attestor.XXXXXX)
cleanup() {
    /usr/bin/rm -f -- "$submitter_temporary" "$sudoers_temporary"
}
trap cleanup EXIT

git_as_user -C "$REPO" show "$CANDIDATE_HASH:$SOURCE" >"$submitter_temporary"
actual_submitter_sha=$(/usr/bin/sha256sum "$submitter_temporary")
[[ ${actual_submitter_sha%% *} == "$SUBMITTER_SHA256" ]] ||
    die "reviewed submitter digest differs"
/usr/bin/python3 -m py_compile "$submitter_temporary" ||
    die "reviewed submitter is not valid Python"

/usr/bin/printf '%s\n' \
    'bmarti44 ALL=(root) NOPASSWD: /usr/local/sbin/glm52-w1-submit *' \
    >"$sudoers_temporary"
/usr/bin/chown root:root "$sudoers_temporary"
/usr/bin/chmod 0440 "$sudoers_temporary"
/usr/sbin/visudo -cf "$sudoers_temporary"

# Membership in docker plus an active socket is equivalent to unrestricted
# root. Close both paths before installing an authority that claims UID
# separation. Existing shells retain the old supplementary group, but the
# stopped socket leaves them no daemon to control; future logins lose the group.
/usr/bin/systemctl disable --now docker.socket docker.service
if /usr/bin/id -nG bmarti44 | /usr/bin/tr ' ' '\n' |
    /usr/bin/grep -qx docker; then
    /usr/sbin/gpasswd -d bmarti44 docker
fi
if /usr/bin/systemctl is-active --quiet docker.socket; then
    die "docker socket remained active"
fi
if /usr/bin/getent group docker | /usr/bin/cut -d: -f4 |
    /usr/bin/tr ',' '\n' | /usr/bin/grep -qx bmarti44; then
    die "bmarti44 remained in docker group"
fi

/usr/bin/install -d -o root -g root -m 0700 "$STATE_ROOT"
/usr/bin/install -d -o root -g root -m 0700 \
    "$STATE_ROOT/requests" "$STATE_ROOT/by-composite"
/usr/bin/install -o root -g root -m 0755 "$submitter_temporary" "$SUBMITTER"
/usr/bin/install -o root -g root -m 0440 "$sudoers_temporary" "$RULE"
/usr/sbin/visudo -c

trap - EXIT
cleanup
printf 'Installed %s; Docker root delegation is closed. No reboot is needed.\n' \
    "$SUBMITTER"
