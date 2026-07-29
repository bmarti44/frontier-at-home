#!/usr/bin/env bash
# One-time install of the root-owned, narrowly delegated GLM W1 authority.
set -Eeuo pipefail
umask 077
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SOURCE=scripts/65_glm52_w1_submit.py
readonly SUBMITTER=/usr/local/sbin/glm52-w1-submit
readonly LIBEXEC=/usr/local/libexec/glm52-w1
readonly HARNESS=/usr/local/libexec/glm52-w1/harness
readonly STATE_ROOT=/var/lib/glm52-w1
readonly RULE=/etc/sudoers.d/glm52-w1-attestor
readonly TMPFILES_RULE=/etc/tmpfiles.d/frontier-at-home.conf
readonly SUBMITTER_SHA256='ad15c685148dab5eaf7388809146c244''7411014469413db952aece5a95a65cf0'

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
harness_temporary=$(/usr/bin/mktemp -d /run/glm52-w1-harness.XXXXXX)
install_complete=0
harness_installed=0
cleanup() {
    if (( install_complete == 0 && harness_installed == 1 )); then
        /usr/bin/rm -rf -- "$HARNESS"
        if [[ -d $harness_temporary/previous-harness ]]; then
            /usr/bin/mv -- "$harness_temporary/previous-harness" "$HARNESS"
        fi
    fi
    /usr/bin/rm -f -- "$submitter_temporary" "$sudoers_temporary"
    /usr/bin/rm -rf -- "$harness_temporary"
}
trap cleanup EXIT

git_as_user -C "$REPO" show "$CANDIDATE_HASH:$SOURCE" >"$submitter_temporary"
actual_submitter_sha=$(/usr/bin/sha256sum "$submitter_temporary")
[[ ${actual_submitter_sha%% *} == "$SUBMITTER_SHA256" ]] ||
    die "reviewed submitter digest differs"
/usr/bin/python3 -m py_compile "$submitter_temporary" ||
    die "reviewed submitter is not valid Python"
/usr/bin/git -c core.hooksPath=/dev/null clone --no-local --no-checkout \
    "$REPO" "$harness_temporary/repository"
/usr/bin/git -c core.hooksPath=/dev/null -C "$harness_temporary/repository" \
    checkout --detach "$CANDIDATE_HASH"
harness_head=$(
    /usr/bin/git -C "$harness_temporary/repository" rev-parse HEAD
)
[[ $harness_head == "$CANDIDATE_HASH" ]] || die "root harness candidate differs"
[[ -z $(/usr/bin/git -C "$harness_temporary/repository" status --porcelain) ]] ||
    die "root harness is not clean"

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
/usr/bin/systemctl disable --now docker.socket docker.service containerd.service
if /usr/bin/id -nG bmarti44 | /usr/bin/tr ' ' '\n' |
    /usr/bin/grep -qx docker; then
    /usr/sbin/gpasswd -d bmarti44 docker
fi
if /usr/bin/pgrep -x dockerd >/dev/null 2>&1; then
    /usr/bin/pkill -TERM -x dockerd
    for _ in 1 2 3 4 5; do
        /usr/bin/pgrep -x dockerd >/dev/null 2>&1 || break
        /usr/bin/sleep 1
    done
fi
if /usr/bin/pgrep -x dockerd >/dev/null 2>&1; then
    /usr/bin/pkill -KILL -x dockerd
fi
if /usr/bin/getent group docker >/dev/null; then
    /usr/sbin/groupdel docker
fi
for unit in docker.socket docker.service containerd.service; do
    if /usr/bin/systemctl is-active --quiet "$unit"; then
        die "container runtime remained active: $unit"
    fi
done
if /usr/bin/pgrep -x dockerd >/dev/null 2>&1 ||
    /usr/bin/pgrep -x containerd >/dev/null 2>&1; then
    die "container runtime process remained active"
fi
if /usr/bin/getent group docker | /usr/bin/cut -d: -f4 |
    /usr/bin/tr ',' '\n' | /usr/bin/grep -qx bmarti44; then
    die "bmarti44 remained in docker group"
fi

/usr/bin/install -d -o root -g root -m 0755 "$STATE_ROOT"
/usr/bin/install -d -o root -g root -m 0711 "$STATE_ROOT/requests"
/usr/bin/install -d -o root -g root -m 0755 "$STATE_ROOT/by-composite"
/usr/bin/install -d -o root -g root -m 0755 "$STATE_ROOT/controller-attempts"
/usr/bin/install -d -o root -g root -m 0755 "$LIBEXEC"
/usr/bin/printf '%s\n' \
    'd /run/lock/frontier-at-home 0750 root dsv4 -' \
    'f /run/lock/frontier-at-home/inference.lock 0660 root dsv4 -' \
    >"$harness_temporary/frontier-at-home.conf"
/usr/bin/install -o root -g root -m 0644 \
    "$harness_temporary/frontier-at-home.conf" "$TMPFILES_RULE"
/usr/bin/systemd-tmpfiles --create "$TMPFILES_RULE"
if [[ -e $HARNESS ]]; then
    /usr/bin/mv -- "$HARNESS" "$harness_temporary/previous-harness"
fi
/usr/bin/cp -a -- "$harness_temporary/repository" "$HARNESS"
harness_installed=1
/usr/bin/chown -R root:root "$HARNESS"
/usr/bin/find "$HARNESS" -type d -exec /usr/bin/chmod 0755 {} +
/usr/bin/find "$HARNESS" -type f -exec /usr/bin/chmod 0644 {} +
/usr/bin/chmod 0755 \
    "$HARNESS/scripts/glm52_w1_affine_campaign.py" \
    "$HARNESS/results/glm52-gates/harness/glm_cgroup_run.sh" \
    "$HARNESS/results/glm52-gates/harness/glm_safe_run.sh"
/usr/bin/install -o root -g root -m 0755 "$submitter_temporary" "$SUBMITTER"
/usr/bin/install -o root -g root -m 0440 "$sudoers_temporary" "$RULE"
/usr/sbin/visudo -c

install_complete=1
trap - EXIT
cleanup
printf 'Installed %s; Docker root delegation is closed. No reboot is needed.\n' \
    "$SUBMITTER"
