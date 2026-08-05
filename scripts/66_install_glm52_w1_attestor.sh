#!/usr/bin/env bash
# One-time install of the root-owned, narrowly delegated GLM W1 authority.
set -Eeuo pipefail
umask 077
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

die() { printf '66_install_glm52_w1_attestor.sh: %s\n' "$*" >&2; exit 1; }

(( EUID == 0 )) || die "must run as root"
reviewed_installer_sha=${GLM52_REVIEWED_INSTALLER_SHA256-}
[[ $reviewed_installer_sha =~ ^[0-9a-f]{64}$ ]] ||
    die "installer must be executed from a reviewed root-owned staged copy"
[[ $(/usr/bin/stat -c '%u:%g:%a:%F' -- "$0") == \
    "0:0:500:regular file" ]] ||
    die "installer must be executed from a reviewed root-owned staged copy"
actual_installer_sha=$(/usr/bin/sha256sum -- "$0")
[[ ${actual_installer_sha%% *} == "$reviewed_installer_sha" ]] ||
    die "reviewed staged installer digest differs"
unset GLM52_REVIEWED_INSTALLER_SHA256 reviewed_installer_sha actual_installer_sha

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SOURCE_UPLOAD_PACK="/usr/bin/git -c safe.directory=$REPO/.git upload-pack"
readonly SOURCE=scripts/65_glm52_w1_submit.py
readonly SUBMITTER=/usr/local/sbin/glm52-w1-submit
readonly LIBEXEC=/usr/local/libexec/glm52-w1
readonly HARNESS=/usr/local/libexec/glm52-w1/harness
readonly PYTHON_RUNTIME=$LIBEXEC/python
readonly PYTHON_DEPENDENCY_SOURCE=/home/bmarti44/.local/lib/python3.12/site-packages
readonly PYTHON_DEPENDENCY_SHA256=39eccffb7a0a2c627bad322ab42a2f07a3b9c55f4952d2867819766c3870bddf
readonly APPROVAL=/usr/local/libexec/glm52-w1/p1-approved.json
readonly CONTROLLER_SOURCE=scripts/81_glm_union_baseline.py
readonly STATE_ROOT=/var/lib/glm52-w1
readonly RULE=/etc/sudoers.d/glm52-w1-attestor
readonly TMPFILES_RULE=/etc/tmpfiles.d/frontier-at-home.conf
readonly LEGACY_LOCK=/run/dsv4/inference.lock
readonly SUBMITTER_SHA256=31d2ea2d72db874767ca647d124c829bed755a8d10529a924796a9332c54309f
readonly CONTAINED_RUNTIME_DIRS=(
    "$HARNESS"
    "$HARNESS/results"
    "$HARNESS/results/glm52-gates"
    "$HARNESS/results/glm52-gates/harness"
    "$HARNESS/scripts"
)
readonly CONTAINED_RUNTIME_FILES=(
    "$HARNESS/results/glm52-gates/harness/glm_safe_run.sh"
    "$HARNESS/scripts/03_memory_guard.py"
)
readonly PYTHON_DEPENDENCIES=(
    numpy numpy.libs nvidia tokenizers torch torchgen typing_extensions.py
)

git_as_user() {
    /usr/sbin/runuser -u bmarti44 -- /usr/bin/env -i \
        HOME=/nonexistent PATH=/usr/bin:/bin LANG=C.UTF-8 \
        GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
        /usr/bin/git -c core.fsmonitor=false -c core.hooksPath=/dev/null "$@"
}

dependency_tree_sha() {
    /usr/bin/env -i HOME=/nonexistent PATH=/usr/bin:/bin LANG=C.UTF-8 \
        LC_ALL=C.UTF-8 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
        /usr/bin/python3 -S - "$1" <<'PY'
import hashlib
import os
import stat
import sys

root = os.path.realpath(sys.argv[1])
names = (
    "numpy", "numpy.libs", "nvidia", "tokenizers", "torch", "torchgen",
    "typing_extensions.py",
)
if set(os.listdir(root)) != set(names):
    raise SystemExit("Python dependency package set differs")
entries = []
for name in names:
    start = os.path.join(root, name)
    details = os.lstat(start)
    if stat.S_ISREG(details.st_mode):
        if details.st_nlink != 1:
            raise SystemExit("unsafe Python dependency file")
        digest = hashlib.sha256()
        with open(start, "rb", buffering=0) as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        entries.append(("F", name, details.st_size, digest.hexdigest()))
        continue
    if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
        raise SystemExit("unsafe Python dependency root")
    for base, directories, files in os.walk(start, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        base_details = os.lstat(base)
        if not stat.S_ISDIR(base_details.st_mode):
            raise SystemExit("unsafe Python dependency directory")
        entries.append(("D", os.path.relpath(base, root), 0, ""))
        for item in directories:
            child_details = os.lstat(os.path.join(base, item))
            if not stat.S_ISDIR(child_details.st_mode) or stat.S_ISLNK(child_details.st_mode):
                raise SystemExit("unsafe Python dependency directory")
        for item in files:
            path = os.path.join(base, item)
            file_details = os.lstat(path)
            if not stat.S_ISREG(file_details.st_mode) or file_details.st_nlink != 1:
                raise SystemExit("unsafe Python dependency file")
            digest = hashlib.sha256()
            with open(path, "rb", buffering=0) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            entries.append((
                "F", os.path.relpath(path, root), file_details.st_size,
                digest.hexdigest(),
            ))
result = hashlib.sha256()
for kind, relative, size, digest in entries:
    result.update(f"{kind}\0{relative}\0{size}\0{digest}\0".encode("utf-8"))
print(result.hexdigest())
PY
}

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
approval_temporary=$(/usr/bin/mktemp /run/glm52-p1-approval.XXXXXX)
sudoers_temporary=$(/usr/bin/mktemp /etc/sudoers.d/.glm52-w1-attestor.XXXXXX)
harness_temporary=$(/usr/bin/mktemp -d /run/glm52-w1-harness.XXXXXX)
install_complete=0
harness_installed=0
python_runtime_installed=0
cleanup() {
    if (( install_complete == 0 && python_runtime_installed == 1 )); then
        /usr/bin/rm -rf -- "$PYTHON_RUNTIME"
        if [[ -d $harness_temporary/previous-python-runtime ]]; then
            /usr/bin/mv -- "$harness_temporary/previous-python-runtime" \
                "$PYTHON_RUNTIME"
        fi
    fi
    if (( install_complete == 0 && harness_installed == 1 )); then
        /usr/bin/rm -rf -- "$HARNESS"
        if [[ -d $harness_temporary/previous-harness ]]; then
            /usr/bin/mv -- "$harness_temporary/previous-harness" "$HARNESS"
        fi
    fi
    /usr/bin/rm -f -- "$submitter_temporary" "$approval_temporary" "$sudoers_temporary"
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
    --upload-pack="$SOURCE_UPLOAD_PACK" \
    "$REPO" "$harness_temporary/repository"
/usr/bin/git -c core.hooksPath=/dev/null -C "$harness_temporary/repository" \
    checkout --detach "$CANDIDATE_HASH"
harness_head=$(
    /usr/bin/git -C "$harness_temporary/repository" rev-parse HEAD
)
[[ $harness_head == "$CANDIDATE_HASH" ]] || die "root harness candidate differs"
[[ -z $(/usr/bin/git -C "$harness_temporary/repository" status --porcelain) ]] ||
    die "root harness is not clean"
python_temporary=$harness_temporary/python-runtime
/usr/bin/install -d -o root -g root -m 0700 "$python_temporary"
for dependency in "${PYTHON_DEPENDENCIES[@]}"; do
    [[ -e $PYTHON_DEPENDENCY_SOURCE/$dependency &&
       ! -L $PYTHON_DEPENDENCY_SOURCE/$dependency ]] ||
        die "frozen Python dependency is absent or unsafe: $dependency"
    /usr/bin/cp -a --reflink=auto -- \
        "$PYTHON_DEPENDENCY_SOURCE/$dependency" "$python_temporary/$dependency"
done
[[ $(dependency_tree_sha "$python_temporary") == "$PYTHON_DEPENDENCY_SHA256" ]] ||
    die "frozen Python dependency content differs"
/usr/bin/chown -R root:root "$python_temporary"
/usr/bin/find "$python_temporary" -type d -exec /usr/bin/chmod 0555 '{}' +
/usr/bin/find "$python_temporary" -type f -exec /usr/bin/chmod 0444 '{}' +
[[ $(dependency_tree_sha "$python_temporary") == "$PYTHON_DEPENDENCY_SHA256" ]] ||
    die "sealed Python dependency content differs"
/usr/bin/env -i HOME=/nonexistent PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$python_temporary" /usr/bin/python3 -S - <<'PY'
import numpy, tokenizers, torch
if not torch.cuda.is_available():
    raise SystemExit("frozen root scorer CUDA runtime is unavailable")
PY
controller_sha=$(/usr/bin/sha256sum \
    "$harness_temporary/repository/$CONTROLLER_SOURCE")
controller_sha=${controller_sha%% *}
[[ $controller_sha =~ ^[0-9a-f]{64}$ ]] ||
    die "reviewed controller digest is invalid"
/usr/bin/python3 - "$approval_temporary" "$CANDIDATE_HASH" "$controller_sha" <<'PY'
import json
import os
import sys

path, candidate, controller = sys.argv[1:]
payload = {
    "schema_version": 1,
    "classification": "GLM52_P1_ROOT_APPROVED_CANDIDATE",
    "candidate_hash": candidate,
    "controller_sha256": controller,
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
/usr/bin/chown root:root "$approval_temporary"
/usr/bin/chmod 0444 "$approval_temporary"

/usr/bin/printf '%s\n' \
    'bmarti44 ALL=(root) NOPASSWD: /usr/local/sbin/glm52-w1-submit *' \
    >"$sudoers_temporary"
/usr/bin/chown root:root "$sudoers_temporary"
/usr/bin/chmod 0440 "$sudoers_temporary"
/usr/sbin/visudo -cf "$sudoers_temporary"

# Convert the legacy namespace before opening it from shell. O_NOFOLLOW plus
# descriptor/path identity checks prevent a dsv4-controlled symlink swap; the
# root-owned sticky directory then makes the root-owned lock nonreplaceable.
/usr/bin/install -d -o root -g dsv4 -m 1770 /run/dsv4
/usr/bin/python3 - "$LEGACY_LOCK" <<'PY'
import fcntl
import grp
import os
import stat
import sys

path = sys.argv[1]
flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
try:
    descriptor = os.open(path, flags)
except FileNotFoundError:
    try:
        descriptor = os.open(
            path,
            flags | os.O_CREAT | os.O_EXCL,
            0o660,
        )
    except FileExistsError:
        descriptor = os.open(path, flags)
try:
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
        raise SystemExit("legacy inference lock is not a private regular file")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("a pre-migration inference server is still running")
    os.fchown(descriptor, 0, grp.getgrnam("dsv4").gr_gid)
    os.fchmod(descriptor, 0o660)
    opened = os.fstat(descriptor)
    visible = os.lstat(path)
    if (
        stat.S_ISLNK(visible.st_mode)
        or opened.st_dev != visible.st_dev
        or opened.st_ino != visible.st_ino
        or visible.st_uid != 0
        or visible.st_nlink != 1
    ):
        raise SystemExit("legacy inference lock changed during conversion")
finally:
    os.close(descriptor)
PY
exec 8<>"$LEGACY_LOCK"
/usr/bin/flock -n -E 75 8 ||
    die "a pre-migration inference server is still running"

# Membership in docker plus an active socket is equivalent to unrestricted
# root. Close both paths before installing an authority that claims UID
# separation. Existing shells retain the old supplementary group, but the
# stopped socket leaves them no daemon to control; future logins lose the group.
/usr/bin/systemctl disable --now docker.socket docker.service containerd.service
if /usr/bin/id -nG bmarti44 | /usr/bin/tr ' ' '\n' |
    /usr/bin/grep -qx docker; then
    /usr/bin/gpasswd -d bmarti44 docker
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
/usr/bin/install -d -o root -g root -m 0700 "$STATE_ROOT/p1-results"
/usr/bin/install -d -o root -g root -m 0700 "$STATE_ROOT/p1-result-receipts"
/usr/bin/install -d -o root -g root -m 0755 "$LIBEXEC"
/usr/bin/printf '%s\n' \
    'd /run/dsv4 1770 root dsv4 -' \
    'f /run/dsv4/inference.lock 0660 root dsv4 -' \
    'd /run/lock/frontier-at-home 0750 root dsv4 -' \
    'f /run/lock/frontier-at-home/inference.lock 0660 root dsv4 -' \
    >"$harness_temporary/frontier-at-home.conf"
/usr/bin/install -o root -g root -m 0644 \
    "$harness_temporary/frontier-at-home.conf" "$TMPFILES_RULE"
/usr/bin/systemd-tmpfiles --create "$TMPFILES_RULE"
for unit in \
    deepseek-v4-flash-ds4.service \
    deepseek-v4-flash-llamacpp.service
do
    dropin="/etc/systemd/system/$unit.d"
    /usr/bin/install -d -o root -g root -m 0755 "$dropin"
    /usr/bin/printf '%s\n' '[Service]' 'RuntimeDirectory=' \
        >"$dropin/frontier-runtime.conf"
    /usr/bin/chown root:root "$dropin/frontier-runtime.conf"
    /usr/bin/chmod 0644 "$dropin/frontier-runtime.conf"
done
/usr/bin/systemctl daemon-reload
if [[ -e $HARNESS ]]; then
    /usr/bin/mv -- "$HARNESS" "$harness_temporary/previous-harness"
fi
/usr/bin/cp -a -- "$harness_temporary/repository" "$HARNESS"
harness_installed=1
/usr/bin/chown -R root:root "$HARNESS"
if [[ -e $PYTHON_RUNTIME ]]; then
    /usr/bin/mv -- "$PYTHON_RUNTIME" \
        "$harness_temporary/previous-python-runtime"
fi
python_runtime_installed=1
/usr/bin/cp -a -- "$python_temporary" "$PYTHON_RUNTIME"
/usr/bin/chown -R root:root "$PYTHON_RUNTIME"
[[ $(dependency_tree_sha "$PYTHON_RUNTIME") == "$PYTHON_DEPENDENCY_SHA256" ]] ||
    die "installed Python dependency content differs"
for path in "${CONTAINED_RUNTIME_DIRS[@]}"; do
    [[ -d $path && ! -L $path ]] ||
        die "contained runtime directory is absent or unsafe"
done
for path in "${CONTAINED_RUNTIME_FILES[@]}"; do
    [[ -f $path && ! -L $path ]] ||
        die "contained runtime file is absent or unsafe"
done
/usr/bin/chmod 0555 "${CONTAINED_RUNTIME_DIRS[@]}"
/usr/bin/chmod 0444 "${CONTAINED_RUNTIME_FILES[@]}"
for path in "${CONTAINED_RUNTIME_DIRS[@]}"; do
    /usr/sbin/runuser -u dsv4 -- /usr/bin/test -x "$path" ||
        die "contained account cannot traverse runtime directory"
done
for path in "${CONTAINED_RUNTIME_FILES[@]}"; do
    /usr/sbin/runuser -u dsv4 -- /usr/bin/test -r "$path" ||
        die "contained account cannot read runtime file"
done
/usr/bin/install -o root -g root -m 0755 "$submitter_temporary" "$SUBMITTER"
/usr/bin/install -o root -g root -m 0444 "$approval_temporary" "$APPROVAL"
/usr/bin/install -o root -g root -m 0440 "$sudoers_temporary" "$RULE"
/usr/sbin/visudo -c

install_complete=1
trap - EXIT
cleanup
printf 'Installed %s; Docker root delegation is closed. No reboot is needed.\n' \
    "$SUBMITTER"
