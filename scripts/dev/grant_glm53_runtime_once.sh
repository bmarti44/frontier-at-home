#!/usr/bin/env bash
# Install only the owner-requested Docker/containerd qualification permissions.
set -euo pipefail
if (( EUID != 0 )); then
    echo 'Run this one-time installer with sudo.' >&2
    exit 1
fi
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
readonly directory=/etc/sudoers.d
readonly rule=/etc/sudoers.d/glm53-runtime-control
[[ -d $directory && ! -L $directory ]]
[[ $(/usr/bin/stat -c %u "$directory") == 0 ]]
mode=$(/usr/bin/stat -c %a "$directory")
(( (8#$mode & 0022) == 0 ))
/usr/sbin/visudo -c
temporary=$(/usr/bin/mktemp "$directory/.glm53-runtime-control.XXXXXX")
trap '/usr/bin/rm -f -- "$temporary"' EXIT
/usr/bin/cat >"$temporary" <<'POLICY'
# Optional GLM qualification runtime controls; no shell or general ctr access.
bmarti44 ALL=(root) NOPASSWD: /usr/bin/systemctl start docker.service
bmarti44 ALL=(root) NOPASSWD: /usr/bin/systemctl stop docker.service
bmarti44 ALL=(root) NOPASSWD: /usr/bin/systemctl start docker.socket
bmarti44 ALL=(root) NOPASSWD: /usr/bin/systemctl stop docker.socket
bmarti44 ALL=(root) NOPASSWD: /usr/bin/systemctl start containerd.service
bmarti44 ALL=(root) NOPASSWD: /usr/bin/systemctl stop containerd.service
bmarti44 ALL=(root) NOPASSWD: /usr/bin/ctr namespaces list -q
bmarti44 ALL=(root) NOPASSWD: /usr/bin/ctr ^--namespace [A-Za-z0-9_][A-Za-z0-9_.-]* tasks list -q$
POLICY
/usr/bin/chown root:root "$temporary"
/usr/bin/chmod 0440 "$temporary"
/usr/sbin/visudo -cf "$temporary"
if [[ -e $rule || -L $rule ]]; then
    [[ -f $rule && ! -L $rule ]] || { echo 'Existing rule is not a regular file.' >&2; exit 1; }
    /usr/bin/cmp -s "$temporary" "$rule" || { echo 'A different grant already exists; left unchanged.' >&2; exit 1; }
    [[ $(/usr/bin/stat -c %u:%g:%a "$rule") == 0:0:440 ]]
    echo 'The scoped runtime grant is already installed.'
    exit 0
fi
/usr/bin/mv -T -- "$temporary" "$rule"
if ! /usr/sbin/visudo -c; then
    /usr/bin/rm -f -- "$rule"
    echo 'Full sudoers validation failed; the new rule was removed.' >&2
    exit 1
fi
echo 'Scoped runtime access installed. Service state has not changed.'
echo 'Future approved runtime operations can use sudo -n without a password prompt.'
