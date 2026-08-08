#!/bin/bash
# Install the reviewed W9 runner behind one root-owned, narrow sudo boundary.
set -Eeuo pipefail
umask 077

die() { echo "95_install_glm52_w9_runner.sh: $*" >&2; exit 1; }

[[ $EUID == 0 ]] || die "must run as root"
[[ $# == 1 && $1 =~ ^[0-9a-f]{40}$ ]] || die "usage: $0 REVIEW_COMMIT"
readonly REVIEW_COMMIT=$1
readonly EXPECTED_INSTALLER_SHA=${GLM52_REVIEWED_INSTALLER_SHA256-}
[[ $EXPECTED_INSTALLER_SHA =~ ^[0-9a-f]{64}$ ]] ||
  die "GLM52_REVIEWED_INSTALLER_SHA256 is required"
installer_info=$(/usr/bin/stat -Lc '%U:%G:%a:%h' -- "$0")
[[ $installer_info == root:root:500:1 ]] ||
  die "installer must be a root-owned 0500 single-link staged copy"
[[ $(/usr/bin/sha256sum -- "$0") == "$EXPECTED_INSTALLER_SHA  $0" ]] ||
  die "staged installer digest differs"
unset GLM52_REVIEWED_INSTALLER_SHA256 installer_info

readonly ORIGIN=https://github.com/bmarti44/frontier-at-home.git
readonly TARGET=/usr/local/libexec/glm52-w9
readonly RUNNER=/usr/local/sbin/glm52-w9-submit
readonly SUDOERS=/etc/sudoers.d/glm52-w9-runner
readonly REVIEW_REL=results/glm52-gates/W9-fp4-falsifier-review-r254.json
readonly FREEZE_REL=results/glm52-gates/W9-fp4-falsifier-candidate4-freeze.json
readonly SOURCE_NODE=/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node
readonly SOURCE_NOBLE=/home/bmarti44/.cache/glm52-drand-client-1.4.2/node_modules/@noble
readonly ROOT_NUMPY=/usr/local/libexec/glm52-w1/python

clone_root=$(/usr/bin/mktemp -d /run/glm52-w9-install.XXXXXX)
install_root=$(/usr/bin/mktemp -d /usr/local/libexec/.glm52-w9.XXXXXX)
sudoers_tmp=$(/usr/bin/mktemp /etc/sudoers.d/.glm52-w9.XXXXXX)
runner_tmp=$(/usr/bin/mktemp /usr/local/sbin/.glm52-w9-submit.XXXXXX)
published=0
cleanup() {
  if [[ $published == 1 ]]; then
    /usr/bin/rm -rf -- "$TARGET"
    /usr/bin/rm -f -- "$RUNNER" "$SUDOERS"
  fi
  /usr/bin/rm -rf -- "$clone_root" "$install_root"
  /usr/bin/rm -f -- "$sudoers_tmp" "$runner_tmp"
}
trap cleanup EXIT

[[ ! -e $TARGET && ! -L $TARGET ]] || die "W9 install already exists"
/usr/bin/git clone --no-checkout -- "$ORIGIN" "$clone_root/repository"
/usr/bin/git -C "$clone_root/repository" checkout --detach "$REVIEW_COMMIT"
[[ $(/usr/bin/git -C "$clone_root/repository" rev-parse HEAD) == "$REVIEW_COMMIT" ]] ||
  die "review commit checkout differs"
[[ -z $(/usr/bin/git -C "$clone_root/repository" status --porcelain) ]] ||
  die "review checkout is dirty"

/usr/bin/python3 -S - "$clone_root/repository" "$REVIEW_COMMIT" "$REVIEW_REL" "$FREEZE_REL" <<'PY'
import hashlib, json, pathlib, re, subprocess, sys
repo = pathlib.Path(sys.argv[1]); review_commit, review_rel, freeze_rel = sys.argv[2:]
review = json.loads((repo / review_rel).read_text(encoding="utf-8"))
candidate = review.get("candidate_hash")
if not isinstance(candidate, str) or not re.fullmatch(r"[0-9a-f]{40}", candidate):
    raise SystemExit("review candidate is invalid")
if (review.get("review_round") != 254 or review.get("critical") != [] or
        review.get("high") != [] or review.get("verdict") != "PASS_RUNTIME_ALLOWED"):
    raise SystemExit("review does not authorize runtime")
freeze_bytes = subprocess.run(
    ["/usr/bin/git", "-C", str(repo), "show", f"{candidate}:{freeze_rel}"],
    check=True, capture_output=True).stdout
freeze = json.loads(freeze_bytes)
if freeze.get("schema") != "glm52-w9-fp4-falsifier-freeze-v4":
    raise SystemExit("candidate freeze schema differs")
for relative, expected in freeze.get("component_sha256", {}).items():
    current = (repo / relative).read_bytes()
    candidate_bytes = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "show", f"{candidate}:{relative}"],
        check=True, capture_output=True).stdout
    if current != candidate_bytes or hashlib.sha256(current).hexdigest() != expected:
        raise SystemExit(f"reviewed component differs: {relative}")
if not subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "merge-base", "--is-ancestor", candidate, review_commit]).returncode == 0:
    raise SystemExit("reviewed candidate is not an ancestor")
print(candidate)
PY
candidate=$(/usr/bin/python3 -S -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["candidate_hash"])' \
  "$clone_root/repository/$REVIEW_REL")

/usr/bin/cp -a -- "$clone_root/repository" "$install_root/repository"
/usr/bin/install -o root -g root -m 0555 -- "$SOURCE_NODE" "$install_root/node"
/usr/bin/cp -a -- "$SOURCE_NOBLE" "$install_root/noble"
/usr/bin/install -o root -g root -m 0555 -- \
  "$clone_root/repository/scripts/94_glm52_w9_submit.py" "$runner_tmp"

/usr/bin/python3 -S - "$install_root" "$runner_tmp" "$candidate" "$REVIEW_COMMIT" "$ROOT_NUMPY" <<'PY'
import hashlib, json, pathlib, stat, sys
root=pathlib.Path(sys.argv[1]); runner=pathlib.Path(sys.argv[2])
candidate=sys.argv[3]; review=sys.argv[4]; runtime=pathlib.Path(sys.argv[5])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 << 20): h.update(block)
    return h.hexdigest()
def tree(path):
    rows=[]
    for item in sorted(path.rglob('*')):
        info=item.lstat()
        if stat.S_ISDIR(info.st_mode): continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1: raise SystemExit('unsafe tree')
        rows.append((item.relative_to(path).as_posix(), info.st_size, sha(item)))
    h=hashlib.sha256()
    for name,size,digest in rows:
        h.update(name.encode()+b'\0'+str(size).encode()+b'\0'+digest.encode()+b'\n')
    return h.hexdigest()
repository=root/'repository'
manifest={
  'schema':'glm52-w9-install-v1', 'candidate_hash':candidate, 'review_commit':review,
  'runner_sha256':sha(runner),
  'scorer_sha256':sha(repository/'scripts/93_score_w9_fp4_falsifier.py'),
  'python_sha256':sha(pathlib.Path('/usr/bin/python3')),
  'node_sha256':sha(root/'node'), 'noble_tree_sha256':tree(root/'noble'),
  'runtime_tree_sha256':hashlib.sha256(
      (tree(runtime/'numpy')+'\0'+tree(runtime/'numpy.libs')).encode()).hexdigest(),
  'repository_tree_sha256':tree(repository),
  'review_receipt_sha256':sha(repository/'results/glm52-gates/W9-fp4-falsifier-review-r254.json'),
}
(root/'install.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n')
PY

/usr/bin/chown -R root:root -- "$install_root"
/usr/bin/find "$install_root" -type d -exec /usr/bin/chmod 0555 {} +
/usr/bin/find "$install_root" -type f -exec /usr/bin/chmod 0444 {} +
/usr/bin/git -C "$install_root/repository" ls-files -s -z | \
  while IFS=$' \t' read -r -d '' mode object stage relative; do
    if [[ $mode == 100755 ]]; then
      /usr/bin/chmod 0555 "$install_root/repository/$relative"
    fi
  done
/usr/bin/chmod 0555 "$install_root/node"
/usr/bin/chmod 0555 "$runner_tmp"
/usr/bin/chown root:root "$runner_tmp"

/usr/bin/printf '%s\n' \
  'bmarti44 ALL=(root) NOPASSWD: /usr/local/sbin/glm52-w9-submit *' > "$sudoers_tmp"
/usr/bin/chown root:root "$sudoers_tmp"
/usr/bin/chmod 0440 "$sudoers_tmp"
/usr/sbin/visudo -cf "$sudoers_tmp"

# Atomic RENAME_NOREPLACE publications; no user-writable path is executed.
[[ ! -e $RUNNER && ! -L $RUNNER ]] || die "W9 runner already exists"
/usr/bin/python3 -S - "$install_root" "$TARGET" <<'PY'
import ctypes, errno, os, sys
libc=ctypes.CDLL(None,use_errno=True); fn=libc.renameat2
fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
source,dest=sys.argv[1:]
if fn(-100,os.fsencode(source),-100,os.fsencode(dest),1):
    code=ctypes.get_errno(); raise OSError(code,os.strerror(code),dest)
PY
/usr/bin/python3 -S - "$runner_tmp" "$RUNNER" <<'PY' || {
import ctypes, os, sys
libc=ctypes.CDLL(None,use_errno=True); fn=libc.renameat2
if fn(-100,os.fsencode(sys.argv[1]),-100,os.fsencode(sys.argv[2]),1):
    raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()),sys.argv[2])
PY
  /usr/bin/rm -rf -- "$TARGET"
  die "runner publication failed; install rolled back"
}
published=1
/usr/bin/install -o root -g root -m 0440 -- "$sudoers_tmp" "$SUDOERS"
/usr/sbin/visudo -c
/usr/bin/mkdir -p /var/lib/glm52-w9/{work,attempts,failures}
/usr/bin/chown -R root:root /var/lib/glm52-w9
/usr/bin/chmod 0555 /var/lib/glm52-w9 /var/lib/glm52-w9/attempts
/usr/bin/chmod 0700 /var/lib/glm52-w9/work /var/lib/glm52-w9/failures
published=0
trap - EXIT
/usr/bin/rm -rf -- "$clone_root"
/usr/bin/rm -f -- "$sudoers_tmp"
echo "Installed root-owned glm52-w9-submit for candidate $candidate; no reboot is needed."
