#!/usr/bin/env bash
# W7.1 one-arm diagnostic. With no arguments this launcher creates a fresh
# evidence directory and executes its --driver arm through the reviewed GLM
# cgroup + safe-run containment path.
set -Eeuo pipefail
umask 077

readonly ROOT=/home/bmarti44/spark-deepseek-v4-flash
readonly HARNESS="$ROOT/results/glm52-gates/harness/w7_cache_generation_smoke_v1.sh"
readonly CGROUP="$ROOT/results/glm52-gates/harness/glm_cgroup_run.sh"
readonly SAFE="$ROOT/results/glm52-gates/harness/glm_safe_run.sh"
readonly SCORER="$ROOT/scripts/89_score_w7_cache_generation.py"
readonly MEMORY_GUARD="$ROOT/scripts/03_memory_guard.py"
readonly BIN=/home/bmarti44/.cache/glm52-w7-stable-remap-bccf0b6/ds4-server
readonly CANDIDATE_SRC=/home/bmarti44/.cache/glm52-w7-stable-remap-bccf0b6
readonly BINARY_SHA256=eec10ca8aae5ef685e5420b02a56a1b76afaac9416acd58efb4230b15678a4d2
readonly MODEL=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
readonly MODEL_BYTES=211075856448
readonly MODEL_SHA256=a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0
readonly LIVE=/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/live-request.json
readonly LIVE_SHA256=d1def599a8bbfcd3a49e97d3c467fe30264caa241e9fa7cf717e5550c2bb601a
readonly PRIMARY=/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/primary-request.json
readonly PRIMARY_SHA256=a453691312004c144474d0fc8f27c17e38aec055a353a20bb2e9946f265667f3
readonly PORT=8097
server_pid=
attempt=
out=
candidate=${DS4_W7_CANDIDATE_HASH:-}
outer_finalized=0
failure_reason=before-attempt
containment_stdout=
containment_rc=
seal_holder_pid=${DS4_W7_SEAL_HOLDER_PID:-}
seal_holder_start_ticks=${DS4_W7_SEAL_HOLDER_START_TICKS:-}
seal_holder_parent_pid=${DS4_W7_SEAL_HOLDER_PARENT_PID:-}
harness_fd_path=${DS4_W7_SEALED_HARNESS_PATH:-}
cgroup_fd_path=${DS4_W7_SEALED_CGROUP_PATH:-}
safe_fd_path=${DS4_W7_SEALED_SAFE_PATH:-}
scorer_fd_path=${DS4_W7_SEALED_SCORER_PATH:-}
memory_guard_fd_path=${DS4_W7_SEALED_MEMORY_GUARD_PATH:-}
live_fd_path=${DS4_W7_SEALED_LIVE_PATH:-$LIVE}
primary_fd_path=${DS4_W7_SEALED_PRIMARY_PATH:-$PRIMARY}
harness_sha256=${DS4_W7_PINNED_HARNESS_SHA256:-}
cgroup_sha256=${DS4_W7_SEALED_CGROUP_SHA256:-}
safe_sha256=${DS4_W7_SEALED_SAFE_SHA256:-}
scorer_sha256=${DS4_W7_SEALED_SCORER_SHA256:-}
memory_guard_sha256=${DS4_W7_SEALED_MEMORY_GUARD_SHA256:-}
engine_lock_fd=
engine_lock_fd_path=
engine_lock_holder_pid=
engine_lock_holder_start_ticks=
engine_lock_holder_parent_pid=
engine_lock_identity=
engine_lock_metadata_fd=
environment_sha256=

has_full_seal() {
  /usr/bin/python3 - "$1" <<'PY'
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDONLY | os.O_CLOEXEC)
try:
    required = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    raise SystemExit(0 if fcntl.fcntl(fd, fcntl.F_GET_SEALS) == required else 1)
finally:
    os.close(fd)
PY
}

if [[ $0 == "$HARNESS" && ! -L $0 && $(readlink -f -- "$0") == "$HARNESS" ]]; then
  :
elif [[ $0 =~ ^/proc/[1-9][0-9]*/fd/[0-9]+$ &&
        ${1:-} =~ ^(--sealed-outer|--sealed-holder-loss-test|--sealed-lineage-self-test|--driver|--driver-lineage-self-test)$ &&
        ${DS4_W7_PINNED_HARNESS_SHA256:-} =~ ^[0-9a-f]{64}$ &&
        $(/usr/bin/sha256sum -- "$0" | /usr/bin/awk '{print $1}') == "$DS4_W7_PINNED_HARNESS_SHA256" ]] &&
        has_full_seal "$0"; then
  :
else
  exit 2
fi

verify_file() {
  [[ -f $1 && ! -L $1 && $(sha256sum -- "$1" | awk '{print $1}') == "$2" ]]
}

verify_sealed_file() {
  [[ $1 =~ ^/proc/[1-9][0-9]*/fd/[0-9]+$ && $2 =~ ^[0-9a-f]{64}$ ]]
  python3 - "$1" "$2" <<'PY'
import fcntl, hashlib, os, sys
path, expected = sys.argv[1:]
fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
try:
    required = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    if fcntl.fcntl(fd, fcntl.F_GET_SEALS) != required:
        raise SystemExit(1)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit(1)
finally:
    os.close(fd)
PY
}

verify_dependencies_fast() {
  verify_file "$BIN" "$BINARY_SHA256"
  [[ -f $MODEL && ! -L $MODEL && $(stat -Lc '%s' -- "$MODEL") == "$MODEL_BYTES" ]]
  if [[ $live_fd_path == /proc/*/fd/* ]]; then
    verify_sealed_file "$live_fd_path" "$LIVE_SHA256"
    verify_sealed_file "$primary_fd_path" "$PRIMARY_SHA256"
  else
    verify_file "$live_fd_path" "$LIVE_SHA256"
    verify_file "$primary_fd_path" "$PRIMARY_SHA256"
  fi
  [[ -f $CGROUP && ! -L $CGROUP && -f $SAFE && ! -L $SAFE && -f $SCORER && ! -L $SCORER ]]
}

prepare_engine_lock() {
  local directory=$1 leaf=$1/.ds4-engine-lock
  [[ -d $directory && ! -L $directory && $(stat -Lc '%a:%u' -- "$directory") == 700:$(id -u) ]] || return 1
  exec {engine_lock_metadata_fd}< <(
    /usr/bin/python3 - "$directory" "$$" <<'PY'
import ctypes, os, pathlib, signal, stat, sys

directory, expected_parent = sys.argv[1:]
libc = ctypes.CDLL(None, use_errno=True)
libc.prctl(1, signal.SIGTERM)
if str(os.getppid()) != expected_parent:
    raise SystemExit("engine lock holder parent mismatch")
directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
try:
    try:
        descriptor = os.open(
            ".ds4-engine-lock",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
    except FileExistsError:
        raise SystemExit(17)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise SystemExit("new engine lock is not a one-link regular file")
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise SystemExit("new engine lock ownership or mode mismatch")
    os.unlink(".ds4-engine-lock", dir_fd=directory_fd)
finally:
    os.close(directory_fd)
metadata = os.fstat(descriptor)
if metadata.st_nlink != 0:
    raise SystemExit("engine lock descriptor remains pathname-linked")
start_ticks = pathlib.Path("/proc/self/stat").read_text().split()[21]
print(
    f"{os.getpid()}\t{start_ticks}\t{os.getppid()}\t{descriptor}\t{metadata.st_dev}:{metadata.st_ino}",
    flush=True,
)
while True:
    signal.pause()
PY
  )
  IFS=$'\t' read -r engine_lock_holder_pid engine_lock_holder_start_ticks \
    engine_lock_holder_parent_pid engine_lock_fd engine_lock_identity <&"$engine_lock_metadata_fd" || return 1
  [[ $engine_lock_holder_pid =~ ^[1-9][0-9]*$ &&
     $engine_lock_holder_start_ticks =~ ^[1-9][0-9]*$ &&
     $engine_lock_holder_parent_pid == "$$" && $engine_lock_fd =~ ^[0-9]+$ &&
     $engine_lock_identity =~ ^[0-9]+:[0-9]+$ ]] || return 1
  verify_engine_lock_holder_identity || return 1
  engine_lock_fd_path="/proc/$engine_lock_holder_pid/fd/$engine_lock_fd"
  environment_sha256=$(printf 'DS4_CUDA_STABLE_MODEL_REMAP=1\nDS4_LOCK_FILE=%s\n' "$engine_lock_fd_path" | sha256sum | awk '{print $1}')
  [[ $environment_sha256 =~ ^[0-9a-f]{64}$ ]] || return 1
}

verify_engine_lock_holder_identity() {
  /usr/bin/python3 - "$engine_lock_holder_pid" "$engine_lock_holder_start_ticks" \
    "$engine_lock_holder_parent_pid" "$engine_lock_fd" "$engine_lock_identity" <<'PY'
import os, pathlib, stat, sys
pid, expected_start, expected_parent, descriptor, expected_identity = sys.argv[1:]
process_stat = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
status = pathlib.Path(f"/proc/{pid}/status").read_text().splitlines()
parent = next(line.split()[1] for line in status if line.startswith("PPid:"))
metadata = os.stat(f"/proc/{pid}/fd/{descriptor}")
identity = f"{metadata.st_dev}:{metadata.st_ino}"
if process_stat[21] != expected_start or parent != expected_parent:
    raise SystemExit("engine lock holder identity changed")
if identity != expected_identity or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 0:
    raise SystemExit("engine lock descriptor identity changed")
PY
}

stop_engine_lock_holder() {
  if [[ ${engine_lock_holder_pid:-} =~ ^[1-9][0-9]*$ ]]; then
    verify_engine_lock_holder_identity || return 1
    kill -TERM "$engine_lock_holder_pid" 2>/dev/null || true
    wait "$engine_lock_holder_pid" 2>/dev/null || true
    engine_lock_holder_pid=
  fi
  if [[ ${engine_lock_metadata_fd:-} =~ ^[0-9]+$ ]]; then
    exec {engine_lock_metadata_fd}<&-
    engine_lock_metadata_fd=
  fi
}

verify_reviewed_sources() {
  local candidate=$1 path
  [[ $candidate =~ ^[0-9a-f]{40}$ ]]
  /usr/bin/git --no-replace-objects -C "$ROOT" cat-file -e "$candidate^{commit}"
  /usr/bin/git --no-replace-objects -C "$ROOT" merge-base --is-ancestor "$candidate" HEAD
  for path in \
    results/glm52-gates/harness/w7_cache_generation_smoke_v1.sh \
    results/glm52-gates/harness/glm_cgroup_run.sh \
    results/glm52-gates/harness/glm_safe_run.sh \
    scripts/89_score_w7_cache_generation.py \
    scripts/03_memory_guard.py
  do
    [[ -f $ROOT/$path && ! -L $ROOT/$path ]]
    /usr/bin/git --no-replace-objects -C "$ROOT" show "$candidate:$path" | /usr/bin/cmp -s - "$ROOT/$path"
  done
}

verify_sealed_candidate_scripts() {
  local candidate=$1 item tracked descriptor variable digest
  [[ $candidate =~ ^[0-9a-f]{40}$ ]]
  [[ $(/usr/bin/git --no-replace-objects -C "$ROOT" rev-parse HEAD) == "$candidate" ]]
  for item in \
    "results/glm52-gates/harness/w7_cache_generation_smoke_v1.sh:$harness_fd_path:harness_sha256" \
    "results/glm52-gates/harness/glm_cgroup_run.sh:$cgroup_fd_path:cgroup_sha256" \
    "results/glm52-gates/harness/glm_safe_run.sh:$safe_fd_path:safe_sha256" \
    "scripts/89_score_w7_cache_generation.py:$scorer_fd_path:scorer_sha256" \
    "scripts/03_memory_guard.py:$memory_guard_fd_path:memory_guard_sha256"
  do
    IFS=: read -r tracked descriptor variable <<<"$item"
    has_full_seal "$descriptor"
    /usr/bin/git --no-replace-objects -C "$ROOT" show "$candidate:$tracked" | /usr/bin/cmp -s - "$descriptor"
    digest=$(/usr/bin/sha256sum -- "$descriptor" | /usr/bin/awk '{print $1}')
    printf -v "$variable" '%s' "$digest"
  done
}

verify_seal_holder_identity() {
  [[ $seal_holder_pid =~ ^[1-9][0-9]*$ && $seal_holder_start_ticks =~ ^[1-9][0-9]*$ &&
     $seal_holder_parent_pid == "$$" ]]
  /usr/bin/python3 - "$seal_holder_pid" "$seal_holder_start_ticks" "$seal_holder_parent_pid" <<'PY'
import pathlib, sys
pid, expected_start, expected_parent = sys.argv[1:]
stat = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
status = pathlib.Path(f"/proc/{pid}/status").read_text().splitlines()
parent = next(line.split()[1] for line in status if line.startswith("PPid:"))
raise SystemExit(0 if stat[21] == expected_start and parent == expected_parent else 1)
PY
}

seal_runtime_snapshots() {
  local candidate=$1 metadata harness_fd cgroup_fd safe_fd scorer_fd memory_guard_fd live_fd primary_fd
  exec {seal_metadata_fd}< <(
    python3 - "$ROOT" "$candidate" "$LIVE" "$LIVE_SHA256" "$PRIMARY" "$PRIMARY_SHA256" <<'PY'
import ctypes, fcntl, hashlib, os, pathlib, signal, subprocess, sys

root, candidate, live_path, live_sha, primary_path, primary_sha = sys.argv[1:]
libc = ctypes.CDLL(None, use_errno=True)
libc.prctl(1, signal.SIGTERM)
if os.getppid() == 1:
    raise SystemExit("snapshot parent exited")

def seal(name, payload):
    fd = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise SystemExit("short snapshot write")
        view = view[written:]
    os.lseek(fd, 0, os.SEEK_SET)
    seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
    if fcntl.fcntl(fd, fcntl.F_GET_SEALS) != seals:
        raise SystemExit("snapshot sealing failed")
    return fd, hashlib.sha256(payload).hexdigest()

def git_payload(path):
    return subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", "-C", root, "show", f"{candidate}:{path}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    ).stdout

def file_payload(path, expected):
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise SystemExit("request changed while sealing")
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise SystemExit("request digest mismatch")
    return payload

tracked = (
    ("harness", "results/glm52-gates/harness/w7_cache_generation_smoke_v1.sh"),
    ("cgroup", "results/glm52-gates/harness/glm_cgroup_run.sh"),
    ("safe", "results/glm52-gates/harness/glm_safe_run.sh"),
    ("scorer", "scripts/89_score_w7_cache_generation.py"),
    ("memory-guard", "scripts/03_memory_guard.py"),
)
snapshots = [seal(name, git_payload(path)) for name, path in tracked]
snapshots.append(seal("live-request", file_payload(live_path, live_sha)))
snapshots.append(seal("primary-request", file_payload(primary_path, primary_sha)))
start_ticks = pathlib.Path("/proc/self/stat").read_text().split()[21]
fields = [str(os.getpid()), start_ticks, str(os.getppid())]
for fd, digest in snapshots:
    fields.extend((str(fd), digest))
print("\t".join(fields), flush=True)
while True:
    signal.pause()
PY
  )
  IFS=$'\t' read -r seal_holder_pid seal_holder_start_ticks seal_holder_parent_pid \
    harness_fd harness_sha256 cgroup_fd cgroup_sha256 safe_fd safe_sha256 \
    scorer_fd scorer_sha256 memory_guard_fd memory_guard_sha256 \
    live_fd live_digest primary_fd primary_digest <&"$seal_metadata_fd"
  [[ $seal_holder_pid =~ ^[1-9][0-9]*$ && $harness_fd =~ ^[0-9]+$ && $cgroup_fd =~ ^[0-9]+$ &&
     $safe_fd =~ ^[0-9]+$ && $scorer_fd =~ ^[0-9]+$ && $memory_guard_fd =~ ^[0-9]+$ &&
     $live_fd =~ ^[0-9]+$ && $primary_fd =~ ^[0-9]+$ ]]
  [[ $live_digest == "$LIVE_SHA256" && $primary_digest == "$PRIMARY_SHA256" ]]
  harness_fd_path="/proc/$seal_holder_pid/fd/$harness_fd"
  cgroup_fd_path="/proc/$seal_holder_pid/fd/$cgroup_fd"
  safe_fd_path="/proc/$seal_holder_pid/fd/$safe_fd"
  scorer_fd_path="/proc/$seal_holder_pid/fd/$scorer_fd"
  memory_guard_fd_path="/proc/$seal_holder_pid/fd/$memory_guard_fd"
  live_fd_path="/proc/$seal_holder_pid/fd/$live_fd"
  primary_fd_path="/proc/$seal_holder_pid/fd/$primary_fd"
  for descriptor in "$harness_fd_path" "$cgroup_fd_path" "$safe_fd_path" "$scorer_fd_path" "$memory_guard_fd_path" "$live_fd_path" "$primary_fd_path"; do
    [[ -r $descriptor ]]
  done
  verify_seal_holder_identity
}

stop_seal_holder() {
  if [[ ${seal_holder_pid:-} =~ ^[1-9][0-9]*$ ]]; then
    verify_seal_holder_identity || return 1
    kill -TERM "$seal_holder_pid" 2>/dev/null || true
    wait "$seal_holder_pid" 2>/dev/null || true
    seal_holder_pid=
  fi
}

verify_driver_containment() {
  local unit=${GLM_SAFE_CGROUP_UNIT:-} path dir high max swap oom_group
  [[ ${GLM_SAFE_REQUIRE_CGROUP:-} == 1 ]]
  [[ $unit =~ ^glm52-w7-c14-[0-9a-f]{12}-[0-9]+$ ]]
  path=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
  [[ $path == */"$unit.service" ]]
  dir=/sys/fs/cgroup$path
  [[ -r $dir/memory.high && -r $dir/memory.max && -r $dir/memory.swap.max && -r $dir/memory.oom.group ]]
  read -r high <"$dir/memory.high"
  read -r max <"$dir/memory.max"
  read -r swap <"$dir/memory.swap.max"
  read -r oom_group <"$dir/memory.oom.group"
  [[ $high == 83751862272 && $max == 85899345920 && $swap == 0 && $oom_group == 1 ]]
}

verify_driver_safe_lineage() {
  local safe_pid=${DS4_W7_SAFE_PID:-} safe_start=${DS4_W7_SAFE_START_TICKS:-}
  local safe_path=${DS4_W7_SAFE_SCRIPT_PATH:-} safe_unit=${DS4_W7_SAFE_CGROUP_UNIT:-}
  local lock_pid=${DS4_W7_LOCK_PARENT_PID:-} lock_start=${DS4_W7_LOCK_PARENT_START_TICKS:-}
  local lock_fd=${DS4_W7_LOCK_FD:-}
  [[ ${GLM_SAFE_W7_DRIVER_LINEAGE:-} == 1 ]] || return 1
  [[ $safe_pid =~ ^[1-9][0-9]*$ && $safe_start =~ ^[1-9][0-9]*$ ]] || return 1
  [[ $safe_path == "$safe_fd_path" && $safe_unit == "${GLM_SAFE_CGROUP_UNIT:-}" ]] || return 1
  [[ $lock_pid =~ ^[1-9][0-9]*$ && $lock_start =~ ^[1-9][0-9]*$ ]] || return 1
  [[ $lock_fd =~ ^[3-9][0-9]*$ ]] || return 1
  has_full_seal "$safe_path" || return 1
  /usr/bin/git --no-replace-objects -C "$ROOT" show \
    "$candidate:results/glm52-gates/harness/glm_safe_run.sh" | /usr/bin/cmp -s - "$safe_path" || return 1
  /usr/bin/python3 - "$safe_pid" "$safe_start" "$safe_path" "$$" "$lock_pid" "$lock_start" "$lock_fd" <<'PY'
import errno, fcntl, os, pathlib, sys

safe_pid, expected_start, expected_path, child_pid, lock_pid, lock_start, lock_fd_text = sys.argv[1:]
stat = pathlib.Path(f"/proc/{safe_pid}/stat").read_text().split()
if stat[21] != expected_start:
    raise SystemExit("safe wrapper start identity changed")
cmdline = pathlib.Path(f"/proc/{safe_pid}/cmdline").read_bytes().split(b"\0")
if cmdline[:2] != [b"/usr/bin/bash", expected_path.encode()]:
    raise SystemExit("safe wrapper command identity mismatch")
current = child_pid
seen = set()
while current not in seen and current != "1":
    seen.add(current)
    if current == safe_pid:
        break
    status = pathlib.Path(f"/proc/{current}/status").read_text().splitlines()
    current = next(line.split()[1] for line in status if line.startswith("PPid:"))
else:
    raise SystemExit("safe wrapper is not a driver ancestor")
safe_status = pathlib.Path(f"/proc/{safe_pid}/status").read_text().splitlines()
safe_parent = next(line.split()[1] for line in safe_status if line.startswith("PPid:"))
lock_stat = pathlib.Path(f"/proc/{lock_pid}/stat").read_text().split()
if safe_parent != lock_pid or lock_stat[21] != lock_start:
    raise SystemExit("inference-lock parent identity mismatch")
safe_argv = pathlib.Path(f"/proc/{safe_pid}/cmdline").read_bytes().split(b"\0")[:-1]
lock_argv = pathlib.Path(f"/proc/{lock_pid}/cmdline").read_bytes().split(b"\0")[:-1]
lock_path = "/run/lock/frontier-at-home/inference.lock"
if lock_argv != [b"/usr/bin/flock", b"-n", b"-E", b"75", lock_path.encode()] + safe_argv:
    raise SystemExit("inference-lock parent command mismatch")
lock = os.stat(lock_path)
identity = f"{os.major(lock.st_dev):02x}:{os.minor(lock.st_dev):02x}:{lock.st_ino}"
if not any(
    len(fields := line.split()) >= 8
    and fields[1:4] == ["FLOCK", "ADVISORY", "WRITE"]
    and fields[4] == lock_pid
    and fields[5].lower() == identity.lower()
    for line in pathlib.Path("/proc/locks").read_text().splitlines()
):
    raise SystemExit("inference-lock ownership is absent")
lock_fd = int(lock_fd_text)
inherited = os.fstat(lock_fd)
if (inherited.st_dev, inherited.st_ino) != (lock.st_dev, lock.st_ino):
    raise SystemExit("inherited inference-lock descriptor identity mismatch")
fdinfo_lines = pathlib.Path(f"/proc/self/fdinfo/{lock_fd}").read_text().splitlines()
if not any(
    len(fields := line.split()) >= 9
    and fields[0] == "lock:"
    and fields[2:5] == ["FLOCK", "ADVISORY", "WRITE"]
    and fields[6].lower() == identity.lower()
    for line in fdinfo_lines
):
    raise SystemExit("inherited inference-lock descriptor does not own the expected lock")
probe = os.open(lock_path, os.O_RDONLY | os.O_CLOEXEC)
try:
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EAGAIN):
            raise
    else:
        raise SystemExit("inherited inference-lock descriptor is no longer held")
finally:
    os.close(probe)
PY
}

sync_parent() {
  sync -d "$1" 2>/dev/null || sync
  sync -f "$(dirname -- "$1")" 2>/dev/null || sync
}

write_child_exit() {
  local out=$1 exit_status=$2
  python3 - "$out/child-exit.json" "$exit_status" <<'PY'
import json, os, pathlib, sys
p = pathlib.Path(sys.argv[1])
data = {"shutdown_requested": True, "forced_kill": False, "exit_status": int(sys.argv[2])}
fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(data, f, sort_keys=True, separators=(",", ":")); f.write("\n"); f.flush(); os.fsync(f.fileno())
dfd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(dfd); os.close(dfd)
PY
}

stop_server_gracefully() {
  local out=$1 rc
  [[ ${server_pid:-} =~ ^[0-9]+$ ]] || return 1
  kill -TERM "$server_pid" 2>/dev/null || return 1
  for _ in $(seq 1 300); do
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 "$server_pid" 2>/dev/null && return 1
  set +e
  wait "$server_pid"
  rc=$?
  set -e
  server_pid=
  write_child_exit "$out" "$rc"
  [[ $rc == 0 ]]
}

cleanup_driver() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ ${server_pid:-} =~ ^[0-9]+$ ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 300); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 0.1
    done
    wait "$server_pid" 2>/dev/null || true
  fi
  exit "$rc"
}

run_driver() {
  [[ $# == 1 ]]
  local out=$1 code= model_hash before after model_fd_path
  verify_driver_containment
  [[ $out =~ ^/home/bmarti44/\.local/state/glm52-w7-cache-generation/attempt-[0-9a-f]{32}/on$ ]]
  [[ -d $out && ! -L $out && -z $(find "$out" -mindepth 1 -maxdepth 1 -print -quit) ]]
  [[ ${DS4_CUDA_STABLE_MODEL_REMAP:-} == 1 ]]
  verify_dependencies_fast
  mkdir "$out/kv"

  # Keep one verified model description open while the child opens the same
  # inode via procfs; a path replacement cannot change the bytes served.
  exec {model_fd}<"$MODEL"
  model_fd_path="/proc/$$/fd/$model_fd"
  before=$(stat -Lc '%d:%i:%s' -- "$model_fd_path")
  [[ ${before##*:} == "$MODEL_BYTES" ]]
  model_hash=$(sha256sum -- "$model_fd_path" | awk '{print $1}')
  [[ $model_hash == "$MODEL_SHA256" ]]
  python3 - "$out/model.identity.json" "$before" "$model_hash" "$model_fd_path" <<'PY'
import json, os, pathlib, sys
p = pathlib.Path(sys.argv[1]); dev, ino, size = sys.argv[2].split(":")
obj = {"bytes": int(size), "device": int(dev), "inode": int(ino), "sha256": sys.argv[3], "executed_path": sys.argv[4]}
fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(obj, f, sort_keys=True, separators=(",", ":")); f.write("\n"); f.flush(); os.fsync(f.fileno())
PY

  "$BIN" --cuda -m "$model_fd_path" -c 8192 --host 127.0.0.1 --port "$PORT" \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    --kv-disk-dir "$out/kv" --kv-disk-space-mb 4096 \
    --kv-cache-boundary-align-tokens 4 --kv-cache-boundary-trim-tokens 8 \
    >"$out/server.log" 2>&1 &
  server_pid=$!
  trap cleanup_driver EXIT INT TERM HUP

  for _ in $(seq 1 600); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 \
      "http://127.0.0.1:$PORT/v1/models" || true)
    [[ $code == 200 ]] && break
    sleep 1
  done
  [[ $code == 200 ]]

  curl -sS --fail-with-body --max-time 900 -H 'Content-Type: application/json' \
    -o "$out/live-response.json" -w '%{http_code}\n' -d @"$live_fd_path" \
    "http://127.0.0.1:$PORT/v1/completions" >"$out/live-http-status"
  curl -sS --fail-with-body --max-time 1200 -H 'Content-Type: application/json' \
    -o "$out/primary-response.json" -w '%{http_code}\n' -d @"$primary_fd_path" \
    "http://127.0.0.1:$PORT/v1/completions" >"$out/primary-http-status"

  stop_server_gracefully "$out"
  trap - EXIT INT TERM HUP
  after=$(stat -Lc '%d:%i:%s' -- "$model_fd_path")
  [[ $after == "$before" ]]
  sync -f "$out"
}

publish_outer_evidence() {
  local attempt=$1 out=$2 containment_rc=$3 containment_stdout=$4 crash_dir=$5 candidate=$6 score_rc execution_head
  execution_head=$(/usr/bin/git --no-replace-objects -C "$ROOT" rev-parse HEAD)
  set +e
  python3 - "$scorer_fd_path" "$attempt" "$out" "$crash_dir" "$candidate" "$execution_head" \
    "$BINARY_SHA256" "$MODEL_SHA256" "$MODEL_BYTES" "$LIVE_SHA256" "$PRIMARY_SHA256" "${environment_sha256:-unknown}" \
    "$scorer_sha256" "$harness_sha256" "$cgroup_sha256" "$safe_sha256" \
    "$memory_guard_sha256" "$containment_rc" "$containment_stdout" <<'PY'
import importlib.machinery, importlib.util, pathlib, sys
scorer_path=pathlib.Path(sys.argv[1]); attempt=pathlib.Path(sys.argv[2]); out=pathlib.Path(sys.argv[3]); crash=pathlib.Path(sys.argv[4])
candidate, execution_head, binary, model, model_bytes, live, primary, environment, scorer_sha, harness_sha, cgroup_sha, safe_sha, memory_guard_sha, containment_rc_text, containment_stdout=sys.argv[5:]
loader=importlib.machinery.SourceFileLoader("w7_scorer",str(scorer_path)); spec=importlib.util.spec_from_loader(loader.name,loader); module=importlib.util.module_from_spec(spec); loader.exec_module(module)
identities={"candidate_hash":candidate,"execution_head":execution_head,"binary_sha256":binary,
 "model_sha256":model,"model_bytes":int(model_bytes),"live_request_sha256":live,
 "primary_request_sha256":primary,"executed_environment_sha256":environment,
 "scorer_sha256":scorer_sha,"harness_sha256":harness_sha,"cgroup_sha256":cgroup_sha,
 "safe_run_sha256":safe_sha,"memory_guard_sha256":memory_guard_sha,"containment":{"memory_high_gib":78,"memory_max_gib":80,
 "kill_floor_gib":24,"minimum_start_gib":110,"timeout_seconds":2400,"swap_max":0}}
containment_rc=int(containment_rc_text)
result=module.score_and_publish_bound_attempt(attempt=attempt,out=out,crash_dir=crash,evidence_dir=out/"evidence",identities=identities,containment_stdout=containment_stdout,containment_rc=containment_rc)
raise SystemExit(0 if result["verdict"] == "PASS" else 1)
PY
  score_rc=$?
  set -e
  [[ -d $out/evidence ]] && outer_finalized=1
  [[ $containment_rc == 0 && $score_rc == 0 ]]
}

publish_failure_triplet() {
  observed_final_head=$(/usr/bin/git --no-replace-objects -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)
  /usr/bin/python3 - "$attempt" "$out" "${candidate:-unknown}" "$observed_final_head" "$failure_reason" "$1" \
    "$BINARY_SHA256" "$MODEL_SHA256" "$MODEL_BYTES" "$LIVE_SHA256" "$PRIMARY_SHA256" "$environment_sha256" \
    "$scorer_sha256" "$harness_sha256" "$cgroup_sha256" "$safe_sha256" \
    "$memory_guard_sha256" "$containment_rc" "$containment_stdout" <<'PY'
import ctypes, errno, hashlib, json, os, pathlib, sys, tempfile
attempt, out = map(pathlib.Path, sys.argv[1:3]); candidate, execution_head, reason=sys.argv[3:6]; rc=int(sys.argv[6])
binary, model, model_bytes, live, primary, environment, scorer_sha, harness_sha, cgroup_sha, safe_sha, memory_guard_sha=sys.argv[7:18]
containment_rc, containment_stdout = sys.argv[18:20]

def read_once(path):
    try:
        fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0))
    except FileNotFoundError:
        return b""
    try:
        before=os.fstat(fd); chunks=[]
        while True:
            chunk=os.read(fd,1024*1024)
            if not chunk: break
            chunks.append(chunk)
        after=os.fstat(fd)
    finally:
        os.close(fd)
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise SystemExit("failure artifact changed while reading")
    return b"".join(chunks)

payloads={
 "containment.stdout":containment_stdout.encode(),
 "containment.stderr":read_once(attempt/"containment.stderr"),
 "containment.rc":(containment_rc+"\n").encode(),
}
rows=[{"source":name,"text":payload.decode("utf-8",errors="strict")} for name,payload in payloads.items()]
summary={"checks":{"runtime_completed":False,"evidence_finalizer_completed":True},"observed":{"failure_reason":reason,"outer_exit_status":rc},"verdict":"FAIL"}
raw_bytes=b"".join((json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode() for row in rows)
summary_bytes=(json.dumps(summary,sort_keys=True,separators=(",",":"))+"\n").encode()
artifacts={name:hashlib.sha256(payload).hexdigest() for name,payload in payloads.items()}
artifacts.update({"raw.jsonl":hashlib.sha256(raw_bytes).hexdigest(),"summary.json":hashlib.sha256(summary_bytes).hexdigest()})
manifest={"schema":"glm52-w7-runtime-v3","candidate_hash":candidate,"execution_head":execution_head,
 "binary_sha256":binary,"model_sha256":model,"model_bytes":int(model_bytes),
 "live_request_sha256":live,"primary_request_sha256":primary,"executed_environment_sha256":environment,
 "scorer_sha256":scorer_sha,"harness_sha256":harness_sha,"cgroup_sha256":cgroup_sha,"safe_run_sha256":safe_sha,"memory_guard_sha256":memory_guard_sha,
 "containment":{"memory_high_gib":78,"memory_max_gib":80,"kill_floor_gib":24,"minimum_start_gib":110,"timeout_seconds":2400,"swap_max":0},
 "purpose":"failed W7 production-path diagnostic","artifacts":artifacts}
manifest_bytes=(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n").encode()
destination=out/"evidence"
if destination.exists(): raise FileExistsError(destination)
temporary=pathlib.Path(tempfile.mkdtemp(prefix=".w7-failure-",dir=out))
try:
    for name,payload in (("raw.jsonl",raw_bytes),("summary.json",summary_bytes),("manifest.json",manifest_bytes)):
        fd=os.open(temporary/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
        with os.fdopen(fd,"wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    directory_fd=os.open(temporary,os.O_RDONLY|os.O_DIRECTORY); os.fsync(directory_fd); os.close(directory_fd)
    libc=ctypes.CDLL(None,use_errno=True); renameat2=getattr(libc,"renameat2",None)
    if renameat2 is None: raise OSError(errno.ENOSYS,"renameat2 is required")
    renameat2.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    renameat2.restype=ctypes.c_int
    if renameat2(-100,os.fsencode(temporary),-100,os.fsencode(destination),1)!=0:
        error=ctypes.get_errno(); raise OSError(error,os.strerror(error),destination)
    parent_fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY); os.fsync(parent_fd); os.close(parent_fd)
except BaseException:
    if temporary.exists():
        for child in temporary.glob("*"): child.unlink(missing_ok=True)
        temporary.rmdir()
    raise
PY
}

finalize_outer() {
  local original_rc=$? final_rc
  trap - EXIT
  [[ $original_rc != 0 ]] || original_rc=1
  if [[ ${outer_finalized:-0} == 0 && -n ${attempt:-} && -n ${out:-} && -d $attempt && ! -L $attempt ]]; then
    set +e
    [[ -e $out ]] || mkdir -m 0700 "$out"
    if [[ ! -d $out || -L $out ]]; then
      exit 125
    fi
    containment_stdout="$containment_stdout" containment_rc="$containment_rc" publish_failure_triplet "$original_rc"
    final_rc=$?
    set -e
    [[ $final_rc == 0 ]] || original_rc=125
  fi
  stop_engine_lock_holder || true
  stop_seal_holder || true
  exit "$original_rc"
}

if [[ ${1:-} == --self-test ]]; then
  [[ $# == 1 ]]
  verify_dependencies_fast
  echo W7_CACHE_GENERATION_SMOKE_SELFTEST_OK
  exit 0
fi

if [[ ${1:-} == --engine-lock-self-test ]]; then
  [[ $# == 1 ]]
  test_directory=$(mktemp -d /home/bmarti44/.local/state/.w7-engine-lock-test.XXXXXX)
  trap 'rm -f -- "$test_directory/.ds4-engine-lock"; rmdir -- "$test_directory"' EXIT
  chmod 0700 "$test_directory"
  ln -s /tmp/ds4.lock "$test_directory/.ds4-engine-lock"
  if prepare_engine_lock "$test_directory"; then
    echo "precreated engine-lock symlink was accepted" >&2
    exit 1
  fi
  stop_engine_lock_holder || true
  rm -f -- "$test_directory/.ds4-engine-lock"
  echo W7_ENGINE_LOCK_PRECREATE_REJECTED
  prepare_engine_lock "$test_directory"
  expected_identity=$(stat -Lc '%d:%i' -- "$engine_lock_fd_path")
  ln -s /tmp/ds4.lock "$test_directory/.ds4-engine-lock"
  /usr/bin/python3 - "$engine_lock_fd_path" "$expected_identity" <<'PY'
import os, sys
path, expected = sys.argv[1:]
descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
try:
    metadata = os.fstat(descriptor)
    actual = f"{metadata.st_dev}:{metadata.st_ino}"
    if actual != expected:
        raise SystemExit("descriptor path was redirected by leaf replacement")
finally:
    os.close(descriptor)
PY
  stop_engine_lock_holder
  echo W7_ENGINE_LOCK_DESCRIPTOR_SELFTEST_OK
  exit 0
fi

if [[ ${1:-} == --sealed-self-test ]]; then
  [[ $# == 1 ]]
  candidate=$(/usr/bin/git --no-replace-objects -C "$ROOT" rev-parse HEAD)
  seal_runtime_snapshots "$candidate"
  python3 - "$harness_fd_path" "$cgroup_fd_path" "$safe_fd_path" "$scorer_fd_path" "$memory_guard_fd_path" "$live_fd_path" "$primary_fd_path" <<'PY'
import errno, fcntl, os, sys
required = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
for path in sys.argv[1:]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != required:
            raise SystemExit("snapshot is not fully sealed")
    finally:
        os.close(descriptor)
    writable = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    try:
        try:
            os.write(writable, b"mutation")
        except OSError as error:
            if error.errno not in (errno.EPERM, errno.EACCES):
                raise
        else:
            raise SystemExit("sealed snapshot accepted a write")
    finally:
        os.close(writable)
PY
  stop_seal_holder
  echo W7_SEALED_SNAPSHOTS_SELFTEST_OK
  exit 0
fi

if [[ ${1:-} == --driver-lineage-self-test ]]; then
  [[ $# == 1 ]]
  [[ $candidate =~ ^[0-9a-f]{40}$ && $(/usr/bin/git --no-replace-objects -C "$ROOT" rev-parse HEAD) == "$candidate" ]]
  /usr/bin/git --no-replace-objects -C "$ROOT" show "$candidate:results/glm52-gates/harness/w7_cache_generation_smoke_v1.sh" | /usr/bin/cmp -s - "$0"
  verify_driver_safe_lineage
  original_safe_pid=$DS4_W7_SAFE_PID
  original_safe_start=$DS4_W7_SAFE_START_TICKS
  original_safe_path=$DS4_W7_SAFE_SCRIPT_PATH
  original_safe_unit=$DS4_W7_SAFE_CGROUP_UNIT
  original_lock_pid=$DS4_W7_LOCK_PARENT_PID
  original_lock_start=$DS4_W7_LOCK_PARENT_START_TICKS
  original_lock_fd=$DS4_W7_LOCK_FD
  expect_lineage_rejection() {
    local name=$1
    if verify_driver_safe_lineage; then
      echo "lineage mutation was accepted name=$name" >&2
      return 1
    fi
    echo "W7_LINEAGE_MUTATION_REJECTED name=$name"
  }
  DS4_W7_SAFE_PID=1; expect_lineage_rejection bad-safe-pid
  DS4_W7_SAFE_PID=$original_safe_pid; DS4_W7_SAFE_START_TICKS=1; expect_lineage_rejection bad-safe-start
  DS4_W7_SAFE_START_TICKS=$original_safe_start; DS4_W7_SAFE_SCRIPT_PATH=$harness_fd_path; expect_lineage_rejection wrong-safe-script
  DS4_W7_SAFE_SCRIPT_PATH=$original_safe_path; DS4_W7_SAFE_CGROUP_UNIT=wrong; expect_lineage_rejection wrong-cgroup-unit
  DS4_W7_SAFE_CGROUP_UNIT=$original_safe_unit; DS4_W7_LOCK_PARENT_PID=1; expect_lineage_rejection bad-lock-pid
  DS4_W7_LOCK_PARENT_PID=$original_lock_pid; DS4_W7_LOCK_PARENT_START_TICKS=1; expect_lineage_rejection bad-lock-start
  DS4_W7_LOCK_PARENT_START_TICKS=$original_lock_start
  DS4_W7_LOCK_FD=9999; expect_lineage_rejection bad-lock-fd
  exec {unowned_lock_fd}<>/run/lock/frontier-at-home/inference.lock
  DS4_W7_LOCK_FD=$unowned_lock_fd; expect_lineage_rejection same-inode-unowned-lock-fd
  exec {unowned_lock_fd}>&-
  DS4_W7_LOCK_FD=$original_lock_fd
  echo W7_DRIVER_LINEAGE_SELFTEST_OK
  sleep 1
  exit 0
fi

if [[ ${1:-} == --driver ]]; then
  [[ $# == 3 && $2 == on ]]
  [[ $candidate =~ ^[0-9a-f]{40}$ && $(/usr/bin/git --no-replace-objects -C "$ROOT" rev-parse HEAD) == "$candidate" ]]
  /usr/bin/git --no-replace-objects -C "$ROOT" show "$candidate:results/glm52-gates/harness/w7_cache_generation_smoke_v1.sh" | /usr/bin/cmp -s - "$0"
  verify_driver_safe_lineage
  run_driver "$3"
  exit
fi

if [[ ${1:-} == --candidate || ${1:-} == --holder-loss-self-test || ${1:-} == --lineage-self-test ]]; then
  [[ $# == 2 && $2 =~ ^[0-9a-f]{40}$ ]]
  initial_mode=$1
  candidate=$2
  verify_dependencies_fast
  verify_reviewed_sources "$candidate"
  seal_runtime_snapshots "$candidate"
  exec /usr/bin/env \
    DS4_W7_SEAL_HOLDER_PID="$seal_holder_pid" \
    DS4_W7_SEAL_HOLDER_START_TICKS="$seal_holder_start_ticks" \
    DS4_W7_SEAL_HOLDER_PARENT_PID="$seal_holder_parent_pid" \
    DS4_W7_CANDIDATE_HASH="$candidate" \
    DS4_W7_SEALED_HARNESS_PATH="$harness_fd_path" \
    DS4_W7_SEALED_CGROUP_PATH="$cgroup_fd_path" \
    DS4_W7_SEALED_SAFE_PATH="$safe_fd_path" \
    DS4_W7_SEALED_SCORER_PATH="$scorer_fd_path" \
    DS4_W7_SEALED_LIVE_PATH="$live_fd_path" \
    DS4_W7_SEALED_PRIMARY_PATH="$primary_fd_path" \
    DS4_W7_PINNED_HARNESS_SHA256="$harness_sha256" \
    DS4_W7_SEALED_CGROUP_SHA256="$cgroup_sha256" \
    DS4_W7_SEALED_SAFE_SHA256="$safe_sha256" \
    DS4_W7_SEALED_SCORER_SHA256="$scorer_sha256" \
    DS4_W7_SEALED_MEMORY_GUARD_PATH="$memory_guard_fd_path" \
    DS4_W7_SEALED_MEMORY_GUARD_SHA256="$memory_guard_sha256" \
    /usr/bin/bash "$harness_fd_path" \
      "$([[ $initial_mode == --candidate ]] && printf %s --sealed-outer || { [[ $initial_mode == --holder-loss-self-test ]] && printf %s --sealed-holder-loss-test || printf %s --sealed-lineage-self-test; })" \
      "$candidate"
fi

[[ $# == 2 && $1 =~ ^(--sealed-outer|--sealed-holder-loss-test|--sealed-lineage-self-test)$ && $2 =~ ^[0-9a-f]{40}$ ]] || exit 2
sealed_mode=$1
candidate=$2
[[ $0 == "$harness_fd_path" && $seal_holder_pid =~ ^[1-9][0-9]*$ ]]
for descriptor in "$harness_fd_path" "$cgroup_fd_path" "$safe_fd_path" "$scorer_fd_path" "$memory_guard_fd_path" "$live_fd_path" "$primary_fd_path"; do
  [[ $descriptor =~ ^/proc/$seal_holder_pid/fd/[0-9]+$ && -r $descriptor ]]
done
verify_seal_holder_identity
verify_sealed_candidate_scripts "$candidate"
verify_dependencies_fast
[[ $(id -u) != 0 && $(id -un) == bmarti44 ]]
if [[ $sealed_mode == --sealed-lineage-self-test ]]; then
  nonce=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
  tag="w7-c14-${nonce:0:12}"
  GLM_SAFE_KILL_FLOOR_GIB=24 \
  GLM_SAFE_MIN_START_GIB=110 \
  GLM_SAFE_TIMEOUT_S=120 \
  GLM_SAFE_RUN_AS_CURRENT_USER=1 \
  GLM_SAFE_PINNED_SAFE_PATH="$safe_fd_path" \
  GLM_SAFE_PINNED_SAFE_SHA256="$safe_sha256" \
  GLM_SAFE_W7_DRIVER_LINEAGE=1 \
  GLM_SAFE_MEMORY_GUARD_PATH="$memory_guard_fd_path" \
  GLM_SAFE_EXPECTED_MEMORY_GUARD_SHA256="$memory_guard_sha256" \
  DS4_W7_PINNED_HARNESS_SHA256="$harness_sha256" \
  DS4_W7_CANDIDATE_HASH="$candidate" \
  DS4_W7_SEALED_SAFE_PATH="$safe_fd_path" \
  DS4_W7_SEALED_SAFE_SHA256="$safe_sha256" \
  /usr/bin/bash "$cgroup_fd_path" --tag "$tag" -- /usr/bin/bash "$harness_fd_path" --driver-lineage-self-test
  stop_seal_holder
  echo W7_LINEAGE_CONTAINMENT_SELFTEST_OK
  exit 0
fi
base=/home/bmarti44/.local/state/glm52-w7-cache-generation
mkdir -p "$base"
nonce=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
attempt="$base/attempt-$nonce"
out="$attempt/on"
mkdir -m 0700 "$attempt"
trap finalize_outer EXIT
mkdir -m 0700 "$out"
prepare_engine_lock "$out"
failure_reason=containment-launch
tag="w7-c14-${nonce:0:12}"
final_artifacts="$out/server.log,$out/live-response.json,$out/live-http-status,$out/primary-response.json,$out/primary-http-status,$out/child-exit.json,$out/model.identity.json"

if [[ $sealed_mode == --sealed-holder-loss-test ]]; then
  failure_reason=seal-holder-loss-self-test
  stop_seal_holder
  false
fi

set +e
containment_stdout=$(
GLM_SAFE_MEMORY_HIGH_GIB=78 \
GLM_SAFE_KILL_FLOOR_GIB=24 \
GLM_SAFE_MIN_START_GIB=110 \
GLM_SAFE_TIMEOUT_S=2400 \
GLM_SAFE_RUN_AS_CURRENT_USER=1 \
GLM_SAFE_LOG_CANDIDATE_PROVENANCE=1 \
GLM_SAFE_EXPECTED_BINARY_SHA256="$BINARY_SHA256" \
GLM_SAFE_PROVENANCE_ENV_ALLOWLIST=DS4_CUDA_STABLE_MODEL_REMAP,DS4_LOCK_FILE \
GLM_SAFE_EXPECTED_ENV_SHA256="$environment_sha256" \
GLM_SAFE_FINAL_ARTIFACTS="$final_artifacts" \
GLM_SAFE_DONE_DIGESTS=1 \
GLM_SAFE_PINNED_SAFE_PATH="$safe_fd_path" \
GLM_SAFE_PINNED_SAFE_SHA256="$safe_sha256" \
GLM_SAFE_W7_DRIVER_LINEAGE=1 \
GLM_SAFE_MEMORY_GUARD_PATH="$memory_guard_fd_path" \
GLM_SAFE_EXPECTED_MEMORY_GUARD_SHA256="$memory_guard_sha256" \
GLM_CANDIDATE_SRC="$CANDIDATE_SRC" \
DS4_W7_PINNED_HARNESS_SHA256="$harness_sha256" \
DS4_W7_CANDIDATE_HASH="$candidate" \
DS4_W7_SEALED_SAFE_PATH="$safe_fd_path" \
DS4_W7_SEALED_SAFE_SHA256="$safe_sha256" \
DS4_W7_SEALED_LIVE_PATH="$live_fd_path" \
DS4_W7_SEALED_PRIMARY_PATH="$primary_fd_path" \
DS4_CUDA_STABLE_MODEL_REMAP=1 \
DS4_LOCK_FILE="$engine_lock_fd_path" \
DS4_CUDA_EXPERT_CACHE_GB=40 \
DS4_CUDA_EXPERT_CACHE_PIN=1 \
DS4_CUDA_EXPERT_CACHE_SLRU=1 \
DS4_CUDA_FETCH_THREADS=6 \
DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
DS4_GLM_SYNC_TRACE=1 \
/usr/bin/bash "$cgroup_fd_path" --tag "$tag" -- /usr/bin/bash "$harness_fd_path" --driver on "$out" \
  2>"$attempt/containment.stderr"
)
containment_rc=$?
set -e
printf '%s\n' "$containment_stdout" >"$attempt/containment.stdout"
printf '%s\n' "$containment_rc" >"$attempt/containment.rc"
sync_parent "$attempt/containment.rc"
failure_reason=containment-result-validation
crash_dir=$(printf '%s\n' "$containment_stdout" | sed -nE 's#^SAFE_RUN_DONE rc=[0-9]+ killed=[a-z]+ dir=(/home/bmarti44/\.local/state/glm52-crashlog/[A-Za-z0-9._-]+) main_sha256=[0-9a-f]{64} samples_sha256=[0-9a-f]{64} kernel_sha256=[0-9a-f]{64}$#\1#p')
[[ $(printf '%s\n' "$crash_dir" | sed '/^$/d' | wc -l) == 1 && -d $crash_dir && ! -L $crash_dir ]]
failure_reason=evidence-binding-and-scoring
publish_outer_evidence "$attempt" "$out" "$containment_rc" "$containment_stdout" "$crash_dir" "$candidate"
stop_engine_lock_holder
stop_seal_holder
trap - EXIT
printf 'W7_CACHE_GENERATION_ATTEMPT=%s\n' "$attempt"
