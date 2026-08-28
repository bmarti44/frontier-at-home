#!/usr/bin/env bash
# Transactional profile switch for the unchanged :8010 auth chain.
set -Eeuo pipefail
umask 077

if [[ -n ${ENGINE_SWITCH_SOURCE_ONLY_FIXTURE_ROOT:-} ]]; then
    [[ ! ${ENGINE_SWITCH_TESTING+x} && ${BASH_SOURCE[0]} != "$0" &&
            $ENGINE_SWITCH_SOURCE_ONLY_FIXTURE_ROOT == /* &&
            -d $ENGINE_SWITCH_SOURCE_ONLY_FIXTURE_ROOT ]] || {
        printf 'invalid engine-switch source-only fixture invocation\n' >&2
        return 2 2>/dev/null || exit 2
    }
    REPO=$ENGINE_SWITCH_SOURCE_ONLY_FIXTURE_ROOT/repo
    PROD_STATE=$ENGINE_SWITCH_SOURCE_ONLY_FIXTURE_ROOT/state
    PROD_LAGUNA_BINARY=$ENGINE_SWITCH_SOURCE_ONLY_FIXTURE_ROOT/artifacts/bin/laguna-server
    PROD_LAGUNA_MODEL=$ENGINE_SWITCH_SOURCE_ONLY_FIXTURE_ROOT/artifacts/models/laguna-00001-of-00003.gguf
    PROD_LAGUNA_DRAFT=$ENGINE_SWITCH_SOURCE_ONLY_FIXTURE_ROOT/artifacts/models/laguna-dflash.gguf
else
    REPO=/home/bmarti44/spark-deepseek-v4-flash
    PROD_STATE=/home/dsv4/ds4-project/engine-switch
    PROD_LAGUNA_BINARY=/home/bmarti44/.cache/llamacpp-laguna-06f8cebd/src/build/bin/llama-server
    PROD_LAGUNA_MODEL=/home/bmarti44/models/laguna-s-2.1/unsloth/UD-Q4_K_XL/Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf
    PROD_LAGUNA_DRAFT=/home/bmarti44/models/laguna-s-2.1/poolside/laguna-s-2.1-DFlash-BF16.gguf
fi
readonly REPO PROD_STATE PROD_LAGUNA_BINARY PROD_LAGUNA_MODEL PROD_LAGUNA_DRAFT
readonly PROD_BINARY=/home/bmarti44/.cache/glm52-dynexp2-patched/ds4-server
readonly PROD_GGUF=/home/bmarti44/models/glm52-full-denseq40.gguf
readonly PROFILE_MANIFEST=$REPO/configs/glm52-fullq4-production-profile.json
readonly QWEN_PROFILE_MANIFEST=$REPO/configs/qwen38-production-profile.json
readonly QWEN_1M_PROFILE_MANIFEST=$REPO/configs/qwen38-1m-production-profile.json
readonly LAGUNA_PROFILE_MANIFEST=$REPO/configs/laguna-production-profile.json
readonly LAGUNA_BUILD_MANIFEST=$REPO/configs/build-manifests/llamacpp-laguna-06f8cebd.json
readonly PROD_QWEN_BINARY=/home/bmarti44/.cache/llamacpp-qwen38-9d77fa17/src/build/bin/llama-server
readonly PROD_QWEN_MODEL=/home/bmarti44/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf
readonly PROD_QWEN_MMPROJ=/home/bmarti44/models/qwen3.8-27b/mmproj-Qwen3.8-27B-f16.gguf
readonly PORT=8013
readonly AUTH_PORT=8010

if [[ ${ENGINE_SWITCH_TESTING:-0} == 1 ]]; then
    [[ -n ${ENGINE_SWITCH_TEST_ROOT:-} ]] || {
        echo "ENGINE_SWITCH_TEST_ROOT is required" >&2; exit 2;
    }
    STATE=$ENGINE_SWITCH_TEST_ROOT
    BINARY=$STATE/source/ds4-server
    GGUF=$STATE/model.gguf
    QWEN_BINARY=$STATE/source/llama-server
    QWEN_MODEL=$STATE/qwen-model.gguf
    QWEN_MMPROJ=$STATE/qwen-mmproj.gguf
    LAGUNA_BINARY=$STATE/source/laguna-server
    LAGUNA_MODEL=$STATE/laguna-model-00001-of-00003.gguf
    LAGUNA_DRAFT=$STATE/laguna-dflash.gguf
else
    unset ENGINE_SWITCH_TEST_ROOT ENGINE_PORT DS4_GLM_TOPK_KEEP \
        DS4_GLM_TOPK_SKIP_LOAD DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL \
        DSV4_ALLOW_RETRY_AFTER_FAILED_START || true
    STATE=$PROD_STATE
    BINARY=$PROD_BINARY
    GGUF=$PROD_GGUF
    QWEN_BINARY=$PROD_QWEN_BINARY
    QWEN_MODEL=$PROD_QWEN_MODEL
    QWEN_MMPROJ=$PROD_QWEN_MMPROJ
    LAGUNA_BINARY=$PROD_LAGUNA_BINARY
    LAGUNA_MODEL=$PROD_LAGUNA_MODEL
    LAGUNA_DRAFT=$PROD_LAGUNA_DRAFT
fi
readonly STATE BINARY GGUF QWEN_BINARY QWEN_MODEL QWEN_MMPROJ \
    LAGUNA_BINARY LAGUNA_MODEL LAGUNA_DRAFT
readonly ACTIVE=$STATE/active.json
readonly LOCK=$STATE/switch.lock
readonly GLM_PROCESS=$STATE/glm52.process.json
readonly GLM_WATCHDOG_TARGET=$STATE/glm52.memwatch.target
readonly GLM_WATCHDOG_READY=$STATE/glm52.memwatch.ready
readonly GLM_WATCHDOG_LOG=$STATE/glm52.memwatch.log
readonly QWEN_PROCESS=$STATE/qwen38.process.json
readonly QWEN_UNIT=qwen38-engine.service
readonly QWEN_WATCHDOG_TARGET=$STATE/qwen38.memwatch.target
readonly QWEN_WATCHDOG_READY=$STATE/qwen38.memwatch.ready
readonly QWEN_WATCHDOG_LOG=$STATE/qwen38.memwatch.log
readonly LAGUNA_PROCESS=$STATE/laguna.process.json
readonly LAGUNA_UNIT=laguna-engine.service
readonly LAGUNA_WATCHDOG_TARGET=$STATE/laguna.memwatch.target
readonly LAGUNA_WATCHDOG_READY=$STATE/laguna.memwatch.ready
readonly LAGUNA_WATCHDOG_LOG=$STATE/laguna.memwatch.log
rollback_needed=false
previous_profile=
qwen_hashes_verified_profile=
qwen_verified_identities=
laguna_hashes_verified=false
laguna_verified_identities=

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Acquire the switch lock on fd 9 with a bounded wait. An unbounded flock
# turned a leaked-fd bug into a silent multi-hour hang (2026-08-21); on
# timeout, name the holder from /proc/locks so the operator can see whether
# the holding PID is alive or the lock is a leaked open-file description
# surviving in a spawned child (see docs/RUNBOOK-stuck-switch.md).
readonly SWITCH_LOCK_TIMEOUT_SECONDS=${SWITCH_LOCK_TIMEOUT_SECONDS:-300}
acquire_switch_lock() {
    exec 9>"$LOCK"
    flock -w "$SWITCH_LOCK_TIMEOUT_SECONDS" -x 9 && return 0
    local inode holder
    inode=$(stat -Lc %i -- "$LOCK" 2>/dev/null || echo '?')
    holder=$(awk -v ino=":$inode " \
        '$0 ~ ino && $0 !~ /->/ {print $5; exit}' /proc/locks 2>/dev/null \
        || true)
    if [[ -n ${holder:-} && -d /proc/$holder ]]; then
        die "switch lock not acquired after ${SWITCH_LOCK_TIMEOUT_SECONDS}s; held by live pid $holder ($(tr '\0' ' ' <"/proc/$holder/cmdline" 2>/dev/null || echo unknown))"
    fi
    die "switch lock not acquired after ${SWITCH_LOCK_TIMEOUT_SECONDS}s; holder pid ${holder:-unknown} is not running — likely a leaked open-file description in a spawned child (memwatch/server); see docs/RUNBOOK-stuck-switch.md"
}

clean_python() {
    env -i PATH=/usr/bin:/bin HOME=/nonexistent LANG=C.UTF-8 \
        /usr/bin/python3 "$@"
}

clean_curl() {
    env -i PATH=/usr/bin:/bin HOME=/nonexistent LANG=C.UTF-8 \
        /usr/bin/curl --disable "$@"
}

# Declarative profile sources (docs/PROFILE-SCHEMA.md). Launch argv, env,
# and containment properties render from these validated profiles; the
# per-alias binary/model/draft paths stay in the readonly variables above so
# the ENGINE_SWITCH_TESTING fixture harness keeps working unchanged.
readonly PROFILE_DSV4=configs/profiles/deepseek-v4-flash/cuda-spark-128g-1m-fast.json
readonly PROFILE_GLM52=configs/profiles/glm-5.2/cuda-spark-128g.json
readonly PROFILE_QWEN38=configs/profiles/qwen3.8-27b/cuda-spark-128g.json
readonly PROFILE_QWEN38_1M=configs/profiles/qwen3.8-27b/cuda-spark-128g-1m.json
readonly PROFILE_LAGUNA=configs/profiles/laguna-s-2.1/cuda-spark-128g.json

profile_path_for() {
    case "$1" in
        dsv4) printf '%s\n' "$PROFILE_DSV4" ;;
        glm52) printf '%s\n' "$PROFILE_GLM52" ;;
        qwen38) printf '%s\n' "$PROFILE_QWEN38" ;;
        qwen38-1m) printf '%s\n' "$PROFILE_QWEN38_1M" ;;
        laguna) printf '%s\n' "$PROFILE_LAGUNA" ;;
        *) die "no profile for alias $1" ;;
    esac
}

# Emit one NUL-delimited field list from a validated profile.
# Fields: args | env | properties | unit | log_name
profile_field() {
    local alias=$1 field=$2 relpath
    relpath=$(profile_path_for "$alias")
    clean_python - "$REPO" "$relpath" "$field" "$alias" <<'PY'
import sys
repo, relpath, field, alias = sys.argv[1:]
sys.path.insert(0, repo + "/scripts/lib")
import profile_resolver
parts = relpath.split("/")
model_slug, name = parts[-2], parts[-1]
profile = profile_resolver.load_profile(model_slug, name)
if profile.get("switch_alias") != alias:
    raise SystemExit(f"profile {relpath} does not carry switch_alias {alias}")
launch = profile["launch"]
out = sys.stdout
if field == "args":
    for token in launch.get("args", []):
        out.write(token + "\0")
elif field == "env":
    for key, value in (launch.get("env") or {}).items():
        out.write(f"{key}={value}\0")
elif field == "properties":
    containment = profile["containment"]
    properties = {
        "Type": "exec",
        "User": launch["user"],
        "MemoryHigh": containment["memory_high"],
        "MemoryMax": containment["memory_max"],
        "MemorySwapMax": containment["memory_swap_max"],
        "OOMPolicy": containment["oom_policy"],
        "KillMode": containment["kill_mode"],
        "Delegate": "no",
    }
    properties.update(containment.get("extra_properties", {}))
    for key, value in properties.items():
        out.write(f"{key}={value}\0")
elif field == "unit":
    out.write(profile["containment"]["unit"] + "\0")
elif field == "log_name":
    out.write(launch["log_name"] + "\0")
else:
    raise SystemExit(f"unknown field {field}")
PY
}

# Fill an array from one NUL-delimited profile field, failing closed on any
# render error (a process substitution would swallow the exit status).
read_profile_array() {
    local -n out_ref=$1
    local alias=$2 field=$3 buffer
    buffer=$(mktemp) || die 'cannot create profile render buffer'
    if ! profile_field "$alias" "$field" >"$buffer"; then
        rm -f -- "$buffer"
        die "profile render failed for $alias $field"
    fi
    out_ref=()
    local token
    while IFS= read -r -d '' token; do
        out_ref+=("$token")
    done <"$buffer"
    rm -f -- "$buffer"
}

# Substitute launch placeholders in an array in place. Any placeholder left
# unresolved fails closed; JSON-literal tokens (chat-template kwargs) pass
# through untouched because they never match a bare {placeholder} token.
subst_placeholders() {
    local -n tokens_ref=$1
    shift
    local -a pairs=("$@")
    local index token pair key
    for index in "${!tokens_ref[@]}"; do
        token=${tokens_ref[$index]}
        for pair in "${pairs[@]}"; do
            key=${pair%%=*}
            token=${token//"{$key}"/${pair#*=}}
        done
        if [[ $token =~ ^\{[a-z_]+\}$ ]]; then
            die "unresolved launch placeholder $token"
        fi
        tokens_ref[$index]=$token
    done
}

launch_systemd_profile() {
    local alias=$1 binary=$2 model=$3 mmproj=$4 draft=$5
    local -a argv props property_args subs
    local unit log_name pair
    read_profile_array argv "$alias" args
    (( ${#argv[@]} > 0 )) || die "$alias profile rendered an empty argv"
    subs=("model=$model" "port=$PORT")
    [[ -z $mmproj ]] || subs+=("mmproj=$mmproj")
    [[ -z $draft ]] || subs+=("draft_model=$draft")
    subst_placeholders argv "${subs[@]}"
    read_profile_array props "$alias" properties
    (( ${#props[@]} > 0 )) || die "$alias profile rendered no containment properties"
    unit=$(profile_field "$alias" unit | tr -d '\0') || die "$alias unit render failed"
    log_name=$(profile_field "$alias" log_name | tr -d '\0') || die "$alias log render failed"
    property_args=()
    for pair in "${props[@]}"; do
        property_args+=(--property "$pair")
    done
    systemd-run --unit="$unit" --collect --quiet \
        "${property_args[@]}" \
        --property "StandardOutput=append:$STATE/$log_name.server.log" \
        --property "StandardError=append:$STATE/$log_name.server.log" \
        /usr/bin/flock --nonblock --no-fork \
        /run/lock/frontier-at-home/inference.lock \
        "$binary" "${argv[@]}"
}

render_snapshot() {
    local alias=$1
    case "$alias" in
        dsv4)
            local -a env_pairs
            read_profile_array env_pairs dsv4 env
            subst_placeholders env_pairs "port=$PORT" "repo=$REPO"
            clean_python - dsv4 "$REPO/scripts/21_serve_llamacpp.sh" \
                "${env_pairs[@]}" <<'PY'
import json, sys
alias, delegate, *pairs = sys.argv[1:]
print(json.dumps({"alias": alias, "mechanism": "delegated-launcher",
                  "runuser": "dsv4", "delegate": delegate,
                  "env": dict(pair.split("=", 1) for pair in pairs)},
                 indent=1))
PY
            ;;
        glm52)
            local -a env_pairs argv
            read_profile_array env_pairs glm52 env
            read_profile_array argv glm52 args
            subst_placeholders argv "model=$GGUF" "port=$PORT"
            clean_python - glm52 "$BINARY" "${#env_pairs[@]}" \
                "${env_pairs[@]}" "${argv[@]}" <<'PY'
import json, sys
alias, binary, count, *rest = sys.argv[1:]
count = int(count)
print(json.dumps({"alias": alias, "mechanism": "setsid-memwatch",
                  "binary": binary,
                  "env": dict(pair.split("=", 1) for pair in rest[:count]),
                  "argv": rest[count:]}, indent=1))
PY
            ;;
        qwen38|qwen38-1m|laguna)
            local binary model mmproj draft unit log_name
            case "$alias" in
                laguna)
                    binary=$LAGUNA_BINARY model=$LAGUNA_MODEL
                    mmproj= draft=$LAGUNA_DRAFT
                    ;;
                *)
                    binary=$QWEN_BINARY model=$QWEN_MODEL
                    mmproj=$QWEN_MMPROJ draft=
                    ;;
            esac
            local -a argv props subs
            read_profile_array argv "$alias" args
            subs=("model=$model" "port=$PORT")
            [[ -z $mmproj ]] || subs+=("mmproj=$mmproj")
            [[ -z $draft ]] || subs+=("draft_model=$draft")
            subst_placeholders argv "${subs[@]}"
            read_profile_array props "$alias" properties
            unit=$(profile_field "$alias" unit | tr -d '\0')
            log_name=$(profile_field "$alias" log_name | tr -d '\0')
            clean_python - "$alias" "$binary" "$unit" \
                "$STATE/$log_name.server.log" "${#props[@]}" \
                "${props[@]}" "${argv[@]}" <<'PY'
import json, sys
alias, binary, unit, server_log, count, *rest = sys.argv[1:]
count = int(count)
print(json.dumps({"alias": alias, "mechanism": "systemd-run",
                  "binary": binary, "env": {}, "argv": rest[count:],
                  "systemd": {"unit": unit,
                              "properties": dict(
                                  pair.split("=", 1) for pair in rest[:count]),
                              "server_log": server_log,
                              "flock": "/run/lock/frontier-at-home/inference.lock"}},
                 indent=1))
PY
            ;;
        *) die "no profile for alias $alias" ;;
    esac
}

dsv4_launcher() {
    # The tuned env map renders from PROFILE_DSV4 (the 1M+fast profile);
    # scripts/tests/test_profile_conformance.py pins its values.
    local -a launcher_env
    read_profile_array launcher_env dsv4 env
    (( ${#launcher_env[@]} > 0 )) || die 'dsv4 profile rendered an empty environment'
    subst_placeholders launcher_env "port=$PORT" "repo=$REPO"
    install -d -o root -g dsv4 -m 1770 /run/dsv4
    /usr/sbin/runuser -u dsv4 -- env -i \
        "${launcher_env[@]}" \
        "$REPO/scripts/21_serve_llamacpp.sh" "$@"
}

sha256() { sha256sum -- "$1" | awk '{print $1}'; }

json_status() {
    clean_python - "$ACTIVE" <<'PY'
import json, os, sys
path = sys.argv[1]
profile = None
state = "inactive"
try:
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") == 1 and value.get("profile") in {"dsv4", "glm52", "qwen38", "qwen38-1m", "laguna"}:
        profile = value["profile"]
        state = "recorded"
except (OSError, ValueError, TypeError):
    pass
print(json.dumps({"schema_version": 1, "active_profile": profile, "state": state},
                 separators=(",", ":"), sort_keys=True))
PY
}

read_active_profile() {
    clean_python - "$ACTIVE" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        value = json.load(stream)
    profile = value.get("profile")
    print(profile if profile in {"dsv4", "glm52", "qwen38", "qwen38-1m", "laguna"} else "")
except (OSError, ValueError, TypeError):
    print("")
PY
}

glm_qualified() {
    [[ ${ENGINE_SWITCH_TESTING:-0} != 1 ]] || return 1
    env -i PATH=/usr/bin:/bin HOME=/nonexistent LANG=C.UTF-8 \
        "$REPO/scripts/glm52_goal.py" release-check --json >/dev/null
}

verify_glm_hashes() {
    clean_python - "$PROFILE_MANIFEST" "$BINARY" "$GGUF" <<'PY'
import hashlib, json, os, stat, sys
manifest_path, binary_path, model_path = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as stream:
    manifest = json.load(stream)
def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()
if manifest.get("binary_path") != binary_path:
    raise SystemExit("GLM binary path is not approved")
if digest(binary_path) != manifest["binary_sha256"]:
    raise SystemExit("GLM binary hash is not approved")
if manifest.get("model_path") != model_path:
    raise SystemExit("GLM model path is not approved")
identity = manifest.get("model_identity", {})
if identity.get("first_bytes") != 1048576:
    raise SystemExit("GLM model prefix length is not approved")
with open(model_path, "rb") as stream:
    before = os.fstat(stream.fileno())
    first_bytes = stream.read(identity["first_bytes"])
    after = os.fstat(stream.fileno())
stable_before = (
    before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
    before.st_uid, before.st_gid, before.st_size, before.st_mtime_ns,
    before.st_ctime_ns,
)
stable_after = (
    after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
    after.st_uid, after.st_gid, after.st_size, after.st_mtime_ns,
    after.st_ctime_ns,
)
if stable_before != stable_after or not stat.S_ISREG(before.st_mode):
    raise SystemExit("GLM model identity changed during verification")
if (before.st_size, before.st_dev, before.st_ino) != (
    identity.get("size_bytes"), identity.get("device"), identity.get("inode")
):
    raise SystemExit("GLM model stat identity is not approved")
if len(first_bytes) != identity.get("first_bytes"):
    raise SystemExit("GLM model prefix is short")
if hashlib.sha256(first_bytes).hexdigest() != identity.get("first_bytes_sha256"):
    raise SystemExit("GLM model prefix hash is not approved")
if manifest.get("context_cap") != 32768:
    raise SystemExit("GLM profile context cap is not 32768")
PY
}

verify_qwen_profile_hashes() {
    local manifest_path=$1 expected_profile=$2 expected_context=$3
    qwen_verified_identities=$(clean_python - "$manifest_path" "$QWEN_BINARY" "$QWEN_MODEL" \
            "$QWEN_MMPROJ" "$expected_profile" "$expected_context" <<'PY'
import hashlib, json, os, stat, sys
manifest_path, binary_path, model_path, mmproj_path, expected_profile, expected_context = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as stream:
    manifest = json.load(stream)

def digest_and_identity(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    value = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit(f"Qwen artifact is not a regular file: {path}")
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(chunk)
        after = os.fstat(stream.fileno())
    fields = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_nlink, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )
    if fields(before) != fields(after):
        raise SystemExit(f"Qwen artifact changed during verification: {path}")
    return value.hexdigest(), fields(after)

identities = {}
for label, path, path_key, hash_key in (
    ("binary", binary_path, "binary_path", "binary_sha256"),
    ("model", model_path, "model_path", "model_sha256"),
    ("mmproj", mmproj_path, "mmproj_path", "mmproj_sha256"),
):
    if manifest.get(path_key) != path:
        raise SystemExit(f"Qwen {label} path is not approved")
    digest, identity = digest_and_identity(path)
    if digest != manifest.get(hash_key):
        raise SystemExit(f"Qwen {label} hash is not approved")
    identities[label] = identity
if manifest.get("profile") != expected_profile or manifest.get("schema_version") != 3:
    raise SystemExit("Qwen profile identity is not approved")
if manifest.get("port") != 8013 or manifest.get("context_cap") != int(expected_context):
    raise SystemExit("Qwen serving topology is not approved")
print(json.dumps(identities, separators=(",", ":"), sort_keys=True))
PY
    )
    qwen_hashes_verified_profile=$expected_profile
}

verify_qwen_hashes() {
    verify_qwen_profile_hashes "$QWEN_PROFILE_MANIFEST" qwen38 32768
}

verify_qwen_1m_hashes() {
    verify_qwen_profile_hashes "$QWEN_1M_PROFILE_MANIFEST" qwen38-1m 1048576
}

revalidate_qwen_identities() {
    [[ -n $qwen_verified_identities ]] || die "Qwen artifacts lack verified identities"
    clean_python - "$qwen_verified_identities" "$QWEN_BINARY" "$QWEN_MODEL" \
            "$QWEN_MMPROJ" <<'PY'
import json, os, stat, sys
expected = json.loads(sys.argv[1])
for label, path in zip(("binary", "model", "mmproj"), sys.argv[2:]):
    info = os.lstat(path)
    actual = [
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    ]
    if not stat.S_ISREG(info.st_mode) or actual != expected[label]:
        raise SystemExit(f"Qwen {label} identity changed after hash approval")
PY
}

verify_laguna_profile_hashes() {
    laguna_verified_identities=$(clean_python - "$LAGUNA_PROFILE_MANIFEST" \
            "$LAGUNA_BUILD_MANIFEST" "$LAGUNA_BINARY" "$LAGUNA_MODEL" \
            "$LAGUNA_DRAFT" <<'PY'
import hashlib, json, os, re, stat, sys
manifest_path, build_manifest_path, binary_path, model_path, draft_path = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as stream:
    manifest = json.load(stream)
with open(build_manifest_path, encoding="utf-8") as stream:
    build_manifest = json.load(stream)

def digest_and_identity(path, expected_bytes, *, nofollow=True,
                        required_directory=None):
    flags = os.O_RDONLY | os.O_CLOEXEC
    if nofollow:
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    if required_directory is not None:
        opened_path = os.path.realpath(f"/proc/self/fd/{descriptor}")
        if os.path.dirname(opened_path) != os.path.realpath(required_directory):
            os.close(descriptor)
            raise SystemExit(f"Laguna artifact escaped its approved directory: {path}")
    value = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or
                (expected_bytes is not None and before.st_size != expected_bytes)):
            raise SystemExit(f"Laguna artifact size or type is not approved: {path}")
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(chunk)
        after = os.fstat(stream.fileno())
    fields = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_nlink, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )
    if fields(before) != fields(after):
        raise SystemExit(f"Laguna artifact changed during verification: {path}")
    return value.hexdigest(), fields(after)

if manifest.get("profile") != "laguna" or manifest.get("schema_version") != 3:
    raise SystemExit("Laguna profile identity is not approved")
if manifest.get("port") != 8013 or manifest.get("context_cap") != 393216:
    raise SystemExit("Laguna serving topology is not approved")
if manifest.get("binary_path") != binary_path:
    raise SystemExit("Laguna binary path is not approved")
identities = {}
digest, identity = digest_and_identity(binary_path, manifest.get("binary_bytes"))
if digest != manifest.get("binary_sha256"):
    raise SystemExit("Laguna binary hash is not approved")
identities["binary"] = identity
if build_manifest.get("schema_version") != 1:
    raise SystemExit("Laguna build manifest identity is not approved")
build_binary = build_manifest.get("binaries", {}).get("llama-server")
if (not isinstance(build_binary, dict) or
        build_binary.get("path") != binary_path or
        build_binary.get("sha256") != manifest.get("binary_sha256")):
    raise SystemExit("Laguna build manifest does not approve the serving binary")
shared_libraries = build_manifest.get("shared_libraries")
if not isinstance(shared_libraries, dict) or not shared_libraries:
    raise SystemExit("Laguna shared-library inventory is not approved")
binary_directory = os.path.dirname(binary_path)
for name, record in sorted(shared_libraries.items()):
    if (not isinstance(name, str) or os.path.basename(name) != name or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*[.]so", name)):
        raise SystemExit(f"Laguna shared-library name is unsafe: {name!r}")
    if not isinstance(record, dict):
        raise SystemExit(f"Laguna shared-library record is invalid: {name}")
    library_path = os.path.join(binary_directory, name)
    if os.path.dirname(os.path.realpath(library_path)) != os.path.realpath(binary_directory):
        raise SystemExit(f"Laguna shared library escapes the binary directory: {name}")
    digest, identity = digest_and_identity(
        library_path, None, nofollow=False, required_directory=binary_directory)
    if digest != record.get("sha256"):
        raise SystemExit(f"Laguna shared-library hash is not approved: {name}")
    identities[f"library:{name}"] = identity
shards = manifest.get("model_shards")
if not isinstance(shards, list) or len(shards) != 3:
    raise SystemExit("Laguna shard inventory is not approved")
if manifest.get("model_path") != model_path or shards[0].get("path") != model_path:
    raise SystemExit("Laguna model load path is not approved")
shard1_suffix = "-00001-of-00003.gguf"
if not model_path.endswith(shard1_suffix):
    raise SystemExit("Laguna load shard does not have the approved split-model suffix")
shard_prefix = model_path[:-len(shard1_suffix)]
derived_shard_paths = [
    model_path,
    shard_prefix + "-00002-of-00003.gguf",
    shard_prefix + "-00003-of-00003.gguf",
]
for index, (shard, path) in enumerate(zip(shards, derived_shard_paths)):
    if shard.get("path") != path:
        raise SystemExit(f"Laguna profile shard {index + 1} differs from loader path")
    digest, identity = digest_and_identity(path, shard.get("bytes"))
    if digest != shard.get("sha256"):
        raise SystemExit(f"Laguna shard {index + 1} hash is not approved")
    identities[f"shard{index + 1}"] = identity
if manifest.get("model_sha256") != shards[0].get("sha256"):
    raise SystemExit("Laguna load-shard hash is inconsistent")
if manifest.get("draft_model_path") != draft_path:
    raise SystemExit("Laguna draft path is not approved")
digest, identity = digest_and_identity(draft_path, manifest.get("draft_model_bytes"))
if digest != manifest.get("draft_model_sha256"):
    raise SystemExit("Laguna draft hash is not approved")
identities["draft"] = identity
print(json.dumps(identities, separators=(",", ":"), sort_keys=True))
PY
    ) || die "Laguna artifact verification failed"
    laguna_hashes_verified=true
}

revalidate_laguna_identities() {
    [[ -n $laguna_verified_identities ]] || die "Laguna artifacts lack verified identities"
    clean_python - "$laguna_verified_identities" "$LAGUNA_BINARY" \
            "$LAGUNA_MODEL" "$LAGUNA_DRAFT" <<'PY'
import json, os, re, stat, sys
expected = json.loads(sys.argv[1])
paths = [("binary", sys.argv[2], False), ("shard1", sys.argv[3], False),
         ("draft", sys.argv[4], False)]
shard1_suffix = "-00001-of-00003.gguf"
shard_prefix = sys.argv[3][:-len(shard1_suffix)]
paths.extend((f"shard{i}", shard_prefix + f"-0000{i}-of-00003.gguf", False)
             for i in (2, 3))
binary_directory = os.path.dirname(sys.argv[2])
shared_libraries = sorted(
    label.removeprefix("library:") for label in expected
    if label.startswith("library:")
)
if not shared_libraries:
    raise SystemExit("Laguna verified shared-library inventory is empty")
for name in shared_libraries:
    if (os.path.basename(name) != name or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*[.]so", name)):
        raise SystemExit(f"Laguna verified shared-library name is unsafe: {name!r}")
paths.extend((f"library:{name}", os.path.join(binary_directory, name), True)
             for name in shared_libraries)
for label, path, follow_symlinks in paths:
    if follow_symlinks and os.path.dirname(os.path.realpath(path)) != os.path.realpath(binary_directory):
        raise SystemExit(f"Laguna {label} escaped the binary directory")
    info = os.stat(path) if follow_symlinks else os.lstat(path)
    actual = [info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
              info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    if not stat.S_ISREG(info.st_mode) or actual != expected[label]:
        raise SystemExit(f"Laguna {label} identity changed after hash approval")
PY
}

proc_identity() {
    local pid=$1 line
    [[ $pid =~ ^[0-9]+$ && $pid -gt 1 && -r /proc/$pid/stat ]] || return 1
    IFS= read -r line <"/proc/$pid/stat" || return 1
    line=${line##*) }
    local -a fields
    read -r -a fields <<<"$line"
    [[ ${fields[2]} =~ ^[0-9]+$ && ${fields[19]} =~ ^[0-9]+$ ]] || return 1
    printf '%s %s\n' "${fields[2]}" "${fields[19]}"
}

stop_glm_verified() {
    [[ -f $GLM_PROCESS ]] || return 0
    local values pid expected_pgid expected_ticks expected_sha memwatch_pid
    local memwatch_ticks current current_pgid current_ticks exe cmdline
    values=$(clean_python - "$GLM_PROCESS" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
print(value["pid"], value["pgid"], value["start_ticks"], value["exe_sha256"],
      value["memwatch_pid"], value["memwatch_start_ticks"])
PY
    ) || die "invalid GLM process record"
    read -r pid expected_pgid expected_ticks expected_sha memwatch_pid \
        memwatch_ticks <<<"$values"
    current=$(proc_identity "$pid" 2>/dev/null || true)
    if [[ -n $current ]]; then
        read -r current_pgid current_ticks <<<"$current"
        exe=$(readlink -f "/proc/$pid/exe") ||
            die "cannot resolve recorded GLM executable"
        [[ $current_pgid == "$expected_pgid" &&
                $current_ticks == "$expected_ticks" ]] ||
            die "stale GLM PID identity; refusing to signal"
        [[ $(sha256 "$exe") == "$expected_sha" ]] ||
            die "GLM executable hash changed; refusing to signal"
        kill -TERM -- "-$expected_pgid"
        for _ in $(seq 1 600); do
            [[ $(proc_identity "$pid" 2>/dev/null || true) != "$current" ]] && break
            sleep 0.1
        done
        if [[ $(proc_identity "$pid" 2>/dev/null || true) == "$current" ]]; then
            kill -KILL -- "-$expected_pgid"
        fi
    fi
    if [[ $(proc_identity "$memwatch_pid" 2>/dev/null || true) == *" $memwatch_ticks" ]]; then
        cmdline=$(tr '\0' ' ' <"/proc/$memwatch_pid/cmdline")
        [[ $cmdline == *"$REPO/scripts/01_memwatch.sh"* &&
                $cmdline == *"$GLM_WATCHDOG_TARGET"* ]] ||
            die "GLM memwatch identity changed; refusing to disarm"
        printf 'DISARM %s %s %s\n' "$pid" "$expected_pgid" "$expected_ticks" \
            >"$GLM_WATCHDOG_TARGET.tmp"
        mv -- "$GLM_WATCHDOG_TARGET.tmp" "$GLM_WATCHDOG_TARGET"
        for _ in $(seq 1 50); do
            [[ -d /proc/$memwatch_pid ]] || break
            sleep 0.1
        done
        [[ ! -d /proc/$memwatch_pid ]] ||
            die "GLM memwatch did not accept authenticated disarm"
    fi
    rm -f -- "$GLM_PROCESS" "$GLM_WATCHDOG_TARGET" "$GLM_WATCHDOG_READY"
}

stop_qwen_verified() {
    local values pid expected_pgid expected_ticks expected_sha expected_unit
    local memwatch_pid memwatch_ticks current current_pgid current_ticks
    local exe unit_pid cmdline ready
    if [[ ! -f $QWEN_PROCESS ]]; then
        if ! unit_pid=$(systemctl show "$QWEN_UNIT" --property=MainPID --value \
                2>/dev/null); then
            die "cannot query Qwen unit MainPID; refusing to assume it is stopped"
        fi
        if [[ $unit_pid =~ ^[0-9]+$ && $unit_pid -gt 1 ]]; then
            die "Qwen unit is live without an identity record; refusing to continue"
        fi
        return 0
    fi
    values=$(clean_python - "$QWEN_PROCESS" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
print(value["pid"], value["pgid"], value["start_ticks"],
      value["exe_sha256"], value["unit"], value["memwatch_pid"],
      value["memwatch_start_ticks"])
PY
    ) || die "invalid Qwen process record"
    read -r pid expected_pgid expected_ticks expected_sha expected_unit \
        memwatch_pid memwatch_ticks <<<"$values"
    [[ $expected_unit == "$QWEN_UNIT" ]] ||
        die "Qwen process record names an unexpected unit"
    current=$(proc_identity "$pid" 2>/dev/null || true)
    if ! unit_pid=$(systemctl show "$QWEN_UNIT" --property=MainPID --value \
            2>/dev/null); then
        die "cannot query Qwen unit MainPID; refusing to stop it"
    fi
    if [[ -n $current ]]; then
        read -r current_pgid current_ticks <<<"$current"
        exe=$(readlink -f "/proc/$pid/exe") ||
            die "cannot resolve recorded Qwen executable"
        [[ $unit_pid == "$pid" && $current_pgid == "$expected_pgid" &&
                $current_ticks == "$expected_ticks" ]] ||
            die "stale Qwen PID identity; refusing to stop unit"
        [[ $(sha256 "$exe") == "$expected_sha" ]] ||
            die "Qwen executable hash changed; refusing to stop unit"
        systemctl stop "$QWEN_UNIT"
        for _ in $(seq 1 600); do
            [[ $(proc_identity "$pid" 2>/dev/null || true) != "$current" ]] && break
            sleep 0.1
        done
        [[ $(proc_identity "$pid" 2>/dev/null || true) != "$current" ]] ||
            die "Qwen transient unit did not stop its recorded process"
    elif [[ $unit_pid =~ ^[0-9]+$ && $unit_pid -gt 1 ]]; then
        die "Qwen unit has an unrecorded live MainPID; refusing to stop it"
    fi
    if [[ $(proc_identity "$memwatch_pid" 2>/dev/null || true) == *" $memwatch_ticks" ]]; then
        cmdline=$(tr '\0' ' ' <"/proc/$memwatch_pid/cmdline")
        [[ $cmdline == *"$REPO/scripts/01_memwatch.sh"* &&
                $cmdline == *"$QWEN_WATCHDOG_TARGET"* ]] ||
            die "Qwen memwatch identity changed; refusing to disarm"
        printf 'DISARM %s %s %s\n' "$pid" "$expected_pgid" "$expected_ticks" \
            >"$QWEN_WATCHDOG_TARGET.tmp"
        mv -- "$QWEN_WATCHDOG_TARGET.tmp" "$QWEN_WATCHDOG_TARGET"
        for _ in $(seq 1 50); do
            [[ -d /proc/$memwatch_pid ]] || break
            sleep 0.1
        done
        ready=$(cat "$QWEN_WATCHDOG_READY" 2>/dev/null) ||
            die "Qwen memwatch disarm acknowledgement is missing"
        [[ $ready == "DISARMED $pid $expected_pgid $expected_ticks" ]] ||
            die "Qwen memwatch disarm acknowledgement does not match engine identity"
        [[ ! -d /proc/$memwatch_pid ]] ||
            die "Qwen memwatch did not accept authenticated disarm"
    fi
    rm -f -- "$QWEN_PROCESS" "$QWEN_WATCHDOG_TARGET" "$QWEN_WATCHDOG_READY"
}

stop_laguna_verified() {
    local values pid expected_pgid expected_ticks expected_sha expected_unit
    local memwatch_pid memwatch_ticks current current_pgid current_ticks
    local exe unit_pid cmdline ready
    if [[ ! -f $LAGUNA_PROCESS ]]; then
        if ! unit_pid=$(systemctl show "$LAGUNA_UNIT" --property=MainPID --value \
                2>/dev/null); then
            die "cannot query Laguna unit MainPID; refusing to assume it is stopped"
        fi
        if [[ $unit_pid =~ ^[0-9]+$ && $unit_pid -gt 1 ]]; then
            die "Laguna unit is live without an identity record; refusing to continue"
        fi
        return 0
    fi
    values=$(clean_python - "$LAGUNA_PROCESS" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
print(value["pid"], value["pgid"], value["start_ticks"], value["exe_sha256"],
      value["unit"], value["memwatch_pid"], value["memwatch_start_ticks"])
PY
    ) || die "invalid Laguna process record"
    read -r pid expected_pgid expected_ticks expected_sha expected_unit \
        memwatch_pid memwatch_ticks <<<"$values"
    [[ $expected_unit == "$LAGUNA_UNIT" ]] ||
        die "Laguna process record names an unexpected unit"
    current=$(proc_identity "$pid" 2>/dev/null || true)
    if ! unit_pid=$(systemctl show "$LAGUNA_UNIT" --property=MainPID --value \
            2>/dev/null); then
        die "cannot query Laguna unit MainPID; refusing to stop it"
    fi
    if [[ -n $current ]]; then
        read -r current_pgid current_ticks <<<"$current"
        exe=$(readlink -f "/proc/$pid/exe") ||
            die "cannot resolve recorded Laguna executable"
        [[ $unit_pid == "$pid" && $current_pgid == "$expected_pgid" &&
                $current_ticks == "$expected_ticks" ]] ||
            die "stale Laguna PID identity; refusing to stop unit"
        [[ $(sha256 "$exe") == "$expected_sha" ]] ||
            die "Laguna executable hash changed; refusing to stop unit"
        systemctl stop "$LAGUNA_UNIT"
        for _ in $(seq 1 600); do
            [[ $(proc_identity "$pid" 2>/dev/null || true) != "$current" ]] && break
            sleep 0.1
        done
        [[ $(proc_identity "$pid" 2>/dev/null || true) != "$current" ]] ||
            die "Laguna transient unit did not stop its recorded process"
    elif [[ $unit_pid =~ ^[0-9]+$ && $unit_pid -gt 1 ]]; then
        die "Laguna unit has an unrecorded live MainPID; refusing to stop it"
    fi
    if [[ $(proc_identity "$memwatch_pid" 2>/dev/null || true) == *" $memwatch_ticks" ]]; then
        cmdline=$(tr '\0' ' ' <"/proc/$memwatch_pid/cmdline")
        [[ $cmdline == *"$REPO/scripts/01_memwatch.sh"* &&
                $cmdline == *"$LAGUNA_WATCHDOG_TARGET"* ]] ||
            die "Laguna memwatch identity changed; refusing to disarm"
        printf 'DISARM %s %s %s\n' "$pid" "$expected_pgid" "$expected_ticks" \
            >"$LAGUNA_WATCHDOG_TARGET.tmp"
        mv -- "$LAGUNA_WATCHDOG_TARGET.tmp" "$LAGUNA_WATCHDOG_TARGET"
        for _ in $(seq 1 50); do
            [[ -d /proc/$memwatch_pid ]] || break
            sleep 0.1
        done
        ready=$(cat "$LAGUNA_WATCHDOG_READY" 2>/dev/null) ||
            die "Laguna memwatch disarm acknowledgement is missing"
        [[ $ready == "DISARMED $pid $expected_pgid $expected_ticks" ]] ||
            die "Laguna memwatch disarm acknowledgement does not match engine identity"
        [[ ! -d /proc/$memwatch_pid ]] ||
            die "Laguna memwatch did not accept authenticated disarm"
    fi
    rm -f -- "$LAGUNA_PROCESS" "$LAGUNA_WATCHDOG_TARGET" "$LAGUNA_WATCHDOG_READY"
}

stop_profile() {
    case "$1" in
        dsv4)
            if dsv4_launcher stop; then
                return 0
            fi
            # The launcher fails closed after removing a dead, identity-checked
            # state record. Treat only that exact absent-and-unreachable result
            # as already stopped; a live or ambiguous endpoint still fails.
            [[ ! -e /run/dsv4/llamacpp.state.json ]] || return 1
            if clean_curl -fsS --max-time 2 \
                    "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
                return 1
            fi
            return 0
            ;;
        glm52) stop_glm_verified ;;
        qwen38|qwen38-1m) stop_qwen_verified ;;
        laguna) stop_laguna_verified ;;
        "") return 0 ;;
        *) die "unknown previous profile $1" ;;
    esac
}

start_dsv4() {
    dsv4_launcher start
}

start_glm52() {
    local pid identity pgid ticks exe_sha current
    local memwatch_pid memwatch_ticks ready
    [[ ! -e $GLM_PROCESS ]] ||
        die "GLM process record already exists; refusing a second model"
    "$REPO/scripts/03_memory_guard.py" --required-gib 110 \
        --stable-samples 3 --interval-seconds 1 --timeout-seconds 180
    rm -f -- "$GLM_WATCHDOG_TARGET" "$GLM_WATCHDOG_READY" \
        "$GLM_PROCESS.tmp"
    "$REPO/scripts/01_memwatch.sh" \
        --target-file "$GLM_WATCHDOG_TARGET" \
        --ready-file "$GLM_WATCHDOG_READY" \
        --threshold-gib 18 --interval-sec 1 --log "$GLM_WATCHDOG_LOG" 9>&- &
    memwatch_pid=$!
    memwatch_ticks=
    ready=
    for _ in $(seq 1 50); do
        memwatch_ticks=$(proc_identity "$memwatch_pid" 2>/dev/null || true)
        memwatch_ticks=${memwatch_ticks#* }
        ready=$(cat "$GLM_WATCHDOG_READY" 2>/dev/null || true)
        [[ -n $memwatch_ticks && $ready == READY ]] && break
        sleep 0.1
    done
    if [[ -z $memwatch_ticks || $ready != READY ]]; then
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "GLM memory watchdog failed to initialize"
    fi
    local -a glm_env glm_argv
    read_profile_array glm_env glm52 env
    read_profile_array glm_argv glm52 args
    (( ${#glm_env[@]} > 0 )) || die 'GLM profile rendered an empty environment'
    (( ${#glm_argv[@]} > 0 )) || die 'GLM profile rendered an empty argv'
    subst_placeholders glm_argv "model=$GGUF" "port=$PORT"
    setsid env -i "${glm_env[@]}" \
        "$BINARY" "${glm_argv[@]}" \
        >"$STATE/glm52.server.log" 2>&1 9>&- &
    pid=$!
    identity=
    for _ in $(seq 1 20); do
        identity=$(proc_identity "$pid" 2>/dev/null || true)
        [[ -n $identity ]] && break
        sleep 1
    done
    if [[ -z $identity ]]; then
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "GLM startup died before identity capture"
    fi
    read -r pgid ticks <<<"$identity"
    exe_sha=$(sha256 "$BINARY")
    if ! clean_python - "$GLM_PROCESS.tmp" "$pid" "$pgid" "$ticks" \
            "$exe_sha" "$memwatch_pid" "$memwatch_ticks" <<'PY'
import json, os, sys
path, pid, pgid, ticks, digest, watchdog_pid, watchdog_ticks = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump({"schema_version":1, "pid":int(pid), "pgid":int(pgid),
               "start_ticks":int(ticks), "exe_sha256":digest,
               "memwatch_pid":int(watchdog_pid),
               "memwatch_start_ticks":int(watchdog_ticks)}, stream)
    stream.flush(); os.fsync(stream.fileno())
PY
    then
        current=$(proc_identity "$pid" 2>/dev/null || true)
        if [[ $current == "$identity" ]]; then
            kill -TERM -- "-$pgid" 2>/dev/null || true
            for _ in $(seq 1 100); do
                [[ $(proc_identity "$pid" 2>/dev/null || true) != "$identity" ]] && break
                sleep 0.1
            done
            if [[ $(proc_identity "$pid" 2>/dev/null || true) == "$identity" ]]; then
                kill -KILL -- "-$pgid" 2>/dev/null || true
            fi
        fi
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        rm -f -- "$GLM_PROCESS.tmp"
        if [[ $(proc_identity "$pid" 2>/dev/null || true) == "$identity" ]]; then
            rollback_needed=false
            die "GLM cleanup failed after process record creation failure; rollback suppressed"
        fi
        die "GLM process identity record could not be created"
    fi
    mv -- "$GLM_PROCESS.tmp" "$GLM_PROCESS"
    printf '%s %s %s provisional\n' "$pid" "$pgid" "$ticks" \
        >"$GLM_WATCHDOG_TARGET.tmp"
    mv -- "$GLM_WATCHDOG_TARGET.tmp" "$GLM_WATCHDOG_TARGET"
    ready=
    for _ in $(seq 1 50); do
        ready=$(cat "$GLM_WATCHDOG_READY" 2>/dev/null || true)
        [[ $ready == "ARMED $pid $pgid $ticks provisional" ]] && break
        sleep 0.1
    done
    [[ $ready == "ARMED $pid $pgid $ticks provisional" ]] ||
        die "GLM memory watchdog did not arm provisional process"
    printf '%s %s %s engine\n' "$pid" "$pgid" "$ticks" \
        >"$GLM_WATCHDOG_TARGET.tmp"
    mv -- "$GLM_WATCHDOG_TARGET.tmp" "$GLM_WATCHDOG_TARGET"
    for _ in $(seq 1 50); do
        ready=$(cat "$GLM_WATCHDOG_READY" 2>/dev/null || true)
        [[ $ready == "ARMED $pid $pgid $ticks engine" ]] && break
        sleep 0.1
    done
    [[ $ready == "ARMED $pid $pgid $ticks engine" ]] ||
        die "GLM memory watchdog did not arm final process"
}

launch_qwen38() {
    launch_systemd_profile qwen38 "$QWEN_BINARY" "$QWEN_MODEL" "$QWEN_MMPROJ" ""
}

launch_qwen38-1m() {
    launch_systemd_profile qwen38-1m "$QWEN_BINARY" "$QWEN_MODEL" "$QWEN_MMPROJ" ""
}

launch_laguna() {
    launch_systemd_profile laguna "$LAGUNA_BINARY" "$LAGUNA_MODEL" "" "$LAGUNA_DRAFT"
}

cleanup_laguna_killed_unit() {
    local load_state active_state
    load_state=$(systemctl show "$LAGUNA_UNIT" --property=LoadState --value \
        2>/dev/null || true)
    [[ -z $load_state || $load_state == not-found ]] && return 0
    active_state=$(systemctl show "$LAGUNA_UNIT" --property=ActiveState --value \
        2>/dev/null || true)
    if [[ $active_state == failed || $active_state == inactive ]]; then
        systemctl reset-failed "$LAGUNA_UNIT" 2>/dev/null || true
        for _ in $(seq 1 50); do
            load_state=$(systemctl show "$LAGUNA_UNIT" --property=LoadState --value \
                2>/dev/null || true)
            [[ -z $load_state || $load_state == not-found ]] && return 0
            sleep 0.1
        done
    fi
    die "Laguna transient unit already exists (LoadState=$load_state, ActiveState=$active_state)"
}

start_laguna_profile() {
    local pid identity pgid ticks exe_sha approved_sha unit_pid current
    local live_exe approved_exe
    local memwatch_pid memwatch_identity memwatch_ticks ready
    [[ ! -e $LAGUNA_PROCESS ]] ||
        die "Laguna process record already exists; refusing a second model"
    { "$laguna_hashes_verified" || verify_laguna_profile_hashes; } ||
        die "Laguna artifact verification failed"
    "$REPO/scripts/03_memory_guard.py" --required-gib 100 \
        --stable-samples 3 --interval-seconds 1 --timeout-seconds 180 ||
        die "pre-load memory release gate failed"
    cleanup_laguna_killed_unit ||
        die "Laguna killed-unit cleanup failed"
    rm -f -- "$LAGUNA_PROCESS.tmp" "$LAGUNA_WATCHDOG_TARGET" \
        "$LAGUNA_WATCHDOG_READY" ||
        die "Laguna stale startup-state cleanup failed"
    "$REPO/scripts/01_memwatch.sh" \
        --target-file "$LAGUNA_WATCHDOG_TARGET" \
        --ready-file "$LAGUNA_WATCHDOG_READY" \
        --threshold-gib 8 --interval-sec 1 --log "$LAGUNA_WATCHDOG_LOG" 9>&- &
    memwatch_pid=$!
    memwatch_ticks=
    ready=
    for _ in $(seq 1 50); do
        memwatch_identity=$(proc_identity "$memwatch_pid" 2>/dev/null || true)
        memwatch_ticks=${memwatch_identity#* }
        ready=$(cat "$LAGUNA_WATCHDOG_READY" 2>/dev/null || true)
        [[ -n $memwatch_ticks && $ready == READY ]] && break
        sleep 0.1
    done
    if [[ -z $memwatch_ticks || $ready != READY ]]; then
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna memory watchdog failed to initialize"
    fi
    if ! revalidate_laguna_identities; then
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna artifact identity changed before execution"
    fi
    if ! launch_laguna; then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna transient unit failed to start"
    fi
    pid=
    identity=
    for _ in $(seq 1 100); do
        unit_pid=$(systemctl show "$LAGUNA_UNIT" --property=MainPID --value \
            2>/dev/null || true)
        if [[ $unit_pid =~ ^[0-9]+$ && $unit_pid -gt 1 ]]; then
            pid=$unit_pid
            identity=$(proc_identity "$pid" 2>/dev/null || true)
            [[ -n $identity ]] && break
        fi
        sleep 0.1
    done
    if [[ -z $pid || -z $identity ]]; then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna transient unit died before identity capture"
    fi
    if ! read -r pgid ticks <<<"$identity"; then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna process identity could not be parsed"
    fi
    if [[ $pgid != "$pid" ]]; then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna transient-unit server is not its process-group leader"
    fi
    if ! live_exe=$(readlink -f "/proc/$pid/exe"); then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna transient unit executable could not be resolved"
    fi
    if ! approved_exe=$(readlink -f "$LAGUNA_BINARY"); then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna approved executable could not be resolved"
    fi
    if [[ $live_exe != "$approved_exe" ]]; then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna transient unit executable identity is wrong"
    fi
    if ! exe_sha=$(sha256 "/proc/$pid/exe"); then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna executable could not be hashed after launch"
    fi
    if ! approved_sha=$(clean_python - "$LAGUNA_PROFILE_MANIFEST" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["binary_sha256"])
PY
    ); then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna approved executable hash could not be read"
    fi
    if [[ $exe_sha != "$approved_sha" ]]; then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna transient unit executed an unapproved binary"
    fi
    if ! clean_python - "$LAGUNA_PROCESS.tmp" "$pid" "$pgid" "$ticks" \
            "$exe_sha" "$LAGUNA_UNIT" "$memwatch_pid" "$memwatch_ticks" <<'PY'
import json, os, sys
path, pid, pgid, ticks, digest, unit, watchdog_pid, watchdog_ticks = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump({"schema_version":1, "pid":int(pid), "pgid":int(pgid),
               "start_ticks":int(ticks), "exe_sha256":digest, "unit":unit,
               "memwatch_pid":int(watchdog_pid),
               "memwatch_start_ticks":int(watchdog_ticks)}, stream)
    stream.flush(); os.fsync(stream.fileno())
PY
    then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        rm -f -- "$LAGUNA_PROCESS.tmp" || true
        die "Laguna process identity record could not be created"
    fi
    current=$(proc_identity "$pid" 2>/dev/null || true)
    if [[ $current != "$identity" ]]; then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        rm -f -- "$LAGUNA_PROCESS.tmp" || true
        die "Laguna process identity changed before record publication"
    fi
    if ! mv -- "$LAGUNA_PROCESS.tmp" "$LAGUNA_PROCESS"; then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Laguna process identity record could not be published"
    fi
    if ! printf '%s %s %s provisional\n' "$pid" "$pgid" "$ticks" \
            >"$LAGUNA_WATCHDOG_TARGET.tmp"; then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        rm -f -- "$LAGUNA_PROCESS" "$LAGUNA_WATCHDOG_TARGET.tmp" || true
        die "Laguna provisional watchdog target could not be written"
    fi
    if ! mv -- "$LAGUNA_WATCHDOG_TARGET.tmp" "$LAGUNA_WATCHDOG_TARGET"; then
        systemctl stop "$LAGUNA_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        rm -f -- "$LAGUNA_PROCESS" "$LAGUNA_WATCHDOG_TARGET.tmp" || true
        die "Laguna provisional watchdog target could not be published"
    fi
    ready=
    for _ in $(seq 1 50); do
        ready=$(cat "$LAGUNA_WATCHDOG_READY" 2>/dev/null || true)
        [[ $ready == "ARMED $pid $pgid $ticks provisional" ]] && break
        sleep 0.1
    done
    [[ $ready == "ARMED $pid $pgid $ticks provisional" ]] ||
        die "Laguna memory watchdog did not arm provisional process"
    printf '%s %s %s engine\n' "$pid" "$pgid" "$ticks" \
        >"$LAGUNA_WATCHDOG_TARGET.tmp" ||
        die "Laguna final watchdog target could not be written"
    mv -- "$LAGUNA_WATCHDOG_TARGET.tmp" "$LAGUNA_WATCHDOG_TARGET" ||
        die "Laguna final watchdog target could not be published"
    ready=
    for _ in $(seq 1 50); do
        ready=$(cat "$LAGUNA_WATCHDOG_READY" 2>/dev/null || true)
        [[ $ready == "ARMED $pid $pgid $ticks engine" ]] && break
        sleep 0.1
    done
    [[ $ready == "ARMED $pid $pgid $ticks engine" ]] ||
        die "Laguna memory watchdog did not arm final process"
}

start_laguna() {
    start_laguna_profile
}

cleanup_qwen_killed_unit() {
    local load_state active_state
    load_state=$(systemctl show "$QWEN_UNIT" --property=LoadState --value \
        2>/dev/null || true)
    [[ -z $load_state || $load_state == not-found ]] && return 0
    active_state=$(systemctl show "$QWEN_UNIT" --property=ActiveState --value \
        2>/dev/null || true)
    if [[ $active_state == failed || $active_state == inactive ]]; then
        systemctl reset-failed "$QWEN_UNIT" 2>/dev/null || true
        for _ in $(seq 1 50); do
            load_state=$(systemctl show "$QWEN_UNIT" --property=LoadState --value \
                2>/dev/null || true)
            [[ -z $load_state || $load_state == not-found ]] && return 0
            sleep 0.1
        done
    fi
    die "Qwen transient unit already exists (LoadState=$load_state, ActiveState=$active_state)"
}

start_qwen_profile() {
    local profile=$1 manifest_path=$2 verify_function=$3
    local pid identity pgid ticks exe_sha approved_sha unit_pid current
    local live_exe approved_exe
    local memwatch_pid memwatch_identity memwatch_ticks ready
    # Watchdog floor: 18 GiB for the 32K profile. The 1m profile measured
    # MemAvailable bottoming at 11 GiB with all four slots filled to ~260K
    # (gate 4), so an 18 GiB floor would false-trip near full load; 8 GiB
    # mirrors the owner-accepted DSV4 1M floor and still protects the box.
    local watchdog_floor_gib=18
    [[ $profile != qwen38-1m ]] || watchdog_floor_gib=8
    [[ ! -e $QWEN_PROCESS ]] ||
        die "Qwen process record already exists; refusing a second model"
    { [[ $qwen_hashes_verified_profile == "$profile" ]] || "$verify_function"; } ||
        die "Qwen artifact verification failed"
    "$REPO/scripts/03_memory_guard.py" --required-gib 100 \
        --stable-samples 3 --interval-seconds 1 --timeout-seconds 180 ||
        die "pre-load memory release gate failed"
    cleanup_qwen_killed_unit ||
        die "Qwen killed-unit cleanup failed"
    rm -f -- "$QWEN_PROCESS.tmp" "$QWEN_WATCHDOG_TARGET" \
        "$QWEN_WATCHDOG_READY" ||
        die "Qwen stale startup-state cleanup failed"
    "$REPO/scripts/01_memwatch.sh" \
        --target-file "$QWEN_WATCHDOG_TARGET" \
        --ready-file "$QWEN_WATCHDOG_READY" \
        --threshold-gib "$watchdog_floor_gib" --interval-sec 1 --log "$QWEN_WATCHDOG_LOG" 9>&- &
    memwatch_pid=$!
    memwatch_ticks=
    ready=
    for _ in $(seq 1 50); do
        memwatch_identity=$(proc_identity "$memwatch_pid" 2>/dev/null || true)
        memwatch_ticks=${memwatch_identity#* }
        ready=$(cat "$QWEN_WATCHDOG_READY" 2>/dev/null || true)
        [[ -n $memwatch_ticks && $ready == READY ]] && break
        sleep 0.1
    done
    if [[ -z $memwatch_ticks || $ready != READY ]]; then
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen memory watchdog failed to initialize"
    fi
    # The approved hashes were computed before stopping the old profile. Close
    # the pathname reopen window as far as systemd permits by checking the full
    # stat identities again immediately before systemd-run opens the paths.
    if ! revalidate_qwen_identities; then
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen artifact identity changed before execution"
    fi
    if ! "launch_$profile"; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen transient unit failed to start"
    fi
    pid=
    identity=
    for _ in $(seq 1 100); do
        unit_pid=$(systemctl show "$QWEN_UNIT" --property=MainPID --value \
            2>/dev/null || true)
        if [[ $unit_pid =~ ^[0-9]+$ && $unit_pid -gt 1 ]]; then
            pid=$unit_pid
            identity=$(proc_identity "$pid" 2>/dev/null || true)
            [[ -n $identity ]] && break
        fi
        sleep 0.1
    done
    if [[ -z $pid || -z $identity ]]; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen transient unit died before identity capture"
    fi
    if ! read -r pgid ticks <<<"$identity"; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen process identity could not be parsed"
    fi
    if [[ $pgid != "$pid" ]]; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen transient-unit server is not its process-group leader"
    fi
    if ! live_exe=$(readlink -f "/proc/$pid/exe"); then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen transient unit executable could not be resolved"
    fi
    if ! approved_exe=$(readlink -f "$QWEN_BINARY"); then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen approved executable could not be resolved"
    fi
    if [[ $live_exe != "$approved_exe" ]]; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen transient unit executable identity is wrong"
    fi
    if ! exe_sha=$(sha256 "/proc/$pid/exe"); then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen executable could not be hashed after launch"
    fi
    if ! approved_sha=$(clean_python - "$manifest_path" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["binary_sha256"])
PY
    ); then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen approved executable hash could not be read"
    fi
    if [[ $exe_sha != "$approved_sha" ]]; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen transient unit executed an unapproved binary"
    fi
    if ! clean_python - "$QWEN_PROCESS.tmp" "$pid" "$pgid" "$ticks" \
            "$exe_sha" "$QWEN_UNIT" "$memwatch_pid" "$memwatch_ticks" <<'PY'
import json, os, sys
path, pid, pgid, ticks, digest, unit, watchdog_pid, watchdog_ticks = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump({"schema_version":1, "pid":int(pid), "pgid":int(pgid),
               "start_ticks":int(ticks), "exe_sha256":digest,
               "unit":unit, "memwatch_pid":int(watchdog_pid),
               "memwatch_start_ticks":int(watchdog_ticks)}, stream)
    stream.flush(); os.fsync(stream.fileno())
PY
    then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        rm -f -- "$QWEN_PROCESS.tmp" || true
        die "Qwen process identity record could not be created"
    fi
    current=$(proc_identity "$pid" 2>/dev/null || true)
    if [[ $current != "$identity" ]]; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        rm -f -- "$QWEN_PROCESS.tmp" || true
        die "Qwen process identity changed before record publication"
    fi
    if ! mv -- "$QWEN_PROCESS.tmp" "$QWEN_PROCESS"; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        die "Qwen process identity record could not be published"
    fi
    if ! printf '%s %s %s provisional\n' "$pid" "$pgid" "$ticks" \
            >"$QWEN_WATCHDOG_TARGET.tmp"; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        rm -f -- "$QWEN_PROCESS" "$QWEN_WATCHDOG_TARGET.tmp" || true
        die "Qwen provisional watchdog target could not be written"
    fi
    if ! mv -- "$QWEN_WATCHDOG_TARGET.tmp" "$QWEN_WATCHDOG_TARGET"; then
        systemctl stop "$QWEN_UNIT" 2>/dev/null || true
        kill -TERM "$memwatch_pid" 2>/dev/null || true
        wait "$memwatch_pid" 2>/dev/null || true
        rm -f -- "$QWEN_PROCESS" "$QWEN_WATCHDOG_TARGET.tmp" || true
        die "Qwen provisional watchdog target could not be published"
    fi
    ready=
    for _ in $(seq 1 50); do
        ready=$(cat "$QWEN_WATCHDOG_READY" 2>/dev/null || true)
        [[ $ready == "ARMED $pid $pgid $ticks provisional" ]] && break
        sleep 0.1
    done
    [[ $ready == "ARMED $pid $pgid $ticks provisional" ]] ||
        die "Qwen memory watchdog did not arm provisional process"
    printf '%s %s %s engine\n' "$pid" "$pgid" "$ticks" \
        >"$QWEN_WATCHDOG_TARGET.tmp" ||
        die "Qwen final watchdog target could not be written"
    mv -- "$QWEN_WATCHDOG_TARGET.tmp" "$QWEN_WATCHDOG_TARGET" ||
        die "Qwen final watchdog target could not be published"
    ready=
    for _ in $(seq 1 50); do
        ready=$(cat "$QWEN_WATCHDOG_READY" 2>/dev/null || true)
        [[ $ready == "ARMED $pid $pgid $ticks engine" ]] && break
        sleep 0.1
    done
    [[ $ready == "ARMED $pid $pgid $ticks engine" ]] ||
        die "Qwen memory watchdog did not arm final process"
}

start_qwen38() {
    start_qwen_profile qwen38 "$QWEN_PROFILE_MANIFEST" verify_qwen_hashes
}

start_qwen38-1m() {
    start_qwen_profile qwen38-1m "$QWEN_1M_PROFILE_MANIFEST" \
        verify_qwen_1m_hashes
}

api_key() {
    local file=${DSV4_API_KEY_FILE:-/etc/deepseek-v4-flash/api-key}
    [[ -r $file ]] || return 1
    IFS= read -r REPLY <"$file"
    [[ $REPLY =~ ^[A-Za-z0-9._-]{16,512}$ ]]
}

verify_dsv4_context() {
    local body path="/slots"
    body=$(clean_curl -fsS --max-time 5 \
        "http://127.0.0.1:$PORT$path") || return 1
    clean_python - "$body" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
if not isinstance(value, list) or len(value) != 2:
    raise SystemExit("DeepSeek slot topology is invalid")
for slot in value:
    if slot["n_ctx"] != 393216:
        raise SystemExit("DeepSeek per-slot context is not 393216 (2 x 512k)")
PY
}

verify_qwen_1m_context() {
    local body path="/slots"
    body=$(clean_curl -fsS --max-time 5 \
        "http://127.0.0.1:$PORT$path") || return 1
    clean_python - "$body" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
if not isinstance(value, list) or len(value) != 4:
    raise SystemExit("Qwen 1M slot topology is invalid")
for slot in value:
    if slot["n_ctx"] != 262144:
        raise SystemExit("Qwen per-slot context is not 262144 (4 x 262K)")
PY
}

verify_laguna_context() {
    local body path="/slots"
    body=$(clean_curl -fsS --max-time 5 \
        "http://127.0.0.1:$PORT$path") || return 1
    clean_python - "$body" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
if not isinstance(value, list) or len(value) != 4:
    raise SystemExit("Laguna slot topology is invalid")
for slot in value:
    if slot["n_ctx"] != 98304:
        raise SystemExit("Laguna per-slot context is not 98304 (4 x 96K)")
PY
}

verify_qwen_process_ready() {
    local values pid expected_pgid expected_ticks expected_exe expected_unit
    local unit_pid identity pgid ticks live_exe sockets
    [[ -r $QWEN_PROCESS ]] || return 1
    values=$(clean_python - "$QWEN_PROCESS" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
print(value["pid"], value["pgid"], value["start_ticks"], value["unit"])
PY
    ) || return 1
    read -r pid expected_pgid expected_ticks expected_unit <<<"$values"
    [[ $expected_unit == "$QWEN_UNIT" ]] || return 1
    unit_pid=$(systemctl show "$QWEN_UNIT" --property=MainPID --value \
        2>/dev/null || true)
    [[ $unit_pid == "$pid" ]] || return 1
    identity=$(proc_identity "$pid" 2>/dev/null || true)
    [[ -n $identity ]] || return 1
    read -r pgid ticks <<<"$identity"
    [[ $pgid == "$expected_pgid" && $ticks == "$expected_ticks" ]] || return 1
    live_exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
    expected_exe=$(readlink -f "$QWEN_BINARY" 2>/dev/null || true)
    [[ -n $live_exe && $live_exe == "$expected_exe" ]] || return 1
    sockets=$(ss -H -ltnp "sport = :$PORT" 2>/dev/null) || return 1
    [[ -n $sockets && $sockets == *"127.0.0.1:$PORT"* &&
            $sockets == *"pid=$pid,"* ]]
}

verify_laguna_process_ready() {
    local values pid expected_pgid expected_ticks expected_unit
    local unit_pid identity pgid ticks live_exe expected_exe sockets
    [[ -r $LAGUNA_PROCESS ]] || return 1
    values=$(clean_python - "$LAGUNA_PROCESS" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
print(value["pid"], value["pgid"], value["start_ticks"], value["unit"])
PY
    ) || return 1
    read -r pid expected_pgid expected_ticks expected_unit <<<"$values"
    [[ $expected_unit == "$LAGUNA_UNIT" ]] || return 1
    unit_pid=$(systemctl show "$LAGUNA_UNIT" --property=MainPID --value \
        2>/dev/null || true)
    [[ $unit_pid == "$pid" ]] || return 1
    identity=$(proc_identity "$pid" 2>/dev/null || true)
    [[ -n $identity ]] || return 1
    read -r pgid ticks <<<"$identity"
    [[ $pgid == "$expected_pgid" && $ticks == "$expected_ticks" ]] || return 1
    live_exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
    expected_exe=$(readlink -f "$LAGUNA_BINARY" 2>/dev/null || true)
    [[ -n $live_exe && $live_exe == "$expected_exe" ]] || return 1
    sockets=$(ss -H -ltnp "sport = :$PORT" 2>/dev/null) || return 1
    [[ -n $sockets && $sockets == *"127.0.0.1:$PORT"* &&
            $sockets == *"pid=$pid,"* ]]
}

wait_model_ready() {
    local profile=$1 expected body deadline probe_count=0 available
    expected=deepseek-v4-flash
    [[ $profile == glm52 ]] && expected=glm-5.2
    [[ $profile == qwen38 || $profile == qwen38-1m ]] && expected=qwen3.8-27b
    [[ $profile == laguna ]] && expected=laguna-s-2.1
    deadline=$((SECONDS + 1800))
    while (( SECONDS < deadline )); do
        body=$(clean_curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" \
            2>/dev/null || true)
        if clean_python - "$expected" "$body" <<'PY' 2>/dev/null
import json, sys
expected, raw = sys.argv[1:]
value = json.loads(raw)
if not any(expected == item["id"].lower() for item in value["data"]):
    raise SystemExit("exact model identity mismatch")
PY
        then
            if [[ $profile != dsv4 ]] || verify_dsv4_context; then
                if [[ $profile != qwen38 && $profile != qwen38-1m ]] || { \
                        clean_curl -fsS --max-time 3 \
                            "http://127.0.0.1:$PORT/health" >/dev/null &&
                        verify_qwen_process_ready; }; then
                    if [[ $profile != qwen38-1m ]] || verify_qwen_1m_context; then
                        if [[ $profile != laguna ]] || { \
                                clean_curl -fsS --max-time 3 \
                                    "http://127.0.0.1:$PORT/health" >/dev/null &&
                                verify_laguna_process_ready &&
                                verify_laguna_context; }; then
                            return 0
                        fi
                    fi
                fi
            fi
        fi
        probe_count=$((probe_count + 1))
        if (( probe_count % 15 == 0 )); then
            available=$(awk '$1 == "MemAvailable:" {printf "%.2f", $2 / 1048576}' \
                /proc/meminfo)
            printf 'Waiting for %s load: MemAvailable=%s GiB\n' \
                "$profile" "${available:-unknown}" >&2
        fi
        sleep 2
    done
    return 1
}

verify_serving() {
    local profile=$1 expected unauth code key body
    expected=deepseek-v4-flash
    [[ $profile == glm52 ]] && expected=glm-5.2
    [[ $profile == qwen38 || $profile == qwen38-1m ]] && expected=qwen3.8-27b
    [[ $profile == laguna ]] && expected=laguna-s-2.1
    body=$(clean_curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models") ||
        return 1
    clean_python - "$expected" "$body" <<'PY'
import json, sys
expected=sys.argv[1]
value=json.loads(sys.argv[2])
if not any(expected == item["id"].lower() for item in value["data"]):
    raise SystemExit("exact model identity mismatch")
PY
    if [[ $profile == dsv4 ]]; then
        verify_dsv4_context || return 1
    fi
    if [[ $profile == qwen38 || $profile == qwen38-1m ]]; then
        clean_curl -fsS --max-time 5 \
            "http://127.0.0.1:$PORT/health" >/dev/null || return 1
        verify_qwen_process_ready || return 1
    fi
    if [[ $profile == qwen38-1m ]]; then
        verify_qwen_1m_context || return 1
    fi
    if [[ $profile == laguna ]]; then
        clean_curl -fsS --max-time 5 \
            "http://127.0.0.1:$PORT/health" >/dev/null || return 1
        verify_laguna_process_ready || return 1
        verify_laguna_context || return 1
    fi
    unauth=$(clean_curl -sS -o /dev/null -w '%{http_code}' --max-time 5 \
        "http://127.0.0.1:$AUTH_PORT/health" || true)
    [[ $unauth == 401 ]] || return 1
    api_key || return 1
    key=$REPLY
    code=$(
        printf 'header = "Authorization: Bearer %s"\n' "$key" |
            clean_curl --config - -sS -o "$STATE/probe.json.tmp" \
                -w '%{http_code}' --max-time 1800 \
                -H 'Content-Type: application/json' \
                -d '{"model":"default","messages":[{"role":"user","content":"Calculate 2+2. State the decimal answer clearly."}],"max_tokens":768,"temperature":0}' \
                "http://127.0.0.1:$AUTH_PORT/v1/chat/completions" || true
    )
    unset key REPLY
    [[ $code == 200 ]] || return 1
    clean_python - "$STATE/probe.json.tmp" <<'PY'
import json, sys
import re
with open(sys.argv[1], encoding="utf-8") as stream:
    value=json.load(stream)
message=value["choices"][0]["message"]
finish_reason=value["choices"][0]["finish_reason"]
if finish_reason not in {"stop", "length"}:
    raise SystemExit("semantic readiness finish reason is invalid")
usage=value["usage"]
if not isinstance(usage.get("completion_tokens"), int) or usage["completion_tokens"] < 1:
    raise SystemExit("semantic readiness completion count is invalid")
parts=[
    message.get(field)
    for field in ("reasoning_content", "content")
    if isinstance(message.get(field), str)
]
text="\n".join(parts)
if not text.strip() or re.search(r"(?<![0-9])4(?![0-9])", text) is None:
    raise SystemExit("semantic readiness probe failed")
PY
    mv -- "$STATE/probe.json.tmp" "$STATE/$profile.probe.json"
}

commit_active() {
    clean_python - "$ACTIVE.tmp" "$1" <<'PY'
import json, os, sys
with open(sys.argv[1], "x", encoding="utf-8") as stream:
    json.dump({"schema_version":1, "profile":sys.argv[2]}, stream)
    stream.flush(); os.fsync(stream.fileno())
PY
    mv -- "$ACTIVE.tmp" "$ACTIVE"
}

rollback() {
    local rc=$?
    "$rollback_needed" || return "$rc"
    rollback_needed=false
    stop_profile "${1:-}" || true
    if [[ -n $previous_profile ]]; then
        if "start_$previous_profile" && wait_model_ready "$previous_profile" &&
                verify_serving "$previous_profile"; then
            commit_active "$previous_profile"
        else
            echo "ROLLBACK VERIFICATION FAILED" >&2
        fi
    fi
    return "$rc"
}

restore_profile() {
    local profile=$1
    if [[ $profile == glm52 ]] && ! glm_qualified; then
        echo "recorded GLM-5.2 profile is no longer qualified" >&2
        return 1
    fi
    if [[ $profile == qwen38 ]]; then
        verify_qwen_hashes || return 1
        revalidate_qwen_identities || return 1
    fi
    if [[ $profile == qwen38-1m ]]; then
        verify_qwen_1m_hashes || return 1
        revalidate_qwen_identities || return 1
    fi
    if [[ $profile == laguna ]]; then
        verify_laguna_profile_hashes || return 1
        revalidate_laguna_identities || return 1
    fi
    verify_serving "$profile" && return 0
    if [[ $profile != dsv4 || -e /run/dsv4/llamacpp.state.json ]]; then
        stop_profile "$profile" || return 1
    fi
    "start_$profile" || return 1
    wait_model_ready "$profile" || return 1
    verify_serving "$profile"
}

# Exercise complete switch/restore control flow without loading a model. These
# overrides exist only in the explicit test mode and retain the real Qwen launch
# functions so their complete systemd-run argv is covered by subprocess tests.
if [[ ${ENGINE_SWITCH_TESTING:-0} == 1 ]]; then
    test_action() { printf '%s\n' "$*" >>"$STATE/actions.log"; }
    systemd-run() {
        test_action "SYSTEMD_RUN $*"
        if [[ $* == *laguna-engine* ]]; then
            [[ ! -e $STATE/fail-laguna-start ]] || return 1
            : >"$STATE/laguna-running"
        else
            [[ ! -e $STATE/fail-qwen-start ]] || return 1
            : >"$STATE/qwen-running"
        fi
    }
    systemctl() {
        local verb=${1:-} property=${2:-}
        if [[ $verb == show && $property == "$QWEN_UNIT" ]]; then
            if [[ $* == *--property=LoadState* ]]; then
                [[ -e $STATE/qwen-unit-killed ]] && printf 'loaded\n' || printf 'not-found\n'
            elif [[ $* == *--property=ActiveState* ]]; then
                [[ -e $STATE/qwen-unit-killed ]] && printf 'failed\n' || printf 'inactive\n'
            fi
            return 0
        fi
        test_action "SYSTEMCTL $*"
        if [[ $verb == reset-failed && $property == "$QWEN_UNIT" ]]; then
            rm -f -- "$STATE/qwen-unit-killed"
        fi
        if [[ $verb == show && $property == "$LAGUNA_UNIT" ]]; then
            if [[ $* == *--property=LoadState* ]]; then
                [[ -e $STATE/laguna-unit-killed ]] && printf 'loaded\n' || printf 'not-found\n'
            elif [[ $* == *--property=ActiveState* ]]; then
                [[ -e $STATE/laguna-unit-killed ]] && printf 'failed\n' || printf 'inactive\n'
            fi
            return 0
        fi
        if [[ $verb == reset-failed && $property == "$LAGUNA_UNIT" ]]; then
            rm -f -- "$STATE/laguna-unit-killed"
        fi
    }
    dsv4_launcher() {
        test_action "DSV4 $*"
        if [[ $1 == start ]]; then
            : >"$STATE/dsv4-running"
        else
            rm -f -- "$STATE/dsv4-running"
        fi
    }
    verify_qwen_profile_hashes() {
        local _manifest_path=$1 expected_profile=$2 _expected_context=$3
        if [[ ! -e $STATE/qwen-hashes-valid ]]; then
            echo "Qwen test artifact hashes are not approved" >&2
            return 1
        fi
        qwen_hashes_verified_profile=$expected_profile
        qwen_verified_identities=test
        test_action "HASHES $expected_profile"
    }
    revalidate_qwen_identities() {
        [[ $qwen_verified_identities == test ]]
    }
    verify_laguna_profile_hashes() {
        if [[ ! -e $STATE/laguna-hashes-valid ]]; then
            echo "Laguna test artifact hashes are not approved" >&2
            return 1
        fi
        laguna_hashes_verified=true
        laguna_verified_identities=test
        test_action "HASHES laguna"
    }
    revalidate_laguna_identities() {
        [[ $laguna_verified_identities == test ]]
    }
    stop_qwen_verified() {
        test_action "STOP qwen"
        rm -f -- "$STATE/qwen-running"
    }
    stop_laguna_verified() {
        test_action "STOP laguna"
        rm -f -- "$STATE/laguna-running"
    }
    start_qwen_profile() {
        local profile=$1 _manifest_path=$2 verify_function=$3
        [[ $qwen_hashes_verified_profile == "$profile" ]] || "$verify_function"
        cleanup_qwen_killed_unit
        "launch_$profile" || die "Qwen transient unit failed to start"
        test_action "START $profile"
    }
    start_laguna_profile() {
        "$laguna_hashes_verified" || verify_laguna_profile_hashes
        cleanup_laguna_killed_unit
        launch_laguna || die "Laguna transient unit failed to start"
        test_action "START laguna"
    }
    wait_model_ready() {
        test_action "WAIT $1"
        [[ -e $STATE/$([[ $1 == dsv4 ]] && printf dsv4 || \
            { [[ $1 == laguna ]] && printf laguna || printf qwen; })-running ]]
    }
    verify_serving() {
        test_action "VERIFY $1"
        [[ ! -e $STATE/fail-$1-verify ]] &&
            [[ -e $STATE/$([[ $1 == dsv4 ]] && printf dsv4 || \
                { [[ $1 == laguna ]] && printf laguna || printf qwen; })-running ]]
    }
fi

if [[ -n ${ENGINE_SWITCH_SOURCE_ONLY_FIXTURE_ROOT:-} ]]; then
    return 0
fi

if [[ ${1:-} == render ]]; then
    # Test-only: print the fully-assembled launch snapshot for one alias so
    # the conformance suite can compare it against the captured fixtures.
    # Read-only; takes no lock and touches no state.
    [[ ${ENGINE_SWITCH_TESTING:-0} == 1 ]] ||
        die 'render is a test-only verb (set ENGINE_SWITCH_TESTING=1)'
    [[ -n ${2:-} ]] || die 'usage: render <alias>'
    render_snapshot "$2"
    exit 0
fi

command=${1:-status}
if [[ $command == status ]]; then
    [[ ${2:-} == --json ]] && { json_status; exit 0; }
    json_status
    exit 0
fi
[[ $command == restore || $command == stop || $command == dsv4 || \
        $command == glm52 || $command == qwen38 || $command == qwen38-1m || \
        $command == laguna ]] ||
    die "usage: $0 status [--json]|stop|restore|dsv4|glm52|qwen38|qwen38-1m|laguna"
if [[ $command == stop ]]; then
    # Gate-window helper: stop the active engine but leave active.json
    # untouched so `restore` brings the same profile back afterwards.
    mkdir -p -- "$STATE"
    acquire_switch_lock
    stop_target=$(read_active_profile)
    [[ -n $stop_target ]] || exit 0
    ( stop_profile "$stop_target" ) ||
        die "cannot safely stop active profile $stop_target"
    echo "STOPPED $stop_target (active.json unchanged; '$0 restore' restarts it)"
    exit 0
fi
if [[ $command == restore ]]; then
    mkdir -p -- "$STATE"
    acquire_switch_lock
    command=$(read_active_profile)
    [[ -n $command ]] || exit 0
    if ( restore_profile "$command" ); then
        exit 0
    fi
    [[ $command != dsv4 ]] || die "dsv4 boot restoration failed"
    echo "RESTORE FAILED for recorded profile $command; falling back to dsv4" >&2
    ( stop_profile "$command" ) ||
        die "cannot safely stop failed $command profile before dsv4 fallback"
    if ( restore_profile dsv4 ); then
        commit_active dsv4
        echo "RESTORE FALLBACK committed dsv4 in active.json" >&2
        exit 0
    fi
    die "dsv4 boot restoration fallback failed after $command"
fi
if [[ $command == glm52 ]] && ! glm_qualified; then
    die "GLM-5.2 production profile is not qualified"
fi
if [[ $command == glm52 ]]; then
    verify_glm_hashes
fi
if [[ $command == qwen38 ]]; then
    verify_qwen_hashes
fi
if [[ $command == qwen38-1m ]]; then
    verify_qwen_1m_hashes
fi
if [[ $command == laguna ]]; then
    verify_laguna_profile_hashes
fi
mkdir -p -- "$STATE"
acquire_switch_lock
if [[ $command == qwen38 || $command == qwen38-1m ]]; then
    revalidate_qwen_identities
fi
if [[ $command == laguna ]]; then
    revalidate_laguna_identities
fi
previous_profile=$(read_active_profile)
if [[ $previous_profile == "$command" ]] && verify_serving "$command"; then
    exit 0
fi
rollback_needed=true
trap 'rollback "$command"' EXIT
stop_profile "$previous_profile"
"start_$command"
wait_model_ready "$command" || die "$command readiness timed out or model identity is wrong"
verify_serving "$command"
commit_active "$command"
rollback_needed=false
trap - EXIT
