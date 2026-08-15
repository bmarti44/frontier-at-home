#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

[[ $# == 3 ]] || { echo "usage: $0 OUT LABEL SEED" >&2; exit 2; }
OUT=$1
LABEL=$2
SEED=$3
REPO=/home/bmarti44/spark-deepseek-v4-flash
BINARY=${DSV4_MATCHED_BINARY:?DSV4_MATCHED_BINARY is required}
EXPECTED_BINARY_SHA256=${DSV4_MATCHED_BINARY_SHA256:?DSV4_MATCHED_BINARY_SHA256 is required}
MODEL=${DSV4_MATCHED_MODEL_FIRST:?DSV4_MATCHED_MODEL_FIRST is required}
SHARDS_JSON=${DSV4_MATCHED_SHARDS_JSON:?DSV4_MATCHED_SHARDS_JSON is required}
BENCH=${MATCHED_BENCH_PATH:?MATCHED_BENCH_PATH is required}
PYTHON=${MATCHED_PYTHON_PATH:?MATCHED_PYTHON_PATH is required}
TOKENIZER_NATIVE=${MATCHED_TOKENIZER_NATIVE_PATH:?MATCHED_TOKENIZER_NATIVE_PATH is required}
TOKENIZER_NATIVE_SHA256=${MATCHED_TOKENIZER_NATIVE_SHA256:?MATCHED_TOKENIZER_NATIVE_SHA256 is required}
PORT=${MATCHED_PORT:?MATCHED_PORT is required}
PID=
START_TICKS=

[[ -x $BINARY && ! -L $BINARY ]] || { echo "invalid DeepSeek binary" >&2; exit 2; }
actual_binary_sha256=$(sha256sum -- "$BINARY" | awk '{print $1}')
[[ $actual_binary_sha256 == "$EXPECTED_BINARY_SHA256" ]] || {
    echo "DeepSeek matched binary identity mismatch" >&2
    exit 2
}
[[ $PORT =~ ^[0-9]+$ ]] && (( 10#$PORT >= 1024 && 10#$PORT <= 65535 )) || {
    echo "invalid matched port" >&2
    exit 2
}

stop_server() {
    [[ ${PID:-} =~ ^[0-9]+$ ]] || return 0
    [[ $(awk '{print $22}' "/proc/$PID/stat" 2>/dev/null || true) == "$START_TICKS" ]] || return 0
    kill -TERM "$PID" 2>/dev/null || true
    for _ in $(seq 1 90); do
        [[ $(awk '{print $22}' "/proc/$PID/stat" 2>/dev/null || true) != "$START_TICKS" ]] && return 0
        sleep 1
    done
    kill -KILL "$PID" 2>/dev/null || true
}
trap stop_server EXIT

mkdir -p -- "$OUT"
record_shards() {
    local checkpoint=$1
    "$PYTHON" -I -B -S - "$SHARDS_JSON" "$OUT/model.shards.jsonl" "$checkpoint" <<'PY'
import json
import os
import pathlib
import sys
import time

expected = json.load(open(sys.argv[1], encoding="ascii"))["dsv4_shards"]
rows = []
for shard in expected:
    path = pathlib.Path(shard["path"])
    info = path.stat()
    observed = {
        "path": str(path),
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
    }
    for field in ("path", "device", "inode", "bytes"):
        if observed[field] != shard[field]:
            raise SystemExit(f"DeepSeek shard identity changed at {sys.argv[3]}: {path}")
    rows.append(observed)
record = {"checkpoint": sys.argv[3], "monotonic_ns": time.monotonic_ns(), "shards": rows}
fd = os.open(sys.argv[2], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
with os.fdopen(fd, "a", encoding="ascii") as stream:
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
}

record_shards prelaunch
LD_LIBRARY_PATH=$(dirname -- "$BINARY") \
    "$BINARY" --model "$MODEL" --alias deepseek-v4-flash \
    --host 127.0.0.1 --port "$PORT" -c 32768 -np 1 -ngl 999 \
    -b 2048 -ub 512 --no-warmup --cache-ram 0 --no-mmap \
    --no-direct-io --no-cache-prompt --spec-type ngram-map-k4v \
    >"$OUT/server.log" 2>&1 &
PID=$!
START_TICKS=$(awk '{print $22}' "/proc/$PID/stat")

ready=false
for _ in $(seq 1 900); do
    if [[ $(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 \
        "http://127.0.0.1:$PORT/v1/models" || true) == 200 ]]; then
        ready=true
        break
    fi
    [[ $(awk '{print $22}' "/proc/$PID/stat" 2>/dev/null || true) == "$START_TICKS" ]] || break
    sleep 1
done
"$ready" || { tail -100 "$OUT/server.log" >&2; exit 1; }
record_shards ready

"$PYTHON" -I -B -S - "$PID" "$MODEL" "$OUT" "$actual_binary_sha256" <<'PY'
import hashlib
import json
import pathlib
import sys

pid = int(sys.argv[1])
model = pathlib.Path(sys.argv[2])
out = pathlib.Path(sys.argv[3])
binary_sha256 = sys.argv[4]
boot_id = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
stat_fields = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
argv = [value.decode("utf-8") for value in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if value]
live_digest = hashlib.sha256(pathlib.Path(f"/proc/{pid}/exe").read_bytes()).hexdigest()
if live_digest != binary_sha256:
    raise SystemExit("live DeepSeek binary does not match the frozen binary")
live_info = pathlib.Path(f"/proc/{pid}/exe").stat()
binary_dir = pathlib.Path(argv[0]).resolve().parent
mapped = {}
for line in pathlib.Path(f"/proc/{pid}/maps").read_text().splitlines():
    fields = line.split()
    if len(fields) < 6 or not fields[-1].startswith("/"):
        continue
    path = pathlib.Path(fields[-1]).resolve()
    if path.parent != binary_dir or not path.is_file():
        continue
    mapped[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
if str(pathlib.Path(argv[0]).resolve()) not in mapped:
    mapped[str(pathlib.Path(argv[0]).resolve())] = live_digest
model_info = model.stat()
model_identity = f"{model_info.st_dev}:{model_info.st_ino}:{model_info.st_size}"
(out / "process.identity.json").write_text(
    json.dumps(
        {
            "boot_id": boot_id,
            "server_pid": pid,
            "server_start_ticks": int(stat_fields[21]),
        },
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n",
    encoding="ascii",
)
(out / "process.command").write_text(
    json.dumps(
        {
            "argv": argv,
            "binary_sha256": binary_sha256,
            "binary_device_inode": f"{live_info.st_dev}:{live_info.st_ino}",
            "context_cap": 32768,
            "model_device_inode_size": model_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n",
    encoding="ascii",
)
(out / "model.device-inode-size").write_text(model_identity + "\n", encoding="ascii")
(out / "process.runtime-closure.json").write_text(
    json.dumps(mapped, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="ascii",
)
PY

env MATCHED_TOKENIZER_NATIVE_PATH="$TOKENIZER_NATIVE" \
    MATCHED_TOKENIZER_NATIVE_SHA256="$TOKENIZER_NATIVE_SHA256" \
    "$PYTHON" -I -B -S "$BENCH" \
    --base-url "http://127.0.0.1:$PORT" \
    --out "$OUT/result.json" --stack-label "$LABEL" \
    --reps 2 --context-levels 0,28672 --max-tokens 160 \
    --min-completion-tokens 128 --request-timeout 2700 \
    --seed "$SEED" --ignore-eos-supported \
    --prompt-count-log "$OUT/server.log" --prompt-count-format llama

[[ $(awk '{print $22}' "/proc/$PID/stat" 2>/dev/null || true) == "$START_TICKS" ]] || {
    echo "DeepSeek server identity changed after requests" >&2
    exit 1
}
post_status=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 \
    "http://127.0.0.1:$PORT/v1/models" || true)
[[ $post_status == 200 ]] || { echo "DeepSeek post-request health failed" >&2; exit 1; }
record_shards post_requests
"$PYTHON" -I -B -S - "$OUT/process.observations.json" "$PID" "$START_TICKS" "$post_status" <<'PY'
import json
import os
import sys
import time

record = {
    "readiness_http_status": 200,
    "post_requests_http_status": int(sys.argv[4]),
    "server_pid": int(sys.argv[2]),
    "server_start_ticks": int(sys.argv[3]),
    "recorded_monotonic_ns": time.monotonic_ns(),
}
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="ascii") as stream:
    json.dump(record, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY

stop_server
trap - EXIT
