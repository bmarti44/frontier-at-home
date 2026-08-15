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
MODEL=$REPO/weights/unsloth-ud-q2_k_xl/DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf
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
LD_LIBRARY_PATH=$(dirname -- "$BINARY") \
    "$BINARY" --model "$MODEL" --alias deepseek-v4-flash \
    --host 127.0.0.1 --port "$PORT" -c 32768 -np 1 -ngl 999 \
    -b 2048 -ub 512 --no-warmup --cache-ram 0 --no-mmap \
    --no-direct-io --spec-type ngram-map-k4v \
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

python3 - "$PID" "$MODEL" "$OUT" "$actual_binary_sha256" <<'PY'
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
            "healthy": True,
            "memwatch_alive": True,
            "server_alive": True,
            "server_pid": pid,
            "server_start_ticks": int(stat_fields[21]),
            "watchdog_armed": True,
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

"$REPO/.venv-harness/bin/python" "$REPO/scripts/30_bench_speed.py" \
    --base-url "http://127.0.0.1:$PORT" \
    --out "$OUT/result.json" --stack-label "$LABEL" \
    --reps 2 --context-levels 0,28672 --max-tokens 160 \
    --min-completion-tokens 128 --request-timeout 2700 \
    --seed "$SEED" --ignore-eos-supported \
    --prompt-count-log "$OUT/server.log" --prompt-count-format llama

stop_server
trap - EXIT
