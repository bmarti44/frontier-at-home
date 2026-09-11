"""Send the four prepared 250K requests concurrently; sample MemAvailable, top RSS, GPU mem every 5 s."""
import json, os, subprocess, sys, threading, time, urllib.request
D = os.path.expanduser("~/.cache/glm53-flash/context-direct-011")
stop = False
def mem():
    m = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
    return {k: round(m[k] / 2**20, 2) for k in ("MemAvailable", "MemFree", "Cached", "Shmem", "AnonPages", "Mapped")}
def top():
    out = subprocess.run(["ps", "-eo", "rss,pid,comm", "--sort=-rss"], text=True, capture_output=True).stdout.splitlines()[1:5]
    return [(round(int(l.split()[0]) / 2**20, 2), l.split()[2]) for l in out]
def gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], text=True, capture_output=True, timeout=5).stdout.strip()
    except Exception as e:
        return str(e)
def sampler():
    while not stop:
        print(f"t={time.time()-t0:6.0f}s {mem()} top={top()} gpu={gpu()}", flush=True); time.sleep(5)
t0 = time.time(); threading.Thread(target=sampler, daemon=True).start()
def one(i):
    req = json.load(open(f"{D}/{i}-request.json")); req["max_tokens"] = 48; req.pop("return_token_ids", None)
    r = urllib.request.Request("http://127.0.0.1:8015/v1/chat/completions", data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=7200) as resp:
            for line in resp:
                line = line.decode().strip()
                if line.startswith("data: {") and '"usage"' in line:
                    print(f"slot{i} usage {json.loads(line[5:]).get('usage')} wall={time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"slot{i} ERROR {e!r} wall={time.time()-t0:.0f}s", flush=True)
ts = [threading.Thread(target=one, args=(i,)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
stop = True; print(f"done wall={time.time()-t0:.0f}s", flush=True)
