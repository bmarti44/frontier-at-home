"""Send one prepared 250K request; sample MemAvailable + per-process RSS every 5 s."""
import json, os, subprocess, sys, threading, time, urllib.request
req = json.load(open(os.path.expanduser("~/.cache/glm53-flash/context-direct-011/0-request.json")))
req["max_tokens"] = 48; req.pop("return_token_ids", None)
key = open(sys.argv[1]).read().strip()
stop = False
def mem():
    m = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
    return m['MemAvailable'] / 2**20
def rss():
    out = subprocess.run(["ps", "-eo", "pid,rss,args"], text=True, capture_output=True).stdout
    rows = []
    for l in out.splitlines():
        if ("api_server" in l or "EngineCore" in l or "VLLM" in l) and "grep" not in l and "one_250k" not in l:
            p = l.split(None, 2); rows.append((p[0], round(int(p[1]) / 2**20, 2), p[2][:40]))
    return rows
def sampler():
    while not stop:
        print(f"t={time.time()-t0:6.0f}s avail={mem():6.2f}GiB rss={rss()}", flush=True); time.sleep(5)
t0 = time.time(); threading.Thread(target=sampler, daemon=True).start()
r = urllib.request.Request("http://127.0.0.1:8015/v1/chat/completions", data=json.dumps(req).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
first = None; text = []
with urllib.request.urlopen(r, timeout=7200) as resp:
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]": continue
        ev = json.loads(line[5:])
        for c in ev.get("choices", []):
            d = c.get("delta", {})
            if d.get("content") or d.get("reasoning_content"):
                if first is None: first = time.time() - t0; print(f"TTFT={first:.1f}s", flush=True)
                text.append(d.get("content") or d.get("reasoning_content"))
        if ev.get("usage"): print("usage", ev["usage"], flush=True)
stop = True
print(f"done wall={time.time()-t0:.1f}s ttft={first} text={''.join(text)[:200]!r}", flush=True)
