import json, sys
from pathlib import Path
ev = Path(sys.argv[1]); mut = sys.argv[2]
(ev/"observe").mkdir(parents=True); (ev/"kit/speed").mkdir(parents=True); (ev/"kit/toolcall").mkdir(parents=True)
swapfree = lambda i: 16777212 - (400000 if mut == "swap" and i > 50 else 0)
(ev/"observe/meminfo.log").write_text("".join(
    f"t MemAvailable:={110000000 - i*1000} SwapTotal:=16777212 SwapFree:={swapfree(i)} \n" for i in range(100)))
(ev/"observe/kernel.log").write_text("x NVRM: Xid 31\n" if mut == "xid" else "")
(ev/"observe/memwatch.log").write_text("ts=1 START\n")
(ev/"observe/server-start1.log").write_text("INFO ready\n")
(ev/"observe/driver.log").write_text("ok\n")
(ev/"kit/toolcall/toolcall.json").write_text(json.dumps({"passed": 15 if mut == "tool" else 19, "total": 20}))
def rep(tps, ctx):
    n = 384; ts = [int(i * (n/150) / tps * 1e9) for i in range(150)]
    err = "early stop: 150 timestamped tokens, minimum is 200"
    if mut == "mismatch": err = "timestamp/client token mismatch: x"
    return {"valid": False, "error": err, "client_completion_tokens": n, "token_timestamps_ns": ts,
            "timing_source": "sse_content_events", "ttft_s": max(ctx, 100)/1600, "prompt_tokens": max(ctx, 100)}
slow = 10 if mut == "slow" else 40
cells = [{"ctx_tokens": c, "reps": [rep(slow, c), rep(slow, c)], "median_decode": None} for c in (0, 28672)]
if mut == "slow":
    for c in cells:
        for r in c["reps"]: r["ttft_s"] = r["prompt_tokens"] / 500
(ev/"kit/speed/speed.json").write_text(json.dumps({"cells": cells, "suite_valid": False}))
probe = {"greedy": [{"sha256": "a"}], "granularity": []}
for n in (1, 2):
    if not (mut == "noprobe" and n == 2):
        (ev/f"probe-start{n}.json").write_text(json.dumps(probe))
