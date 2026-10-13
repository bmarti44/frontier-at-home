#!/usr/bin/env python3
"""Show verdict.py rejects malformed or broken evidence (PREREG.md, Scorer).

Builds a synthetic passing evidence directory, checks it scores FITS, then
applies one mutation at a time and checks the expected verdict. Prints one
line per case and exits nonzero on any mismatch.
"""
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("smoke_verdict", HERE / "verdict.py")
verdict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verdict)

GIB = 2**20  # kB per GiB


def meminfo_line(t, avail_gib, free_gib, swap_used_kb=0):
    return (f"2026-10-13T04:{t // 60:02d}:{t % 60:02d}Z MemFree:={int(free_gib * GIB)} "
            f"MemAvailable:={int(avail_gib * GIB)} SwapTotal:=16777212 "
            f"SwapFree:={16777212 - swap_used_kb} ")


def build(root: Path, peak=100.0):
    obs = root / "observe"
    obs.mkdir(parents=True)
    rows = []
    for t in range(600):
        avail = 115.0 if t < 10 else 115.0 - peak if t < 400 else 115.0
        rows.append(meminfo_line(t, avail, avail))
    (obs / "meminfo.log").write_text("\n".join(rows) + "\n")
    (obs / "driver.log").write_text("\n".join([
        "2026-10-13T04:00:05Z start 1: launching in scope mimo26-smoke-1-9",
        "2026-10-13T04:02:00Z start 1: ready after 115 s",
        "2026-10-13T04:06:50Z mimo26-smoke-1-9.scope: scope empty after stop",
        "2026-10-13T04:08:00Z start 2: launching in scope mimo26-smoke-2-9",
        "2026-10-13T04:09:00Z start 2: ready after 60 s",
        "2026-10-13T04:09:30Z smoke window closed",
        "2026-10-13T04:40:00Z fidelity rc=0",
    ]) + "\n")
    for name in ("kernel-smoke.log", "kernel-tail.log", "memwatch.log", "server-start1.log",
                 "server-start2.log"):
        (obs / name).write_text("ok\n")
    (root / "kit/toolcall").mkdir(parents=True)
    (root / "kit/toolcall/toolcall.json").write_text(json.dumps({"passed": 20, "total": 20}))
    (root / "kit/speed").mkdir(parents=True)
    rep = {"valid": True, "client_completion_tokens": 512, "token_timestamps_ns": [0, 10**10],
           "timing_source": "sse_content_events", "ttft_s": 0.5, "prompt_tokens": 20}
    (root / "kit/speed/speed.json").write_text(json.dumps({"cells": [
        {"ctx_tokens": 0, "reps": [rep]}, {"ctx_tokens": 28672, "reps": [rep]}]}))
    probe = {"greedy": [{"sha256": "a"}, {"sha256": "b"}, {"sha256": "c"}]}
    for n in (1, 2):
        (root / f"probe-start{n}.json").write_text(json.dumps(probe))
    (root / "fidelity").mkdir()
    (root / "fidelity/model_diff.log").write_text(
        " -- A perplexity:  6.20000000\n -- B perplexity:  6.00000000\n"
        " -- A label in top-K:\n      K = 1: 0.5000\n"
        " -- B label in top-K:\n      K = 1: 0.5100\n"
        " -- Top-K agreement, A vs B:\n      K = 1: 0.9000\n"
        " -- KL divergence (A, B):  0.10000000\n -- KL divergence (B, A):  0.11000000\n")
    weights = root / "weights.json"
    weights.write_text(json.dumps({"artifacts": {"exl3-2.52-mikeroz": {"files": [
        {"name": "model-00001-of-00001.safetensors", "bytes": 90 * 2**30},
        {"name": "config.json", "bytes": 10}]}}}))
    return weights


def append(path, text):
    with open(path, "a") as f:
        f.write(text)


CASES = [
    ("baseline", None, "FITS", "MEASURED"),
    ("kernel NVRM", lambda r: append(r / "observe/kernel-smoke.log", "NVRM: Xid 31\n"), "DOES_NOT_FIT", None),
    ("memwatch fail-closed", lambda r: append(r / "observe/memwatch.log", "FAIL_CLOSED reason=x\n"),
     "DOES_NOT_FIT", None),
    ("NVRM after the smoke window", lambda r: append(r / "observe/kernel-tail.log", "NVRM: Xid 31\n"),
     "FITS", "MEASURED"),
    ("sampler gap during load", lambda r: (r / "observe/meminfo.log").write_text("\n".join(
        l for i, l in enumerate((r / "observe/meminfo.log").read_text().splitlines()) if not 60 <= i < 70)
        + "\n"), "NO_RESULT", None),
    ("server CUDA OOM", lambda r: append(r / "observe/server-start2.log", "torch.OutOfMemoryError\n"),
     "DOES_NOT_FIT", None),
    ("memwatch breach", lambda r: append(r / "observe/memwatch.log", "BREACH 7.9\n"), "DOES_NOT_FIT", None),
    ("peak above 106", "peak107", "DOES_NOT_FIT", None),
    ("swap growth", lambda r: append(r / "observe/meminfo.log", meminfo_line(599, 115, 115, 300000) + "\n"),
     "DOES_NOT_FIT", None),
    ("kernel log missing", lambda r: (r / "observe/kernel-smoke.log").unlink(), "NO_RESULT", None),
    ("one server log missing", lambda r: (r / "observe/server-start2.log").unlink(), "NO_RESULT", None),
    ("short meminfo", lambda r: (r / "observe/meminfo.log").write_text(meminfo_line(0, 115, 115) + "\n"),
     "NO_RESULT", None),
    ("toolcall 18/20", lambda r: (r / "kit/toolcall/toolcall.json").write_text(
        json.dumps({"passed": 18, "total": 20})), "FITS_TOOLCALL_LOW", None),
    ("toolcall total 19", lambda r: (r / "kit/toolcall/toolcall.json").write_text(
        json.dumps({"passed": 19, "total": 19})), "NO_RESULT", None),
    ("speed malformed", lambda r: (r / "kit/speed/speed.json").write_text("{"), "NO_RESULT", None),
    ("no eligible rep", lambda r: (r / "kit/speed/speed.json").write_text(json.dumps({"cells": [
        {"ctx_tokens": 0, "reps": [{"valid": False, "error": "timeout"}]}]})), "NO_RESULT", None),
    ("failed attempt", lambda r: append(r / "observe/driver.log",
                                        "2026-10-13T04:10:00Z FAILED_ATTEMPT start 2: x\n"), "NO_RESULT", None),
    ("start 2 never ready", lambda r: (r / "observe/driver.log").write_text(
        "\n".join(l for l in (r / "observe/driver.log").read_text().splitlines()
                  if "start 2: ready" not in l) + "\n"), "NO_RESULT", None),
    ("probe missing", lambda r: (r / "probe-start2.json").unlink(), "NO_RESULT", None),
    ("fidelity log missing", lambda r: (r / "fidelity/model_diff.log").unlink(), "FITS", "NO_RESULT"),
    ("fidelity run failed", lambda r: append(r / "observe/driver.log",
                                             "2026-10-13T04:41:00Z FIDELITY_FAILED rc=137\n"), "FITS", "NO_RESULT"),
    ("fidelity nan", lambda r: (r / "fidelity/model_diff.log").write_text(
        (r / "fidelity/model_diff.log").read_text().replace("0.10000000", "nan")), "FITS", "NO_RESULT"),
    ("fidelity duplicated", lambda r: append(r / "fidelity/model_diff.log", " -- A perplexity:  1.00000000\n"),
     "FITS", "NO_RESULT"),
]


def main():
    bad = 0
    for label, mutate, want, want_fid in CASES:
        root = Path(tempfile.mkdtemp(prefix="mimo-smoke-mut-"))
        try:
            weights = build(root, peak=107.0 if mutate == "peak107" else 100.0)
            if callable(mutate):
                mutate(root)
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                verdict.main(root, weights)
            summary = json.loads((root / "summary.json").read_text())
            ok = summary["verdict"] == want and (want_fid is None or summary["fidelity_status"] == want_fid)
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {label}: {summary['verdict']} / fidelity "
                  f"{summary['fidelity_status']} (want {want}{' / ' + want_fid if want_fid else ''})")
        finally:
            shutil.rmtree(root)
    print(f"{len(CASES) - bad}/{len(CASES)} cases as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
