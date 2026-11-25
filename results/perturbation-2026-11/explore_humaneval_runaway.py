"""EXPLORATORY, written after both arms ran; not part of the preregistered reading (PREREG "Analysis").

Usage: python3 -B explore_humaneval_runaway.py --repo R [--min-available-gib 10] <out.json>

Inkling never ends a raw HumanEval completion: every one runs to the 16,384-token cap, writing numbered
copies of the function after the answer. Protocol v5's extractor applies the classic stop list only when
the plain splice fails to parse, so when the cut-off text happens to parse, the extra code runs with the
answer. For every failed HumanEval job of either model whose plain splice parsed, this re-runs the
bench's own sandbox on the completion cut at the first classic stop sequence, so the first function is
scored alone. One container at a time, each start waiting for MemAvailable >= the floor.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import tempfile
import time
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
from run_perturb import load_bench, mem_available_gib, utc_now  # noqa: E402

CLASSIC_STOPS = ("\ndef ", "\nclass ", "\nif __name__", "\nprint(")


def cut(completion: str) -> str:
    positions = [p for p in (completion.find(s) for s in CLASSIC_STOPS) if p >= 0]
    return completion[:min(positions)] if positions else completion


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--min-available-gib", type=float, default=10.0)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    bench, _ = load_bench(a.repo)
    items = {}
    for line in (here / "items.jsonl").read_text().splitlines():
        job = json.loads(line)
        if job["suite"] == "humaneval":
            items[job["job"]] = job["item"]
    results = []
    with tempfile.TemporaryDirectory(prefix="perturb-runaway-") as tmp:
        for model in ("inkling2mini", "qwen38fn"):
            for path in sorted((here / "runs" / model / "transcripts").glob("humaneval_*.json")):
                record = json.loads(path.read_text())
                if record["scored_correct"]:
                    continue
                item = items[record["job"]]
                try:
                    ast.parse(item["prompt"] + record["completion"])
                except SyntaxError:
                    continue  # the extractor already cut this one at the stop list
                waited = 0
                while mem_available_gib() < a.min_available_gib:
                    if waited >= 600:
                        raise SystemExit(f"MemAvailable stayed below {a.min_available_gib} GiB for 600 s")
                    time.sleep(5)
                    waited += 5
                ok, _, reason, _ = bench.run_humaneval(item, cut(record["completion"]), Path(tmp),
                                                       f"{model}-{path.stem}")
                results.append({"model": model, "job": record["job"], "finish_reason": record.get("finish_reason"),
                                "original_reason": record["reason"], "first_function_passed": ok,
                                "first_function_reason": reason})
                print(f"{model} {record['job']} first function: {reason}", flush=True)
    summary = {"finished_at": utc_now(), "label": "exploratory, post hoc", "stops": CLASSIC_STOPS,
               "rescored": len(results), "results": results}
    for model in ("inkling2mini", "qwen38fn"):
        mine = [r for r in results if r["model"] == model]
        summary[model] = {arm: {"rescored": sum(r["job"].split(":")[2] == arm for r in mine),
                                "first_function_passed": sum(r["job"].split(":")[2] == arm and r["first_function_passed"]
                                                             for r in mine)} for arm in ("O", "P")}
    a.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({m: summary[m] for m in ("inkling2mini", "qwen38fn")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
