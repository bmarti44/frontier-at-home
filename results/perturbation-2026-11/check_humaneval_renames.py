"""Sandbox check of the HumanEval renames (PREREG "HumanEval", review A-L4).

Usage: python3 -B check_humaneval_renames.py --repo R [--min-available-gib 10] <rows/humaneval.jsonl>
                                             <humaneval-names.json> <out.json>

For every task, the bench's own sandbox (`run_humaneval`) executes the canonical solution twice: the
original row (a control that separates sandbox trouble from a bad rename) and the renamed row built
by `build_humaneval.build`. A renamed row that fails while its original passes leaves both arms.
One container at a time, each start waiting for MemAvailable >= the floor.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_humaneval  # noqa: E402
from run_perturb import load_bench, mem_available_gib, utc_now  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--min-available-gib", type=float, default=10.0)
    ap.add_argument("rows", type=Path)
    ap.add_argument("names", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    bench, _ = load_bench(a.repo)
    rows = [json.loads(line) for line in a.rows.read_text().splitlines()]
    names = json.loads(a.names.read_text())
    results = []
    with tempfile.TemporaryDirectory(prefix="perturb-rename-") as tmp:
        for n, row in enumerate(rows):
            renamed, problems = build_humaneval.build(row, names[row["task_id"].split("/")[1]])
            entry = {"task_id": row["task_id"], "from": row["entry_point"], "to": renamed["entry_point"],
                     "static_problems": problems}
            for arm, r in (("original", row), ("renamed", renamed)):
                waited = 0
                while mem_available_gib() < a.min_available_gib:
                    if waited >= 600:
                        raise SystemExit(f"MemAvailable stayed below {a.min_available_gib} GiB for 600 s")
                    time.sleep(5)
                    waited += 5
                ok, _, reason, _ = bench.run_humaneval(r, r["canonical_solution"], Path(tmp), f"{arm}-{n:03d}")
                entry[arm] = {"passed": ok, "reason": reason}
            results.append(entry)
            print(f"[{n + 1}/{len(rows)}] {row['task_id']} original={entry['original']['reason']} "
                  f"renamed={entry['renamed']['reason']}", flush=True)
    excluded = [e["task_id"] for e in results if e["original"]["passed"] and not e["renamed"]["passed"]]
    infra = [e["task_id"] for e in results if not e["original"]["passed"]]
    summary = {"finished_at": utc_now(), "tasks": len(results),
               "renamed_pass": sum(e["renamed"]["passed"] for e in results),
               "original_pass": sum(e["original"]["passed"] for e in results),
               "excluded_by_rename": excluded, "original_failed": infra,
               "static_problems": [e["task_id"] for e in results if e["static_problems"]], "results": results}
    a.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"renamed pass {summary['renamed_pass']}/{len(results)}, original pass {summary['original_pass']}, "
          f"excluded {excluded}, original failed {infra}")
    return 0 if not infra and not summary["static_problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
