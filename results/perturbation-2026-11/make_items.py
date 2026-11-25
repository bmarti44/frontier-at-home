"""Assemble the job list: the original (O) and perturbed (P) arms of every kept item, in one seeded shuffle.

Usage: python3 -B make_items.py <dir with rows/ and variants/> <seed> <out items.jsonl>

Item sets are the rows both models were already scored on: GSM8K dev + holdout (200),
MMLU-Pro dev + holdout (500), HumanEval (164). An item excluded from its variant set (a GSM8K row
without a faithful template, an MMLU-Pro row with positional options) is dropped from both arms.
GSM8K has 2 variants per row; MMLU-Pro and HumanEval have 1.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

FIELDS = {"gsm8k": ("question", "answer"),
          "mmlu-pro": ("question_id", "question", "options", "answer", "category"),
          "humaneval": ("task_id", "prompt", "canonical_solution", "test", "entry_point")}


def bench_row(suite: str, row: dict) -> dict:
    return {k: row[k] for k in FIELDS[suite]}


def main() -> int:
    base, seed, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    jobs = []
    variant_files = {"gsm8k": "gsm8k-variants.jsonl", "mmlu-pro": "mmlu-pro-shuffled.jsonl"}
    for suite in ("gsm8k", "mmlu-pro", "humaneval"):
        rows = [json.loads(line) for line in (base / "rows" / f"{suite}.jsonl").read_text().splitlines()]
        if suite == "humaneval":
            renamed = [json.loads(line) for line in (base / "variants/humaneval-renamed.jsonl").read_text().splitlines()]
            variants = [{"split": r["split"], "row": r["row"], "variant": 0, "perturbation": r["perturbation"],
                         "item": bench_row(suite, r)} for r in renamed]
        else:
            variants = [json.loads(line) for line in (base / "variants" / variant_files[suite]).read_text().splitlines()]
        by_row: dict[int, list[dict]] = {}
        for v in variants:
            by_row.setdefault(v["row"], []).append(v)
        for r in rows:
            if r["row"] not in by_row:
                continue
            jobs.append({"job": f"{suite}:{r['row']}:O:0", "suite": suite, "row": r["row"], "split": r["split"],
                         "arm": "O", "variant": 0, "item": bench_row(suite, r)})
            for v in sorted(by_row[r["row"]], key=lambda x: x["variant"]):
                assert v["split"] == r["split"]
                jobs.append({"job": f"{suite}:{r['row']}:P:{v['variant']}", "suite": suite, "row": r["row"],
                             "split": r["split"], "arm": "P", "variant": v["variant"],
                             "perturbation": v["perturbation"], "item": v["item"]})
    random.Random(hashlib.sha256(f"{seed}:order".encode()).hexdigest()).shuffle(jobs)
    with open(out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(Counter((j["suite"], j["arm"]) for j in jobs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
