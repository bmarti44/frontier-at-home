"""Score the perturbation experiment (rules fixed in PREREG.md before any model run).

Usage: python3 -B analyze.py --items items.jsonl --model qwen38fn=DIR --model inkling2mini=DIR --out summary.json

Per model and suite, over the items both arms completed:
  O_i = 1 if the original was answered correctly, P_i = the mean over the item's variants;
  drop = mean(O_i - P_i) in percentage points, with a 95% paired bootstrap interval over items.
Between models, on the items both models completed:
  difference = drop(first model) - drop(second model), bootstrapping items jointly.
Readings: a model "drops" when its interval is above 0; the first model "drops more" when the
difference interval is above 0, the second when it is below 0; otherwise no detectable difference.
A suite is NO_RESULT for a model when more than 2% of its jobs end in a transport error or a
sandbox infrastructure failure, or any job is missing. Below that, an item with any such job leaves
both arms and the between-model set, and is listed.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

B = 10000
BOOT_SEED = 20261123
INFRA = re.compile(r"^invalid:|^pending sandbox$|^sandbox launch failed|^sandbox failed with exit (125|126|127)$")
SUITES = ("gsm8k", "mmlu-pro", "humaneval")


def load(items_path: Path, out_dir: Path):
    jobs = [json.loads(line) for line in items_path.read_text().splitlines()]
    rec = {}
    for p in (out_dir / "transcripts").glob("*.json"):
        r = json.loads(p.read_text())
        rec[r["job"]] = r
    return jobs, rec


def per_item(jobs, rec, suite):
    """Return ({row: (O, P)}, missing, infra, n_jobs, tokens, infra_rows, letters).

    An item with any infrastructure-failed job leaves both arms (review A-M2)."""
    rows: dict[int, dict] = {}
    missing = infra = n = 0
    infra_rows: set[int] = set()
    letters: dict[str, dict[str, list[float]]] = {"O": {}, "P": {}}
    tokens = {"O": [], "P": []}
    for j in jobs:
        if j["suite"] != suite:
            continue
        n += 1
        r = rec.get(j["job"])
        if r is None:
            missing += 1
            continue
        # A timeout is the model running out the clock: scored wrong like the holdout, never dropped (A-L1).
        if INFRA.search(r["reason"] or "") and not r.get("timed_out"):
            infra += 1
            infra_rows.add(j["row"])
        slot = rows.setdefault(j["row"], {"O": None, "P": []})
        ok = 1.0 if r.get("scored_correct") else 0.0
        if suite == "mmlu-pro":
            letters[j["arm"]].setdefault(j["item"]["answer"].upper(), []).append(ok)
        if j["arm"] == "O":
            slot["O"] = ok
        else:
            slot["P"].append(ok)
        usage = r.get("usage") or {}
        if isinstance(usage.get("completion_tokens"), int):
            tokens[j["arm"]].append(usage["completion_tokens"])
    complete = {k: (v["O"], sum(v["P"]) / len(v["P"])) for k, v in rows.items()
                if v["O"] is not None and v["P"] and k not in infra_rows}
    return complete, missing, infra, n, tokens, infra_rows, letters


def boot_ci(diffs: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return means[int(0.025 * B)], means[int(0.975 * B) - 1]


def reading(lo: float, hi: float, above: str, below: str) -> str:
    return above if lo > 0 else below if hi < 0 else "no detectable difference"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="name=out_dir; the first is the model under suspicion")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    models = [m.split("=", 1) for m in a.model]
    summary = {"bootstrap_resamples": B, "bootstrap_seed": BOOT_SEED, "suites": {}}
    for s_i, suite in enumerate(SUITES):
        entry = {"models": {}}
        data = {}
        for name, d in models:
            jobs, rec = load(a.items, Path(d))
            complete, missing, infra, n, tokens, infra_rows, letters = per_item(jobs, rec, suite)
            no_result = missing > 0 or infra > 0.02 * n
            diffs = [o - p for o, p in complete.values()]
            lo, hi = boot_ci(diffs, BOOT_SEED + s_i) if diffs else (float("nan"), float("nan"))
            entry["models"][name] = {
                "jobs": n, "missing": missing, "infra_failures": infra,
                "items_dropped_for_infra": sorted(infra_rows), "items": len(complete),
                "verdict": "NO_RESULT" if no_result else reading(lo, hi, "drops", "improves"),
                "original_accuracy": sum(o for o, _ in complete.values()) / len(complete) if complete else None,
                "perturbed_accuracy": sum(p for _, p in complete.values()) / len(complete) if complete else None,
                "drop_pp": 100 * statistics.fmean(diffs) if diffs else None,
                "drop_ci95_pp": [100 * lo, 100 * hi],
                "median_completion_tokens": {arm: statistics.median(v) if v else None for arm, v in tokens.items()},
            }
            if suite == "mmlu-pro":
                # Reported, not read (review A-L3): the shuffle evens out the correct letter's position.
                entry["models"][name]["accuracy_by_correct_letter"] = {
                    arm: {k: [round(sum(v) / len(v), 4), len(v)] for k, v in sorted(by.items())}
                    for arm, by in letters.items()}
            data[name] = (complete, no_result)
        (n1, _), (n2, _) = models[0], models[1]
        c1, bad1 = data[n1]
        c2, bad2 = data[n2]
        common = sorted(set(c1) & set(c2))
        diffs = [(c1[k][0] - c1[k][1]) - (c2[k][0] - c2[k][1]) for k in common]
        lo, hi = boot_ci(diffs, BOOT_SEED + 100 + s_i) if diffs else (float("nan"), float("nan"))
        entry["difference"] = {
            "definition": f"drop({n1}) - drop({n2}) on common items",
            "items": len(common),
            "pp": 100 * statistics.fmean(diffs) if diffs else None,
            "ci95_pp": [100 * lo, 100 * hi],
            "verdict": "NO_RESULT" if (bad1 or bad2) else reading(lo, hi, f"{n1} drops more", f"{n2} drops more"),
        }
        summary["suites"][suite] = entry
    a.out.write_text(json.dumps(summary, indent=1))
    for suite, e in summary["suites"].items():
        print(f"== {suite}")
        for name, m in e["models"].items():
            if m["items"]:
                print(f"  {name}: O {100*m['original_accuracy']:.1f}%  P {100*m['perturbed_accuracy']:.1f}%  "
                      f"drop {m['drop_pp']:+.1f} pp [{m['drop_ci95_pp'][0]:+.1f}, {m['drop_ci95_pp'][1]:+.1f}]  "
                      f"{m['verdict']}  (items {m['items']}, infra {m['infra_failures']}, missing {m['missing']})")
        d = e["difference"]
        if d["items"]:
            print(f"  difference: {d['pp']:+.1f} pp [{d['ci95_pp'][0]:+.1f}, {d['ci95_pp'][1]:+.1f}]  {d['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
