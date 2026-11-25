#!/usr/bin/env python3
"""Diagnostic, written after the draw: per-suite z against Flash-Next, the paired
(McNemar) comparison, and the long-generation breakdown detector.

Reads only this bundle's transcripts and Flash-Next's holdout transcripts.
Run from anywhere: python3 breakdown_analysis.py > breakdown-analysis.json
"""
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FN = HERE.parent.parent / "qwen38fn-gates" / "holdout-2026-10-11"
WORD_RUN = re.compile(r"\b(\w+)(?:\s+\1\b){4,}")
CHAR_RUN = re.compile(r"(.)\1{59,}", re.S)
STRAY = re.compile(r"[ऀ-ॿ฀-๿가-힯؀-ۿ]")
LATE = 3000  # generated tokens; the onset estimate below is relative to the completion


def z(x1, n1, x2, n2):
    p = (x1 + x2) / (n1 + n2)
    return (x1 / n1 - x2 / n2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def load(directory):
    return {os.path.basename(p): json.load(open(p)) for p in glob.glob(f"{directory}/*.json")}


def onset(t, patterns):
    """Estimated generated-token position of the first signal, or None."""
    text = t.get("completion", "")
    ct = ((t.get("response") or {}).get("usage") or {}).get("completion_tokens") or 0
    hits = [m.start() for m in (p.search(text) for p in patterns) if m]
    if not hits or not ct:
        return None
    return int(min(hits) / max(1, len(text)) * ct)


def paired(m, f, keys):
    b = sum(1 for k in keys if m[k]["scored_correct"] and not f[k]["scored_correct"])
    c = sum(1 for k in keys if f[k]["scored_correct"] and not m[k]["scored_correct"])
    return {"rows": len(keys), "mimo": sum(bool(m[k]["scored_correct"]) for k in keys),
            "flash_next": sum(bool(f[k]["scored_correct"]) for k in keys),
            "mimo_only": b, "flash_next_only": c, "mcnemar_exact_p_two_sided": round(mcnemar(b, c), 4)}


out = {}
for suite in ("gsm8k", "mmlu-pro", "humaneval"):
    acc_m = json.load(open(HERE / "accuracy" / f"acc-{suite}.json"))
    acc_f = json.load(open(FN / "accuracy" / f"acc-{suite}.json"))
    m = load(HERE / "accuracy" / "transcripts" / suite)
    f = load(FN / "accuracy" / "transcripts" / suite)
    if set(m) != set(f):
        sys.exit(f"{suite}: transcript names differ")
    # Code legitimately holds long digit/letter runs, so HumanEval uses words and stray scripts only.
    patterns = (WORD_RUN, STRAY) if suite == "humaneval" else (WORD_RUN, CHAR_RUN, STRAY)
    late = sorted(k for k in m if (onset(m[k], patterns) or -1) >= LATE)
    late_fn = sorted(k for k in f if (onset(f[k], patterns) or -1) >= LATE)
    n, nf = acc_m["n"], acc_f["n"]
    out[suite] = {
        "mimo": acc_m["correct"], "flash_next": acc_f["correct"], "n": n,
        "delta": acc_m["correct"] - acc_f["correct"],
        "z_one_sided": round(z(acc_m["correct"], n, acc_f["correct"], nf), 3),
        "invalid_mimo": acc_m["invalid_count"], "invalid_flash_next": acc_f["invalid_count"],
        "rowset_equal": acc_m["rowset_sha256"] == acc_f["rowset_sha256"],
        "rowset_sha256_16": acc_m["rowset_sha256"][:16],
        "mimo_completion_tokens": sum(((t.get("response") or {}).get("usage") or {}).get("completion_tokens") or 0
                                      for t in m.values()),
        "flash_next_completion_tokens": sum(((t.get("response") or {}).get("usage") or {}).get("completion_tokens") or 0
                                            for t in f.values()),
        "mimo_at_budget": sum(1 for t in m.values() if t.get("finish_reason") == "length"),
        "flash_next_at_budget": sum(1 for t in f.values() if t.get("finish_reason") == "length"),
        "late_breakdowns_mimo": len(late), "late_breakdowns_flash_next": len(late_fn),
        "paired_all": paired(m, f, list(m)),
        "paired_mimo_not_broken": paired(m, f, [k for k in m if k not in late]),
        "paired_mimo_broken": paired(m, f, late),
        "late_breakdown_rows": late,
    }
json.dump(out, sys.stdout, indent=1)
print()
