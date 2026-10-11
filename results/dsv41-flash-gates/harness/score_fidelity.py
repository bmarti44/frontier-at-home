#!/usr/bin/env python3
"""Fixed scorer for the DeepSeek V4.1 Flash CUDA fidelity smoke.

Compares a CUDA score_official TSV against the upstream Metal reference TSV
for the same 112-case official-API fixture (ds4 gguf-tools/quality-testing/
deepseek-v4.1-flash-20260919-router). This is a fidelity smoke, not a
capability or lossy-change qualification.

Formulas (preregistered in ../smoke-2026-09-25/PREREGISTRATION.md):
  token-weighted NLL  = sum(nll) / sum(target_tokens)       over all cases
  API top-1 agreement = sum(api_top1_match) / sum(api_top1_count)
                        over cases except case_047 (known tokenizer mismatch)
  delta = cuda - metal; one-sided 95% bound from a paired case bootstrap
  (10,000 resamples, seed 20260925).
Input validation (candidate 3, review finding H3): both arms' case IDs must
  equal the fixture manifest's IDs exactly (manifest bytes hashed into the
  summary); target_tokens and api counts must be integers with
  target_tokens > 0, 0 <= api_top1_match <= api_top1_count; nll must be finite
  and >= 0; target_tokens and api_top1_count must be equal between arms.
PASS iff: every validation holds, identical case-ID sets of size 112,
  every case target_tokens equal between arms,
  NLL delta upper bound <= 0.01, and
  top-1 agreement loss upper bound <= 0.5 percentage points.
"""
import csv, hashlib, json, math, random, sys

EXCLUDE_API = {"case_047"}
FIELDS = ("nll", "target_tokens", "api_top1_match", "api_top1_count")


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            cid = r["id"]
            if cid in rows:
                raise SystemExit(f"duplicate id {cid} in {path}")
            vals = {}
            for k in FIELDS:
                if r.get(k) in (None, ""):
                    raise SystemExit(f"missing {k} for {cid} in {path}")
                v = float(r[k])
                if not math.isfinite(v):
                    raise SystemExit(f"non-finite {k} for {cid} in {path}")
                vals[k] = v
            for k in ("target_tokens", "api_top1_match", "api_top1_count"):
                if vals[k] != int(vals[k]):
                    raise SystemExit(f"non-integer {k} for {cid} in {path}")
            if vals["target_tokens"] <= 0:
                raise SystemExit(f"target_tokens <= 0 for {cid} in {path}")
            if vals["nll"] < 0:
                raise SystemExit(f"negative nll for {cid} in {path}")
            if not 0 <= vals["api_top1_match"] <= vals["api_top1_count"]:
                raise SystemExit(f"api_top1_match out of range for {cid} in {path}")
            rows[cid] = vals
    return rows


def fixture_ids(path):
    raw = open(path, "rb").read()
    ids = []
    for line in raw.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        ids.append(line.split("\t", 1)[0])
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate id in fixture {path}")
    return set(ids), hashlib.sha256(raw).hexdigest()


def metrics(rows, ids):
    nll = sum(rows[i]["nll"] for i in ids) / sum(rows[i]["target_tokens"] for i in ids)
    api = [i for i in ids if i not in EXCLUDE_API]
    top1 = sum(rows[i]["api_top1_match"] for i in api) / sum(rows[i]["api_top1_count"] for i in api)
    return nll, top1


def main(cuda_path, metal_path, fixture_path, out_path):
    cuda, metal = load(cuda_path), load(metal_path)
    fixture, fixture_sha = fixture_ids(fixture_path)
    checks = {}
    checks["cuda_ids_match_fixture"] = set(cuda) == fixture
    checks["metal_ids_match_fixture"] = set(metal) == fixture
    checks["ids_equal"] = set(cuda) == set(metal)
    checks["case_count_112"] = len(cuda) == 112 and len(metal) == 112
    ids = sorted(set(cuda) & set(metal))
    checks["target_tokens_equal"] = all(cuda[i]["target_tokens"] == metal[i]["target_tokens"] for i in ids)
    checks["api_top1_count_equal"] = all(cuda[i]["api_top1_count"] == metal[i]["api_top1_count"] for i in ids)
    if not all(checks.values()):
        summary = {"kind": "fidelity_smoke_not_capability", "checks": checks,
                   "fixture_manifest": fixture_path, "fixture_sha256": fixture_sha,
                   "verdict": "FAIL", "reason": "input validation failed"}
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
        print(json.dumps(summary, indent=1, sort_keys=True))
        return 1
    c_nll, c_top1 = metrics(cuda, ids)
    m_nll, m_top1 = metrics(metal, ids)
    rng = random.Random(20260925)
    d_nll, d_top1 = [], []
    for _ in range(10000):
        s = [rng.choice(ids) for _ in ids]
        cn, ct = metrics(cuda, s)
        mn, mt = metrics(metal, s)
        d_nll.append(cn - mn)
        d_top1.append((mt - ct) * 100.0)
    d_nll.sort(); d_top1.sort()
    nll_ub = d_nll[int(0.95 * len(d_nll))]
    top1_loss_ub = d_top1[int(0.95 * len(d_top1))]
    checks["nll_delta_ub_le_0.01"] = nll_ub <= 0.01
    checks["top1_loss_ub_le_0.5pp"] = top1_loss_ub <= 0.5
    summary = {
        "kind": "fidelity_smoke_not_capability",
        "cuda_tsv": cuda_path, "metal_reference_tsv": metal_path,
        "fixture_manifest": fixture_path, "fixture_sha256": fixture_sha,
        "reference_note": "Metal reference GGUF byte identity with the pinned CUDA Q2 is not established; a delta measures combined artifact+backend fidelity",
        "cases": len(ids),
        "cuda": {"token_weighted_nll": c_nll, "api_top1_agreement": c_top1},
        "metal": {"token_weighted_nll": m_nll, "api_top1_agreement": m_top1},
        "nll_delta": c_nll - m_nll, "nll_delta_upper95": nll_ub,
        "top1_loss_pp": (m_top1 - c_top1) * 100.0, "top1_loss_upper95_pp": top1_loss_ub,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: score_fidelity.py CUDA.tsv METAL.tsv FIXTURE_MANIFEST.tsv summary.json")
    sys.exit(main(*sys.argv[1:]))
