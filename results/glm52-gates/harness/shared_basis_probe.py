#!/usr/bin/env python3
"""Shared hidden-space basis energy probe (creative round 2, probe 2 — the
MoBE-class go/no-go sol-B specified).

For one layer's gate_exps tensor (256 experts, each 2048x6144 IQ2_XXS):
accumulate S = sum_e We^T We over a subsample of experts, eigendecompose,
and report retained Frobenius energy at ranks 128/256/512/1024/1536 vs the
isotropic baseline r/6144. Go: >=90% at rank<=1024. Kill: otherwise.

Usage: shared_basis_probe.py RAWFILE N_EXPERTS_TOTAL ROWS COLS N_SAMPLE
RAWFILE = dd-extracted tensor bytes (IQ2_XXS, whole tensor).
"""
import sys, numpy as np
sys.path.insert(0, '/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/gguf-py')
from gguf.quants import IQ2_XXS

raw_path, n_exp, rows, cols, n_samp = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]))
BLK_B, BLK_W = 66, 256
exp_weights = rows * cols
exp_bytes = exp_weights // BLK_W * BLK_B
raw = open(raw_path, 'rb')
S = np.zeros((cols, cols), dtype=np.float64)
idx = np.linspace(0, n_exp - 1, n_samp).astype(int)
tot_energy = 0.0
for k, e in enumerate(idx):
    raw.seek(e * exp_bytes)
    b = raw.read(exp_bytes)
    w = IQ2_XXS.dequantize(np.frombuffer(b, dtype=np.uint8),
                           ).astype(np.float32).reshape(rows, cols)
    S += (w.T @ w).astype(np.float64)
    tot_energy += float((w * w).sum())
    if (k + 1) % 8 == 0:
        print(f"  accumulated {k+1}/{n_samp} experts", flush=True)
ev = np.linalg.eigvalsh(S)[::-1]          # descending
cum = np.cumsum(ev) / ev.sum()
print(f"experts sampled={n_samp}/{n_exp} matrix={rows}x{cols} "
      f"total Frobenius energy={tot_energy:.4g}")
for r in (128, 256, 512, 1024, 1536, 2048):
    if r <= cols:
        print(f"rank {r:5d}: retained energy {cum[r-1]*100:6.2f}%  "
              f"(isotropic baseline {r/cols*100:5.2f}%)")
print("GO if >=90% at rank<=1024; else the aggressive shared-basis design dies")
