#!/usr/bin/env python3
"""Per-stream entropy of IQ2_XXS expert data (creative round 2, probe 1).

IQ2_XXS block (256 weights, 66 bytes): fp16 d (2B) + qs[32] uint16 (64B).
Per 8-weight group the uint16s carry E8-lattice indices (low bytes of the
first 4 uint16 in each 4-uint16 cluster) and sign/scale bits (high bytes /
last uint16s). The old 3.8-4.2% compressibility ceiling was measured on the
INTERLEAVED packed bytes; this probe separates streams:
  stream A = even-position low bytes (mostly lattice indices)
  stream B = even-position high bytes
  stream C = odd-position low/high bytes (signs/scales mix)
plus first-order (order-1 context) entropy of stream A, and the fp16 scale
stream. If any stream's entropy is well under 8 bits/byte, model-aware
entropy coding has real headroom that byte-level zstd could not see.

Usage: iq2_stream_entropy.py GGUF OFFSET NBYTES
"""
import sys, numpy as np

path, off, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
BLK = 66
with open(path, 'rb') as f:
    f.seek(off)
    raw = f.read((n // BLK) * BLK)
a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, BLK)
d16 = a[:, :2].copy().view(np.uint8)            # scale stream (fp16 bytes)
qs = a[:, 2:].reshape(-1, 32, 2)                # 32 uint16 per block

def H(x):
    c = np.bincount(x.ravel(), minlength=256).astype(np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())

def H1(x):  # order-1: H(next | prev), on a flat stream
    x = x.ravel()
    pairs = x[:-1].astype(np.uint16) * 256 + x[1:]
    c = np.bincount(pairs, minlength=65536).astype(np.float64)
    pj = c / c.sum()
    cm = np.bincount(x[:-1], minlength=256).astype(np.float64)
    pm = cm / cm.sum()
    nz = pj > 0
    hj = float(-(pj[nz] * np.log2(pj[nz])).sum())
    hm = float(-(pm[pm > 0] * np.log2(pm[pm > 0])).sum())
    return hj - hm

lo = qs[:, :, 0]                                 # low bytes of each uint16
hi = qs[:, :, 1]                                 # high bytes
even_lo, even_hi = lo[:, 0::2], hi[:, 0::2]
odd_lo, odd_hi = lo[:, 1::2], hi[:, 1::2]

tot_bytes = a.size
streams = {
    "even_lo(idx-ish)": even_lo, "even_hi": even_hi,
    "odd_lo": odd_lo, "odd_hi": odd_hi,
    "scale_fp16_lo": d16[:, 0], "scale_fp16_hi": d16[:, 1],
}
weighted = 0.0
print(f"blocks={len(a)} bytes={tot_bytes}")
for name, s in streams.items():
    h0 = H(s)
    frac = s.size / tot_bytes
    weighted += h0 * frac
    extra = f" H1={H1(s):.3f}" if name == "even_lo(idx-ish)" else ""
    print(f"{name:18s} H0={h0:.3f} bits/byte  share={frac*100:5.2f}%{extra}")
print(f"WEIGHTED H0 = {weighted:.3f} bits/byte -> ideal order-0 per-stream "
      f"compression = {(1 - weighted/8)*100:.2f}%")
print("(old packed-bytes ceiling was 3.8-4.2%; >8-10% here validates "
      "stream-separated entropy coding)")
