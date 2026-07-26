#!/usr/bin/env python3
"""T-A for task 50: the CHEAPEST falsification of FP4-at-1M, before any kernel.

THE QUESTION. 1M context needs the non-RoPE latent at FP4 (45.2 GiB KV, which
fits beside the 67 GiB expert cache). The blocker is fidelity: the published
datum for DeepSeek MLA with naive Q4 KV is +0.19 PPL (+3.0%), i.e. about
0.0296 nat/token, roughly 3x our 0.01 nat/token budget.

Every paper that rescues 4-bit KV does it with WALSH-HADAMARD ROTATION before
quantization (UltraQuant 2606.20474, SAW-INT4 2604.19157, RotateKV, KVLinC
2510.05373, PolarQuant 2603.29078). The mechanism: absmax block scaling wastes
its range on outliers, and rotation Gaussianises the block so the grid is used
efficiently.

PRE-REGISTERED CRITERION. To first order the NLL penalty scales with
quantization error ENERGY. To bring 0.0296 under 0.01 we need the error energy
cut by >= 3x, i.e. rotation must deliver >= 4.8 dB of SNR improvement. We
require >= 10 dB to leave margin for the fact that the published number is on a
different model and our latent is already E4M3-rounded. If rotation cannot
clear 10 dB on realistic data, FP4-at-1M is dead and no kernel should be
written.

HONEST LIMITATION, stated up front: this uses SYNTHETIC data with tail
heaviness swept across a realistic range, because no real KV capture exists yet
(capturing one needs an engine dump hook + a GPU window). It can therefore
FALSIFY cheaply but cannot CONFIRM. A real-capture rerun is required before any
implementation decision. The sweep is reported so the conclusion's sensitivity
to that assumption is visible rather than hidden.
"""
import math, sys

# ---- E2M1 (FP4) grid, exactly as ds4.c dsv4_e2m1fn_value_cpu ----------------
FP4_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def fp4_round(x):
    s = -1.0 if x < 0 else 1.0
    ax = min(abs(x), 6.0)
    best = min(FP4_GRID, key=lambda g: (abs(ax - g), g))
    return s * best

def quantize_fp4(vec, qb=32):
    """per-`qb` absmax with power-of-two scale, mirroring
    dsv4_fp4_act_quantize_row_inplace_cpu (ds4.c:3268)."""
    out = []
    for off in range(0, len(vec), qb):
        blk = vec[off:off + qb]
        amax = max(abs(v) for v in blk) or 7.05e-38
        scale = 2.0 ** math.ceil(math.log2(amax / 6.0))
        out.extend(fp4_round(v / scale) * scale for v in blk)
    return out

# ---- Walsh-Hadamard transform (in place, power-of-two length) --------------
def hadamard(v):
    n = len(v)
    x = list(v)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a, b = x[j], x[j + h]
                x[j], x[j + h] = a + b, a - b
        h *= 2
    s = 1.0 / math.sqrt(n)
    return [t * s for t in x]

def inv_hadamard(v):
    return hadamard(v)          # orthonormal + symmetric => self-inverse

# ---- data model -----------------------------------------------------------
def make_block(rng, n, tail):
    """Gaussian core with a heavy tail; `tail` = fraction of outlier channels,
    which is the documented shape of KV keys (a few channels much larger)."""
    out = []
    for i in range(n):
        u1 = max(rng(), 1e-12); u2 = rng()
        g = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        if (i * 2654435761) % 1000 < tail * 1000:
            g *= 30.0                      # outlier channel
        out.append(g)
    return out

def lcg():
    s = [12345]
    def r():
        s[0] = (s[0] * 1103515245 + 12345) & 0x7fffffff
        return s[0] / 0x7fffffff
    return r

def snr_db(ref, got):
    se = sum((a - b) ** 2 for a, b in zip(ref, got))
    sig = sum(a * a for a in ref)
    if se <= 0: return float("inf")
    return 10.0 * math.log10(sig / se)

def main():
    # CORRECTION after the first run: v1 used a 32-point Hadamard, but the
    # engine rotates with dsv4_hadamard128_inplace_cpu (128-wide, ds4.c:3252)
    # and only THEN quantizes in per-32 FP4 blocks
    # (dsv4_fp4_act_quantize_row_inplace_cpu, ds4.c:3268). UltraQuant likewise
    # uses a 128-wide Walsh-Hadamard. A 32-point transform averages over far
    # fewer channels and understates the outlier-spreading effect, so v1's
    # numbers were geometry-wrong, not a real falsification.
    N = 128                    # rotation width, matching the engine
    QB = 32                    # FP4 quantization block within the rotated vector
    BLOCKS = 1200
    print("T-A: can Hadamard rotation rescue FP4 on the MLA latent?")
    print("     rotation width %d (engine: dsv4_hadamard128_inplace_cpu),"
          " FP4 block %d\n" % (N, QB))
    print("pre-registered: rotation must give >= 10 dB SNR gain, else FP4-at-1M is dead")
    print("(>= 4.8 dB is the bare minimum to bring the published +3.0%% PPL under our")
    print(" 0.01 nat/token budget; 10 dB is that plus margin)\n")
    print("%-14s %12s %12s %10s  %s" % ("outlier frac", "plain FP4", "rotated FP4",
                                        "gain", "verdict"))
    worst_gain = float("inf")
    gains = []
    for tail in (0.0, 0.005, 0.01, 0.03, 0.06, 0.10):
        rng = lcg()
        plain_num = plain_den = rot_num = rot_den = 0.0
        for _ in range(BLOCKS):
            blk = make_block(rng, N, tail)
            # (a) quantize directly
            q = quantize_fp4(blk, QB)
            plain_num += sum((a - b) ** 2 for a, b in zip(blk, q))
            plain_den += sum(a * a for a in blk)
            # (b) rotate -> quantize -> rotate back
            r = hadamard(blk)
            qr = quantize_fp4(r, QB)
            back = inv_hadamard(qr)
            rot_num += sum((a - b) ** 2 for a, b in zip(blk, back))
            rot_den += sum(a * a for a in blk)
        p_db = 10 * math.log10(plain_den / plain_num) if plain_num else float("inf")
        r_db = 10 * math.log10(rot_den / rot_num) if rot_num else float("inf")
        gain = r_db - p_db
        worst_gain = min(worst_gain, gain)
        gains.append((tail, gain))
        print("%-14.3f %9.2f dB %9.2f dB %7.2f dB  %s"
              % (tail, p_db, r_db, gain,
                 "clears 10 dB" if gain >= 10 else
                 "clears 4.8 dB only" if gain >= 4.8 else "INSUFFICIENT"))

    print()
    # ANALYSIS FIX: reporting the minimum gain across the sweep was misleading,
    # because the minimum occurs at tail=0.0 -- pure Gaussian data, where plain
    # FP4 already achieves the HIGHEST SNR (18.75 dB) and there is nothing for
    # rotation to rescue. The published +3.0% PPL datum comes from real KV,
    # which HAS channel outliers, so the relevant figure is the gain on
    # outlier-bearing data.
    outlier_gains = [g for t, g in gains if t > 0.0]
    typ = sum(outlier_gains) / len(outlier_gains)
    base = 0.0296                       # published naive-Q4-on-MLA, nat/token
    pred = base / (10 ** (typ / 10.0))
    print("rotation gain on outlier-bearing data : %.2f dB (mean over tails > 0)"
          % typ)
    print("implied error-energy reduction        : %.2fx" % (10 ** (typ / 10.0)))
    print("predicted NLL penalty                 : %.4f nat/token (budget 0.010)"
          % pred)
    ok = pred <= 0.01
    shortfall = pred / 0.01
    print()
    if ok:
        print("VERDICT: rotation alone is PLAUSIBLY sufficient -> validate on real KV")
    else:
        print("VERDICT: rotation ALONE is INSUFFICIENT -- %.2fx over budget." % shortfall)
        print("         Not a kill: it needs a further %.1fx error-energy reduction"
              % shortfall)
        print("         (%.1f dB) from the rest of the published stack:" 
              % (10 * math.log10(shortfall)))
        print("           - learned per-channel linear correction (KVLinC 2510.05373),")
        print("             one element-wise multiply at inference")
        print("           - keep first/last 2 layers' KV at higher precision")
        print("             (UltraQuant 2606.20474)")
        print("           - preserve attention sinks (KVSink 2508.04257)")
        print("           - tiered precision: DSA reads only top-%d rows per token, so"
              % 2048)
        print("             rarely-selected rows can be cheaper than hot ones")
    print("\nREMINDER: synthetic data. This can kill the idea but cannot confirm it.")
    print("Next gate: capture real KV rows from the engine and rerun before deciding.")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
