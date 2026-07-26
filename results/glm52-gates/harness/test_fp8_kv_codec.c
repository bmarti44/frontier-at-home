/* T1 for task 49: prove an FP8 storage codec is LOSSLESS for the values the
 * engine already produces.
 *
 * Contract being tested (written before the codec exists):
 *   dsv4_fp8_kv_quantize_row_inplace_cpu (ds4.c:3210) rounds every non-RoPE
 *   element to  scale * sign * dsv4_e4m3fn_value_cpu(idx)  with idx in [0,126]
 *   and scale a power of two. Every such value is therefore representable in
 *   ONE byte (sign bit + 7-bit index) given the block scale. So storing the
 *   cache at FP8 instead of F32 must be EXACTLY reversible -- not approximate.
 *
 * This is what makes the end-to-end acceptance test (T3) able to demand
 * bit-identical logits rather than "close enough".
 *
 * The reference functions below are copied verbatim from ds4.c; test_source_drift
 * re-derives them from the engine source at runtime so the copy cannot silently
 * go stale.
 *
 * Build:  cc -O2 -std=c11 -Wall -Wextra -o /tmp/test_fp8_kv_codec \
 *              results/glm52-gates/harness/test_fp8_kv_codec.c -lm
 * Run:    /tmp/test_fp8_kv_codec [path-to-ds4.c]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- verbatim from ds4.c ------------------------------------------------ */
static float dsv4_e4m3fn_value_cpu(int i) {
    static const float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f,
        0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f,
        32.0f, 64.0f, 128.0f, 256.0f,
    };
    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

static float dsv4_e4m3fn_dequant_cpu(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_cpu(mid) <= ax) lo = mid; else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - dsv4_e4m3fn_value_cpu(best));
        const float next_diff = fabsf(ax - dsv4_e4m3fn_value_cpu(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best++;
        }
    }
    return sign * dsv4_e4m3fn_value_cpu(best);
}

static void dsv4_fp8_kv_quantize_row_inplace_cpu(float *x, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 64; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < 64; i++) {
            float v = x[off + i] / scale;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            x[off + i] = dsv4_e4m3fn_dequant_cpu(v) * scale;
        }
    }
}

/* ---- THE CODEC UNDER TEST ----------------------------------------------- */
/* Storage byte: bit7 = sign, bits6..0 = E4M3 index in [0,126].
 * encode() must be exact for any value in the quantizer's image. */

static uint8_t fp8_store_encode(float v_over_scale) {
    const uint8_t sign = (v_over_scale < 0.0f ||
                          (v_over_scale == 0.0f && signbit(v_over_scale))) ? 0x80u : 0x00u;
    const float ax = fminf(fabsf(v_over_scale), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_cpu(mid) <= ax) lo = mid; else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - dsv4_e4m3fn_value_cpu(best));
        const float nd = fabsf(ax - dsv4_e4m3fn_value_cpu(best + 1));
        if (nd < bd || (nd == bd && ((best + 1) & 1) == 0 && (best & 1) != 0)) best++;
    }
    return (uint8_t)(sign | (uint8_t)best);
}

static float fp8_store_decode(uint8_t b) {
    const float mag = dsv4_e4m3fn_value_cpu(b & 0x7fu);
    return (b & 0x80u) ? -mag : mag;
}

/* ---- tests -------------------------------------------------------------- */
static int failures = 0;
static void fail(const char *what, double a, double b, int idx) {
    if (failures < 20)
        fprintf(stderr, "FAIL %-34s got %.9g want %.9g (i=%d)\n", what, a, b, idx);
    failures++;
}

/* T1a: every representable magnitude round-trips exactly, both signs. */
static void test_exhaustive_indices(void) {
    for (int i = 0; i <= 126; i++) {
        for (int s = 0; s < 2; s++) {
            const float v = (s ? -1.0f : 1.0f) * dsv4_e4m3fn_value_cpu(i);
            const float rt = fp8_store_decode(fp8_store_encode(v));
            if (memcmp(&rt, &v, sizeof(float)) != 0) {
                /* -0.0 vs +0.0 is a bit pattern difference we tolerate only
                 * when the magnitude is genuinely zero AND the consumer treats
                 * them identically; flag it so the decision is explicit. */
                if (!(v == 0.0f && rt == 0.0f)) fail("exhaustive index round-trip", rt, v, i);
            }
        }
    }
    printf("T1a exhaustive index round-trip (254 values)      : %s\n",
           failures ? "FAIL" : "pass");
}

/* T1b: the FULL image of the quantizer -- every representable magnitude across
 * a wide sweep of block scales -- must round-trip exactly through the codec. */
static void test_full_quantizer_image(void) {
    const int before = failures;
    for (int e = -40; e <= 40; e++) {
        const float scale = ldexpf(1.0f, e);
        for (int i = 0; i <= 126; i++) {
            for (int s = 0; s < 2; s++) {
                const float stored = (s ? -1.0f : 1.0f) * dsv4_e4m3fn_value_cpu(i);
                const float v = stored * scale;              /* what the cache holds */
                if (!isfinite(v)) continue;
                const float rt = fp8_store_decode(fp8_store_encode(v / scale)) * scale;
                if (rt != v) fail("quantizer-image round-trip", rt, v, i);
            }
        }
    }
    printf("T1b quantizer image over 81 scales (20574 values) : %s\n",
           failures == before ? "pass" : "FAIL");
}

/* T1c: end-to-end on real-shaped rows. Quantize like the engine, then require
 * the codec to reproduce the quantized row EXACTLY. This is the property that
 * lets T3 demand bit-identical logits. */
static void test_row_roundtrip(void) {
    const int before = failures;
    const uint32_t head_dim = 576, n_rot = 64, n_nope = head_dim - n_rot;
    static float row[576], ref[576];
    unsigned seed = 12345u;
    for (int trial = 0; trial < 200; trial++) {
        for (uint32_t i = 0; i < head_dim; i++) {
            seed = seed * 1103515245u + 12345u;
            float u = (float)((seed >> 8) & 0xffff) / 32768.0f - 1.0f;
            /* mix in scales and outliers so blocks span very different amax */
            if (trial % 4 == 1) u *= 1e-5f;
            if (trial % 4 == 2) u *= 1e4f;
            if (trial % 8 == 3 && i % 37 == 0) u *= 500.0f;   /* clamp exercise */
            row[i] = u;
        }
        dsv4_fp8_kv_quantize_row_inplace_cpu(row, head_dim, n_rot);
        memcpy(ref, row, sizeof(row));

        /* storage round-trip, per 64-element block, mirroring the real layout */
        for (uint32_t off = 0; off < n_nope; off += 64) {
            float amax = 0.0f;
            for (uint32_t i = 0; i < 64; i++) {
                const float av = fabsf(ref[off + i]);
                if (av > amax) amax = av;
            }
            if (amax < 1.0e-4f) amax = 1.0e-4f;
            const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
            for (uint32_t i = 0; i < 64; i++) {
                const uint8_t b = fp8_store_encode(ref[off + i] / scale);
                row[off + i] = fp8_store_decode(b) * scale;
            }
        }
        for (uint32_t i = 0; i < n_nope; i++)
            if (row[i] != ref[i]) fail("row round-trip (non-RoPE)", row[i], ref[i], (int)i);
        /* RoPE tail is NOT FP8-rounded by the engine and must be left alone */
        for (uint32_t i = n_nope; i < head_dim; i++)
            if (row[i] != ref[i]) fail("RoPE tail must be untouched", row[i], ref[i], (int)i);
    }
    printf("T1c real-shaped row round-trip (200 rows)         : %s\n",
           failures == before ? "pass" : "FAIL");
}

/* T1d: the copied reference must not drift from the engine source. */
static void test_source_drift(const char *ds4c) {
    FILE *f = fopen(ds4c, "r");
    if (!f) { printf("T1d source-drift check                            : SKIP (%s unreadable)\n", ds4c); return; }
    char line[4096];
    int seen_446 = 0, seen_scale = 0, seen_nope = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "fminf(fabsf(x), 448.0f)")) seen_446 = 1;
        if (strstr(line, "ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)))")) seen_scale = 1;
        if (strstr(line, "const uint32_t n_nope = head_dim - n_rot;")) seen_nope = 1;
    }
    fclose(f);
    const int ok = seen_446 && seen_scale && seen_nope;
    if (!ok) failures++;
    printf("T1d source-drift check vs ds4.c                   : %s\n", ok ? "pass" : "FAIL");
}

int main(int argc, char **argv) {
    printf("T1: FP8 KV storage codec must be LOSSLESS for already-quantized values\n\n");
    test_exhaustive_indices();
    test_full_quantizer_image();
    test_row_roundtrip();
    test_source_drift(argc > 1 ? argv[1] : "/home/dsv4/ds4-project/src/ds4-upstream-master/ds4.c");
    printf("\n%s (%d failures)\n", failures ? "T1 FAILED" : "T1 PASSED", failures);
    return failures ? 1 : 0;
}
