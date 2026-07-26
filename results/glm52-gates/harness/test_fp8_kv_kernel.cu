/* T2 for task 49: validate the CUDA FP8 KV store/load kernels in ISOLATION,
 * before any engine modification.
 *
 * Contract (same as T1, now on-device): for rows that have already been
 * E4M3-rounded by the engine's quantizer, the sequence
 *      F32 staging  ->  fp8 store kernel  ->  fp8 load kernel  ->  F32
 * must reproduce the input BITWISE. Any deviation means the end-to-end
 * bit-identity gate (T3) cannot hold, so this test gates the port.
 *
 * Layout under test (matches the plan in task 49):
 *   576-dim row = 512 non-RoPE dims stored 1 byte each (sign + 7-bit E4M3 idx)
 *               +  64 RoPE dims stored as __half (NOT FP8: SnapMLA shows the
 *                  RoPE component spans +/-10^3 with outlier tails and FP8
 *                  there raises MSE by an order of magnitude)
 *   per 64-element block: one power-of-two scale, exactly as the engine does.
 *
 * Build: nvcc -O2 -arch=native -o /tmp/test_fp8_kv_kernel \
 *              results/glm52-gates/harness/test_fp8_kv_kernel.cu -lm
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(2);} } while(0)

/* ROPE_F32=1 keeps the RoPE tail at F32 so the whole row is BIT-IDENTICAL to
 * the F32 baseline (2.92x). ROPE_F32=0 stores RoPE at F16 for 3.48x, which is
 * what SnapMLA does (they keep RoPE at BF16) but is lossy, so that mode must be
 * gated by the paired NLL suite instead of by bit-identity. */
#ifndef ROPE_F32
#define ROPE_F32 1
#endif

#define HEAD_DIM 576u
#define N_ROT     64u
#define N_NOPE   (HEAD_DIM - N_ROT)

/* ---- E4M3 reference, identical maths to ds4.c ---------------------------- */
__host__ __device__ static float e4m3_value(int i) {
    const float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f, 0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f };
    const int e = (i >> 3) & 0x0f, m = i & 0x07;
    return e == 0 ? (float)m * 0.001953125f
                  : (1.0f + (float)m * 0.125f) * exp_scale[e];
}
__host__ __device__ static uint8_t e4m3_encode(float v) {
    /* T2 caught this: (v < 0.0f) is FALSE for -0.0f, which silently dropped the
     * sign bit on 217/2097152 values. |delta| was 0 so arithmetic was unaffected,
     * but it broke the bitwise contract that T3's bit-identity gate rests on.
     * The engine's quantizer really does emit -0.0 (sign * e4m3_value(0)). */
    const uint8_t sign = signbit(v) ? 0x80u : 0x00u;
    const float ax = fminf(fabsf(v), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1;
        if (e4m3_value(mid) <= ax) lo = mid; else hi = mid - 1; }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - e4m3_value(best));
        const float nd = fabsf(ax - e4m3_value(best + 1));
        if (nd < bd || (nd == bd && ((best + 1) & 1) == 0 && (best & 1) != 0)) best++;
    }
    return (uint8_t)(sign | (uint8_t)best);
}
__host__ __device__ static float e4m3_decode(uint8_t b) {
    const float mag = e4m3_value(b & 0x7f);
    return (b & 0x80u) ? -mag : mag;
}

/* ---- kernels under test -------------------------------------------------- */
/* store: F32 staging row -> packed (1 byte/nope dim, __half/rope dim) + scales */
__global__ static void fp8_kv_store(const float *__restrict__ src,
                                    uint8_t *__restrict__ nope,
#if ROPE_F32
                                    float   *__restrict__ rope,
#else
                                    __half  *__restrict__ rope,
#endif
                                    float   *__restrict__ scales,
                                    uint32_t rows) {
    const uint32_t r = blockIdx.x;
    if (r >= rows) return;
    const float *s = src + (size_t)r * HEAD_DIM;

    /* one block scale per 64 non-RoPE elements, computed exactly as the engine */
    for (uint32_t off = threadIdx.x * 64u; off < N_NOPE; off += blockDim.x * 64u) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 64u; i++) amax = fmaxf(amax, fabsf(s[off + i]));
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        scales[(size_t)r * (N_NOPE / 64u) + off / 64u] = scale;
        for (uint32_t i = 0; i < 64u; i++)
            nope[(size_t)r * N_NOPE + off + i] = e4m3_encode(s[off + i] / scale);
    }
    for (uint32_t i = threadIdx.x; i < N_ROT; i += blockDim.x)
#if ROPE_F32
        rope[(size_t)r * N_ROT + i] = s[N_NOPE + i];
#else
        rope[(size_t)r * N_ROT + i] = __float2half(s[N_NOPE + i]);
#endif
}

/* load: packed -> F32 row */
__global__ static void fp8_kv_load(float *__restrict__ dst,
                                   const uint8_t *__restrict__ nope,
#if ROPE_F32
                                   const float   *__restrict__ rope,
#else
                                   const __half  *__restrict__ rope,
#endif
                                   const float   *__restrict__ scales,
                                   uint32_t rows) {
    const uint32_t r = blockIdx.x;
    if (r >= rows) return;
    float *d = dst + (size_t)r * HEAD_DIM;
    for (uint32_t i = threadIdx.x; i < N_NOPE; i += blockDim.x) {
        const float scale = scales[(size_t)r * (N_NOPE / 64u) + i / 64u];
        d[i] = e4m3_decode(nope[(size_t)r * N_NOPE + i]) * scale;
    }
    for (uint32_t i = threadIdx.x; i < N_ROT; i += blockDim.x)
#if ROPE_F32
        d[N_NOPE + i] = rope[(size_t)r * N_ROT + i];
#else
        d[N_NOPE + i] = __half2float(rope[(size_t)r * N_ROT + i]);
#endif
}

/* ---- host reference quantizer (verbatim maths from ds4.c) ---------------- */
static float h_e4m3_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1;
        if (e4m3_value(mid) <= ax) lo = mid; else hi = mid - 1; }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - e4m3_value(best));
        const float nd = fabsf(ax - e4m3_value(best + 1));
        if (nd < bd || (nd == bd && ((best + 1) & 1) == 0 && (best & 1) != 0)) best++;
    }
    return sign * e4m3_value(best);
}
static void h_quantize_row(float *x) {
    for (uint32_t off = 0; off < N_NOPE; off += 64) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 64; i++) amax = fmaxf(amax, fabsf(x[off + i]));
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < 64; i++) {
            float v = x[off + i] / scale;
            v = fminf(fmaxf(v, -448.0f), 448.0f);
            x[off + i] = h_e4m3_dequant(v) * scale;
        }
    }
}

int main(void) {
    const uint32_t rows = 4096;              /* a realistic slab of context */
    printf("T2: CUDA FP8 KV store/load must round-trip BITWISE  [mode: RoPE %s]\n",
           ROPE_F32 ? "F32 strict" : "F16 compact");
    printf("    %u rows x %u dims (%u non-RoPE @1B + %u RoPE @2B)\n\n",
           rows, HEAD_DIM, N_NOPE, N_ROT);

    float *h_src = (float *)malloc((size_t)rows * HEAD_DIM * sizeof(float));
    float *h_out = (float *)malloc((size_t)rows * HEAD_DIM * sizeof(float));
    unsigned seed = 987654321u;
    for (uint32_t r = 0; r < rows; r++) {
        float *row = h_src + (size_t)r * HEAD_DIM;
        for (uint32_t i = 0; i < HEAD_DIM; i++) {
            seed = seed * 1103515245u + 12345u;
            float u = (float)((seed >> 8) & 0xffff) / 32768.0f - 1.0f;
            if (r % 5 == 1) u *= 1e-5f;                 /* tiny-amax blocks */
            if (r % 5 == 2) u *= 1e4f;                  /* large-amax blocks */
            if (r % 7 == 3 && i % 53 == 0) u *= 900.0f; /* clamp exercise */
            row[i] = u;
        }
        h_quantize_row(row);   /* engine already did this before storage */
    }

    float *d_src, *d_out, *d_scales; uint8_t *d_nope;
#if ROPE_F32
    float *d_rope; const size_t rope_elem = sizeof(float);
#else
    __half *d_rope; const size_t rope_elem = sizeof(__half);
#endif
    CUDA_OK(cudaMalloc(&d_src, (size_t)rows * HEAD_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_out, (size_t)rows * HEAD_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_nope, (size_t)rows * N_NOPE));
    CUDA_OK(cudaMalloc(&d_rope, (size_t)rows * N_ROT * rope_elem));
    CUDA_OK(cudaMalloc(&d_scales, (size_t)rows * (N_NOPE / 64) * sizeof(float)));
    CUDA_OK(cudaMemcpy(d_src, h_src, (size_t)rows * HEAD_DIM * sizeof(float),
                       cudaMemcpyHostToDevice));

    fp8_kv_store<<<rows, 128>>>(d_src, d_nope, d_rope, d_scales, rows);
    CUDA_OK(cudaGetLastError());
    fp8_kv_load<<<rows, 128>>>(d_out, d_nope, d_rope, d_scales, rows);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(h_out, d_out, (size_t)rows * HEAD_DIM * sizeof(float),
                       cudaMemcpyDeviceToHost));

    size_t bad_nope = 0, bad_rope = 0; float worst = 0.0f;
    for (uint32_t r = 0; r < rows; r++) {
        const float *a = h_src + (size_t)r * HEAD_DIM;
        const float *b = h_out + (size_t)r * HEAD_DIM;
        for (uint32_t i = 0; i < N_NOPE; i++)
            if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                bad_nope++; worst = fmaxf(worst, fabsf(a[i] - b[i]));
                if (bad_nope <= 5)
                    fprintf(stderr, "  nope r=%u i=%u got %.9g want %.9g\n", r, i, b[i], a[i]);
            }
        for (uint32_t i = N_NOPE; i < HEAD_DIM; i++)
            if (fabsf(a[i] - b[i]) > 0.0f) {           /* RoPE: F16 is lossy */
                bad_rope++; }
    }
    const size_t n_nope_total = (size_t)rows * N_NOPE;
    printf("T2a non-RoPE bitwise round-trip : %s  (%zu/%zu mismatched, worst |d|=%.3g)\n",
           bad_nope ? "FAIL" : "pass", bad_nope, n_nope_total, worst);
#if ROPE_F32
    printf("T2b RoPE stored at F32 (strict) : %s  (%zu/%zu differ) -- whole row is\n"
           "                                  bit-identical, so T3 can demand max|d|==0\n",
           bad_rope ? "FAIL" : "pass", bad_rope, (size_t)rows * N_ROT);
    if (bad_rope) bad_nope += bad_rope;
#else
    printf("T2b RoPE stored at F16 (compact): %zu/%zu differ -- EXPECTED, F16 is lossy.\n"
           "                                  This mode CANNOT use bit-identity; gate it\n"
           "                                  with the paired NLL suite instead.\n",
           bad_rope, (size_t)rows * N_ROT);
#endif

    const double bytes_f32 = (double)HEAD_DIM * 4.0;
    const double bytes_new = (double)N_NOPE * 1.0 + (double)N_ROT * (double)rope_elem
                           + (double)(N_NOPE / 64) * 4.0;   /* + block scales */
    printf("\nrow bytes: F32 %.0f -> packed %.0f  (%.2fx reduction, scales included)\n",
           bytes_f32, bytes_new, bytes_f32 / bytes_new);
    printf("%s\n", bad_nope ? "T2 FAILED" : "T2 PASSED (non-RoPE path is exact)");
    return bad_nope ? 1 : 0;
}
