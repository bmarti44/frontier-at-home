#include "ds4_gpu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t kComponents = 1048576u;
constexpr uint32_t kTokens = 8u;
constexpr uint32_t kTopK = 2048u;
constexpr uint32_t kBlocks = 5u;
constexpr float kOneSidedT95Df4 = 2.131846786f;

float float_from_bits(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool cuda_ok(cudaError_t rc, const char *what) {
    if (rc == cudaSuccess) return true;
    std::fprintf(stderr, "w4-topk: %s: %s\n", what, cudaGetErrorString(rc));
    return false;
}

void set_candidate(bool enabled) {
    if (enabled) {
        setenv("DS4_CUDA_TOPK2048_CUB", "1", 1);
    } else {
        unsetenv("DS4_CUDA_TOPK2048_CUB");
    }
}

bool run_once(ds4_gpu_tensor *selected,
              const ds4_gpu_tensor *scores,
              bool candidate,
              float *elapsed_ms) {
    set_candidate(candidate);
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    if (!cuda_ok(cudaEventCreate(&begin), "create begin event") ||
        !cuda_ok(cudaEventCreate(&end), "create end event")) {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        return false;
    }
    bool ok = cuda_ok(cudaEventRecord(begin), "record begin") &&
              ds4_gpu_indexer_topk_tensor(selected, scores,
                                           kComponents, kTokens, kTopK) != 0 &&
              cuda_ok(cudaEventRecord(end), "record end") &&
              cuda_ok(cudaEventSynchronize(end), "synchronize end") &&
              cuda_ok(cudaEventElapsedTime(elapsed_ms, begin, end), "elapsed time");
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    return ok;
}

bool check_exact(const uint32_t *got, const uint32_t *expected, const char *arm) {
    for (uint64_t i = 0; i < (uint64_t)kTokens * kTopK; ++i) {
        if (got[i] != expected[i]) {
            std::fprintf(stderr,
                         "w4-topk: %s mismatch at element %llu: got=%u expected=%u\n",
                         arm, (unsigned long long)i, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

uint32_t sha256_rotr(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32u - bits));
}

void sha256_transform(uint32_t state[8], const unsigned char block[64]) {
    static constexpr uint32_t constants[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
    };
    uint32_t words[64];
    for (uint32_t i = 0; i < 16u; ++i) {
        words[i] = (uint32_t)block[4u*i] << 24u |
                   (uint32_t)block[4u*i+1u] << 16u |
                   (uint32_t)block[4u*i+2u] << 8u |
                   (uint32_t)block[4u*i+3u];
    }
    for (uint32_t i = 16u; i < 64u; ++i) {
        const uint32_t s0 = sha256_rotr(words[i-15u],7u) ^
                            sha256_rotr(words[i-15u],18u) ^ (words[i-15u] >> 3u);
        const uint32_t s1 = sha256_rotr(words[i-2u],17u) ^
                            sha256_rotr(words[i-2u],19u) ^ (words[i-2u] >> 10u);
        words[i] = words[i-16u] + s0 + words[i-7u] + s1;
    }
    uint32_t a=state[0], b=state[1], c=state[2], d=state[3];
    uint32_t e=state[4], f=state[5], g=state[6], h=state[7];
    for (uint32_t i = 0; i < 64u; ++i) {
        const uint32_t s1=sha256_rotr(e,6u)^sha256_rotr(e,11u)^sha256_rotr(e,25u);
        const uint32_t choice=(e&f)^(~e&g);
        const uint32_t temp1=h+s1+choice+constants[i]+words[i];
        const uint32_t s0=sha256_rotr(a,2u)^sha256_rotr(a,13u)^sha256_rotr(a,22u);
        const uint32_t majority=(a&b)^(a&c)^(b&c);
        const uint32_t temp2=s0+majority;
        h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

void sha256_bytes(const unsigned char *data, size_t bytes, unsigned char digest[32]) {
    uint32_t state[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                       0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    size_t offset=0;
    while (bytes-offset >= 64u) { sha256_transform(state,data+offset); offset+=64u; }
    unsigned char tail[128]={};
    const size_t remaining=bytes-offset;
    std::memcpy(tail,data+offset,remaining); tail[remaining]=0x80u;
    const size_t tail_bytes=remaining<56u ? 64u : 128u;
    const uint64_t bit_length=(uint64_t)bytes*8u;
    for (uint32_t i=0;i<8u;++i) tail[tail_bytes-1u-i]=(unsigned char)(bit_length>>(8u*i));
    sha256_transform(state,tail);
    if (tail_bytes==128u) sha256_transform(state,tail+64u);
    for (uint32_t i=0;i<8u;++i) {
        digest[4u*i]=(unsigned char)(state[i]>>24u);
        digest[4u*i+1u]=(unsigned char)(state[i]>>16u);
        digest[4u*i+2u]=(unsigned char)(state[i]>>8u);
        digest[4u*i+3u]=(unsigned char)state[i];
    }
}

void selected_sha256(const uint32_t *selected, uint64_t count, char out[65]) {
    unsigned char digest[32];
    sha256_bytes(reinterpret_cast<const unsigned char *>(selected),
                 (size_t)count * sizeof(uint32_t), digest);
    static constexpr char hex[] = "0123456789abcdef";
    for (uint32_t i = 0; i < 32u; ++i) {
        out[2u * i] = hex[digest[i] >> 4u];
        out[2u * i + 1u] = hex[digest[i] & 15u];
    }
    out[64] = '\0';
}

double paired_lower_95(const float *baseline_ms, const float *candidate_ms) {
    double logs[kBlocks];
    double mean = 0.0;
    for (uint32_t i = 0; i < kBlocks; ++i) {
        if (!(baseline_ms[i] > 0.0f) || !(candidate_ms[i] > 0.0f)) return 0.0;
        logs[i] = std::log((double)baseline_ms[i] / (double)candidate_ms[i]);
        mean += logs[i];
    }
    mean /= (double)kBlocks;
    double sumsq = 0.0;
    for (double v : logs) sumsq += (v - mean) * (v - mean);
    const double sample_sd = std::sqrt(sumsq / (double)(kBlocks - 1u));
    return std::exp(mean - (double)kOneSidedT95Df4 * sample_sd /
                           std::sqrt((double)kBlocks));
}

bool check_tie_order() {
    constexpr uint32_t n_components = 16384u;
    constexpr uint32_t n_tokens = 1u;
    constexpr uint32_t top_k = 2048u;
    float *scores_host = static_cast<float *>(
        std::malloc((size_t)n_components * sizeof(float)));
    uint32_t *selected_host = static_cast<uint32_t *>(
        std::malloc((size_t)top_k * sizeof(uint32_t)));
    if (!scores_host || !selected_host) return false;
    for (uint32_t i = 0; i < n_components; ++i) {
        scores_host[i] = (float)(i >> 2u);
    }
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(
        (uint64_t)n_components * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(
        (uint64_t)top_k * sizeof(uint32_t));
    bool ok = scores && selected &&
              ds4_gpu_tensor_write(scores, 0, scores_host,
                                   (uint64_t)n_components * sizeof(float));
    for (uint32_t arm = 0; ok && arm < 2u; ++arm) {
        set_candidate(arm == 1u);
        ok = ds4_gpu_indexer_topk_tensor(selected, scores,
                                         n_components, n_tokens, top_k) != 0 &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(selected, 0, selected_host,
                                  (uint64_t)top_k * sizeof(uint32_t));
        for (uint32_t rank = 0; ok && rank < top_k; ++rank) {
            const uint32_t group = (n_components >> 2u) - 1u - rank / 4u;
            const uint32_t expected = group * 4u + rank % 4u;
            if (selected_host[rank] != expected) {
                std::fprintf(stderr,
                             "w4-topk: %s tie mismatch rank=%u got=%u expected=%u\n",
                             arm ? "candidate" : "baseline", rank,
                             selected_host[rank], expected);
                ok = false;
            }
        }
    }
    set_candidate(false);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    std::free(selected_host);
    std::free(scores_host);
    return ok;
}

bool compare_arms(const float *scores_host,
                  uint32_t n_components,
                  uint32_t top_k,
                  const char *label) {
    uint32_t *baseline = static_cast<uint32_t *>(
        std::malloc((size_t)top_k * sizeof(uint32_t)));
    uint32_t *candidate = static_cast<uint32_t *>(
        std::malloc((size_t)top_k * sizeof(uint32_t)));
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(
        (uint64_t)n_components * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(
        (uint64_t)top_k * sizeof(uint32_t));
    bool ok = baseline && candidate && scores && selected &&
              ds4_gpu_tensor_write(scores, 0, scores_host,
                                   (uint64_t)n_components * sizeof(float));
    for (uint32_t arm = 0; ok && arm < 2u; ++arm) {
        set_candidate(arm == 1u);
        uint32_t *dst = arm ? candidate : baseline;
        ok = ds4_gpu_indexer_topk_tensor(selected, scores,
                                         n_components, 1u, top_k) != 0 &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(selected, 0, dst,
                                  (uint64_t)top_k * sizeof(uint32_t));
    }
    if (ok && std::memcmp(baseline, candidate,
                          (size_t)top_k * sizeof(uint32_t)) != 0) {
        for (uint32_t rank = 0; rank < top_k; ++rank) {
            if (baseline[rank] != candidate[rank]) {
                std::fprintf(stderr,
                             "w4-topk: %s arm mismatch rank=%u baseline=%u candidate=%u\n",
                             label, rank, baseline[rank], candidate[rank]);
                break;
            }
        }
        ok = false;
    }
    set_candidate(false);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    std::free(candidate);
    std::free(baseline);
    return ok;
}

bool check_boundary_and_special_values() {
    constexpr uint32_t top_k = 2048u;
    const uint32_t shapes[] = {
        4097u, 8191u, 8192u, 8193u, 16383u, 16385u,
        24575u, 24576u, 24577u, 1000003u
    };
    for (uint32_t n_components : shapes) {
        float *scores = static_cast<float *>(
            std::malloc((size_t)n_components * sizeof(float)));
        if (!scores) return false;
        for (uint32_t i = 0; i < n_components; ++i) {
            scores[i] = (float)((i * 2654435761u) & 0x00ffffffu);
        }
        char label[64];
        std::snprintf(label, sizeof(label), "boundary-%u", n_components);
        const bool ok = compare_arms(scores, n_components, top_k, label);
        std::free(scores);
        if (!ok) return false;
    }

    constexpr uint32_t special_n = 16384u;
    float *special = static_cast<float *>(
        std::malloc((size_t)special_n * sizeof(float)));
    if (!special) return false;
    for (uint32_t i = 0; i < special_n; ++i) {
        special[i] = (i & 1u) ? 0.0f : -0.0f;
    }
    bool ok = compare_arms(special, special_n, top_k, "signed-zero-cutoff");
    if (ok) {
        for (uint32_t i = 0; i < special_n; ++i) special[i] = (float)i;
        special[17] = INFINITY;
        special[8191] = -INFINITY;
        special[8192] = INFINITY;
        special[16370] = -INFINITY;
        ok = compare_arms(special, special_n, top_k, "infinities-cross-chunk");
    }
    if (ok) {
        for (uint32_t i = 0; i < special_n; ++i) special[i] = (float)i;
        special[3] = float_from_bits(0x7fc00001u);
        special[2047] = float_from_bits(0xffc00002u);
        special[8194] = float_from_bits(0x7fc01234u);
        ok = compare_arms(special, special_n, top_k, "nan-fallback");
    }
    std::free(special);
    return ok;
}

}  // namespace

int main() {
    const uint64_t score_count = (uint64_t)kComponents * kTokens;
    const uint64_t selected_count = (uint64_t)kTopK * kTokens;
    float *scores_host = static_cast<float *>(
        std::malloc((size_t)score_count * sizeof(float)));
    uint32_t *expected = static_cast<uint32_t *>(
        std::malloc((size_t)selected_count * sizeof(uint32_t)));
    uint32_t *baseline_out = static_cast<uint32_t *>(
        std::malloc((size_t)selected_count * sizeof(uint32_t)));
    uint32_t *candidate_out = static_cast<uint32_t *>(
        std::malloc((size_t)selected_count * sizeof(uint32_t)));
    if (!scores_host || !expected || !baseline_out || !candidate_out) return 2;

    const uint32_t mask = kComponents - 1u;
    for (uint32_t t = 0; t < kTokens; ++t) {
        const uint32_t salt = (0x9e3779b9u * (t + 1u)) & mask;
        for (uint32_t i = 0; i < kComponents; ++i) {
            scores_host[(uint64_t)t * kComponents + i] = (float)(i ^ salt);
        }
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            expected[(uint64_t)t * kTopK + rank] =
                (mask - rank) ^ salt;
        }
    }

    if (!ds4_gpu_init()) return 2;
    if (!check_tie_order() || !check_boundary_and_special_values()) {
        ds4_gpu_cleanup();
        return 1;
    }
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(selected_count * sizeof(uint32_t));
    bool ok = scores && selected &&
              ds4_gpu_tensor_write(scores, 0, scores_host,
                                   score_count * sizeof(float));
    float baseline_ms[kBlocks] = {};
    float candidate_ms[kBlocks] = {};
    const char *first_schedule_env = std::getenv("W4_FIRST_SCHEDULE");
    if (!first_schedule_env ||
        (std::strcmp(first_schedule_env, "ABBA") != 0 &&
         std::strcmp(first_schedule_env, "BAAB") != 0)) {
        std::fprintf(stderr,
                     "w4-topk: W4_FIRST_SCHEDULE must be ABBA or BAAB\n");
        ok = false;
    }
    const bool first_baab = first_schedule_env &&
        std::strcmp(first_schedule_env, "BAAB") == 0;

    if (ok) {
        float ignored = 0.0f;
        ok = run_once(selected, scores, false, &ignored) &&
             run_once(selected, scores, true, &ignored);
    }

    for (uint32_t block = 0; ok && block < kBlocks; ++block) {
        const bool baab = first_baab ^ ((block & 1u) != 0u);
        float arm_ms[4] = {};
        const bool candidate_order[4] = {
            baab, !baab, !baab, baab
        };
        for (uint32_t arm = 0; ok && arm < 4u; ++arm) {
            ok = run_once(selected, scores, candidate_order[arm], &arm_ms[arm]);
            uint32_t *dst = candidate_order[arm] ? candidate_out : baseline_out;
            if (ok) {
                ok = ds4_gpu_tensor_read(selected, 0, dst,
                                         selected_count * sizeof(uint32_t)) != 0 &&
                     check_exact(dst, expected,
                                 candidate_order[arm] ? "candidate" : "baseline");
            }
            if (ok) {
                char ids_sha256[65];
                selected_sha256(dst, selected_count, ids_sha256);
                std::fprintf(stderr,
                             "W4_OBSERVATION block=%u sequence=%u arm=%c "
                             "mode=%u exact=1 ids_sha256=%s elapsed_ms=%.9f\n",
                             block, arm, candidate_order[arm] ? 'B' : 'A',
                             candidate_order[arm] ? 1u : 0u, ids_sha256,
                             arm_ms[arm]);
            }
        }
        if (ok) {
            baseline_ms[block] = baab ? (arm_ms[1] + arm_ms[2]) * 0.5f
                                      : (arm_ms[0] + arm_ms[3]) * 0.5f;
            candidate_ms[block] = baab ? (arm_ms[0] + arm_ms[3]) * 0.5f
                                       : (arm_ms[1] + arm_ms[2]) * 0.5f;
        }
    }

    if (ok && std::memcmp(baseline_out, candidate_out,
                          selected_count * sizeof(uint32_t)) != 0) {
        std::fprintf(stderr, "w4-topk: arms selected different IDs\n");
        ok = false;
    }

    const double lower95 = ok ? paired_lower_95(baseline_ms, candidate_ms) : 0.0;
    std::fprintf(stderr,
                 "w4-topk: n_comp=%u n_tokens=%u top_k=%u "
                 "baseline_ms=[%.6f,%.6f,%.6f,%.6f,%.6f] "
                 "candidate_ms=[%.6f,%.6f,%.6f,%.6f,%.6f] "
                 "speedup_lower95=%.9f required=2.0\n",
                 kComponents, kTokens, kTopK,
                 baseline_ms[0], baseline_ms[1], baseline_ms[2],
                 baseline_ms[3], baseline_ms[4],
                 candidate_ms[0], candidate_ms[1], candidate_ms[2],
                 candidate_ms[3], candidate_ms[4], lower95);
    if (ok && lower95 < 2.0) {
        std::fprintf(stderr,
                     "w4-topk RED: default-off exact-select candidate did not reach 2x\n");
        ok = false;
    }

    set_candidate(false);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    ds4_gpu_cleanup();
    std::free(candidate_out);
    std::free(baseline_out);
    std::free(expected);
    std::free(scores_host);
    return ok ? 0 : 1;
}
