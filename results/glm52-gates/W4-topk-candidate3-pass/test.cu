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
                std::fprintf(stderr,
                             "W4_OBSERVATION block=%u sequence=%u arm=%c elapsed_ms=%.9f\n",
                             block, arm, candidate_order[arm] ? 'B' : 'A',
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
