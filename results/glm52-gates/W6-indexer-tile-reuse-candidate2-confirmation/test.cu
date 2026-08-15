#include "ds4_gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kRows = 1048576u;
constexpr uint32_t kRaggedRows = 1048575u;
constexpr uint32_t kTokens = 65u;
constexpr uint32_t kHeads = 32u;
constexpr uint32_t kHeadDim = 128u;
constexpr uint32_t kTopK = 128u;
constexpr uint32_t kCanaryElements = 257u;
constexpr float kScorePoison = -12345.25f;
constexpr uint32_t kIdPoison = 0xdec0addeu;

struct Case {
    uint32_t rows;
    uint32_t tokens;
    uint32_t pos0;
    bool causal;
};

bool write_tensor(ds4_gpu_tensor *tensor, const void *src, uint64_t bytes) {
    return tensor && ds4_gpu_tensor_write(tensor, 0, src, bytes) != 0;
}

bool exact_width(const char *value) {
    return value && ((value[0] == '1' && value[1] == '\0') ||
                     (value[0] == '2' && value[1] == '\0') ||
                     (value[0] == '4' && value[1] == '\0'));
}

bool parse_schedule(const char *text, std::vector<std::string> &out) {
    if (!text || !*text) return false;
    const char *start = text;
    for (const char *p = text;; ++p) {
        if (*p != ',' && *p != '\0') continue;
        std::string value(start, (size_t)(p - start));
        if (!exact_width(value.c_str())) return false;
        out.push_back(value);
        if (*p == '\0') break;
        start = p + 1;
    }
    if (out.size() != 15u) return false;
    for (const char *width : {"1", "2", "4"}) {
        if (std::count(out.begin(), out.end(), std::string(width)) != 5) {
            return false;
        }
    }
    return true;
}

uint64_t logical_k_bytes(uint32_t width) {
    const uint64_t query_tiles = (64u + 15u) / 16u;
    const uint64_t groups = (query_tiles + width - 1u) / width;
    return groups * (uint64_t)kRows * kHeadDim * sizeof(float);
}

}  // namespace

int main() {
    const uint64_t key_count = (uint64_t)kRows * kHeadDim;
    const uint64_t q_count = (uint64_t)kTokens * kHeads * kHeadDim;
    const uint64_t weight_count = (uint64_t)kTokens * kHeads;
    const uint64_t score_count = (uint64_t)kTokens * kRows;
    const uint64_t id_count = (uint64_t)kTokens * kTopK;
    const uint64_t score_capacity = score_count + kCanaryElements;
    const uint64_t id_capacity = id_count + kCanaryElements;

    std::vector<float> key(key_count);
    std::vector<float> q(q_count);
    std::vector<float> weights(weight_count);
    std::vector<float> score_host(score_capacity);
    std::vector<uint32_t> id_host(id_capacity);
    std::vector<float> baseline_scores;
    std::vector<uint32_t> baseline_ids;
    std::vector<float> reference64_scores;
    std::vector<uint32_t> reference64_ids;
    for (uint64_t i = 0; i < key_count; ++i) {
        key[i] = (float)((int32_t)((i * 17u + 5u) % 4093u) - 2046) / 257.0f;
    }
    for (uint64_t i = 0; i < q_count; ++i) {
        q[i] = (float)((int32_t)((i * 29u + 11u) % 2039u) - 1019) / 509.0f;
    }
    for (uint64_t i = 0; i < weight_count; ++i) {
        weights[i] = 0.5f + (float)(i % 17u) / 32.0f;
    }

    std::vector<std::string> schedule;
    if (!parse_schedule(std::getenv("W6_TIMING_SCHEDULE"), schedule)) {
        std::fprintf(stderr, "FAIL: W6_TIMING_SCHEDULE must contain five exact instances each of 1,2,4\n");
        return 2;
    }

    if (!ds4_gpu_init()) return 2;
    ds4_gpu_tensor *key_gpu = ds4_gpu_tensor_alloc(key_count * sizeof(float));
    ds4_gpu_tensor *q_gpu = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *weights_gpu = ds4_gpu_tensor_alloc(weight_count * sizeof(float));
    ds4_gpu_tensor *scores_gpu = ds4_gpu_tensor_alloc(score_capacity * sizeof(float));
    ds4_gpu_tensor *ids_gpu = ds4_gpu_tensor_alloc(id_capacity * sizeof(uint32_t));
    bool ok = key_gpu && q_gpu && weights_gpu && scores_gpu && ids_gpu &&
              write_tensor(key_gpu, key.data(), key_count * sizeof(float)) &&
              write_tensor(q_gpu, q.data(), q_count * sizeof(float)) &&
              write_tensor(weights_gpu, weights.data(), weight_count * sizeof(float));
    const float scale = 1.0f / std::sqrt((float)(kHeads * kHeadDim));

    auto poison = [&]() {
        std::fill(score_host.begin(), score_host.end(), kScorePoison);
        std::fill(id_host.begin(), id_host.end(), kIdPoison);
        return write_tensor(scores_gpu, score_host.data(), score_capacity * sizeof(float)) &&
               write_tensor(ids_gpu, id_host.data(), id_capacity * sizeof(uint32_t)) &&
               ds4_gpu_synchronize() != 0;
    };

    auto validate_complete = [&](uint64_t expected_scores, uint64_t expected_ids,
                                 const char *width, const Case &c) {
        bool complete = true;
        for (uint64_t i = 0; i < expected_scores; ++i) {
            if (score_host[i] == kScorePoison) { complete = false; break; }
        }
        for (uint64_t i = 0; complete && i < expected_ids; ++i) {
            if (id_host[i] == kIdPoison) { complete = false; break; }
        }
        for (uint64_t i = score_count; complete && i < score_capacity; ++i) {
            if (score_host[i] != kScorePoison) { complete = false; break; }
        }
        for (uint64_t i = id_count; complete && i < id_capacity; ++i) {
            if (id_host[i] != kIdPoison) { complete = false; break; }
        }
        if (!complete) {
            std::fprintf(stderr,
                         "FAIL: incomplete output or damaged canary width=%s rows=%u tokens=%u pos0=%u causal=%u\n",
                         width, c.rows, c.tokens, c.pos0, c.causal ? 1u : 0u);
        }
        return complete;
    };

    auto run_width = [&](const char *width, const Case &c,
                         std::vector<float> &scores,
                         std::vector<uint32_t> &ids) {
        if (width) setenv("DS4_CUDA_GLM_INDEXER_QUERY_TILES", width, 1);
        else unsetenv("DS4_CUDA_GLM_INDEXER_QUERY_TILES");
        const uint64_t expected_scores = (uint64_t)c.tokens * c.rows;
        const uint64_t expected_ids = (uint64_t)c.tokens * kTopK;
        bool run_ok = poison() &&
                      ds4_gpu_glm_indexer_scores_batch_tensor(
                          scores_gpu, q_gpu, weights_gpu, key_gpu, c.rows,
                          c.tokens, c.pos0, kHeads, kHeadDim, scale, c.causal) != 0 &&
                      ds4_gpu_synchronize() != 0 &&
                      ds4_gpu_indexer_topk_tensor(
                          ids_gpu, scores_gpu, c.rows, c.tokens, kTopK) != 0 &&
                      ds4_gpu_synchronize() != 0 &&
                      ds4_gpu_tensor_read(scores_gpu, 0, score_host.data(),
                                          score_capacity * sizeof(float)) != 0 &&
                      ds4_gpu_tensor_read(ids_gpu, 0, id_host.data(),
                                          id_capacity * sizeof(uint32_t)) != 0;
        run_ok = run_ok && validate_complete(expected_scores, expected_ids,
                                              width ? width : "unset", c);
        if (run_ok) {
            scores.assign(score_host.begin(), score_host.begin() + expected_scores);
            ids.assign(id_host.begin(), id_host.begin() + expected_ids);
        }
        return run_ok;
    };

    const std::vector<Case> cases = {
        {kRows, 1u, 97u, false},  {kRows, 16u, 97u, false},
        {kRows, 17u, 97u, false}, {kRows, 32u, 97u, false},
        {kRows, 33u, 97u, false}, {kRows, 64u, 97u, false},
        {kRows, 65u, 97u, false}, {kRows, 17u, 0u, true},
        {kRows, 17u, 15u, true},  {kRows, 33u, 31u, true},
        {kRows, 65u, 63u, true},  {kRaggedRows, 17u, 15u, true},
    };
    for (const Case &c : cases) {
        if (!ok) break;
        ok = run_width(nullptr, c, baseline_scores, baseline_ids);
        if (ok && c.rows == kRows && c.tokens == 64u && !c.causal) {
            reference64_scores = baseline_scores;
            reference64_ids = baseline_ids;
        }
        for (const char *width : {"2", "4"}) {
            if (!ok) break;
            std::vector<float> candidate_scores;
            std::vector<uint32_t> candidate_ids;
            ok = run_width(width, c, candidate_scores, candidate_ids) &&
                 baseline_scores == candidate_scores && baseline_ids == candidate_ids;
            if (!ok) {
                std::fprintf(stderr,
                             "FAIL: width %s changes exact output rows=%u tokens=%u pos0=%u causal=%u\n",
                             width, c.rows, c.tokens, c.pos0, c.causal ? 1u : 0u);
            }
        }
    }

    for (const char *bad : {"", " ", "+1", "01", "3", "１２"}) {
        if (!ok) break;
        setenv("DS4_CUDA_GLM_INDEXER_QUERY_TILES", bad, 1);
        const bool rejected =
            ds4_gpu_glm_indexer_scores_batch_tensor(
                scores_gpu, q_gpu, weights_gpu, key_gpu, kRows, 1u, 97u,
                kHeads, kHeadDim, scale, false) == 0;
        if (!rejected) {
            std::fprintf(stderr, "FAIL: malformed width accepted bytes=%zu\n", std::strlen(bad));
            ok = false;
        }
    }

    if (ok) {
        setenv("DS4_CUDA_GLM_INDEXER_QUERY_TILES", "2", 1);
        ds4_gpu_set_quality(true);
        const bool rejected =
            ds4_gpu_glm_indexer_scores_batch_tensor(
                scores_gpu, q_gpu, weights_gpu, key_gpu, kRows, 1u, 97u,
                kHeads, kHeadDim, scale, false) == 0;
        ds4_gpu_set_quality(false);
        if (!rejected) {
            std::fprintf(stderr, "FAIL: width 2 accepted in quality mode\n");
            ok = false;
        }
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    const Case timed_case = {kRows, 64u, 97u, false};
    if (ok && (reference64_scores.empty() || reference64_ids.empty())) ok = false;
    if (ok) {
        ok = cudaEventCreate(&start) == cudaSuccess &&
             cudaEventCreate(&stop) == cudaSuccess;
    }
    for (size_t sequence = 0; ok && sequence < schedule.size(); ++sequence) {
        const std::string &width = schedule[sequence];
        setenv("DS4_CUDA_GLM_INDEXER_QUERY_TILES", width.c_str(), 1);
        ok = poison() && cudaEventRecord(start) == cudaSuccess &&
             ds4_gpu_glm_indexer_scores_batch_tensor(
                 scores_gpu, q_gpu, weights_gpu, key_gpu, timed_case.rows,
                 timed_case.tokens, timed_case.pos0, kHeads, kHeadDim,
                 scale, timed_case.causal) != 0 &&
             cudaEventRecord(stop) == cudaSuccess &&
             cudaEventSynchronize(stop) == cudaSuccess;
        float elapsed_ms = 0.0f;
        ok = ok && cudaEventElapsedTime(&elapsed_ms, start, stop) == cudaSuccess &&
             std::isfinite(elapsed_ms) && elapsed_ms > 0.0f &&
             ds4_gpu_indexer_topk_tensor(ids_gpu, scores_gpu, timed_case.rows,
                                         timed_case.tokens, kTopK) != 0 &&
             ds4_gpu_synchronize() != 0 &&
             ds4_gpu_tensor_read(scores_gpu, 0, score_host.data(),
                                 score_capacity * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(ids_gpu, 0, id_host.data(),
                                 id_capacity * sizeof(uint32_t)) != 0 &&
             validate_complete((uint64_t)timed_case.tokens * timed_case.rows,
                               (uint64_t)timed_case.tokens * kTopK,
                               width.c_str(), timed_case);
        if (ok) {
            ok = std::equal(reference64_scores.begin(), reference64_scores.end(),
                            score_host.begin()) &&
                 std::equal(reference64_ids.begin(), reference64_ids.end(),
                            id_host.begin());
        }
        if (ok) {
            const uint32_t width_value = (uint32_t)(width[0] - '0');
            std::printf("{\"kind\":\"timing\",\"sequence\":%zu,\"width\":%u,\"elapsed_ms\":%.9g,\"logical_k_bytes\":%llu,\"complete_write\":true,\"exact_scores\":true,\"exact_ids\":true,\"canaries_intact\":true}\n",
                        sequence, width_value, elapsed_ms,
                        (unsigned long long)logical_k_bytes(width_value));
        }
    }
    if (stop) cudaEventDestroy(stop);
    if (start) cudaEventDestroy(start);

    if (ok) {
        std::printf("{\"kind\":\"result\",\"verdict\":\"PASS\",\"correctness_cases\":12,\"causal_cases\":5,\"ragged_row_cases\":1,\"invalid_values_rejected\":6,\"quality_rejected\":true}\n");
    }
    unsetenv("DS4_CUDA_GLM_INDEXER_QUERY_TILES");
    ds4_gpu_tensor_free(ids_gpu);
    ds4_gpu_tensor_free(scores_gpu);
    ds4_gpu_tensor_free(weights_gpu);
    ds4_gpu_tensor_free(q_gpu);
    ds4_gpu_tensor_free(key_gpu);
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
