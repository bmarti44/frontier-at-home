#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr uint64_t kBytes = 2ull * 1024ull * 1024ull * 1024ull;
constexpr int kSamples = 5;
constexpr int kBlocks = 4096;
constexpr int kThreads = 256;

[[noreturn]] void fail(const char *operation, cudaError_t error) {
    std::fprintf(
        stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
    std::exit(1);
}

void require_cuda(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) fail(operation, error);
}

__global__ void initialize_kernel(float4 *values, uint64_t count) {
    const uint64_t start =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride =
        static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = start; index < count; index += stride) {
        const float value = static_cast<float>((index % 251u) + 1u) / 251.0f;
        values[index] = make_float4(value, value + 1.0f, value + 2.0f,
                                    value + 3.0f);
    }
}

__global__ void read_bandwidth_kernel(
    const float4 *__restrict__ values,
    uint64_t count,
    float *__restrict__ block_sums) {
    float sum = 0.0f;
    const uint64_t start =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride =
        static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = start; index < count; index += stride) {
        const float4 value = values[index];
        sum += value.x + value.y + value.z + value.w;
    }

    __shared__ float scratch[kThreads];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) block_sums[blockIdx.x] = scratch[0];
}

}  // namespace

int main() {
    static_assert(kBytes % sizeof(float4) == 0);
    const uint64_t element_count = kBytes / sizeof(float4);
    float4 *values = nullptr;
    float *block_sums = nullptr;
    require_cuda(cudaMalloc(&values, kBytes), "cudaMalloc(values)");
    require_cuda(
        cudaMalloc(&block_sums, kBlocks * sizeof(float)),
        "cudaMalloc(block_sums)");

    initialize_kernel<<<kBlocks, kThreads>>>(values, element_count);
    require_cuda(cudaGetLastError(), "initialize kernel launch");
    require_cuda(cudaDeviceSynchronize(), "initialize synchronization");

    for (int warmup = 0; warmup < 2; warmup++) {
        read_bandwidth_kernel<<<kBlocks, kThreads>>>(
            values, element_count, block_sums);
        require_cuda(cudaGetLastError(), "warmup kernel launch");
    }
    require_cuda(cudaDeviceSynchronize(), "warmup synchronization");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    require_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    require_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    double bandwidth[kSamples] = {};
    double elapsed[kSamples] = {};
    for (int sample = 0; sample < kSamples; sample++) {
        require_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
        read_bandwidth_kernel<<<kBlocks, kThreads>>>(
            values, element_count, block_sums);
        require_cuda(cudaGetLastError(), "measurement kernel launch");
        require_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
        require_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        float milliseconds = 0.0f;
        require_cuda(
            cudaEventElapsedTime(&milliseconds, start, stop),
            "cudaEventElapsedTime");
        if (!(milliseconds > 0.0f) || !std::isfinite(milliseconds)) {
            std::fprintf(stderr, "invalid CUDA event duration\n");
            return 1;
        }
        elapsed[sample] = milliseconds;
        bandwidth[sample] =
            static_cast<double>(kBytes) / (milliseconds / 1000.0) / 1.0e9;
    }

    std::vector<float> host_sums(kBlocks);
    require_cuda(
        cudaMemcpy(host_sums.data(), block_sums,
                   kBlocks * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy(checksum)");
    double checksum = 0.0;
    for (float value : host_sums) checksum += value;
    if (!std::isfinite(checksum) || checksum == 0.0) {
        std::fprintf(stderr, "invalid read checksum\n");
        return 1;
    }

    std::printf(
        "{\"schema_version\":1,\"bytes\":%llu,\"samples\":%d,"
        "\"bandwidth_gb_s\":[",
        static_cast<unsigned long long>(kBytes), kSamples);
    for (int sample = 0; sample < kSamples; sample++) {
        std::printf("%s%.17g", sample ? "," : "", bandwidth[sample]);
    }
    std::printf("],\"elapsed_ms\":[");
    for (int sample = 0; sample < kSamples; sample++) {
        std::printf("%s%.17g", sample ? "," : "", elapsed[sample]);
    }
    std::printf("],\"checksum\":%.17g}\n", checksum);

    require_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    require_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    require_cuda(cudaFree(block_sums), "cudaFree(block_sums)");
    require_cuda(cudaFree(values), "cudaFree(values)");
    return 0;
}
