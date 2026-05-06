#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CURAND(call) \
    do { \
        curandStatus_t status = call; \
        if (status != CURAND_STATUS_SUCCESS) { \
            std::cerr << "cuRAND error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void float_to_half_kernel(const float* src, __half* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}

inline void FillRandomDevice(__half* d_ptr, size_t n, unsigned long long seed = 42)
{
    float* d_tmp;
    CHECK_CUDA(cudaMalloc(&d_tmp, n * sizeof(float)));

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CHECK_CURAND(curandGenerateUniform(gen, d_tmp, n));
    CHECK_CURAND(curandDestroyGenerator(gen));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    float_to_half_kernel<<<blocks, threads>>>(d_tmp, d_ptr, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_tmp));
}

__global__ void naive_attention_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int N, int d, float scale)
{
    int dd   = blockIdx.x * blockDim.x + threadIdx.x;
    int i    = blockIdx.y * blockDim.y + threadIdx.y;
    int head = blockIdx.z;

    if (dd >= d || i >= N) return;

    int head_off = head * N * d;
    const __half* q = Q + head_off;
    const __half* k = K + head_off;
    const __half* v = V + head_off;

    float row_max = -INFINITY;
    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        for (int c = 0; c < d; c++)
            dot += __half2float(q[i * d + c]) * __half2float(k[j * d + c]);
        row_max = fmaxf(row_max, dot * scale);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        for (int c = 0; c < d; c++)
            dot += __half2float(q[i * d + c]) * __half2float(k[j * d + c]);
        float p = expf(dot * scale - row_max);
        sum += p;
        acc += p * __half2float(v[j * d + dd]);
    }

    O[head_off + i * d + dd] = __float2half(acc / sum);
}

inline void NaiveAttentionGPU(
    const __half* d_Q, const __half* d_K, const __half* d_V, __half* d_O,
    int batch, int n_heads, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);
    dim3 block(32, 8);
    dim3 grid((d + block.x - 1) / block.x,
              (N + block.y - 1) / block.y,
              batch * n_heads);
    naive_attention_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d, scale);
    CHECK_CUDA(cudaDeviceSynchronize());
}

inline bool VerifyAttention(const __half* d_O, const __half* d_O_ref,
                            int size, float threshold = 0.05f)
{
    std::vector<__half> h_O(size), h_O_ref(size);
    CHECK_CUDA(cudaMemcpy(h_O.data(), d_O, size * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_O_ref.data(), d_O_ref, size * sizeof(__half),
                          cudaMemcpyDeviceToHost));

    double max_diff = 0.0, avg_diff = 0.0;
    int worst_idx = 0;
    for (int i = 0; i < size; i++) {
        double diff = std::fabs((double)__half2float(h_O[i]) -
                                (double)__half2float(h_O_ref[i]));
        if (diff > max_diff) { max_diff = diff; worst_idx = i; }
        avg_diff += diff;
    }
    avg_diff /= size;

    printf("  [verify] max_diff: %.6f (idx %d), avg_diff: %.6f",
           max_diff, worst_idx, avg_diff);

    if (avg_diff > threshold) { printf(" FAIL\n"); return false; }
    printf(" OK\n");
    return true;
}

struct BenchmarkResult {
    std::string label;
    int N;
    float time_ms;
    float tflops;
};

inline void WriteCSV(const std::vector<BenchmarkResult>& results,
                     const std::string& filename)
{
    std::ofstream file(filename);
    file << "Label,N,TimeMs,TFLOPS\n";
    for (const auto& r : results)
        file << "\"" << r.label << "\"," << r.N << ","
             << r.time_ms << "," << r.tflops << "\n";
    file.close();
}

inline double AttentionFLOPs(int batch, int n_heads, int N, int d)
{
    return (double)batch * n_heads * (4.0 * N * N * d + 6.0 * N * N);
}

template<typename Kernel>
BenchmarkResult RunBenchmarkNoVerify(
    const char* label,
    const __half* d_Q, const __half* d_K, const __half* d_V,
    __half* d_O,
    int batch, int n_heads, int N, int d,
    int warmup_runs = 5,
    int timed_runs = 20)
{
    for (int i = 0; i < warmup_runs; i++)
        Kernel::Run(d_Q, d_K, d_V, d_O, batch, n_heads, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < timed_runs; i++)
        Kernel::Run(d_Q, d_K, d_V, d_O, batch, n_heads, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / timed_runs;

    double flops = AttentionFLOPs(batch, n_heads, N, d);
    float tflops = static_cast<float>(flops / (avg_ms * 1e9));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::cout << label << ": " << avg_ms << " ms, "
              << tflops << " TFLOPS" << std::endl;

    return BenchmarkResult{label, N, avg_ms, tflops};
}
