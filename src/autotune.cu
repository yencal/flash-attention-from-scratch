// autotune.cu
// Sweep tile configs (BR, BC, NUM_WARPS) for all kernels.
// Usage: ./autotune [start_kernel] [end_kernel]
//   defaults: 01 07

#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <vector>
#include <string>
#include <functional>
#include <cuda_runtime.h>

#include "utils.cuh"
#include "01_base.cuh"
#include "02_swizzle.cuh"
#include "03_kvpipe.cuh"
#include "04_vpair.cuh"
#include "05_exp2f.cuh"
#include "06_batched.cuh"
#include "07_dbuf.cuh"

struct TuneConfig {
    std::string name;
    std::function<void(const __half*, const __half*, const __half*,
                       __half*, int, int, int)> run;
};

// All tile configs: M_TILES=1 and M_TILES=2
template<template<int, int, int, int> class Kernel>
std::vector<TuneConfig> GetAllVariants(const char* prefix) {
    std::vector<TuneConfig> v;
    auto add = [&](const char* tag, auto fn) {
        v.push_back({std::string(prefix) + "_" + tag, fn});
    };
    // M_TILES=1 (BR = NUM_WARPS * 16)
    add("64x32_W4",   Kernel<64, 32, 128, 4>::Run);
    add("64x64_W4",   Kernel<64, 64, 128, 4>::Run);
    add("64x128_W4",  Kernel<64, 128, 128, 4>::Run);
    add("128x32_W8",  Kernel<128, 32, 128, 8>::Run);
    add("128x64_W8",  Kernel<128, 64, 128, 8>::Run);
    add("128x128_W8", Kernel<128, 128, 128, 8>::Run);
    // M_TILES=2 (BR = NUM_WARPS * 32)
    add("64x32_W2",   Kernel<64, 32, 128, 2>::Run);
    add("64x64_W2",   Kernel<64, 64, 128, 2>::Run);
    add("64x128_W2",  Kernel<64, 128, 128, 2>::Run);
    add("128x32_W4",  Kernel<128, 32, 128, 4>::Run);
    add("128x64_W4",  Kernel<128, 64, 128, 4>::Run);
    add("128x128_W4", Kernel<128, 128, 128, 4>::Run);
    return v;
}

void RunAutotune(const char* kernel_name,
                 const std::vector<TuneConfig>& variants,
                 const __half* Q, const __half* K, const __half* V,
                 __half* O, __half* O_ref,
                 int batch, int n_heads, int N, int d,
                 int warmup = 3, int iters = 10)
{
    printf("\n--- %s (N=%d) ---\n", kernel_name, N);

    float best_time = FLT_MAX;
    std::string best_name;
    double best_tflops = 0;

    for (const auto& config : variants) {
        for (int i = 0; i < warmup; i++)
            config.run(Q, K, V, O, batch, n_heads, N);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  %-28s SKIP (%s)\n", config.name.c_str(),
                   cudaGetErrorString(err));
            cudaGetLastError();
            continue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < iters; i++)
            config.run(Q, K, V, O, batch, n_heads, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= iters;

        double tflops = AttentionFLOPs(batch, n_heads, N, d) / (ms * 1e9);
        printf("  %-28s %7.3f ms  %6.1f TFLOPS\n",
               config.name.c_str(), ms, tflops);

        if (ms < best_time) {
            best_time = ms;
            best_name = config.name;
            best_tflops = tflops;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("  >> Best: %-28s %.1f TFLOPS\n", best_name.c_str(), best_tflops);
}

int main(int argc, char** argv) {
    int k_start = 1, k_end = 7;
    if (argc >= 2) k_start = atoi(argv[1]);
    if (argc >= 3) k_end = atoi(argv[2]);

    printf("Autotuning kernels %02d-%02d\n", k_start, k_end);

    constexpr int D = 128;
    constexpr int BATCH = 16, N_HEADS = 16;
    std::vector<int> seq_lens = {1024, 2048, 4096, 8192};

    struct KernelEntry {
        int id;
        const char* name;
        std::vector<TuneConfig> variants;
    };

    std::vector<KernelEntry> kernels;
    if (k_start <= 1 && k_end >= 1)
        kernels.push_back({1, "01_base",
            GetAllVariants<FlashBase>("01")});
    if (k_start <= 2 && k_end >= 2)
        kernels.push_back({2, "02_swizzle",
            GetAllVariants<FlashSwizzle>("02")});
    if (k_start <= 3 && k_end >= 3)
        kernels.push_back({3, "03_kvpipe",
            GetAllVariants<FlashKVPipe>("03")});
    if (k_start <= 4 && k_end >= 4)
        kernels.push_back({4, "04_vpair",
            GetAllVariants<FlashVPair>("04")});
    if (k_start <= 5 && k_end >= 5)
        kernels.push_back({5, "05_exp2f",
            GetAllVariants<FlashExp2f>("05")});
    if (k_start <= 6 && k_end >= 6)
        kernels.push_back({6, "06_batched",
            GetAllVariants<FlashBatched>("06")});
    if (k_start <= 7 && k_end >= 7)
        kernels.push_back({7, "07_dbuf",
            GetAllVariants<FlashDbuf>("07")});

    for (int N : seq_lens) {
        size_t total = (size_t)BATCH * N_HEADS * N * D;

        __half *d_Q, *d_K, *d_V, *d_O, *d_O_ref;
        CHECK_CUDA(cudaMalloc(&d_Q, total * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_K, total * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_V, total * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_O, total * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_O_ref, total * sizeof(__half)));

        FillRandomDevice(d_Q, total, 42);
        FillRandomDevice(d_K, total, 43);
        FillRandomDevice(d_V, total, 44);

        printf("\n========== N = %d (%.2f GFLOPs) ==========\n",
               N, AttentionFLOPs(BATCH, N_HEADS, N, D) / 1e9);

        for (auto& ke : kernels) {
            RunAutotune(ke.name, ke.variants,
                        d_Q, d_K, d_V, d_O, d_O_ref,
                        BATCH, N_HEADS, N, D);
        }

        CHECK_CUDA(cudaFree(d_Q));
        CHECK_CUDA(cudaFree(d_K));
        CHECK_CUDA(cudaFree(d_V));
        CHECK_CUDA(cudaFree(d_O));
        CHECK_CUDA(cudaFree(d_O_ref));
    }

    return 0;
}
