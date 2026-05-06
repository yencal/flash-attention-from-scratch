#include <cstdio>
#include <vector>
#include <iostream>

#include "utils.cuh"
#include "01_base.cuh"
#include "02_swizzle.cuh"
#include "03_kvpipe.cuh"
#include "04_vpair.cuh"
#include "05_exp2f.cuh"
#include "06_batched.cuh"
#include "07_dbuf.cuh"

int main()
{
    constexpr int D = 128;

    // Verify correctness
    {
        constexpr int V_BATCH = 1, V_HEADS = 1, V_N = 1024;
        size_t v_total = (size_t)V_BATCH * V_HEADS * V_N * D;

        __half *d_Q, *d_K, *d_V, *d_O, *d_O_ref;
        CHECK_CUDA(cudaMalloc(&d_Q,     v_total * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_K,     v_total * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_V,     v_total * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_O,     v_total * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_O_ref, v_total * sizeof(__half)));

        FillRandomDevice(d_Q, v_total, 42);
        FillRandomDevice(d_K, v_total, 43);
        FillRandomDevice(d_V, v_total, 44);

        printf("Verifying correctness (batch=1, heads=1, N=%d)...\n", V_N);
        NaiveAttentionGPU(d_Q, d_K, d_V, d_O_ref, V_BATCH, V_HEADS, V_N, D);

        FlashBase<64, 32, D, 2>::Run(
            d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        CHECK_CUDA(cudaDeviceSynchronize());
        bool pass = VerifyAttention(d_O, d_O_ref, v_total);
        printf("  01_base:       %s\n", pass ? "PASS" : "FAIL");

        FlashSwizzle<64, 128, D, 4>::Run(
            d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        CHECK_CUDA(cudaDeviceSynchronize());
        bool pass2 = VerifyAttention(d_O, d_O_ref, v_total);
        printf("  02_swizzle:    %s\n", pass2 ? "PASS" : "FAIL");

        FlashKVPipe<64, 128, D, 4>::Run(
            d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        CHECK_CUDA(cudaDeviceSynchronize());
        bool pass3 = VerifyAttention(d_O, d_O_ref, v_total);
        printf("  03_kvpipe:     %s\n", pass3 ? "PASS" : "FAIL");

        FlashVPair<64, 128, D, 4>::Run(
            d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        CHECK_CUDA(cudaDeviceSynchronize());
        bool pass4 = VerifyAttention(d_O, d_O_ref, v_total);
        printf("  04_vpair:      %s\n", pass4 ? "PASS" : "FAIL");

        FlashExp2f<64, 128, D, 4>::Run(
            d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        CHECK_CUDA(cudaDeviceSynchronize());
        bool pass5 = VerifyAttention(d_O, d_O_ref, v_total);
        printf("  05_exp2f:      %s\n", pass5 ? "PASS" : "FAIL");

        FlashBatched<128, 64, D, 4>::Run(
            d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        CHECK_CUDA(cudaDeviceSynchronize());
        bool pass6 = VerifyAttention(d_O, d_O_ref, v_total);
        printf("  06_batched:    %s\n", pass6 ? "PASS" : "FAIL");

        FlashDbuf<128, 64, D, 4>::Run(
            d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        CHECK_CUDA(cudaDeviceSynchronize());
        bool pass7 = VerifyAttention(d_O, d_O_ref, v_total);
        printf("  07_dbuf:       %s\n", pass7 ? "PASS" : "FAIL");

        pass = pass && pass2 && pass3 && pass4 && pass5 && pass6 && pass7;

        CHECK_CUDA(cudaFree(d_Q));
        CHECK_CUDA(cudaFree(d_K));
        CHECK_CUDA(cudaFree(d_V));
        CHECK_CUDA(cudaFree(d_O));
        CHECK_CUDA(cudaFree(d_O_ref));

        if (!pass) return 1;
    }

    // Benchmark
    constexpr int BATCH   = 16;
    constexpr int N_HEADS = 16;

    std::vector<int> seq_lens = {256, 512, 1024, 2048, 4096, 8192};
    std::vector<BenchmarkResult> results;

    printf("\nFlash Attention Benchmark\n");
    printf("batch=%d, heads=%d, d=%d\n", BATCH, N_HEADS, D);

    for (int N : seq_lens) {
        size_t total = (size_t)BATCH * N_HEADS * N * D;

        printf("\n=== N = %d (%.2f GFLOPs) ===\n", N,
               AttentionFLOPs(BATCH, N_HEADS, N, D) / 1e9);

        __half *d_Q, *d_K, *d_V, *d_O;
        CHECK_CUDA(cudaMalloc(&d_Q, total * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_K, total * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_V, total * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_O, total * sizeof(__half)));

        FillRandomDevice(d_Q, total, 42);
        FillRandomDevice(d_K, total, 43);
        FillRandomDevice(d_V, total, 44);

        results.push_back(RunBenchmarkNoVerify<FlashBase<64, 32, D, 2>>(
            "01_base", d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N, D));

        results.push_back(RunBenchmarkNoVerify<FlashSwizzle<64, 128, D, 4>>(
            "02_swizzle", d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N, D));

        results.push_back(RunBenchmarkNoVerify<FlashKVPipe<64, 128, D, 4>>(
            "03_kvpipe", d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N, D));

        results.push_back(RunBenchmarkNoVerify<FlashVPair<64, 128, D, 4>>(
            "04_vpair", d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N, D));

        results.push_back(RunBenchmarkNoVerify<FlashExp2f<64, 128, D, 4>>(
            "05_exp2f", d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N, D));

        results.push_back(RunBenchmarkNoVerify<FlashBatched<128, 64, D, 4>>(
            "06_batched", d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N, D));

        results.push_back(RunBenchmarkNoVerify<FlashDbuf<128, 64, D, 4>>(
            "07_dbuf", d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N, D));

        CHECK_CUDA(cudaFree(d_Q));
        CHECK_CUDA(cudaFree(d_K));
        CHECK_CUDA(cudaFree(d_V));
        CHECK_CUDA(cudaFree(d_O));
    }

    WriteCSV(results, "flash_attention_results.csv");
    std::cout << "\nResults saved to flash_attention_results.csv" << std::endl;

    return 0;
}
