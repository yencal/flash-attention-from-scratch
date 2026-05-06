// profile.cu
// Minimal driver for NCU profiling - one kernel per run
// Usage: ./profile <kernel_num>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "utils.cuh"
#include "01_base.cuh"
#include "02_swizzle.cuh"
#include "03_kvpipe.cuh"
#include "04_vpair.cuh"
#include "05_exp2f.cuh"
#include "06_batched.cuh"
#include "07_dbuf.cuh"

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <kernel_num>\n", argv[0]);
        printf("  01 = FlashBase       (64x32)\n");
        printf("  02 = FlashSwizzle    (64x128)\n");
        printf("  03 = FlashKVPipe     (64x128)\n");
        printf("  04 = FlashVPair      (64x128)\n");
        printf("  05 = FlashExp2f      (64x128)\n");
        printf("  06 = FlashBatched    (128x64)\n");
        printf("  07 = FlashDbuf       (128x64)\n");
        return 1;
    }

    constexpr int D = 128;
    const char* kernel = argv[1];
    int knum = atoi(kernel);

    // ---- Verify correctness at small config ----
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

        if (knum == 1) {
            FlashBase<64, 32, D, 2>::Run(
                d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        } else if (knum == 2) {
            FlashSwizzle<64, 128, D, 4>::Run(
                d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        } else if (knum == 3) {
            FlashKVPipe<64, 128, D, 4>::Run(
                d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        } else if (knum == 4) {
            FlashVPair<64, 128, D, 4>::Run(
                d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        } else if (knum == 5) {
            FlashExp2f<64, 128, D, 4>::Run(
                d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        } else if (knum == 6) {
            FlashBatched<128, 64, D, 4>::Run(
                d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        } else if (knum == 7) {
            FlashDbuf<128, 64, D, 4>::Run(
                d_Q, d_K, d_V, d_O, V_BATCH, V_HEADS, V_N);
        } else {
            printf("Unknown kernel: %s\n", kernel);
            return 1;
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        bool pass = VerifyAttention(d_O, d_O_ref, v_total);
        printf("  Kernel %s: %s\n", kernel, pass ? "PASS" : "FAIL");

        CHECK_CUDA(cudaFree(d_Q));
        CHECK_CUDA(cudaFree(d_K));
        CHECK_CUDA(cudaFree(d_V));
        CHECK_CUDA(cudaFree(d_O));
        CHECK_CUDA(cudaFree(d_O_ref));

        if (!pass) return 1;
    }

    // ---- Profile at full config ----
    constexpr int BATCH   = 16;
    constexpr int N_HEADS = 16;
    constexpr int N       = 8192;
    size_t total = (size_t)BATCH * N_HEADS * N * D;

    __half *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, total * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_K, total * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_V, total * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_O, total * sizeof(__half)));

    FillRandomDevice(d_Q, total, 42);
    FillRandomDevice(d_K, total, 43);
    FillRandomDevice(d_V, total, 44);

    printf("\nProfiling kernel %s (batch=%d, heads=%d, N=%d, d=%d)...\n",
           kernel, BATCH, N_HEADS, N, D);

    if (knum == 1) {
        FlashBase<64, 32, D, 2>::Run(d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N);
    } else if (knum == 2) {
        FlashSwizzle<64, 128, D, 4>::Run(d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N);
    } else if (knum == 3) {
        FlashKVPipe<64, 128, D, 4>::Run(d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N);
    } else if (knum == 4) {
        FlashVPair<64, 128, D, 4>::Run(d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N);
    } else if (knum == 5) {
        FlashExp2f<64, 128, D, 4>::Run(d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N);
    } else if (knum == 6) {
        FlashBatched<128, 64, D, 4>::Run(d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N);
    } else if (knum == 7) {
        FlashDbuf<128, 64, D, 4>::Run(d_Q, d_K, d_V, d_O, BATCH, N_HEADS, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Kernel %s launched successfully.\n", kernel);

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    return 0;
}
