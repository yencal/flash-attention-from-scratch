#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "kernel_helpers.cuh"
#include "online_softmax.cuh"
#include "gemm.cuh"
#include "epilogue.cuh"

template<int BR, int BC, int D, int NUM_WARPS>
__global__ void flash_swizzle(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int N)
{
    constexpr int NUM_THREADS = NUM_WARPS * 32;
    constexpr int M_TILES = BR / (NUM_WARPS * MMA_M);

    constexpr int QK_N_TILES = BC / MMA_N;
    constexpr int QK_K_TILES = D / MMA_K;
    constexpr int PV_N_TILES = D / MMA_N;

    constexpr int SMEM_K_OFF = BR * D;
    constexpr int SMEM_V_OFF = BR * D + BC * D;

    const float SOFTMAX_SCALE = 1.0f / sqrtf((float)D);

    static_assert(BR % (NUM_WARPS * MMA_M) == 0);
    static_assert(BC % (2 * MMA_N) == 0);
    static_assert(D % MMA_K == 0);
    static_assert((BR * D) % (NUM_THREADS * 8) == 0);
    static_assert((BC * D) % (NUM_THREADS * 8) == 0);

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int warp_base_row = warp_id * M_TILES * MMA_M;

    int q_block = blockIdx.x;
    int head    = blockIdx.y;
    int batch   = blockIdx.z;

    int head_stride  = N * D;
    int batch_stride = gridDim.y * head_stride;

    const __half* Q_head = Q + batch * batch_stride + head * head_stride;
    const __half* K_head = K + batch * batch_stride + head * head_stride;
    const __half* V_head = V + batch * batch_stride + head * head_stride;
    __half*       O_head = O + batch * batch_stride + head * head_stride;

    const __half* Q_block_ptr = Q_head + q_block * BR * D;
    __half*       O_block_ptr = O_head + q_block * BR * D;

    extern __shared__ __half smem[];
    __half* smem_Q = smem;
    __half* smem_K = smem + SMEM_K_OFF;
    __half* smem_V = smem + SMEM_V_OFF;

    // Prologue: load Q (swizzled) — stays in SMEM for entire mainloop
    load_tile_async_swizzled<BR, D, NUM_THREADS>(Q_block_ptr, smem_Q, D, tid);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    // Init per-thread state
    FragmentC O_accum[M_TILES][PV_N_TILES];
    #pragma unroll
    for (int mt = 0; mt < M_TILES; mt++)
        #pragma unroll
        for (int n = 0; n < PV_N_TILES; n++)
            O_accum[mt][n].fill(0.0f);

    float m[2 * M_TILES];
    float l[2 * M_TILES];
    #pragma unroll
    for (int i = 0; i < 2 * M_TILES; i++) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }

    // Mainloop
    int n_kv_blocks = N / BC;

    for (int j = 0; j < n_kv_blocks; j++) {
        load_tile_async_swizzled<BC, D, NUM_THREADS>(
            K_head + j * BC * D, smem_K, D, tid);
        load_tile_async_swizzled<BC, D, NUM_THREADS>(
            V_head + j * BC * D, smem_V, D, tid);
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();

        #pragma unroll
        for (int mt = 0; mt < M_TILES; mt++) {
            FragmentC S_accum[QK_N_TILES];
            gemm_qk_smemq<QK_K_TILES, QK_N_TILES, D>(
                S_accum, smem_Q, smem_K,
                warp_base_row + mt * MMA_M);

            #pragma unroll
            for (int n = 0; n < QK_N_TILES; n++) {
                S_accum[n].reg[0] *= SOFTMAX_SCALE;
                S_accum[n].reg[1] *= SOFTMAX_SCALE;
                S_accum[n].reg[2] *= SOFTMAX_SCALE;
                S_accum[n].reg[3] *= SOFTMAX_SCALE;
            }

            if (j == 0)
                online_softmax<true,  QK_N_TILES, PV_N_TILES>(
                    S_accum, O_accum[mt], &m[mt * 2], &l[mt * 2]);
            else
                online_softmax<false, QK_N_TILES, PV_N_TILES>(
                    S_accum, O_accum[mt], &m[mt * 2], &l[mt * 2]);

            gemm_pv_regp_swizzled<QK_N_TILES, PV_N_TILES, D>(
                O_accum[mt], S_accum, smem_V);
        }

        __syncthreads();
    }

    // Epilogue
    #pragma unroll
    for (int mt = 0; mt < M_TILES; mt++)
        epilogue_scatter_swizzled<D, PV_N_TILES>(
            O_accum[mt], &l[mt * 2], smem_Q,
            warp_base_row + mt * MMA_M);

    __syncthreads();

    epilogue_store_swizzled<BR, D, NUM_THREADS>(
        smem_Q, O_block_ptr, D);
}

template<int BR, int BC, int D, int NUM_WARPS>
struct FlashSwizzle {
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr size_t SMEM_BYTES = (BR + 2 * BC) * D * sizeof(__half);

    static void Run(const __half* Q, const __half* K, const __half* V,
                    __half* O, int batch, int n_heads, int N) {
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(
                flash_swizzle<BR, BC, D, NUM_WARPS>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_BYTES
            );
            configured = true;
        }

        dim3 grid(N / BR, n_heads, batch);
        dim3 block(NUM_THREADS);
        flash_swizzle<BR, BC, D, NUM_WARPS>
            <<<grid, block, SMEM_BYTES>>>(Q, K, V, O, N);
    }
};
