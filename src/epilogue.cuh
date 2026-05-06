#pragma once

#include <cuda_fp16.h>
#include "fragment.cuh"
#include "kernel_helpers.cuh"

// Split epilogue for multi-tile warps.
// Phase 1: normalize O by l, FP32->FP16 (paired cvt), scatter to SMEM.
template<int D, int PV_N_TILES>
__device__ __forceinline__
void epilogue_scatter(
    FragmentC O_accum[PV_N_TILES],
    float l[2],
    __half* smem,
    int warp_row_offset)
{
    int lane = threadIdx.x & 31;
    float inv_l[2] = { 1.0f / l[0], 1.0f / l[1] };

    #pragma unroll
    for (int n = 0; n < PV_N_TILES; n++) {
        int col_offset = n * MMA_N;
        #pragma unroll
        for (int pair = 0; pair < 2; pair++) {
            int base_i = pair * 2;
            int row = warp_row_offset + FragmentC::get_row(lane, base_i);
            int col = col_offset + FragmentC::get_col(lane, base_i);
            float v0 = O_accum[n].reg[base_i]     * inv_l[pair];
            float v1 = O_accum[n].reg[base_i + 1] * inv_l[pair];
            uint32_t packed;
            asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n"
                : "=r"(packed) : "f"(v1), "f"(v0));
            *reinterpret_cast<uint32_t*>(
                &smem[row * D + col]) = packed;
        }
    }
}

// Phase 2: coalesced SMEM -> GMEM store. Called once after __syncthreads.
template<int BR, int D, int NUM_THREADS>
__device__ __forceinline__
void epilogue_store(
    __half* smem,
    __half* O_gmem,
    int gmem_stride)
{
    int tid = threadIdx.x;
    constexpr int TOTAL_VEC = (BR * D) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    static_assert(D % 8 == 0);
    static_assert(TOTAL_VEC % NUM_THREADS == 0);

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; i++) {
        int vec_idx = tid + i * NUM_THREADS;
        int row = vec_idx / (D / 8);
        int col8 = vec_idx % (D / 8);
        float4 val = *reinterpret_cast<float4*>(&smem[row * D + col8 * 8]);
        *reinterpret_cast<float4*>(&O_gmem[row * gmem_stride + col8 * 8]) = val;
    }
}

// =========================================================================
// Swizzled variants
// =========================================================================

template<int D, int PV_N_TILES>
__device__ __forceinline__
void epilogue_scatter_swizzled(
    FragmentC O_accum[PV_N_TILES],
    float l[2],
    __half* smem,
    int warp_row_offset)
{
    int lane = threadIdx.x & 31;
    float inv_l[2] = { 1.0f / l[0], 1.0f / l[1] };

    #pragma unroll
    for (int n = 0; n < PV_N_TILES; n++) {
        int col_offset = n * MMA_N;
        #pragma unroll
        for (int pair = 0; pair < 2; pair++) {
            int base_i = pair * 2;
            int row = warp_row_offset + FragmentC::get_row(lane, base_i);
            int col = col_offset + FragmentC::get_col(lane, base_i);
            float v0 = O_accum[n].reg[base_i]     * inv_l[pair];
            float v1 = O_accum[n].reg[base_i + 1] * inv_l[pair];
            uint32_t packed;
            asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n"
                : "=r"(packed) : "f"(v1), "f"(v0));
            *reinterpret_cast<uint32_t*>(
                &smem[swizzle_offset<D>(row, col)]) = packed;
        }
    }
}

template<int BR, int D, int NUM_THREADS>
__device__ __forceinline__
void epilogue_store_swizzled(
    __half* smem,
    __half* O_gmem,
    int gmem_stride)
{
    int tid = threadIdx.x;
    constexpr int TOTAL_VEC = (BR * D) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    static_assert(D % 8 == 0);
    static_assert(TOTAL_VEC % NUM_THREADS == 0);

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; i++) {
        int vec_idx = tid + i * NUM_THREADS;
        int row = vec_idx / (D / 8);
        int col8 = vec_idx % (D / 8);
        int sw_col8 = col8 ^ (row & 7);
        float4 val = *reinterpret_cast<float4*>(&smem[row * D + sw_col8 * 8]);
        *reinterpret_cast<float4*>(&O_gmem[row * gmem_stride + col8 * 8]) = val;
    }
}
