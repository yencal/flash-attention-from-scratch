#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// cp.async.cg: 16-byte async copy, GMEM -> SMEM, bypasses L1 (uses L2 only)
__device__ __forceinline__
void cp_async_cg(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__
void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__
void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

template<int STRIDE>
__device__ __forceinline__
int swizzle_offset(int row, int col) {
    return row * STRIDE + (((col >> 3) ^ (row & 7)) << 3) + (col & 7);
}

// Cooperative tile loader: all threads load [ROWS x COLS] from GMEM to SMEM.
// Each cp.async copies 16 bytes = 8 FP16 elements.
template<int ROWS, int COLS, int NUM_THREADS>
__device__ void load_tile_async(
    const __half* gmem_ptr,
    __half* smem_ptr,
    int gmem_stride,
    int tid)
{
    constexpr int TOTAL_VEC = (ROWS * COLS) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    static_assert((ROWS * COLS) % 8 == 0);
    static_assert(TOTAL_VEC % NUM_THREADS == 0);
    static_assert(COLS % 8 == 0);

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; i++) {
        int idx = tid + i * NUM_THREADS;
        int row = idx / (COLS / 8);
        int col8 = idx % (COLS / 8);
        cp_async_cg(
            &smem_ptr[row * COLS + col8 * 8],
            &gmem_ptr[row * gmem_stride + col8 * 8]
        );
    }
}

template<int ROWS, int COLS, int NUM_THREADS>
__device__ void load_tile_async_swizzled(
    const __half* gmem_ptr,
    __half* smem_ptr,
    int gmem_stride,
    int tid)
{
    constexpr int TOTAL_VEC = (ROWS * COLS) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    static_assert((ROWS * COLS) % 8 == 0);
    static_assert(TOTAL_VEC % NUM_THREADS == 0);
    static_assert(COLS % 8 == 0);

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; i++) {
        int idx = tid + i * NUM_THREADS;
        int row = idx / (COLS / 8);
        int col8 = idx % (COLS / 8);
        int sw_col8 = col8 ^ (row & 7);
        cp_async_cg(
            &smem_ptr[row * COLS + sw_col8 * 8],
            &gmem_ptr[row * gmem_stride + col8 * 8]
        );
    }
}
