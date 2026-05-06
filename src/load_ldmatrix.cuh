#pragma once

#include <cuda_fp16.h>
#include "fragment.cuh"
#include "kernel_helpers.cuh"

// Load A fragment (16x16) via ldmatrix.x4
// smem_ptr: top-left of the 16x16 sub-tile in SMEM
// STRIDE: leading dimension of the SMEM tile (FP16 elements)
template<int STRIDE>
__device__ __forceinline__
void load_fragment_ldmatrix(FragmentA& frag, const __half* smem_ptr) {
    int lane = threadIdx.x & 31;

    int tile_id = lane >> 3;
    int row_in_tile = lane & 7;
    int row = (tile_id & 1) * 8 + row_in_tile;
    int col = (tile_id >> 1) * 8;

    uint32_t addr = __cvta_generic_to_shared(smem_ptr + row * STRIDE + col);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1]),
          "=r"(frag.reg[2]), "=r"(frag.reg[3])
        : "r"(addr)
    );
}

// Load two B fragments from K stored row-major, interpreted as K^T.
// ldmatrix.x4 loads 16x16 of K, registers reordered into two 16x8 B fragments.
//
//   r0 = K[n=0..7,  k=0..7]    r2 = K[n=0..7,  k=8..15]
//   r1 = K[n=8..15, k=0..7]    r3 = K[n=8..15, k=8..15]
//
//   b0 (n=0..7):  {r0, r2}
//   b1 (n=8..15): {r1, r3}
template<int STRIDE>
__device__ __forceinline__
void load_kt_fragment_pair(FragmentB& b0, FragmentB& b1,
                           const __half* smem_ptr) {
    int lane = threadIdx.x & 31;

    int tile_id = lane >> 3;
    int row_in_tile = lane & 7;
    int row = (tile_id & 1) * 8 + row_in_tile;
    int col = (tile_id >> 1) * 8;

    uint32_t addr = __cvta_generic_to_shared(smem_ptr + row * STRIDE + col);

    uint32_t r0, r1, r2, r3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );

    b0.reg[0] = r0;  b0.reg[1] = r2;
    b1.reg[0] = r1;  b1.reg[1] = r3;
}

// Load B fragment (16x8) via ldmatrix.x2.trans
// Transposes each 8x8 sub-tile during load: row-major SMEM -> column-major registers.
// Used for V in the PV GEMM.
template<int STRIDE>
__device__ __forceinline__
void load_fragment_ldmatrix(FragmentB& frag, const __half* smem_ptr) {
    int lane = threadIdx.x & 31;
    int row = lane & 15;

    uint32_t addr = __cvta_generic_to_shared(smem_ptr + row * STRIDE);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1])
        : "r"(addr)
    );
}

// =========================================================================
// Swizzled variants: same logic, swizzle_offset replaces row*STRIDE+col
// =========================================================================

template<int STRIDE>
__device__ __forceinline__
void load_a_swizzled(FragmentA& frag, const __half* smem_tile,
                     int row_offset, int col_offset) {
    int lane = threadIdx.x & 31;
    int tile_id = lane >> 3;
    int row_in_tile = lane & 7;
    int row = row_offset + (tile_id & 1) * 8 + row_in_tile;
    int col = col_offset + (tile_id >> 1) * 8;

    uint32_t addr = __cvta_generic_to_shared(
        &smem_tile[swizzle_offset<STRIDE>(row, col)]);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1]),
          "=r"(frag.reg[2]), "=r"(frag.reg[3])
        : "r"(addr)
    );
}

template<int STRIDE>
__device__ __forceinline__
void load_kt_pair_swizzled(FragmentB& b0, FragmentB& b1,
                           const __half* smem_tile,
                           int row_offset, int col_offset) {
    int lane = threadIdx.x & 31;
    int tile_id = lane >> 3;
    int row_in_tile = lane & 7;
    int row = row_offset + (tile_id & 1) * 8 + row_in_tile;
    int col = col_offset + (tile_id >> 1) * 8;

    uint32_t addr = __cvta_generic_to_shared(
        &smem_tile[swizzle_offset<STRIDE>(row, col)]);

    uint32_t r0, r1, r2, r3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );

    b0.reg[0] = r0;  b0.reg[1] = r2;
    b1.reg[0] = r1;  b1.reg[1] = r3;
}

// Load single K^T B-fragment via ldmatrix.x2.trans
// K stored row-major [BC, D]. Same 8 rows, two column groups (col, col+8)
// gives k=0..7 in r0 and k=8..15 in r1 for the same 8 n-values.
template<int STRIDE>
__device__ __forceinline__
void load_kt_single_swizzled(FragmentB& frag, const __half* smem_tile,
                             int row_offset, int col_offset) {
    int lane = threadIdx.x & 31;
    int row = row_offset + (lane & 7);
    int col = col_offset + ((lane >> 3) & 1) * 8;

    uint32_t addr = __cvta_generic_to_shared(
        &smem_tile[swizzle_offset<STRIDE>(row, col)]);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1])
        : "r"(addr)
    );
}

template<int STRIDE>
__device__ __forceinline__
void load_b_swizzled(FragmentB& frag, const __half* smem_tile,
                     int row_offset, int col_offset) {
    int lane = threadIdx.x & 31;
    int row = row_offset + (lane & 15);

    uint32_t addr = __cvta_generic_to_shared(
        &smem_tile[swizzle_offset<STRIDE>(row, col_offset)]);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1])
        : "r"(addr)
    );
}

template<int STRIDE>
__device__ __forceinline__
void load_vt_pair_swizzled(FragmentB& b0, FragmentB& b1,
                           const __half* smem_tile,
                           int row_offset, int col_offset) {
    int lane = threadIdx.x & 31;
    int tile_id = lane >> 3;
    int row_in_tile = lane & 7;
    int row = row_offset + (tile_id & 1) * 8 + row_in_tile;
    int col = col_offset + (tile_id >> 1) * 8;

    uint32_t addr = __cvta_generic_to_shared(
        &smem_tile[swizzle_offset<STRIDE>(row, col)]);

    uint32_t r0, r1, r2, r3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );

    b0.reg[0] = r0;  b0.reg[1] = r1;
    b1.reg[0] = r2;  b1.reg[1] = r3;
}
