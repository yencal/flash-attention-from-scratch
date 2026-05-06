#pragma once

#include "fragment.cuh"

template<int N_TILES>
__device__ __forceinline__
float compute_rowmax(const FragmentC S_accum[], int row) {
    float row_max = -INFINITY;
    #pragma unroll
    for (int n = 0; n < N_TILES; n++) {
        row_max = fmaxf(row_max, S_accum[n].reg[row * 2 + 0]);
        row_max = fmaxf(row_max, S_accum[n].reg[row * 2 + 1]);
    }
    row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 1));
    row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 2));
    return row_max;
}

template<int N_TILES>
__device__ __forceinline__
void rescale_o(FragmentC O_accum[], int row, float alpha) {
    #pragma unroll
    for (int n = 0; n < N_TILES; n++) {
        O_accum[n].reg[row * 2 + 0] *= alpha;
        O_accum[n].reg[row * 2 + 1] *= alpha;
    }
}

template<int N_TILES>
__device__ __forceinline__
float exp_and_rowsum(FragmentC S_accum[], int row, float row_max) {
    float sum = 0.0f;
    #pragma unroll
    for (int n = 0; n < N_TILES; n++) {
        S_accum[n].reg[row * 2 + 0] = __expf(S_accum[n].reg[row * 2 + 0] - row_max);
        S_accum[n].reg[row * 2 + 1] = __expf(S_accum[n].reg[row * 2 + 1] - row_max);
        sum += S_accum[n].reg[row * 2 + 0] + S_accum[n].reg[row * 2 + 1];
    }
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
    return sum;
}

template<bool IS_FIRST, int QK_N_TILES, int PV_N_TILES>
__device__ __forceinline__
void online_softmax(FragmentC S_accum[], FragmentC O_accum[],
                    float m[2], float l[2]) {
    #pragma unroll
    for (int row = 0; row < 2; row++) {
        float row_max = compute_rowmax<QK_N_TILES>(S_accum, row);

        if constexpr (!IS_FIRST) {
            float alpha = __expf(m[row] - row_max);
            rescale_o<PV_N_TILES>(O_accum, row, alpha);
            l[row] *= alpha;
        }
        m[row] = row_max;

        l[row] += exp_and_rowsum<QK_N_TILES>(S_accum, row, row_max);
    }
}

// =========================================================================
// exp2f + fused scale: S is raw (unscaled), scale absorbed into exp2f
//
// exp(S * scale - m) = exp2f(S * scale_log2 - m_log2)
// where scale_log2 = log2(e) / sqrt(D), m stored in log2 space.
// exp2f(S * scale_log2 - m) compiles to FFMA + MUFU.EX2 (2 ops)
// vs the original FMUL + FADD + FMUL + MUFU.EX2 (4 ops).
// =========================================================================

template<int N_TILES>
__device__ __forceinline__
float exp2f_and_rowsum(FragmentC S_accum[], int row,
                       float row_max, float scale_log2) {
    float sum = 0.0f;
    #pragma unroll
    for (int n = 0; n < N_TILES; n++) {
        S_accum[n].reg[row * 2 + 0] = exp2f(
            S_accum[n].reg[row * 2 + 0] * scale_log2 - row_max);
        S_accum[n].reg[row * 2 + 1] = exp2f(
            S_accum[n].reg[row * 2 + 1] * scale_log2 - row_max);
        sum += S_accum[n].reg[row * 2 + 0] + S_accum[n].reg[row * 2 + 1];
    }
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
    return sum;
}

template<bool IS_FIRST, int QK_N_TILES, int PV_N_TILES>
__device__ __forceinline__
void online_softmax_exp2f(FragmentC S_accum[], FragmentC O_accum[],
                          float m[2], float l[2], float scale_log2) {
    #pragma unroll
    for (int row = 0; row < 2; row++) {
        float row_max_raw = compute_rowmax<QK_N_TILES>(S_accum, row);
        float row_max = row_max_raw * scale_log2;

        if constexpr (!IS_FIRST) {
            float alpha = exp2f(m[row] - row_max);
            rescale_o<PV_N_TILES>(O_accum, row, alpha);
            l[row] *= alpha;
        }
        m[row] = row_max;

        l[row] += exp2f_and_rowsum<QK_N_TILES>(S_accum, row,
                                                row_max, scale_log2);
    }
}
