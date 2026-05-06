#pragma once

#include "fragment.cuh"
#include "mma_ops.cuh"
#include "load_ldmatrix.cuh"

// =========================================================================
// GEMM QK: S = Q @ K^T  [16xD] @ [DxBC] -> [16xBC]
//
// Q_frags pre-loaded in registers (one per k-tile).
// K^T loaded from SMEM via ldmatrix.x4 + register reorder (2 B frags per load).
// =========================================================================

template<int K_TILES, int N_TILES, int D_STRIDE, int BC_STRIDE>
__device__ __forceinline__
void gemm_qk(FragmentC S_accum[],
             const FragmentA Q_frags[],
             const __half* smem_K)
{
    #pragma unroll
    for (int n = 0; n < N_TILES; n++)
        S_accum[n].fill(0.0f);

    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        #pragma unroll
        for (int n = 0; n < N_TILES; n += 2) {
            FragmentB K_frag0, K_frag1;
            load_kt_fragment_pair<D_STRIDE>(K_frag0, K_frag1,
                &smem_K[n * MMA_N * D_STRIDE + k * MMA_K]);

            mma_sync(S_accum[n],   Q_frags[k], K_frag0);
            mma_sync(S_accum[n+1], Q_frags[k], K_frag1);
        }
    }
}

// =========================================================================
// P staging: FP32 S fragments -> FP16 SMEM
//
// Writes P (post-softmax S) to SMEM so it can be loaded as A operand
// for the PV GEMM via ldmatrix.
// =========================================================================

template<int N_TILES, int SMEM_STRIDE>
__device__ __forceinline__
void store_p_to_smem(FragmentC S_accum[],
                     __half* smem_p,
                     int warp_row_offset)
{
    int lane = threadIdx.x & 31;

    #pragma unroll
    for (int n = 0; n < N_TILES; n++) {
        int col_offset = n * MMA_N;
        #pragma unroll
        for (int i = 0; i < FragmentC::num_elements; i++) {
            int row = warp_row_offset + FragmentC::get_row(lane, i);
            int col = col_offset + FragmentC::get_col(lane, i);
            smem_p[row * SMEM_STRIDE + col] = __float2half(S_accum[n].reg[i]);
        }
    }
}

// =========================================================================
// GEMM PV: O += P @ V  [16xBC] @ [BCxD] -> [16xD]
//
// P loaded from SMEM via ldmatrix.x4 (A operand).
// V loaded from SMEM via ldmatrix.x2.trans (B operand).
// =========================================================================

template<int K_TILES, int N_TILES, int BC_STRIDE, int D_STRIDE>
__device__ __forceinline__
void gemm_pv(FragmentC O_accum[],
             const __half* smem_P, const __half* smem_V,
             int warp_row_offset)
{
    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        FragmentA P_frag;
        load_fragment_ldmatrix<BC_STRIDE>(P_frag,
            &smem_P[warp_row_offset * BC_STRIDE + k * MMA_K]);

        #pragma unroll
        for (int n = 0; n < N_TILES; n++) {
            FragmentB V_frag;
            load_fragment_ldmatrix<D_STRIDE>(V_frag,
                &smem_V[k * MMA_K * D_STRIDE + n * MMA_N]);

            mma_sync(O_accum[n], P_frag, V_frag);
        }
    }
}

// =========================================================================
// P in registers: FragmentC -> FragmentA conversion
// =========================================================================

__device__ __forceinline__
uint32_t cvt_f32x2_to_f16x2(float lo, float hi) {
    uint32_t result;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n"
        : "=r"(result) : "f"(hi), "f"(lo));
    return result;
}

__device__ __forceinline__
void convert_s_to_p(FragmentA& p,
                    const FragmentC& s_even, const FragmentC& s_odd) {
    p.reg[0] = cvt_f32x2_to_f16x2(s_even.reg[0], s_even.reg[1]);
    p.reg[1] = cvt_f32x2_to_f16x2(s_even.reg[2], s_even.reg[3]);
    p.reg[2] = cvt_f32x2_to_f16x2(s_odd.reg[0],  s_odd.reg[1]);
    p.reg[3] = cvt_f32x2_to_f16x2(s_odd.reg[2],  s_odd.reg[3]);
}

template<int PV_K_TILES>
__device__ __forceinline__
void convert_s_to_p_frags(const FragmentC S_accum[], FragmentA P_frags[]) {
    #pragma unroll
    for (int k = 0; k < PV_K_TILES; k++)
        convert_s_to_p(P_frags[k], S_accum[2*k], S_accum[2*k+1]);
}

// =========================================================================
// Q-in-SMEM variant (non-swizzled): Q reloaded from SMEM each k-tile
// =========================================================================

template<int K_TILES, int N_TILES, int D_STRIDE>
__device__ __forceinline__
void gemm_qk_smemq_base(FragmentC S_accum[],
                         const __half* smem_Q,
                         const __half* smem_K,
                         int q_row_offset)
{
    #pragma unroll
    for (int n = 0; n < N_TILES; n++)
        S_accum[n].fill(0.0f);

    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        FragmentA Q_frag;
        load_fragment_ldmatrix<D_STRIDE>(Q_frag,
            &smem_Q[q_row_offset * D_STRIDE + k * MMA_K]);

        #pragma unroll
        for (int n = 0; n < N_TILES; n += 2) {
            FragmentB K_frag0, K_frag1;
            load_kt_fragment_pair<D_STRIDE>(K_frag0, K_frag1,
                &smem_K[n * MMA_N * D_STRIDE + k * MMA_K]);

            mma_sync(S_accum[n],   Q_frag, K_frag0);
            mma_sync(S_accum[n+1], Q_frag, K_frag1);
        }
    }
}

// =========================================================================
// P-in-registers PV GEMM (non-swizzled): inline S->P convert, V from SMEM
// =========================================================================

template<int QK_N_TILES, int PV_N_TILES, int D_STRIDE>
__device__ __forceinline__
void gemm_pv_regp(FragmentC O_accum[],
                  FragmentC S_accum[],
                  const __half* smem_V)
{
    constexpr int PV_K_TILES = QK_N_TILES / 2;

    #pragma unroll
    for (int k = 0; k < PV_K_TILES; k++) {
        FragmentA P_frag;
        convert_s_to_p(P_frag, S_accum[2*k], S_accum[2*k+1]);

        #pragma unroll
        for (int n = 0; n < PV_N_TILES; n++) {
            FragmentB V_frag;
            load_fragment_ldmatrix<D_STRIDE>(V_frag,
                &smem_V[k * MMA_K * D_STRIDE + n * MMA_N]);

            mma_sync(O_accum[n], P_frag, V_frag);
        }
    }
}

// =========================================================================
// Swizzled variants
// =========================================================================

template<int K_TILES, int N_TILES, int D_STRIDE>
__device__ __forceinline__
void gemm_qk_swizzled(FragmentC S_accum[],
                      const FragmentA Q_frags[],
                      const __half* smem_K)
{
    #pragma unroll
    for (int n = 0; n < N_TILES; n++)
        S_accum[n].fill(0.0f);

    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        #pragma unroll
        for (int n = 0; n < N_TILES; n += 2) {
            FragmentB K_frag0, K_frag1;
            load_kt_pair_swizzled<D_STRIDE>(K_frag0, K_frag1,
                smem_K, n * MMA_N, k * MMA_K);

            mma_sync(S_accum[n],   Q_frags[k], K_frag0);
            mma_sync(S_accum[n+1], Q_frags[k], K_frag1);
        }
    }
}

template<int K_TILES, int N_TILES, int D_STRIDE>
__device__ __forceinline__
void gemm_qk_smemq(FragmentC S_accum[],
                    const __half* smem_Q,
                    const __half* smem_K,
                    int q_row_offset)
{
    #pragma unroll
    for (int n = 0; n < N_TILES; n++)
        S_accum[n].fill(0.0f);

    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        FragmentA Q_frag;
        load_a_swizzled<D_STRIDE>(Q_frag, smem_Q,
            q_row_offset, k * MMA_K);

        #pragma unroll
        for (int n = 0; n < N_TILES; n += 2) {
            FragmentB K_frag0, K_frag1;
            load_kt_pair_swizzled<D_STRIDE>(K_frag0, K_frag1,
                smem_K, n * MMA_N, k * MMA_K);

            mma_sync(S_accum[n],   Q_frag, K_frag0);
            mma_sync(S_accum[n+1], Q_frag, K_frag1);
        }
    }
}

template<int N_TILES, int SMEM_STRIDE>
__device__ __forceinline__
void store_p_swizzled(FragmentC S_accum[],
                      __half* smem_p,
                      int warp_row_offset)
{
    int lane = threadIdx.x & 31;

    #pragma unroll
    for (int n = 0; n < N_TILES; n++) {
        int col_offset = n * MMA_N;
        #pragma unroll
        for (int i = 0; i < FragmentC::num_elements; i++) {
            int row = warp_row_offset + FragmentC::get_row(lane, i);
            int col = col_offset + FragmentC::get_col(lane, i);
            smem_p[swizzle_offset<SMEM_STRIDE>(row, col)] =
                __float2half(S_accum[n].reg[i]);
        }
    }
}

template<int K_TILES, int N_TILES, int BC_STRIDE, int D_STRIDE>
__device__ __forceinline__
void gemm_pv_swizzled(FragmentC O_accum[],
                      const __half* smem_P, const __half* smem_V,
                      int warp_row_offset)
{
    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        FragmentA P_frag;
        load_a_swizzled<BC_STRIDE>(P_frag, smem_P,
            warp_row_offset, k * MMA_K);

        #pragma unroll
        for (int n = 0; n < N_TILES; n++) {
            FragmentB V_frag;
            load_b_swizzled<D_STRIDE>(V_frag, smem_V,
                k * MMA_K, n * MMA_N);

            mma_sync(O_accum[n], P_frag, V_frag);
        }
    }
}

template<int QK_N_TILES, int PV_N_TILES, int D_STRIDE>
__device__ __forceinline__
void gemm_pv_regp_swizzled(FragmentC O_accum[],
                           FragmentC S_accum[],
                           const __half* smem_V)
{
    constexpr int PV_K_TILES = QK_N_TILES / 2;

    #pragma unroll
    for (int k = 0; k < PV_K_TILES; k++) {
        FragmentA P_frag;
        convert_s_to_p(P_frag, S_accum[2*k], S_accum[2*k+1]);

        #pragma unroll
        for (int n = 0; n < PV_N_TILES; n++) {
            FragmentB V_frag;
            load_b_swizzled<D_STRIDE>(V_frag, smem_V,
                k * MMA_K, n * MMA_N);

            mma_sync(O_accum[n], P_frag, V_frag);
        }
    }
}

template<int K_TILES, int N_TILES, int D_STRIDE>
__device__ __forceinline__
void gemm_pv_prefrag_swizzled(FragmentC O_accum[],
                              const FragmentA P_frags[],
                              const __half* smem_V)
{
    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        #pragma unroll
        for (int n = 0; n < N_TILES; n++) {
            FragmentB V_frag;
            load_b_swizzled<D_STRIDE>(V_frag, smem_V,
                k * MMA_K, n * MMA_N);

            mma_sync(O_accum[n], P_frags[k], V_frag);
        }
    }
}

// =========================================================================
// x4.trans V pair loading: 1 ldmatrix.x4.trans -> 2 B fragments
// =========================================================================

template<int K_TILES, int N_TILES, int D_STRIDE>
__device__ __forceinline__
void gemm_pv_x4trans(FragmentC O_accum[],
                     const FragmentA P_frags[],
                     const __half* smem_V)
{
    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        #pragma unroll
        for (int n = 0; n < N_TILES; n += 2) {
            FragmentB V_frag0, V_frag1;
            load_vt_pair_swizzled<D_STRIDE>(V_frag0, V_frag1,
                smem_V, k * MMA_K, n * MMA_N);

            mma_sync(O_accum[n],   P_frags[k], V_frag0);
            mma_sync(O_accum[n+1], P_frags[k], V_frag1);
        }
    }
}

// =========================================================================
// Batched M-tile GEMM: both M-tiles processed per k-tile iteration
// =========================================================================

template<int K_TILES, int N_TILES, int D_STRIDE, int M_TILES>
__device__ __forceinline__
void gemm_qk_smemq_batched(FragmentC S_accum[][N_TILES],
                            const __half* smem_Q,
                            const __half* smem_K,
                            int warp_base_row)
{
    #pragma unroll
    for (int mt = 0; mt < M_TILES; mt++)
        #pragma unroll
        for (int n = 0; n < N_TILES; n++)
            S_accum[mt][n].fill(0.0f);

    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        FragmentA Q_frag[M_TILES];
        #pragma unroll
        for (int mt = 0; mt < M_TILES; mt++)
            load_a_swizzled<D_STRIDE>(Q_frag[mt], smem_Q,
                warp_base_row + mt * MMA_M, k * MMA_K);

        #pragma unroll
        for (int n = 0; n < N_TILES; n += 2) {
            FragmentB K_frag0, K_frag1;
            load_kt_pair_swizzled<D_STRIDE>(K_frag0, K_frag1,
                smem_K, n * MMA_N, k * MMA_K);

            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++) {
                mma_sync(S_accum[mt][n],   Q_frag[mt], K_frag0);
                mma_sync(S_accum[mt][n+1], Q_frag[mt], K_frag1);
            }
        }
    }
}

// =========================================================================
// Double-buffered batched QK GEMM: K-pair ping-pong
// =========================================================================

template<int K_TILES, int N_TILES, int D_STRIDE, int M_TILES>
__device__ __forceinline__
void gemm_qk_smemq_batched_dbuf(FragmentC S_accum[][N_TILES],
                                 const __half* smem_Q,
                                 const __half* smem_K,
                                 int warp_base_row)
{
    #pragma unroll
    for (int mt = 0; mt < M_TILES; mt++)
        #pragma unroll
        for (int n = 0; n < N_TILES; n++)
            S_accum[mt][n].fill(0.0f);

    constexpr int N_PAIRS = N_TILES / 2;
    constexpr int TOTAL_PAIRS = K_TILES * N_PAIRS;

    FragmentB K_pp[2][2];
    load_kt_pair_swizzled<D_STRIDE>(K_pp[0][0], K_pp[0][1],
        smem_K, 0, 0);

    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        FragmentA Q_frag[M_TILES];
        #pragma unroll
        for (int mt = 0; mt < M_TILES; mt++)
            load_a_swizzled<D_STRIDE>(Q_frag[mt], smem_Q,
                warp_base_row + mt * MMA_M, k * MMA_K);

        #pragma unroll
        for (int np = 0; np < N_PAIRS; np++) {
            int idx = k * N_PAIRS + np;
            int cur = idx & 1;
            int nxt = 1 - cur;

            if (idx + 1 < TOTAL_PAIRS) {
                int next_np = (np + 1 < N_PAIRS) ? np + 1 : 0;
                int next_k  = (np + 1 < N_PAIRS) ? k : k + 1;
                load_kt_pair_swizzled<D_STRIDE>(K_pp[nxt][0], K_pp[nxt][1],
                    smem_K, next_np * 2 * MMA_N, next_k * MMA_K);
            }

            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++) {
                mma_sync(S_accum[mt][np * 2],     Q_frag[mt], K_pp[cur][0]);
                mma_sync(S_accum[mt][np * 2 + 1], Q_frag[mt], K_pp[cur][1]);
            }
        }
    }
}

// =========================================================================
// Double-buffered batched PV GEMM: V-pair ping-pong from SMEM
// =========================================================================

template<int K_TILES, int N_TILES, int D_STRIDE, int M_TILES>
__device__ __forceinline__
void gemm_pv_x4trans_batched_dbuf(FragmentC O_accum[][N_TILES],
                                   const FragmentA P_frags[][K_TILES],
                                   const __half* smem_V)
{
    constexpr int N_PAIRS = N_TILES / 2;
    constexpr int TOTAL_PAIRS = K_TILES * N_PAIRS;

    FragmentB V_pp[2][2];
    load_vt_pair_swizzled<D_STRIDE>(V_pp[0][0], V_pp[0][1],
        smem_V, 0, 0);

    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        #pragma unroll
        for (int np = 0; np < N_PAIRS; np++) {
            int idx = k * N_PAIRS + np;
            int cur = idx & 1;
            int nxt = 1 - cur;

            if (idx + 1 < TOTAL_PAIRS) {
                int next_np = (np + 1 < N_PAIRS) ? np + 1 : 0;
                int next_k  = (np + 1 < N_PAIRS) ? k : k + 1;
                load_vt_pair_swizzled<D_STRIDE>(V_pp[nxt][0], V_pp[nxt][1],
                    smem_V, next_k * MMA_K, next_np * 2 * MMA_N);
            }

            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++) {
                mma_sync(O_accum[mt][np * 2],     P_frags[mt][k], V_pp[cur][0]);
                mma_sync(O_accum[mt][np * 2 + 1], P_frags[mt][k], V_pp[cur][1]);
            }
        }
    }
}

// =========================================================================
// V in registers: preload all V, PV GEMM is pure register-to-register
// =========================================================================

template<int K_TILES, int N_TILES, int D_STRIDE>
__device__ __forceinline__
void preload_v_x4trans(FragmentB V_frags[][N_TILES],
                       const __half* smem_V)
{
    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        #pragma unroll
        for (int n = 0; n < N_TILES; n += 2) {
            load_vt_pair_swizzled<D_STRIDE>(V_frags[k][n], V_frags[k][n+1],
                smem_V, k * MMA_K, n * MMA_N);
        }
    }
}

template<int K_TILES, int N_TILES, int M_TILES>
__device__ __forceinline__
void gemm_pv_vreg_batched(FragmentC O_accum[][N_TILES],
                          const FragmentA P_frags[][K_TILES],
                          const FragmentB V_frags[][N_TILES])
{
    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        #pragma unroll
        for (int n = 0; n < N_TILES; n++) {
            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++) {
                mma_sync(O_accum[mt][n], P_frags[mt][k], V_frags[k][n]);
            }
        }
    }
}

