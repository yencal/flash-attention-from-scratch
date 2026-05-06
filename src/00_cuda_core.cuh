#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

template<int BR, int BC, int D>
__global__ void flash_cuda_core(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int N)
{
    const float SCALE = 1.0f / sqrtf((float)D);

    int tid   = threadIdx.x;
    int q_blk = blockIdx.x;
    int head  = blockIdx.y;
    int batch = blockIdx.z;

    int base = (batch * gridDim.y + head) * N * D;

    const __half* Q_tile = Q + base + q_blk * BR * D;
    const __half* K_head = K + base;
    const __half* V_head = V + base;
    __half*       O_tile = O + base + q_blk * BR * D;

    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = smem + BR * D;
    float* sV = smem + (BR + BC) * D;

    for (int i = tid; i < BR * D; i += BR)
        sQ[i] = __half2float(Q_tile[i]);
    __syncthreads();

    float O_acc[D];
    for (int d = 0; d < D; d++)
        O_acc[d] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    int n_kv_blocks = N / BC;

    for (int j = 0; j < n_kv_blocks; j++) {
        const __half* Kj = K_head + j * BC * D;
        const __half* Vj = V_head + j * BC * D;
        for (int i = tid; i < BC * D; i += BR) {
            sK[i] = __half2float(Kj[i]);
            sV[i] = __half2float(Vj[i]);
        }
        __syncthreads();

        float S[BC];
        float new_max = m;
        for (int c = 0; c < BC; c++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++)
                dot += sQ[tid * D + d] * sK[c * D + d];
            S[c] = dot * SCALE;
            new_max = fmaxf(new_max, S[c]);
        }

        float alpha = __expf(m - new_max);
        for (int d = 0; d < D; d++)
            O_acc[d] *= alpha;
        l *= alpha;
        m = new_max;

        for (int c = 0; c < BC; c++) {
            float p = __expf(S[c] - m);
            l += p;
            for (int d = 0; d < D; d++)
                O_acc[d] += p * sV[c * D + d];
        }

        __syncthreads();
    }

    float inv_l = 1.0f / l;
    for (int d = 0; d < D; d++)
        O_tile[tid * D + d] = __float2half(O_acc[d] * inv_l);
}

template<int BR, int BC, int D>
struct FlashCudaCore {
    static constexpr int NUM_THREADS = BR;
    static constexpr size_t SMEM_BYTES = (BR + 2 * BC) * D * sizeof(float);

    static void Run(const __half* Q, const __half* K, const __half* V,
                    __half* O, int batch, int n_heads, int N) {
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(
                flash_cuda_core<BR, BC, D>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_BYTES);
            configured = true;
        }
        dim3 grid(N / BR, n_heads, batch);
        flash_cuda_core<BR, BC, D><<<grid, NUM_THREADS, SMEM_BYTES>>>(Q, K, V, O, N);
    }
};
