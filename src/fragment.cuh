#pragma once

#include <cstdint>
#include <cuda_fp16.h>

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

struct FragmentA {
    static constexpr int num_elements = 8;
    static constexpr int num_regs = 4;
    uint32_t reg[num_regs];

    static __device__ int get_row(int lane, int i) {
        return (lane >> 2) + 8 * ((i >> 1) & 1);
    }

    static __device__ int get_col(int lane, int i) {
        return (lane & 3) * 2 + (i & 1) + 8 * (i >> 2);
    }
};

struct FragmentB {
    static constexpr int num_elements = 4;
    static constexpr int num_regs = 2;
    uint32_t reg[num_regs];

    static __device__ int get_row(int lane, int i) {
        return (lane & 3) * 2 + (i & 1) + 8 * (i >> 1);
    }

    static __device__ int get_col(int lane, int i) {
        return lane >> 2;
    }
};

struct FragmentC {
    static constexpr int num_elements = 4;
    static constexpr int num_regs = 4;
    float reg[num_regs];

    __device__ __forceinline__ void fill(float val) {
        reg[0] = val; reg[1] = val; reg[2] = val; reg[3] = val;
    }

    static __device__ int get_row(int lane, int i) {
        return (lane >> 2) + 8 * (i >> 1);
    }

    static __device__ int get_col(int lane, int i) {
        return (lane & 3) * 2 + (i & 1);
    }
};
