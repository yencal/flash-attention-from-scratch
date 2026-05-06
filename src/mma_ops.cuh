#pragma once

#include "fragment.cuh"

__device__ __forceinline__
void mma_sync(FragmentC& D,
              const FragmentA& A,
              const FragmentB& B,
              const FragmentC& C) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D.reg[0]), "=f"(D.reg[1]), "=f"(D.reg[2]), "=f"(D.reg[3])
        : "r"(A.reg[0]), "r"(A.reg[1]), "r"(A.reg[2]), "r"(A.reg[3]),
          "r"(B.reg[0]), "r"(B.reg[1]),
          "f"(C.reg[0]), "f"(C.reg[1]), "f"(C.reg[2]), "f"(C.reg[3])
    );
}

__device__ __forceinline__
void mma_sync(FragmentC& acc,
              const FragmentA& A,
              const FragmentB& B) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc.reg[0]), "+f"(acc.reg[1]), "+f"(acc.reg[2]), "+f"(acc.reg[3])
        : "r"(A.reg[0]), "r"(A.reg[1]), "r"(A.reg[2]), "r"(A.reg[3]),
          "r"(B.reg[0]), "r"(B.reg[1])
    );
}
