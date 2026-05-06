#!/usr/bin/env python3
# bench_fa2.py
# Benchmark PyTorch's FlashAttention (SDPA) at the same configs as the C++ benchmark.
# Output: fa2_results.csv (same format as flash_attention_results.csv)

import torch
import torch.nn.functional as F

BATCH, HEADS, D = 16, 16, 128
SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192]
WARMUP, ITERS = 10, 50


def attention_flops(batch, heads, N, d):
    return batch * heads * (4.0 * N * N * d + 6.0 * N * N)


results = []
for N in SEQ_LENS:
    # SDPA expects (batch, heads, seqlen, headdim)
    Q = torch.randn(BATCH, HEADS, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(BATCH, HEADS, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(BATCH, HEADS, N, D, device="cuda", dtype=torch.float16)

    # Force flash attention backend
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        for _ in range(WARMUP):
            F.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            F.scaled_dot_product_attention(Q, K, V)
        end.record()
        torch.cuda.synchronize()

    ms = start.elapsed_time(end) / ITERS
    flops = attention_flops(BATCH, HEADS, N, D)
    tflops = flops / (ms * 1e9)
    print(f"N={N:5d}  {ms:.3f} ms  {tflops:.2f} TFLOPS")
    results.append((N, ms, tflops))

with open("fa2_results.csv", "w") as f:
    f.write("Label,N,TimeMs,TFLOPS\n")
    for N, ms, tflops in results:
        f.write(f"fa2,{N},{ms:.6f},{tflops:.4f}\n")

print("Results saved to fa2_results.csv")
