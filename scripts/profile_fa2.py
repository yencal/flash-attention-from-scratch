import torch
import torch.nn.functional as F

BATCH, HEADS, N, D = 16, 16, 8192, 128
Q = torch.randn(BATCH, HEADS, N, D, device="cuda", dtype=torch.float16)
K = torch.randn(BATCH, HEADS, N, D, device="cuda", dtype=torch.float16)
V = torch.randn(BATCH, HEADS, N, D, device="cuda", dtype=torch.float16)

with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
    O = F.scaled_dot_product_attention(Q, K, V)
torch.cuda.synchronize()
