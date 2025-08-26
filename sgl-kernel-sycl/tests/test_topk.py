import itertools

import torch
import triton
from sgl_kernel_sycl import topk_softmax

num_tokens = 1
num_experts = 4
topk = 1

gating_output = torch.randn(
    (num_tokens, num_experts), dtype=torch.float32, device="cpu"
).to("xpu")
topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cpu").to("xpu")
topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cpu").to("xpu")

gating_output_cpu = gating_output.to("cpu")

topk_softmax(
    topk_weights,
    topk_indices,
    gating_output,
)
topk_weights_cpu = topk_weights.to("cpu")
topk_indices_cpu = topk_indices.to("cpu")
# Native torch implementation
softmax_output = torch.softmax(gating_output_cpu, dim=-1)
topk_weights_ref, topk_indices_ref = torch.topk(softmax_output, topk, dim=-1)

# Verify the top-k weights and indices match the torch native ones
assert torch.allclose(
    topk_weights_ref, topk_weights_cpu, atol=1e-3, rtol=1e-3
), f"Weights mismatch: torch={topk_weights_ref} vs SGLang={topk_weights_cpu}"
assert torch.allclose(
    topk_indices_ref.int(), topk_indices_cpu, atol=0, rtol=0
), f"Indices mismatch: torch={topk_indices_ref}, SGLang={topk_indices_cpu}"
print(topk_weights_ref)
print(topk_weights_cpu)
