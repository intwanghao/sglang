import torch


def gptq_marlin_repack(
    b_q_weight,
    perm,
    size_k,
    size_n,
    num_bits,
):
    return torch.ops.sgl_kernel_sycl.gptq_marlin_repack.default(
        b_q_weight,
        perm,
        size_k,
        size_n,
        num_bits,
    )

def moe_wna16_marlin_gemm(
    a,
    c_or_none,
    b_q_weight,
    b_scales,
    b_zeros_or_none,
    g_idx_or_none,
    perm_or_none,
    workspace,
    sorted_token_ids,
    expert_ids,
    num_tokens_past_padded,
    topk_weights,
    moe_block_size,
    top_k,
    mul_topk_weights,
    is_ep,
    b_q_type_id,
    size_m,
    size_n,
    size_k,
    is_k_full,
    use_atomic_add,
    use_fp32_reduce,
    is_zp_float,
):
    return torch.ops.sgl_kernel_sycl.moe_wna16_marlin_gemm.default(
        a,
        c_or_none,
        b_q_weight,
        b_scales,
        b_zeros_or_none,
        g_idx_or_none,
        perm_or_none,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_past_padded,
        topk_weights,
        int(moe_block_size),
        int(top_k),
        bool(mul_topk_weights),
        bool(is_ep),
        int(b_q_type_id),
        int(size_m),
        int(size_n),
        int(size_k),
        bool(is_k_full),
        bool(use_atomic_add),
        bool(use_fp32_reduce),
        bool(is_zp_float),
    )
  


def awq_marlin_repack(
    b_q_weight: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    return torch.ops.sgl_kernel_sycl.awq_marlin_repack(b_q_weight, size_k, size_n, num_bits)


def awq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty(
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = torch.ops.sgl_kernel_sycl.awq_marlin_repack(
            b_q_weight[e], size_k, size_n, num_bits
        )
    return output
