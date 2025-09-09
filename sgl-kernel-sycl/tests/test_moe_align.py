import itertools

import pytest
import torch
import triton
import triton.language as tl
from sgl_kernel_sycl import moe_align_block_size


def ceil_div(a, b):
    return (a + b - 1) // b

@torch.no_grad()
def moe_align_block_size_torch(
    topk_ids: torch.Tensor,            # 任意形状，元素为 [0, num_experts)
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,    # 输出：长度 >= total_tokens_post_pad
    expert_ids: torch.Tensor,          # 输出：长度 = total_tokens_post_pad // block_size
    num_tokens_post_pad: torch.Tensor, # 输出：标量 tensor
) -> None:
    """
    纯 PyTorch 版本（无 Triton 依赖）。
    - 接受任意形状的 topk_ids；会按 .reshape(-1) 的线性顺序处理（与 Triton 一致）。
    - 每个 expert 的区间长度为 ceil(count / block_size) * block_size。
    - 在各自 expert 区间的前 count 个位置按线性顺序写入原始线性索引。
    - expert_ids 以 block 为单位写 expert id。
    """
    assert isinstance(num_experts, int) and num_experts >= 0
    assert isinstance(block_size, int) and block_size > 0

    device = topk_ids.device
    ne = num_experts
    bs = block_size

    # 关键改动：按 Triton 的做法把任意形状线性化
    flat = topk_ids.reshape(-1)
    numel = flat.numel()

    # 1) 按 expert 统计
    counts = torch.bincount(flat.to(torch.int64), minlength=ne).to(torch.int32)  # [ne]

    # 2) 计算 padding 后长度和前缀和
    padded = ((counts + bs - 1) // bs) * bs                                     # [ne]
    cumsum = torch.empty(ne + 1, dtype=torch.int32, device=device)              # [ne+1]
    cumsum[0] = 0
    if ne > 0:
        cumsum[1:] = torch.cumsum(padded, dim=0)
    total_padded = int(cumsum[-1].item())
    num_tokens_post_pad.fill_(total_padded)

    # 3) 写 expert_ids（按 block）
    if total_padded > 0:
        expected_blocks = total_padded // bs
        if expert_ids.numel() < expected_blocks:
            raise ValueError(f"expert_ids 太短，需要 {expected_blocks}，实际 {expert_ids.numel()}")
        for e in range(ne):
            blk_l = int(cumsum[e].item()) // bs
            blk_r = int(cumsum[e + 1].item()) // bs
            if blk_r > blk_l:
                expert_ids[blk_l:blk_r] = e

    # 4) 回填每个 expert 的有效 token 索引（线性索引）
    if total_padded > 0:
        if sorted_token_ids.numel() < total_padded:
            raise ValueError(f"sorted_token_ids 太短，需要 >= {total_padded}，实际 {sorted_token_ids.numel()}")
        arange_idx = torch.arange(numel, device=device, dtype=torch.int64)

        for e in range(ne):
            cnt = int(counts[e].item())
            if cnt == 0:
                continue
            mask = (flat == e)
            idxs = arange_idx[mask]                  # 线性顺序的原始索引
            base = int(cumsum[e].item())
            sorted_token_ids[base: base + cnt] = idxs.to(sorted_token_ids.dtype)

    return

@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)
    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = ceil_div(numel, num_experts)

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )
    moe_align_block_size_stage2[grid](
        tokens_cnts,
        num_experts,
    )
    moe_align_block_size_stage3[(1,)](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )

@pytest.mark.parametrize(
    "block_size,num_tokens,topk,num_experts,pad_sorted_token_ids",
    list(
        itertools.product(
             [32, 64],  # block_size
             [1, 2],  # num_tokens
             [1, 2],  # topk
             [32, 67],  #  num_experts
             [True, False],  # pad_sorted_token_ids
        )
    ),
)
def test_moe_align_block_size_compare_implementations(
    block_size, num_tokens, topk, num_experts, pad_sorted_token_ids
):

    topk_ids = torch.argsort(torch.rand(num_tokens, num_experts, device="cpu"), dim=1)[
        :, :topk
    ]

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)

    sorted_ids_cuda = torch.zeros(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    if not pad_sorted_token_ids:
        sorted_ids_cuda.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids_cuda = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad_cuda = torch.zeros(
        (1), dtype=torch.int32, device=topk_ids.device
    )
    token_cnts_buffer = torch.zeros(
        (num_experts + 1) * num_experts,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    cumsum_buffer = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )

    sorted_ids_triton = torch.empty_like(sorted_ids_cuda)
    sorted_ids_triton.fill_(topk_ids.numel())
    expert_ids_triton = torch.zeros_like(expert_ids_cuda)
    num_tokens_post_pad_triton = torch.empty_like(num_tokens_post_pad_cuda)

    topk_ids_xpu = topk_ids.to("xpu")
    num_experts_xpu = num_experts
    block_size_xpu = block_size
    sorted_ids_cuda_xpu =  sorted_ids_cuda.to("xpu")
    expert_ids_cuda_xpu = expert_ids_cuda.to("xpu")
    num_tokens_post_pad_cuda_xpu = num_tokens_post_pad_cuda.to("xpu")
    token_cnts_buffer_xpu = token_cnts_buffer.to("xpu")
    cumsum_buffer_xpu = cumsum_buffer.to("xpu")
    pad_sorted_token_ids_xpu = pad_sorted_token_ids

    moe_align_block_size(
        topk_ids_xpu,
        num_experts_xpu,
        block_size_xpu,
        sorted_ids_cuda_xpu,
        expert_ids_cuda_xpu,
        num_tokens_post_pad_cuda_xpu,
        token_cnts_buffer_xpu,
        cumsum_buffer_xpu,
        pad_sorted_token_ids_xpu,
    )

    #return
    moe_align_block_size_torch(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_triton,
        expert_ids_triton,
        num_tokens_post_pad_triton,
    )

    #sorted_ids_triton_xpu = sorted_ids_triton.to("xpu")
    #expert_ids_triton_xpu = expert_ids_triton.to("xpu")
    #num_tokens_post_pad_triton_xpu = num_tokens_post_pad_triton.to("xpu")

    assert torch.allclose(expert_ids_cuda_xpu.to("cpu"), expert_ids_triton, atol=0, rtol=0), (
        f"Expert IDs mismatch for block_size={block_size_xpu}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA expert_ids1: {expert_ids_cuda_xpu.to("cpu")}\n"
        f"Triton expert_ids1: {expert_ids_triton}"
    )

    assert torch.allclose(
        num_tokens_post_pad_cuda_xpu.to("cpu"), num_tokens_post_pad_triton, atol=0, rtol=0
    ), (
        f"Num tokens post pad mismatch for block_size={block_size_xpu}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA num_tokens_post_pad2: {num_tokens_post_pad_cuda_xpu.to("cpu")}\n"
        f"Triton num_tokens_post_pad2: {num_tokens_post_pad_triton}"
    )
    #return
    # Select an expert to check
    expert_ids_cuda = expert_ids_cuda_xpu.to("cpu")
    expert_idx = expert_ids_cuda.max().item()

    # Get the first and last block id where expert_ids_cuda == expert_idx
    matching_indices = torch.where(expert_ids_cuda == expert_idx)[0]
    block_sorted_start = matching_indices[0].item() * block_size
    block_sorted_end = min(
        (matching_indices[-1].item() + 1) * block_size, max_num_tokens_padded
    )
    sorted_ids_cuda = sorted_ids_cuda_xpu.to("cpu")
    selected_sorted_ids_cuda = sorted_ids_cuda[
        block_sorted_start:block_sorted_end
    ].sort()[0]
    selected_sorted_ids_triton = sorted_ids_triton[
        block_sorted_start:block_sorted_end
    ].sort()[0]

    assert torch.allclose(
        selected_sorted_ids_cuda,
        selected_sorted_ids_triton,
        atol=0,
        rtol=0,
    ), (
        f"Sorted IDs mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA sorted_ids3: {selected_sorted_ids_cuda}\n"
        f"Triton sorted_ids3: {selected_sorted_ids_triton}"
    )


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
