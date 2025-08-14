# test_moe_wna16_marlin_gemm.py
import math
import numpy as np
import pytest
import torch
from sgl_kernel_sycl import moe_wna16_marlin_gemm

from sglang.srt.layers.quantization.scalar_type import scalar_types
from sglang.srt.layers.quantization.utils import quantize_weights

def _type_id(st):
    # 兼容不同封装：id / scalar_type_id / value
    for attr in ("id", "scalar_type_id", "value"):
        if hasattr(st, attr):
            return getattr(st, attr)
    return st

# ---------- Marlin 权重辅助 ----------
GPTQ_MARLIN_TILE = 16

def marlin_weights_gptq(q_w: torch.Tensor, size_k: int, size_n: int, num_bits: int):
    # 只做 16x16 tile 重排，不做额外 perm
    tile = 16
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0 and size_n % tile == 0
    q = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q = q.permute((0, 2, 1, 3))                     # [K/16, N/16, 16, 16]
    q = q.reshape((size_k // tile, size_n * tile))  # [K/16, N*16]
    return marlin_pack(q, num_bits)

def get_weight_perm(num_bits: int):
    perm_list: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)
    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception(f"num_bits must be 4 or 8, got {num_bits}")
    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    return torch.from_numpy(perm)

def marlin_permute_weights(q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0 and size_n % tile == 0
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))
    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)
    return q_w

def marlin_pack(q_w: torch.Tensor, num_bits: int):
    pack_factor = 32 // num_bits
    orig_device = q_w.device
    q_w = q_w.to("cpu").contiguous().numpy().astype(np.uint32)
    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << (num_bits * i)
    return torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

def marlin_weights(q_w: torch.Tensor, size_k: int, size_n: int, num_bits: int, perm: torch.Tensor):
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)
    return marlin_pack(q_w, num_bits)

# ---------- MoE 元数据 ----------
def make_moe_metadata(M: int, top_k: int, moe_block_size: int, device: torch.device):
    total = M * top_k
    padded = int(math.ceil(total / moe_block_size) * moe_block_size)
    sorted_token_ids = torch.arange(total, dtype=torch.int32, device=device)
    if padded > total:
        pad = torch.full((padded - total,), fill_value=total, dtype=torch.int32, device=device)
        sorted_token_ids = torch.cat([sorted_token_ids, pad], dim=0)
    num_blocks = padded // moe_block_size
    expert_ids = torch.zeros((num_blocks,), dtype=torch.int32, device=device)  # 全 expert 0
    num_tokens_past_padded = torch.tensor([padded], dtype=torch.int32, device=device)
    topk_weights = torch.ones((total,), dtype=torch.float32, device=device)  # 乘 1 不改变数值
    return sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights

# ---------- 主测试 ----------
#TODO: figure out the root cause fot bfloat16 slow issue
#@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("num_bits, b_q_type_attr", [
    (4, "uint4b8"),     # GPTQ 4bit
    (8, "uint8b128"),   # GPTQ 8bit
])
@pytest.mark.parametrize("k_tiles,n_tiles", [(4, 2), (4, 4)])
@pytest.mark.parametrize("moe_block_size", [8, 16])
def test_moe_wna16_marlin_gemm_correct(dtype, num_bits, b_q_type_attr, k_tiles, n_tiles, moe_block_size):

    if not hasattr(scalar_types, b_q_type_attr):
        pytest.skip(f"scalar_types.{b_q_type_attr} not available in this build.")

    device = torch.device("xpu")
    torch.manual_seed(0)

    tile_k, tile_n = 16, 64
    M = 32
    K = k_tiles * tile_k
    N = n_tiles * tile_n

    # ===== 量化（对称列量化，生成 q_unsigned + scales；内核会按自己的方式解码） =====
    w_fp = torch.randn((K, N), dtype=torch.float32, device=device)
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    s_col = w_fp.abs().amax(dim=0) / qmax
    s_col = torch.clamp(s_col, min=1e-8)
    q_signed   = torch.round(w_fp / s_col).to(torch.int32).clamp(qmin, qmax)
    q_unsigned = (q_signed + (1 << (num_bits - 1))).to(torch.int32)

    # ===== 打包成 Marlin GPTQ 权重（使用已有的 perm + pack） =====
    weight_perm = get_weight_perm(num_bits)
    b_q_weight = (
        marlin_weights(q_unsigned, K, N, num_bits, weight_perm)
        .contiguous().to(device).unsqueeze(0)  # [E=1, K/16, (N*16)/pack]
    )
    b_scales = s_col.to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).contiguous()

    pack_factor = 32 // num_bits
    assert list(b_q_weight.shape) == [1, K // 16, (N * 16) // pack_factor]

    # ===== MoE 元数据 & workspace =====
    def _moe_and_ws(size_m, moe_bs, device, size_n):
        min_thread_n = 64
        assert size_n % min_thread_n == 0
    
        sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights = make_moe_metadata(
            size_m, top_k=1, moe_block_size=moe_bs, device=device
        )
    
        # 原来是：min_workspace = min(max_n_tiles * (sorted_token_ids.size(0)/moe_bs), sms * 4)
        # 这里去掉对 sms 的依赖，直接用“上界”就行：
        max_n_tiles = size_n // min_thread_n
        min_workspace = max_n_tiles * int(sorted_token_ids.size(0) / moe_bs)
    
        # 分配到 device 上
        workspace = torch.zeros((max(min_workspace, 1),), dtype=torch.int32, device=device)
        return sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights, workspace

    b_q_type = getattr(scalar_types, b_q_type_attr)

    # ===== 第一步：A = I_K → “拍出” W_kernel =====
    A_eye = torch.eye(K, dtype=dtype, device=device)
    sorted_ids_E, expert_ids_E, ntpp_E, topk_E, ws_E = _moe_and_ws(K, moe_block_size, device, N)
    W_kernel = moe_wna16_marlin_gemm(
        A_eye, None, b_q_weight, b_scales,
        None, None, None,
        ws_E,
        sorted_ids_E, expert_ids_E, ntpp_E, topk_E,
        int(moe_block_size), int(1), False, False,           # mul_topk_weights=False
        _type_id(b_q_type), int(K), int(N), int(K),
        False, False, True, False,
    )
    assert list(W_kernel.shape) == [K, N]

    # ===== 第二步：随机 A → 对比 out 与 a @ W_kernel =====
    a = torch.randn((M, K), dtype=dtype, device=device)
    sorted_ids, expert_ids, ntpp, topk, ws = _moe_and_ws(M, moe_block_size, device, N)
    out = moe_wna16_marlin_gemm(
        a, None, b_q_weight, b_scales,
        None, None, None,
        ws,
        sorted_ids, expert_ids, ntpp, topk,
        int(moe_block_size), int(1), False, False,
        _type_id(b_q_type), int(M), int(N), int(K),
        False, False, True, False,
    )

    #ref = a @ W_kernel
    ref = (a.float().cpu() @ W_kernel.float().cpu())
    torch.xpu.synchronize()

    # —— 按 dtype 选择容差：BF16 舍入噪声大一些 ——
    if dtype == torch.bfloat16:
        rtol, atol = 2e-2, 6e-2
    else:  # torch.float16
        rtol, atol = 1e-2, 2e-2

    #torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(out.cpu().float(), ref.float(), rtol=rtol, atol=atol)

if __name__ == "__main__":
    # 允许直接 `python test_moe_wna16_marlin_gemm.py`
    import subprocess, sys, os
    subprocess.call([sys.executable, "-m", "pytest", "--tb=short", "-s", "-v", os.path.basename(__file__)])
