import numpy as np
import pytest
import torch
from typing import Optional
from sgl_kernel_sycl import gptq_marlin_repack

from sglang.srt.layers.quantization.utils import (
    get_pack_factor,
    pack_cols,
)

GPTQ_MARLIN_TILE_K = 16
GPTQ_MARLIN_TILE_N = 64

def gptq_pack_along_k(q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int) -> torch.Tensor:
    assert q_w.shape == (size_k, size_n)
    assert num_bits in (4, 8)
    pack_factor = 32 // num_bits
    assert size_k % pack_factor == 0

    dev = q_w.device
    q = q_w.detach().to("cpu").numpy().astype(np.uint32)  # [K, N]
    k_groups = size_k // pack_factor
    packed = np.zeros((k_groups, size_n), dtype=np.uint32)
    for i in range(pack_factor):
        shift = num_bits * i  # LSB-first
        packed |= (q[i::pack_factor, :] << shift)
    return torch.from_numpy(packed.astype(np.int32)).to(dev)

def gptq_marlin_reference(
    q_w: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    has_perm: bool,
    perm: Optional[torch.Tensor],
) -> torch.Tensor:
    assert q_w.shape == (size_k, size_n)
    assert num_bits in (4, 8)
    pack_factor = 32 // num_bits
    tile_k_size, tile_n_size = 16, 64
    assert size_k % tile_k_size == 0 and size_n % tile_n_size == 0

    q = q_w.detach().to("cpu").numpy().astype(np.uint32)
    if has_perm:
        assert perm is not None and perm.numel() == size_k
        perm_np = perm.detach().to("cpu").numpy().astype(np.int64)
    else:
        perm_np = None

    k_tiles = size_k // tile_k_size
    n_tiles = size_n // tile_n_size

    out_cols_per_tile = tile_k_size * tile_n_size // pack_factor  # = 16*64/pack_factor
    out = np.zeros((k_tiles, n_tiles * out_cols_per_tile), dtype=np.uint32)

    tc_offsets = [0, 1, 8, 9]
    if num_bits == 4:
        pack_idx = [0, 2, 4, 6, 1, 3, 5, 7]
    else:
        pack_idx = [0, 2, 1, 3]

    mask = (1 << num_bits) - 1

    for k_tile_id in range(k_tiles):
        first_k = k_tile_id * tile_k_size
        if has_perm:
            sh_perm = perm_np[first_k:first_k + tile_k_size]

        for n_tile_id in range(n_tiles):
            base_col = n_tile_id * out_cols_per_tile  # 列块起点

            for th_id in range(32):
                tc_col = th_id // 4
                tc_row_base = (th_id % 4) * 2

                for warp_id in range(4):
                    cur_n = warp_id * 16 + tc_col
                    vals = [0] * 8
                    for i in range(4):
                        k_idx_local = tc_row_base + tc_offsets[i]
                        if has_perm:
                            k_idx_global = int(sh_perm[k_idx_local])
                        else:
                            k_idx_global = first_k + k_idx_local

                        first_n = n_tile_id * tile_n_size
                        v1 = int(q[k_idx_global, first_n + cur_n]) & mask
                        v2 = int(q[k_idx_global, first_n + cur_n + 8]) & mask
                        vals[i] = v1
                        vals[4 + i] = v2

                    if num_bits == 4:
                        res = 0
                        for i in range(8):
                            res |= (vals[pack_idx[i]] & mask) << (i * 4)
                        out[k_tile_id, base_col + th_id * 4 + warp_id] = res
                    else:
                        res1 = 0
                        res2 = 0
                        for i in range(4):
                            res1 |= (vals[pack_idx[i]] & mask) << (i * 8)
                            res2 |= (vals[4 + pack_idx[i]] & mask) << (i * 8)
                        base = base_col + th_id * 8 + (warp_id * 2)
                        out[k_tile_id, base + 0] = res1
                        out[k_tile_id, base + 1] = res2

    return torch.from_numpy(out.view(np.int32))

def marlin_permute_weights(q_w, size_k, size_n, perm, tile_k=GPTQ_MARLIN_TILE_K, tile_n=GPTQ_MARLIN_TILE_N):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile_k == 0, f"size_k = {size_k}, tile_k = {tile_k}"
    assert size_n % tile_n == 0, f"size_n = {size_n}, tile_n = {tile_n}"

    q_w = q_w.reshape((size_k // tile_k, tile_k, size_n // tile_n, tile_n))
    q_w = q_w.permute((0, 2, 1, 3))                        # [KT, NT, tk, tn]
    q_w = q_w.reshape((size_k // tile_k, size_n * tile_k)) # [KT, NT*tn*tk] -> [KT, N*tk]

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)
    return q_w


def marlin_weights(q_w, size_k, size_n, num_bits, perm):
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)

    pack_factor = get_pack_factor(num_bits)  # 32/num_bits
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)
    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << (num_bits * i)

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)
    return q_packed


def get_weight_perm_gptq(num_bits: int):
    perm_list: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
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


@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("k_tiles,n_tiles", [(1, 1), (2, 2)])
@pytest.mark.parametrize("use_act_order", [False, True])
def test_gptq_marlin_repack_correct(num_bits, k_tiles, n_tiles, use_act_order):
    tile_k, tile_n = GPTQ_MARLIN_TILE_K, GPTQ_MARLIN_TILE_N
    size_k = k_tiles * tile_k
    size_n = n_tiles * tile_n
    pack_factor = 32 // num_bits

    q_w = torch.randint(
        low=0, high=(1 << num_bits), size=(size_k, size_n), dtype=torch.int32, device="xpu"
    )

    b_q_weight = gptq_pack_along_k(q_w, num_bits, size_k, size_n)

    weight_perm = get_weight_perm_gptq(num_bits).to(device="xpu", dtype=torch.int32)
    q_w_marlin = marlin_weights(q_w, size_k, size_n, num_bits, weight_perm)

    if use_act_order:
        perm_kernel = torch.arange(size_k, device="xpu", dtype=torch.int32)
    else:
        perm_kernel = torch.empty(0, device="xpu", dtype=torch.int32)

    #out_gpu = torch.ops.sgl_kernel_sycl.gptq_marlin_repack(b_q_weight, perm_kernel, size_k, size_n, num_bits)
    out_gpu = gptq_marlin_repack(b_q_weight, perm_kernel, size_k, size_n, num_bits)
    assert out_gpu.is_xpu and out_gpu.dtype == torch.int32
    q_w_marlin_ref = gptq_marlin_reference(
        q_w, size_k, size_n, num_bits, has_perm=use_act_order, perm=perm_kernel if use_act_order else None
    ).to(out_gpu.device)
    expected_cols = size_n * tile_k // pack_factor
    assert list(out_gpu.shape) == [size_k // tile_k, expected_cols]

    torch.xpu.synchronize()
    torch.testing.assert_close(out_gpu, q_w_marlin_ref)


if __name__ == "__main__":
    import subprocess
    subprocess.call(["pytest", str(__file__)])
