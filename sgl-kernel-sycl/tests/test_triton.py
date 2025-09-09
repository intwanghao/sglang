import triton
import torch
import triton.language as tl
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

x = torch.randn(1024, device='xpu')
y = torch.randn(1024, device='xpu')
out = torch.empty_like(x)

add_kernel[(x.numel() // 128,)](x, y, out, x.numel(), BLOCK_SIZE=128)
