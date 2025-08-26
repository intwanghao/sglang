import numpy as np, sys, os
import torch
from sgl_kernel_sycl import vec_add_usm

a = torch.tensor([1, 2, 3], dtype=torch.float32, device="xpu")
b = torch.tensor([1, 2, 3], dtype=torch.float32, device="xpu")
c = torch.tensor([0, 0, 0], dtype=torch.float32, device="xpu")

vec_add_usm(a, b, c, 3)
d = c.to("cpu")
print("c:", d)
