import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row * row_stride + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(out_ptr + row * row_stride + offsets, x * rstd * w, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    num_warps = 4 if block_size < 2048 else 8 if block_size < 4096 else 16
    out = torch.empty_like(x)
    rmsnorm_kernel[(n_rows,)](
        x, weight, out, x.stride(0), n_cols, eps, BLOCK_SIZE=block_size, num_warps=num_warps
    )
    return out
