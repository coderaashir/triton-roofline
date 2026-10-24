import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(in_ptr + row * in_row_stride + offsets, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    out = numerator / tl.sum(numerator, axis=0)
    tl.store(out_ptr + row * out_row_stride + offsets, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    num_warps = 4 if block_size < 2048 else 8 if block_size < 4096 else 16
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        out, x, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=block_size, num_warps=num_warps
    )
    return out


def softmax_unfused(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    return numerator / denominator[:, None]
