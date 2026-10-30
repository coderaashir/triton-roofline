import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    return output


@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


def relu(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=block_size)
    return output


@triton.jit
def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(output_ptr + offsets, output, mask=mask)


def gelu(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=block_size)
    return output
