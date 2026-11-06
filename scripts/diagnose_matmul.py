"""Why is the fp16 matmul slow? Time every tile config on its own and inspect the compiled PTX.

Run on a GPU (Colab T4): python scripts/diagnose_matmul.py
"""

import torch
import triton

from troof.kernels.matmul import _TILES, matmul_kernel

N = 2048
FLOPS = 2 * N**3


def run(dtype, bm, bn, bk, warps, stages):
    a = torch.randn(N, N, device="cuda", dtype=dtype)
    b = torch.randn(N, N, device="cuda", dtype=dtype)
    c = torch.empty(N, N, device="cuda", dtype=dtype)
    grid = (triton.cdiv(N, bm) * triton.cdiv(N, bn),)

    def launch():
        return matmul_kernel.fn[grid](
            a, b, c, N, N, N, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
            num_warps=warps, num_stages=stages,
        )

    name = f"{str(dtype)[6:]:<8}{bm}x{bn}x{bk} w{warps} s{stages}"
    try:
        compiled = launch()
        ms = triton.testing.do_bench(launch, return_mode="median")
    except Exception as e:  # noqa: BLE001 (e.g. out of shared memory)
        print(f"{name:<32} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
        return
    ptx = compiled.asm["ptx"]
    print(f"{name:<32}{FLOPS / ms / 1e9:>7.2f} TFLOP/s  regs={getattr(compiled, 'n_regs', '?'):<4}"
          f" spills={getattr(compiled, 'n_spills', '?'):<4} mma={ptx.count('mma.sync')}"
          f" ldmatrix={ptx.count('ldmatrix')} fma={ptx.count('fma.rn')} cp.async={ptx.count('cp.async')}")


def main():
    print(f"torch {torch.__version__} | triton {triton.__version__} | "
          f"{torch.cuda.get_device_name()} sm_{''.join(map(str, torch.cuda.get_device_capability()))}")
    print(f"matmul {N}x{N}x{N}; cuBLAS fp16 for reference:", end=" ")
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    ms = triton.testing.do_bench(lambda: a @ a, return_mode="median")
    print(f"{FLOPS / ms / 1e9:.2f} TFLOP/s\n")
    for dtype in (torch.float16, torch.float32):
        for bm, bn, bk, warps in _TILES:
            for stages in (1, 2, 3):
                run(dtype, bm, bn, bk, warps, stages)
        print()


if __name__ == "__main__":
    main()
