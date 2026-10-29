"""Run every benchmark on the GPU and write results/*.csv.

Run on a GPU (Colab T4): python scripts/run_bench.py
Then make the charts and tables anywhere: python scripts/report.py
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from troof.bench.harness import measure, print_table, save_csv
from troof.hardware import detect
from troof.kernels.elementwise import add, copy, gelu, relu
from troof.kernels.matmul import best_config, matmul
from troof.kernels.rmsnorm import rmsnorm
from troof.kernels.softmax import softmax, softmax_unfused

RESULTS = Path(__file__).resolve().parents[1] / "results"

ELEMENTWISE_SIZES = [2**p for p in range(10, 28, 2)]  # 1K to 64M elements
ROWS = 4096
ROW_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
MATMUL_SIZES = [128, 256, 512, 1024, 2048, 4096]

softmax_compiled = torch.compile(lambda x: torch.softmax(x, dim=-1))
rmsnorm_compiled = torch.compile(lambda x, w: F.rms_norm(x, (x.shape[-1],), w, eps=1e-6))


def run_cases(kernel, shape, cases, hw, reference, **tol):
    """Check every provider against the reference, then time each one."""
    results = []
    for provider, fn in cases.items():
        torch.testing.assert_close(fn(), reference, **tol)
        results.append(measure(kernel, provider, fn, shape, hw))
    return results


def bench_elementwise_size(n, hw):
    x = torch.rand(n, device="cuda")
    y = torch.rand(n, device="cuda")
    out = torch.empty_like(x)
    return [
        *run_cases("copy", (n,), {"triton": lambda: copy(x), "torch": lambda: out.copy_(x)},
                   hw, x),
        *run_cases("add", (n,), {"triton": lambda: add(x, y), "torch": lambda: x + y}, hw, x + y),
        *run_cases("relu", (n,), {"triton": lambda: relu(x), "torch": lambda: torch.relu(x)},
                   hw, torch.relu(x)),
        *run_cases("gelu", (n,), {"triton": lambda: gelu(x), "torch": lambda: F.gelu(x)},
                   hw, F.gelu(x)),
    ]


def bench_softmax_shape(n, hw):
    x = torch.randn(ROWS, n, device="cuda")
    cases = {
        "triton": lambda: softmax(x),
        "torch": lambda: torch.softmax(x, dim=-1),
        "torch.compile": lambda: softmax_compiled(x),
        "torch unfused": lambda: softmax_unfused(x),
    }
    return run_cases("softmax", (ROWS, n), cases, hw, torch.softmax(x, dim=-1))


def bench_rmsnorm_shape(n, hw):
    x = torch.randn(ROWS, n, device="cuda")
    w = torch.rand(n, device="cuda")
    cases = {
        "triton": lambda: rmsnorm(x, w),
        "torch": lambda: F.rms_norm(x, (n,), w, eps=1e-6),
        "torch.compile": lambda: rmsnorm_compiled(x, w),
    }
    return run_cases("rmsnorm", (ROWS, n), cases, hw, F.rms_norm(x, (n,), w, eps=1e-6))


def bench_matmul_size(s, dtype, hw):
    a = torch.randn(s, s, device="cuda", dtype=dtype)
    b = torch.randn(s, s, device="cuda", dtype=dtype)
    kernel = "matmul_fp16" if dtype == torch.float16 else "matmul_fp32"
    tol = {"atol": 0.1, "rtol": 0.02} if dtype == torch.float16 else {"atol": 1e-2, "rtol": 1e-3}
    torch.testing.assert_close(matmul(a, b), a @ b, **tol)  # also runs the autotuner
    return [
        measure(kernel, "triton", lambda: matmul(a, b), (s, s, s), hw, config=best_config()),
        measure(kernel, "torch", lambda: a @ b, (s, s, s), hw),
    ]


def main():
    hw = detect()
    print(f"GPU: {hw.name} ({hw.mem_bw_gbs:.0f} GB/s, {hw.peak_tflops} TFLOP/s spec)\n")
    suites = {
        "elementwise": lambda: [r for n in ELEMENTWISE_SIZES for r in bench_elementwise_size(n, hw)],
        "softmax": lambda: [r for n in ROW_LENGTHS for r in bench_softmax_shape(n, hw)],
        "rmsnorm": lambda: [r for n in ROW_LENGTHS for r in bench_rmsnorm_shape(n, hw)],
        "matmul": lambda: [
            r
            for dtype in (torch.float16, torch.float32)
            for s in MATMUL_SIZES
            for r in bench_matmul_size(s, dtype, hw)
        ],
    }
    for name, run in suites.items():
        print(f"== {name}")
        results = run()
        print_table(results)
        save_csv(results, RESULTS / f"{name}.csv")
        print()


if __name__ == "__main__":
    main()
