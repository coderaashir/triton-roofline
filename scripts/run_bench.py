"""Regenerate every CSV in results/ and every chart in figures/.

Run on a GPU (Colab T4): python scripts/run_bench.py
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from troof.bench.harness import measure, print_table, save_csv
from troof.hardware import detect
from troof.kernels.elementwise import add, gelu, relu

RESULTS = Path(__file__).resolve().parents[1] / "results"
SIZES = [2**p for p in range(10, 28, 2)]  # 1K to 64M elements


def bench_elementwise_size(n, hw):
    x = torch.rand(n, device="cuda")
    y = torch.rand(n, device="cuda")
    b = x.element_size()
    # (name, bytes moved, our kernel, torch baseline)
    cases = [
        ("add", 3 * n * b, lambda: add(x, y), lambda: x + y),
        ("relu", 2 * n * b, lambda: relu(x), lambda: torch.relu(x)),
        ("gelu", 2 * n * b, lambda: gelu(x), lambda: F.gelu(x)),
    ]
    results = []
    for kernel, bytes_moved, ours, baseline in cases:
        torch.testing.assert_close(ours(), baseline())
        results.append(measure(kernel, "triton", ours, n, bytes_moved, hw))
        results.append(measure(kernel, "torch", baseline, n, bytes_moved, hw))
    return results


def bench_elementwise(hw):
    return [r for n in SIZES for r in bench_elementwise_size(n, hw)]


def main():
    hw = detect()
    print(f"GPU: {hw.name} ({hw.mem_bw_gbs:.0f} GB/s spec)\n")
    results = bench_elementwise(hw)
    print_table(results)
    save_csv(results, RESULTS / "elementwise.csv")
    print(f"\nSaved {RESULTS / 'elementwise.csv'}")


if __name__ == "__main__":
    main()
