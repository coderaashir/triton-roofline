"""Benchmark harness: time a kernel and report GB/s and % of peak bandwidth."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import triton

from troof.hardware import Hardware


@dataclass
class Result:
    kernel: str
    provider: str  # "triton" (ours) or "torch" (baseline)
    n: int
    bytes_moved: int
    ms: float
    gbs: float
    pct_peak: float


def measure(
    kernel: str,
    provider: str,
    fn: Callable[[], object],
    n: int,
    bytes_moved: int,
    hw: Hardware,
) -> Result:
    # do_bench waits for the GPU, warms up, and repeats; we keep the median time.
    ms = triton.testing.do_bench(fn, return_mode="median")
    gbs = bytes_moved / (ms * 1e-3) / 1e9
    return Result(kernel, provider, n, bytes_moved, ms, gbs, 100 * gbs / hw.mem_bw_gbs)


def save_csv(results: list[Result], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


def print_table(results: list[Result]) -> None:
    print(f"{'kernel':<8}{'provider':<10}{'n':>12}{'ms':>10}{'GB/s':>9}{'% peak':>9}")
    for r in results:
        print(
            f"{r.kernel:<8}{r.provider:<10}{r.n:>12,}{r.ms:>10.4f}{r.gbs:>9.1f}{r.pct_peak:>8.1f}%"
        )
