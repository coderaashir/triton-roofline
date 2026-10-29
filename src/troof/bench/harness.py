"""Benchmark harness: time a kernel and compare it with its roofline prediction."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from troof.costmodel import predict
from troof.hardware import Hardware


@dataclass
class Result:
    kernel: str  # key into costmodel.SPECS
    provider: str  # "triton" (ours) or a PyTorch baseline
    shape: str  # e.g. "4096x1024"
    ms: float
    gbs: float  # minimum bytes / time ("effective bandwidth")
    tflops: float
    predicted_ms: float
    pct_roof: float  # predicted / measured
    bound: str
    config: str = ""  # autotuner's pick, for matmul


def measure(
    kernel: str,
    provider: str,
    fn: Callable[[], object],
    shape: tuple[int, ...],
    hw: Hardware,
    config: str = "",
) -> Result:
    import triton

    # do_bench waits for the GPU, warms up, and repeats; we keep the median time.
    ms = triton.testing.do_bench(fn, return_mode="median")
    p = predict(kernel, shape, hw)
    s = ms * 1e-3
    return Result(
        kernel=kernel,
        provider=provider,
        shape="x".join(map(str, shape)),
        ms=ms,
        gbs=p.bytes_moved / s / 1e9,
        tflops=p.flops / s / 1e12,
        predicted_ms=p.seconds * 1e3,
        pct_roof=100 * p.seconds / s,
        bound=p.bound,
        config=config,
    )


def save_csv(results: list[Result], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(Result)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


def load_csv(path: str | Path) -> list[Result]:
    types = {f.name: f.type for f in fields(Result)}
    with Path(path).open() as f:
        return [
            Result(**{k: float(v) if types[k] == "float" else v for k, v in row.items()})
            for row in csv.DictReader(f)
        ]


def print_table(results: list[Result]) -> None:
    print(f"{'kernel':<13}{'provider':<15}{'shape':>16}{'ms':>10}{'GB/s':>8}{'TFLOP/s':>9}"
          f"{'% roof':>8}  config")
    for r in results:
        print(f"{r.kernel:<13}{r.provider:<15}{r.shape:>16}{r.ms:>10.4f}{r.gbs:>8.1f}"
              f"{r.tflops:>9.2f}{r.pct_roof:>7.1f}%  {r.config}")
