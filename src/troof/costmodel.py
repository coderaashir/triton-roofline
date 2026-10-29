"""Per-kernel cost model: each kernel declares the FLOPs and bytes it must move for a shape.

The roofline then gives a lower bound on its time on any `Hardware`:

    predicted time = max(FLOPs / peak FLOP/s,  bytes / peak bandwidth)

Bytes are the *minimum* traffic: every input read once, every output written once.
A kernel that moves more than that (e.g. unfused softmax) shows up as a low % of roofline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from troof.hardware import Hardware
from troof.roofline import peak_bw, peak_flops

FP32, FP16 = 4, 2  # bytes per element


@dataclass(frozen=True)
class KernelSpec:
    name: str
    dtype: str  # which compute roof applies: "fp32" (FMA units) or "fp16" (tensor cores)
    flops: Callable[..., float]
    bytes_moved: Callable[..., float]


SPECS = {
    s.name: s
    for s in [
        KernelSpec("copy", "fp32", lambda n: 0, lambda n: 2 * n * FP32),
        KernelSpec("add", "fp32", lambda n: n, lambda n: 3 * n * FP32),
        KernelSpec("relu", "fp32", lambda n: n, lambda n: 2 * n * FP32),
        # 4 arithmetic ops + erf (~10-op polynomial): an estimate.
        KernelSpec("gelu", "fp32", lambda n: 14 * n, lambda n: 2 * n * FP32),
        # per element: max, subtract, exp, sum, divide
        KernelSpec("softmax", "fp32", lambda m, n: 5 * m * n, lambda m, n: 2 * m * n * FP32),
        # per element: square, sum, scale by 1/rms, scale by weight; plus the weight vector
        KernelSpec("rmsnorm", "fp32", lambda m, n: 4 * m * n, lambda m, n: (2 * m * n + n) * FP32),
        KernelSpec("matmul_fp16", "fp16", lambda m, n, k: 2 * m * n * k,
                   lambda m, n, k: (m * k + k * n + m * n) * FP16),
        KernelSpec("matmul_fp32", "fp32", lambda m, n, k: 2 * m * n * k,
                   lambda m, n, k: (m * k + k * n + m * n) * FP32),
    ]
}


def softmax_unfused_bytes(m: int, n: int) -> int:
    """Traffic of softmax written as 5 separate PyTorch ops (max, sub, exp, sum, div).

    Reads 5*m*n + 2*m elements and writes 3*m*n + 2*m: about 4x the fused minimum.
    """
    return (8 * m * n + 4 * m) * FP32


@dataclass(frozen=True)
class Prediction:
    flops: float
    bytes_moved: float
    compute_s: float
    memory_s: float

    @property
    def seconds(self) -> float:
        return max(self.compute_s, self.memory_s)

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes_moved

    @property
    def bound(self) -> str:
        return "compute" if self.compute_s > self.memory_s else "memory"


def predict(kernel: str, shape: tuple[int, ...], hw: Hardware) -> Prediction:
    spec = SPECS[kernel]
    flops = spec.flops(*shape)
    nbytes = spec.bytes_moved(*shape)
    return Prediction(flops, nbytes, flops / peak_flops(hw, spec.dtype), nbytes / peak_bw(hw))
