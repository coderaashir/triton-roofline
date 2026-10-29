"""Roofline model: the best FLOP/s a kernel can reach given its arithmetic intensity."""

from __future__ import annotations

from troof.hardware import Hardware


def peak_flops(hw: Hardware, dtype: str = "fp32") -> float:
    return hw.peak_tflops[dtype] * 1e12


def peak_bw(hw: Hardware) -> float:
    return hw.mem_bw_gbs * 1e9


def attainable_flops(intensity: float, hw: Hardware, dtype: str = "fp32") -> float:
    """The roof: the best FLOP/s any kernel with this intensity (FLOPs/byte) can reach."""
    return min(peak_flops(hw, dtype), intensity * peak_bw(hw))


def predicted_time(flops: float, bytes_moved: float, hw: Hardware, dtype: str = "fp32") -> float:
    """Lower bound in seconds: a kernel can't beat its math time or its memory time."""
    return max(flops / peak_flops(hw, dtype), bytes_moved / peak_bw(hw))
