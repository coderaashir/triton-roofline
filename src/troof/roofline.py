"""Roofline model: arithmetic intensity -> predicted lower-bound time, plus plots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from troof.hardware import Hardware


@dataclass
class KernelPoint:
    name: str
    flops: float
    bytes_moved: float
    seconds: float  # measured

    @property
    def intensity(self) -> float:
        """Arithmetic intensity: FLOPs per byte moved."""
        return self.flops / self.bytes_moved

    @property
    def achieved_flops(self) -> float:
        return self.flops / self.seconds


def peak_flops(hw: Hardware, dtype: str = "fp32") -> float:
    return hw.peak_tflops[dtype] * 1e12


def peak_bw(hw: Hardware) -> float:
    return hw.mem_bw_gbs * 1e9


def attainable_flops(intensity: float, hw: Hardware, dtype: str = "fp32") -> float:
    """The roof: the best FLOP/s any kernel with this intensity can reach."""
    return min(peak_flops(hw, dtype), intensity * peak_bw(hw))


def predicted_time(flops: float, bytes_moved: float, hw: Hardware, dtype: str = "fp32") -> float:
    """Lower bound in seconds: a kernel can't beat its math time or its memory time."""
    return max(flops / peak_flops(hw, dtype), bytes_moved / peak_bw(hw))


def efficiency(point: KernelPoint, hw: Hardware, dtype: str = "fp32") -> float:
    """Fraction of the roofline reached: predicted lower bound / measured time."""
    return predicted_time(point.flops, point.bytes_moved, hw, dtype) / point.seconds


# Chart colors: fixed categorical order, validated for colorblind separation.
SERIES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "D")]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def plot_roofline(
    hw: Hardware,
    points: list[KernelPoint],
    path: str | Path,
    dtype: str = "fp32",
    measured_bw_gbs: float | None = None,
    label_at: dict[str, tuple[float, float]] | None = None,
) -> None:
    """label_at moves a point's label to (intensity, GFLOP/s) with a leader line."""
    label_at = label_at or {}
    fig = Figure(figsize=(8, 5.5), dpi=150, facecolor=SURFACE)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.12)
    ax = fig.add_subplot(facecolor=SURFACE)
    ax.set(xscale="log", yscale="log", xlim=(0.01, 1000), ylim=(1, 30_000))

    xs = np.logspace(-2, 3, 400)
    peak_g = peak_flops(hw, dtype) / 1e9
    ax.plot(xs, np.minimum(peak_g, xs * hw.mem_bw_gbs), color=INK, lw=2, zorder=2)
    if measured_bw_gbs:
        ax.plot(xs, np.minimum(peak_g, xs * measured_bw_gbs), color=INK_2, lw=1.2, ls="--")

    ridge = hw.ridge_point(dtype)
    ax.axvline(ridge, color=MUTED, lw=1, ls=":", zorder=1)

    def slope_angle(x0: float, x1: float) -> float:
        (a, b), (c, d) = ax.transData.transform([(x0, x0), (x1, x1)])
        return math.degrees(math.atan2(d - b, c - a))

    angle = slope_angle(0.02, 2)
    ax.text(0.03, 0.03 * hw.mem_bw_gbs * 1.5, f"Memory bandwidth: {hw.mem_bw_gbs:.0f} GB/s spec",
            rotation=angle, rotation_mode="anchor", color=INK, fontsize=9)
    if measured_bw_gbs:
        ax.text(6, 6 * measured_bw_gbs / 2.2, f"best measured: {measured_bw_gbs:.0f} GB/s",
                rotation=angle, rotation_mode="anchor", color=INK_2, fontsize=8.5)
    ax.text(ridge * 1.3, peak_g * 1.25, f"Peak compute: {hw.peak_tflops[dtype]} TFLOP/s ({dtype})",
            color=INK, fontsize=9)
    ax.text(ridge * 1.08, 1.4, f"ridge point\n{ridge:.0f} FLOPs/byte", color=MUTED, fontsize=8)
    ax.text(0.012, 15_000, "MEMORY-BOUND", color=MUTED, fontsize=8, fontweight="bold")
    ax.text(60, 2500, "COMPUTE-BOUND", color=MUTED, fontsize=8, fontweight="bold")

    for (color, marker), p in zip(SERIES, points):
        y = p.achieved_flops / 1e9
        ax.scatter(p.intensity, y, s=70, color=color, marker=marker,
                   edgecolor=SURFACE, linewidth=2, zorder=3, label=p.name)
        text = f"{p.name}: {efficiency(p, hw, dtype):.0%} of roof"
        if p.name in label_at:
            ax.annotate(text, (p.intensity, y), xytext=label_at[p.name], color=INK, fontsize=9,
                        va="center", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
        else:
            ax.annotate(text, (p.intensity, y), xytext=(10, -4), textcoords="offset points",
                        color=INK, fontsize=9)

    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)", color=INK_2)
    ax.set_ylabel("Performance (GFLOP/s)", color=INK_2)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, which="both")
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93), frameon=False, fontsize=9,
              labelcolor=INK_2)
    fig.suptitle(f"Triton kernels on the {hw.name} roofline", x=0.11, ha="left",
                 color=INK, fontsize=13, fontweight="bold")
    fig.text(0.11, 0.905, "Largest size measured; dot = achieved speed, line = best possible",
             color=INK_2, fontsize=9)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
