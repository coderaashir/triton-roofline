"""Charts for the README."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullLocator

from troof.hardware import Hardware

# Fixed categorical order from a colorblind-validated palette; a color follows its entity.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "D", "^", "v"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def _figure(title: str, subtitle: str, ncols: int = 1, figsize=(8, 5.5), top: float = 0.86):
    fig = Figure(figsize=figsize, dpi=150, facecolor=SURFACE)
    fig.subplots_adjust(left=0.1, right=0.97, top=top, bottom=0.12, wspace=0.08)
    axes = fig.subplots(1, ncols, sharey=True, squeeze=False)[0]
    for ax in axes:
        ax.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.1, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.1, 0.895, subtitle, color=INK_2, fontsize=9)
    return fig, axes


def _style(ax, xlabel: str, ylabel: str | None) -> None:
    ax.set_xlabel(xlabel, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, which="both")
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _compact(v: int) -> str:
    if v >= 2**20:
        return f"{v >> 20}M"
    return f"{v >> 10}K" if v >= 2**10 else str(v)


def _pow2_axis(ax, values: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_minor_locator(NullLocator())
    fmt = _compact if max(values) >= 100_000 else "{:,}".format
    ax.set_xticklabels([fmt(v) for v in values])


def _save(fig: Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


@dataclass
class RooflineSeries:
    name: str
    points: list[tuple[float, float]]  # (FLOPs/byte, GFLOP/s)
    labels: dict[int, tuple[str, tuple[float, float]]] = field(default_factory=dict)
    # point index -> (text, where to put it in data coords; a leader line joins them)
    connect: bool = False


def plot_roofline(
    hw: Hardware,
    series: list[RooflineSeries],
    path: str | Path,
    measured_bw_gbs: float | None = None,
) -> None:
    fig, (ax,) = _figure(
        f"Triton kernels on the {hw.name} roofline",
        "Dot = measured speed of my kernel; lines = best possible. Matmul shown at every size.",
    )
    ax.set(xscale="log", yscale="log", xlim=(0.05, 3000), ylim=(5, 150_000))

    xs = np.logspace(math.log10(0.05), math.log10(3000), 400)
    bw = hw.mem_bw_gbs
    fp32, fp16 = (hw.peak_tflops[d] * 1e3 for d in ("fp32", "fp16"))  # GFLOP/s
    ax.plot(xs, np.minimum(fp16, xs * bw), color=INK_2, lw=1.4, zorder=2)
    ax.plot(xs, np.minimum(fp32, xs * bw), color=INK, lw=2, zorder=2)
    if measured_bw_gbs:
        ax.plot(xs, np.minimum(fp16, xs * measured_bw_gbs), color=MUTED, lw=1, ls="--", zorder=1)

    (a, b), (c, d) = ax.transData.transform([(0.1, 0.1), (10, 10)])
    angle = math.degrees(math.atan2(d - b, c - a))
    ax.text(0.09, 0.09 * bw * 1.6, f"memory: {bw:.0f} GB/s spec", rotation=angle,
            rotation_mode="anchor", color=INK, fontsize=9)
    if measured_bw_gbs:
        ax.text(0.5, 0.5 * measured_bw_gbs / 2.6, f"best measured: {measured_bw_gbs:.0f} GB/s",
                rotation=angle, rotation_mode="anchor", color=MUTED, fontsize=8)
    ax.text(2900, fp32 * 1.12, f"fp32 peak: {fp32 / 1e3:.1f} TFLOP/s", color=INK, fontsize=9,
            ha="right")
    ax.text(2900, fp16 * 1.12, f"fp16 tensor-core peak: {fp16 / 1e3:.0f} TFLOP/s", color=INK_2,
            fontsize=9, ha="right")

    for i, s in enumerate(series):
        color, marker = SERIES[i], MARKERS[i]
        x, y = zip(*s.points)
        if s.connect:
            ax.plot(x, y, color=color, lw=1.5, zorder=3)
        ax.scatter(x, y, s=55, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.8,
                   zorder=4, label=s.name)
        for idx, (text, where) in s.labels.items():
            ax.annotate(text, s.points[idx], xytext=where, color=INK, fontsize=8.5, va="center",
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7})

    _style(ax, "Arithmetic intensity (FLOPs / byte)", "Performance (GFLOP/s)")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    _save(fig, path)


def plot_lines(
    path: str | Path,
    title: str,
    subtitle: str,
    panels: list[tuple[str, dict[str, list[tuple[int, float]]]]],
    xlabel: str,
    ylabel: str,
    refs: list[tuple[float, str]] = (),
    log_y: bool = False,
    styles: dict[str, tuple[str, str]] | None = None,
    figsize=(8, 5.5),
) -> None:
    """One or more side-by-side panels sharing a y axis.

    panels: (panel title, {series name: [(x, y), ...]}). refs: horizontal reference lines.
    styles: series name -> (color, linestyle); defaults to categorical order.
    """
    titled = any(t for t, _ in panels)
    fig, axes = _figure(title, subtitle, ncols=len(panels), figsize=figsize,
                        top=0.76 if titled else 0.8)
    names = list(dict.fromkeys(name for _, data in panels for name in data))
    styles = styles or {name: (SERIES[i], "-") for i, name in enumerate(names)}
    for col, (ax, (panel_title, data)) in enumerate(zip(axes, panels)):
        if panel_title:
            ax.set_title(panel_title, color=INK, fontsize=10, loc="left")
        for name, pts in data.items():
            color, ls = styles[name]
            marker = MARKERS[names.index(name) % len(MARKERS)]
            x, y = zip(*pts)
            ax.plot(x, y, color=color, ls=ls, lw=2, marker=marker, ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        for value, text in refs:
            ax.axhline(value, color=MUTED, lw=1, ls=":")
            if col == len(panels) - 1:
                ax.text(1.0, value, f" {text}", transform=ax.get_yaxis_transform(),
                        color=MUTED, fontsize=8, va="bottom", ha="right")
        _pow2_axis(ax, sorted({x for pts in data.values() for x, _ in pts}))
        if log_y:
            ax.set_yscale("log")
        _style(ax, xlabel, ylabel if col == 0 else None)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for h, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(h)
                labels.append(label)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.09, 0.885), ncol=len(labels),
               frameon=False, fontsize=9, labelcolor=INK_2)
    _save(fig, path)
