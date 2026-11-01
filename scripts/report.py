"""Turn results/*.csv into the README's charts (figures/) and tables (results/summary.md).

No GPU needed: python scripts/report.py
"""

import sys
from pathlib import Path

from troof.bench.harness import load_csv
from troof.costmodel import SPECS, predict, softmax_unfused_bytes
from troof.hardware import T4
from troof.plots import SERIES, RooflineSeries, plot_lines, plot_roofline

ROOT = Path(__file__).resolve().parents[1]
# Optional args: python scripts/report.py [results_dir] [figures_dir]
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
FIGURES = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "figures"
hw = T4


def dims(shape: str) -> tuple[int, ...]:
    return tuple(int(d) for d in shape.split("x"))


def pick(rows, kernel, provider, shape=None):
    """Rows for one kernel+provider; with shape=None, every shape in run order."""
    out = [r for r in rows if r.kernel == kernel and r.provider == provider]
    return [r for r in out if r.shape == shape] if shape else out


def largest(rows, kernel, provider="triton"):
    return max(pick(rows, kernel, provider), key=lambda r: dims(r.shape))


def point(r):
    p = predict(r.kernel, dims(r.shape), hw)
    return p.intensity, r.tflops * 1e3


def fmt_us(ms: float) -> str:
    us = ms * 1e3
    return f"{us:,.0f} µs" if us >= 100 else f"{us:.1f} µs"


def roofline_chart(rows, ceiling):
    ew = [largest(rows, k) for k in ("add", "relu", "gelu")]
    sm, rn = largest(rows, "softmax"), largest(rows, "rmsnorm")
    mm16, mm32 = pick(rows, "matmul_fp16", "triton"), pick(rows, "matmul_fp32", "triton")

    def pct(r):
        return f"{r.pct_roof:.0f}%"

    series = [
        RooflineSeries("elementwise (add, relu, gelu)", [point(r) for r in ew], {
            0: (f"add {pct(ew[0])}", (0.07, 9)),
            1: (f"relu {pct(ew[1])}", (0.2, 12)),
            2: (f"gelu {pct(ew[2])}", (3, 60)),
        }),
        RooflineSeries("softmax", [point(sm)], {0: (f"softmax {pct(sm)}", (2.2, 30))}),
        RooflineSeries("rmsnorm", [point(rn)], {0: (f"rmsnorm {pct(rn)}", (1.5, 15))}),
        RooflineSeries("matmul fp32", [point(r) for r in mm32], {
            len(mm32) - 1: (f"fp32 {mm32[-1].shape.split('x')[0]}³: {pct(mm32[-1])}", (400, 900)),
        }, connect=True),
        RooflineSeries("matmul fp16", [point(r) for r in mm16], {
            0: (f"fp16 {mm16[0].shape.split('x')[0]}³", (60, 90)),
            len(mm16) - 1: (f"fp16 {mm16[-1].shape.split('x')[0]}³: {pct(mm16[-1])}",
                            (150, 33000)),
        }, connect=True),
    ]
    plot_roofline(hw, series, FIGURES / "roofline.png", measured_bw_gbs=ceiling)


def elementwise_chart(rows):
    data = {
        k: [(dims(r.shape)[0], r.gbs) for r in pick(rows, k, "triton")]
        for k in ("copy", "add", "relu", "gelu")
    }
    plot_lines(
        FIGURES / "elementwise.png",
        "Elementwise kernels: bandwidth vs input size",
        "Small inputs are dominated by launch overhead; large ones reach the T4's practical ceiling.",
        [("", data)],
        "Elements (fp32)",
        "Bandwidth (GB/s)",
        refs=[(hw.mem_bw_gbs, f"{hw.mem_bw_gbs:.0f} GB/s spec")],
        figsize=(8, 5),
    )


def fusion_chart(rows, ceiling):
    providers = ["triton", "torch", "torch.compile", "torch unfused"]
    panels = [
        (title, {
            p: [(dims(r.shape)[1], r.gbs) for r in pick(rows, kernel, p)]
            for p in providers if pick(rows, kernel, p)
        })
        for kernel, title in (("softmax", "Softmax"), ("rmsnorm", "RMSNorm"))
    ]
    plot_lines(
        FIGURES / "fusion.png",
        "Fusion: softmax and RMSNorm on 4096 rows",
        "Effective bandwidth = minimum bytes / time. Higher is better.",
        panels,
        "Row length",
        "Effective bandwidth (GB/s)",
        refs=[(hw.mem_bw_gbs, "spec"), (ceiling, "copy ceiling")],
        figsize=(9, 5),
    )


def matmul_chart(rows):
    series = {
        "Triton fp16": ("matmul_fp16", "triton"), "cuBLAS fp16": ("matmul_fp16", "torch"),
        "Triton fp32": ("matmul_fp32", "triton"), "cuBLAS fp32": ("matmul_fp32", "torch"),
    }
    data = {
        name: [(dims(r.shape)[0], r.tflops) for r in pick(rows, k, p)]
        for name, (k, p) in series.items()
    }
    styles = {
        "Triton fp16": (SERIES[0], "-"), "cuBLAS fp16": (SERIES[1], "-"),
        "Triton fp32": (SERIES[0], "--"), "cuBLAS fp32": (SERIES[1], "--"),
    }
    plot_lines(
        FIGURES / "matmul.png",
        "Matmul: TFLOP/s vs matrix size (square, N×N×N)",
        "Solid = fp16 on tensor cores, dashed = fp32. Dotted lines are the T4's peaks.",
        [("", data)],
        "N",
        "TFLOP/s",
        refs=[(hw.peak_tflops["fp16"], "fp16 peak"), (hw.peak_tflops["fp32"], "fp32 peak")],
        log_y=True,
        styles=styles,
    )


def summary(rows, ceiling) -> str:
    out = [f"# Results summary ({hw.name})", "",
           (f"Best measured bandwidth (copy kernel): **{ceiling:.0f} GB/s**, "
           f"{100 * ceiling / hw.mem_bw_gbs:.0f}% of the {hw.mem_bw_gbs:.0f} GB/s spec."), ""]

    out += ["## Roofline efficiency per kernel (largest shape)", "",
            ("| Kernel | Shape | Bound | FLOPs/byte | Predicted | Measured | % of roofline "
            "| vs PyTorch |"), "|---|---|---|---|---|---|---|---|"]
    for k in ("copy", "add", "relu", "gelu", "softmax", "rmsnorm", "matmul_fp16", "matmul_fp32"):
        r = largest(rows, k)
        p = predict(k, dims(r.shape), hw)
        torch_r = pick(rows, k, "torch", r.shape)[0]
        ai = f"{p.intensity:.2f}" if p.intensity < 10 else f"{p.intensity:.0f}"
        out.append(f"| {k} | {r.shape} | {p.bound} | {ai} | {fmt_us(r.predicted_ms)} "
                   f"| {fmt_us(r.ms)} | {r.pct_roof:.0f}% | {torch_r.ms / r.ms:.2f}× |")

    m, n = dims(largest(rows, "softmax").shape)
    out += ["", f"## Fusion: softmax on {m}×{n} fp32", "",
            "| Version | Bytes moved (model) | Time | Effective GB/s |", "|---|---|---|---|"]
    for prov in ("triton", "torch", "torch.compile", "torch unfused"):
        r = pick(rows, "softmax", prov, f"{m}x{n}")[0]
        nbytes = softmax_unfused_bytes(m, n) if prov == "torch unfused" else \
            SPECS["softmax"].bytes_moved(m, n)
        out.append(f"| {prov} | {nbytes / 1e6:,.0f} MB | {fmt_us(r.ms)} | {r.gbs:.0f} |")

    out += ["", "## Matmul by size", "",
            ("| N | Triton fp16 | cuBLAS fp16 | % of roofline | Autotuned tile "
            "| Triton fp32 | cuBLAS fp32 |"), "|---|---|---|---|---|---|---|"]
    for r16 in pick(rows, "matmul_fp16", "triton"):
        c16 = pick(rows, "matmul_fp16", "torch", r16.shape)[0]
        r32 = pick(rows, "matmul_fp32", "triton", r16.shape)[0]
        c32 = pick(rows, "matmul_fp32", "torch", r16.shape)[0]
        out.append(f"| {dims(r16.shape)[0]} | {r16.tflops:.2f} TFLOP/s | {c16.tflops:.2f} "
                   f"| {r16.pct_roof:.0f}% | {r16.config} | {r32.tflops:.2f} | {c32.tflops:.2f} |")

    out += ["", "## Every kernel and shape (cost model vs measured, Triton only)", "",
            "| Kernel | Shape | Bound | Predicted | Measured | % of roofline |",
            "|---|---|---|---|---|---|"]
    for r in rows:
        if r.provider == "triton":
            out.append(f"| {r.kernel} | {r.shape} | {r.bound} | {fmt_us(r.predicted_ms)} "
                       f"| {fmt_us(r.ms)} | {r.pct_roof:.0f}% |")
    return "\n".join(out) + "\n"


def main():
    rows = [r for name in ("elementwise", "softmax", "rmsnorm", "matmul")
            for r in load_csv(RESULTS / f"{name}.csv")]
    ceiling = max(r.gbs for r in rows if r.kernel == "copy")
    roofline_chart(rows, ceiling)
    elementwise_chart(rows)
    fusion_chart(rows, ceiling)
    matmul_chart(rows)
    text = summary(rows, ceiling)
    (RESULTS / "summary.md").write_text(text)
    print(text)
    print(f"Saved charts to {FIGURES} and tables to {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    main()
