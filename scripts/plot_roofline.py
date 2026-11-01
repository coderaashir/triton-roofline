"""Plot the elementwise kernels on the T4 roofline from results/elementwise.csv.

No GPU needed: python scripts/plot_roofline.py
"""

import csv
from pathlib import Path

from troof.hardware import T4
from troof.roofline import KernelPoint, efficiency, plot_roofline, predicted_time

ROOT = Path(__file__).resolve().parents[1]

# FLOPs per element. gelu = 4 arithmetic ops + erf (~10-op polynomial): an estimate.
FLOPS_PER_ELEMENT = {"add": 1, "relu": 1, "gelu": 14}


def load_points(path, provider="triton"):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["provider"] == provider]
    n_max = max(int(r["n"]) for r in rows)
    return [
        KernelPoint(
            name=r["kernel"],
            flops=FLOPS_PER_ELEMENT[r["kernel"]] * n_max,
            bytes_moved=int(r["bytes_moved"]),
            seconds=float(r["ms"]) * 1e-3,
        )
        for r in rows
        if int(r["n"]) == n_max
    ]


def best_bandwidth(path):
    with open(path) as f:
        return max(float(r["gbs"]) for r in csv.DictReader(f))


def main():
    csv_path = ROOT / "results" / "elementwise.csv"
    points = load_points(csv_path)

    print(f"{'kernel':<8}{'FLOPs/byte':>12}{'predicted µs':>14}{'measured µs':>13}{'% of roof':>11}")
    for p in points:
        predicted = predicted_time(p.flops, p.bytes_moved, T4) * 1e6
        print(f"{p.name:<8}{p.intensity:>12.3f}{predicted:>14.0f}{p.seconds * 1e6:>13.0f}"
              f"{efficiency(p, T4):>10.0%}")

    out = ROOT / "figures" / "roofline_elementwise.png"
    plot_roofline(T4, points, out, measured_bw_gbs=best_bandwidth(csv_path),
                  label_at={"relu": (0.35, 9), "add": (0.35, 4.5)})
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
