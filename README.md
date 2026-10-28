# triton-roofline

Triton GPU kernels, benchmarked and compared against a roofline model I built myself.

> **Status:** in progress (started Oct 27, 2026). This README turns into the full report in week 3; the roofline chart will go here.

## What this is

For each kernel (elementwise ops, fused softmax, RMSNorm, matmul) the repo:

1. checks correctness against PyTorch,
2. measures speed (GB/s, TFLOP/s),
3. predicts the best possible speed from a roofline model of the GPU,
4. reports how close the kernel gets to that limit, and why.

## Layout

```
src/troof/
  hardware.py      GPU specs (bandwidth, peak FLOP/s)
  roofline.py      arithmetic intensity -> predicted lower-bound time, plots
  kernels/         Triton kernels
  bench/           benchmark harness
tests/             correctness tests vs PyTorch
scripts/           run_bench.py regenerates results/ and figures/
notebooks/         colab_runner.ipynb runs everything on a Colab T4
results/           raw benchmark CSVs
figures/           charts
```

## Running

Triton needs an NVIDIA GPU on Linux. Open `notebooks/colab_runner.ipynb` in Google Colab with a T4 runtime, then run all cells.
