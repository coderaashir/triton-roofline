# triton-roofline

A small project I built to learn GPU programming: a few Triton kernels, timed on a Colab T4 and compared against a roofline model of the GPU.

![Roofline chart](figures/roofline.png)

## Kernels

- elementwise: copy, add, relu, gelu
- softmax and RMSNorm, each fused into one kernel
- matmul, tiled and autotuned, in fp32 and fp16

Each one is checked against PyTorch before it's timed

## Results

| Kernel | % of roofline | vs PyTorch |
|---|---|---|
| add / relu / gelu | 72–75% | about the same |
| softmax | 72% | about the same |
| RMSNorm | 72% | 1.45× faster |
| matmul fp32 | 49% | 1.11× faster than cuBLAS |
| matmul fp16 | 3% | much slower than cuBLAS |

More detail is in [results/summary.md](results/summary.md), and the other charts are in [figures/](figures/).

## Running it

Open [notebooks/colab_runner.ipynb](notebooks/colab_runner.ipynb) in Colab with a T4 GPU and run all cells
