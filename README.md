# triton-roofline

A small project I built to learn GPU programming: a few Triton kernels, timed on a Colab T4 and compared against a roofline model of the GPU.

![Roofline chart](figures/roofline.png)

## Kernels

- elementwise: copy, add, relu, gelu
- softmax and RMSNorm, each fused into one kernel
- matmul, tiled and autotuned, in fp32 and fp16

Each one is checked against PyTorch before it's timed.

## Results

| Kernel | % of roofline | vs PyTorch |
|---|---|---|
| add / relu / gelu | 72–75% | about the same |
| softmax | 72% | about the same |
| RMSNorm | 72% | 1.45× faster |
| matmul fp32 | 49% | 1.11× faster than cuBLAS |
| matmul fp16 | 3% | much slower than cuBLAS |

More detail is in [results/summary.md](results/summary.md), and the other charts are in [figures/](figures/).

## Things I learned

Most of these kernels are limited by memory, not math. GELU does a lot more math than ReLU but runs at the same speed, because both spend their time moving data. The best any kernel got was about 75% of the T4's spec bandwidth, and PyTorch tops out in the same place.

Fusion makes a big difference. Writing softmax as five separate PyTorch ops moves about 4× as much data as a single fused kernel, and it ran 3.8× slower.

The fp16 matmul was the surprise. It came out slower than fp32, at about 3% of the roofline. I wrote a [diagnostic script](scripts/diagnose_matmul.py) to look at the compiled code, and on the T4 Triton wasn't using the tensor cores at all ([output](results/matmul_diagnostic.txt)). The bigger tile sizes were also spilling a lot of registers.

## Ideas for later

- Try the fp16 matmul with an upcast to fp32 before the dot product
- Run everything on a newer GPU (L4 or A100) where Triton uses tensor cores
- Look at why `torch.compile`'s softmax is slower on long rows

## Running it

Open [notebooks/colab_runner.ipynb](notebooks/colab_runner.ipynb) in Colab with a T4 GPU and run all cells. `python scripts/report.py` rebuilds the charts from the CSVs.
