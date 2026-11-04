# Results summary (NVIDIA T4)

Best measured bandwidth (copy kernel): **233 GB/s**, 73% of the 320 GB/s spec.

## Roofline efficiency per kernel (largest shape)

| Kernel | Shape | Bound | FLOPs/byte | Predicted | Measured | % of roofline | vs PyTorch |
|---|---|---|---|---|---|---|---|
| copy | 67108864 | memory | 0.00 | 1,678 µs | 2,319 µs | 72% | 0.99× |
| add | 67108864 | memory | 0.08 | 2,517 µs | 3,346 µs | 75% | 1.00× |
| relu | 67108864 | memory | 0.12 | 1,678 µs | 2,318 µs | 72% | 1.00× |
| gelu | 67108864 | memory | 1.75 | 1,678 µs | 2,301 µs | 73% | 1.00× |
| softmax | 4096x8192 | memory | 0.62 | 839 µs | 1,158 µs | 72% | 0.97× |
| rmsnorm | 4096x8192 | memory | 0.50 | 839 µs | 1,161 µs | 72% | 1.45× |
| matmul_fp16 | 4096x4096x4096 | compute | 1365 | 2,114 µs | 60,462 µs | 3% | 0.10× |
| matmul_fp32 | 4096x4096x4096 | compute | 683 | 16,968 µs | 34,317 µs | 49% | 1.11× |

## Fusion: softmax on 4096×8192 fp32

| Version | Bytes moved (model) | Time | Effective GB/s |
|---|---|---|---|
| triton | 268 MB | 1,158 µs | 232 |
| torch | 268 MB | 1,122 µs | 239 |
| torch.compile | 268 MB | 2,281 µs | 118 |
| torch unfused | 1,074 MB | 4,385 µs | 61 |

## Matmul by size

| N | Triton fp16 | cuBLAS fp16 | % of roofline | Autotuned tile | Triton fp32 | cuBLAS fp32 |
|---|---|---|---|---|---|---|
| 128 | 0.40 TFLOP/s | 0.68 | 3% | 32x32x32 w2 | 0.30 | 0.41 |
| 256 | 1.38 TFLOP/s | 2.96 | 5% | 32x32x32 w2 | 1.49 | 1.35 |
| 512 | 2.32 TFLOP/s | 14.56 | 4% | 32x32x32 w2 | 3.89 | 4.55 |
| 1024 | 2.45 TFLOP/s | 34.95 | 4% | 32x32x32 w2 | 4.19 | 4.46 |
| 2048 | 2.30 TFLOP/s | 30.45 | 4% | 32x32x32 w2 | 4.03 | 4.01 |
| 4096 | 2.27 TFLOP/s | 22.08 | 3% | 32x32x32 w2 | 4.01 | 3.59 |

## Every kernel and shape (cost model vs measured, Triton only)

| Kernel | Shape | Bound | Predicted | Measured | % of roofline |
|---|---|---|---|---|---|
| copy | 1024 | memory | 0.0 µs | 6.6 µs | 0% |
| add | 1024 | memory | 0.0 µs | 6.2 µs | 1% |
| relu | 1024 | memory | 0.0 µs | 4.2 µs | 1% |
| gelu | 1024 | memory | 0.0 µs | 4.2 µs | 1% |
| copy | 4096 | memory | 0.1 µs | 4.1 µs | 2% |
| add | 4096 | memory | 0.2 µs | 4.2 µs | 4% |
| relu | 4096 | memory | 0.1 µs | 4.1 µs | 2% |
| gelu | 4096 | memory | 0.1 µs | 4.3 µs | 2% |
| copy | 16384 | memory | 0.4 µs | 4.9 µs | 8% |
| add | 16384 | memory | 0.6 µs | 5.4 µs | 11% |
| relu | 16384 | memory | 0.4 µs | 4.9 µs | 8% |
| gelu | 16384 | memory | 0.4 µs | 5.0 µs | 8% |
| copy | 65536 | memory | 1.6 µs | 6.1 µs | 27% |
| add | 65536 | memory | 2.5 µs | 7.7 µs | 32% |
| relu | 65536 | memory | 1.6 µs | 6.1 µs | 27% |
| gelu | 65536 | memory | 1.6 µs | 6.3 µs | 26% |
| copy | 262144 | memory | 6.6 µs | 12.1 µs | 54% |
| add | 262144 | memory | 9.8 µs | 17.0 µs | 58% |
| relu | 262144 | memory | 6.6 µs | 12.0 µs | 55% |
| gelu | 262144 | memory | 6.6 µs | 12.3 µs | 53% |
| copy | 1048576 | memory | 26.2 µs | 38.9 µs | 67% |
| add | 1048576 | memory | 39.3 µs | 57.0 µs | 69% |
| relu | 1048576 | memory | 26.2 µs | 38.9 µs | 67% |
| gelu | 1048576 | memory | 26.2 µs | 38.9 µs | 67% |
| copy | 4194304 | memory | 105 µs | 149 µs | 70% |
| add | 4194304 | memory | 157 µs | 215 µs | 73% |
| relu | 4194304 | memory | 105 µs | 148 µs | 71% |
| gelu | 4194304 | memory | 105 µs | 149 µs | 70% |
| copy | 16777216 | memory | 419 µs | 585 µs | 72% |
| add | 16777216 | memory | 629 µs | 844 µs | 75% |
| relu | 16777216 | memory | 419 µs | 584 µs | 72% |
| gelu | 16777216 | memory | 419 µs | 584 µs | 72% |
| copy | 67108864 | memory | 1,678 µs | 2,319 µs | 72% |
| add | 67108864 | memory | 2,517 µs | 3,346 µs | 75% |
| relu | 67108864 | memory | 1,678 µs | 2,318 µs | 72% |
| gelu | 67108864 | memory | 1,678 µs | 2,301 µs | 73% |
| softmax | 4096x256 | memory | 26.2 µs | 38.4 µs | 68% |
| softmax | 4096x512 | memory | 52.4 µs | 74.0 µs | 71% |
| softmax | 4096x1024 | memory | 105 µs | 148 µs | 71% |
| softmax | 4096x2048 | memory | 210 µs | 291 µs | 72% |
| softmax | 4096x4096 | memory | 419 µs | 576 µs | 73% |
| softmax | 4096x8192 | memory | 839 µs | 1,158 µs | 72% |
| rmsnorm | 4096x256 | memory | 26.2 µs | 38.6 µs | 68% |
| rmsnorm | 4096x512 | memory | 52.4 µs | 75.8 µs | 69% |
| rmsnorm | 4096x1024 | memory | 105 µs | 148 µs | 71% |
| rmsnorm | 4096x2048 | memory | 210 µs | 293 µs | 72% |
| rmsnorm | 4096x4096 | memory | 419 µs | 577 µs | 73% |
| rmsnorm | 4096x8192 | memory | 839 µs | 1,161 µs | 72% |
| matmul_fp16 | 128x128x128 | memory | 0.3 µs | 10.4 µs | 3% |
| matmul_fp16 | 256x256x256 | memory | 1.2 µs | 24.3 µs | 5% |
| matmul_fp16 | 512x512x512 | memory | 4.9 µs | 116 µs | 4% |
| matmul_fp16 | 1024x1024x1024 | compute | 33.0 µs | 877 µs | 4% |
| matmul_fp16 | 2048x2048x2048 | compute | 264 µs | 7,461 µs | 4% |
| matmul_fp16 | 4096x4096x4096 | compute | 2,114 µs | 60,462 µs | 3% |
| matmul_fp32 | 128x128x128 | memory | 0.6 µs | 14.1 µs | 4% |
| matmul_fp32 | 256x256x256 | compute | 4.1 µs | 22.5 µs | 18% |
| matmul_fp32 | 512x512x512 | compute | 33.1 µs | 69.1 µs | 48% |
| matmul_fp32 | 1024x1024x1024 | compute | 265 µs | 513 µs | 52% |
| matmul_fp32 | 2048x2048x2048 | compute | 2,121 µs | 4,266 µs | 50% |
| matmul_fp32 | 4096x4096x4096 | compute | 16,968 µs | 34,317 µs | 49% |
