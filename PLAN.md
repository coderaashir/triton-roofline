# Triton Roofline Project — 16-Day Plan

**Goal:** a polished project to show the MTIA Kernels & Optimizations hiring manager: Triton kernels, measured, and compared against a roofline model I built myself.
**Send date:** Mon Nov 17, 2026 · **Budget:** ~2–2.5 hrs/day, 16 working days (~35–40 hrs)

---

## The 2-minute primer

- **GPU:** a chip with thousands of small cores doing math in parallel. The math is fast; moving data in and out of GPU memory is usually the slow part.
- **Kernel:** a function that runs on the GPU.
- **Triton:** a Python-like language for writing kernels (simpler than CUDA; MTIA uses it).
- **Roofline:** a chart of how fast a kernel *could* go, capped by memory speed ("memory-bound") or math speed ("compute-bound"). Compare real speed to that cap.

The whole project: write kernels → measure them → compare to the cap.

---

## Setup: GPU

- **Yes, you need a GPU.** Triton only runs on NVIDIA GPUs (and won't install natively on macOS), and the project is about measuring speed.
- **Don't buy one.** Use **Google Colab's free T4**: Runtime → Change runtime type → T4 GPU.
- **Workflow:** edit on Mac → `git push` → in Colab: `git pull`, `pip install -e .`, `pytest`, run benchmarks → commit results back.
- **Colab caveats:** sessions disconnect after idle time and there are daily usage caps. Colab Pro (~$10/mo) is optional if free access gets annoying.
- **Optional, Day 14 only:** ~$5–10 of L4/A10 time on Modal or RunPod for bf16/fp8. The T4 doesn't support those.

**T4 numbers to use:** 320 GB/s memory bandwidth · 8.1 TFLOP/s fp32 · 65 TFLOP/s fp16 (tensor cores)

---

## Week 1: basics and your first roofline

- [ ] **Day 1 (Mon Oct 27): mental model and setup**
  - Learn: Horace He, "Making Deep Learning Go Brrrr From First Principles". Compute vs memory vs overhead.
  - Build: GitHub repo + skeleton. Colab on T4: `!nvidia-smi`, `torch.cuda.is_available()`.
  - Done when: you can say why adding two big lists is limited by memory, not math.

- [ ] **Day 2 (Tue Oct 28): your first kernel**
  - Learn: Triton tutorial 01 (vector add). Programs, `BLOCK_SIZE`, `tl.load`/`tl.store`, masks.
  - Build: run it in Colab → `kernels/elementwise.py` + test.
  - Done when: the test passes and you can explain the mask.

- [ ] **Day 3 (Wed Oct 29): more kernels and testing**
  - Learn: `torch.testing.assert_close`, tolerances, fp32 vs fp16.
  - Build: `relu`, `gelu` kernels + tests over several sizes (include odd ones like 1000003).

- [ ] **Day 4 (Thu Oct 30): measuring speed**
  - Learn: GPU work is async; use `triton.testing.do_bench`. GB/s = bytes ÷ time. fp32 add of length N moves 12N bytes.
  - Build: `bench/harness.py` → GB/s and % of 320 GB/s. Add a copy kernel to measure *achievable* bandwidth.
  - Done when: you have a size vs GB/s table. Note why small sizes are slow (overhead).

- [ ] **Day 5 (Sat Nov 1): the roofline**
  - Learn: Williams et al. roofline paper, pp. 1–4. Arithmetic intensity = FLOPs ÷ bytes. Ridge point = peak FLOPs ÷ peak BW (T4 fp32 ≈ 25 FLOPs/byte).
  - Build: `roofline.py`. Log-log roof plus dots for add/relu/gelu.
  - Done when: the first chart is in `figures/`. 🎉

- [ ] **Day 6 (Sun Nov 2): catch-up and notes**
  - Finish leftovers. Write the Week 1 section of `NOTES.md` and start `GLOSSARY.md`.

## Week 2: fusion and matmul

- [ ] **Day 7 (Mon Nov 3): fused softmax**
  - Learn: Triton tutorial 02. Fusion = read once, compute on-chip, write once. Reductions (max, sum).
  - Build: softmax kernel + test.

- [ ] **Day 8 (Tue Nov 4): proving fusion helps**
  - Build: benchmark unfused step-by-step torch vs `torch.softmax` vs `torch.compile` vs yours. Count bytes for each (fused ≈ 4× less traffic).
  - Done when: you have a before/after chart and a one-sentence explanation. **Headline result.**

- [ ] **Day 9 (Thu Nov 6): RMSNorm**
  - Learn: RMSNorm (used in Llama). Another fusion win. Tutorial 05 patterns, forward pass only.
  - Build: kernel + test + benchmark vs torch and `torch.compile`.

- [ ] **Day 10 (Fri Nov 7): matmul, part 1**
  - Learn: matmul = 2·M·N·K FLOPs. Tiling = load once, reuse many times. Tensor cores. (Simon Boehm's CUDA matmul blog has great diagrams.)
  - Build: Triton tutorial 03 in fp16 → passing test.

- [ ] **Day 11 (Sat Nov 8): matmul, part 2**
  - Learn: `@triton.autotune`, `num_warps`, `num_stages`.
  - Build: sweep sizes 128 → 4096, plot TFLOP/s vs size against `torch.matmul`.
  - Done when: the chart shows small = memory-bound, large = compute-bound, and you can say why.

## Week 3: the roofline tool and the write-up

- [ ] **Day 12 (Mon Nov 10): the cost model**
  - Build: each kernel declares `flops(shape)` and `bytes(shape)`. One function returns predicted time, measured time, and efficiency. Generate the table automatically.

- [ ] **Day 13 (Tue Nov 11): one command to rebuild everything**
  - Build: `scripts/run_bench.py` regenerates all CSVs and charts. **Hero chart:** every kernel on one roofline.
  - Decision point: on schedule → stretch goal on Day 14; behind → polish.

- [ ] **Day 14 (Thu Nov 13): stretch goal or polish**
  - On the T4: fp32 vs fp16 matmul to show the tensor-core speedup on the roofline. (bf16/fp8 needs a rented L4.)

- [ ] **Day 15 (Fri Nov 14): README as a report**
  - Order: hero chart → headline table → per-kernel sections (predicted vs achieved, why the gap) → memory- vs compute-bound and why → what I'd try next.

- [ ] **Day 16 (Sun Nov 16): polish and prep**
  - AI review of the README for clarity. 1-page summary for the HM. Reread `GLOSSARY.md`, practice a 2-min walkthrough of the chart.

- [ ] **Mon Nov 17: send it.** Lead with the roofline chart.

---

## Rules of thumb

- **Use AI freely** for scaffolding, plots, CI, and formatting.
- **Keep for yourself:** the numbers and what they mean, the "what I'd try next" section, and the vocabulary.
- **Every number in the README comes from a real run in `results/`.**
- **When confused,** paste the code and ask AI: *"Explain this like I'm new to GPUs. What's happening in memory, and why?"* Then put the answer in `GLOSSARY.md` in your own words.
- **Commit every 1–2 days.** A steady contribution graph signals motivation.
- **Finishing cleanly beats ambition.**
