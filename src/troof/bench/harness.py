"""Benchmark harness: time a kernel and report GB/s and % of peak bandwidth.

TODO(Day 4):
- time with triton.testing.do_bench (handles warmup and GPU async)
- GB/s = bytes moved / time
- % of peak = GB/s / hw.mem_bw_gbs
- a plain copy kernel to measure the bandwidth actually achievable
"""
