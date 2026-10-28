"""Roofline model: arithmetic intensity -> predicted lower-bound time, plus plots.

TODO(Day 5):
- arithmetic_intensity(flops, bytes_moved) -> FLOPs per byte
- predicted_time(flops, bytes_moved, hw, dtype) -> max(compute time, memory time)
- plot_roofline(hw, points) -> log-log chart saved to figures/
"""
