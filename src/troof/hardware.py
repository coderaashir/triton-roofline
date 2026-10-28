"""GPU specs used by the roofline model.

The roofline code takes a `Hardware` object and never hardcodes numbers,
so the same model works for any accelerator: add an entry here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hardware:
    name: str
    mem_bw_gbs: float  # peak memory bandwidth from the spec sheet, GB/s (1 GB = 1e9 bytes)
    peak_tflops: dict[str, float] = field(default_factory=dict)  # dtype -> peak TFLOP/s

    def ridge_point(self, dtype: str = "fp32") -> float:
        """Arithmetic intensity (FLOPs/byte) where a kernel stops being memory-bound.

        Below this, memory bandwidth limits speed; above it, math throughput does.
        """
        return (self.peak_tflops[dtype] * 1e12) / (self.mem_bw_gbs * 1e9)


# NVIDIA T4 (Turing, sm_75), the free Colab GPU. Public spec sheet numbers.
# fp16 uses tensor cores. No bf16/fp8 support.
T4 = Hardware(
    name="NVIDIA T4",
    mem_bw_gbs=320.0,
    peak_tflops={"fp32": 8.1, "fp16": 65.0},
)

HARDWARE = {"T4": T4}


def detect() -> Hardware:
    """Return the spec for the GPU we're running on."""
    import torch

    gpu_name = torch.cuda.get_device_name()
    for key, hw in HARDWARE.items():
        if key in gpu_name:
            return hw
    raise ValueError(f"No spec for {gpu_name!r}; add it to HARDWARE in hardware.py")
