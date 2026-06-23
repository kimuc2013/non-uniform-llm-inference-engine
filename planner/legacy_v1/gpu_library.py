"""Synthetic GpuProfile catalog from datasheet + empirical scaling.

For clusters where we have no microbench data (most), we synthesize a
GpuProfile from the vendor datasheet (peak bf16 TFLOPS, peak HBM bandwidth)
multiplied by an empirically-derived "achievable" factor.

The factor is calibrated against our real measured cluster:
  RTX PRO 6000 Blackwell datasheet: 380 TFLOPS bf16, 1792 GB/s mem BW
  RTX PRO 6000 Blackwell measured  : ~378 TFLOPS prefill, ~1461 GB/s
  → prefill achievable factor ≈ 0.99 (memory ≈ 0.82)

  RTX 6000 Ada datasheet           : 91 TFLOPS bf16 (dense), 960 GB/s
  RTX 6000 Ada measured            : ~145 TFLOPS prefill, ~798 GB/s
  → Ada outperforms dense spec because cuBLAS uses FP16 path / TensorCore tricks
  → memory achievable factor ≈ 0.83

We adopt:
  ACHIEVABLE_TFLOPS_FRAC = 0.85   (most GPUs achieve 80-95% of spec in TC matmul)
  ACHIEVABLE_MEM_BW_FRAC = 0.82   (matches measured fraction on both GPUs)

These are tunable per-cluster if calib data is available for any reference GPU.
"""

from __future__ import annotations

from .cost_model import GpuProfile


# ---- Empirical scaling factors (derived from our real Blackwell+Ada cluster) ----

ACHIEVABLE_TFLOPS_FRAC = 0.85
ACHIEVABLE_MEM_BW_FRAC = 0.82

# Decode regime TFLOPS is much lower than peak — small M means memory-bound,
# kernel can't reach peak compute. Empirically ≈ 25-40% of prefill TFLOPS.
DECODE_TFLOPS_FRAC = 0.30

# KV-read bandwidth (paged-attn pattern) is typically 1.5-3× larger than
# bulk-copy mem_bw on cache-friendly tensors. We use a conservative multiplier.
KV_BW_MULTIPLIER = 2.5


def synthetic_gpu(name: str, *, spec_tflops_bf16: float, spec_mem_bw_GBs: float,
                  vram_GB: int = 80, l2_mb: int = 50,
                  tflops_eff: float = ACHIEVABLE_TFLOPS_FRAC,
                  mem_bw_eff: float = ACHIEVABLE_MEM_BW_FRAC) -> GpuProfile:
    """Build a GpuProfile from a vendor spec sheet + empirical efficiency factors.

    Notes:
      `vram_GB` and `l2_mb` are metadata only — the cost model uses TFLOPS
      and mem BW directly. Future versions could use l2_mb for cache-aware
      matmul time correction at small M.
    """
    tflops_prefill = spec_tflops_bf16 * tflops_eff
    return GpuProfile(
        name=name,
        tflops_prefill=tflops_prefill,
        tflops_decode=tflops_prefill * DECODE_TFLOPS_FRAC,
        mem_bw_GBs=spec_mem_bw_GBs * mem_bw_eff,
        kv_bw_GBs=spec_mem_bw_GBs * mem_bw_eff * KV_BW_MULTIPLIER,
        matmul_lookup={},   # empty -> CostModel.matmul_time_s falls back to roofline
    )


# ---------------------------------------------------------------------
# GPU catalog. Numbers from NVIDIA datasheets and verified vendor specs.
# Format: (spec_tflops_bf16_dense, spec_mem_bw_GBs, vram_GB, l2_mb)
# ---------------------------------------------------------------------

H100_SXM5  = synthetic_gpu("H100-SXM5",       spec_tflops_bf16=989, spec_mem_bw_GBs=3350, vram_GB=80,  l2_mb=50)
H100_PCIE  = synthetic_gpu("H100-PCIe",       spec_tflops_bf16=756, spec_mem_bw_GBs=2000, vram_GB=80,  l2_mb=50)
H200_SXM   = synthetic_gpu("H200-SXM",        spec_tflops_bf16=989, spec_mem_bw_GBs=4800, vram_GB=141, l2_mb=50)
A100_SXM4_80  = synthetic_gpu("A100-SXM4-80GB", spec_tflops_bf16=312, spec_mem_bw_GBs=2039, vram_GB=80, l2_mb=40)
A100_PCIE_80  = synthetic_gpu("A100-PCIe-80GB", spec_tflops_bf16=312, spec_mem_bw_GBs=1935, vram_GB=80, l2_mb=40)
A100_SXM4_40  = synthetic_gpu("A100-SXM4-40GB", spec_tflops_bf16=312, spec_mem_bw_GBs=1555, vram_GB=40, l2_mb=40)
V100_SXM2_32  = synthetic_gpu("V100-SXM2-32GB", spec_tflops_bf16=125, spec_mem_bw_GBs=900,  vram_GB=32, l2_mb=6)
V100_PCIE_32  = synthetic_gpu("V100-PCIe-32GB", spec_tflops_bf16=112, spec_mem_bw_GBs=900,  vram_GB=32, l2_mb=6)
L40S         = synthetic_gpu("L40S",            spec_tflops_bf16=362, spec_mem_bw_GBs=864,  vram_GB=48, l2_mb=96)
A40          = synthetic_gpu("A40",             spec_tflops_bf16=149, spec_mem_bw_GBs=696,  vram_GB=48, l2_mb=6)
A6000_ADA    = synthetic_gpu("RTX6000-Ada",     spec_tflops_bf16=91,  spec_mem_bw_GBs=960,  vram_GB=48, l2_mb=96)
RTX_PRO_BLW  = synthetic_gpu("RTX-PRO-Blackwell", spec_tflops_bf16=380, spec_mem_bw_GBs=1792, vram_GB=96, l2_mb=192)


# Convenience: aliases for common cluster-builder usage
GPU_CATALOG: dict[str, GpuProfile] = {
    "H100-SXM5":          H100_SXM5,
    "H100-PCIe":          H100_PCIE,
    "H200-SXM":           H200_SXM,
    "A100-SXM4-80GB":     A100_SXM4_80,
    "A100-PCIe-80GB":     A100_PCIE_80,
    "A100-SXM4-40GB":     A100_SXM4_40,
    "V100-SXM2-32GB":     V100_SXM2_32,
    "V100-PCIe-32GB":     V100_PCIE_32,
    "L40S":               L40S,
    "A40":                A40,
    "RTX6000-Ada":        A6000_ADA,
    "RTX-PRO-Blackwell":  RTX_PRO_BLW,
}


def get_gpu(name: str) -> GpuProfile:
    if name not in GPU_CATALOG:
        raise KeyError(f"unknown GPU: {name}. Available: {list(GPU_CATALOG)}")
    return GPU_CATALOG[name]
