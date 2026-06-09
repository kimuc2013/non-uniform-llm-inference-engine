"""Compute-time model: per-op FLOPs + bytes-moved, per-GPU efficiency.

We attach (`num_ops`, `tensor_size`) to each Chakra COMP_NODE; ASTRA-sim's
roofline engine ("roofline-enabled": 1) then computes per-op wall time as:

    time = max(num_ops / peak_perf, tensor_size / local_mem_bw)

Where `peak_perf` and `local_mem_bw` come from the per-GPU system JSON.
ASTRA-sim already does the roofline math, so all WE need to provide is the
op's FLOPs and bytes. Those depend only on shape, not GPU type — the GPU
type lives in the system config.

The single source of truth here is FLOPs + bytes for bf16 matmul.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeProfile:
    """Per-GPU performance constants used to populate the ASTRA-sim system
    JSON. Numbers are achievable bf16 TFLOPS + HBM bandwidth, not raw
    datasheet peak. The efficiency factor follows our calibration on real
    Blackwell/Ada measurements (~85% / ~82%)."""

    name: str
    spec_tflops_bf16: float    # raw datasheet
    spec_mem_bw_GBs: float     # raw datasheet
    achievable_tflops_frac: float = 0.85
    achievable_mem_bw_frac: float = 0.82

    @property
    def peak_perf_tflops(self) -> float:
        return self.spec_tflops_bf16 * self.achievable_tflops_frac

    @property
    def local_mem_bw_GBs(self) -> float:
        return self.spec_mem_bw_GBs * self.achievable_mem_bw_frac


# ---------- shape-level FLOPs / bytes formulas (bf16) ----------------------

DTYPE_BYTES = 2     # bf16


def matmul_flops_bytes(M: int, K: int, N: int) -> tuple[int, int]:
    """(num_ops, tensor_size) for a bf16 matmul of shape (M,K) @ (K,N).

    num_ops counts mac as 2. tensor_size = inputs + output bytes (the
    bytes ASTRA-sim's mem-bound roofline checks against local_mem_bw).
    """
    num_ops = 2 * M * K * N
    bytes_ = (M * K + K * N + M * N) * DTYPE_BYTES
    return num_ops, bytes_


def attention_flops_bytes(B: int, S_q: int, S_kv: int, n_heads: int,
                          head_dim: int) -> tuple[int, int]:
    """(num_ops, tensor_size) for one rank's attention compute.

    For decode (S_q=1, S_kv=kv_len): tiny compute, big KV-cache read.
    For prefill (S_q=S_kv=S): heavy compute O(B * h * S^2 * d).
    """
    # Scores: Q @ K^T of shape (B*h, S_q, head_dim) @ (B*h, head_dim, S_kv) -> (B*h, S_q, S_kv)
    flops_qk = 2 * B * n_heads * S_q * S_kv * head_dim
    # Softmax: cheap, ignored at this fidelity.
    # Out: scores @ V of shape (B*h, S_q, S_kv) @ (B*h, S_kv, head_dim) -> (B*h, S_q, head_dim)
    flops_av = 2 * B * n_heads * S_q * S_kv * head_dim
    num_ops = flops_qk + flops_av

    # Bytes: K, V cache reads dominate when S_kv >> S_q (decode).
    # Plus Q activation, output activation.
    bytes_q  = B * n_heads * S_q  * head_dim * DTYPE_BYTES
    bytes_kv = B * n_heads * S_kv * head_dim * DTYPE_BYTES * 2   # K + V
    bytes_o  = B * n_heads * S_q  * head_dim * DTYPE_BYTES
    bytes_ = bytes_q + bytes_kv + bytes_o
    return num_ops, bytes_


def layernorm_flops_bytes(B: int, S: int, hidden: int) -> tuple[int, int]:
    """Minimal LayerNorm/RMSNorm cost (just to keep nodes consistent)."""
    n_tokens = B * S
    num_ops = n_tokens * hidden * 8       # ~mean+var+scale, rough
    bytes_ = n_tokens * hidden * DTYPE_BYTES * 2   # read + write
    return num_ops, bytes_


def residual_add_flops_bytes(B: int, S: int, hidden: int) -> tuple[int, int]:
    n_tokens = B * S
    num_ops = n_tokens * hidden
    bytes_ = n_tokens * hidden * DTYPE_BYTES * 3   # 2 reads + 1 write
    return num_ops, bytes_


def silu_mul_flops_bytes(B: int, S: int, ffn: int) -> tuple[int, int]:
    """gate ⊙ silu(up) — pointwise."""
    n = B * S * ffn
    num_ops = 5 * n   # silu + multiply
    bytes_ = n * DTYPE_BYTES * 3
    return num_ops, bytes_


def compute_op_runtime(num_ops: int, tensor_size_bytes: int,
                       peak_tflops: float, mem_bw_GBs: float) -> float:
    """Manual roofline; useful for sanity checks. ASTRA-sim itself will
    apply the same formula when roofline-enabled=1."""
    t_flops = num_ops / (peak_tflops * 1e12)
    t_mem = tensor_size_bytes / (mem_bw_GBs * 1e9)
    return max(t_flops, t_mem)
