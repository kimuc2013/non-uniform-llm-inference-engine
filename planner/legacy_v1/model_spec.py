"""Model architecture specs needed by the cost model.

Keeps just the dimensions that matter for compute/comm prediction. Constants
are taken from HuggingFace configs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int

    @property
    def gqa_group(self) -> int:
        """num_q_heads / num_kv_heads — head_splits must be a multiple of this."""
        assert self.num_q_heads % self.num_kv_heads == 0
        return self.num_q_heads // self.num_kv_heads

    @property
    def q_proj_dim(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def qkv_combined_dim(self) -> int:
        return self.q_proj_dim + 2 * self.kv_proj_dim


llama_3_3_70b = ModelSpec(
    name="meta-llama/Llama-3.3-70B-Instruct",
    num_layers=80,
    hidden_size=8192,
    num_q_heads=64,
    num_kv_heads=8,
    head_dim=128,
    intermediate_size=28672,
    vocab_size=128256,
)

llama_3_1_8b = ModelSpec(
    name="meta-llama/Llama-3.1-8B-Instruct",
    num_layers=32,
    hidden_size=4096,
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
    intermediate_size=14336,
    vocab_size=128256,
)

qwen_3_32b = ModelSpec(
    name="Qwen/Qwen3-32B",
    num_layers=64,
    hidden_size=5120,
    num_q_heads=64,
    num_kv_heads=8,
    head_dim=128,
    intermediate_size=25600,
    vocab_size=151936,
)

opt_30b = ModelSpec(
    name="facebook/opt-30b",
    num_layers=48,
    hidden_size=7168,
    num_q_heads=56,
    num_kv_heads=56,        # dense MHA, not GQA
    head_dim=128,
    intermediate_size=28672,
    vocab_size=50272,
)

# Registry by alias used elsewhere in the launcher/experiments.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "llama8b":   llama_3_1_8b,
    "llama70b":  llama_3_3_70b,
    "qwen3_32b": qwen_3_32b,
    "opt30b":    opt_30b,
}


def get_model(alias: str) -> ModelSpec:
    if alias not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model alias {alias!r}. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[alias]
