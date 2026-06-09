"""Qwen3 preset: qwen3_32b. Non-uniform TP patch lives on qwen3.py
(qwen2 is intentionally left stock per user decision 2026-05-28).
"""
from __future__ import annotations


# Qwen3-32B:
#   num_hidden_layers   = 64
#   hidden_size         = 5120
#   num_attention_heads = 64
#   num_kv_heads        = 8
#   head_dim            = 128
#   intermediate_size   = 25600
#   vocab_size          = 151936
def apply_qwen3_32b_preset(env: dict[str, str]) -> None:
    env.setdefault("MODEL", "Qwen/Qwen3-32B")
    env.setdefault("MODEL_FAMILY", "qwen3")
    env.setdefault("MODEL_SIZE_LABEL", "32b")
    env.setdefault("NUM_LAYERS", "64")
    env.setdefault("HIDDEN_SIZE", "5120")
    env.setdefault("NUM_Q_HEADS", "64")
    env.setdefault("NUM_KV_HEADS", "8")
    env.setdefault("HEAD_DIM", "128")
    env.setdefault("INTERMEDIATE_SIZE", "25600")
    env.setdefault("VOCAB_SIZE", "151936")

    env.setdefault("TP", "8")
    env.setdefault("PP", "1")
    env.setdefault("EXECUTOR_BACKEND", "ray")

    # Uniform baseline: 64/8 = 8, 8/8 = 1, 25600/8 = 3200
    env.setdefault("TP_HEAD_SPLITS", "8,8,8,8,8,8,8,8")
    env.setdefault("TP_FFN_SPLITS", "3200,3200,3200,3200,3200,3200,3200,3200")
    env.setdefault("TP_KV_SPLITS", "1,1,1,1,1,1,1,1")

    env.setdefault("MAX_MODEL_LEN", "4096")
    env.setdefault("MAX_NUM_BATCHED_TOKENS", "2048")
