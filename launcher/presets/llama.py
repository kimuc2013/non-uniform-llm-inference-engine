"""LLaMA presets: llama8b (single-node 4-GPU) and llama70b (multi-node 8-GPU).

Each preset sets architectural facts (NUM_Q_HEADS, INTERMEDIATE_SIZE, etc.)
and a *uniform-baseline* TP_*_SPLITS string for the standard TP size.
The planner (M5) can replace these with non-uniform splits.
"""
from __future__ import annotations


# Llama-3.1-8B-Instruct:
#   num_hidden_layers   = 32
#   hidden_size         = 4096
#   num_attention_heads = 32
#   num_kv_heads        = 8
#   head_dim            = 128
#   intermediate_size   = 14336
#   vocab_size          = 128256
def apply_llama8b_preset(env: dict[str, str]) -> None:
    env.setdefault("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    env.setdefault("MODEL_FAMILY", "llama")
    env.setdefault("MODEL_SIZE_LABEL", "8b")
    env.setdefault("NUM_LAYERS", "32")
    env.setdefault("HIDDEN_SIZE", "4096")
    env.setdefault("NUM_Q_HEADS", "32")
    env.setdefault("NUM_KV_HEADS", "8")
    env.setdefault("HEAD_DIM", "128")
    env.setdefault("INTERMEDIATE_SIZE", "14336")
    env.setdefault("VOCAB_SIZE", "128256")

    env.setdefault("TP", "4")
    env.setdefault("PP", "1")
    env.setdefault("SINGLE_NODE_HINT", "1")
    env.setdefault("EXECUTOR_BACKEND", "mp")

    # Uniform baseline: 32/4 = 8 Q heads, 8/4 = 2 KV heads, 14336/4 = 3584 FFN
    env.setdefault("TP_HEAD_SPLITS", "8,8,8,8")
    env.setdefault("TP_FFN_SPLITS", "3584,3584,3584,3584")
    env.setdefault("TP_KV_SPLITS", "2,2,2,2")

    env.setdefault("MAX_MODEL_LEN", "8192")
    env.setdefault("MAX_NUM_BATCHED_TOKENS", "2048")


# Llama-3.3-70B-Instruct:
#   num_hidden_layers   = 80
#   hidden_size         = 8192
#   num_attention_heads = 64
#   num_kv_heads        = 8
#   head_dim            = 128
#   intermediate_size   = 28672
#   vocab_size          = 128256
def apply_llama70b_preset(env: dict[str, str]) -> None:
    env.setdefault("MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    env.setdefault("MODEL_FAMILY", "llama")
    env.setdefault("MODEL_SIZE_LABEL", "70b")
    env.setdefault("NUM_LAYERS", "80")
    env.setdefault("HIDDEN_SIZE", "8192")
    env.setdefault("NUM_Q_HEADS", "64")
    env.setdefault("NUM_KV_HEADS", "8")
    env.setdefault("HEAD_DIM", "128")
    env.setdefault("INTERMEDIATE_SIZE", "28672")
    env.setdefault("VOCAB_SIZE", "128256")

    env.setdefault("TP", "8")
    env.setdefault("PP", "1")
    env.setdefault("EXECUTOR_BACKEND", "ray")

    # Uniform baseline: 64/8 = 8 Q heads, 8/8 = 1 KV head, 28672/8 = 3584 FFN
    env.setdefault("TP_HEAD_SPLITS", "8,8,8,8,8,8,8,8")
    env.setdefault("TP_FFN_SPLITS", "3584,3584,3584,3584,3584,3584,3584,3584")
    env.setdefault("TP_KV_SPLITS", "1,1,1,1,1,1,1,1")

    env.setdefault("MAX_MODEL_LEN", "16384")
    env.setdefault("MAX_NUM_BATCHED_TOKENS", "2048")
