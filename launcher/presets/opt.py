"""OPT preset: opt30b. Non-uniform TP patch will live on opt.py
(dense MHA + ReLU FFN, learned positional embeddings).

NOTE: OPT-30B has max_position_embeddings=2050; do not exceed that or
loading will fail. Old runs on opt30b were blocked at in=2048 because
the prompt length plus generation exceeded 2050.
"""
from __future__ import annotations


# facebook/opt-30b:
#   num_hidden_layers   = 48
#   hidden_size         = 7168
#   num_attention_heads = 56
#   head_dim            = 128
#   intermediate_size   = 28672 (4 * hidden_size)
#   vocab_size          = 50272
#   NOTE: dense MHA (kv_heads == q_heads), not GQA.
def apply_opt30b_preset(env: dict[str, str]) -> None:
    env.setdefault("MODEL", "facebook/opt-30b")
    env.setdefault("MODEL_FAMILY", "opt")
    env.setdefault("MODEL_SIZE_LABEL", "30b")
    env.setdefault("NUM_LAYERS", "48")
    env.setdefault("HIDDEN_SIZE", "7168")
    env.setdefault("NUM_Q_HEADS", "56")
    env.setdefault("NUM_KV_HEADS", "56")     # dense MHA
    env.setdefault("HEAD_DIM", "128")
    env.setdefault("INTERMEDIATE_SIZE", "28672")
    env.setdefault("VOCAB_SIZE", "50272")

    env.setdefault("TP", "8")
    env.setdefault("PP", "1")
    env.setdefault("EXECUTOR_BACKEND", "ray")

    # Uniform baseline: 56/8 = 7 Q heads = 7 KV heads, 28672/8 = 3584 FFN
    env.setdefault("TP_HEAD_SPLITS", "7,7,7,7,7,7,7,7")
    env.setdefault("TP_FFN_SPLITS", "3584,3584,3584,3584,3584,3584,3584,3584")
    env.setdefault("TP_KV_SPLITS", "7,7,7,7,7,7,7,7")

    # OPT-30B max_position_embeddings = 2050, must not exceed.
    env.setdefault("MAX_MODEL_LEN", "2048")
    env.setdefault("MAX_NUM_BATCHED_TOKENS", "2048")
