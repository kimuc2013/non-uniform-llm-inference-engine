"""Preset registry. Each preset is a callable (env_dict) -> None that
applies setdefault() values to the env dict.

Preset functions only set env vars; cluster topology is layered on top
by config.resolve_launch_config() so that presets are cluster-agnostic.
"""
from __future__ import annotations

from collections.abc import Callable

from .common import apply_common_defaults
from .llama import apply_llama8b_preset, apply_llama70b_preset
from .qwen import apply_qwen3_32b_preset
from .opt import apply_opt30b_preset

PresetFn = Callable[[dict], None]

REGISTRY: dict[str, PresetFn] = {
    "llama8b": apply_llama8b_preset,
    "llama70b": apply_llama70b_preset,
    "qwen3_32b": apply_qwen3_32b_preset,
    "opt30b": apply_opt30b_preset,
}

ALL_TARGETS = tuple(REGISTRY.keys())


def get_preset(name: str) -> PresetFn:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown target {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


__all__ = ["REGISTRY", "ALL_TARGETS", "get_preset", "apply_common_defaults"]
