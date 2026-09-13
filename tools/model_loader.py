"""Unified model loader for vanilla Llama vs SelfExtend Llama.

The diagnostic_forward extraction and the kernel-retrieval identity hold
identically on both, but the model loading paths differ:

  - vanilla:    transformers.AutoModelForCausalLM
  - selfextend: models.selfExtend.llama2.LlamaForCausalLM + per-layer
                group_size / neighbor_size configuration

This helper hides that difference so the experiments scripts can take a
`--method` flag and otherwise be identical.
"""

from __future__ import annotations

import torch


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model(
    model_name: str,
    *,
    method: str = "vanilla",
    dtype: str = "bfloat16",
    device: str = "cuda",
    attn_implementation: str = "eager",
    group_size: int = 4,
    neighbor_size: int = 1024,
):
    """Load a Llama model in either vanilla or SelfExtend mode.

    Args:
        model_name: HF id or local path (e.g. "meta-llama/Llama-2-7b-hf").
        method: "vanilla" or "selfextend".
        dtype: one of "float32", "float16", "bfloat16".
        device: torch device string ("cuda", "cpu", "cuda:0", ...).
        attn_implementation: "eager", "sdpa", or "flash_attention_2".
            SE runs its own attention (models/selfExtend), which is eager;
            anything else under method="selfextend" is refused rather than
            silently coerced, so a driver's kernel setting cannot be
            ignored without a word (working rules 3.13).
        group_size: SE group_size (ignored for vanilla).
        neighbor_size: SE neighbor_size (ignored for vanilla).

    Returns:
        model: a .eval()-ed, device-placed model ready for forward passes.
               For SE, every layer's self_attn.{group_size, neighbor_size}
               has already been set.
    """
    if dtype not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {list(_DTYPES)}")
    torch_dtype = _DTYPES[dtype]

    if attn_implementation == "sdpa" and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        # torch's SDPA dispatcher may pick the cuDNN backend for bf16 on
        # Ampere/Hopper cards (torch >= 2.5). Its graphs fail on some shapes
        # -- "cuDNN Frontend error: No execution plans support the graph" in
        # the I2CL calibration's training forward, job 845166, RESULTS
        # 63.23 -- and its numerics are a third kernel beside flash and
        # memory-efficient, while the cell's cache gate is measured under
        # one. Pin the backend set to flash + memory-efficient (+ math).
        torch.backends.cuda.enable_cudnn_sdp(False)

    if method == "vanilla":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

    elif method == "selfextend":
        if attn_implementation != "eager":
            raise ValueError(
                f"SelfExtend runs its own eager attention (models/selfExtend); "
                f"attn_implementation={attn_implementation!r} would be ignored, so "
                "it is refused -- pass 'eager' (the driver's ATTN=eager)")
        from models.selfExtend.llama2 import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        # Per-layer SE configuration (matches test_unified.py:130-132).
        for layer in model.model.layers:
            layer.self_attn.group_size = group_size
            layer.self_attn.neighbor_size = neighbor_size

    else:
        raise ValueError(
            f"unknown method {method!r}; expected 'vanilla' or 'selfextend'"
        )

    model = model.to(device).eval()
    return model


def describe_model(model, method: str, **se_kwargs) -> str:
    """One-line description of the loaded model, for logging."""
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    hidden = cfg.hidden_size
    base = (f"method={method} layers={n_layers} heads={n_heads} kv={n_kv} "
            f"hidden={hidden}")
    if method == "selfextend":
        gs = se_kwargs.get("group_size", "?")
        ns = se_kwargs.get("neighbor_size", "?")
        base += f" group_size={gs} neighbor_size={ns}"
    return base
