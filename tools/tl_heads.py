"""TL head registry for the kernel-retrieval framework.

Per `method_kernel_retrieval.md` §2: T = set of (layer, head) tuples that
TSLA (Yang/Cho/Inoue 2025, arXiv:2509.24164) identifies as TL heads. Theorem 1's
identity holds for the aggregate Delta_TL summed over T.

This module is a thin **config layer**: load / save a TLHeadSet and ship a
small placeholder default. Identifying heads from scratch (re-running TSLA's
TL-score procedure) is OUT OF SCOPE here; populate from the published paper
or from a fitted set produced by `experiments/exp_identity_verification.py`.

Config file format (JSON):
    {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "source": "TSLA paper (Yang/Cho/Inoue 2025, Table X)",
        "heads": [[layer, head], [layer, head], ...]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class TLHeadSet:
    """T = set of (layer, head) tuples identified as TL heads."""
    model_name: str
    heads: set[tuple[int, int]]
    source: str = ""

    def __post_init__(self) -> None:
        # Coerce list-of-list (JSON) -> set-of-tuple; normalize types.
        self.heads = {tuple(int(x) for x in lk) for lk in self.heads}
        for lk in self.heads:
            if len(lk) != 2:
                raise ValueError(f"expected (layer, head) pair, got {lk}")

    def layers(self) -> set[int]:
        """Distinct layers touched by this head set."""
        return {l for (l, _) in self.heads}

    def heads_at(self, layer: int) -> set[int]:
        """Head indices within a given layer."""
        return {k for (l, k) in self.heads if l == layer}

    def __len__(self) -> int:
        return len(self.heads)

    def __iter__(self):
        return iter(sorted(self.heads))

    def __contains__(self, key) -> bool:
        return tuple(key) in self.heads


# ---------------------------------------------------------------------------
# Registry of known TLHeadSets
# ---------------------------------------------------------------------------

# PLACEHOLDER values. Replace with TSLA-published heads or fitted output before
# claiming Result 1. The actual TL heads for Llama-2-7B are reported in TSLA
# (Yang/Cho/Inoue 2025) Tables / supplementary; we have not transcribed them
# yet. Until then, these placeholders let us smoke-test the pipeline.

_PLACEHOLDER_LLAMA2_7B_TL_HEADS: set[tuple[int, int]] = {
    # NOTE: PLACEHOLDER. Do not use for paper results. Pulled to a small,
    # mid-layer-cluster pattern matching the doc's TL-head intuition
    # (mid-late layers, multiple heads concentrated).
    (15, 17),
    (15, 23),
    (16, 5),
    (16, 26),
    (17, 12),
}

_REGISTRY: dict[str, TLHeadSet] = {
    "meta-llama/Llama-2-7b-hf": TLHeadSet(
        model_name="meta-llama/Llama-2-7b-hf",
        heads=set(_PLACEHOLDER_LLAMA2_7B_TL_HEADS),
        source="PLACEHOLDER -- replace with TSLA published heads before "
               "Result 1.",
    ),
}


def load_tsla_heads(
    model_name: str,
    config_path: Optional[str | Path] = None,
) -> TLHeadSet:
    """Load the TLHeadSet for a model.

    Args:
        model_name: HF model id (e.g. 'meta-llama/Llama-2-7b-hf').
        config_path: Optional JSON file path. If provided, overrides the
                     registry default.

    Returns:
        TLHeadSet. Raises FileNotFoundError if config_path is provided but
        doesn't exist, or KeyError if model_name is not in the registry and
        no config_path is given.
    """
    if config_path is not None:
        return load_heads(config_path)
    if model_name in _REGISTRY:
        return _REGISTRY[model_name]
    raise KeyError(
        f"No TL heads registered for {model_name!r}. Pass `config_path` to a "
        f"JSON file, or extend tools.tl_heads._REGISTRY."
    )


def load_heads(path: str | Path) -> TLHeadSet:
    """Load a TLHeadSet from a JSON config file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return TLHeadSet(
        model_name=cfg["model_name"],
        heads=set(tuple(lk) for lk in cfg["heads"]),
        source=cfg.get("source", ""),
    )


def save_heads(head_set: TLHeadSet, path: str | Path) -> None:
    """Save a TLHeadSet to JSON."""
    path = Path(path)
    cfg = {
        "model_name": head_set.model_name,
        "source": head_set.source,
        "heads": sorted([list(lk) for lk in head_set.heads]),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def register(head_set: TLHeadSet) -> None:
    """Register a TLHeadSet in the in-process registry under its model_name."""
    _REGISTRY[head_set.model_name] = head_set


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # Roundtrip via JSON
    original = load_tsla_heads("meta-llama/Llama-2-7b-hf")
    print(f"Default registry entry: {len(original)} heads, source={original.source!r}")
    print(f"  Layers touched: {sorted(original.layers())}")
    print(f"  Sample iter:    {list(original)[:3]}")

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        tmp_path = tf.name
    try:
        save_heads(original, tmp_path)
        reloaded = load_heads(tmp_path)
        assert reloaded.heads == original.heads
        assert reloaded.model_name == original.model_name
        print(f"Roundtrip OK: {len(reloaded)} heads loaded from {tmp_path}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Membership / lookup
    sample = next(iter(original.heads))
    assert sample in original
    assert original.heads_at(sample[0]) >= {sample[1]}
    print("Membership and per-layer lookup OK")
