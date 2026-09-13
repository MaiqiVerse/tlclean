"""Section 3.2: per-query-head realized prototype write, as an attention wrapper.

THE PROBLEM. Under GQA there is no per-query-head value anywhere in the model.
`v_proj` emits one slot per KV head, and `repeat_kv` expands those slots inside
the attention function with no module boundary in between. So a `v_proj` hook
can only write a whole KV group (that is section 3.1, Method A-group), and an
`o_proj` hook writes the head's OUTPUT rather than its value. To give four
query heads that share a KV slot four different values, the write has to happen
after `repeat_kv` and before `attn_weights @ value_states`.

THE IMPLEMENTATION, AND WHY IT DOES NOT COPY THE EAGER BODY. Section 11 asks
for a narrow adapter aligned line by line with the installed Transformers eager
function, and warns against two shortcuts: patching only the answer row (misses
propagation) and adding a vector after `o_proj` (changes the bf16 reduction
order). A third hazard it does not name is the obvious way to satisfy it --
copying the eager body into this file -- which silently rots the day the
installed Transformers changes.

So nothing is copied. `repeat_kv` is applied here, the selected query-head label
rows are overwritten, and the ALREADY-EXPANDED key and value are handed to the
stock `eager_attention_forward` with `num_key_value_groups` temporarily set to
1. Stock `repeat_kv(x, 1)` returns `x` unchanged, so every remaining line --
the scaled matmul, the mask slice, the float32 softmax, dropout, the second
matmul, the transpose -- is the installed library's own code executing exactly
once. "Line-by-line aligned" becomes true by construction instead of by
inspection, and the empty-write case is bitwise identical to an unwrapped
forward because it performs the identical operations.

WHAT THIS BUYS AND WHAT IT COSTS. Because the overwrite precedes the attention
matmul, it acts on every destination row that can causally read the label row,
not just the answer row, and causally-masked entries stay zero on their own.
The cost is that `value_states` is materialised and cloned once per patched
layer, which is a few megabytes at these sequence lengths.

The write is silent-failure-prone in one specific way: if the attention
interface is not actually re-routed, every arm quietly degrades to natural and
the whole experiment reads as a null result. `RealizedWriteContext` therefore
counts adapter invocations and raises on exit if a layer that was supposed to
be patched never ran.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Torch is needed to APPLY a write, not to describe one. Importing it lazily
# keeps LayerWrite, group_of and group_members usable from torch-free code --
# the runner's write construction and its CPU fixture both build LayerWrite
# objects without ever touching a tensor.
try:
    import torch
except Exception as _e:                      # pragma: no cover - environment
    torch, _TORCH_ERR = None, f"{type(_e).__name__}: {_e}"
else:
    _TORCH_ERR = None

from tools.prototype_targets import (LayerWrite, group_members,  # noqa: F401,E402
                                     group_of)

ADAPTER_NAME = "eager_realized_write"


def _need_torch():
    if torch is None:
        raise RuntimeError(
            f"torch is required to apply a realized write ({_TORCH_ERR}). "
            "Describing one -- LayerWrite, group_of, group_members -- does not "
            "need it, which is why the import is lazy.")


def realized_value_states(value, n_rep, write, repeat_kv_fn):
    """post-`repeat_kv` value tensor with the selected query-head rows written.

    Split out from the attention wrapper because ALL of the head/group index
    arithmetic lives here and none of it needs a model, a config, or
    Transformers -- so section 11's group semantics can be tested directly.

    Returns the expanded tensor unmodified when there is nothing to write, so
    the sham arm travels the identical code path.
    """
    _need_torch()
    v = repeat_kv_fn(value, n_rep)
    if write is None or not write.values:
        return v
    b, n_q, n_tok, d_head = v.shape
    if b != 1:
        raise ValueError(f"realized write assumes one prompt per forward; got "
                         f"batch {b}. Batching would make the frozen label-row "
                         f"positions ambiguous across prompts.")
    rows = torch.as_tensor(write.rows, dtype=torch.long, device=v.device).reshape(-1)
    if rows.numel() and (rows.min() < 0 or rows.max() >= n_tok):
        raise ValueError(f"label rows {rows.tolist()} outside the sequence "
                         f"(length {n_tok})")
    if rows.numel() != len(set(rows.tolist())):
        raise ValueError(f"duplicate label rows {rows.tolist()}: two demos "
                         "cannot share a label token position")
    bad = [h for h in write.heads() if not 0 <= h < n_q]
    if bad:
        raise ValueError(f"query head index out of range: {bad} (model has "
                         f"{n_q} query heads)")
    v = v.clone()
    for h, vals in write.values.items():
        vals = torch.as_tensor(vals)
        if vals.shape != (rows.numel(), d_head):
            raise ValueError(
                f"head {h}: values have shape {tuple(vals.shape)}, expected "
                f"{(rows.numel(), d_head)} (one d_head vector per label row)")
        v[0, int(h), rows, :] = vals.to(device=v.device, dtype=v.dtype)
    return v


def _stock():
    """The installed eager attention and repeat_kv. Imported at call time so
    this module stays importable on a machine without Transformers (the arm
    arithmetic and the value-construction fixture do not need it)."""
    try:
        from transformers.models.llama.modeling_llama import (
            eager_attention_forward, repeat_kv)
    except Exception as e:                    # report the real cause, rule 6
        raise RuntimeError(
            f"cannot import the installed Llama eager attention "
            f"({type(e).__name__}: {e}). The realized-write adapter wraps the "
            "library's own function on purpose and has no fallback copy of it; "
            "run this where Transformers is installed.") from e
    return eager_attention_forward, repeat_kv


def realized_write_attention(module, query, key, value, attention_mask,
                             scaling, dropout=0.0, **kwargs):
    """Attention interface: stock eager, with the value overwrite spliced in."""
    eager, repeat_kv = _stock()
    write = getattr(module, "_realized_write", None)
    n_rep = int(module.num_key_value_groups)
    key_states = repeat_kv(key, n_rep)
    value_states = realized_value_states(value, n_rep, write, repeat_kv)

    module._realized_write_calls = getattr(module, "_realized_write_calls", 0) + 1
    saved = module.num_key_value_groups
    module.num_key_value_groups = 1     # K and V are already expanded
    try:
        return eager(module, query, key_states, value_states, attention_mask,
                     scaling, dropout=dropout, **kwargs)
    finally:
        module.num_key_value_groups = saved


def register_adapter():
    """Make the adapter reachable by name, preferring the public interface."""
    try:
        from transformers.modeling_utils import AttentionInterface
        AttentionInterface.register(ADAPTER_NAME, realized_write_attention)
        return "AttentionInterface.register"
    except Exception:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS[ADAPTER_NAME] = realized_write_attention
        return "ALL_ATTENTION_FUNCTIONS"


def verify_adapter_ran(expect, calls, how=""):
    """Raise unless every layer that was supposed to be patched actually ran.

    Split out of the context manager so it can be tested without Transformers,
    and because it is the single check standing between a failed re-route and a
    result that looks exactly like a true null.
    """
    missed = sorted(i for i in expect if calls.get(i, 0) == 0)
    if missed:
        raise RuntimeError(
            f"the realized-write adapter never ran on layers {missed}"
            + (f" (registered via {how})" if how else "") +
            ". The forward completed, so every arm would have silently "
            "returned natural values. Check that the attention interface "
            "lookup honours config._attn_implementation="
            f"{ADAPTER_NAME!r} in this Transformers version.")


class RealizedWriteContext:
    """Route a model's attention through the adapter for the duration.

    `writes` maps layer index -> LayerWrite. Layers not named are still routed
    through the adapter (so every layer runs the same code path and the
    comparison is not confounded by two different attention implementations),
    but write nothing.

    On exit it verifies the adapter actually ran on every patched layer. Without
    that check a failed re-route is indistinguishable from a true null: the
    forward succeeds, the logits are natural, and every arm reports no effect.
    """

    def __init__(self, model, writes, *, expect_layers=None):
        self.model = model
        self.writes = dict(writes or {})
        self.layers = model.model.layers
        self.expect = (set(self.writes) if expect_layers is None
                       else set(expect_layers))
        self._saved_impl = None
        self._how = None

    def __enter__(self):
        self._how = register_adapter()
        cfg = self.model.config
        self._saved_impl = cfg._attn_implementation
        if self._saved_impl != "eager":
            raise RuntimeError(
                f"the model is running {self._saved_impl!r} attention, but the "
                "realized write is only defined against the eager path "
                "(section 2.2 fixes vanilla/bf16/eager). Load with "
                "attn_implementation='eager'.")
        cfg._attn_implementation = ADAPTER_NAME
        for i, layer in enumerate(self.layers):
            layer.self_attn._realized_write = self.writes.get(i)
            layer.self_attn._realized_write_calls = 0
        return self

    def __exit__(self, *exc):
        cfg = self.model.config
        cfg._attn_implementation = self._saved_impl
        calls = {i: getattr(layer.self_attn, "_realized_write_calls", 0)
                 for i, layer in enumerate(self.layers)}
        for layer in self.layers:
            layer.self_attn._realized_write = None
        if exc[0] is None:
            verify_adapter_ran(self.expect, calls, self._how)
        return False

    def call_counts(self):
        return {i: getattr(layer.self_attn, "_realized_write_calls", 0)
                for i, layer in enumerate(self.layers)}
