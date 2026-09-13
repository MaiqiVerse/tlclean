"""Answer-row attention, captured through the attention interface. GPU.

WHY NOT `output_attentions=True`. It keeps the whole [B, H, N, N] matrix for
every layer. At the K=5 prompt length (N ~ 3038, H = 32) that is ~1.2 GB per
layer in float32 and ~38 GB over the model -- not a tuning problem, a wall.

WHY NOT A `self_attn.forward` MONKEY-PATCH. `probe_prototype_shrinkage`'s
runner says in as many words that it avoids one because it would have to
coexist with the realized-write adapter's re-routing of the same module, and
"two interacting interventions on the same module, one of them unnecessary,
is not worth the risk".

WHAT THIS DOES INSTEAD. It registers a named attention function, exactly the
way `realized_write` does, that calls STOCK EAGER and keeps only the answer
row -- `attn_weights[0, :, -1, :]`, shape [heads, tokens]. Eager already
materialises the full matrix internally, so keeping one row of it costs
nothing extra; what it avoids is the matrix being RETAINED.

AND IT REFUSES TO COEXIST WITH A WRITE. The two adapters both take over
`config._attn_implementation`, and the donor this feeds is chosen from the
NATURAL forward anyway, so they never need to run together. That is enforced
rather than documented: `CaptureContext` raises if a realized write is
active. This is the docstring's own worry, made mechanical.

⚠ THE ANSWER ROW IS THE LAST QUERY POSITION. Section 4.1 already uses "demo
label rows 的平均 answer attention" for matched-head selection, so this is
that same quantity and not a new convention.
"""

from __future__ import annotations

import numpy as np

ADAPTER_NAME = "answer_attention_capture"


def _stock():
    """The stock eager attention function, by whatever route exists."""
    from transformers.models.llama.modeling_llama import \
        eager_attention_forward
    return eager_attention_forward


# Unit round-off (round-to-nearest relative error) by significand width.
# bf16 keeps 8 significand bits, fp16 11, fp32 24, fp64 53.
DTYPE_ROUNDOFF = {"torch.bfloat16": 2.0 ** -8, "torch.float16": 2.0 ** -11,
                  "torch.float32": 2.0 ** -24, "torch.float64": 2.0 ** -53}

# The float32 softmax residue and the float64 summation, both orders below the
# bf16 term but named rather than absorbed.
_SUM_SLACK = 1e-6


def roundoff_for(dtype):
    """Unit round-off for the dtype the weights were STORED in.

    Not the dtype they were computed in. Llama's eager attention runs the
    softmax in float32 and then casts back:

        attn_weights = softmax(..., dtype=torch.float32).to(query.dtype)

    so on a bf16 model the row that exists afterwards is bf16, and its entries
    carry bf16 rounding however the softmax was computed.
    """
    key = str(dtype)
    if key not in DTYPE_ROUNDOFF:
        raise ValueError(f"no unit round-off registered for {key!r}; add it "
                         f"rather than guessing. Known: {sorted(DTYPE_ROUNDOFF)}")
    return DTYPE_ROUNDOFF[key]


def answer_row_faults(row, *, roundoff):
    """Why `row` is not a softmax row over key positions. Empty means it is.

    THE CHECK THAT CATCHES THE WRONG TENSOR. Attention weights sum to 1 along
    the key axis and are non-negative; logits, pre-softmax scores and value
    states satisfy neither. A capture that silently grabbed the wrong operand
    would otherwise look like a plausible array of numbers and pick donors
    from it.

    THE TOLERANCE IS DERIVED, and `roundoff` is REQUIRED so that no caller can
    supply a guess by omission. A softmax row sums to 1 before storage; every
    entry is then rounded to the storage dtype, so

        |sum - 1|  <=  sum_i |w_i - w^_i|  <=  u * sum_i w_i  =  u

    with u the unit round-off. That is a HARD bound, not an estimate.

    ⚠ THIS DEFAULTED TO 1e-3 AND FIRED ON A CORRECT CAPTURE (job 838469, all
    five carrier layers). On bf16 u = 2^-8 = 3.906e-3, and the observed
    deviations were 1.600e-3 to 2.898e-3 -- every one inside the bound, at
    0.41 to 0.74 of it, spread below it the way partial cancellation of
    independent roundings looks. 1e-3 was 0.26 of the bound: tighter than the
    arithmetic permits, which is working rules 11b, and the docstring calling it
    "float slack and not a fitted threshold" was wrong twice -- it was neither.
    """
    a = np.asarray(row, dtype=np.float64)
    bad = []
    if a.ndim != 2:
        return [f"answer row is {a.ndim}-D, expected [heads, tokens]"]
    if not np.all(np.isfinite(a)):
        bad.append(f"{int((~np.isfinite(a)).sum())} non-finite entries")
        return bad
    if a.min() < 0:
        bad.append(f"minimum {a.min():.3e} is negative; softmax weights "
                   "cannot be, so this is not the attention matrix")
    tol = float(roundoff) + _SUM_SLACK
    s = a.sum(axis=1)
    off = np.abs(s - 1.0).max()
    if off > tol:
        bad.append(f"rows sum to between {s.min():.6f} and {s.max():.6f}; the "
                   f"worst is off by {off:.3e}, which is {off / tol:.2f}x the "
                   f"bound {tol:.3e} that the storage dtype's unit round-off "
                   f"{roundoff:.3e} permits. A softmax row cannot exceed it, "
                   "so this is not one")
    return bad


class CaptureContext:
    """Route attention through the capture adapter for the duration.

    `store` is filled as {layer_index: [heads, tokens] float32} for the layers
    named in `layers`; other layers run the adapter too (the switch is
    config-wide) but keep nothing.
    """

    def __init__(self, model, layers, store):
        self.model = model
        self.layers = sorted(int(l) for l in layers)
        self.store = store
        self._saved = None
        self._dtypes = []

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        cfg = self.model.config
        if getattr(cfg, "_attn_implementation", None) not in ("eager", None):
            raise RuntimeError(
                f"attention implementation is "
                f"{cfg._attn_implementation!r}, not 'eager'. This adapter "
                "wraps stock eager; entering from another adapter -- the "
                "realized write, say -- would nest two re-routes of the same "
                "interface, which is the coexistence this file exists to "
                "avoid.")
        for l in self.layers:
            mod = self.model.model.layers[l].self_attn
            if getattr(mod, "_realized_write", None) is not None:
                raise RuntimeError(
                    f"layer {l} still carries a realized write. The donor is "
                    "chosen from the NATURAL forward, so capture and write "
                    "never need to run together.")
        want = set(self.layers)
        store = self.store
        eager = _stock()
        self._dtypes = self_dtype = []

        def capture(module, query, key, value, attention_mask, scaling,
                    dropout=0.0, **kwargs):
            out, weights = eager(module, query, key, value, attention_mask,
                                 scaling, dropout=dropout, **kwargs)
            idx = getattr(module, "layer_idx", None)
            if idx in want and weights is not None:
                # ONE ROW, then straight to CPU. Keeping `weights` itself
                # would reintroduce the [H, N, N] retention this avoids.
                # THE STORAGE DTYPE IS RECORDED BEFORE THE CAST. `.float()`
                # is what makes the numbers convenient; it also erases the
                # precision the check has to be derived from.
                self_dtype.append(weights.dtype)
                store[int(idx)] = (weights[0, :, -1, :]
                                   .detach().float().cpu().numpy())
                module._answer_attention_calls = getattr(
                    module, "_answer_attention_calls", 0) + 1
            return out, weights

        try:
            from transformers.modeling_utils import AttentionInterface
            AttentionInterface.register(ADAPTER_NAME, capture)
        except Exception:                                    # noqa: BLE001
            ALL_ATTENTION_FUNCTIONS[ADAPTER_NAME] = capture
        self._saved = cfg._attn_implementation
        cfg._attn_implementation = ADAPTER_NAME
        for l in self.layers:
            self.model.model.layers[l].self_attn._answer_attention_calls = 0
        return self

    def __exit__(self, *exc):
        self.model.config._attn_implementation = self._saved
        return False

    def verify_ran(self):
        """Raise unless every requested layer actually captured a row.

        A re-route that silently does not happen leaves `store` empty and the
        donor would then be chosen from whatever the caller defaulted to --
        the same failure realized_write's verify_adapter_ran guards, and the
        same reason: the forward completes either way.
        """
        missed = [l for l in self.layers
                  if getattr(self.model.model.layers[l].self_attn,
                             "_answer_attention_calls", 0) == 0]
        if missed:
            raise RuntimeError(
                f"the attention-capture adapter never ran on layers {missed}. "
                "The forward completed, so nothing else would have reported "
                "this. Check that the attention-interface lookup honours "
                f"config._attn_implementation={ADAPTER_NAME!r}.")
        seen = {str(d) for d in getattr(self, "_dtypes", [])}
        if len(seen) != 1:
            raise RuntimeError(
                f"the captured rows came back in {sorted(seen) or 'no'} "
                "dtype(s); the tolerance is derived from exactly one, so a "
                "mixture has to be resolved rather than averaged over")
        u = roundoff_for(seen.pop())
        bad = []
        for l in self.layers:
            bad += [f"layer {l}: {b}"
                    for b in answer_row_faults(self.store[l], roundoff=u)]
        if bad:
            raise RuntimeError("the captured rows are not attention weights:\n"
                               "  " + "\n  ".join(bad))


def label_row_attention(store, layer, label_rows, *, heads=None):
    """[n_demos] attention from the answer position to each demo's label row.

    `heads` averages over a KV group's query heads. Section 4.1 already
    selects matched heads on "demo label rows 的平均 answer attention", so
    averaging over the group is that section's convention rather than a new
    one -- and under GQA the write IS per group, so a per-head donor would be
    choosing on a finer axis than the intervention acts on.
    """
    a = np.asarray(store[int(layer)], dtype=np.float64)
    rows = np.asarray(label_rows, dtype=np.int64)
    if rows.max() >= a.shape[1]:
        raise ValueError(f"label row {int(rows.max())} is outside the "
                         f"{a.shape[1]} captured key positions")
    sub = a if heads is None else a[np.asarray(heads, dtype=np.int64)]
    return sub[:, rows].mean(axis=0)
