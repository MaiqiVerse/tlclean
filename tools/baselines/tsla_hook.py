"""TSLA-TL steering: the label subspace, the head score, the head count.

Section 13.2.4. W is the [n_candidates, d] block of unembedding rows for the
frozen candidate tokens, and the label subspace projector is

    P = W^T (W W^T)^+ W

with the pseudo-inverse pinned to atol=0, rtol = n_candidates * eps(float32).
The rank, the smallest non-zero singular value and the condition number are
RETURNED, not printed and forgotten: 13.2.4 requires them on the record and
forbids falling back to a plain inverse.

Each head's answer-row OV output o is scored

    s = mean over c != y of <o, W_y - W_c>   /   || o P ||_2

and a denominator that is zero or non-finite BLOCKS. No epsilon is added --
13.2.4 says so, and the reason is that the denominator is the length of o's
component inside the label subspace, so zero means the head writes nothing
the label readout can see. Nudging it produces a huge finite score for a head
that contributed nothing, which is the opposite of the ranking's intent.

The head count is the official top 3 %: floor(0.03 * n_layers * n_heads),
which is 30 of 1024 here. Computed, never typed -- a hardcoded 30 stops being
3 % the moment anything about the model changes, and would still look right.

Injection reuses fv_hook.inject: the two baselines share the operator (answer
row of one decoder layer's output, scaled by a global alpha) and differ only
in the layer and in how the vector was built. Writing it twice is what the
ZeroTuning rescale did before it was collapsed to one.
"""

from __future__ import annotations

import numpy as np

# 13.2.4: the upstream's fraction, and the upstream's mid-stack edit layer.
# Neither is searched; alpha is the only configured axis (2.4(5)).
TOP_FRACTION = 0.03
EDIT_LAYER = 16


def edit_layer(n_layers):
    """The upstream's mid-stack edit layer for a model of `n_layers`: L // 2.
    16 on the 32-layer Llamas (EDIT_LAYER above, kept for the tests and the
    artifacts that record it), 14 on Qwen2.5-7B, 18 on Qwen3-8B."""
    return int(n_layers) // 2


def label_projector(w):
    """(P, diagnostics) for the label subspace spanned by W's rows.

    `w` is [n_candidates, d]. Returns P = W^T (W W^T)^+ W, which is the
    ORTHOGONAL PROJECTOR onto the row space of W, plus the rank, smallest
    non-zero singular value and condition number that 13.2.4 puts on the
    record.

    The rtol coefficient is n_candidates, following the registered rule. It is
    read off `w` rather than typed: the candidate set has already changed size
    once in this project (the 30/36/50 confusion of working rules 3.10) and a
    literal would have silently stopped matching the rule.
    """
    W = np.asarray(w, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"W is {W.ndim}-D, expected [n_candidates, d]")
    n, d = W.shape
    if n > d:
        raise ValueError(
            f"{n} candidates in {d} dimensions: W W^T cannot have full rank, "
            "and the projector would not be onto an n-dimensional subspace")
    G = W @ W.T
    s = np.linalg.svd(G, compute_uv=False)
    rtol = n * float(np.finfo(np.float32).eps)
    cutoff = rtol * float(s[0]) if s.size else 0.0
    keep = s > cutoff
    rank = int(keep.sum())
    if rank < n:
        raise ValueError(
            f"the candidate Gram matrix has rank {rank} of {n} at the "
            f"registered rtol {rtol:.3e}: two candidate tokens' unembedding "
            "rows are linearly dependent, so the label subspace is smaller "
            "than the candidate set and the projector does not separate them.")
    Ginv = np.linalg.pinv(G, rcond=rtol)
    P = W.T @ Ginv @ W
    smin = float(s[keep][-1]) if rank else float("nan")
    return P, {"rank": rank, "n_candidates": n, "rtol": rtol,
               "smallest_nonzero_singular_value": smin,
               "condition_number": float(s[0] / smin) if smin else float("inf"),
               "rule": "torch.linalg.pinv(W W^T, atol=0, rtol=n*eps(float32))"}


def projector_faults(P, w, tol=None):
    """Why `P` is not the projector it claims to be. Empty means it is.

    Three exact algebraic facts, checked rather than assumed, with a tolerance
    derived from the operand sizes and eps -- NOT from any observed deviation
    (working rules 11): P is symmetric, P is idempotent, and it fixes every row of
    W, because P W^T = W^T (W W^T)^+ (W W^T) = W^T.
    """
    P = np.asarray(P, dtype=np.float64)
    W = np.asarray(w, dtype=np.float64)
    scale = float(np.abs(P).max()) or 1.0
    t = tol if tol is not None else 1e3 * np.finfo(np.float64).eps * scale * P.shape[0]
    bad = []
    if abs(float(np.abs(P - P.T).max())) > t:
        bad.append(f"P is not symmetric (max |P - P^T| "
                   f"{float(np.abs(P - P.T).max()):.3e} > {t:.3e})")
    d = float(np.abs(P @ P - P).max())
    if d > t:
        bad.append(f"P is not idempotent (max |PP - P| {d:.3e} > {t:.3e})")
    e = float(np.abs(P @ W.T - W.T).max())
    if e > t:
        bad.append(f"P does not fix the label rows (max |P W^T - W^T| "
                   f"{e:.3e} > {t:.3e}); the subspace it projects onto is not "
                   "the one W spans")
    return bad


def head_scores(ov, w, gold_class, P):
    """s_lh for one query: alignment with the gold margin, per unit label norm.

    `ov` is [layer, head, d]: each head's answer-row OV output. The numerator
    is the MEAN over the non-gold candidates of <o, W_y - W_c>, i.e. how much
    this head pushes the gold candidate above the others; the denominator is
    the length of o inside the label subspace.

    A zero or non-finite denominator RAISES. 13.2.4 forbids an epsilon, and
    the reason is not tidiness: the denominator vanishing means the head wrote
    nothing the label readout can see, and dividing a near-zero numerator by a
    nudged near-zero denominator manufactures a large score for a head that
    contributed nothing.
    """
    O = np.asarray(ov, dtype=np.float64)
    W = np.asarray(w, dtype=np.float64)
    if O.ndim != 3 or O.shape[2] != W.shape[1]:
        raise ValueError(f"ov {O.shape} does not match W {W.shape}; expected "
                         "[layer, head, d]")
    y = int(gold_class)
    if not 0 <= y < W.shape[0]:
        raise ValueError(f"gold index {y} outside the {W.shape[0]} candidates")
    others = [c for c in range(W.shape[0]) if c != y]
    # mean_c <o, W_y - W_c> = <o, W_y - mean_c W_c>, one contraction not n
    margin_dir = W[y] - W[others].mean(axis=0)
    num = O @ margin_dir
    den = np.linalg.norm(O @ np.asarray(P, dtype=np.float64), axis=2)
    # NUMERICALLY ZERO, NOT ONLY EXACTLY ZERO.
    #
    # 13.2.4 says to block on a denominator that is zero or non-finite. Taken
    # literally as `den == 0.0` that check CANNOT FIRE: a head writing nothing
    # inside the label subspace gives the projection's rounding error, order
    # 1e-16 times ||o||, never a hard zero. And 1/1e-16 is exactly the huge
    # spurious score the rule exists to prevent, so the literal reading admits
    # the pathology it was written against -- working rules 11b, a criterion
    # nothing can satisfy.
    #
    # The threshold is RELATIVE and derived, not tuned: ||o P|| / ||o|| at the
    # level of float64 rounding for a d-dimensional projection means the
    # component is indistinguishable from zero at this precision. Nothing here
    # is fitted to an observed value.
    #
    # This BLOCKS MORE than the literal rule, never less. Widening a refusal is
    # the safe direction; adding an epsilon to the denominator -- which 13.2.4
    # forbids by name -- would be the other one.
    norm_o = np.linalg.norm(O, axis=2)
    floor = 64.0 * float(np.finfo(np.float64).eps) * np.sqrt(O.shape[2]) * norm_o
    bad = ~np.isfinite(den) | (den <= floor)
    if bad.any():
        l, h = np.argwhere(bad)[0]
        raise ValueError(
            f"{int(bad.sum())} head(s) have a label-subspace norm at or below "
            f"float64 rounding, first (layer {l}, head {h}): ||o P|| = "
            f"{den[l, h]:.3e} vs ||o|| = {norm_o[l, h]:.3e} (floor "
            f"{floor[l, h]:.3e}). 13.2.4 blocks here rather than adding an "
            "epsilon: the denominator vanishing means the head wrote nothing "
            "the label readout can see, and dividing by rounding noise would "
            "rank it as if it had written a great deal.")
    return num / den


def n_top_heads(n_layers, n_heads, fraction=TOP_FRACTION):
    """floor(fraction * n_layers * n_heads). 30 of 1024 at the registered 3 %.

    Computed from the model's own shape. A literal 30 would keep looking
    correct while silently ceasing to be three per cent.
    """
    n = int(np.floor(float(fraction) * int(n_layers) * int(n_heads)))
    if n < 1:
        raise ValueError(
            f"{fraction} of {n_layers}x{n_heads} rounds to {n} heads")
    return n


def top_heads(scores, k):
    """The k highest-scoring heads; ties take the smaller (layer, head).

    Same registered tie rule as the FV top-10 and the carrier top-8. Stated
    once per place it is used because "whatever argsort returned" is not a
    rule anybody can reproduce, and averaging over 50 prompts leaves ties.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"scores is {s.ndim}-D, expected [layer, head]")
    if not np.all(np.isfinite(s)):
        raise ValueError(f"{int((~np.isfinite(s)).sum())} of {s.size} scores "
                         "are not finite")
    order = sorted(((int(l), int(h)) for l in range(s.shape[0])
                    for h in range(s.shape[1])),
                   key=lambda lh: (-s[lh[0], lh[1]], lh[0], lh[1]))
    if not 0 < int(k) <= s.size:
        raise ValueError(f"k={k} out of range for {s.size} heads")
    return order[:int(k)]


def steering_vector(ov_sum_by_prompt):
    """v_TSLA = mean over prompts of the SUM of the selected heads' OV output.

    Sum inside a prompt, mean across prompts -- in that order. Averaging first
    and summing after gives the same number only because both are linear, but
    the shapes differ and the mistake would not be visible in the result.
    """
    a = np.asarray(ov_sum_by_prompt, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected [prompt, d], got {a.shape}")
    if a.shape[0] == 0:
        raise ValueError("no prompts: the mean is undefined, and an empty "
                         "vector is bitwise identical to the alpha=0 placebo")
    return a.mean(axis=0)


# THE UPSTREAM'S HEAD CLASSES (Localizing_TR_TL/experiments/get_demos_and_heads.py
# 236-262). It ranks every head three ways and builds one task vector from
# the top 3 % of each (ablation_tv.py:81/196):
#     TL      margin_add   = mean_c <o, W_y - W_c> / ||o P||   -- head_scores above
#     TR      cossim_norm  = ||o P||                           -- tr_scores
#     random  a control draw                                   -- random_heads
# The paper's own reading is that TR vectors are what matters for fixed-label
# classification and TL vectors for open generation; this project ran TL
# first (baseline_under_review 3.4) and adds TR and random as further
# families under the same alpha grid, layer and construction.
HEAD_CLASSES = ("tl", "tr", "random")


def tr_scores(ov, P):
    """s_lh for one query, TR: ||o P||_2, the head's length inside the label
    subspace -- how much it writes into the task subspace at all, regardless
    of which label. The upstream's `cossim_norm`, summed over prompts by the
    caller exactly as head_scores is."""
    O = np.asarray(ov, dtype=np.float64)
    if O.ndim != 3:
        raise ValueError(f"ov {O.shape}: expected [layer, head, d]")
    return np.linalg.norm(O @ np.asarray(P, dtype=np.float64), axis=2)


def random_heads(n_layers, n_heads, k, seed):
    """k heads drawn uniformly without replacement -- the upstream's
    `random_head` control -- seeded by the demo seed so the draw is on the
    record and re-runs reproduce it."""
    n = int(n_layers) * int(n_heads)
    if not 0 < int(k) <= n:
        raise ValueError(f"k={k} out of range for {n} heads")
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(k), replace=False)
    return sorted((int(i // int(n_heads)), int(i % int(n_heads))) for i in idx)


def sidecar_key(cls):
    """Where a class's vectors live in the sidecar: `tsla_vectors` for TL
    (the original key, unchanged), `tsla_vectors_<cls>` otherwise."""
    if cls not in HEAD_CLASSES:
        raise ValueError(f"head class {cls!r}: expected one of {HEAD_CLASSES}")
    return "tsla_vectors" if cls == "tl" else f"tsla_vectors_{cls}"
