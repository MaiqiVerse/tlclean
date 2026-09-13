"""The Function Vector arithmetic, in one place and with no model in it.

Section 13.2.3 fixes every step:

    v_FV^(s) = sum over the CIE top-10 heads of  W_O^(l,h) abar_lh^(s)

where abar_lh^(s) is head (l,h)'s mean answer-row output over prefix seed s's
own 144 validation prompts, and W_O^(l,h) is layer l's o_proj weight sliced to
head h's columns. The vector is added as alpha * v to the ANSWER ROW ONLY of
decoder layer 9's output.

ONE VECTOR PER PREFIX SEED, not one shared vector. Section 13.2.3 used to say
"a global v_FV across the three prefix seeds" and to draw its prompts from the
abolished discovery split; section 2.4(4), section 14.0b-7 and the registry's
per_seed_artifacts all say per seed, and the runner is machine-checked against
the last of those. 14.0b-14 records the correction.

TWO PROPERTIES THAT HOLD EXACTLY, and are checked rather than trusted:

  * alpha = 0 is the identity, bitwise. It is the registered placebo, and a
    placebo that merely "looks the same" tests nothing -- adding 0.0 * v is
    not an approximation, so anything other than bitwise equality means the
    hook is doing something the formula does not say.

  * the grid {0, 0.25, 0.5, 1, 2} is DYADIC, so scaling is exact in binary
    floating point and the injected deltas are exactly proportional:
    inject(h, 2a) - h  ==  2 * (inject(h, a) - h), bitwise. This is algebra,
    not a tolerance, and it catches a hook that renormalises, clips, or
    applies the vector somewhere the scale does not pass through linearly.
    It would NOT hold for a non-dyadic grid, which is one more reason not to
    quietly widen the registered five points.

Nothing here imports torch. The functions take arrays and are written so the
same code runs on numpy in a fixture and on tensors on the GPU: no isinstance
branch, no dtype conversion, no second implementation. Writing the rescale
twice is exactly what went wrong in the ZeroTuning runner.
"""

from __future__ import annotations

import numpy as np

# Section 13.2.3: the upstream CIE default, not a searched quantity.
N_CIE_HEADS = 10
# Section 13.2.3: the upstream edit layer for the Llama-family evaluator. NOT
# searched -- searching it would make the grid a Cartesian product while still
# being called five points, which 2.4(5) forbids by name.
EDIT_LAYER = 9


def head_slice(o_proj_weight, head, head_dim):
    """W_O^(l,h): layer l's o_proj weight restricted to head h's columns.

    HF Llama stores o_proj.weight as [hidden, n_query_heads * head_dim] and
    computes y = concat(head outputs) @ W^T, so head h owns the column block
    [h*head_dim, (h+1)*head_dim) and its contribution to the residual stream is
    that block times its own output. Under GQA the o_proj input is the
    concatenation of the QUERY heads, so this slicing is per query head and
    needs no KV-group arithmetic -- FV reads, it does not intervene per head.
    """
    d = int(head_dim)
    h = int(head)
    if o_proj_weight.shape[1] % d:
        raise ValueError(
            f"o_proj has {o_proj_weight.shape[1]} columns, not a multiple of "
            f"head_dim {d}; the head blocks would not line up")
    n = o_proj_weight.shape[1] // d
    if not 0 <= h < n:
        raise ValueError(f"head {h} out of range for {n} query heads")
    return o_proj_weight[:, h * d:(h + 1) * d]


def fv_from_heads(mean_acts, o_proj_by_layer, heads, head_dim):
    """v_FV = sum_(l,h) W_O^(l,h) abar_lh, over the given heads.

    `mean_acts` is [layer, head, head_dim]: each head's MEAN answer-row output
    as seen at the o_proj INPUT, which is where the upstream defines it.
    `o_proj_by_layer[l]` is layer l's o_proj weight [hidden, n_heads*head_dim].

    The sum is over the top-10 heads only. Summing over all of them would give
    the layer's whole attention output and is a different object.
    """
    a = np.asarray(mean_acts)
    if a.ndim != 3:
        raise ValueError(
            f"mean_acts is {a.ndim}-D, expected [layer, head, head_dim]")
    if a.shape[2] != int(head_dim):
        raise ValueError(
            f"mean_acts last axis is {a.shape[2]}, head_dim is {head_dim}")
    if not heads:
        raise ValueError(
            "no heads given. An empty top-k makes v the zero vector, which is "
            "bitwise identical to the alpha=0 placebo at every alpha -- the "
            "arm would run, produce natural numbers, and look fine")
    v = None
    for layer, head in heads:
        w = head_slice(o_proj_by_layer[int(layer)], head, head_dim)
        term = w @ a[int(layer), int(head)]
        v = term if v is None else v + term
    return v


def top_cie_heads(scores, k=N_CIE_HEADS):
    """The k heads with the largest mean CIE; ties take the smaller (l, h).

    `scores` is [layer, head]. The tie rule is registered (13.2.3) and matters:
    CIE is a mean over 25 corrupted prompts of a probability recovery, so exact
    ties are not rare on a coarse grid, and "whatever argsort returned" is not
    a rule anybody can reproduce.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"scores is {s.ndim}-D, expected [layer, head]")
    if not np.all(np.isfinite(s)):
        bad = int((~np.isfinite(s)).sum())
        raise ValueError(
            f"{bad} of {s.size} CIE scores are not finite; 13.4(13) blocks the "
            "freeze on a NaN rather than letting it sort to one end")
    n = int(k)
    if not 0 < n <= s.size:
        raise ValueError(f"k={n} out of range for {s.size} heads")
    order = sorted(((int(l), int(h)) for l in range(s.shape[0])
                    for h in range(s.shape[1])),
                   key=lambda lh: (-s[lh[0], lh[1]], lh[0], lh[1]))
    return order[:n]


def inject(hidden, alpha, v, row=-1):
    """hidden[..., row, :] += alpha * v, returning a NEW array.

    Expression-only and allocation-returning on purpose. An in-place `+=` on
    the layer output mutates the tensor the model is still using, so a second
    arm reading the same buffer would see the first arm's edit; and a second
    implementation for torch is how the ZeroTuning rescale came to exist twice.

    alpha = 0 returns a bitwise copy: `0.0 * v` is exactly zero for every
    finite v, and adding exact zero changes no bit of a finite float. It is
    NOT bitwise for a non-finite hidden state, which is a state the model
    should never be in and which the caller checks for separately.
    """
    a = alpha * v
    out = hidden.copy() if hasattr(hidden, "copy") else hidden.clone()
    out[..., row, :] = hidden[..., row, :] + a
    return out


def corrupted_label_order(n_demos, prompt_index):
    """A deterministic derangement of the demo DISPLAY labels (13.2.3).

    The corruption shuffles which label is shown next to each demonstration.
    The query and the gold answer do not move -- the CIE asks how much of the
    gold probability a head restores after the task's label mapping has been
    destroyed, so destroying the query too would measure something else.

    A DERANGEMENT, not a shuffle: a random permutation leaves demos in place
    with probability ~1/e per demo, and a "corrupted" prompt that happens to
    be the natural one contributes a zero recovery to every head equally,
    which silently shrinks every CIE toward zero. The rotation by
    1 + (prompt_index mod (n_demos-1)) is a derangement for every input and
    needs no rejection loop.
    """
    n = int(n_demos)
    if n < 2:
        raise ValueError(
            f"{n} demonstrations: a label corruption needs at least two "
            "demos to have anything to exchange")
    shift = 1 + (int(prompt_index) % (n - 1))
    return [(i + shift) % n for i in range(n)]
