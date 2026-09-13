"""EXPLORATORY: same-class donor chosen by attention. NO torch, NO GPU.

NOT A PREREGISTERED ARM. `prereg_method_A.md` section 4 fixes eleven arms and
`tools/prototype_targets.py` holds exactly those; putting a twelfth in there
would make that module's own docstring false. These live here so the line
between "registered" and "added after seeing the data" is a file boundary
rather than a comment.

WHY IT EXISTS. Section 4 already carries `single-same donor` -- "same-class
demos in stable demo-ID cyclic order; i's donor is the next one" -- whose
donor rule is deliberately UNINFORMATIVE, because its job is to answer "does
one substitution behave like the mean". That is a different question from
"does the RIGHT one behave better than the mean", and the test-set readout
gives a reason to ask the second: the effect is monotone in a candidate's
natural rank, and gold at rank >= 9 is actively HARMED (RESULTS 44.4d/44.4e).
A plausible reading is that the class mean is dragged by four demos that do
not match this query, which is exactly what replacing it with the
best-matching exemplar would fix.

TWO VARIANTS, and they are different experiments:

  frozen    the donor is chosen ONCE per (seed, layer, carrier, class) from
            VALIDATION-AVERAGED label-row attention. The write stays a fixed
            per-seed vector, so gamma*, the freeze and "the write is
            determined by the prompt's natural values" all keep their
            meaning. Precedent: section 4.2's `non-label dose` scales by a
            validation attention ratio, and 4.1 selects matched heads by
            "demo label rows 的平均 answer attention".
  dynamic   the donor is chosen PER QUERY from that query's own natural
            attention. The write becomes query-adaptive, which is a different
            estimand -- it can no longer be described as a fixed edit to the
            prompt's values, and any comparison against a frozen arm has to
            say so.

COLLAPSE, NOT PERMUTATION. `single_same` is a permutation: every demo gets a
different donor and the class keeps N_c distinct values. This one COLLAPSES
the class onto one exemplar, so at gamma=1 all N_c slots hold the same vector
and the winner is unchanged by construction. The two therefore probe
different things and neither is a stronger version of the other; that has to
be said whenever they are compared.

THIS DOES NOT FILTER POSITION. The original question that led here was
whether the prototype filters positional content. It does not, and neither
does this: a single donor swaps demo i's positional content for donor j's
rather than averaging it away. What makes that tolerable here is measured
rather than assumed -- RESULTS section 28.5 finds the LABEL-row channel has
slot excess +0.006 against demo excess +0.201 (33x), with the sink absent
from label rows and the mass mid-prompt rather than at the end.
"""

from __future__ import annotations

import numpy as np

from tools.prototype_targets import bf16_round

ARMS = ("maxattn_frozen", "maxattn_dynamic")


def max_attention_donor(labels, attn, demo_ids):
    """Index of the highest-attention demo OF EACH DEMO'S OWN CLASS.

    Returns [n] int, where out[i] is the donor for demo i -- the same index
    for every demo of a class, since the class collapses onto one exemplar.

    SELF IS INCLUDED. The winner's donor is itself, so it is written with its
    own value and does not move. Excluding self would send the winner to the
    runner-up, which is neither what "replace the class by its best exemplar"
    means nor something any hypothesis here asks about.

    TIES BREAK ON STABLE DEMO ID, not on row order. Attention arrives as
    float32 from a bf16 model and exact ties are ordinary at this precision
    (the test readout found 27 of 750 queries with a tied top candidate), so
    an unspecified tie-break would make the arm depend on how the prompt
    happened to be assembled -- the same reason section 4's cyclic donor is
    keyed on demo ID.
    """
    labels = np.asarray(labels)
    a = np.asarray(attn, dtype=np.float64)
    ids = [str(x) for x in demo_ids]
    if a.shape != labels.shape:
        raise ValueError(f"attention is {a.shape}, labels are {labels.shape}; "
                         "one attention weight per demo label row is required")
    if len(ids) != labels.size:
        raise ValueError(f"{len(ids)} demo ids for {labels.size} demos")
    if len(set(ids)) != len(ids):
        raise ValueError("demo IDs must be unique to break attention ties "
                         "reproducibly")
    if not np.all(np.isfinite(a)):
        raise ValueError("attention contains non-finite values; a donor "
                         "chosen from NaN is not a choice")
    if np.any(a < 0):
        raise ValueError("attention has negative entries; these are softmax "
                         "weights and a negative one means the wrong tensor "
                         "was captured")
    out = np.empty(labels.size, dtype=int)
    for c in np.unique(labels):
        rows = np.where(labels == c)[0]
        # stable ID order first, then a stable argmax over it: the first
        # maximal entry in ID order wins, so ties are resolved by ID.
        order = rows[np.argsort([ids[r] for r in rows], kind="stable")]
        best = order[int(np.argmax(a[order]))]
        out[rows] = best
    return out


def donor_concentration(labels, attn):
    """Per class, the winner's share of that class's label-row attention.

    THE NUMBER THAT DECIDES WHETHER THIS ARM IS ANYTHING. With N_c demos in a
    class, a share near 1/N_c means the attention is flat and the argmax is
    picking noise -- the arm then degenerates into `single_same` with an
    arbitrary donor, and "best-matching exemplar" is a name for a coin flip.
    A share well above 1/N_c means there is something to select.

    Returns {class: (share, n_c, uniform_share)}.
    """
    labels = np.asarray(labels)
    a = np.asarray(attn, dtype=np.float64)
    out = {}
    for c in np.unique(labels):
        rows = np.where(labels == c)[0]
        tot = float(a[rows].sum())
        share = float(a[rows].max() / tot) if tot > 0 else float("nan")
        out[int(c)] = (share, int(rows.size), 1.0 / rows.size)
    return out


def donor_targets(labels, attn, demo_ids):
    """T_i for the max-attention arms: the donor's natural value, per demo.

    Takes v0 through `written_rows` rather than here so that the donor choice
    and the shrinkage stay separable -- the choice is the experiment, the
    shrinkage is section 3's convention and must not be re-implemented.
    """
    return max_attention_donor(labels, attn, demo_ids)


def written_rows_from_donor(gamma, v0, donor, *, cast_bf16=True):
    """v~_i = v0_i + gamma (v0_{donor(i)} - v0_i). Section 3's convention.

    TAKES THE DONOR, NOT THE ATTENTION. The choice and the write are separate
    steps and only the choice differs between the two variants -- frozen picks
    from validation-averaged attention, dynamic from this query's own. Feeding
    attention in here would make the write itself look variant-specific and
    would let a frozen run recompute a donor at write time, which is the
    failure `frozen_donor_faults` exists to catch.
    """
    v0 = np.asarray(v0, dtype=np.float64)
    d = np.asarray(donor, dtype=np.int64)
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma={gamma} outside [0, 1]")
    if v0.ndim != 2:
        raise ValueError(f"v0 is {v0.shape}, expected [n_demos, d_head]")
    if d.shape != (v0.shape[0],):
        raise ValueError(f"donor is {d.shape}, expected one index per demo "
                         f"({v0.shape[0]})")
    if d.min() < 0 or d.max() >= v0.shape[0]:
        raise ValueError(f"donor indices span [{d.min()}, {d.max()}], outside "
                         f"the {v0.shape[0]} demos")
    out = v0 + float(gamma) * (v0[d] - v0)
    return bf16_round(out) if cast_bf16 else out


def written_rows(gamma, v0, labels, attn, demo_ids, *, cast_bf16=True):
    """v~_i = v0_i + gamma (T_i - v0_i), with T_i the class's best exemplar.

    Section 3's shrinkage convention, applied to a donor this file chooses.
    The formula is written out rather than imported because
    `written_label_rows` refuses arms outside the registered eleven, and
    loosening that guard to admit an exploratory arm would remove the check
    that keeps the registered set closed.
    """
    v0 = np.asarray(v0, dtype=np.float64)
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma={gamma} outside [0, 1]")
    if v0.ndim != 2 or v0.shape[0] != np.asarray(labels).size:
        raise ValueError(f"v0 is {v0.shape}, expected [n_demos, d_head]")
    donor = max_attention_donor(labels, attn, demo_ids)
    return written_rows_from_donor(gamma, v0, donor, cast_bf16=cast_bf16)


def frozen_donor_faults(manifest, seeds, layers_by_seed):
    """Why a frozen-donor manifest may not be used. Empty means it may.

    The frozen variant's whole claim is that the donor was fixed BEFORE the
    run, per (seed, layer, carrier, class), from validation. A manifest that
    is missing a cell would otherwise be filled in at run time from the
    query's own attention -- which is silently the dynamic arm wearing the
    frozen arm's name.
    """
    bad = []
    if not isinstance(manifest, dict):
        return [f"donor manifest is {type(manifest).__name__}, expected an "
                "object keyed by seed"]
    for s in seeds:
        cell = manifest.get(str(s))
        if not isinstance(cell, dict):
            bad.append(f"seed {s}: no frozen donors")
            continue
        for layer, carriers in sorted(layers_by_seed.get(s, {}).items()):
            for g in carriers:
                key = f"{layer}:{g}"
                if key not in cell:
                    bad.append(f"seed {s} has no frozen donor for layer "
                               f"{layer} carrier {g}; a missing cell would be "
                               "filled from the query at run time, which is "
                               "the dynamic arm under the frozen name")
    return bad
