"""13.5.2's selective demo memory: the slice, and the per-head visibility mask.

Two pieces, both pure and both checkable without a model.

  slice_demo_kv   from an offline prefill of C_s || D_s, keep the post-RoPE
                  keys and projected values whose tokens belong to D_s.
  visibility_mask the additive [1, heads, q, kv] mask that lets ONLY the
                  frozen H_8 query heads see those columns.

WHY THE MASK IS THE ONLY PART THAT IS NOT FREE. Everything else 13.5.2 asks
for falls out of doing it the straightforward way: concatenating the demo K/V
and running attention ONCE gives the single softmax denominator it requires
(violating that would mean deliberately running two attentions and adding);
giving the receiver its ORIGINAL position ids from the full K=5 prompt makes
every relative offset identical to the monolithic case, because RoPE is
relative; and leaving Q alone preserves the early-write -> late-query ->
late-read path.

The mask is different because GQA shares physical K/V across a group while
13.5.2 requires visibility PER QUERY HEAD: the other members of a carrier's KV
group must not read the memory. Eager attention accepts a 4-D mask, so this is
a tensor to build rather than an attention to rewrite -- but it is the step
that has to be right, because 13.5.2 says an implementation that can only open
the whole group must be renamed `KV-group-selective`, registered as a
deviation, and NOT merged with this design's results.

ZERO GPU.
"""

from __future__ import annotations

import numpy as np

NEG = -np.inf


def mask_pad_for(w_mask, w_model):
    """Zero columns to append so an additive mask lines up with the model's.

    THE MODEL'S MASK CAN BE ONE COLUMN WIDER THAN THE KEYS. With a
    DynamicCache and no 2D `attention_mask` argument, transformers builds its
    4D causal mask at

        target_length = past_seen_tokens + sequence_length + 1

    (modeling_llama.py, `_update_causal_mask`), while the keys are only
    `past + q` long; eager attention then slices it back with
    `attention_mask[:, :, :, : key_states.shape[-2]]`, so the extra column is
    inert. A visibility mask built at the true width `n_common + n_demo +
    n_live` is therefore CORRECT and still fails to broadcast against it.

    THE SLACK IS CATEGORICAL, NOT A TOLERANCE (working rules 11b). It is exactly
    1 when the mask argument is None and exactly 0 when a 2D one is supplied,
    because those are the two branches of that expression. Any other
    difference means the run's (n_common, n_demo, n_live) disagree with the
    cache the model is actually holding -- which is the misalignment this
    width is checked for -- so it raises rather than padding to fit.

    Padding on the RIGHT is what keeps the meaning: C_s and D_s are the
    leading spans, so their column indices must not move.
    """
    a, b = int(w_mask), int(w_model)
    slack = b - a
    if slack not in (0, 1):
        raise ValueError(
            f"the visibility mask is {a} columns wide and the model's "
            f"attention mask is {b}. The only difference this code accepts is "
            "transformers' +1 causal-mask padding (past + q + 1 when no 2D "
            f"attention_mask is given), so a gap of {slack} means n_common + "
            "n_demo + n_live is not the cache the model is holding. Do not "
            "pad past it: the demo columns would be reading the wrong keys.")
    return slack


def slice_demo_kv(k, v, n_common, n_through_demos):
    """(K_D, V_D): the columns whose tokens belong to D_s.

    `k` and `v` are one layer's cached tensors from the offline prefill of
    C_s || D_s, indexed [..., position, dim]. The boundaries come from
    k0_decomposition.prefix_cuts, which verified them against the FULL
    prompt's tokenisation rather than against separately rendered pieces.
    """
    a, b = int(n_common), int(n_through_demos)
    if not 0 <= a < b:
        raise ValueError(
            f"boundaries {a} and {b} do not delimit a non-empty D_s; the "
            "memory would be built from nothing and the arm would be a "
            "K0-offset run wearing another name")
    n = np.asarray(k).shape[-2]
    if b > n:
        raise ValueError(
            f"the demo boundary {b} is past the prefill's {n} positions. The "
            "offline pass must cover C_s || D_s exactly")
    return np.asarray(k)[..., a:b, :], np.asarray(v)[..., a:b, :]


def visibility_mask(n_heads, n_query, n_common, n_demo, n_live,
                    allowed_heads, base_cols=None):
    """Additive [1, heads, n_query, n_common + n_demo + n_live] mask.

    Three column spans, and only the middle one is ever touched:

      [0, n_common)                 C_s -- OPEN TO EVERY HEAD, always.
      [n_common, n_common+n_demo)   D_s -- gated by `allowed_heads`.
      the rest                      the live receiver -- untouched.

    C_s IS A SPAN, NOT AN OMISSION. This function took only (n_demo, n_live)
    and produced a mask starting at the demo block, which is wrong twice: the
    width no longer matches the model's cache_len + q_len, and 13.5.2 says in
    as many words that C_s must not be hidden inside the selective cache --
    "不得把 C_s 只藏进 selective cache" -- because FV, TSLA, the memory arm and
    both K=0 natural arms have to see the same common format. Making it a
    named argument is what stops it being forgotten again.

    The LIVE columns stay zero for every head: their causal structure is the
    model's own and this mask must not quietly become a second causal mask.

    `base_cols` OPENS PART OF THE DEMO SPAN TO EVERY HEAD, given as column
    indices within [0, n_demo). Default None reproduces the three-span
    behaviour exactly.

    It is a SET and not a fourth span because of what the draw does. The
    K=5 demonstrations ARE a subset of the K=10 ones -- checked, all three
    seeds -- but the task shuffles the whole list at the end, on 180 items at
    K=5 and 360 at K=10, so the same demonstrations land SCATTERED through the
    longer prompt (seed 42: positions 2, 3, 10, 11, 14, 20, ... 356, 358,
    359). A split point cannot name that, and one that tried would open a
    contiguous 180 columns that are not the registered K=5 set.
    """
    h, q = int(n_heads), int(n_query)
    c, d, l = int(n_common), int(n_demo), int(n_live)
    if d <= 0:
        raise ValueError(f"n_demo={d}: nothing to make visible")
    if c < 0:
        raise ValueError(f"n_common={c} is negative")
    allowed = {int(x) for x in allowed_heads}
    bad = [x for x in allowed if not 0 <= x < h]
    if bad:
        raise ValueError(f"head(s) {bad} outside the layer's {h} query heads")
    blocked = np.ones(d, dtype=bool)
    if base_cols is not None:
        b = np.asarray(list(base_cols), dtype=np.int64)
        if b.size and (b.min() < 0 or b.max() >= d):
            raise ValueError(
                f"base_cols outside the demo span [0, {d}): min {int(b.min())}, "
                f"max {int(b.max())}")
        if np.unique(b).size != b.size:
            raise ValueError(
                f"base_cols holds {b.size - np.unique(b).size} duplicate "
                "column(s); a column is open or it is not, and a duplicate "
                "means the caller built the set by concatenation rather than "
                "by membership")
        blocked[b] = False
    cols = c + np.nonzero(blocked)[0]
    m = np.zeros((1, h, q, c + d + l), dtype=np.float64)
    for head in range(h):
        if head not in allowed:
            m[0, head][:, cols] = NEG
    return m


def mask_prefix_rows(masks):
    """{layer: [heads, n_prefix] float64}: the per-head column pattern of a
    masks dict built at n_query = 1 and n_live = 1, the one live column
    dropped.

    visibility_mask writes the SAME row for every query position and leaves
    every live column at zero, so a mask for any query length is that
    pattern repeated n_live times with n_live zero columns appended --
    expand_mask below does that on the device. Building the full
    [1, heads, n_live, n_cols] array in numpy for every arm of every query
    and moving it to the card cost ~1 s and 5 GB per arm at a 5.4k-token
    prefix (dbpedia, RESULTS 63.12), which is where a 750-query read spent
    its day. The pattern is built once per arm per seed instead.
    """
    out = {}
    for l, m in masks.items():
        a = np.asarray(m)
        if a.ndim != 4 or a.shape[0] != 1 or a.shape[2] != 1:
            raise ValueError(f"layer {l}: mask is {a.shape}; build the pattern with "
                             "n_query = 1 and n_live = 1 ([1, heads, 1, n_prefix + 1])")
        if a.shape[3] < 2:
            raise ValueError(f"layer {l}: mask has {a.shape[3]} column(s); no prefix")
        if np.any(a[0, :, 0, -1] != 0.0):
            raise ValueError(f"layer {l}: the live column is not zero; that is not a "
                             "visibility mask")
        out[int(l)] = np.ascontiguousarray(a[0, :, 0, :-1])
    return out


def expand_mask(row, n_live, torch, device=None, dtype=None):
    """[1, heads, n_live, n_prefix + n_live] on `device`: `row` ([heads,
    n_prefix], a mask_prefix_rows pattern or its tensor) repeated over the
    n_live query rows, then n_live zero columns -- element for element what
    visibility_mask builds for that query length, in `dtype` (default: the
    row's). Casting the pattern before repeating it gives the same values as
    casting the full array, since the cast is elementwise."""
    r = row if torch.is_tensor(row) else torch.as_tensor(np.asarray(row))
    if device is not None:
        r = r.to(device)
    if dtype is not None:
        r = r.to(dtype)
    if r.ndim != 2:
        raise ValueError(f"row is {tuple(r.shape)}; expected [heads, n_prefix]")
    h, p = r.shape
    n = int(n_live)
    if n < 1:
        raise ValueError(f"n_live={n}: the receiver runs at least one token")
    live = torch.zeros((1, h, n, n), dtype=r.dtype, device=r.device)
    return torch.cat([r.view(1, h, 1, p).expand(1, h, n, p), live], dim=-1)


def mask_faults(m, n_demo, allowed_heads, n_common=0, n_kv_group_width=None,
                kv_group_of=None, base_cols=None):
    """Why `m` is not the mask 13.5.2 describes. Empty means it is.

    The last two arguments turn on the check that matters under GQA: given a
    carrier head, its OTHER group members must be closed. Passing them is how
    a run proves it implemented per-head visibility rather than per-group --
    the distinction 13.5.2 makes renaming conditional on.
    """
    a = np.asarray(m)
    bad = []
    if a.ndim != 4 or a.shape[0] != 1:
        return [f"mask is {a.shape}, expected [1, heads, query, kv]"]
    base = None
    if base_cols is not None:
        base = np.zeros(int(n_demo), dtype=bool)
        b = np.asarray(list(base_cols), dtype=np.int64)
        if b.size and (b.min() < 0 or b.max() >= int(n_demo)):
            return [f"base_cols outside the demo span [0, {int(n_demo)})"]
        base[b] = True
    h, d, c = a.shape[1], int(n_demo), int(n_common)
    allowed = {int(x) for x in allowed_heads}
    if not allowed:
        bad.append("no head is allowed to read the memory; that is the "
                   "K0-offset baseline, not the selective-memory arm")
    for head in range(h):
        common = a[0, head, :, :c]
        demo = a[0, head, :, c:c + d]
        live = a[0, head, :, c + d:]
        # C_s IS OPEN TO EVERYONE, in every arm. 13.5.2: it may not be hidden
        # inside the selective cache, because every method must see the same
        # common format.
        if c and not np.all(common == 0.0):
            bad.append(
                f"head {head}: {int((common != 0).sum())} of the C_s columns "
                "are blocked. C_s is the query-independent common prefix and "
                "13.5.2 forbids hiding it inside the selective cache")
        if head in allowed:
            if not np.all(demo == 0.0):
                bad.append(f"head {head} is in H_8 but {int((demo != 0).sum())} "
                           "of its demo entries are blocked")
            continue
        # A head outside H_8 must see the BASE columns and nothing else in the
        # demo span. Checking only "all blocked" would refuse a correct
        # base-span mask; checking only "the rest blocked" would let a mask
        # that opens everything pass. Both halves are needed.
        if base is None:
            if not np.all(np.isneginf(demo)):
                n_open = int((~np.isneginf(demo)).sum())
                bad.append(
                    f"head {head} is NOT in H_8 but {n_open} demo entries are "
                    "open to it. Under GQA the physical K/V are shared, so "
                    "this is exactly the failure that turns the arm into "
                    "`KV-group-selective` and forbids merging it with this "
                    "design")
            continue
        if not np.all(np.isneginf(demo[:, ~base])):
            n_open = int((~np.isneginf(demo[:, ~base])).sum())
            bad.append(
                f"head {head} is NOT in H_8 but {n_open} entries OUTSIDE the "
                "base columns are open to it. Those are the increment, which "
                "only the carriers may read")
        if not np.all(demo[:, base] == 0.0):
            n_shut = int((demo[:, base] != 0.0).sum())
            bad.append(
                f"head {head} has {n_shut} BASE columns blocked. The base is "
                "the K=5 prompt every head sees in every arm; blocking it "
                "makes the baseline a condition nothing has measured")
        if not np.all(live == 0.0):
            bad.append(f"head {head}: the mask touches {int((live != 0).sum())} "
                       "LIVE columns. It may only govern the demo memory; the "
                       "causal structure is the model's own")
    if kv_group_of is not None and n_kv_group_width:
        for head in sorted(allowed):
            g = kv_group_of(head)
            sibs = [x for x in range(h)
                    if kv_group_of(x) == g and x != head]
            open_sibs = [x for x in sibs
                         if not np.all(np.isneginf(a[0, x, :, c:c + d]))]
            if open_sibs:
                bad.append(
                    f"carrier head {head}'s KV-group siblings {open_sibs} can "
                    "read the memory. 13.5.2 requires visibility per QUERY "
                    "head; an implementation that can only open the group must "
                    "be renamed and registered, not reported as this design")
    return bad
