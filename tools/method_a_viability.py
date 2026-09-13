"""The L3.1 Method A viability gate (prereg section 13.6). ZERO GPU, analysis only.

The question is *should we spend the six baseline implementations?* -- not
"does Method A work", which is H4, lives on the test set, and is not touched
here. Nothing in this file reads a test prediction; the whole gate runs on the
144 validation queries that gamma* was always going to be chosen on, so it adds
no leakage and does not weaken H4 or H6. What it changes is engineering order.

WHY CROSS-FITTING. A five-point grid selected and judged on the same 144
queries reports the minimum of five noisy numbers as if it were an estimate.
Two deterministic folds fix that: gamma is chosen on one half and the difference
is measured on the other, then swapped, so every one of the 144 queries is
evaluated under a gamma that did not see it.

BOTH INFERENCES TEST THE SAME THING: THE SELECTION PROCEDURE.
This is the part an earlier version got wrong. The bootstrap reselected gamma
inside every resample, but the p-value was a plain sign-flip on the
already-selected out-of-fold differences -- which tests "given the two gammas
this run happened to pick, is d zero?", a different estimand from the one the
interval covers. GO required both, so its two conjuncts were about two
different quantities.

PER SEED SINCE SECTION 14.0a. Each prefix seed draws its OWN 144 validation
queries, so there is no shared query axis: position q under seed 42 and under
seed 43 are different queries, and "average over the three seeds within a
query" is undefined. The unit is the (query, seed) CELL -- 3 x 144 = 432 of
them -- and each seed gets its own 72/72 folds and its own two selections, six
in all.

The permutation is over the PROCEDURE. Under the null that a query's
difference vector is symmetric about zero, one sign eps per unique QUERY ID is
applied to every cell that query has, across whichever seeds drew it, and then
the fold-internal selection and the cross-fitted evaluation are re-run inside
the permuted data. Clusters are therefore of size 1, 2 or 3 rather than always
3 -- which is why the inference clusters on query_id and why query_id must
stay free of the seed.

It still reduces. Write e[g,s,q] = nll[g,s,q] - natural[s,q]; the natural term
does not depend on gamma, so

    selection on (s, F)  =  argmin_g (1/|F_s|) sum_{q in F_s} eps_q e[g,s,q]
    out-of-fold value    =  eps_q * e[ghat_(s, other), s, q]

which is one block per (seed, fold) -- SIX blocks, not the two matrix products
the shared axis allowed. The cost is stated as blocks rather than as a
multiplication count because the count depends on how the folds pack. With
eps == +1 it must reproduce `cross_fit` exactly, which is asserted rather than
assumed.

WHAT THE INTERVAL COVERS. The bootstrap resamples unique query ids -- a drawn
query brings all of its cells -- and reselects gamma inside every draw, but
holds the six head sets H_sF fixed, because the e of a head set never
forwarded cannot be computed offline. The interval is therefore conditional on
those six, and section 13.6.4 says so rather than claiming a direction for the
resulting understatement: a narrower interval makes GO and STOP both easier
and squeezes INCONCLUSIVE, so it is not conservative either way.

GAMMA = 0 IS THE IDENTITY OPERATOR, which gives a free correctness check on the
entire forward path: the gamma=0 slice must equal the natural arm BITWISE. It
also survives permutation -- e[0] is identically zero, so a flipped dataset
still has gamma=0 as its identity -- and it makes futility exact: if both folds
select gamma=0 then d is identically zero and the verdict is STOP, the same
conclusion section 9 registers for gamma*_group = 0.

A VERDICT IS ONLY A VERDICT UNDER THE REGISTERED CONFIGURATION. Running with a
reduced bootstrap, without the opponent freeze, or on an npz that cannot be
shown to be the registered experiment yields status DIAGNOSTIC and
decision=None -- not a GO/STOP with a warning printed above it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# All inherited, none chosen here.
GAMMA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)   # section 3
DELTA_MIN = 0.02                           # see the note in `decide`
TIE_TOL = 1e-8                             # section 5(4)
N_BOOT = 10_000                            # section 9
N_PERM = 1_000_000                         # section 6.2
STAT_SEED = 20260826                       # section 9
ALPHA = 0.05
# Section 2.1, from tools/prereg_config. N_VALIDATION is DERIVED there
# (N_ELIGIBLE * VALIDATION_PER_CLASS); it used to be the literal 144 here and
# in freeze_baseline_spec, so changing either input left both asserting a
# shape no run produced.
from tools.prereg_config import (N_ELIGIBLE,  # noqa: E402,F401
                                 N_VALIDATION, REGISTERED_SEEDS,
                                 VALIDATION_PER_CLASS)
GROUP_IMPL = "gqa_group_v"                 # section 3.1: Method A-GROUP
# The metrics' own ranges. Checked because finiteness is not the same
# question: -10 is perfectly finite and impossible as an NLL.
DOMAIN = {"nll": (0.0, np.inf), "acc": (0.0, 1.0),
          "brier": (0.0, 2.0)}
# accuracy is stronger than its range: each CELL is one query's correctness,
# so it is an indicator, not a rate. 0.5 sits inside [0, 1] and is not a
# value a single query can take.
INDICATOR = ("acc", "natural_acc")
DECISIONS = ("GO", "STOP", "INCONCLUSIVE")
VERIFIED, DIAGNOSTIC = "VERIFIED", "DIAGNOSTIC"

assert list(GAMMA_GRID) == sorted(GAMMA_GRID), \
    "the tie rule takes the first winning index as the smallest gamma"


# ==========================================================================
# folds
# ==========================================================================
def assign_folds(query_ids, class_idx, *, require_registered_shape=True):
    """Deterministic, class-balanced, no RNG (section 13.6.2), PER SEED.

    `require_registered_shape=False` keeps the RANKING RULE and drops only the
    "exactly VALIDATION_PER_CLASS rows in every class" guard. Two callers need
    the rule and cannot satisfy the guard: discover_carriers' --limit smoke
    path, which is MEANT to build folds the bundle check then refuses, and the
    fixtures that exercise it. The guard belongs to the gate, where an
    unbalanced split really is a confound; the rule belongs to both, and
    splitting them this way is what keeps there being one implementation of
    it. There was briefly a second copy and it ranked by a different key --
    see fold_labels for what that cost.

    `query_ids` and `class_idx` are [S, Q]: each seed draws its own 144, so
    each seed gets its own 72/72 split. An ODD per-class count is supported:
    the folds then differ by at most one query per seed, and by zero when the
    class count is even, because the extra query alternates between folds
    class by class rather than always going to fold 0. A single split over a shared axis is
    what this used to compute and there is no shared axis left.

    Duplicates are checked WITHIN a seed, not across. The same query drawn by
    two seeds is one query in two cells, which is legal and is exactly what
    the clustering downstream is for; a repeat inside one seed would put one
    query in both folds of that seed and is not.
    """
    qs = np.asarray(query_ids)
    cs = np.asarray(class_idx)
    if qs.ndim != 2 or cs.ndim != 2 or qs.shape != cs.shape:
        raise ValueError(
            f"query_ids {qs.shape} and class_idx {cs.shape} must both be "
            "[seed, query]. A 1-D array is the pre-14.0a shared axis, which "
            "no longer exists.")
    fold = np.empty(qs.shape, dtype=np.int8)
    for si in range(qs.shape[0]):
        row_q = [str(q) for q in qs[si]]
        row_c = [int(c) for c in cs[si]]
        if len(set(row_q)) != len(row_q):
            raise ValueError(
                f"seed index {si}: duplicate query ids within one seed; that "
                "query would be in both of that seed's folds")
        by_class = {}
        for i, (q, c) in enumerate(zip(row_q, row_c)):
            by_class.setdefault(c, []).append((q, i))
        bad = {c: len(v) for c, v in by_class.items()
               if len(v) != VALIDATION_PER_CLASS} if require_registered_shape \
            else {}
        if bad:
            raise ValueError(
                f"seed index {si}: section 2.1 fixes validation at "
                f"{VALIDATION_PER_CLASS} queries per class; these classes have "
                f"something else: {bad}. An unequal split would make the two "
                "folds differ in class composition, which is precisely the "
                "confound the deterministic split removes.")
        # The starting parity ALTERNATES by class. With an even per-class
        # count this only decides which half of a class is called fold 0 --
        # both folds still get exactly half of every class -- but with an ODD
        # count it is what keeps the folds balanced overall: without it every
        # class hands its extra query to fold 0, so per_class=5 over 36
        # classes gives 108/72 rather than 90/90, silently and always in the
        # same direction.
        for ci, c in enumerate(sorted(by_class)):
            for rank, (_q, i) in enumerate(sorted(by_class[c])):
                fold[si, i] = (rank + ci) % 2
    return fold


# ==========================================================================
# the reduced form: everything below works on e[g, s, q]
# ==========================================================================
def seed_mean_effect(nll, natural):
    """e = nll - natural, with the natural term broadcast over the head axis.

    Accepts nll[gamma, seed, query] or nll[head, gamma, seed, query] and
    returns the same shape. NATURAL IS HEAD-INDEPENDENT -- it writes nothing,
    so there is one natural forward per (seed, query) however many head sets
    the intervened forwards used. Broadcasting it here rather than asking
    callers to tile it keeps that asymmetry visible: it is a fact about the
    arm, not a shape convenience.

    NO average over seeds. Everything downstream -- the fold selection, the
    cross-fitted difference, every permutation -- is a function of this array
    alone, because the natural term is the same for every gamma and cancels
    out of every argmin and every difference. That is exact, not an
    approximation, and it is what makes a million permutations affordable.
    """
    a = np.asarray(nll, dtype=np.float64)
    nat = np.asarray(natural, dtype=np.float64)
    if nat.ndim != 2:
        raise ValueError(f"natural is {nat.ndim}-D, expected [seed, query]")
    if a.ndim == 3:
        if a.shape[1:] != nat.shape:
            raise ValueError(f"nll {a.shape} and natural {nat.shape} disagree")
        return a - nat[None]
    if a.ndim == 4:
        if a.shape[2:] != nat.shape:
            raise ValueError(f"nll {a.shape} and natural {nat.shape} disagree")
        return a - nat[None, None]
    raise ValueError(
        f"nll is {a.ndim}-D, expected [gamma, seed, query] or "
        "[head, gamma, seed, query]")


def _pick(means):
    """argmin along the last axis with section 5(4)'s tie rule, vectorised.

    GAMMA_GRID is ascending and index order is gamma order, so the FIRST entry
    within tolerance of the minimum is the smallest winning gamma.
    """
    m = np.asarray(means, dtype=np.float64)
    return (m <= m.min(axis=-1, keepdims=True) + TIE_TOL).argmax(axis=-1)


def _as_hgsq(e, fold):
    """(e as [head, gamma, seed, query], fold as [seed, query]).

    The head axis is length 2 and is NOT optional: 13.6.3 evaluates each fold
    under the other fold's head set, so an array with one head set cannot
    express the design. Accepting a 3-D array by broadcasting it would make
    the head layer of the selection silently disappear, which is the state
    this replaces.
    """
    a = np.asarray(e, dtype=np.float64)
    fold = np.asarray(fold)
    if a.ndim != 4:
        raise ValueError(
            f"e is {a.ndim}-D, expected e[head, gamma, seed, query]. Since "
            "13.6.3 the two folds select their own head sets and each is "
            "evaluated under the other's, so a single-head-set array is a "
            "different experiment")
    if a.shape[0] != 2:
        raise ValueError(f"e has {a.shape[0]} head sources, expected 2 (the "
                         "set selected on fold 0 and on fold 1)")
    if fold.shape != a.shape[2:]:
        raise ValueError(f"e {a.shape} and fold {fold.shape} disagree; "
                         "expected fold[seed, query]")
    return a, fold


def select_gamma(e, si, rows, hi=0):
    """The registered rule for ONE seed and ONE head set, on a subset of rows.

    `hi` indexes the head-source axis: 13.6.3 chooses gamma_sF under H_sF, so
    the head set and the gamma come from the same fold by construction rather
    than by the caller remembering to pair them.
    """
    a = np.asarray(e, dtype=np.float64)[hi]
    idx = np.asarray(rows)
    if idx.size == 0:
        raise ValueError(f"cannot select gamma on an empty fold (seed {si})")
    means = a[:, si, :][:, idx].mean(axis=1)
    return int(_pick(means)), means


def cross_fit(e, fold, gammas=GAMMA_GRID):
    """Out-of-fold per-cell differences, plus each (seed, fold)'s gamma.

    SIX selections, not two: (seed, fold) is the unit, because each seed's
    folds hold that seed's own queries and a gamma chosen on seed 42's fold A
    says nothing about seed 43.
    """
    a, fold = _as_hgsq(e, fold)
    S = a.shape[2]
    chosen, d = {}, np.full(a.shape[2:], np.nan)
    for si in range(S):
        rows = {f: np.flatnonzero(fold[si] == f) for f in (0, 1)}
        for f, r in rows.items():
            if r.size == 0:
                raise ValueError(f"seed index {si}: fold {f} is empty")
        for f in (0, 1):
            # fold f's gamma is chosen under fold f's HEAD SET
            chosen[(si, f)] = select_gamma(a, si, rows[f], hi=f)[0]
        for f in (0, 1):
            r = rows[f]
            # the head set AND gamma selected on the OTHER fold of this seed.
            # Both come from index 1-f, so they cannot be paired wrongly.
            d[si, r] = a[1 - f, chosen[(si, 1 - f)], si, r]
    if not np.all(np.isfinite(d)):
        raise ValueError("some cell got no out-of-fold difference")
    meta = {f"seed{si}_fold{f}_selected": gammas[chosen[(si, f)]]
            for si in range(S) for f in (0, 1)}
    meta["_head_source"] = {f"seed{si}_fold{f}": f
                            for si in range(S) for f in (0, 1)}
    meta["_idx"] = chosen
    return d, meta


def apply_choice(e, fold, chosen):
    """Out-of-fold values at a GIVEN set of six selections.

    Split out of cross_fit so an auxiliary metric can be reported at the
    configuration the primary criterion selected. Reusing cross_fit for that
    silently turns the auxiliary into a second selection procedure.
    """
    a, fold = _as_hgsq(e, fold)
    d = np.full(a.shape[2:], np.nan)
    for si in range(a.shape[2]):
        for f in (0, 1):
            r = np.flatnonzero(fold[si] == f)
            key = (si, 1 - f)
            if key not in chosen:
                raise KeyError(
                    f"no selection recorded for seed index {si}, fold "
                    f"{1 - f}; cross_fit must have produced all six")
            d[si, r] = a[1 - f, chosen[key], si, r]
    if not np.all(np.isfinite(d)):
        raise ValueError("some cell got no out-of-fold value")
    return d


def strict_int(v, what):
    """int(v) only when v IS an integer. Raises otherwise.

    Bare int() truncates, so an identity field silently rounds: seed 42.9
    became seed 42 and class 0.75 became class 0, and both then matched the
    registered value. bool is excluded because it is an int subclass and
    True would pass as 1.
    """
    # np.bool_ FIRST, and by name: it is not a Python bool, so the isinstance
    # check below misses it, and it is not numbers.Real either -- but float()
    # happily turns it into 1.0. Checking bool alone let np.bool_(True)
    # through as the integer 1.
    if isinstance(v, (bool, np.bool_)):
        raise ValueError(f"{what} is a bool ({v!r}); bool is an int subclass "
                         "and would compare as 0/1")
    # A NUMBER, not something float() can parse. float("5.0") succeeds, so
    # the string "5.0" was accepted as K=5 -- an identity field taken from a
    # file must not be coerced from text, because a file that says "5.0"
    # where the schema says 5 was not written by the runner.
    if not isinstance(v, numbers.Real):
        raise ValueError(
            f"{what} is {type(v).__name__} {v!r}, not a real number. "
            "float() would parse a numeric string, so a JSON field holding "
            '"5.0" would have been accepted as 5.')
    # INTEGRAL FIRST, without going through float(). Python ints are
    # unbounded and float(10**1000) raises OverflowError -- which callers
    # catching ValueError/TypeError did not see, so a huge integer broke the
    # "return faults" contract by escaping as an exception.
    if isinstance(v, numbers.Integral):
        return int(v)
    try:
        f = float(v)
    except (OverflowError, ValueError) as e:
        raise ValueError(f"{what} is {v!r}, which does not convert to a "
                         f"finite float: {e}") from None
    if not np.isfinite(f):
        raise ValueError(f"{what} is {v!r}, which is not a finite number")
    if f != int(f):
        raise ValueError(
            f"{what} is {v!r}, which is not an integer. int() truncates, so "
            "this would have been accepted as "
            f"{int(f)} and matched whatever that equals.")
    return int(f)


def cluster_index(query_ids):
    """Map each [S, Q] cell to its query's cluster id.

    A query drawn by two seeds is ONE cluster with two cells. Section 13.6.4
    flips and resamples the query, not the cell, so this mapping is what makes
    the two inferences agree with the estimand.
    """
    qs = np.asarray(query_ids)
    uniq = {}
    idx = np.empty(qs.shape, dtype=np.int64)
    for si in range(qs.shape[0]):
        for qi in range(qs.shape[1]):
            k = str(qs[si, qi])
            if k not in uniq:
                uniq[k] = len(uniq)
            idx[si, qi] = uniq[k]
    return idx, len(uniq)


def bootstrap_T(e, fold, cluster, *, n_boot=N_BOOT, seed=STAT_SEED):
    """Resample unique QUERY IDS, with the gamma selection repeated inside.

    Not stratified by fold. Under per-seed validation a query can sit in
    several seeds, so stratifying by fold splits the very cluster the
    clustering exists to keep whole. A drawn query brings ALL of its cells,
    with the same multiplicity, wherever they fall.

    Fold sizes therefore vary between draws. That is accepted; only a draw
    that empties some (seed, fold) is rejected and redrawn, since a selection
    on nothing is undefined rather than merely noisy.

    NOTE what is NOT resampled: the head sets. Section 13.6.4 states the
    interval is conditional on the six selected H_sF and says so rather than
    claiming a direction for the resulting understatement.
    """
    a, fold = _as_hgsq(e, fold)
    cl = np.asarray(cluster)
    n_cl = int(cl.max()) + 1
    S, Q = a.shape[2], a.shape[3]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    b = 0
    rejected = 0
    while b < n_boot:
        draw = rng.integers(0, n_cl, size=n_cl)
        mult = np.bincount(draw, minlength=n_cl)      # times each query drawn
        w = mult[cl].astype(np.float64)               # per-cell weight
        ok = True
        g = {}
        for si in range(S):
            for f in (0, 1):
                # w[si], not w: `w` is [S, Q] and `fold[si]` is [Q],
                # so the bare form broadcast to a [S, Q] mask and then
                # indexed a [Q] row with it.
                m = (fold[si] == f) & (w[si] > 0)
                if not m.any():
                    ok = False
                    break
                wm = w[si][m]
                # fold f's head set, as in cross_fit
                g[(si, f)] = int(_pick(
                    (a[f][:, si, :][:, m] * wm).sum(axis=1) / wm.sum()))
            if not ok:
                break
        if not ok:
            rejected += 1
            if rejected > 10 * n_boot:
                raise RuntimeError(
                    "more than 10x n_boot draws emptied a (seed, fold); the "
                    "cluster structure cannot support this resample")
            continue
        tot = 0.0
        for si in range(S):
            for f in (0, 1):
                m = fold[si] == f
                tot += (a[1 - f, g[(si, 1 - f)], si, :][m] * w[si][m]).sum()
        out[b] = tot / (w.sum())
        b += 1
    return out


def procedure_sign_flip_p(e, fold, cluster, *, n_perm=N_PERM, seed=STAT_SEED,
                          chunk=20_000):
    """Two-sided sign-flip on the PROCEDURE, not on a fixed set of gammas.

    One sign per unique QUERY ID, applied to every cell that query has across
    whichever seeds drew it, then the fold-internal selection and the
    cross-fitted evaluation are REDONE inside the permuted data.

    Flipping the d of already-chosen gammas would test "with these six gammas
    held fixed, is d zero", which is not the quantity the interval covers --
    and GO requires both (section 14.1 C15).

    Head selection is NOT permuted, and does not need to be: score_margin
    comes from un-intervened forwards, so epsilon acts on a quantity the head
    ranking never sees. Fold-internal re-selection of H is the identity here.
    That is different in kind from the bootstrap's fixed H, which IS an
    approximation; the two are not to be conflated.

    The reduction is per (seed, fold) BLOCK -- six of them -- not the two
    matrix products the shared-axis form allowed. The cost is stated as blocks
    rather than as a multiplication count, because the count depends on how
    the ragged folds pack.
    """
    a, fold = _as_hgsq(e, fold)
    cl = np.asarray(cluster)
    n_cl = int(cl.max()) + 1
    S = a.shape[2]
    d0, _ = cross_fit(a, fold)
    T0 = float(d0.mean())
    rng = np.random.default_rng(seed)
    ge = 0
    done = 0
    n_cells = a.shape[2] * a.shape[3]
    while done < n_perm:
        m = min(chunk, n_perm - done)
        eps_cl = rng.choice(np.array([-1.0, 1.0]), size=(m, n_cl))
        eps = eps_cl[:, cl]                        # [m, S, Q] per-cell signs
        tot = np.zeros(m)
        for si in range(S):
            rows = {f: np.flatnonzero(fold[si] == f) for f in (0, 1)}
            gsel = {}
            for f in (0, 1):
                r = rows[f]
                # [m, G]: the permuted fold mean for every gamma, under
                # THIS fold's head set -- the same pairing cross_fit uses
                mm = np.einsum("mq,gq->mg", eps[:, si, r],
                               a[f][:, si, r]) / r.size
                gsel[f] = _pick(mm)
            for f in (0, 1):
                r = rows[f]
                pick = gsel[1 - f]                 # the OTHER fold's choice
                # ...and the OTHER fold's head set, index 1-f for both
                tot += (eps[:, si, r]
                        * a[1 - f][pick][:, si, :][:, r]).sum(axis=1)
        Tstar = tot / n_cells
        ge += int(np.count_nonzero(np.abs(Tstar) >= abs(T0) - 1e-12))
        done += m
    return (1.0 + ge) / (1.0 + n_perm), T0


def decide(T, lo, hi, p, *, delta_min=DELTA_MIN, alpha=ALPHA,
           status=VERIFIED, why_diagnostic=None):
    """Section 13.6.5. Mutually exclusive and exhaustive; no fourth branch.

    delta_min = 0.02 nats. ⚠ This is the SAME NUMERIC SCALE as section 9's
    non-inferiority margin, adopted as a symmetric engineering threshold -- it
    is NOT the logical converse of that margin. Section 9 says "gamma=1 is at
    most 0.02 nats WORSE than natural, so treat it as approximately
    non-inferior". "An improvement of at least 0.02 nats is the smallest worth
    building six baselines for" is a different proposition about a different
    direction; it merely borrows the scale, and borrowing a scale is a choice
    that has to be declared rather than derived.

    Raising delta_min makes STOP easier and GO harder.

    A DIAGNOSTIC run returns decision=None. A reduced bootstrap or a skipped
    freeze does not produce a GO or a STOP with a caveat printed above it --
    the field is absent, so nothing downstream can read a verdict that the
    configuration did not earn.
    """
    for name, v in (("T", T), ("lo", lo), ("hi", hi), ("p", p)):
        if isinstance(v, bool) or not isinstance(v, (int, float, np.floating,
                                                     np.integer)):
            raise TypeError(f"{name} must be a real number, got {v!r} "
                            f"({type(v).__name__}); bool is an int subclass "
                            "and would silently compare as 0/1")
        if not np.isfinite(float(v)):
            raise ValueError(f"{name} is {v}, which no comparison can decide")
    if lo > hi:
        raise ValueError(f"interval [{lo}, {hi}] is inverted")
    if status not in (VERIFIED, DIAGNOSTIC):
        raise ValueError(f"unknown status {status!r}")

    if hi < -delta_min and p < alpha and T < 0:
        d = "GO"
    elif lo > -delta_min:
        d = "STOP"
    else:
        d = "INCONCLUSIVE"

    out = {
        "status": status,
        "decision": d if status == VERIFIED else None,
        "provisional_decision": d,
        "T": float(T), "ci95": [float(lo), float(hi)], "p_signflip": float(p),
        "delta_min": float(delta_min),
        "go_condition": f"upper {hi:+.5f} < {-delta_min:+.5f} and p {p:.4g} < "
                        f"{alpha} and T {T:+.5f} < 0",
        "stop_condition": f"lower {lo:+.5f} > {-delta_min:+.5f}",
        "means": {
            "GO": "confident of at least delta_min improvement -- implement "
                  "the six baselines. NOT a result and not a claim about H4.",
            "STOP": "even the most favourable end falls short of delta_min -- "
                    "terminate the method line as an ENGINEERING decision. "
                    "This is not a preregistered negative result: H4 lives on "
                    "test and was never run.",
            "INCONCLUSIVE": "direction may be favourable but the interval "
                            "straddles delta_min. Validation is fixed at 144 "
                            "and cannot be enlarged without revising 2.1.",
        }[d],
    }
    if status == DIAGNOSTIC:
        out["why_diagnostic"] = list(why_diagnostic or ["unspecified"])
        out["means"] = (
            "DIAGNOSTIC: this configuration cannot produce a verdict. "
            "`decision` is null; `provisional_decision` is what the same "
            "numbers would have said under the registered configuration and "
            "must not be reported as the gate's outcome.")
    return out


# ==========================================================================
# input: prove this is the registered experiment
# ==========================================================================
REQUIRED = ("nll", "natural_nll", "acc", "natural_acc", "brier",
            "natural_brier", "query_ids", "class_idx", "gammas", "seeds")
META_REQUIRED = ("model", "task", "K", "dtype", "attn_implementation",
                 "carrier_impl", "carrier_sha256", "query_manifest_sha256",
                 "label_space_sha256",
                 # computed by the forward runner from the LIVE tokenizer; the
                 # only evidence available to a zero-GPU analysis that the ids
                 # were produced by the tokenizer the label space names
                 "tokenizer_provenance")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


REGISTERED_MODEL = "meta-llama/Llama-3.1-8B"
REGISTERED_TASK = "trec_fine_per_class"
REGISTERED_DTYPE = ("torch.bfloat16", "bfloat16", "bf16")
REGISTERED_K = 5


def load_viability_npz(path, *, query_manifest, label_space, carrier_json,
                       task=None, meta_path=None):
    """Read the validation run and PROVE it is the registered experiment.

    A self-consistent npz is not evidence. Ten classes, four queries each,
    arbitrary seeds and the wrong carrier implementation would satisfy every
    shape check and yield a perfectly formatted verdict. So the identity of the
    experiment is checked against the artifacts themselves -- the query
    manifest's own validation list, the label space, the recorded carrier hash
    -- rather than against fields the npz asserts about itself.
    """
    z = dict(np.load(path, allow_pickle=False))
    missing = [k for k in REQUIRED if k not in z]
    if missing:
        raise ValueError(
            f"{path}: missing {missing} (has {sorted(z)}). Section 13.6.5 "
            "requires out-of-fold accuracy and Brier to be reported alongside "
            "NLL, so they are inputs, not optional extras.")
    meta = json.loads(Path(meta_path or str(path).replace(".npz", ".json"))
                      .read_text(encoding="utf-8"))
    m_missing = [k for k in META_REQUIRED if meta.get(k) is None]
    if m_missing:
        raise ValueError(f"{path}: the meta records no {m_missing}; the run "
                         "cannot be attributed to a model, a carrier set or a "
                         "class space")

    bad = []
    # strict about TYPE, not only value: float(False) is 0.0 and
    # float("0.0") is 0.0, so a bool grid and a grid of strings both compared
    # equal to the registered one. The numbers would be right and the proof
    # that a runner wrote the file to schema would not.
    for _i, _g in enumerate(z["gammas"]):
        if isinstance(_g, (bool, np.bool_)) or not isinstance(_g,
                                                              numbers.Real):
            bad.append(
                f"gammas[{_i}] is {type(_g).__name__} {_g!r}, not a real "
                "number. float() would accept False as 0.0 and \"0.0\" as "
                "0.0, so the grid could match while the file was not written "
                "to schema.")
    gammas = tuple(float(g) for g in z["gammas"])
    if gammas != GAMMA_GRID:
        bad.append(f"gamma grid {gammas} is not the registered {GAMMA_GRID}")
    seeds = tuple(strict_int(s, f"seeds[{i}]")
                  for i, s in enumerate(z["seeds"]))
    if seeds != REGISTERED_SEEDS:
        bad.append(f"seeds {seeds} are not the registered {REGISTERED_SEEDS}")

    # THREE sets of 144, each aligned position-by-position with
    # validation_by_seed[s] -- 432 (seed, query) cells (section 13.6.4b).
    # Checking one set would have passed whenever seed 42 matched, whatever
    # the other two held.
    qarr = np.asarray(z["query_ids"])
    carr = np.asarray(z["class_idx"])
    man = json.loads(Path(query_manifest).read_text(encoding="utf-8"))
    vbs = man.get("validation_by_seed")
    if vbs is None:
        bad.append(
            "the query manifest has no 'validation_by_seed'; a single "
            "'validation' list predates section 14.0a and describes a draw "
            "that no longer exists")
        vbs = {}
    want_cls = []
    if qarr.ndim != 2 or carr.shape != qarr.shape:
        bad.append(
            f"query_ids {qarr.shape} and class_idx {carr.shape} must both be "
            "[seed, query]; a 1-D array is the pre-14.0a shared axis")
    elif qarr.shape[0] != len(seeds):
        bad.append(f"query_ids has {qarr.shape[0]} seed rows but the npz "
                   f"declares {len(seeds)} seeds")
    else:
        for si, sd in enumerate(seeds):
            rows_ = vbs.get(str(sd))
            if rows_ is None:
                bad.append(f"the manifest has no validation for seed {sd}")
                continue
            w_ids = [v["query_id"] for v in rows_]
            w_cls = [int(v["class_idx"]) for v in rows_]
            want_cls += w_cls
            got_ids = [str(q) for q in qarr[si]]
            if got_ids != w_ids:
                overlap = len(set(got_ids) & set(w_ids))
                bad.append(
                    f"seed {sd}: the {len(got_ids)} queries are not the "
                    f"manifest's {len(w_ids)} validation queries in order "
                    f"({overlap} ids in common). Positional indexing means a "
                    "reordering pairs each row with another query's gold.")
            elif [strict_int(c, f"class_idx[{si},{j}]")
                  for j, c in enumerate(carr[si])] != w_cls:
                bad.append(f"seed {sd}: class_idx disagrees with the "
                           "manifest's validation classes")
            if len(got_ids) != N_VALIDATION:
                bad.append(f"seed {sd}: {len(got_ids)} queries; section 2.1 "
                           f"fixes validation at {N_VALIDATION} PER SEED")
    ncls = len(set(int(c) for c in np.asarray(z["class_idx"]).ravel()))
    if ncls != N_ELIGIBLE:
        bad.append(f"{ncls} distinct classes; section 2.1 fixes the decision "
                   f"space at {N_ELIGIBLE}")

    if meta["carrier_impl"] != GROUP_IMPL:
        bad.append(
            f"carrier_impl {meta['carrier_impl']!r} is not {GROUP_IMPL!r}. The "
            "gate is defined on Method A-GROUP (section 3.1); Method A-head is "
            "a secondary implementation with its own gamma* and does not enter "
            "H4, so it cannot stand in here.")
    # Exact comparisons, not endswith/substring. "Llama-3.1-8B-Instruct" ends
    # with nothing useful and "not-bfloat16" contains "bfloat16"; a check that
    # a wrong value can satisfy is not a check.
    if str(meta["attn_implementation"]) != "eager":
        bad.append(f"attn_implementation {meta['attn_implementation']!r} is "
                   "not the registered 'eager' (section 2.2)")
    if str(meta["dtype"]) not in REGISTERED_DTYPE:
        bad.append(f"dtype {meta['dtype']!r} is not one of the registered "
                   f"spellings of bfloat16 {REGISTERED_DTYPE}")
    if strict_int(meta["K"], "meta K") != REGISTERED_K:
        bad.append(f"K={meta['K']}, registered is {REGISTERED_K}")
    if str(meta["model"]) != REGISTERED_MODEL:
        bad.append(f"model {meta['model']!r} != the registered "
                   f"{REGISTERED_MODEL!r}")
    if str(meta["task"]) != REGISTERED_TASK:
        bad.append(f"task {meta['task']!r} != the registered "
                   f"{REGISTERED_TASK!r}")
    for key, path_ in (("query_manifest_sha256", query_manifest),
                       ("label_space_sha256", label_space),
                       # THE CARRIER. Previously required in the meta and never
                       # checked against anything, so a run on the wrong heads
                       # with a plausible hash typed in reached VERIFIED.
                       ("carrier_sha256", carrier_json)):
        got = _sha(path_)
        if meta[key] != got:
            bad.append(f"{key}: the run recorded {str(meta[key])[:12]}, but "
                       f"{Path(path_).name} on disk hashes to {got[:12]}")

    # The label space, loaded rather than merely hashed, so its internal
    # consistency and its class space are re-derived.
    #
    # ⚠ NOT the tokenizer provenance. `FrozenLabelSpace.load` only recomputes
    # that when handed a live tokenizer, and this gate is a zero-GPU analysis
    # that has none. Claiming otherwise would be describing a check that does
    # not run. What the tokenizer IS verified against is the forward runner:
    # it holds the live tokenizer, records the provenance it computed, and the
    # comparison happens below.
    try:
        from tools.label_space import FrozenLabelSpace, provenance_mismatch
        ls = FrozenLabelSpace.load(label_space, model=str(meta["model"]),
                                   query_manifest=query_manifest)
        if [int(c) for c in ls.eligible_classes] != sorted(set(want_cls)):
            bad.append("the label space's eligible classes are not the "
                       "manifest's validation classes")
        if len(ls.candidate_token_ids) != N_ELIGIBLE:
            bad.append(f"the label space has {len(ls.candidate_token_ids)} "
                       f"candidates, registered is {N_ELIGIBLE}")
        rp = meta.get("tokenizer_provenance")
        if not rp:
            bad.append(
                "the run records no tokenizer_provenance. The forward runner "
                "had the live tokenizer and is the only place that can prove "
                "it was the one that produced these ids; without that record "
                "this gate can check the label space's internal consistency "
                "and nothing about the tokenizer.")
        else:
            hard = [b for b in provenance_mismatch(rp, ls.data["provenance"])
                    if not b.startswith("[advisory]")]
            bad += [f"tokenizer provenance recorded by the forward runner "
                    f"disagrees with the label space: {b}" for b in hard]
    except Exception as e:                                  # noqa: BLE001
        bad.append(f"the label space did not load and verify: {e}")

    # The pool that used to be validated here is gone (section 14.0a):
    # discovery now runs on the SAME 144 validation queries this npz is scored
    # on, so "was it discovered on the right data" is answered by the carrier's
    # query_manifest_sha256 below rather than by a separate artifact.

    # The carrier artifact, against the schema frozen in tools/carrier_schema.py
    # -- which re-derives the top-8 from the recorded ranking rather than
    # counting eight entries. Eight duplicated or invented pairs used to pass.
    try:
        from tools.carrier_schema import validate as validate_carrier
        carr = json.loads(Path(carrier_json).read_text(encoding="utf-8"))
        # A FOLD BUNDLE OR A SINGLE CARRIER, because the run decides which.
        #
        # The flat schema predates 13.6.3's fold axis: it describes ONE head
        # set, and a run with a head axis used SIX. The bundle is the artifact
        # that pins all six, so refusing it was the identity check demanding
        # the wrong file -- the gate would only accept a carrier the
        # experiment did not use.
        #
        # This changes WHICH ARTIFACT SCHEMA is accepted as proof of identity.
        # It changes no criterion: delta_min, the fold split, the estimator
        # and the three-state boundaries are untouched, and it was made before
        # any verdict was seen (14.0b-18).
        if "bundle_version" in carr:
            from tools import carrier_bundle as CB
            man = json.loads(Path(query_manifest).read_text(encoding="utf-8"))
            vbs = man["validation_by_seed"]
            ids = {s: [r["query_id"] for r in vbs[str(s)]]
                   for s in REGISTERED_SEEDS}
            cls = {s: {r["query_id"]: int(r["class_idx"]) for r in vbs[str(s)]}
                   for s in REGISTERED_SEEDS}
            from tools.discover_carriers import fold_labels
            bad += [f"{Path(carrier_json).name}: {b}" for b in CB.bundle_faults(
                carr, scope=CB.SCOPE_FOLD, validation_by_seed=ids,
                fold_by_seed={s: fold_labels(ids[s], cls[s])
                              for s in REGISTERED_SEEDS},
                query_manifest_sha256=_sha(query_manifest),
                model=str(meta["model"]))]
        else:
            # The manifest hash is PASSED, not omitted. It was left empty when
            # the field was renamed, so validate_carrier compared the carrier
            # against nothing and a carrier discovered on another validation
            # set reached VERIFIED. Hashed from DISK: comparing the carrier's
            # field with the meta's would only show the two agreed.
            bad += [f"{Path(carrier_json).name}: {b}"
                    for b in validate_carrier(
                        carr, model=str(meta["model"]),
                        query_manifest_sha256=_sha(query_manifest))]
    except Exception as e:                                  # noqa: BLE001
        bad.append(f"the carrier artifact did not parse: {e}")

    # THE TASK-BACKED REBUILD, actually invoked. Its absence is what made
    # --task fail-open: run() granted VERIFIED for any non-empty string while
    # nothing called the rebuild, so a hand-authored but self-consistent
    # manifest passed. Failures append to `bad` like every other fault.
    if task:
        try:
            bad += task_rebuild_inputs(task, query_manifest)
        except Exception as e:                              # noqa: BLE001
            bad.append(f"the task-backed rebuild could not run: {e}")

    nll, nat = np.asarray(z["nll"], float), np.asarray(z["natural_nll"], float)
    # N_VALIDATION, not the array's own length: an input under
    # verification must not define the size it is checked against.
    nq, ns = N_VALIDATION, len(seeds)
    # EVERY metric, not just NLL. accuracy and Brier were checked only for
    # finiteness and the gamma=0 identity, so a transposed or truncated
    # array would have reached the auxiliary report.
    # FOUR AXES since 13.6.3. The head axis is length 2 -- the set selected
    # on fold 0 and the one selected on fold 1 -- because each fold is
    # evaluated under the OTHER fold's heads. A three-axis array is not a
    # smaller version of this experiment, it is a different one, so it is
    # named as such rather than accepted and broadcast.
    want = (2, len(gammas), ns, nq)
    for name in ("nll", "acc", "brier"):
        got = np.asarray(z[name], float).shape
        if got != want:
            bad.append(
                f"{name} has shape {got}, expected {want} = "
                "(head, gamma, seed, query)"
                + ("; this is the pre-13.6.3 three-axis layout, which has "
                   "one head set for both folds" if len(got) == 3 else ""))
    for name in ("natural_nll", "natural_acc", "natural_brier"):
        got = np.asarray(z[name], float).shape
        if got != (ns, nq):
            bad.append(f"{name} has shape {got}, expected {(ns, nq)} = "
                       "(seed, query)")
    for name in ("nll", "natural_nll", "acc", "natural_acc", "brier",
                 "natural_brier"):
        a = np.asarray(z[name], float)
        if not np.all(np.isfinite(a)):
            bad.append(f"{name} holds {int((~np.isfinite(a)).sum())} "
                       "non-finite values")
            continue
        # THE DOMAIN, not just finiteness. A negative NLL is the dangerous
        # one: e = nll - natural goes hugely negative, T with it, and GO
        # requires precisely that sign -- so an impossible number reaches a
        # decision with nothing in the way. These bounds are the definitions,
        # not tolerances: NLL is -log p with p <= 1; Brier sums
        # (p_c - y_c)^2 over classes and maxes at 2 when all mass is on one
        # wrong class; accuracy is a mean of indicators.
        lo, hi = DOMAIN["nll" if name.endswith("nll")
                        else "acc" if name.endswith("acc") else "brier"]
        out = (a < lo) | (a > hi)
        if not out.any() and name in INDICATOR:
            out = a != np.round(a)
            if out.any():
                bad.append(
                    f"{name}: {int(out.sum())} of {a.size} values are not 0 "
                    "or 1. Each cell is ONE query's correctness, so it is an "
                    "indicator; 0.5 is inside [0, 1] and is not something a "
                    "single query can be.")
                continue
        if out.any():
            bad.append(
                f"{name}: {int(out.sum())} of {a.size} values lie outside "
                f"[{lo}, {hi}] (min {a.min():.4g}, max {a.max():.4g}). That "
                "is not a number this metric can take, so no comparison "
                "involving it means anything.")
    if bad:
        raise ValueError(f"{path} is not the registered viability experiment:\n"
                         "  " + "\n  ".join(bad))

    # gamma=0 is the identity operator: a free check on the whole forward path.
    i0 = gammas.index(0.0)
    for name, arm in (("nll", "natural_nll"), ("acc", "natural_acc"),
                      ("brier", "natural_brier")):
        a, b = np.asarray(z[name], float), np.asarray(z[arm], float)
        # [:, i0], not [i0]: under [head, gamma, seed, query] a bare [i0]
        # takes a HEAD slab. gamma=0 must be the identity under BOTH head
        # sets -- writing nothing is writing nothing whichever heads were
        # chosen -- so the check covers the whole 2 x seed x query block.
        got = a[:, i0]
        want_ = np.broadcast_to(b, got.shape)
        if not np.array_equal(got, want_):
            n = int((got != want_).sum())
            raise ValueError(
                f"{path}: the gamma=0 slice of {name} differs from {arm} in {n} "
                f"of {want_.size} cells (max |delta| "
                f"{np.abs(got - want_).max():.3e}). gamma=0 is the IDENTITY "
                "operator (section 3): v_i = v_i^0 + 0*Delta_i. If these "
                "differ, the write path is doing something the preregistration "
                "does not describe and no number here is interpretable. This "
                "is a runner bug, not a tolerance to widen.")
    return z, meta, gammas


def validation_row_faults(drawn, want, seed):
    """Compare a rebuilt validation draw with what the manifest claims.

    PURE: rows in, faults out, no task and no download -- which is what lets
    it be armed. Inside `task_rebuild_inputs` it had no direct arm at all,
    because the only test that reached that function stubbed the whole thing.

    Three checks, and the third is the one a permutation cannot survive:

      * the ids, in order -- positional indexing means a reordering pairs
        each row with another query's gold;
      * every other field, row by row. class_idx and source_text come from
        the manifest and were compared against nothing, so permuting them
        among the ids -- leaving the id list and the reservation set
        untouched -- rebuilt clean, and the gold class decides the fold;
      * query_id == content_hash(class_idx, source_text), RECOMPUTED. The id
        is defined as that hash, so the three fields cannot disagree without
        the hash disagreeing with them.
    """
    # imported HERE, not inherited: content_hash is a function-level import
    # in task_rebuild_inputs, so extracting this comparison out of it
    # reintroduced exactly the unbound name the gate was built for -- and the
    # gate reported it on the first run after the split.
    from tools.prereg_ids import content_hash
    faults = []
    got_ids = [v["query_id"] for v in drawn]
    want_ids = [v["query_id"] for v in want]
    if got_ids != want_ids:
        overlap = len(set(got_ids) & set(want_ids))
        faults.append(
            f"seed {seed}: the manifest's validation set does not rebuild "
            f"from the train split ({overlap} of {len(got_ids)} ids in "
            "common, order included). A manifest can be internally "
            "consistent and still describe a draw that never happened.")
    else:
        # CANONICAL types, not str(). Comparing str(3) with str("3")
        # accepted a manifest whose class_idx was the STRING "3" -- which
        # contradicts the rule that a file's identity fields must already be
        # numbers (fe9ab5c). strict_int raises on the string, and the
        # exception is turned into a fault rather than escaping.
        wrong = []
        for i, (g, w) in enumerate(zip(drawn, want)):
            try:
                if strict_int(w["class_idx"], f"row {i} class_idx") != int(
                        g["class_idx"]):
                    wrong.append((i, "class_idx"))
            except (ValueError, KeyError, TypeError) as e:
                faults.append(f"seed {seed}: row {i} class_idx is not a "
                              f"usable integer: {e}")
            if str(g.get("source_text")) != str(w.get("source_text")):
                wrong.append((i, "source_text"))
        if wrong:
            faults.append(
                f"seed {seed}: {len(wrong)} field(s) differ from the rebuilt "
                f"draw although every query_id matches (first: row "
                f"{wrong[0][0]}, {wrong[0][1]}). The ids agreeing says the "
                "right queries were named, not that they carry the right "
                "gold or the right text.")
    broken = []
    for i, w in enumerate(want):
        try:
            c = strict_int(w["class_idx"], f"row {i} class_idx")
        except (ValueError, KeyError, TypeError):
            continue                     # already reported above
        if content_hash(c, w["source_text"]) != w["query_id"]:
            broken.append(i)
    if broken:
        faults.append(
            f"seed {seed}: {len(broken)} row(s) whose query_id is not "
            f"content_hash(class_idx, source_text) (first at {broken[0]}). "
            "The id is DEFINED as that hash, so a row where they disagree "
            "was assembled, not drawn.")
    return faults


def task_rebuild_inputs(task_name, query_manifest):
    """Rebuild the three validation draws and the three prefixes from TRAIN.

    Returns a list of faults. This is the only check that a manifest was not
    simply authored: every hash inside it can be made self-consistent, but the
    draws themselves are a deterministic function of the train split and the
    seed, so they either reproduce or they do not.

    ⚠ It used to return (rows, prefix_hashes) and HAD NO CALLER, while run()
    granted VERIFIED for any non-empty --task string. The rebuild was a
    function nobody invoked guarding a flag nobody honoured.
    """
    # content_hash was used below and imported nowhere -- neither here nor
    # at module level -- so every --task run failed on NameError inside the
    # try/except and could never reach VERIFIED. ast.parse cannot see this.
    from tools.prereg_ids import content_hash
    from tools.prereg_task import (draw_validation_from_train,
                                   eligible_classes, load_task,
                                   prefix_demo_rows, train_rows)
    man = json.loads(Path(query_manifest).read_text(encoding="utf-8"))
    vbs = man.get("validation_by_seed") or {}
    task = load_task(task_name, 5, REGISTERED_SEEDS[0])
    # DERIVED from the task, not read from the manifest. Taking the class
    # space from the file under verification lets a manifest, a label space
    # and three validation sets all be swapped to a different 36 classes
    # together and still rebuild -- the rebuild would be checking the manifest
    # against itself. The rule (train AND test AND >= 15 deduplicated train
    # rows, section 2.1) is what the class space actually is.
    derived = [int(c) for c in eligible_classes(task, task_name)]
    claimed = [int(c) for c in man.get("eligible_classes", [])]
    faults = []
    if claimed != derived:
        faults.append(
            f"the manifest claims {len(claimed)} eligible classes but the "
            f"task's rule derives {len(derived)}"
            + (f"; first difference at position "
               f"{next(i for i, (a, b) in enumerate(zip(claimed, derived)) if a != b)}"
               if claimed and derived
               and len(claimed) == len(derived) else "")
            + ". Every downstream draw is conditioned on this list.")
    eligible = derived
    rows = train_rows(task, task_name)
    for sd in REGISTERED_SEEDS:
        want = vbs.get(str(sd))
        if want is None:
            faults.append(f"the manifest has no validation for seed {sd}")
            continue
        drawn, _diag = draw_validation_from_train(
            rows, eligible, set(), sd, per_class=VALIDATION_PER_CLASS)
        faults += validation_row_faults(drawn, want, sd)
        # and the prefix it implies must not contain that seed's own queries
        reserved_s = {(int(v["class_idx"]), v["source_text"]) for v in want}
        demos = prefix_demo_rows(task_name, 5, [sd],
                                 excluded_docs=reserved_s)[sd]
        ph = {content_hash(c, t) for c, t in demos}
        clash = ph & {v["query_id"] for v in want}
        if clash:
            faults.append(
                f"seed {sd}: {len(clash)} of its validation queries appear in "
                "its own prefix; the reservation did not take effect")
    return faults


def require_freeze(path):
    """Section 13.6.1(2): the opponents must be frozen before this runs.

    TWO questions, and the earlier version asked only the second.

      1. Is this freeze VALID? A file written by the earlier fail-open tool --
         or authored by hand -- can record every repository as unreadable and
         still be perfectly self-consistent. A consumer that only asks "has
         anything changed since this was written" accepts it. So the validity
         conditions are re-checked against the RECORDED content here, by the
         code about to rely on it, rather than trusted because some earlier
         run is assumed to have checked them.
      2. Has anything drifted since?
    """
    from tools.baselines.freeze_baseline_spec import (build_freeze, compare,
                                                      validate_freeze)
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"{p}: the six baselines are not frozen. Section 13.6.1 requires "
            "the freeze BEFORE this gate, so that a visible viability result "
            "cannot lead to adjusting the opponents. Run "
            "`python tools/baselines/freeze_baseline_spec.py --write "
            "--stage spec` first.")
    rec = json.loads(p.read_text(encoding="utf-8"))
    invalid = validate_freeze(rec)
    if invalid:
        raise SystemExit(
            f"{p} is not a VALID freeze, independently of drift:\n  "
            + "\n  ".join(invalid[:8])
            + "\nIt pins nothing, so it cannot license a verdict. Regenerate "
              "it with the current tool.")
    drift = compare(rec, build_freeze(rec.get("stage", "spec")))
    if drift:
        raise SystemExit(
            f"{p}: {len(drift)} item(s) drifted since the freeze:\n  "
            + "\n  ".join(drift[:8])
            + "\nThe gate will not run against an unfrozen opponent set.")
    return _sha(p)


# ==========================================================================
# CLI
# ==========================================================================
def run(path, *, query_manifest, label_space, carrier_json,
        task=None, freeze=None, n_boot=N_BOOT,
        n_perm=N_PERM, delta_min=DELTA_MIN, meta_path=None):
    why = []
    freeze_sha = None
    if freeze:
        freeze_sha = require_freeze(freeze)
    else:
        why.append("--no-require-freeze: the opponents were not shown to be "
                   "frozen, so a visible result could still change them")
    if n_boot != N_BOOT:
        why.append(f"--n-boot {n_boot} != the registered {N_BOOT}")
    if n_perm != N_PERM:
        why.append(f"--n-perm {n_perm} != the registered {N_PERM}")
    if delta_min != DELTA_MIN:
        why.append(f"--delta-min {delta_min} != the registered {DELTA_MIN}")
    if not task:
        why.append(
            "--task was not given, so the validation sets were checked only "
            "for self-consistency against the manifest. A forged manifest "
            "with every internal hash updated passes every check except a "
            "rebuild from the train split.")
    status = DIAGNOSTIC if why else VERIFIED

    z, meta, gammas = load_viability_npz(
        path, query_manifest=query_manifest, label_space=label_space,
        carrier_json=carrier_json,
        task=task, meta_path=meta_path)
    # [S, Q] throughout: each seed has its own queries, its own folds, and
    # its own six selections. `cluster` maps every cell to its query id, so a
    # query drawn by two seeds moves as one unit under both inferences.
    fold = assign_folds(z["query_ids"], z["class_idx"])
    cluster, n_clusters = cluster_index(z["query_ids"])
    ebar = seed_mean_effect(z["nll"], z["natural_nll"])
    d, chosen = cross_fit(ebar, fold, gammas)
    T = float(d.mean())

    p, T_perm = procedure_sign_flip_p(ebar, fold, cluster, n_perm=n_perm)
    if abs(T_perm - T) > 1e-12:
        raise AssertionError(
            f"the permutation's identity case gives T={T_perm:.12f} but "
            f"cross_fit gives {T:.12f}. The reduced form and the direct "
            "computation have drifted apart; neither number is usable.")
    boot = bootstrap_T(ebar, fold, cluster, n_boot=n_boot)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    verdict = decide(T, lo, hi, p, delta_min=delta_min, status=status,
                     why_diagnostic=why)

    aux = {}
    for name, arm in (("accuracy", "natural_acc"), ("brier", "natural_brier")):
        key = "acc" if name == "accuracy" else "brier"
        eb = seed_mean_effect(z[key], z[arm])
        # AT THE GAMMAS NLL CHOSE, not reselected. Calling cross_fit again
        # would make each auxiliary its own selection procedure, and since the
        # rule is argmin, the accuracy figure would pick the gamma with the
        # WORST accuracy -- reported as though it described the configuration
        # the gate actually decided on.
        da = apply_choice(eb, fold, chosen["_idx"])
        aux[name] = {"T_out_of_fold": float(da.mean()),
                     "note": "AUXILIARY (section 13.6.5): reported at the "
                             "gammas the NLL cross-fit selected, not "
                             "reselected; not part of the decision"}
    # the in-sample gamma curve: descriptive, optimistically biased, and
    # reported so "weak effect" can be told from "noisy selection"
    # (section 13.6.3). Over ALL cells, which is what T averages over.
    # PER SEED. Pooling the cells gives one argmin describing a selection
    # nobody makes: gamma is chosen per seed (six times), so a single pooled
    # curve is not the in-sample counterpart of anything the gate computes.
    # AVERAGED OVER THE HEAD AXIS for the descriptive curve only. Each fold
    # is really evaluated under one specific head set (13.6.3), so no single
    # curve describes the criterion -- which is why this one is labelled
    # DESCRIPTIVE and the selection below reads cross_fit's six choices
    # instead. Averaging keeps the printed curve from silently being head
    # source 0's alone.
    in_sample_by_seed = ebar.mean(axis=(0, 3))         # [gamma, seed]
    in_sample = in_sample_by_seed.mean(axis=1)         # kept for the report

    print("=" * 78)
    print("L3.1 METHOD A VIABILITY GATE (prereg 13.6) -- validation only")
    print("=" * 78)
    print(f"  {meta['model']}  {meta['task']} K={meta['K']}  "
          f"{meta['dtype']}/{meta['attn_implementation']}  "
          f"carrier {meta['carrier_impl']} {meta['carrier_sha256'][:12]}")
    # NEGATIVE INDICES, because ebar is [gamma, seed, query] OR
    # [head, gamma, seed, query] and this line was written before the head
    # axis existed. With shape[1]/shape[2] it printed "5 seeds x 3 queries =
    # 15 cells" for a 3-seed, 144-query run -- the gamma axis read as seeds
    # and the seed axis as queries. Cosmetic, in that every number below it
    # was right, and worth fixing anyway: a header that miscounts the design
    # invites doubt about the numbers that follow.
    print(f"  {ebar.shape[-2]} seeds x {ebar.shape[-1]} queries = "
          f"{ebar.shape[-2] * ebar.shape[-1]} cells over {n_clusters} "
          f"distinct queries, seeds {[int(s) for s in z['seeds']]}")
    for _si, _sd in enumerate(z["seeds"]):
        print(f"    seed {_sd}: folds "
              f"{int((fold[_si] == 0).sum())}/{int((fold[_si] == 1).sum())}")
    print(f"  [PASS] the run is the registered experiment (manifest, label "
          f"space, seeds, class space, carrier implementation)")
    print(f"  [PASS] gamma=0 reproduces natural bitwise on NLL, accuracy and "
          f"Brier")
    if freeze_sha:
        print(f"  [PASS] opponents frozen, {freeze_sha[:12]}")

    print()
    for _si, _sd in enumerate(z["seeds"]):
        _g0 = chosen[f"seed{_si}_fold0_selected"]
        _g1 = chosen[f"seed{_si}_fold1_selected"]
        print(f"  seed {_sd}: fold 0 -> gamma {_g0} (evaluates fold 1), "
              f"fold 1 -> gamma {_g1} (evaluates fold 0)")
    _sel = {chosen[f"seed{_si}_fold{_f}_selected"]
            for _si in range(len(z["seeds"])) for _f in (0, 1)}
    if len(_sel) > 1:
        print("    the folds disagree -- cross-fitting working, not a defect, "
              "but selection noise is part of what is being measured")

    # THREE curves, one per seed, because gamma is selected per seed. The
    # pooled version printed one argmin describing a selection nobody makes,
    # and called 432 cells "all 144".
    print("\n  in-sample gamma curves, PER SEED (DESCRIPTIVE, optimistically "
          "biased, NOT the criterion):")
    # _pick, not np.argmin: the registered rule takes the SMALLEST gamma
    # within 1e-8 of the minimum (section 5(4)). np.argmin takes the numeric
    # minimum, so a 1e-12 improvement at gamma=0.25 was reported as the
    # in-sample choice where the rule selects 0.0 -- the descriptive curve
    # and the actual selection disagreeing about the same array.
    best_by_seed = [int(_pick(in_sample_by_seed[:, si]))
                    for si in range(len(z["seeds"]))]
    for si, sd in enumerate(z["seeds"]):
        marks = "".join(
            f"   gamma {g:<5} {in_sample_by_seed[gi, si]:+.6f}"
            + ("*" if gi == best_by_seed[si] else " ")
            for gi, g in enumerate(gammas))
        print(f"    seed {sd}:{marks}")
    print(f"    in-sample argmin per seed: "
          f"{[gammas[b] for b in best_by_seed]}"
          + ("   (they DISAGREE, which is what per-seed selection allows)"
             if len(set(best_by_seed)) > 1 else "   (all three agree)"))
    best = int(_pick(in_sample))

    print("\n  OUT-OF-FOLD (the criterion)")
    print(f"    T = mean_q d_q       {T:+.6f} nats")
    print(f"    95% bootstrap CI     [{lo:+.6f}, {hi:+.6f}]   ({n_boot} draws, "
          "gamma reselected in each)")
    print(f"    sign-flip p          {p:.4g}   ({n_perm} permutations, "
          "gamma RESELECTED in each -- same estimand as the CI)")
    print(f"    delta_min            {delta_min} nats (section 9's scale, "
          "adopted symmetrically; see `decide`)")
    print("\n  auxiliary, reported but NOT part of the decision:")
    for k, v in aux.items():
        print(f"    out-of-fold {k:<9} {v['T_out_of_fold']:+.6f}")

    print(f"\n  STATUS: {verdict['status']}")
    if verdict["status"] == DIAGNOSTIC:
        for w in why:
            print(f"    - {w}")
        print(f"    decision is null. The same numbers would have said "
              f"{verdict['provisional_decision']} under the registered "
              "configuration; that is not the gate's outcome.")
    else:
        print(f"  VERDICT: {verdict['decision']}")
        print(f"    {verdict['means']}")
    return verdict, {"chosen": {k: v for k, v in chosen.items()
                                if not k.startswith("_")},
                     "in_sample_mean_dnll_by_seed": {
                         str(sd): in_sample_by_seed[:, si].tolist()
                         for si, sd in enumerate(z["seeds"])},
                     "in_sample_argmin_by_seed": {
                         str(sd): gammas[best_by_seed[si]]
                         for si, sd in enumerate(z["seeds"])},
                     "auxiliary": aux,
                     "freeze_sha256": freeze_sha,
                     "run_meta": meta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--viability-npz")
    ap.add_argument("--viability-meta", default=None)
    ap.add_argument("--query-manifest")
    ap.add_argument("--label-space")
    ap.add_argument("--task", default=None,
                    help="enables the task-backed rebuild of the validation "
                         "sets; without it the run is DIAGNOSTIC")
    # The retired --discovery-split flag is REMOVED, not kept as a no-op:
    # argparse then refuses an old command line outright instead of accepting
    # it and silently ignoring the argument it names.
    ap.add_argument("--carrier-json",
                    help="the frozen Method A carriers; hashed and "
                         "compared against the run's recorded value")
    ap.add_argument("--freeze", default="results/baseline_spec_freeze.json")
    ap.add_argument("--no-require-freeze", action="store_true",
                    help="DIAGNOSTIC only -- the verdict becomes null")
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help="DIAGNOSTIC unless the registered 10000")
    ap.add_argument("--n-perm", type=int, default=N_PERM,
                    help="DIAGNOSTIC unless the registered 1000000")
    ap.add_argument("--delta-min", type=float, default=DELTA_MIN)
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        from tools.test_method_a_viability import main as fixtures
        return fixtures()
    for need in ("viability_npz", "query_manifest", "label_space",
                 "carrier_json"):
        if not getattr(args, need):
            raise SystemExit(f"--{need.replace('_', '-')} is required "
                             "(or --self-test)")
    verdict, extra = run(
        args.viability_npz, query_manifest=args.query_manifest,
        label_space=args.label_space, carrier_json=args.carrier_json,
        task=args.task,
        meta_path=args.viability_meta,
        freeze=None if args.no_require_freeze else args.freeze,
        n_boot=args.n_boot, n_perm=args.n_perm, delta_min=args.delta_min)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"spec": "prereg_method_A.md section 13.6",
                        "source": str(args.viability_npz),
                        **verdict, **extra}, indent=2), encoding="utf-8")
        print(f"\n  [output] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
