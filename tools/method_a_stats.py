"""Sections 6.1, 6.2 and 9 as arithmetic: metrics and the ONE confirmatory test.

Every constant here is preregistered, so they are module-level defaults rather
than call-site arguments. A seed or a resample count passed in at the call site
is a number someone can change after seeing a result; a default that the fixture
asserts is not.

    N_PERMUTATIONS = 1_000_000     sign flips        (section 6.2)
    N_BOOTSTRAP    = 10_000        query resamples   (section 6.2)
    STAT_SEED      = 20260826      both              (section 6.2)
    NLL_MARGIN     = 0.02 nat      one-sided upper   (section 9)
    ACC_MARGIN     = -0.03         one-sided lower   (section 9)

TWO REGIMES, BECAUSE THE TWO SETTINGS SAMPLE QUERIES DIFFERENTLY.

  L2, the mechanism setting: one query set common to the three demo seeds. The
  same query is observed under all three, so `paired_difference_common`
  averages the seeds INSIDE the query and the unit of analysis is the query.

  L3.1, the method main table and H6: 250 queries are drawn per demo seed,
  independently, so seed 42's query 17 and seed 43's query 17 are different
  questions and cannot be paired. The estimand is

      T = (1/3) sum_s dbar_s ,   dbar_s = (1/250) sum_{q in Q_s} d_{qs},

  the mean of the three per-seed means, and the confirmatory test is ONE
  aggregate test on T -- not three tests that all have to pass.

THE CLUSTER IS THE UNIQUE query_id, AND THAT IS THE WHOLE POINT. The three
draws come from one pool, so they overlap by construction. A query drawn by two
seeds contributes two differences that are not independent observations: same
text, same gold label, different prefix. Treating the 750 (query, seed) cells as
750 independent units would understate the variance and inflate significance.

So the randomisation and the bootstrap both operate on unique query_ids. Writing
T as a sum over clusters makes this exact:

      T = sum_c w_c ,   w_c = (1/S) sum_{s containing c} d_{c,s} / n_s

-- every seed in which query c appears contributes to the SAME w_c, so a
sign flip or a bootstrap draw moves all of that query's observations together.
`cluster_weights` builds w, and nothing downstream ever sees the flat 750.

ONE AGGREGATE TEST, NOT A PER-SEED CONJUNCTION. Requiring each seed to be
individually significant would be a different and much stricter claim, and it is
not what is preregistered: the claim is about the average effect across the
three fixed prefix seeds. Per-seed means, intervals and directions are reported
in full, but descriptively -- they are not gates. There is deliberately no
`max(per_seed_p)` anywhere in this module.

WHAT THE INFERENCE SUPPORTS. "The average effect given these three
preregistered prefix seeds." Three seeds do not estimate a seed population, and
the seed-to-seed SD is reported as a descriptive spread, not as a standard error
for generalising to unseen prefixes.

DIRECTION IS A SEPARATE CONJUNCT. Section 6.2 requires the point estimate to
move the right way IN ADDITION to a significant two-sided p. Both `decide` and
`decide_aggregate` return the two separately and refuse to collapse them,
because a two-sided test is symmetric and would otherwise hand a "pass" to a
large effect in the wrong direction.

THE CANDIDATE SPACE IS THE ELIGIBLE CLASSES -- 36 on TREC-fine. Every metric
here normalizes over the candidate set it is given, and that set comes from the
manifest: the classes present in both splits with at least 15 deduplicated
training examples. The task as preregistered is closed-set over those; the rest
are outside it by definition of the task. (This is a statement about the task
definition, not about what an intervention can do -- a V write changes logits
across the whole vocabulary.) The candidate set is a required argument
everywhere for this reason: there is no default class space to get wrong.
"""

from __future__ import annotations

import json
import numbers
from pathlib import Path

import numpy as np


def _is_number(x):
    """A real number that is NOT a bool.

    `isinstance(True, int)` is True in Python, so a plain numeric check accepts
    `T=True, p=True, ci95=[True, True]` as a complete result. Booleans are
    excluded explicitly, and `numbers.Real` is used so numpy floats still pass.
    """
    return (isinstance(x, numbers.Real)
            and not isinstance(x, (bool, np.bool_)))

N_PERMUTATIONS = 1_000_000
N_BOOTSTRAP = 10_000
STAT_SEED = 20260826
ALPHA = 0.05
NLL_MARGIN = 0.02        # section 9: one-sided 95% UPPER bound must be below
ACC_MARGIN = -0.03       # section 9: one-sided 95% LOWER bound must be above


# --------------------------------------------------------------------------
# section 6.1 metrics
# --------------------------------------------------------------------------
# EVERY function below indexes FULL-VOCABULARY logits with LABEL TOKEN IDS, and
# the gold argument is likewise a token id. The parameter names say so, because
# class indices and token ids are both small integers: on TREC-fine the classes
# are 0..49 and the label tokens are scattered vocabulary positions, so passing
# class indices selects perfectly valid but completely wrong columns and nothing
# raises. Callers working in classes should use CandidateSpace, which owns the
# mapping and never exposes a raw column index.

def candidate_nll(logits, candidate_token_ids, gold_token_id):
    """-z_gold + logsumexp over the CANDIDATE label tokens (section 6.1).

    The normalizer runs over the preregistered candidate tokens: not the full
    vocabulary, not every label token in the header, and not TL-alone logits.
    Those give different numbers, and the project has already produced a
    spurious result by mixing two class spaces (working rules rule 10), so the
    candidate set is required rather than defaulted.
    """
    z = np.asarray(logits, dtype=np.float64)
    cols = np.asarray(candidate_token_ids, dtype=int)
    if int(gold_token_id) not in set(cols.tolist()):
        raise ValueError(
            f"gold token {gold_token_id} is not among the {len(cols)} candidate "
            "tokens. Section 6.1 normalizes over the candidate tokens; a gold "
            "outside that set makes the NLL undefined rather than large. (If "
            "this looks like a class index rather than a token id, that is the "
            "bug -- see CandidateSpace.)")
    zc = z[..., cols]
    m = zc.max(axis=-1, keepdims=True)
    lse = np.squeeze(m, -1) + np.log(np.exp(zc - m).sum(axis=-1))
    return lse - z[..., int(gold_token_id)]


def candidate_argmax_token(logits, candidate_token_ids):
    """argmax restricted to the candidate tokens, returned as a TOKEN id."""
    cols = np.asarray(candidate_token_ids, dtype=int)
    return cols[np.asarray(logits)[..., cols].argmax(axis=-1)]


def candidate_probs(logits, candidate_token_ids):
    z = np.asarray(logits, dtype=np.float64)[
        ..., np.asarray(candidate_token_ids, int)]
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def brier(logits, candidate_token_ids, gold_token_id):
    """Multiclass Brier over the candidate simplex."""
    p = candidate_probs(logits, candidate_token_ids)
    cols = list(np.asarray(candidate_token_ids, int))
    if int(gold_token_id) not in cols:
        raise ValueError(f"gold token {gold_token_id} is not a candidate")
    onehot = np.zeros_like(p)
    onehot[..., cols.index(int(gold_token_id))] = 1.0
    return ((p - onehot) ** 2).sum(axis=-1)


def fixed_opponent_token(natural_logits, candidate_token_ids, gold_token_id):
    """The top NON-gold candidate under the NATURAL logits (section 6.1).

    Fixed once per (query, seed) and shared by every arm. Recomputing it per arm
    would let the opponent move to whatever the arm happened to promote, so a
    margin could improve while the arm made things worse -- the comparison would
    silently change its own question.
    """
    cols = np.asarray(candidate_token_ids, dtype=int)
    z = np.asarray(natural_logits, dtype=np.float64)
    mask = cols != int(gold_token_id)
    if not mask.any():
        raise ValueError("no opponent: the candidate set holds only the gold "
                         "token")
    return int(cols[mask][z[cols[mask]].argmax()])


def margin(logits, gold_token_id, opponent_token_id):
    z = np.asarray(logits, dtype=np.float64)
    return z[..., int(gold_token_id)] - z[..., int(opponent_token_id)]


class CandidateSpace:
    """The preregistered decision space, as classes paired with their tokens.

    Callers speak CLASSES; the class-to-token mapping happens in here and no
    raw column index is ever handed out. That is the whole purpose: the metric
    functions index full-vocabulary logits with token ids, and a runner that
    passed class indices instead would get valid columns, wrong numbers and no
    error. With this object that mistake is not expressible.

    Built from a frozen per-model label space (tools/label_space.py), which
    selected the tokens out of a calibration header by class id -- nothing is
    re-discovered or renumbered there either. NOT from the query manifest:
    that file is model-independent and no longer carries token ids.
    """

    def __init__(self, class_ids, token_ids):
        self.class_ids = [int(c) for c in class_ids]
        self.token_ids = [int(t) for t in token_ids]
        if len(self.class_ids) != len(self.token_ids):
            raise ValueError(f"{len(self.class_ids)} classes vs "
                             f"{len(self.token_ids)} tokens")
        if not self.class_ids:
            raise ValueError("empty candidate space")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError("candidate token ids are not distinct")
        if len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("candidate class ids are not distinct")
        self._tok = dict(zip(self.class_ids, self.token_ids))
        self._cls = dict(zip(self.token_ids, self.class_ids))

    @classmethod
    def from_label_space(cls, label_space, *, model=None, query_manifest=None,
                         tokenizer=None):
        """Build from a frozen per-model label space, verifying it first.

        This is the only supported route. Token ids are tokenizer-specific, so
        they belong to a model, not to the class space.
        """
        from tools.label_space import FrozenLabelSpace
        ls = (label_space if isinstance(label_space, FrozenLabelSpace)
              else FrozenLabelSpace.load(label_space, model=model,
                                         query_manifest=query_manifest,
                                         tokenizer=tokenizer))
        return cls(ls.eligible_classes, ls.candidate_token_ids)

    @classmethod
    def from_manifest(cls, manifest):
        """Removed. The query manifest is model-independent by design.

        It used to carry candidate token ids, and that is precisely how a set
        of Llama-2 ids came to describe a Llama-3.1 run: nothing in the file
        recorded which tokenizer had produced them. Refusing the old shape is
        the point -- a stale manifest must not quietly keep working.
        """
        raise ValueError(
            "candidate token ids no longer live in the query manifest: they "
            "are tokenizer-specific and the manifest is model-independent. "
            "Use CandidateSpace.from_label_space(<label_space.json>, "
            "model=..., query_manifest=...) -- see tools/build_label_space.py. "
            "A manifest still carrying 'candidate_space' predates 2026-09-01 "
            "and its ids may belong to a different tokenizer entirely.")

    def __len__(self):
        return len(self.class_ids)

    def token_of(self, class_idx):
        c = int(class_idx)
        if c not in self._tok:
            raise ValueError(
                f"class {c} is not in the candidate space "
                f"{self.class_ids[:5]}... ({len(self)} classes). Queries are "
                "restricted to the train classes, so a gold outside it means "
                "the query set is wrong, not the metric.")
        return self._tok[c]

    def class_of(self, token_id):
        t = int(token_id)
        if t not in self._cls:
            raise ValueError(f"token {t} is not a candidate token")
        return self._cls[t]

    def nll(self, logits, gold_class):
        return candidate_nll(logits, self.token_ids, self.token_of(gold_class))

    def probs(self, logits):
        return candidate_probs(logits, self.token_ids)

    def argmax_class(self, logits):
        return self.class_of(candidate_argmax_token(logits, self.token_ids))

    def correct(self, logits, gold_class):
        return int(self.argmax_class(logits) == int(gold_class))

    def brier(self, logits, gold_class):
        return brier(logits, self.token_ids, self.token_of(gold_class))

    def opponent_class(self, natural_logits, gold_class):
        return self.class_of(fixed_opponent_token(
            natural_logits, self.token_ids, self.token_of(gold_class)))

    def margin(self, logits, gold_class, opponent_class):
        return margin(logits, self.token_of(gold_class),
                      self.token_of(opponent_class))


# --------------------------------------------------------------------------
# section 6.2 pairing
# --------------------------------------------------------------------------
def paired_difference_common(arm_by_seed, natural_by_seed):
    """L2 ONLY: d_q = mean_s (metric^A_qs - metric^0_qs). Shape [n_queries].

    Both inputs are [n_seeds, n_queries] over ONE common query set. The mean is
    over seeds, inside the query, which is what makes the query the unit.

    Never use this for L3.1. Under the PCW protocol column q of seed 42 and
    column q of seed 43 are different queries, and averaging them would be
    arithmetic over unrelated observations that no shape check can catch --
    the arrays line up perfectly and the result is meaningless.
    """
    a = np.asarray(arm_by_seed, dtype=np.float64)
    b = np.asarray(natural_by_seed, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"arm {a.shape} vs natural {b.shape}: the paired "
                         "difference needs the same queries and seeds in both")
    if a.ndim != 2:
        raise ValueError(f"expected [n_seeds, n_queries], got {a.shape}")
    return (a - b).mean(axis=0)


def per_seed_differences(arm_by_seed, natural_by_seed):
    """L3.1 / PCW: one paired-difference vector per seed, kept apart.

    Inputs are sequences of per-seed arrays; the seeds may hold different
    queries and are NOT required to be the same length. Within a seed the pairing
    is by query, which is what the PCW protocol guarantees: every arm and every
    baseline sees that seed's same 250.
    """
    out = []
    if len(arm_by_seed) != len(natural_by_seed):
        raise ValueError(f"{len(arm_by_seed)} arm seeds vs "
                         f"{len(natural_by_seed)} natural seeds")
    if not arm_by_seed:
        raise ValueError("no seeds")
    for s, (a, b) in enumerate(zip(arm_by_seed, natural_by_seed)):
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        if a.shape != b.shape:
            raise ValueError(
                f"seed index {s}: arm has {a.size} queries, natural has "
                f"{b.size}. Within a seed the two arms must run on the SAME "
                "drawn queries, or the pairing is broken.")
        if a.size == 0:
            raise ValueError(f"seed index {s}: no queries")
        out.append(a - b)
    return out


def summarize_across_seeds(values):
    """Mean and SD of the per-seed estimates. SD uses ddof=1.

    ddof=1 because these are a sample of seeds, not the population, and because
    ddof=0 has already produced a wrong answer in this project once: it halves
    the variance of a pair and made unrelated groups look alike.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 2:
        return {"mean": float(v.mean()) if v.size else float("nan"),
                "sd": float("nan"), "n_seeds": int(v.size),
                "per_seed": v.tolist(),
                "note": "SD undefined for fewer than two seeds"}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "n_seeds": int(v.size), "per_seed": v.tolist()}


def cluster_weights(per_seed_d, per_seed_ids):
    """Collapse the per-seed differences onto unique query_ids.

    Returns (ids, w) with

        w_c = (1/S) sum_{s containing c} d_{c,s} / n_s

    so that `w.sum()` is exactly T = (1/S) sum_s mean(d_{.,s}), the mean of the
    per-seed means, while every observation of one query sits in ONE weight.

    That identity is what makes the clustering enforceable rather than
    advisory: the statistic is a plain signed sum over clusters, so a sign flip
    or a bootstrap draw necessarily moves all of a repeated query's
    observations together. There is no code path that reaches the flat
    (query, seed) table.
    """
    if len(per_seed_d) != len(per_seed_ids):
        raise ValueError(f"{len(per_seed_d)} difference vectors vs "
                         f"{len(per_seed_ids)} id vectors")
    if not per_seed_d:
        raise ValueError("no seeds")
    s_count = len(per_seed_d)
    acc: dict = {}
    for si, (d, ids) in enumerate(zip(per_seed_d, per_seed_ids)):
        d = np.asarray(d, dtype=np.float64).ravel()
        ids = list(ids)
        if d.size != len(ids):
            raise ValueError(
                f"seed index {si}: {d.size} differences but {len(ids)} query "
                "ids. Clustering is by query_id, so every observation must "
                "carry one.")
        if d.size == 0:
            raise ValueError(f"seed index {si}: no queries")
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"seed index {si}: repeated query_id WITHIN one seed. The draw "
                "is without replacement, so this means the input is wrong "
                "rather than that a query legitimately recurs.")
        if not np.all(np.isfinite(d)):
            raise ValueError(
                f"seed index {si}: {int((~np.isfinite(d)).sum())} non-finite "
                "difference(s). These must be refused rather than aggregated: "
                "a NaN gives a NaN interval and an inf reports the smallest "
                "achievable p as if it were evidence.")
        for q, val in zip(ids, d):
            acc[q] = acc.get(q, 0.0) + float(val) / (s_count * d.size)
    ids = sorted(acc)
    return ids, np.array([acc[q] for q in ids], dtype=np.float64)


def aggregate_statistic(w):
    """T = sum_c w_c: the mean of the per-seed means (see cluster_weights)."""
    return float(np.asarray(w, dtype=np.float64).sum())


def clustered_sign_flip_p(w, n_permutations=N_PERMUTATIONS, seed=STAT_SEED,
                          chunk=20_000):
    """Two-sided sign-flip p for T = sum_c w_c, flipping whole CLUSTERS.

    One Rademacher sign per unique query_id. A query drawn by two seeds is
    flipped once, not twice, so its two observations never act as independent
    evidence.
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.size == 0:
        raise ValueError("no clusters")
    obs = abs(w.sum())
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n_permutations:
        b = min(chunk, n_permutations - done)
        signs = rng.integers(0, 2, size=(b, w.size), dtype=np.int8) * 2 - 1
        hits += int((np.abs(signs @ w) >= obs - 1e-12).sum())
        done += b
    return (1 + hits) / (1 + n_permutations)


def _cluster_index(per_seed_ids):
    """Unique query_ids, plus each seed's observations as cluster positions."""
    uniq = sorted({q for ids in per_seed_ids for q in ids})
    pos = {q: i for i, q in enumerate(uniq)}
    return uniq, [np.array([pos[q] for q in ids], dtype=np.int64)
                  for ids in per_seed_ids]


def cluster_bootstrap_stats(per_seed_d, per_seed_ids, n_boot=N_BOOTSTRAP,
                            seed=STAT_SEED):
    """Bootstrap T by resampling query_ids, RECOMPUTING each seed's mean.

    The estimator being bootstrapped is "fixed n per seed, mean within a seed,
    then mean across seeds". Preserving it requires keeping seed membership: a
    resample draws a multiplicity m_q per unique query and each seed re-averages
    over ITS OWN queries with those multiplicities,

        dbar*_s = sum_{q in Q_s} m_q d_qs / sum_{q in Q_s} m_q ,
        T*      = (1/S) sum_s dbar*_s .

    Resampling the pre-aggregated cluster weights w_c and summing them -- which
    is what this function used to do -- is NOT the same estimator, because a
    query's total weight depends on how many seeds drew it, so a resample
    silently changes the per-seed denominators. The failure is visible with no
    statistics at all: set every d_qs = 1, and every seed mean and T are exactly
    1 by construction, so the interval must be [1, 1]. The old code returned
    [0.9636, 1.0366].

    A replicate in which some seed drew none of its own queries has an undefined
    mean; those are dropped and counted rather than filled in.
    """
    ds = [np.asarray(d, dtype=np.float64).ravel() for d in per_seed_d]
    if len(ds) != len(per_seed_ids):
        raise ValueError(f"{len(ds)} difference vectors vs "
                         f"{len(per_seed_ids)} id vectors")
    for si, (d, ids) in enumerate(zip(ds, per_seed_ids)):
        if d.size != len(ids):
            raise ValueError(f"seed index {si}: {d.size} differences but "
                             f"{len(ids)} query ids")
    uniq, idx = _cluster_index(per_seed_ids)
    c = len(uniq)
    rng = np.random.default_rng(seed)
    # counts of a with-replacement sample of C clusters from C
    m = rng.multinomial(c, np.full(c, 1.0 / c), size=n_boot)
    total = np.zeros(n_boot, dtype=np.float64)
    usable = np.ones(n_boot, dtype=bool)
    for d, ix in zip(ds, idx):
        mm = m[:, ix]                                  # [n_boot, n_s]
        den = mm.sum(axis=1).astype(np.float64)
        num = mm.astype(np.float64) @ d
        usable &= den > 0
        total += np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    total /= len(ds)
    if not usable.all():
        # At the preregistered 250-per-seed scale this cannot realistically
        # happen -- a seed would have to lose all 250 of its queries in one
        # resample -- so any dropping at all means the input is much smaller
        # than the design, which is fine for a fixture and a red flag for a run.
        n_bad = int((~usable).sum())
        if n_bad > n_boot // 5:
            raise RuntimeError(
                f"{n_bad}/{n_boot} bootstrap replicates left some seed with no "
                "queries of its own, so its mean was undefined. At that rate "
                "the interval is not trustworthy; the seeds are far smaller "
                "than the preregistered 250.")
        total = total[usable]
    return total


def cluster_bootstrap_ci(per_seed_d, per_seed_ids, level=0.95,
                         n_boot=N_BOOTSTRAP, seed=STAT_SEED):
    b = cluster_bootstrap_stats(per_seed_d, per_seed_ids, n_boot, seed)
    lo = (1 - level) / 2 * 100
    return float(np.percentile(b, lo)), float(np.percentile(b, 100 - lo))


def cluster_bootstrap_bound(per_seed_d, per_seed_ids, side, level=0.95,
                            n_boot=N_BOOTSTRAP, seed=STAT_SEED):
    if side not in ("upper", "lower"):
        raise ValueError(f"side must be 'upper' or 'lower', got {side!r}")
    b = cluster_bootstrap_stats(per_seed_d, per_seed_ids, n_boot, seed)
    return float(np.percentile(b, level * 100 if side == "upper"
                               else (1 - level) * 100))


# --------------------------------------------------------------------------
# section 6.2 inference
# --------------------------------------------------------------------------
def sign_flip_p(d, n_permutations=N_PERMUTATIONS, seed=STAT_SEED,
                chunk=20_000):
    """Two-sided paired sign-flip permutation p-value.

    Under the null that each query's paired difference is symmetric about zero,
    flipping its sign leaves the distribution alone. The statistic is the mean.

    p = (1 + #{|T*| >= |T_obs|}) / (1 + B). The +1s make it a valid Monte-Carlo
    p-value: without them a null that never once reaches |T_obs| reports p=0,
    which claims more resolution than B draws can provide.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    if d.size == 0:
        raise ValueError("no queries")
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n_permutations:
        b = min(chunk, n_permutations - done)
        signs = rng.integers(0, 2, size=(b, d.size), dtype=np.int8) * 2 - 1
        hits += int((np.abs(signs @ d) >= obs * d.size - 1e-12).sum())
        done += b
    return (1 + hits) / (1 + n_permutations)


def exact_sign_flip_p(d):
    """Exhaustive two-sided sign-flip p for small n. Reference for the fixture.

    Enumerating 2^n is the only way to know whether the Monte-Carlo version is
    right; comparing it against another sampler would just compare two samplers.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    n = d.size
    if n > 22:
        raise ValueError(f"2^{n} is too many; this is the small-n reference")
    signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    return float((np.abs(signs @ d) >= abs(d.sum()) - 1e-12).sum()) / 2 ** n


def bootstrap_means(d, n_boot=N_BOOTSTRAP, seed=STAT_SEED):
    """Query-bootstrap distribution of the mean. Resamples QUERIES, not cells."""
    d = np.asarray(d, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    return d[idx].mean(axis=1)


def bootstrap_ci(d, level=0.95, n_boot=N_BOOTSTRAP, seed=STAT_SEED):
    """Two-sided percentile interval (section 6.2)."""
    b = bootstrap_means(d, n_boot, seed)
    lo = (1 - level) / 2 * 100
    return float(np.percentile(b, lo)), float(np.percentile(b, 100 - lo))


def bootstrap_bound(d, side, level=0.95, n_boot=N_BOOTSTRAP, seed=STAT_SEED):
    """One-sided percentile bound. side='upper' -> the 95th percentile."""
    if side not in ("upper", "lower"):
        raise ValueError(f"side must be 'upper' or 'lower', got {side!r}")
    b = bootstrap_means(d, n_boot, seed)
    return float(np.percentile(b, level * 100 if side == "upper"
                               else (1 - level) * 100))


def mcnemar_floor_p(net):
    """The SMALLEST two-sided exact McNemar p any split with this net can give.

    Why this exists. An accuracy difference is a NET: `net = c - b`, so many
    (broken, fixed) splits produce it and each has its own p. McNemar
    conditions on the discordant total b + c, and adding a matched pair to
    both sides only makes the split more balanced, so p falls monotonically as
    b goes to 0. The floor is therefore b = 0, c = |net|:

        p_min = 2 * 2 ** -|net|        (1.0 when net = 0)

    THE POINT IS THAT IT NEEDS NO DATA BEYOND THE NET. Given a net of +4, the
    most favourable outcome imaginable -- the arm broke nothing and fixed four
    -- gives p = 0.125, so no discordant split can reach 0.05 and the counts
    do not have to be looked at to know that. Equivalently: a net of fewer
    than SIX queries can never be significant, whatever n is and however many
    decisions were disturbed.

    Report it beside any accuracy difference. Without it a p of 1.0000 reads
    as "measured, and nothing there" when it may mean "this test could not
    have fired" -- working rules 3.4's rule that a gate no input can turn red is
    not checking anything, and 11b's that a criterion has to be attainable.
    """
    k = abs(int(net))
    return 1.0 if k == 0 else min(1.0, 2.0 * 2.0 ** -k)


def mcnemar_can_signify(net, alpha=ALPHA):
    """Could ANY split with this net reach `alpha`? (b=0 is the best case.)"""
    return mcnemar_floor_p(net) < alpha


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the input order.

    Adjusted p_(i) = max_{j<=i} min(1, (m-j+1) p_(j)); the running max enforces
    the monotonicity that makes the step-down procedure coherent -- without it a
    later, larger raw p could be reported as more significant than an earlier
    one.
    """
    p = np.asarray(pvals, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("no p-values")
    if ((p < 0) | (p > 1)).any():
        raise ValueError(f"p-values outside [0,1]: {p[(p < 0) | (p > 1)]}")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adj = np.maximum.accumulate(adj)
    out = np.empty(m)
    out[order] = adj
    return out


def decide(d, direction, alpha=ALPHA, **kw):
    """The full section 6.2 decision: two-sided p AND the required direction.

    direction='decrease' for NLL, 'increase' for accuracy. Returned as separate
    fields, never pre-combined: a two-sided test is symmetric, so significance
    alone would pass a large effect pointing the wrong way.
    """
    if direction not in ("decrease", "increase"):
        raise ValueError(f"direction must be 'decrease' or 'increase', got "
                         f"{direction!r}")
    d = np.asarray(d, dtype=np.float64).ravel()
    mean = float(d.mean())
    p = sign_flip_p(d, **kw)
    lo, hi = bootstrap_ci(d)
    right_way = mean < 0 if direction == "decrease" else mean > 0
    return {"mean": mean, "p": p, "ci95": (lo, hi), "n_queries": int(d.size),
            "direction_ok": bool(right_way),
            "significant": bool(p < alpha),
            "passes": bool(p < alpha and right_way)}


def decide_aggregate(d_by_seed, ids_by_seed, direction, alpha=ALPHA,
                     n_permutations=N_PERMUTATIONS, perm_seed=STAT_SEED,
                     n_bootstrap=N_BOOTSTRAP, boot_seed=STAT_SEED, **kw):
    """The L3.1 decision: ONE aggregate clustered test on T.

    T is the mean of the per-seed means; the sign-flip randomisation and the
    bootstrap both cluster on unique query_id. The claim passes on a two-sided
    p below alpha AND the point estimate pointing the right way.

    Per-seed means and directions come back in `per_seed`, and the seed-to-seed
    SD in `across_seeds`, but neither is a gate: they are descriptive. There is
    no per-seed significance requirement and no max-of-p anywhere.
    """
    if direction not in ("decrease", "increase"):
        raise ValueError(f"direction must be 'decrease' or 'increase', got "
                         f"{direction!r}")
    ids, w = cluster_weights(d_by_seed, ids_by_seed)
    T = aggregate_statistic(w)
    p = clustered_sign_flip_p(w, n_permutations=n_permutations, seed=perm_seed,
                              **kw)
    # the sign-flip may use the collapsed weights (the statistic is a signed sum
    # over clusters); the bootstrap may NOT, because it has to re-average within
    # each seed -- see cluster_bootstrap_stats.
    lo, hi = cluster_bootstrap_ci(d_by_seed, ids_by_seed, n_boot=n_bootstrap,
                                  seed=boot_seed)
    per_seed_means = [float(np.asarray(d, dtype=np.float64).mean())
                      for d in d_by_seed]
    n_obs = int(sum(np.asarray(d).size for d in d_by_seed))
    right_way = T < 0 if direction == "decrease" else T > 0
    return {"T": T, "p": p, "ci95": (lo, hi),
            # what actually produced the numbers above, so a reader never has to
            # infer it and assert_confirmatory can check it
            "params": {"direction": direction, "alpha": float(alpha),
                       "n_permutations": int(n_permutations),
                       "perm_seed": int(perm_seed),
                       "n_bootstrap": int(n_bootstrap),
                       "boot_seed": int(boot_seed)},
            "n_clusters": len(ids), "n_observations": n_obs,
            "n_repeated_queries": n_obs - len(ids),
            "direction_ok": bool(right_way),
            "significant": bool(p < alpha),
            "passes": bool(p < alpha and right_way),
            "per_seed_means": per_seed_means,          # descriptive only
            "across_seeds": summarize_across_seeds(per_seed_means),
            "note": "T is the average effect given these three preregistered "
                    "prefix seeds. Three seeds do not estimate a seed "
                    "population; the seed SD is a descriptive spread."}


# Section 2.1/2.2, from tools/prereg_config. Not extendable at run time.
from tools.prereg_config import N_TEST_PER_SEED  # noqa: E402,F401
from tools.prereg_config import REGISTERED_SEEDS as REGISTERED_DEMO_SEEDS


def _load_verified_manifest(manifest, freeze_manifest, allow_unverified):
    """The query manifest, having proved it is the FROZEN one.

    Checking results against "whatever manifest was handed in" proves only that
    the caller is self-consistent: an edited manifest paired with results that
    match it passes every id check. So unless the caller explicitly opts out,
    the manifest must be a PATH, a freeze manifest must accompany it, every
    registered artifact must still hash to its recorded value, and the query
    manifest itself must be one of those registered artifacts.
    """
    from tools.build_query_manifest import freeze_manifest_blockers

    if allow_unverified:
        if isinstance(manifest, (str, Path)):
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        return manifest, None

    if not isinstance(manifest, (str, Path)):
        raise ValueError(
            "a confirmatory comparison needs the query manifest as a PATH so "
            "it can be hashed against the freeze manifest. A dict proves only "
            "that the caller is self-consistent with itself. Pass the path, or "
            "allow_unverified_manifest=True for a fixture.")
    if freeze_manifest is None:
        raise ValueError(
            "a confirmatory comparison needs --freeze-manifest: the query "
            "manifest must be shown to be the frozen one, not merely a "
            "manifest that happens to match the results.")

    # THE SAME AUTHORITY CHECK THE TEST LOCK USES, and bound to THIS manifest.
    # This used to call verify_freeze_manifest and then hunt for the manifest
    # among the registered files -- which a two-line self-signed JSON carrying
    # exactly one correct hash satisfies. A confirmatory result computed under
    # that is a result computed under no freeze at all, so the weaker path is
    # gone rather than kept alongside.
    bad = freeze_manifest_blockers(
        freeze_manifest, expected_roles={"query_manifest": manifest})
    if bad:
        raise PermissionError(
            "the freeze manifest may not authorise a confirmatory result: "
            + "; ".join(bad))
    p = Path(manifest)
    return json.loads(p.read_text(encoding="utf-8")), str(p)


def confirmatory_differences(arm_by_seed, ref_by_seed, manifest, *,
                             freeze_manifest=None,
                             allow_unverified_manifest=False,
                             seeds=REGISTERED_DEMO_SEEDS,
                             n_per_seed=N_TEST_PER_SEED):
    """Paired differences for a CONFIRMATORY comparison, aligned by query_id.

    arm_by_seed / ref_by_seed : {demo_seed: {query_id: metric}}
    manifest                  : path to the frozen query manifest
    freeze_manifest           : path to the freeze manifest that pins it

    `decide_aggregate` accepts any number of seeds of any length, which is right
    for a general statistical routine and wrong for H4/H4b/H6. This is the
    registered gate in front of it, and it refuses anything that is not the
    preregistered design:

      * the query manifest hashes to its registered value in a freeze manifest
        that itself still matches disk;
      * exactly demo seeds 42, 43, 44;
      * exactly 250 unique queries per seed;
      * query ids identical to that seed's frozen list;
      * arm and reference matched BY query id, not by array position;
      * every metric and every difference finite.

    Those last conditions are what stop a silent wrong answer. If a forward job
    fails to write ten queries, position-based pairing still yields two
    same-length arrays and a perfectly ordinary p-value; here it raises and
    names the missing ids.
    """
    manifest, _mpath = _load_verified_manifest(manifest, freeze_manifest,
                                               allow_unverified_manifest)
    frozen = manifest.get("test_by_seed")
    if not frozen:
        raise ValueError(
            "the manifest has no 'test_by_seed'. A confirmatory L3.1 "
            "comparison must run on the frozen per-seed draws; without them "
            "there is nothing to check the query ids against.")

    want = tuple(int(s) for s in seeds)
    for name, got in (("arm", arm_by_seed), ("reference", ref_by_seed)):
        have = tuple(sorted(int(s) for s in got))
        if have != tuple(sorted(want)):
            raise ValueError(
                f"{name}: demo seeds {have} but the preregistration fixes "
                f"{tuple(sorted(want))} (section 2.1). Seeds are not added or "
                "dropped at analysis time.")

    per_seed_d, per_seed_ids = [], []
    for s in want:
        ids = list(frozen[str(s)])
        if len(ids) != n_per_seed or len(set(ids)) != n_per_seed:
            raise ValueError(
                f"seed {s}: the manifest records {len(ids)} draws "
                f"({len(set(ids))} unique) but the design fixes exactly "
                f"{n_per_seed}. The manifest itself is wrong; do not proceed.")
        a, r = arm_by_seed[s], ref_by_seed[s]
        for name, got in (("arm", a), ("reference", r)):
            missing = [q for q in ids if q not in got]
            extra = [q for q in got if q not in set(ids)]
            if missing or extra:
                raise ValueError(
                    f"seed {s} {name}: {len(missing)} of the frozen "
                    f"{n_per_seed} queries are missing and {len(extra)} are "
                    f"not in the frozen draw. First missing: {missing[:3]}; "
                    f"first extra: {extra[:3]}. Results must cover the frozen "
                    "draw exactly -- a partial forward would otherwise produce "
                    "an ordinary-looking p-value on a different sample.")
        # Non-finite metrics must stop the analysis, not travel into it. A NaN
        # propagates to T and yields a NaN interval; an inf makes |T| exceed
        # every sign-flipped statistic and reports p = 1/(1+B), which reads as
        # overwhelming significance. Both look like results.
        av = np.array([float(a[q]) for q in ids], dtype=np.float64)
        rv = np.array([float(r[q]) for q in ids], dtype=np.float64)
        d = av - rv
        for label, vals in (("arm", av), ("reference", rv),
                            ("difference", d)):
            bad = ~np.isfinite(vals)
            if bad.any():
                where = [ids[i] for i in np.flatnonzero(bad)[:3]]
                raise ValueError(
                    f"seed {s}: {int(bad.sum())} non-finite {label} value(s), "
                    f"first at {where}. A NaN would give a NaN interval and an "
                    "inf would report p = 1/(1+B) as if it were overwhelming "
                    "evidence, so this stops here rather than propagating.")
        per_seed_d.append(d)
        per_seed_ids.append(ids)
    return per_seed_d, per_seed_ids


def confirmatory_decision(arm_by_seed, ref_by_seed, manifest, direction,
                          alpha=ALPHA, freeze_manifest=None,
                          allow_unverified_manifest=False, **kw):
    """The registered H4 / H4b / H6 entry point: 3 x 250, then the aggregate.

    An UNVERIFIED run does not produce a verdict. Recording
    `manifest_verified: False` beside an ordinary `passes: True` would leave the
    distinction to whoever remembers to read the metadata, and the whole point
    of this gate is not to rely on that. So when the manifest was not checked
    against a freeze manifest, `passes` and `significant` come back as None --
    falsy, so `if result["passes"]` cannot read as success -- and the booleans
    move to `diagnostic_passes` / `diagnostic_significant`, which a caller has
    to ask for by name.

    `assert_confirmatory` is the hard gate a reporter calls.
    """
    verified = not allow_unverified_manifest
    if verified:
        # A VERIFIED run may not be a fast run. Every one of these is fixed by
        # the preregistration, and a result carrying 2,000 permutations instead
        # of 1,000,000 looks exactly like a real one once it is written down --
        # which is why this is refused here rather than noticed later.
        overridden = [f"{name}={got!r} (registered {want!r})"
                      for name, got, want in
                      (("alpha", alpha, ALPHA),
                       ("n_permutations", kw.get("n_permutations",
                                                 N_PERMUTATIONS),
                        N_PERMUTATIONS),
                       ("perm_seed", kw.get("perm_seed", STAT_SEED), STAT_SEED),
                       ("n_bootstrap", kw.get("n_bootstrap", N_BOOTSTRAP),
                        N_BOOTSTRAP),
                       ("boot_seed", kw.get("boot_seed", STAT_SEED), STAT_SEED))
                      if got != want]
        if overridden:
            raise ValueError(
                "a VERIFIED confirmatory run must use the preregistered "
                "statistical constants, but these were overridden: "
                + "; ".join(overridden)
                + ". Reduced settings are for diagnostics only -- pass "
                  "allow_unverified_manifest=True, which yields no verdict.")

    d, ids = confirmatory_differences(
        arm_by_seed, ref_by_seed, manifest, freeze_manifest=freeze_manifest,
        allow_unverified_manifest=allow_unverified_manifest)
    out = decide_aggregate(d, ids, direction, alpha=alpha, **kw)
    out["confirmatory_status"] = "VERIFIED" if verified else "UNVERIFIED"
    out["registered"] = {"demo_seeds": list(REGISTERED_DEMO_SEEDS),
                         "n_per_seed": N_TEST_PER_SEED,
                         "aligned_by": "query_id",
                         "manifest_verified": verified,
                         "n_observations": sum(len(x) for x in ids)}
    if not verified:
        out["diagnostic_passes"] = out["passes"]
        out["diagnostic_significant"] = out["significant"]
        out["passes"] = None
        out["significant"] = None
        out["note"] = (
            "UNVERIFIED: the query manifest was not checked against a freeze "
            "manifest, so this is a diagnostic run and carries no verdict. "
            + out["note"])
    return out


def assert_confirmatory(result):
    """Raise unless `result` is a complete, VERIFIED confirmatory decision.

    A reporter calls this before quoting any H4 / H4b / H6 outcome, so it cannot
    be satisfied by a label alone: `{"confirmatory_status": "VERIFIED"}` is a
    dict someone can type, and it must not be accepted. Every invariant the
    registered path establishes is re-checked here on the object itself --
    seeds, size, alignment, verification, verdict types and finite numbers --
    so the gate holds even for a result that was written, edited or
    round-tripped by something other than confirmatory_decision.

    All problems are reported together; a reporter should not have to fix them
    one run at a time.
    """
    if not isinstance(result, dict):
        raise ValueError(f"expected a decision dict, got {type(result).__name__}")

    status = result.get("confirmatory_status")
    if status is None:
        raise ValueError(
            "this result has no confirmatory_status, so it did not come from "
            "confirmatory_decision. decide_aggregate does not check the demo "
            "seeds, the 250-per-seed size, the frozen query ids or the freeze "
            "manifest; a confirmatory claim cannot rest on it.")
    if status != "VERIFIED":
        raise PermissionError(
            f"confirmatory_status is {status!r}, not 'VERIFIED'. The query "
            "manifest was not checked against a freeze manifest, so this run "
            "carries no verdict -- its numbers are diagnostic only "
            f"(diagnostic_passes={result.get('diagnostic_passes')!r}).")

    bad = []
    reg = result.get("registered")
    if not isinstance(reg, dict):
        bad.append("no 'registered' block, so nothing records what was checked")
        reg = {}
    if reg.get("manifest_verified") is not True:
        bad.append(f"registered.manifest_verified is "
                   f"{reg.get('manifest_verified')!r}, not True")
    if tuple(reg.get("demo_seeds") or ()) != REGISTERED_DEMO_SEEDS:
        bad.append(f"registered.demo_seeds is {reg.get('demo_seeds')!r}, not "
                   f"{list(REGISTERED_DEMO_SEEDS)}")
    if reg.get("n_per_seed") != N_TEST_PER_SEED:
        bad.append(f"registered.n_per_seed is {reg.get('n_per_seed')!r}, not "
                   f"{N_TEST_PER_SEED}")
    want_obs = len(REGISTERED_DEMO_SEEDS) * N_TEST_PER_SEED
    if reg.get("n_observations") != want_obs:
        bad.append(f"registered.n_observations is "
                   f"{reg.get('n_observations')!r}, not {want_obs}")
    if reg.get("aligned_by") != "query_id":
        bad.append(f"registered.aligned_by is {reg.get('aligned_by')!r}, not "
                   "'query_id'")

    for key in ("passes", "significant", "direction_ok"):
        if not isinstance(result.get(key), bool):
            bad.append(f"{key} is {result.get(key)!r}; a VERIFIED result must "
                       "carry a real boolean verdict")
    for key in ("diagnostic_passes", "diagnostic_significant"):
        if key in result:
            bad.append(f"{key} is present, which only an UNVERIFIED run "
                       "produces; this result is internally inconsistent")

    for key in ("T", "p"):
        v = result.get(key)
        if not _is_number(v) or not np.isfinite(v):
            bad.append(f"{key} is {v!r}, not a finite number")
    p = result.get("p")
    if _is_number(p) and np.isfinite(p) and not 0 < p <= 1:
        bad.append(f"p is {p!r}, outside (0, 1]")
    ci = result.get("ci95")
    if (not isinstance(ci, (tuple, list)) or len(ci) != 2
            or not all(_is_number(x) and np.isfinite(x) for x in ci)):
        bad.append(f"ci95 is {ci!r}, not a pair of finite numbers")
    elif ci[0] > ci[1]:
        bad.append(f"ci95 is inverted: {ci}")

    # ---- the registered statistical constants, and the verdict they imply ---
    par = result.get("params")
    if not isinstance(par, dict):
        bad.append("no 'params' block, so nothing records which alpha, how "
                   "many permutations or which seeds produced these numbers")
        par = {}
    for name, want in (("alpha", ALPHA),
                       ("n_permutations", N_PERMUTATIONS),
                       ("perm_seed", STAT_SEED),
                       ("n_bootstrap", N_BOOTSTRAP),
                       ("boot_seed", STAT_SEED)):
        if par.get(name) != want:
            bad.append(f"params.{name} is {par.get(name)!r}, not the "
                       f"preregistered {want!r}")
    direction = par.get("direction")
    if direction not in ("decrease", "increase"):
        bad.append(f"params.direction is {direction!r}, not 'decrease' or "
                   "'increase'; without it the sign of T means nothing")

    # The booleans must FOLLOW from the numbers. Checking only
    # passes == significant and direction_ok leaves a result that reports
    # significant=True beside p=0.9 entirely unchallenged.
    p, T, alpha = result.get("p"), result.get("T"), par.get("alpha")
    sig, dok = result.get("significant"), result.get("direction_ok")
    if (_is_number(p) and np.isfinite(p) and _is_number(alpha)
            and isinstance(sig, bool)):
        if sig != (p < alpha):
            bad.append(f"significant={sig} but p={p!r} and alpha={alpha!r}, "
                       f"so p < alpha is {p < alpha}")
    if (_is_number(T) and np.isfinite(T) and isinstance(dok, bool)
            and direction in ("decrease", "increase")):
        want_dok = T < 0 if direction == "decrease" else T > 0
        if dok != want_dok:
            bad.append(f"direction_ok={dok} but T={T!r} with direction="
                       f"{direction!r}, which requires {want_dok}")
    if all(isinstance(result.get(k), bool)
           for k in ("passes", "significant", "direction_ok")):
        if result["passes"] != (result["significant"]
                                and result["direction_ok"]):
            bad.append(
                f"passes={result['passes']} contradicts significant="
                f"{result['significant']} and direction_ok="
                f"{result['direction_ok']}; the verdict is not the conjunction "
                "it is defined to be")

    if bad:
        raise ValueError(
            "the result claims confirmatory_status VERIFIED but fails "
            f"{len(bad)} invariant(s) of the registered design:\n  - "
            + "\n  - ".join(bad)
            + "\nA status string is not evidence; it must be a decision "
              "confirmatory_decision actually produced.")
    return result


def non_inferiority(d_nll, d_acc, nll_margin=NLL_MARGIN,
                    acc_margin=ACC_MARGIN):
    """Section 9's gamma=1 "preserved" criterion: TWO one-sided bounds.

    Not "p is not significant". A non-significant result is compatible with a
    large effect the sample was too small to resolve, so section 9 requires the
    bootstrap bound itself to sit inside the margin -- which fails loudly when
    the data are uninformative instead of passing quietly.
    """
    up = bootstrap_bound(d_nll, "upper")
    lo = bootstrap_bound(d_acc, "lower")
    return {"nll_upper95": up, "nll_margin": nll_margin,
            "nll_ok": bool(up < nll_margin),
            "acc_lower95": lo, "acc_margin": acc_margin,
            "acc_ok": bool(lo > acc_margin),
            "passes": bool(up < nll_margin and lo > acc_margin)}


def non_inferiority_aggregate(d_nll_by_seed, d_acc_by_seed, ids_by_seed,
                              nll_margin=NLL_MARGIN, acc_margin=ACC_MARGIN):
    """Section 9's gamma=1 criterion on the AGGREGATE T, cluster-bootstrapped.

    Adjudicated on T, not seed by seed: the claim is about the average effect
    across the three prefix seeds, so the bound belongs on the same estimand the
    hypothesis is about.

    The clustering is what keeps this honest. Resampling (query, seed) cells
    would split a repeated query across draws, shrink the interval, and for an
    EQUIVALENCE claim a narrower interval is easier to fit inside the margin --
    the error would run toward passing.
    """
    if len(d_nll_by_seed) != len(d_acc_by_seed):
        raise ValueError(f"{len(d_nll_by_seed)} NLL seeds vs "
                         f"{len(d_acc_by_seed)} accuracy seeds")
    _i1, w_nll = cluster_weights(d_nll_by_seed, ids_by_seed)
    _i2, w_acc = cluster_weights(d_acc_by_seed, ids_by_seed)
    up = cluster_bootstrap_bound(d_nll_by_seed, ids_by_seed, "upper")
    lo = cluster_bootstrap_bound(d_acc_by_seed, ids_by_seed, "lower")
    return {"nll_T": aggregate_statistic(w_nll), "nll_upper95": up,
            "nll_margin": nll_margin, "nll_ok": bool(up < nll_margin),
            "acc_T": aggregate_statistic(w_acc), "acc_lower95": lo,
            "acc_margin": acc_margin, "acc_ok": bool(lo > acc_margin),
            "n_clusters": len(_i1),
            "passes": bool(up < nll_margin and lo > acc_margin)}
