"""Shared spine for the six H6 baseline runners: test lock, artifact
quarantine, candidate space, output schema. ZERO GPU at import; torch is
imported lazily by the runners, never here.

Three things every baseline must get right, and none of them are the baseline's
own business, so they live here once.

1. THE TEST LOCK. Section 2.1 forbids generating any test prediction before the
   three gamma*, the control carriers/positions and the analysis code are
   frozen. baseline_under_review.md section 0.1 states the rule as "the runner
   must raise if the query manifest contains test IDs".

   That rule is no longer implementable as written, and pretending otherwise
   would be worse than useless. Under the current schema there is ONE manifest
   and it always contains every test query -- `entries` IS the pool. A runner
   that refused such a manifest would refuse the only manifest that exists, and
   a runner that checked and passed would be checking nothing.

   So the guard is moved from what the manifest CONTAINS to what the run
   MATERIALISES: queries only ever come from `load_split()`, which refuses both
   test splits without a verified freeze manifest, and whatever comes back is
   then asserted disjoint from every test id in that same manifest
   (`assert_no_test_leak`). That is strictly stronger -- it fails on a
   hand-assembled id list, which the containment check never could.

2. THE ARTIFACT QUARANTINE. Section 0.2: no baseline code path may read this
   project's TL-head rankings, kappa/phi dumps or prototype caches. An import-
   time assertion cannot enforce that -- the read happens at runtime, from a
   path that is a string until the moment it is opened. So every file a runner
   opens goes through `read_json`/`read_npz` here, which refuse the quarantined
   patterns, and `tests/baselines/test_common.py` greps the runner sources for
   bare `open(`/`json.load`/`np.load`. The grep is the part that actually
   binds: it is categorical (working rules 11b) and cannot be satisfied by a
   sloppy runner that happens to pass today.

3. THE PLACEBO CONTRACT IS THE PREREGISTRATION'S, NOT A GENERALISATION OF IT.
   Section 11.9 is explicit and already contains the distinction:

     the five MODEL-SIDE runners -- ZeroTuning r=1, FV/TSLA alpha=0, UniBias
     empty attention/FFN sets -- must produce candidate logits equal ELEMENT
     FOR ELEMENT to the SAME full-prompt natural forward;

     Deep-Thinking T=1 is the ONE exception: it matches a same-prefix-cache
     natural reference and is explicitly not required to match the differently
     chunked full-prompt natural bitwise.

   An earlier version of this file replaced that with a two-gate scheme in
   which every runner compared against its own path and only approximated the
   stock one. That was changing a registered criterion, not following it, and
   the justification given -- that a rebuilt attention cannot reproduce a fused
   kernel's reduction order -- does not apply here at all: the registered
   setting is EAGER attention, and an adapter whose neutral parameter cancels
   exactly (r=1 divides by exactly 1.0) and which then calls the same eager
   kernel on the same tensors does reproduce the stock bits. So the contract is
   restored: `placebo_bitwise` against the stock forward, with
   `dt_path_agreement` as Deep-Thinking's registered numeric allowance and
   nothing else permitted to use it.

   ⚠ The adapter must NOT short-circuit to the stock path when its parameter is
   neutral. That would make the gate vacuous -- it would verify a branch, not
   the intervention arithmetic. The neutral case must flow through the same
   code and come out identical because the arithmetic cancels.

WHAT IS STORED, AND WHY NOT THE FULL LOGITS. The readout is 36 frozen candidate
tokens, so the candidate slice is the raw quantity for this analysis. Full-vocab
logits would be 750 cells x 128256 x 4 bytes per arm, which is not a reasonable
thing to write per arm. But a bare candidate slice cannot answer a question
asked in full-vocabulary terms later, and working rules rule 9 says do not store
only summaries. So each cell also carries `full_logsumexp` and
`full_argmax_token`: the first makes every candidate's FULL-space probability
recoverable exactly (log p_full(c) = logit_c - logsumexp), the second preserves
the icl_common convention-2 check that `model_pred_class` is a full-space
argmax.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.build_query_manifest import load_split  # noqa: E402
from tools.label_space import FrozenLabelSpace, file_sha256  # noqa: E402

SCHEMA_VERSION = 1
MODES = ("placebo", "fit", "validation", "frozen")
from tools.prereg_config import (  # noqa: E402,F401
    REGISTERED_SEEDS as REGISTERED_DEMO_SEEDS)

# Section 0.2, by ARTIFACT KIND rather than by fuzzy name.
#
# The first version matched the substring "method_a_" and therefore refused
# `results/prereg_method_A_query_manifest.json` -- the shared, MODEL-INDEPENDENT
# class space that every baseline is required to read. The fixtures missed it
# because they used a synthetic name (`qm.json`), which is working rules rule 4 in
# its purest form: a fixture built from the implementation's own conventions
# can only confirm that the implementation is self-consistent. Every pattern
# below is now written against a REAL filename and the fixtures use those.
#
# The distinction that matters is not "does the name mention Method A" but
# "is this a FINDING of the method, or shared input?":
#
#   shared input   the query manifest (classes, not tokens), the per-model
#                  label space, the calibration prompts -- section 2.4(3)
#                  explicitly permits all of these, and using the same ones is
#                  what makes the comparison paired. Since 14.0a that list no
#                  longer includes a discovery pool: every method fits on the
#                  same validation queries instead;
#   a finding      TL-head rankings, carriers, prototypes, kappa/phi dumps,
#                  probe outputs, the viability gate's verdict. A baseline that
#                  reads one of these is no longer independent of the method it
#                  is being compared against.
DENIED_PATTERNS = (
    "tl_heads*", "tl_score*", "kappa*", "*prototype*", "*carrier*",
    "probe_*", "*_phi_*", "*phi_dump*", "method_a_viability*",
    "*shrinkage*", "*label_profile*", "*substrate*",
)
ALLOWED_PATTERNS = (
    "prereg_method_A_query_manifest.json",   # model-independent class space
    "*query_manifest*.json",                 # ...and its variants
    "label_space_*.json",                    # per-model label space
    "calibration_*.jsonl",                   # rendered prompts + header
    "baseline_*.json", "baseline_*.npz",     # this package's own artifacts
    "freeze_manifest*.json",                 # the test lock's evidence
)
# The one exemption in the spec: section 3.4 lets the ANALYSER read a head list
# to compute a Jaccard overlap. Analysis is not a run path, so it opts in
# explicitly and says so in the artifact.
JACCARD_EXEMPTION = "jaccard-descriptive-only"


class TestLockError(PermissionError):
    """Raised when a run would materialise a test query it may not touch."""


# ==========================================================================
# artifact quarantine
# ==========================================================================
def refuse_quarantined(path, *, exemption=None):
    """Raise unless `path` is a permitted shared input.

    Denied first, then allowed, then REFUSED BY DEFAULT. Fail-closed on an
    unrecognised name is deliberate: "I did not think of this file" is not the
    same as "this file is safe", and the cost of the strictness is one line
    added to ALLOWED_PATTERNS as a deliberate act.
    """
    if exemption == JACCARD_EXEMPTION:
        return
    from fnmatch import fnmatch
    name = Path(path).name
    lo = name.lower()
    hit = [q for q in DENIED_PATTERNS if fnmatch(lo, q.lower())]
    if hit:
        raise PermissionError(
            f"{name}: this is a FINDING of the method (matched {hit!r}), and a "
            "baseline may not read one. Section 0.2: the H6 comparison is only "
            "meaningful if the baselines are independent of what the method "
            "found. Shared INPUTS -- query manifest, label space, calibration "
            "-- are permitted and are what makes the comparison "
            f"paired. If this is the descriptive Jaccard analysis of section "
            f"3.4, pass exemption={JACCARD_EXEMPTION!r}.")
    if not any(fnmatch(lo, q.lower()) for q in ALLOWED_PATTERNS):
        raise PermissionError(
            f"{name}: not on the list of artifacts a baseline may read. This "
            "is fail-closed on purpose -- an unrecognised name is refused "
            "rather than assumed harmless. If it is a legitimate shared input, "
            "add its pattern to ALLOWED_PATTERNS in tools/baselines/common.py "
            "with a one-line reason; that edit is visible in the baseline "
            f"spec freeze. Permitted today: {list(ALLOWED_PATTERNS)}")


def read_json(path, *, exemption=None):
    refuse_quarantined(path, exemption=exemption)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path}: no such file")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from None


def read_npz(path, *, exemption=None):
    """Load an npz EAGERLY, as a plain dict of arrays.

    `np.load` returns a lazy NpzFile that holds the archive open until it is
    closed, and every caller here wants two or three arrays and then forgets
    the handle. Returning a dict removes that whole class of bug -- it also
    makes the file deletable on Windows, which is how the leak surfaced. These
    artifacts are (arms, seeds, queries, candidates) float64, well under a
    megabyte; if one ever gets large enough for this to matter, that is a
    reason to add a lazy accessor deliberately, not to hand out handles by
    default.
    """
    refuse_quarantined(path, exemption=exemption)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path}: no such file")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ==========================================================================
# the test lock
# ==========================================================================
def split_for_mode(mode):
    """Which manifest split a mode is allowed to materialise.

    None of `placebo`, `fit` or `validation` reads the manifest's test side;
    only `frozen` reaches a test split, and `load_split` then demands a
    verified freeze manifest.

    `fit` was `discovery`, and read a separate 250-row pool. That pool is gone
    (section 14.0a) and the phase now reads validation -- the SAME rows
    `validation` reads. The two are kept apart anyway, because they are
    different phases: `fit` builds the method's target (a head set, a steering
    vector, a mask), `validation` scores candidate configurations by NLL. They
    produce different artifacts, the placebo contract distinguishes them, and
    section 2.4(7) requires their forward counts reported separately. Merging
    them because they came to share a data source would lose all three.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    return {"placebo": "validation", "fit": "validation",
            "validation": "validation", "frozen": "test_seed"}[mode]


def test_ids(manifest):
    """Every query id on the test side of the manifest, from BOTH routes.

    `entries` is the L2 common pool and `test_by_seed` the L3.1 per-seed draws.
    The draws are taken from the pool so the second is a subset of the first
    today, but that is a property of one builder version, not of the schema --
    reading only `entries` would be a check that silently narrows if the
    builder ever draws from elsewhere.
    """
    man = manifest if isinstance(manifest, dict) else read_json(manifest)
    ids = {e["query_id"] for e in man.get("entries", [])
           if e.get("role") == "pool"}
    for drawn in (man.get("test_by_seed") or {}).values():
        ids |= set(drawn)
    return ids


def assert_no_test_leak(query_ids, manifest, *, what="this run"):
    """The materialised queries must not intersect the test side. Always."""
    leaked = sorted(set(query_ids) & test_ids(manifest))
    if leaked:
        raise TestLockError(
            f"{what} materialised {len(leaked)} test queries "
            f"(first three {[q[:12] for q in leaked[:3]]}). Section 2.1 forbids "
            "any test prediction before the freeze; section 13.4(4) makes the "
            "freeze manifest the evidence. This check is on the ids actually "
            "loaded, not on what the manifest contains -- a hand-assembled id "
            "list is caught here and nowhere else.")
    return len(set(query_ids))


def load_queries(manifest_path, mode, *, demo_seed=None,
                 freeze_manifest=None, expected_roles=None):
    """The queries one run may see, with the lock enforced on the way out.

    Returns a list of {query_id, class_idx, ...} dicts.

    `expected_roles` binds the freeze to the artifacts THIS run loads and is
    forwarded to `load_split`. Callers must pass the label space and carriers
    they are about to read: binding only the query manifest closed the hole on
    the Method A probe path and left the frozen baselines able to run against
    a label space the freeze does not pin, which changes every argmax while
    the freeze still verifies.

    EVERY mode REQUIRES a demo_seed now, the validation-side ones as much as
    `frozen`. For `frozen` the reason was always PCW: each seed draws its own
    250, so "test_seed" with no seed is not under-specified but incoherent,
    and loading several at once would concatenate unrelated query sets. Since
    14.0a the same holds on the validation side -- each seed draws its own 144
    -- so there is no seed-independent set to return there either, and a
    default would hand a seed-44 run seed 42's queries without saying so.
    """
    split = split_for_mode(mode)
    man = read_json(manifest_path)
    if demo_seed is None:
        raise ValueError(
            f"mode={mode!r} needs exactly one demo_seed. Validation is drawn "
            "per prefix seed (section 14.0a) and the test draws always were, "
            "so no mode has a seed-independent query set to return.")
    # THE RUN MUST BE ATTRIBUTABLE TO A COMMIT. The freeze records a commit
    # rather than source hashes (14.0b-2 (t)), so a test prediction produced
    # from uncommitted code cannot be reproduced from anything. The probe got
    # this check when it was written; the frozen baseline runners reach test
    # through here, and covering one path and not the other would have left
    # H6 able to do what H1-H5 could not.
    #
    # BEFORE the lock, not after: if the tree is dirty no freeze manifest can
    # help, so answering "pass --freeze-manifest" would send the caller down
    # the wrong road. Test splits only -- validation work happens on a dirty
    # tree constantly and none of it is confirmatory.
    if split != "validation":
        from tools.build_query_manifest import attribution_blockers
        blockers = attribution_blockers()
        if blockers:
            raise PermissionError("; ".join(blockers))
    rows = list(load_split(manifest_path, split, demo_seed=demo_seed,
                           freeze_manifest=freeze_manifest,
                           expected_roles=expected_roles))
    ids = [r["query_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{mode}: duplicate query ids in the loaded split")
    if mode != "frozen":
        assert_no_test_leak(ids, man, what=f"mode={mode!r}")
    return rows


# ==========================================================================
# candidate space
# ==========================================================================
def load_candidate_space(label_space_path, *, model, query_manifest,
                         tokenizer=None):
    """The frozen per-model label space, verified against this run's model.

    Every optional argument given is checked -- see FrozenLabelSpace.load. The
    model is NOT optional here: a baseline reading another model's ids would
    index valid but wrong columns and report a perfectly plausible NLL.
    """
    if not model:
        raise ValueError(
            "a baseline run must name its model. Label token ids do not "
            "transfer between tokenizers, and the whole reason the label space "
            "is a separate artifact is that a set of Llama-2 ids once "
            "described a Llama-3.1 run with nothing recording the difference.")
    ls = FrozenLabelSpace.load(label_space_path, model=model,
                               query_manifest=query_manifest,
                               tokenizer=tokenizer)
    return ls


# ==========================================================================
# output schema
# ==========================================================================
class BaselineOutput:
    """Accumulates (arm, seed, query) candidate readouts and writes one npz.

    Cells are stored by key and materialised into a dense array at write time,
    so a runner that skips a cell fails loudly at `finish()` instead of writing
    a silently zero-filled row.
    """

    def __init__(self, name, *, arms, seeds, queries_by_seed, label_space,
                 mode):
        self.name = name
        self.arms = list(arms)
        self.seeds = [int(s) for s in seeds]
        # EACH SEED ITS OWN QUERIES. This was a single `queries` list walked
        # once per seed, which held while validation was a fixed common 144.
        # Since 14.0a it is not: seed 43 would have had to produce results for
        # seed 42's queries, or -- worse, because it yields a file instead of
        # an error -- its rows would have landed at positions belonging to
        # another seed's queries.
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(
                f"{name}: duplicate seeds {self.seeds}. Each would get its own"
                " slice of the array, so the same run would appear twice and"
                " any cross-seed count would be inflated by construction.")
        self.queries_by_seed = {int(k): list(v)
                                for k, v in queries_by_seed.items()}
        extra = sorted(set(self.queries_by_seed) - set(self.seeds))
        if extra:
            raise ValueError(
                f"{name}: queries given for undeclared seed(s) {extra}. They "
                "would be silently dropped, and a caller that meant to run "
                "them would get a file that looks complete.")
        for sd, rows_ in self.queries_by_seed.items():
            ids_ = [r["query_id"] for r in rows_]
            if len(set(ids_)) != len(ids_):
                dup = sorted({i for i in ids_ if ids_.count(i) > 1})
                raise ValueError(
                    f"{name}: seed {sd} lists {len(ids_) - len(set(ids_))} "
                    f"repeated query_id(s) (e.g. {dup[:3]}). One cell would "
                    "fill two positions, which reads downstream as two "
                    "observations of the same query rather than one.")
        missing_seeds = [s for s in self.seeds
                         if s not in self.queries_by_seed]
        if missing_seeds:
            raise ValueError(
                f"{name}: no queries given for seed(s) {missing_seeds}. Each "
                "seed draws its own set, so there is nothing to fall back to.")
        sizes = {s: len(self.queries_by_seed[s]) for s in self.seeds}
        if len(set(sizes.values())) != 1:
            raise ValueError(
                f"{name}: the seeds hold different numbers of queries "
                f"{sizes}. Each seed draws exactly the same COUNT (144 on "
                "validation, 250 on test) even though the rows differ, so an "
                "uneven count means one seed's load went wrong rather than "
                "that a ragged array is needed.")
        self.n_q = next(iter(sizes.values()))
        self.ls = label_space
        self.mode = mode
        self.n_cand = len(label_space.candidate_token_ids)
        if mode == "frozen" and len(self.seeds) != 1:
            raise ValueError(
                f"{name}: a frozen run holds one seed, got {self.seeds}. The "
                "array has a single query axis, and under PCW the three seeds "
                "do not share one -- stacking them would pair each row with "
                "another seed's query.")
        if "natural" not in self.arms:
            raise ValueError(
                f"{name}: every runner must emit a 'natural' arm computed "
                "through ITS OWN path -- that is the reference for gate P1 "
                "(placebo bitwise) and for gate P2 (path vs stock). Without it "
                "the placebo check has nothing meaningful to compare against.")
        self._ids_by_seed = {sd: {r["query_id"] for r in rows_}
                             for sd, rows_ in self.queries_by_seed.items()}
        self._cells = {}
        self._extra = {}

    def add(self, arm, seed, query_id, candidate_logits, *,
            full_logsumexp=None, full_argmax_token=None):
        if arm not in self.arms:
            raise KeyError(f"{self.name}: unknown arm {arm!r} "
                           f"(declared {self.arms})")
        v = np.asarray(candidate_logits, dtype=np.float64).reshape(-1)
        if v.shape[0] != self.n_cand:
            raise ValueError(
                f"{self.name}: arm {arm!r} produced {v.shape[0]} values but the "
                f"label space has {self.n_cand} candidates. The readout must be "
                "the frozen candidate slice, in the label space's class order.")
        if not np.all(np.isfinite(v)):
            raise ValueError(
                f"{self.name}: arm {arm!r} query {query_id[:12]} produced "
                f"{int((~np.isfinite(v)).sum())} non-finite candidate logits. A "
                "NaN here becomes a NaN NLL, which every downstream comparison "
                "would propagate silently.")
        if int(seed) not in self.queries_by_seed:
            raise KeyError(
                f"{self.name}: arm {arm!r} wrote a cell for seed {seed}, which"
                f" this run does not carry (declared {self.seeds}).")
        if query_id not in self._ids_by_seed[int(seed)]:
            raise KeyError(
                f"{self.name}: arm {arm!r} wrote query {query_id[:12]} under "
                f"seed {seed}, which is not one of that seed's "
                f"{len(self._ids_by_seed[int(seed)])} queries. Each seed draws"
                " its own set (14.0a); the cell would sit in _cells and be "
                "dropped by _dense() without a word.")
        key = (arm, int(seed), query_id)
        if key in self._cells:
            raise KeyError(f"{self.name}: cell {key[:2]} {key[2][:12]} written "
                           "twice; the second write would be invisible")
        self._cells[key] = (v, full_logsumexp, full_argmax_token)

    def _n_distinct(self):
        """Distinct query_ids across all seeds. Fewer than S * n_q whenever
        the per-seed draws overlap, which they do -- and which is why the
        inference clusters on query_id rather than treating cells as
        independent."""
        return len({q["query_id"] for s in self.seeds
                    for q in self.queries_by_seed[s]})

    def note(self, key, value):
        """Runner-specific metadata for the meta json (grids, chosen values)."""
        self._extra[key] = value

    def _dense(self):
        A, S, Q = len(self.arms), len(self.seeds), self.n_q
        logits = np.full((A, S, Q, self.n_cand), np.nan)
        lse = np.full((A, S, Q), np.nan)
        amax = np.full((A, S, Q), -1, dtype=np.int64)
        missing = []
        for ai, arm in enumerate(self.arms):
            for si, seed in enumerate(self.seeds):
                for qi, q in enumerate(self.queries_by_seed[seed]):
                    cell = self._cells.get((arm, seed, q["query_id"]))
                    if cell is None:
                        missing.append((arm, seed, q["query_id"][:12]))
                        continue
                    v, l, a = cell
                    logits[ai, si, qi] = v
                    if l is not None:
                        lse[ai, si, qi] = float(l)
                    if a is not None:
                        amax[ai, si, qi] = int(a)
        if missing:
            raise ValueError(
                f"{self.name}: {len(missing)} of {A * S * Q} cells were never "
                f"written (first three {missing[:3]}). A dense array filled "
                "where a cell is missing would be indistinguishable from a "
                "cell that legitimately read zero.")
        return logits, lse, amax

    # The sidecar fields the validation-selection check demands (see
    # freeze_baseline_spec.SIDECAR_REQUIRED). Required HERE so a runner cannot
    # omit one and only find out when the freeze refuses months later.
    #
    # ⚠ The list used to stop at the first five, while the selection check also
    # wanted query_manifest_sha256 and label_space_sha256 -- so a runner could
    # write a file that landed fine and was rejected at freeze time. Those two
    # are not asked of the runner at all now: it supplies the PATHS and
    # `finish` hashes them, which is one fewer thing to get wrong and removes
    # the possibility of a recorded hash that does not match the file used.
    SIDECAR_FROM_RUNNER = ("model", "task", "K", "dtype",
                           "attn_implementation", "query_manifest",
                           "label_space")

    def finish(self, out_dir, *, meta):
        missing = [k for k in self.SIDECAR_FROM_RUNNER if meta.get(k) is None]
        if missing:
            raise ValueError(
                f"{self.name}: meta records no {missing}. A result whose "
                "sidecar cannot state the model, task, K, precision and "
                "attention implementation cannot later be shown to be the "
                "registered run -- and a configuration selected from it would "
                "be selected from an unattributable file.")
        logits, lse, amax = self._dense()
        d = Path(out_dir)
        # `out_dir` is a DIRECTORY -- the file names are built below, so a
        # caller who passes `.../foo.npz` gets a DIRECTORY named `foo.npz`
        # holding `baseline_<name>_<mode>.npz`, and every downstream command
        # looks for the file at the path the caller typed. That happened on
        # 2026-09-08 to the TSLA validation run: 40 GPU-minutes landed at a
        # path nothing reads. mkdir cannot tell the two intents apart, so the
        # check has to be here, and it has to be by SUFFIX -- the only signal
        # the argument carries about which one the caller meant.
        if d.suffix in (".npz", ".json"):
            raise SystemExit(
                f"{self.name}: --out is a DIRECTORY, and {out_dir!r} names a "
                f"file. Pass `--out {d.parent}` -- this run would then write "
                f"{d.parent}/baseline_{self.name}_{self.mode}.npz and .json. "
                "Passing the file name instead creates a directory of that "
                "name and hides the result one level down.")
        d.mkdir(parents=True, exist_ok=True)
        # The seed goes in the NAME for frozen runs, or the three per-seed runs
        # would overwrite one another and the last one home would look like the
        # whole test set.
        stem = f"baseline_{self.name}_{self.mode}"
        if self.mode == "frozen":
            stem += f"_seed{self.seeds[0]}"
        npz = d / f"{stem}.npz"
        np.savez_compressed(
            npz,
            candidate_logits=logits.astype(np.float64),
            full_logsumexp=lse.astype(np.float64),
            full_argmax_token=amax,
            arms=np.array(self.arms),
            seeds=np.array(self.seeds, dtype=np.int64),
            # [S, Q], not [Q]: the rows differ per seed, so a single
            # vector could only have described one of them while reading as
            # though it described the file.
            query_ids=np.array([[q["query_id"]
                                 for q in self.queries_by_seed[s]]
                                for s in self.seeds]),
            gold_class=np.array([[int(q["class_idx"])
                                  for q in self.queries_by_seed[s]]
                                 for s in self.seeds],
                                dtype=np.int64),
            candidate_classes=np.array(self.ls.eligible_classes,
                                       dtype=np.int64),
            candidate_token_ids=np.array(self.ls.candidate_token_ids,
                                         dtype=np.int64),
        )
        full = dict(meta)
        full.update({
            "schema_version": SCHEMA_VERSION,
            "baseline": self.name,
            "mode": self.mode,
            # computed HERE from the paths the run actually used, so a
            # recorded hash cannot disagree with the file that was read
            "query_manifest_sha256": file_sha256(meta["query_manifest"]),
            "label_space_sha256": file_sha256(meta["label_space"]),
            "arms": self.arms,
            "demo_seeds": self.seeds,
            "n_queries_per_seed": self.n_q,
            "n_distinct_queries": self._n_distinct(),
            # S * n_per_seed - n_distinct. NOT n_per_seed - n_distinct, which
            # is negative, and NOT the number of queries that appear in more
            # than one seed -- a query in all three contributes two repeats.
            "n_repeated_cells": (len(self.seeds) * self.n_q
                                 - self._n_distinct()),
            "n_candidates": self.n_cand,
            "candidate_classes": self.ls.eligible_classes,
            "label_space_model": self.ls.model,
            "label_space_provenance": self.ls.data["provenance"],
            "npz_sha256": file_sha256(npz),
            "runner": self._extra,
        })
        from tools.icl_common import run_provenance
        full["provenance"] = run_provenance()
        js = d / f"{stem}.json"
        js.write_text(json.dumps(full, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        return npz, js


# ==========================================================================
# the two placebo gates
# ==========================================================================
def placebo_bitwise(placebo, reference):
    """Section 11.9: the placebo's candidate logits, element for element.

    For the five MODEL-SIDE runners the reference is the SAME full-prompt
    natural forward -- the stock path, not a same-path natural. Deep-Thinking
    is the one registered exception and passes its own prefix-cache reference
    (see `DT_PATH_TOLERANCE`).
    """
    a = np.asarray(placebo)
    b = np.asarray(reference)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    n = int((a != b).sum())
    if n == 0:
        return True, f"all {a.size} candidate logits identical"
    worst = float(np.max(np.abs(a - b)))
    return False, (f"{n}/{a.size} entries differ, max |delta| {worst:.3e}. The "
                   "neutral parameter is not neutral: the adapter is doing "
                   "arithmetic that does not cancel, or it is not reusing the "
                   "stock op sequence.")


# Section 11.9's only numeric allowance, and it belongs to Deep-Thinking alone:
# natural-cache vs full-prompt natural, max |delta| on candidate log-probs.
# baseline_under_review.md section 3.7 fixes it at 0.02 with 100% argmax
# agreement. It is NOT a general escape hatch -- a runner that cannot meet
# `placebo_bitwise` needs a preregistration revision, not this constant.
DT_PATH_TOLERANCE = 0.02


def dt_path_agreement(path_natural, stock_natural, tol=DT_PATH_TOLERANCE):
    """Deep-Thinking only: the prefix-cache path against the full prompt.

    Registered as a numeric check because the two genuinely chunk the sequence
    differently; every other runner is bitwise.
    """
    a = np.asarray(path_natural, dtype=np.float64)
    b = np.asarray(stock_natural, dtype=np.float64)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    worst = float(np.max(np.abs(a - b))) if a.size else 0.0
    agree = float(np.mean(np.argmax(a, axis=-1) == np.argmax(b, axis=-1))) \
        if a.size else 1.0
    ok = worst <= tol and agree == 1.0
    return ok, (f"max |delta| {worst:.3e} (registered tol {tol}), candidate "
                f"argmax agreement {agree:.4f}")


# ==========================================================================
# shared CLI
# ==========================================================================
REQUIRED_FOR_A_RUN = ("mode", "query_manifest", "label_space", "out", "attn")


def base_parser(name, description):
    """The shared CLI.

    Nothing is `required=True`, because `--self-test` must work with no other
    argument: fixtures run on a machine with no manifest, no label space and no
    model, which is the only environment I can check them in before they reach
    the server. The requirement is enforced in `require_run_args` AFTER the
    self-test branch, so a real run is still refused without them -- argparse
    would otherwise make the fixtures unrunnable, which is how a gate quietly
    stops being run at all.
    """
    p = argparse.ArgumentParser(prog=f"run_{name}.py", description=description)
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    # tools/model_args' flags, so a SelfExtend model reaches the baselines
    p.add_argument("--method", choices=("vanilla", "selfextend"),
                   default="vanilla")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--neighbor-size", type=int, default=1024)
    p.add_argument("--dtype", choices=("bfloat16", "float32"),
                   default="bfloat16")
    # the attention kernel (tools/model_args --attn): spelled by every real
    # run -- REQUIRED_FOR_A_RUN holds it, the self-test needs none of these
    p.add_argument("--attn", choices=("eager", "sdpa"), default=None,
                   help="eager or sdpa, the kernel the cell runs (prereg 14.0b-26)")
    p.add_argument("--query-manifest")
    p.add_argument("--label-space")
    p.add_argument("--demo-seeds", default="42,43,44")
    p.add_argument("--freeze-manifest", default=None)
    p.add_argument("--calibration-dir", default=None,
                   help="directory holding the calibration jsonl whose header "
                        "and prefix this run must reproduce")
    p.add_argument("--out")
    p.add_argument("--self-test", action="store_true",
                   help="run this runner's CPU fixtures and exit")
    return p


def require_run_args(args):
    missing = [f"--{k.replace('_', '-')}" for k in REQUIRED_FOR_A_RUN
               if not getattr(args, k, None)]
    if missing:
        raise SystemExit(f"a run needs {', '.join(missing)}. "
                         "(--self-test needs none of them.)")
    return args


def parse_seeds(spec, mode="validation"):
    """The seeds one run may use -- which depends on the mode.

    validation / placebo / fit : all three seeds MAY run in one process, but
      not over one query set. Since section 14.0a each seed draws its OWN 144,
      so the caller must ask load_queries for each seed separately and key its
      output on (seed, query_id). The per-query cross-seed mean that the old
      shared set allowed is no longer defined -- 13.6.3 replaced it with cells
      clustered on query_id.

    frozen : EXACTLY ONE. Under PCW each seed drew its own 250 queries
      (section 2.2), so there is no shared query axis across seeds and one
      process must cover one seed. The three runs are merged afterwards by
      (seed, query_id). Section 11 says the same thing about `--split
      test_seed`.
    """
    seeds = [int(s) for s in str(spec).split(",") if s.strip()]
    if mode == "frozen":
        if len(seeds) != 1:
            raise ValueError(
                f"mode='frozen' takes exactly one demo seed, got {seeds}. "
                "Under PCW the three seeds have DIFFERENT query sets "
                "(Q_42, Q_43, Q_44 of 250 each), so a single output array with "
                "one query axis cannot represent them; running them together "
                "would concatenate unrelated sets. Run once per seed and merge "
                "on (seed, query_id).")
        if seeds[0] not in REGISTERED_DEMO_SEEDS:
            raise ValueError(f"demo seed {seeds[0]} is not one of the "
                             f"registered {list(REGISTERED_DEMO_SEEDS)}")
        return seeds
    if tuple(seeds) != REGISTERED_DEMO_SEEDS:
        raise ValueError(
            f"demo seeds {seeds} are not the registered {list(REGISTERED_DEMO_SEEDS)}. "
            "Section 2.1 fixes them and forbids adding a fourth; a baseline "
            "run on a different seed set is not comparable with the method.")
    return seeds


def sha256_bytes(b) -> str:
    return hashlib.sha256(b).hexdigest()


def config_arm_name(config):
    """The arm label one configuration must be written under. Deterministic.

    Lives here, not in the freeze tool, because BOTH sides need it and a
    convention with two homes is how this project once got two different
    answers for one class space (working rules 2.6.2). BC wrote its arm as "bc"
    while the selection validator looked for `config_arm_name({})` = "default",
    so its NPZ would have been refused for an arm it never named.
    """
    if not config:
        return "default"
    return "_".join(f"{k}={config[k]}" for k in sorted(config))
