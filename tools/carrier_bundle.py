"""Nine carrier sets in two artifacts, and the reasons they are two.

Section 13.6.3 needs SIX head sets for the gate -- H_sF, one per (seed, fold)
-- and section 13.6.3's closing note needs THREE more for after GO, chosen on
each seed's whole 144. That is nine sets, and they must not travel together:

  * the GATE must never see the full-validation sets. Those are chosen on
    every query including the fold the gate evaluates on, so a runner that
    could reach them could evaluate a fold under heads selected using it.
    Separate files make that a missing argument rather than a discipline;
  * the POST-GO configuration must not silently fall back to a fold set,
    which is chosen on 72 queries and is the wrong object for a final
    configuration.

So `scope` is recorded IN the bundle and required BY the reader, and each
side refuses the other by name rather than by shape -- the two have the same
shape, which is exactly why shape cannot be the check.

WHAT IS CHECKED, beyond each set being a valid carrier:

  * the key set is exactly right for the scope. Six keys s<seed>_f<fold>, or
    three keys s<seed>, with the registered seeds and no others;
  * every set records the query ids it was DISCOVERED on, and those rows are
    the ones the scope says they should be: fold f of that seed, or that
    seed's whole validation set. A set discovered on the wrong rows is the
    defect that makes cross-fitting a formality, and nothing about the head
    list itself reveals it;
  * for the fold scope, a seed's two folds are disjoint and their union is
    that seed's validation set -- the same partition the statistical core
    receives, checked here rather than assumed.

⚠ Two folds of one seed selecting the SAME heads is NOT a fault. It is a
legitimate outcome -- the top-8 by score_margin on overlapping data are
correlated -- and refusing it would be the kind of too-strict gate this
project has already had to withdraw twice.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools import carrier_schema
from tools.prereg_config import N_VALIDATION, REGISTERED_SEEDS

BUNDLE_VERSION = 1
SCOPE_FOLD = "fold"                 # the six H_sF the gate cross-fits with
SCOPE_FULL = "full_validation"      # the three H_s for the post-GO config
SCOPES = (SCOPE_FOLD, SCOPE_FULL)

BUNDLE_REQUIRED = ("bundle_version", "scope", "spec", "query_manifest_sha256",
                   "sets")


def fold_key(seed, fold):
    return f"s{int(seed)}_f{int(fold)}"


def full_key(seed):
    return f"s{int(seed)}"


def expected_keys(scope):
    if scope == SCOPE_FOLD:
        return {fold_key(s, f) for s in REGISTERED_SEEDS for f in (0, 1)}
    if scope == SCOPE_FULL:
        return {full_key(s) for s in REGISTERED_SEEDS}
    raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPES}")


def bundle_faults(bundle, *, scope, validation_by_seed=None, fold_by_seed=None,
                  query_manifest_sha256=None, model=None, n_validation=None,
                  task=None, K=None, arch=None):
    """Why this bundle may not be used at `scope`. Empty means it may.

    `validation_by_seed` maps seed -> the ordered query ids of that seed's
    validation set; `fold_by_seed` maps seed -> a list of 0/1 the same length.
    Both are optional, and when they are absent the row checks are SKIPPED
    and said to be skipped -- a caller that omits them gets less checking,
    not a silent pass. The gate passes both.

    RETURNS faults, never raises: a malformed artifact is the state this
    reports on, and an exception here reaches callers catching nothing.
    """
    if scope not in SCOPES:
        return [f"unknown scope {scope!r}; expected one of {SCOPES}"]
    if not isinstance(bundle, dict):
        return [f"the bundle is {type(bundle).__name__}, expected an object"]
    bad = [f"no {k!r}" for k in BUNDLE_REQUIRED if k not in bundle]
    if bad:
        return bad
    if int(bundle.get("bundle_version", -1)) != BUNDLE_VERSION:
        return [f"bundle_version {bundle.get('bundle_version')!r} is not "
                f"v{BUNDLE_VERSION}; rebuild rather than reinterpret"]

    # SCOPE FIRST, and by name. The two bundles have the same shape -- nine
    # sets split six and three -- so a shape check cannot tell them apart,
    # and the whole point of the split is that the gate must not read the
    # full-validation sets.
    got_scope = str(bundle["scope"])
    if got_scope != scope:
        return [f"this is a {got_scope!r} bundle and a {scope!r} one was "
                "asked for. The full-validation sets are chosen on every "
                "query INCLUDING the fold the gate evaluates on, so reading "
                "them there would evaluate a fold under heads selected using "
                "it"]

    sets = bundle["sets"]
    if not isinstance(sets, dict):
        return [f"'sets' is {type(sets).__name__}, expected an object keyed "
                "by set name"]
    want = expected_keys(scope)
    if set(sets) != want:
        missing, extra = sorted(want - set(sets)), sorted(set(sets) - want)
        return [f"the {scope!r} bundle holds {len(sets)} sets; expected "
                f"{len(want)} ({sorted(want)}). missing={missing} "
                f"extra={extra}"]

    if (query_manifest_sha256 is not None
            and bundle["query_manifest_sha256"] != query_manifest_sha256):
        bad.append(
            f"query_manifest_sha256 {str(bundle['query_manifest_sha256'])[:12]}"
            f" != this run's {str(query_manifest_sha256)[:12]}; the sets were "
            "discovered against a different set of draws")

    for key in sorted(sets):
        entry = sets[key]
        if not isinstance(entry, dict):
            bad.append(f"{key}: entry is {type(entry).__name__}, expected an "
                       "object with 'carrier' and 'discovery_query_ids'")
            continue
        carrier = entry.get("carrier")
        if not isinstance(carrier, dict):
            bad.append(f"{key}: no 'carrier' object")
            continue
        # A fold set is chosen on HALF the validation rows (13.6.3); the
        # schema's registered 144 is the full-validation number, so the scope
        # says which to expect rather than the check being dropped.
        n_val = int(n_validation) if n_validation is not None else N_VALIDATION
        bad += [f"{key}: {b}" for b in carrier_schema.validate(
            carrier, query_manifest_sha256=query_manifest_sha256, model=model,
            expect_discovery_queries=(n_val // 2 if scope == SCOPE_FOLD
                                      else n_val),
            expect_task=task, expect_K=K, expect_arch=arch)]
        ids = entry.get("discovery_query_ids")
        if not isinstance(ids, list) or not ids:
            bad.append(f"{key}: no 'discovery_query_ids'. Nothing about a "
                       "head list reveals which rows chose it, so without "
                       "this the cross-fit is a formality")

    if validation_by_seed is None or (scope == SCOPE_FOLD
                                      and fold_by_seed is None):
        bad.append(
            "[not checked] discovery rows: the caller passed no "
            + ("validation_by_seed" if validation_by_seed is None
               else "fold_by_seed")
            + ", so which rows chose each set was not verified. The gate "
              "passes both; this note is here so a silent skip cannot be "
              "mistaken for a pass")
        return bad

    for seed in REGISTERED_SEEDS:
        vids = list(validation_by_seed.get(seed)
                    or validation_by_seed.get(str(seed)) or [])
        if not vids:
            bad.append(f"seed {seed}: no validation rows supplied")
            continue
        if scope == SCOPE_FULL:
            got = _ids(sets, full_key(seed))
            if got is not None and got != vids:
                bad.append(
                    f"{full_key(seed)}: discovered on {len(got)} rows, not "
                    f"this seed's {len(vids)} validation rows. A full-scope "
                    "set chosen on anything less is a fold set with the "
                    "wrong label")
            continue
        folds = list(fold_by_seed.get(seed) or fold_by_seed.get(str(seed))
                     or [])
        if len(folds) != len(vids):
            bad.append(f"seed {seed}: {len(folds)} fold labels for "
                       f"{len(vids)} validation rows")
            continue
        for f in (0, 1):
            want_rows = [q for q, ff in zip(vids, folds) if int(ff) == f]
            got = _ids(sets, fold_key(seed, f))
            if got is None:
                continue
            if got != want_rows:
                bad.append(
                    f"{fold_key(seed, f)}: discovered on {len(got)} rows that "
                    f"are not fold {f} of seed {seed} ({len(want_rows)} rows)."
                    " Choosing heads on the fold they are then evaluated on "
                    "is what the cross-fit exists to prevent, and the head "
                    "list itself cannot show it")
        a, b = (_ids(sets, fold_key(seed, f)) for f in (0, 1))
        if a is not None and b is not None:
            if set(a) & set(b):
                bad.append(f"seed {seed}: the two folds share "
                           f"{len(set(a) & set(b))} discovery rows")
            if sorted(a + b) != sorted(vids):
                bad.append(f"seed {seed}: the two folds' rows do not partition "
                           f"the validation set ({len(a) + len(b)} vs "
                           f"{len(vids)})")
    return bad


def _ids(sets, key):
    e = sets.get(key)
    if not isinstance(e, dict):
        return None
    ids = e.get("discovery_query_ids")
    return list(ids) if isinstance(ids, list) else None


def load(path, *, scope, **kw):
    """Read a bundle and refuse it unless it may be used at `scope`."""
    p = Path(path)
    try:
        bundle = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{p}: unreadable as a carrier bundle -- "
                         f"{type(e).__name__}: {e}")
    bad = bundle_faults(bundle, scope=scope, **kw)
    if bad:
        raise ValueError(f"{p} is not a usable {scope!r} carrier bundle:\n  "
                         + "\n  ".join(bad))
    return bundle


def summary(bundle):
    sets = bundle.get("sets") or {}
    return (f"{bundle.get('scope')!r} bundle, {len(sets)} sets: "
            + ", ".join(f"{k}={len((sets[k].get('carrier') or {}).get('heads') or [])}h"
                        for k in sorted(sets)))


# ==========================================================================
# building the nine sets from one pass of per-query scores
# ==========================================================================
def heads_for(bundle, seed, top_n=None, *, key=None):
    """[(layer, head)] for one seed. THE ONLY place a top-N cut is taken.

    `top_n=None` returns the FROZEN set section 2.4(2) selected -- what every
    registered arm uses. Any other value slices the bundle's own `ranking`,
    which is the same score in the same order, just a different cut, and is
    therefore EXPLORATORY.

    One implementation because two runners want it: `run_k0_receiver` sweeps
    head count for the visibility arm and `run_viability_forward` sweeps it
    for the write arms. Re-ranking in either would be a second copy of
    2.4(2)'s rule, which is how the class-space convention drifted once
    already (working rules 2.6.2).
    """
    carrier = bundle["sets"][key or full_key(seed)]["carrier"]
    if top_n is None:
        return [(int(l), int(h)) for l, h in carrier["heads"]]
    rank = carrier.get("ranking")
    if not rank:
        raise ValueError(
            "the bundle carries no 'ranking', so a top-N other than the "
            "frozen set cannot be cut from it without re-running discovery")
    if top_n > len(rank):
        raise ValueError(f"top_n={top_n} exceeds the {len(rank)} ranked heads")
    pairs = [(int(e["layer"]), int(e["head"])) for e in rank[:top_n]]
    listed = [(int(l), int(h)) for l, h in carrier["heads"]]
    if len(set(listed)) != len(listed):
        raise ValueError(
            f"the frozen head list has {len(listed)} entries but only "
            f"{len(set(listed))} distinct; a carrier set with a repeat is "
            "malformed, and it would also silently shorten the agreement "
            "check below")
    # AGAINST THE LIST LENGTH, not the set's. Comparing top_n to len(set(...))
    # let a frozen list of eight copies of one head skip this check entirely,
    # because the set had size 1 and 8 != 1.
    if top_n == len(listed) and set(pairs) != set(listed):
        raise ValueError(
            f"the first {top_n} ranked heads are not the frozen set; the "
            "ranking and the frozen selection disagree, so one of them is "
            "not what section 2.4(2) produced")
    return pairs


def _rank_and_pick(scores, n_layers, n_attn):
    """(ranking, heads) for one mean score matrix [layer, head].

    RAW score_margin, descending, ties broken by (layer, head) so the result
    does not depend on numpy's sort stability. Section 2.4(2) ranks on the
    raw score and not a layer-normalised one -- normalising rescales every
    layer to the same spread and turns "which heads write the margin" into
    "which heads are unusual for their layer", which is a different question.
    """
    entries = [{"layer": int(l), "head": int(h), "score": float(scores[l][h])}
               for l in range(n_layers) for h in range(n_attn)]
    entries.sort(key=lambda e: (-e["score"], e["layer"], e["head"]))
    heads = [(e["layer"], e["head"]) for e in entries[:carrier_schema.N_TOP]]
    return entries, heads


def build_from_scores(scores_by_seed, validation_by_seed, fold_by_seed, *,
                      model, query_manifest_sha256,
                      n_layers=carrier_schema.N_LAYERS,
                      n_attn=carrier_schema.N_ATTN_HEADS,
                      n_kv=carrier_schema.N_KV_HEADS,
                      task=carrier_schema.TASK, K=carrier_schema.K,
                      carrier_impl="gqa_group_v"):
    """The two bundles, from ONE pass of per-query scores.

    `scores_by_seed[seed]` is [query, layer, head] in the same row order as
    `validation_by_seed[seed]`.

    DISCOVERY DOES NOT DOUBLE WITH THE FOLDS. score_margin is a per-query
    quantity and every set is a MEAN of it over some subset of rows, so the
    nine sets are nine averages of one forward pass -- 13.6.3 says so, and
    building them here rather than in the runner is what keeps that true. A
    runner that recomputed per fold would be free to use a different set of
    forwards for each, and nothing downstream could tell.

    The fold sets are averaged over that fold's rows ONLY. That is the whole
    cross-fit: a mean taken over all 144 and then labelled "fold 0" is the
    leak, it produces a perfectly valid carrier, and the discovery_query_ids
    written alongside are the only record that says which happened.
    """
    fold_sets, full_sets = {}, {}
    for seed in REGISTERED_SEEDS:
        rows = list(validation_by_seed.get(seed)
                    or validation_by_seed.get(str(seed)) or [])
        labels = list(fold_by_seed.get(seed) or fold_by_seed.get(str(seed))
                      or [])
        sc = scores_by_seed.get(seed)
        if sc is None:
            sc = scores_by_seed.get(str(seed))
        if sc is None or not rows or len(labels) != len(rows):
            raise ValueError(
                f"seed {seed}: need scores, {len(rows)} validation rows and "
                f"a fold label for each (got {len(labels)} labels)")
        sc = [[list(map(float, r)) for r in q] for q in sc]
        if len(sc) != len(rows):
            raise ValueError(f"seed {seed}: {len(sc)} score rows for "
                             f"{len(rows)} queries")

        def _mean(idx):
            return [[sum(sc[q][l][h] for q in idx) / len(idx)
                     for h in range(n_attn)] for l in range(n_layers)]

        for f in (0, 1):
            idx = [i for i, ff in enumerate(labels) if int(ff) == f]
            if not idx:
                raise ValueError(f"seed {seed}: fold {f} has no rows")
            ranking, heads = _rank_and_pick(_mean(idx), n_layers, n_attn)
            fold_sets[fold_key(seed, f)] = _entry(
                ranking, heads, [rows[i] for i in idx], model=model,
                query_manifest_sha256=query_manifest_sha256,
                n_layers=n_layers, n_attn=n_attn, n_kv=n_kv,
                carrier_impl=carrier_impl, task=task, K=K)
        ranking, heads = _rank_and_pick(_mean(range(len(rows))), n_layers,
                                        n_attn)
        full_sets[full_key(seed)] = _entry(
            ranking, heads, rows, model=model,
            query_manifest_sha256=query_manifest_sha256, n_layers=n_layers,
            n_attn=n_attn, n_kv=n_kv, carrier_impl=carrier_impl, task=task,
            K=K)

    def _bundle(scope, sets):
        return {"bundle_version": BUNDLE_VERSION, "scope": scope,
                "spec": "prereg_method_A.md section 13.6.3 -- "
                        + ("the six H_sF the gate cross-fits with"
                           if scope == SCOPE_FOLD else
                           "the three H_s for the post-GO configuration"),
                "query_manifest_sha256": query_manifest_sha256, "sets": sets}

    return _bundle(SCOPE_FOLD, fold_sets), _bundle(SCOPE_FULL, full_sets)


def _entry(ranking, heads, rows, *, model, query_manifest_sha256, n_layers,
           n_attn, n_kv, carrier_impl, task=carrier_schema.TASK,
           K=carrier_schema.K):
    groups, dragged = carrier_schema.expected_dragged(heads, n_attn, n_kv)
    return {
        "carrier": {
            "schema_version": carrier_schema.SCHEMA_VERSION,
            "spec": "prereg_method_A.md section 2.4(2)",
            "model": str(model), "task": str(task),
            "K": int(K), "carrier_impl": carrier_impl,
            "score": carrier_schema.SCORE_DEFINITION,
            "n_layers": n_layers, "n_attn_heads": n_attn, "n_kv_heads": n_kv,
            "query_manifest_sha256": query_manifest_sha256,
            "n_discovery_queries": len(rows),
            "n_random_variants": carrier_schema.N_RANDOM_VARIANTS,
            "prefix_seeds": list(carrier_schema.PREFIX_SEEDS),
            "prefix_assignment": carrier_schema.PREFIX_ASSIGNMENT,
            "intervention": carrier_schema.INTERVENTION,
            "random_seed_rule": carrier_schema.RANDOM_SEED_RULE,
            "random_seed_salt": carrier_schema.RANDOM_SEED_SALT,
            "ranking": ranking,
            "heads": [list(h) for h in heads],
            "kv_groups": [list(g) for g in groups],
            "dragged_heads": [list(d) for d in dragged],
        },
        "discovery_query_ids": list(rows),
    }
