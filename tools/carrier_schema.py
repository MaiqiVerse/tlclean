"""The Method A carrier-discovery artifact: schema and validator. No torch.

WRITTEN BEFORE THE RUNNER, DELIBERATELY. The viability gate's first version
"verified" the carriers by hashing the file and counting eight entries, which
eight duplicates or eight invented pairs satisfy. A hash pins bytes; a count
pins a length; neither says the file describes the top-8 that section 2.4(2)
froze. So the schema is fixed here first and the runner is written to it,
rather than the schema being read off whatever the runner happened to emit.

WHAT SECTION 2.4(2) ACTUALLY FIXES, and therefore what has to be checkable:

  * the score, S_lh, on 250 discovery queries with prefix seeds cycling
    42/43/44 and M=5 random_uuid variants each -- so the artifact records which
    discovery split it came from, by hash, and how many queries and variants
    were used;
  * the FULL ranking over all 32x32 query heads, from which the top-8 are the
    first eight, ties broken toward smaller (layer, head). Recording the
    ranking makes "these are the top 8" a checkable claim rather than an
    assertion: the validator re-derives the selection from the scores;
  * the mapping to KV groups. Under GQA the eight QUERY heads drag in their
    whole groups, and section 3.1's group implementation touches the union.
    Those are different sets and conflating them is how "8 heads" turns into
    32 without anyone noticing, so both are recorded and the containment is
    checked.

NOTHING HERE IS A TOLERANCE. Every check is categorical -- membership, range,
uniqueness, ordering, containment -- so none of them needs a threshold anyone
could tune (working rules 11b).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMA_VERSION = 3
N_TOP = 8                    # section 2.4(2)
N_DISCOVERY = 144            # section 2.4(1): the validation set, all
                             # of it, once per prefix seed
N_RANDOM_VARIANTS = 5        # section 2.4(2), M=5
GROUP_IMPL = "gqa_group_v"   # section 3.1

# THE REST OF SECTION 2.4(2), which the first version left in comments. A
# runner using the wrong prefix seeds, a different randomisation, or another
# task would still have produced a schema-valid artifact -- which defeats the
# point of fixing the schema before the runner. These are the experiment's
# IDENTITY, so they are fields with exact expected values, not prose.
PREFIX_SEEDS = (42, 43, 44)
PREFIX_ASSIGNMENT = ("each prefix seed independently over all 144 "
                     "validation queries")
INTERVENTION = "random_uuid"
RANDOM_SEED_SALT = 20260830
# ⚠ RAW string. Written as a normal literal, the `\0` separators become actual
# NUL characters in the value -- which then land in the JSON artifact and in
# every comparison. The prereg writes the rule with literal backslash-zero
# separators, so that is what this must contain. (working rules rule 5's failure
# mode, arriving through a .py file this time rather than a heredoc.)
RANDOM_SEED_RULE = (r'low 32 bits of SHA256("method_A_rand\0<prefix_seed>\0<query_id>\0<m>'
                    r'\0<20260830>")')
TASK = "trec_fine_per_class"
K = 5
# L3.1-8B. Recorded in the artifact and checked against it, so a run on another
# architecture cannot quietly reuse this schema.
N_LAYERS, N_ATTN_HEADS, N_KV_HEADS = 32, 32, 8
REGISTERED_MODEL = "meta-llama/Llama-3.1-8B"
# Another model's artifact records ITS architecture and the caller states
# what it expects (validate's expect_arch / expect_task); the registered
# numbers above are enforced only on the registered model's artifacts.

REQUIRED = ("schema_version", "spec", "model", "task", "K", "carrier_impl",
            "score", "n_layers", "n_attn_heads", "n_kv_heads",
            "query_manifest_sha256", "n_discovery_queries",
            "n_random_variants", "prefix_seeds", "prefix_assignment",
            "intervention", "random_seed_rule", "random_seed_salt", "ranking",
            "heads", "kv_groups", "dragged_heads")

SCORE_DEFINITION = (
    "S_lh = E_q[ <o_gold_qlh, W_U[y_q]> - (1/M) sum_m <o_rand_qlh, W_U[y_q]> ]"
)


def random_uuid_seed(prefix_seed, query_id, m, salt=RANDOM_SEED_SALT):
    r"""Low 32 bits of
    SHA256("method_A_rand\0<prefix_seed>\0<query_id>\0<m>\0<salt>").

    `prefix_seed` leads, and is not optional. Discovery now runs all three
    prefix seeds over the SAME 144 validation queries, so without it the M=5
    control prompts would carry bit-identical label sequences across seeds and
    the three runs' control terms would be correlated -- "independent per seed"
    in name only (prereg 2.4(2)).

    EXECUTABLE, and the runner must call this rather than reimplement it:
    the schema can check that an artifact DECLARES the rule, but only a
    shared implementation makes the declaration true of the run that
    produced it. Called by the
    runner and pinned by a hand-computed fixture, so `random_seed_rule` in the
    artifact describes what actually happened rather than what was intended.

    The separators are NUL bytes, matching `tools/prereg_ids._h`, which is what
    section 2.4(2)'s backslash-zero notation means.
    """
    nul = b"\x00"
    blob = (b"method_A_rand" + nul + str(int(prefix_seed)).encode("utf-8")
            + nul + str(query_id).encode("utf-8") + nul
            + str(int(m)).encode("utf-8") + nul
            + str(int(salt)).encode("utf-8"))
    return int.from_bytes(hashlib.sha256(blob).digest()[-4:], "big")


def group_of(head, n_attn=N_ATTN_HEADS, n_kv=N_KV_HEADS):
    """Which KV group a query head belongs to. g(h) = floor(h / (32/8))."""
    return int(head) // (int(n_attn) // int(n_kv))


def expected_dragged(heads, n_attn=N_ATTN_HEADS, n_kv=N_KV_HEADS):
    """Every query head the GROUP implementation touches, given the top-8.

    This is the number that has to appear in the report: section 3.1 writes to
    whole KV groups, so selecting eight query heads can move up to 32.
    """
    per = int(n_attn) // int(n_kv)
    groups = sorted({(int(l), group_of(h, n_attn, n_kv)) for l, h in heads})
    return groups, sorted({(l, g * per + j) for l, g in groups
                           for j in range(per)})


def validate(carrier, *, query_manifest_sha256=None, model=None,
             expect_discovery_queries=None, expect_task=None, expect_K=None,
             expect_arch=None):
    """Everything wrong with a carrier artifact. Empty list means it is sound.

    `query_manifest_sha256` and `model`, when given, are CHECKED -- the point
    is to tie the carriers to the pool they were discovered on and the model
    they were discovered for, neither of which the file can vouch for alone.
    """
    bad = []
    if not isinstance(carrier, dict):
        return ["the carrier artifact is not a JSON object"]
    missing = [k for k in REQUIRED if k not in carrier]
    if missing:
        return [f"missing {missing} (has {sorted(carrier)})"]

    if int(carrier["schema_version"]) != SCHEMA_VERSION:
        bad.append(f"schema v{carrier['schema_version']}, this code expects "
                   f"v{SCHEMA_VERSION}; rebuild rather than reinterpret")
    if carrier["carrier_impl"] != GROUP_IMPL:
        bad.append(f"carrier_impl {carrier['carrier_impl']!r} is not "
                   f"{GROUP_IMPL!r} (section 3.1 Method A-group)")
    if carrier["score"] != SCORE_DEFINITION:
        bad.append("the recorded score definition is not section 2.4(2)'s "
                   "frozen TL-score")
    # THE ARCHITECTURE. Registered for L3.1-8B; another model's artifact
    # records its own and the caller states what it expects (discover_carriers
    # passes the loaded config). With neither, only internal consistency.
    registered = str(carrier.get("model")) == REGISTERED_MODEL
    if expect_arch is not None:
        want_arch = tuple(int(x) for x in expect_arch)
    elif registered:
        want_arch = (N_LAYERS, N_ATTN_HEADS, N_KV_HEADS)
    else:
        want_arch = None
    if want_arch is not None:
        for key, want in zip(("n_layers", "n_attn_heads", "n_kv_heads"),
                             want_arch):
            if int(carrier[key]) != want:
                bad.append(f"{key}={carrier[key]}, expected {want}"
                           + (" (registered)" if expect_arch is None else ""))
    if int(carrier["n_kv_heads"]) <= 0 or \
            int(carrier["n_attn_heads"]) % int(carrier["n_kv_heads"]):
        bad.append(f"n_attn_heads {carrier['n_attn_heads']} is not a multiple "
                   f"of n_kv_heads {carrier['n_kv_heads']}")
    # 13.6.3 discovers a FOLD set on half the validation rows, so the caller
    # states what this set should have been chosen on. The registered 144 is
    # TREC's full-validation number and applies to the registered model's
    # artifacts by default; another task's count comes from its manifest.
    counts = [("n_random_variants", N_RANDOM_VARIANTS),
              ("K", K if expect_K is None else int(expect_K)),
              ("random_seed_salt", RANDOM_SEED_SALT)]
    if expect_discovery_queries is not None or registered:
        counts.insert(0, ("n_discovery_queries",
                          N_DISCOVERY if expect_discovery_queries is None
                          else int(expect_discovery_queries)))
    for key, want in counts:
        if int(carrier[key]) != want:
            bad.append(f"{key}={carrier[key]}, registered is {want}")
    # The protocol's identity, exactly as section 2.4(2) states it. The task
    # is TREC's on the registered model and the caller's otherwise.
    want_task = expect_task if expect_task is not None else (
        TASK if registered else str(carrier.get("task")))
    for key, want in (("task", want_task), ("intervention", INTERVENTION),
                      ("prefix_assignment", PREFIX_ASSIGNMENT),
                      ("random_seed_rule", RANDOM_SEED_RULE)):
        if str(carrier[key]) != want:
            bad.append(f"{key}={carrier[key]!r}, registered is {want!r}")
    if tuple(int(x) for x in carrier["prefix_seeds"]) != PREFIX_SEEDS:
        bad.append(f"prefix_seeds {carrier['prefix_seeds']} are not the "
                   f"registered {list(PREFIX_SEEDS)}")
    if model is not None and str(carrier["model"]) != str(model):
        bad.append(f"model {carrier['model']!r} != this run's {model!r}")
    if query_manifest_sha256 is not None and \
            carrier["query_manifest_sha256"] != query_manifest_sha256:
        bad.append(
            f"query_manifest_sha256 {str(carrier['query_manifest_sha256'])[:12]}"
            f" != the manifest on disk {query_manifest_sha256[:12]}; these "
            "carriers were not discovered on this validation set")

    # ---- the ranking, and the top-8 RE-DERIVED from it -------------------
    rank = carrier["ranking"]
    if not isinstance(rank, list) or not rank:
        bad.append("ranking is empty; 'these are the top 8' would be an "
                   "assertion rather than a claim anyone can check")
        return bad
    try:
        entries = [(int(r["layer"]), int(r["head"]), float(r["score"]))
                   for r in rank]
    except (TypeError, KeyError, ValueError) as e:
        bad.append(f"ranking entries are not {{layer, head, score}}: {e}")
        return bad
    n_expected = int(carrier["n_layers"]) * int(carrier["n_attn_heads"])
    if len(entries) != n_expected:
        bad.append(f"ranking has {len(entries)} entries; section 2.4(2) ranks "
                   f"all {n_expected} query heads")
    if len({(l, h) for l, h, _ in entries}) != len(entries):
        bad.append("ranking repeats a (layer, head)")
    for l, h, s in entries:
        if not 0 <= l < int(carrier["n_layers"]):
            bad.append(f"ranking has layer {l} outside "
                       f"[0, {carrier['n_layers']})")
            break
    for l, h, s in entries:
        if not 0 <= h < int(carrier["n_attn_heads"]):
            bad.append(f"ranking has head {h} outside "
                       f"[0, {carrier['n_attn_heads']})")
            break
    # isfinite, not `s != s`: the latter catches NaN only, and +/-inf would
    # have sailed through into an argmax.
    if not all(math.isfinite(s) for _l, _h, s in entries):
        bad.append("ranking holds a non-finite score (NaN or +/-inf)")
    # descending by score, ties toward the smaller (layer, head)
    key = [(-s, l, h) for l, h, s in entries]
    if key != sorted(key):
        bad.append("ranking is not sorted by descending score with ties "
                   "toward the smaller (layer, head) -- the frozen tie rule")

    heads = [(int(l), int(h)) for l, h in carrier["heads"]]
    if len(heads) != N_TOP:
        bad.append(f"{len(heads)} heads, section 2.4(2) freezes the top-{N_TOP}")
    if len(set(heads)) != len(heads):
        bad.append(f"the head set repeats an entry: {sorted(heads)}")
    derived = [(l, h) for l, h, _s in entries[:N_TOP]]
    if sorted(heads) != sorted(derived):
        bad.append(
            f"heads {sorted(heads)} are not the first {N_TOP} of the ranking "
            f"{sorted(derived)}. Eight plausible pairs are not the top eight, "
            "and this is the check that tells them apart.")

    # ---- the GQA mapping, kept distinct from the head set ----------------
    groups, dragged = expected_dragged(heads, int(carrier["n_attn_heads"]),
                                       int(carrier["n_kv_heads"]))
    got_g = sorted((int(l), int(g)) for l, g in carrier["kv_groups"])
    got_d = sorted((int(l), int(h)) for l, h in carrier["dragged_heads"])
    if got_g != groups:
        bad.append(f"kv_groups {got_g} != the groups the {N_TOP} query heads "
                   f"map into {groups}")
    if got_d != dragged:
        bad.append(f"dragged_heads has {len(got_d)} entries, the group "
                   f"implementation touches {len(dragged)}")
    if not set(heads) <= set(got_d):
        bad.append("the top-8 are not contained in the dragged set, which is "
                   "impossible if the mapping is right")
    return bad


def load(path, *, query_manifest=None, model=None):
    """Read and validate. Raises ValueError listing everything wrong."""
    p = Path(path)
    carrier = json.loads(p.read_text(encoding="utf-8"))
    sha = None
    if query_manifest is not None:
        sha = hashlib.sha256(Path(query_manifest).read_bytes()).hexdigest()
    bad = validate(carrier, query_manifest_sha256=sha, model=model)
    if bad:
        raise ValueError(f"{p.name} is not a valid carrier artifact:\n  "
                         + "\n  ".join(bad))
    return carrier


MAX_DRAGGED = N_TOP * (N_ATTN_HEADS // N_KV_HEADS)


def summary(carrier):
    """What the group implementation actually touches, for THIS head set.

    ⚠ The count is a property of the discovered heads, not a constant. Eight
    heads spread over eight distinct (layer, KV group) pairs drag in 32 query
    heads; eight heads inside two groups of one layer drag in 8. Until
    discovery has run, the only true statement is "at most 32" -- writing 32
    as though it were the answer is the section 29-32 mistake in the other
    direction.
    """
    heads = [(int(l), int(h)) for l, h in carrier["heads"]]
    groups, dragged = expected_dragged(heads, int(carrier["n_attn_heads"]),
                                       int(carrier["n_kv_heads"]))
    return (f"{len(heads)} top query heads over {len({l for l, _ in heads})} "
            f"layer(s) -> {len(groups)} KV group(s), {len(dragged)} query "
            f"heads touched by the group implementation "
            f"(at most {MAX_DRAGGED} for any 8-head set)")
