"""Gates for the L3.1 Method A viability gate. Pure numpy, a few seconds.

Every arm here corresponds to a way the gate could hand back a confident number
it has not earned:

  * the permutation and the interval could test DIFFERENT estimands -- they did,
    in the first version, and GO required both;
  * a reduced bootstrap or a skipped freeze could still print GO/STOP;
  * a self-consistent npz from the wrong experiment could pass every shape
    check;
  * gamma=0 could stop being the identity and nothing downstream would notice.

Run: python tools/test_method_a_viability.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.method_a_viability import (DECISIONS, DIAGNOSTIC,  # noqa: E402
                                      GAMMA_GRID,
                                      N_VALIDATION, VALIDATION_PER_CLASS,
                                      VERIFIED, apply_choice,
                                      assign_folds, bootstrap_T,
                                      cluster_index, cross_fit, decide,
                                      load_viability_npz,
                                      procedure_sign_flip_p, require_freeze,
                                      REGISTERED_SEEDS, seed_mean_effect,
                                      select_gamma)

NC, NS = 36, 3

# [S, Q]. Each seed draws its own 144 (section 14.0a), so the rows must
# DIFFER -- three identical rows would let a consumer that ignores the seed
# pass every arm -- and must OVERLAP, or every query cluster would have size
# one and the clustering the inference depends on would never be exercised.
# Seed si takes a window offset by 8, out of a pool of 160.
# Ids are DIGESTS, not sequential hex. assign_folds ranks within a class by
# query_id, so monotonic ids make that rank equal the position -- identically
# in every seed -- and the three folds come out the same. A digest's order
# within a class is arbitrary and differs by seed, which is what the fold
# machinery is supposed to face.
def _qid(k):
    import hashlib as _h
    return _h.sha256(str(k).encode()).hexdigest()


# Per class: a pool of 6, of which seed si takes 4 starting at si. The seeds
# therefore SHARE ids (clusters of size 2 and 3 exist) and hold DIFFERENT
# sets (a consumer ignoring the seed fails).
QIDS = np.array([[_qid(c * 1000 + (si + r) % 6)
                  for c in range(NC) for r in range(VALIDATION_PER_CLASS)]
                 for si in range(NS)])
CLS = np.tile(np.repeat(np.arange(NC), VALIDATION_PER_CLASS), (NS, 1))
assert QIDS.shape == (NS, N_VALIDATION) and CLS.shape == QIDS.shape
assert len({tuple(r) for r in QIDS}) == NS, "the three rows must differ"
assert set(QIDS[0]) & set(QIDS[1]), "...and must overlap"


def good_carrier(**over):
    """A schema-valid carrier artifact: a full 32x32 ranking whose first eight
    entries ARE the recorded head set."""
    from tools.carrier_schema import (GROUP_IMPL, INTERVENTION, K,
                                      N_ATTN_HEADS, N_KV_HEADS, N_LAYERS,
                                      PREFIX_ASSIGNMENT, PREFIX_SEEDS,
                                      RANDOM_SEED_RULE, RANDOM_SEED_SALT,
                                      SCHEMA_VERSION, SCORE_DEFINITION, TASK,
                                      expected_dragged)
    # ⚠ The top-8 must SPREAD across layers and KV groups. An earlier version
    # made them layer 0 heads 0..7, which is exactly two whole KV groups, so
    # dragged_heads == heads and the arm testing "conflates the top-8 with the
    # group union" was byte-identical to the correct artifact -- a red arm
    # that does not differ from the green one tests nothing (working rules 4).
    # Spread heads give 8 groups and 32 dragged heads, so the two sets differ.
    top = [(3, 1), (7, 5), (11, 9), (14, 13), (18, 17), (21, 22), (25, 26),
           (29, 30)]
    rank = [{"layer": l, "head": h, "score": 1000.0 - i}
            for i, (l, h) in enumerate(top)]
    rank += [{"layer": l, "head": h,
              "score": 100.0 - (l * N_ATTN_HEADS + h) * 1e-3}
             for l in range(N_LAYERS) for h in range(N_ATTN_HEADS)
             if (l, h) not in set(top)]
    heads = [[r["layer"], r["head"]] for r in rank[:8]]
    groups, dragged = expected_dragged([(l, h) for l, h in heads])
    c = {"schema_version": SCHEMA_VERSION,
         "spec": "prereg_method_A.md section 2.4(2)",
         "model": "meta-llama/Llama-3.1-8B", "carrier_impl": GROUP_IMPL,
         "score": SCORE_DEFINITION, "n_layers": N_LAYERS,
         "n_attn_heads": N_ATTN_HEADS, "n_kv_heads": N_KV_HEADS,
         "task": TASK, "K": K, "prefix_seeds": list(PREFIX_SEEDS),
         "prefix_assignment": PREFIX_ASSIGNMENT, "intervention": INTERVENTION,
         "random_seed_rule": RANDOM_SEED_RULE,
         "random_seed_salt": RANDOM_SEED_SALT,
         "query_manifest_sha256": "d" * 64, "n_discovery_queries": 144,
         "n_random_variants": 5, "ranking": rank, "heads": heads,
         "kv_groups": [list(g) for g in groups],
         "dragged_heads": [list(x) for x in dragged]}
    c.update(over)
    return c


def write_run(d, *, ebar_effect=None, **over):
    """A well-formed viability run and its meta, on disk."""
    import hashlib
    rng = np.random.default_rng(11)
    nat = rng.normal(2.5, 0.3, size=(NS, N_VALIDATION))
    # [head, gamma, seed, query] since 13.6.3: two head sets per seed, one
    # selected on each fold. The producer writes both slabs; `ebar_effect` is
    # given per gamma and applied to both, since fixtures that vary the two
    # are testing the head axis rather than whatever they came for.
    nll = np.tile(nat, (2, len(GAMMA_GRID), 1, 1))
    if ebar_effect is not None:
        for gi in range(len(GAMMA_GRID)):
            nll[:, gi] += ebar_effect[gi]
    nll[:, 0] = nat                                  # gamma=0 IS the identity
    # The auxiliary metrics carry the head axis too: they are reported AT the
    # configuration the primary criterion selected, so they have to be
    # indexable by the same (head, gamma) pair or apply_choice would read a
    # different arm than the one that was chosen.
    acc = rng.integers(0, 2,
                       size=(2, len(GAMMA_GRID), NS, N_VALIDATION)).astype(float)
    nat_acc = acc[0, 0].copy()
    br = rng.uniform(0, 1, size=(2, len(GAMMA_GRID), NS, N_VALIDATION))
    nat_br = br[0, 0].copy()
    # gamma=0 is the identity under BOTH head sets: writing nothing is
    # writing nothing whichever heads were selected. Drawing the two slabs
    # independently and leaving gamma=0 to differ would be a runner bug, and
    # validate_run says so.
    acc[:, 0] = nat_acc
    br[:, 0] = nat_br
    man = Path(d) / "prereg_method_A_query_manifest.json"
    man.write_text(json.dumps({
        "eligible_classes": list(range(NC)),
        "validation_by_seed": {
            str(sd): [{"query_id": q, "class_idx": int(c),
                       "source_text": f"t{si}-{j}"}
                      for j, (q, c) in enumerate(zip(QIDS[si], CLS[si]))]
            for si, sd in enumerate(REGISTERED_SEEDS)}}),
        encoding="utf-8")
    # A real, internally consistent label space -- FrozenLabelSpace now
    # loads it, so a stub with only a provenance block no longer suffices.
    lsp = Path(d) / "label_space_llama31.json"
    lsp.write_text(json.dumps({
        "schema_version": 1,
        "provenance": {"model": "meta-llama/Llama-3.1-8B",
                       "tokenizer_class": "PreTrainedTokenizerFast",
                       "transformers_version": "4.52.3", "revision": None,
                       "is_fast": True, "vocab_size": 128256,
                       "vocab_sha256": "v", "special_tokens_sha256": "s",
                       "backend_tokenizer_sha256": "b"},
        "task": "trec_fine_per_class", "K": 5, "n_classes": 36,
        "eligible_classes": list(range(36)),
        "abstract_labels": [chr(65 + i % 26) + str(i) for i in range(36)],
        "label_token_ids": [1000 + i for i in range(36)],
        "candidate_token_ids": [1000 + i for i in range(36)],
        "query_manifest_sha256": hashlib.sha256(
            man.read_bytes()).hexdigest()}), encoding="utf-8")

    # the carrier is tied to the MANIFEST now, not to a pool file
    man_sha = hashlib.sha256(man.read_bytes()).hexdigest()
    carrier = Path(d) / "method_a_carriers_L31_top8.json"
    carrier.write_text(json.dumps(
        good_carrier(query_manifest_sha256=man_sha)), encoding="utf-8")
    npz = Path(d) / f"viability_{over.pop('tag', 'ok')}.npz"
    arrays = {"nll": nll, "natural_nll": nat, "acc": acc, "natural_acc": nat_acc,
              "brier": br, "natural_brier": nat_br,
              "query_ids": QIDS, "class_idx": CLS,
              "gammas": np.array(GAMMA_GRID),
              "seeds": np.array([42, 43, 44])}
    arrays.update({k: v for k, v in over.items() if isinstance(v, np.ndarray)})
    np.savez(npz, **arrays)
    meta = {"model": "meta-llama/Llama-3.1-8B", "task": "trec_fine_per_class",
            "K": 5, "dtype": "torch.bfloat16", "attn_implementation": "eager",
            "carrier_impl": "gqa_group_v",
            "carrier_sha256": hashlib.sha256(carrier.read_bytes()).hexdigest(),
            "tokenizer_provenance": {
                "model": "meta-llama/Llama-3.1-8B",
                "tokenizer_class": "PreTrainedTokenizerFast",
                "transformers_version": "4.52.3", "revision": None,
                "is_fast": True, "vocab_size": 128256, "vocab_sha256": "v",
                "special_tokens_sha256": "s",
                "backend_tokenizer_sha256": "b"},
            "query_manifest_sha256":
                hashlib.sha256(man.read_bytes()).hexdigest(),
            "label_space_sha256": hashlib.sha256(lsp.read_bytes()).hexdigest()}
    meta.update({k: v for k, v in over.items() if not isinstance(v, np.ndarray)})
    Path(str(npz).replace(".npz", ".json")).write_text(json.dumps(meta),
                                                       encoding="utf-8")
    # four artifacts, not five: the discovery split is gone (14.0a).
    # Callers unpacking five values will fail loudly here rather than
    # binding a stale name.
    return npz, man, lsp, carrier


def main() -> int:  # noqa: C901
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {detail}" if detail else ""))
        ok = ok and bool(cond)

    def red(name, fn, want=Exception):
        try:
            fn()
        except want as e:
            print(f"  [PASS] {name}  correctly refused: {str(e)[:72]}...")
            return
        except Exception as e:
            check(name, False, f"raised {type(e).__name__}, wanted {want}")
            return
        check(name, False, "SILENTLY SUCCEEDED -- the gate does not bite")

    rng = np.random.default_rng(7)

    print("folds: deterministic, class-balanced, no RNG, PER SEED")
    f1 = assign_folds(QIDS, CLS)
    check("each seed splits 72/72 of its OWN 144",
          f1.shape == (NS, N_VALIDATION)
          and all(int((f1[si] == 0).sum()) == 72
                  and int((f1[si] == 1).sum()) == 72 for si in range(NS)),
          f"{[(int((f1[si] == 0).sum()), int((f1[si] == 1).sum())) for si in range(NS)]}")
    check("every class contributes 2 to each fold, in every seed",
          all(int((f1[si][CLS[si] == c] == 0).sum()) == 2
              for si in range(NS) for c in range(NC)))
    check("re-running gives the identical split", np.array_equal(
        f1, assign_folds(QIDS, CLS)))
    shuf = list(rng.permutation(N_VALIDATION))
    back = np.empty_like(f1)
    back[:, shuf] = assign_folds(QIDS[:, shuf], CLS[:, shuf])
    check("the split follows the QUERY, not its row position",
          np.array_equal(back, f1))
    check("...and the seeds get DIFFERENT splits, because their queries differ",
          not np.array_equal(f1[0], f1[1]),
          "identical fold vectors would mean the seed never reached the split")
    red("a class with the wrong count is refused",
        lambda: assign_folds(QIDS[:, :-1], CLS[:, :-1]), ValueError)
    red("duplicate query ids WITHIN a seed are refused",
        lambda: assign_folds(np.array([["a"] * N_VALIDATION] * NS), CLS),
        ValueError)
    check("...but the SAME query in two seeds is fine",
          assign_folds(QIDS, CLS).shape == (NS, N_VALIDATION)
          and bool(set(QIDS[0]) & set(QIDS[1])),
          "one query in two cells is legal and is what the clustering is for")
    red("a 1-D query_ids array is refused as the pre-14.0a shared axis",
        lambda: assign_folds(QIDS[0], CLS[0]), ValueError)

    # ---- an ODD per-class count ----------------------------------------
    # Measured, because the balance is a property of the alternation and not
    # obvious from it: with the extra query always going to fold 0, per_class
    # =5 over 36 classes gives 108/72, silently and always the same way.
    import tools.method_a_viability as _MV

    def _folds_for(n_cls, per):
        q = np.array([[f"{i:064x}" for i in range(n_cls * per)]])
        c = np.array([np.repeat(np.arange(n_cls), per)])
        keep = _MV.VALIDATION_PER_CLASS
        _MV.VALIDATION_PER_CLASS = per
        try:
            f = _MV.assign_folds(q, c)
        finally:
            _MV.VALIDATION_PER_CLASS = keep
        return f[0], c[0]

    for per in (3, 5, 7):
        f_, c_ = _folds_for(NC, per)
        a, b = int((f_ == 0).sum()), int((f_ == 1).sum())
        sizes = sorted({int((f_[c_ == k] == 0).sum()) for k in range(NC)})
        check(f"per_class={per} (odd) splits {a}/{b}, exactly balanced",
              a == b and sizes == [per // 2, per // 2 + 1],
              f"per-class fold-0 counts {sizes}; without the alternation "
              f"every class would hand its extra to fold 0 and the split "
              f"would be {NC * (per // 2 + 1)}/{NC * (per // 2)}")
    f_, _ = _folds_for(35, 5)
    a, b = int((f_ == 0).sum()), int((f_ == 1).sum())
    check("an odd class count leaves the folds differing by at most one",
          abs(a - b) == 1, f"{a}/{b} over 35 classes x 5")
    f_, _ = _folds_for(NC, VALIDATION_PER_CLASS)
    check("...and the registered even count is unaffected",
          int((f_ == 0).sum()) == int((f_ == 1).sum()) == 72)

    print("\nthe reduced form is EXACT, not an approximation")
    nat = rng.normal(2.5, 0.3, size=(NS, N_VALIDATION))
    # TWO HEAD SOURCES (13.6.3): fold f selects under H_f and fold 1-f is
    # evaluated under it. The two slabs are drawn independently, so an
    # implementation that ignores the head axis produces a different number
    # rather than the same one.
    nll = nat[None, None] + rng.normal(0, 0.1, size=(2, 5, NS, N_VALIDATION))
    nll[0] = nat
    ebar = seed_mean_effect(nll, nat)
    check("e is [head, gamma, seed, query] with NO average over seeds",
          ebar.shape == (2, 5, NS, N_VALIDATION)
          and np.allclose(ebar, nll - nat[None, None]),
          "averaging positions across seeds would average different queries")
    # selection through e must equal selection on the raw NLL means, per seed
    _si = 0
    rows = np.flatnonzero(f1[_si] == 0)
    raw = nll[0][:, _si, :][:, rows].mean(axis=1)      # head source 0
    check("argmin over e equals argmin over the raw NLL (natural cancels)",
          int(np.argmin(raw)) == select_gamma(ebar, _si, rows, hi=0)[0],
          "this cancellation is what makes 10^6 permutations affordable")
    check("gamma=0's e slab is identically zero",
          np.array_equal(ebar[0, 0], np.zeros((NS, N_VALIDATION))),
          "so a permuted dataset still has gamma=0 as its identity")

    cl, n_cl = cluster_index(QIDS)
    check("the cluster map covers every cell and merges repeats",
          cl.shape == (NS, N_VALIDATION) and n_cl < NS * N_VALIDATION,
          f"{n_cl} distinct queries over {NS * N_VALIDATION} cells")
    check("...and a query in two seeds gets ONE cluster id",
          all(cl[0][list(QIDS[0]).index(q)] == cl[1][list(QIDS[1]).index(q)]
              for q in list(set(QIDS[0]) & set(QIDS[1]))[:5]),
          "two ids for one query would make every cluster size one")

    # THE HEAD AXIS IS LOAD-BEARING, not decoration. 13.6.3 evaluates fold f
    # under the head set selected on fold 1-f, so swapping the two slabs must
    # change the answer. If it does not, the axis was threaded through the
    # signatures and ignored by the arithmetic -- which is what the
    # three-axis version did to the head layer of the selection, silently.
    print("\nthe head source actually reaches the arithmetic")
    _hA = np.zeros((2, 5, NS, N_VALIDATION))
    _hA[0, 2] = -0.40          # head set A: gamma index 2 helps
    _hA[1, 4] = -0.40          # head set B: gamma index 4 helps
    _dA, _chA = cross_fit(_hA, f1)
    _dB, _chB = cross_fit(_hA[::-1].copy(), f1)
    _pick_of = lambda ch: [ch[f"seed{si}_fold{f}_selected"]
                           for si in range(NS) for f in (0, 1)]
    check("swapping the two head slabs changes the six selections",
          _pick_of(_chA) != _pick_of(_chB),
          f"A {_pick_of(_chA)} vs B {_pick_of(_chB)}")
    check("...and every cell lands on the planted effect either way",
          np.allclose(_dA, -0.40) and np.allclose(_dB, -0.40),
          "fold 1 reads head 0's slab at head 0's gamma and fold 0 reads "
          "head 1's; reading its OWN fold's head set would find 0.0, since "
          "the two slabs put their effect at different gammas")

    print("\nthe permutation tests the PROCEDURE, like the interval does")
    # Both head sources carry the plant: the arm is about the permutation
    # reproducing cross_fit, not about the two head sets differing.
    plant = np.zeros((2, 5, NS, N_VALIDATION))
    plant[:, 2] = -0.30
    e2 = seed_mean_effect(nat[None, None] + plant, nat)
    d, chosen = cross_fit(e2, f1)
    cl2, _ = cluster_index(QIDS)
    p, T_perm = procedure_sign_flip_p(e2, f1, cl2, n_perm=2000)
    check("the permutation's identity case reproduces cross_fit exactly",
          abs(T_perm - d.mean()) < 1e-15,
          f"{T_perm:.12f} vs {d.mean():.12f}")
    # ALL SIX, not one: a single key would pass while another seed picked
    # something else, and every cell of that seed would then be evaluated
    # under the wrong gamma.
    check("all six (seed, fold) selections find the planted gamma, and T "
          "recovers it",
          all(chosen[f"seed{si}_fold{f}_selected"] == 0.5
              for si in range(NS) for f in (0, 1))
          and abs(d.mean() + 0.30) < 1e-12,
          f"{sorted(v for k, v in chosen.items() if not k.startswith('_'))}")
    check("a strong planted effect is significant", p < 0.01, f"p={p:.4g}")

    # THE ARM THIS REDESIGN EXISTS FOR. Under a pure-noise dataset the naive
    # sign-flip holds the two selected gammas fixed and so ignores the fact
    # that selection itself chased noise; the procedure-level permutation does
    # not. The naive p must be the more optimistic of the two.
    from tools.method_a_stats import sign_flip_p
    pp, pn = [], []
    for s in range(24):
        r = np.random.default_rng(100 + s)
        en = np.zeros((2, 5, NS, N_VALIDATION))
        # THE SAME noise under both head sources. This arm is about the GAMMA
        # selection chasing noise it is then evaluated on; drawing the two
        # head slabs independently breaks exactly that and the gap vanishes
        # (12/24 rather than systematic) -- a real property of the design,
        # and an unrealistic null: H_A and H_B are both the top-8 by
        # score_margin on overlapping queries, so their effects are strongly
        # correlated, not independent. Identical slabs is the conservative
        # end of that range and isolates the arm to what it claims.
        en[:, 1:] = r.normal(0, 0.05, size=(4, NS, N_VALIDATION))[None]
        dn, _ = cross_fit(en, f1)
        pp.append(procedure_sign_flip_p(en, f1, cl2, n_perm=4000, seed=s)[0])
        # the naive comparison flattens the cells, which is what makes it
        # naive: it holds the six selections fixed AND ignores the clustering
        pn.append(sign_flip_p(dn.ravel(), n_permutations=4000, seed=s))
    pp, pn = np.array(pp), np.array(pn)
    # ⚠ The claim is DIRECTIONAL, not pointwise. An earlier version of this arm
    # demanded p_proc >= p_naive on every draw and went red at 11/12 -- with a
    # finite permutation count the ordering can flip on an individual draw even
    # when the systematic relationship is exactly as described. Demanding an
    # inequality a correct implementation cannot always satisfy is working rules 
    # 11b, which this project has now paid for three times.
    check("on pure noise the naive sign-flip is SYSTEMATICALLY more optimistic",
          pp.mean() > pn.mean() and int((pp >= pn).sum()) >= 18,
          f"mean p: procedure {pp.mean():.4f} vs naive {pn.mean():.4f}; "
          f"procedure larger on {int((pp >= pn).sum())}/24 draws. The gap is "
          "the selection noise the naive test conditions away -- it holds the "
          "two chosen gammas fixed, so the permuted data never gets to pick "
          "its own best-looking gamma the way the real data did.")

    print("\nthe futility case is EXACT and agrees with section 9")
    flat = np.zeros((2, 5, NS, N_VALIDATION))
    flat[:, 1:] = 0.10
    d0, ch0 = cross_fit(flat, f1)
    check("every (seed, fold) selects gamma=0 when nothing helps",
          all(ch0[f"seed{si}_fold{f}_selected"] == 0.0
              for si in range(NS) for f in (0, 1)))
    check("d is identically zero, bitwise",
          np.array_equal(d0, np.zeros((NS, N_VALIDATION))))
    b0 = bootstrap_T(flat, f1, cl2, n_boot=200)
    p0, _ = procedure_sign_flip_p(flat, f1, cl2, n_perm=500)
    v0 = decide(0.0, float(np.percentile(b0, 2.5)),
                float(np.percentile(b0, 97.5)), p0)
    check("verdict is STOP, matching section 9's gamma*_group=0 futility exit",
          v0["decision"] == "STOP", f"CI {v0['ci95']}, p {p0:.3g}")

    # ---- identity fields must BE integers, not be truncated to them ----
    print("\nidentity fields are STRICT integers")
    from tools.method_a_viability import strict_int
    for v, label in ((42.9, "seed 42.9"), (0.75, "class 0.75"),
                     (5.9, "K=5.9"), (True, "bool True"),
                     (np.bool_(True), "np.bool_(True), not a Python bool"),
                     ("5.0", 'the string "5.0", which float() would parse'),
                     ("42", 'the string "42"'),
                     (float("nan"), "NaN"), (float("inf"), "inf")):
        red(f"{label} is refused rather than truncated",
            lambda v=v: strict_int(v, "x"), ValueError)
    # UNBOUNDED integers: float(10**1000) raises OverflowError, which the
    # callers catching ValueError/TypeError never saw, so a huge integer
    # escaped as an exception and broke the "return faults" contract. The
    # Integral fast path returns it exactly, without float() at all.
    check("a huge Python int is returned EXACTLY, not overflowed",
          strict_int(10 ** 1000, "x") == 10 ** 1000,
          "float(10**1000) raises OverflowError; the Integral path never "
          "calls it")
    check("...and a huge int inside numpy's int64 range is unaffected",
          strict_int(10 ** 18, "x") == 10 ** 18)

    check("...while a genuine integer, int or float, passes",
          strict_int(42, "x") == 42 and strict_int(42.0, "x") == 42,
          "bare int() accepted 42.9 as seed 42, which then matched the "
          "registered value")

    # ---- the reported argmin obeys the SAME tie rule as the selection ----
    # np.argmin takes the numeric minimum; the registered rule takes the
    # smallest gamma within 1e-8 of it. A 1e-12 improvement at gamma=0.25 was
    # reported as the in-sample choice where the rule selects 0.0.
    print("\nthe in-sample argmin uses the registered tie rule")
    from tools.method_a_viability import _pick as _pk
    _near = np.zeros((5, 3))
    _near[1, 0] = -1e-12                      # seed 0: a 1e-12 "improvement"
    _near[1, 1] = -1e-6                       # seed 1: outside the tolerance
    check("a 1e-12 improvement is a TIE, so the smaller gamma wins",
          GAMMA_GRID[int(_pk(_near[:, 0]))] == 0.0,
          f"np.argmin would report "
          f"{GAMMA_GRID[int(np.argmin(_near[:, 0]))]}")
    check("...while 1e-6 is outside the tolerance and is honoured",
          GAMMA_GRID[int(_pk(_near[:, 1]))] == 0.25)

    # ...and THROUGH run(), because the two arms above call _pick directly:
    # reverting run() to np.argmin would leave both of them green while the
    # sidecar recorded the wrong gamma. This drives the real path and reads
    # what the sidecar actually holds.
    import tools.method_a_viability as _MV2
    with tempfile.TemporaryDirectory() as _td:
        _eff = np.zeros((5, NS, N_VALIDATION))
        _eff[1, 0] = -1e-12                   # seed 0 only, a hair better
        _np, _mn, _ls2, _cj = write_run(_td, tag="tie", ebar_effect=_eff)
        _v, _x = _MV2.run(_np, query_manifest=_mn, label_space=_ls2,
                          carrier_json=_cj, task=None, freeze=None,
                          n_boot=40, n_perm=40)
        _am = _x["in_sample_argmin_by_seed"]
        check("run() records the TIE RULE's gamma, not np.argmin's",
              _am[str(REGISTERED_SEEDS[0])] == 0.0,
              f"got {_am}; a 1e-12 improvement at gamma=0.25 is a tie, so "
              "section 5(4) takes 0.0 -- np.argmin would record 0.25")

    from tools.prereg_ids import content_hash as _ch2

    def _row(c, t):
        return {"query_id": _ch2(c, t), "class_idx": c, "source_text": t}

    # ---- the rebuild's row comparison, driven directly -----------------
    # This lived inside task_rebuild_inputs, which needs a real TREC
    # download, so the only test that reached it stubbed the whole function
    # and the comparison had no arm at all. Split out, it takes rows and
    # returns faults, so a permutation can be handed to it.
    print("\nthe rebuild compares ROWS, not just ids")
    from tools.method_a_viability import validation_row_faults
    from tools.prereg_ids import content_hash as _ch2


    _good = [_row(c, f"q{c}") for c in range(4)]
    check("an identical draw yields no faults",
          validation_row_faults(_good, list(_good), 42) == [])
    _reordered = [_good[1], _good[0]] + _good[2:]
    check("a REORDERING is caught",
          len(validation_row_faults(_good, _reordered, 42)) >= 1,
          "positional indexing pairs each row with another query's gold")
    # THE ONE THE ID CHECK MISSES: same ids, same order, metadata permuted
    _perm = [dict(g) for g in _good]
    _perm[0]["class_idx"], _perm[1]["class_idx"] = (_perm[1]["class_idx"],
                                                    _perm[0]["class_idx"])
    _perm[0]["source_text"], _perm[1]["source_text"] = (_perm[1]["source_text"],
                                                        _perm[0]["source_text"])
    check("the id list is UNCHANGED by that permutation",
          [v["query_id"] for v in _perm] == [v["query_id"] for v in _good],
          "so an id-only comparison sees nothing wrong")
    _f = validation_row_faults(_good, _perm, 42)
    check("...but permuted metadata IS caught",
          len(_f) >= 1, str(_f)[:80] or "SILENTLY CLEAN")
    check("...and the hash recomputation names it as assembled, not drawn",
          any("content_hash" in x for x in _f), str(_f)[:80])
    _oneoff = [dict(g) for g in _good]
    _oneoff[2]["class_idx"] = 99
    check("a single wrong gold class is caught",
          len(validation_row_faults(_good, _oneoff, 42)) >= 1,
          "the gold class also decides the fold, so the split moves with it")

    # a manifest whose class_idx is the STRING "3" must not pass as 3
    _strrows = [dict(r) for r in [_row(c, f"q{c}") for c in range(4)]]
    _strrows[0]["class_idx"] = "0"
    check("a class_idx that is the STRING \"0\" is refused, not coerced",
          validation_row_faults([_row(c, f"q{c}") for c in range(4)],
                                _strrows, 42) != [],
          "str(0) == str(\"0\") accepted it; a file's identity fields must "
          "already be numbers")

    # ---- the WIRING: real task_rebuild_inputs, stubbed task data --------
    # The arms above call validation_row_faults directly, and the --task test
    # replaces task_rebuild_inputs wholesale, so deleting the production call
    # leaves both groups green. This keeps the real function and patches only
    # what it reads from the task -- the one arrangement that notices.
    print("\ntask_rebuild_inputs actually CALLS the row comparison")
    import tools.prereg_task as _PT2
    import tools.method_a_viability as _MV3
    with tempfile.TemporaryDirectory() as _td2:
        _rows = [_row(c, f"q{c}") for c in range(4)]
        _bad_rows = [dict(r) for r in _rows]
        _bad_rows[0]["class_idx"] = 99          # ids match, gold does not
        _bad_rows[0]["query_id"] = _rows[0]["query_id"]
        _mf = Path(_td2) / "prereg_method_A_query_manifest.json"
        _mf.write_text(json.dumps({
            "eligible_classes": [0, 1, 2, 3],
            "validation_by_seed": {str(sd): _bad_rows
                                   for sd in REGISTERED_SEEDS}}),
            encoding="utf-8")
        _saved = {n: getattr(_PT2, n) for n in
                  ("load_task", "train_rows", "eligible_classes",
                   "draw_validation_from_train", "prefix_demo_rows")}
        _PT2.load_task = lambda *a, **k: object()
        _PT2.train_rows = lambda *a, **k: []
        _PT2.eligible_classes = lambda *a, **k: [0, 1, 2, 3]
        _PT2.draw_validation_from_train = \
            lambda *a, **k: ([dict(r) for r in _rows], {})
        _PT2.prefix_demo_rows = lambda *a, **k: {a[2][0]: []}
        try:
            _f2 = _MV3.task_rebuild_inputs("trec_fine_per_class", _mf)
            check("the real function reports the row fault the helper finds",
                  any("class_idx" in x or "content_hash" in x for x in _f2),
                  str(_f2)[:90] or "SILENTLY CLEAN -- the call is missing")
            _mf.write_text(json.dumps({
                "eligible_classes": [0, 1, 2, 3],
                "validation_by_seed": {str(sd): _rows
                                       for sd in REGISTERED_SEEDS}}),
                encoding="utf-8")
            check("...and reports nothing when the rows agree",
                  _MV3.task_rebuild_inputs("trec_fine_per_class", _mf) == [],
                  "otherwise the arm above would fire on anything")
        finally:
            for _n, _fn in _saved.items():
                setattr(_PT2, _n, _fn)

    # ---- the invariant the row check rests on -------------------------
    # task_rebuild_inputs needs a real task, so its new row-by-row comparison
    # has no direct arm here -- block B stubs the whole function. What CAN be
    # checked without a download is the property it relies on: query_id IS
    # content_hash(class_idx, source_text), so permuting that metadata among
    # the ids cannot leave the hashes intact. If this stopped holding, the row
    # check would be comparing fields with nothing anchoring them, which is
    # the state it was added to end.
    print("\nquery_id IS the hash of (class, text) -- the row check's anchor")
    from tools.prereg_ids import content_hash as _cht
    _pairs = [(3, "a question"), (11, "another one"), (0, "a third")]
    _ids = [_cht(c, t) for c, t in _pairs]
    check("the id is reproduced by hashing its own pair",
          all(_cht(c, t) == i for (c, t), i in zip(_pairs, _ids)))
    check("...and PERMUTING the metadata between ids breaks every one",
          not any(_cht(c, t) == i for (c, t), i
                  in zip(_pairs[1:] + _pairs[:1], _ids)),
          "this is why recomputing the hash catches a permutation that "
          "leaves the id list and the reservation set untouched")

    print("\nthe selection rule is section 5(4)'s")
    # [gamma, seed, query] with one seed: select_gamma picks for ONE seed
    tie = np.zeros((2, 5, 1, 4))
    tie[:, 3] -= 1e-12
    check("a tie within 1e-8 takes the SMALLER gamma",
          GAMMA_GRID[select_gamma(tie, 0, np.arange(4))[0]] == 0.0)
    tie2 = np.zeros((2, 5, 1, 4))
    tie2[:, 3] -= 1e-6
    check("a difference outside the tolerance is honoured",
          GAMMA_GRID[select_gamma(tie2, 0, np.arange(4))[0]] == 0.75)
    red("selecting on an empty fold is refused",
        lambda: select_gamma(tie, 0, np.array([], dtype=int)), ValueError)
    # and the choice is per SEED: seed 1's rows must not decide seed 0's gamma
    two = np.zeros((2, 5, 2, 4))
    two[:, 3, 0] -= 1e-6                    # seed 0 favours gamma=0.75
    two[:, 1, 1] -= 1e-6                    # seed 1 favours gamma=0.25
    check("each seed selects from its OWN rows",
          GAMMA_GRID[select_gamma(two, 0, np.arange(4))[0]] == 0.75
          and GAMMA_GRID[select_gamma(two, 1, np.arange(4))[0]] == 0.25,
          "one gamma for all three seeds is what the shared axis produced")

    print("\nthree states, mutually exclusive and exhaustive")
    for name, T, lo, hi, p_, want in [
            ("a large, tight improvement", -0.30, -0.34, -0.26, 1e-5, "GO"),
            ("an improvement too small to matter", -0.004, -0.008, -0.001,
             1e-4, "STOP"),
            ("a harmful effect", +0.10, 0.06, 0.14, 1e-5, "STOP"),
            ("a straddling interval", -0.05, -0.11, +0.01, 0.20,
             "INCONCLUSIVE"),
            ("exactly at the threshold", -0.02, -0.05, -0.02, 1e-5,
             "INCONCLUSIVE"),
            ("past the margin but not significant", -0.30, -0.34, -0.26, 0.40,
             "INCONCLUSIVE")]:
        check(f"{name} -> {want}", decide(T, lo, hi, p_)["decision"] == want)
    red("a bool masquerading as a number is refused",
        lambda: decide(True, True, True, True), TypeError)
    red("a NaN statistic is refused", lambda: decide(float("nan"), -1., 1., .01),
        ValueError)
    red("an inverted interval is refused", lambda: decide(-.1, 0.2, -0.2, .01),
        ValueError)

    print("\na verdict requires the REGISTERED configuration")
    v = decide(-0.30, -0.34, -0.26, 1e-5, status=DIAGNOSTIC,
               why_diagnostic=["--n-boot 50"])
    check("a diagnostic run returns decision=None, not GO with a caveat",
          v["decision"] is None and v["status"] == DIAGNOSTIC
          and v["provisional_decision"] == "GO",
          "nothing downstream can read a verdict the configuration did not earn")
    check("...and it says why", v["why_diagnostic"] == ["--n-boot 50"])
    check("a verified run does carry the decision",
          decide(-0.30, -0.34, -0.26, 1e-5)["decision"] == "GO")
    red("an unknown status is refused",
        lambda: decide(-.1, -.2, 0., .01, status="OK"), ValueError)

    print("\nthe npz must PROVE it is the registered experiment")
    with tempfile.TemporaryDirectory() as td:
        npz, man, lsp, carr = write_run(td)
        z, meta, g = load_viability_npz(npz, query_manifest=man,
                                        label_space=lsp,
                                        carrier_json=carr)
        check("a well-formed registered run loads",
              z["nll"].shape == (2, 5, 3, 144) and meta["carrier_impl"]
              == "gqa_group_v",
              "two head sources per seed since 13.6.3, one selected on each "
              "fold")
        for label, over in (
                ("the seeds are not 42,43,44",
                 {"tag": "s", "seeds": np.array([42, 43, 45])}),
                ("the carrier is Method A-HEAD, not group",
                 {"tag": "h", "carrier_impl": "gqa_head_realized"}),
                ("the model is not L3.1", {"tag": "m", "model": "gpt2"}),
                ("K is not 5", {"tag": "k", "K": 10}),
                ("attention is not eager",
                 {"tag": "a", "attn_implementation": "sdpa"}),
                ("the recorded manifest hash does not match the file",
                 {"tag": "q", "query_manifest_sha256": "0" * 64}),
                ("the label-space hash does not match",
                 {"tag": "l", "label_space_sha256": "0" * 64}),
                ("the recorded CARRIER hash does not match the artifact",
                 {"tag": "c", "carrier_sha256": "0" * 64}),
                ("the task is not the registered one",
                 {"tag": "t", "task": "sst2"}),
                ("the model is an INSTRUCT variant that merely ends similarly",
                 {"tag": "i", "model": "meta-llama/Llama-3.1-8B-Instruct"}),
                ("dtype is a string that merely CONTAINS bfloat16",
                 {"tag": "d", "dtype": "not-bfloat16"}),
                ("the gamma grid is not the registered one",
                 {"tag": "g", "gammas": np.array([0., .5, 1., 1.5, 2.])})):
            bad_npz, _, _, _ = write_run(td, **over)
            red(f"a run where {label} is refused",
                lambda p=bad_npz: load_viability_npz(
                    p, query_manifest=man, label_space=lsp, carrier_json=carr), ValueError)

        # SELF-CONSISTENT RUNS FROM THE WRONG EXPERIMENT. Each has the right
        # shapes and is internally coherent; only the artifacts say otherwise.
        #
        # ⚠ The first draft of this arm built its "wrong" ids with the same
        # formula as the good ones and so was bitwise IDENTICAL to the correct
        # run -- it reported SILENTLY SUCCEEDED against a check that was
        # working fine. A red arm that does not differ from the green one tests
        # nothing (working rules rule 4).
        # [S, Q] like QIDS, and flattened for the comparison: `set(QIDS)`
        # over a 2-D array raises rather than comparing, so the disjointness
        # this arm asserts would never have been evaluated.
        other_q = np.array([[f"{i + 10 ** 6 + 10000 * si:064x}"
                             for i in range(N_VALIDATION)]
                            for si in range(NS)])
        check("the 'wrong' ids really are different from the good ones",
              not (set(other_q.ravel()) & set(QIDS.ravel())),
              "otherwise the arm below would be testing the identical run")
        wrong, _, _, _ = write_run(td, tag="otherq", query_ids=other_q)
        red("a self-consistent run on a DIFFERENT validation set is refused",
            lambda: load_viability_npz(wrong, query_manifest=man,
                                       label_space=lsp, carrier_json=carr), ValueError)

        # roll WITHIN each seed: rolling the seed axis would swap whole
        # seeds, which the seed-order check catches for another reason
        rot = np.roll(QIDS, 1, axis=1)
        wrong2, _, _, _ = write_run(td, tag="rot", query_ids=rot)
        red("...and so is the SAME set in a different order (positional "
            "indexing would pair each row with another query's gold)",
            lambda: load_viability_npz(wrong2, query_manifest=man,
                                       label_space=lsp, carrier_json=carr), ValueError)

        ten = np.repeat(np.arange(10), 15)[:N_VALIDATION]
        wrong3, _, _, _ = write_run(td, tag="tencls", class_idx=ten)
        red("a run over 10 classes rather than the registered 36 is refused",
            lambda: load_viability_npz(wrong3, query_manifest=man,
                                       label_space=lsp, carrier_json=carr), ValueError)

        # gamma=0 must be the identity on ALL THREE metrics, not just NLL.
        for metric in ("natural_nll", "natural_acc", "natural_brier"):
            arr = dict(np.load(npz, allow_pickle=False))
            arr[metric] = np.asarray(arr[metric], float) + 1e-9
            p2 = Path(td) / f"broken_{metric}.npz"
            np.savez(p2, **arr)
            Path(str(p2).replace(".npz", ".json")).write_text(
                json.dumps(meta), encoding="utf-8")
            red(f"a gamma=0 slice that differs from {metric} is refused",
                lambda pp=p2: load_viability_npz(pp, query_manifest=man,
                                                 label_space=lsp,
                                                 carrier_json=carr), ValueError)

        arr = dict(np.load(npz, allow_pickle=False))
        del arr["brier"]
        p3 = Path(td) / "nobrier.npz"
        np.savez(p3, **arr)
        Path(str(p3).replace(".npz", ".json")).write_text(json.dumps(meta),
                                                          encoding="utf-8")
        red("a run with no Brier is refused -- 13.6.5 requires the auxiliary "
            "metrics as inputs",
            lambda: load_viability_npz(p3, query_manifest=man,
                                       label_space=lsp, carrier_json=carr), ValueError)
        for field in ("carrier_sha256", "carrier_impl", "model"):
            m2 = {k: v for k, v in meta.items() if k != field}
            p4 = Path(td) / f"nometa_{field}.npz"
            np.savez(p4, **dict(np.load(npz, allow_pickle=False)))
            Path(str(p4).replace(".npz", ".json")).write_text(
                json.dumps(m2), encoding="utf-8")
            red(f"a meta missing {field} is refused",
                lambda pp=p4: load_viability_npz(pp, query_manifest=man,
                                                 label_space=lsp,
                                                 carrier_json=carr), ValueError)

        # The carrier ARTIFACT itself, not just its hash. A hash pins bytes;
        # it does not say the bytes describe the frozen top-8.
        # ⚠ Order matters: write_run REWRITES the carrier with good content, so
        # the artifact must be mangled AFTER the run exists, and the meta's
        # carrier_sha256 updated to match -- otherwise the hash check fires and
        # the CONTENT check, which is what this arm is for, never runs.
        import hashlib as _h
        gc = good_carrier()
        dup = [gc["heads"][0]] * 8
        fake = [[31, 31 - i] for i in range(8)]
        for i, (label, payload) in enumerate((
                ("lists no heads", good_carrier(heads=[])),
                ("lists the wrong number of heads",
                 good_carrier(heads=gc["heads"] + [[9, 0]])),
                ("lists EIGHT DUPLICATES of one head", good_carrier(heads=dup)),
                ("lists eight plausible but INVENTED heads that are not the "
                 "ranking's top-8", good_carrier(heads=fake)),
                ("has no ranking at all, so 'top-8' is an assertion",
                 good_carrier(ranking=[])),
                ("has a ranking that is not sorted by score",
                 good_carrier(ranking=list(reversed(gc["ranking"])))),
                ("names Method A-head rather than the group implementation",
                 good_carrier(carrier_impl="gqa_head_realized")),
                ("was discovered on a DIFFERENT pool",
                 good_carrier(query_manifest_sha256="0" * 64)),
                ("conflates the top-8 with the KV-group union",
                 good_carrier(dragged_heads=gc["heads"])))):
            bad4, _, _, c4 = write_run(td, tag=f"carr{i}")
            c4.write_text(json.dumps(payload), encoding="utf-8")
            mp = Path(str(bad4).replace(".npz", ".json"))
            m4 = json.loads(mp.read_text(encoding="utf-8"))
            m4["carrier_sha256"] = _h.sha256(c4.read_bytes()).hexdigest()
            mp.write_text(json.dumps(m4), encoding="utf-8")
            red(f"a carrier artifact that {label} is refused even though its "
                "hash matches", lambda p=bad4, c=c4: load_viability_npz(
                    p, query_manifest=man, label_space=lsp, carrier_json=c),
                ValueError)
        carr.write_text(json.dumps(gc), encoding="utf-8")

        # THE SELF-REFERENCE. Previously the carrier's discovery hash was
        # compared against the META's copy of it, so writing the SAME wrong
        # value into both satisfied the check. Only hashing the file settles
        # it, and this arm is the one that says so.
        agree, _, _, ac = write_run(
            td, tag="agree", query_manifest_sha256="0" * 64)
        ac.write_text(json.dumps(
            good_carrier(query_manifest_sha256="0" * 64)), encoding="utf-8")
        amp = Path(str(agree).replace(".npz", ".json"))
        am = json.loads(amp.read_text(encoding="utf-8"))
        am["carrier_sha256"] = _h.sha256(ac.read_bytes()).hexdigest()
        amp.write_text(json.dumps(am), encoding="utf-8")
        red("a discovery hash that meta and carrier AGREE on, but that no file "
            "produces, is refused",
            lambda: load_viability_npz(agree, query_manifest=man,
                                       label_space=lsp, carrier_json=ac), ValueError)

        # Three arms tampered with the discovery pool's CONTENTS here --
        # truncated it, reversed it, gave it another task -- to show that
        # the hash alone was not the check. The pool is gone (14.0a) and
        # the property moved to the manifest's validation sets, where the
        # arms already exist: `query_ids=rot` is the reordering,
        # `class_idx=ten` the wrong class space, and the shape arms the
        # truncation. Retargeting these would have duplicated those.

        # THE DOMAIN, not just finiteness. -10 is a perfectly finite NLL and
        # an impossible one, and e = nll - natural then goes hugely negative
        # -- which is the sign GO requires.
        for nm, bad_val, why in (
                ("nll", -10.0, "a negative NLL is -log p with p > 1"),
                ("brier", -1.0, "Brier is a sum of squares"),
                ("brier", 3.0, "Brier maxes at 2"),
                ("acc", 2.0, "accuracy is a mean of indicators")):
            # The gamma=0 slice must EQUAL the natural arm, or the identity
            # check fires first and the arm passes for the wrong reason --
            # which is what the first draft did: disabling the domain check
            # left all four green.
            arr = np.full((5, NS, N_VALIDATION), 0.5)
            arr[1, 0, 0] = bad_val
            natname = "natural_" + ("nll" if nm == "nll" else nm)
            dp, _, _, dc = write_run(
                td, tag=f"dom{nm}{bad_val}",
                **{nm: arr, natname: arr[0].copy()})
            red(f"{nm}={bad_val} is refused ({why})",
                lambda p=dp, c=dc: load_viability_npz(
                    p, query_manifest=man, label_space=lsp,
                    carrier_json=c), ValueError)

        # THE GAMMA GRID'S TYPE, which the value comparison cannot see.
        # Measured at the npz boundary first: a mixed bool/float list is
        # coerced to float64 on save, so that path is unreachable and is not
        # armed. Strings survive as <U4 and an all-bool list as bool dtype,
        # so both are reachable -- but only the STRING case is discriminating,
        # because it parses to exactly the registered grid and the later value
        # check therefore passes it. An all-bool grid is refused by the value
        # check whatever the type check does.
        _sg = np.array(["0.0", "0.25", "0.5", "0.75", "1.0"])
        gp, _, _, gc = write_run(td, tag="gstr", gammas=_sg)
        red("a gamma grid of STRINGS is refused even though it parses to the "
            "registered values",
            lambda: load_viability_npz(gp, query_manifest=man,
                                       label_space=lsp, carrier_json=gc),
            ValueError)

        # accuracy is an INDICATOR per cell, not a rate: 0.5 is inside
        # [0, 1] and is not something one query can be.
        _half = np.full((5, NS, N_VALIDATION), 1.0)
        _half[1, 0, 0] = 0.5
        hp, _, _, hc = write_run(td, tag="acchalf",
                                 acc=_half, natural_acc=_half[0].copy())
        red("accuracy=0.5 is refused -- each cell is one query's correctness",
            lambda: load_viability_npz(hp, query_manifest=man,
                                       label_space=lsp, carrier_json=hc),
            ValueError)

        # a fractional CLASS, through the loader. seeds and K were driven
        # here; class_idx was not, so removing strict_int from the class path
        # left every arm green.
        _fc = CLS.astype(float).copy()
        _fc[0, 0] = 0.75
        cp, _, _, cc = write_run(td, tag="clsfrac", class_idx=_fc)
        red("a fractional class_idx is refused by the loader",
            lambda: load_viability_npz(cp, query_manifest=man,
                                       label_space=lsp, carrier_json=cc),
            (ValueError, TypeError))

        # strict_int THROUGH THE LOADER, not only as a helper. A helper can
        # be right while nothing calls it; these three drive the path a real
        # run takes.
        for fld, val, why in ((("seeds", np.array([42.9, 43.0, 44.0])),
                               None, "a fractional seed"),
                              (None, ("K", 5.9), "a fractional K"),
                              (None, ("K", True), "a bool K")):
            if fld is not None:
                ip, _, _, ic = write_run(td, tag="seedfrac",
                                         **{fld[0]: fld[1]})
                red(f"{why} is refused by the loader",
                    lambda p=ip, c=ic: load_viability_npz(
                        p, query_manifest=man, label_space=lsp,
                        carrier_json=c), (ValueError, TypeError))
            else:
                ip, _, _, ic = write_run(td, tag=f"K{val[1]}")
                _mp = Path(str(ip).replace(".npz", ".json"))
                _m = json.loads(_mp.read_text(encoding="utf-8"))
                _m[val[0]] = val[1]
                _mp.write_text(json.dumps(_m), encoding="utf-8")
                red(f"{why} is refused by the loader",
                    lambda p=ip, c=ic: load_viability_npz(
                        p, query_manifest=man, label_space=lsp,
                        carrier_json=c), (ValueError, TypeError))

        # accuracy and Brier shapes, not only NLL's
        for name, shape in (("acc", (5, 3, 100)), ("brier", (4, 3, 144)),
                            ("natural_acc", (3, 100)),
                            ("natural_brier", (144, 3))):
            bs, _, _, bcj = write_run(
                td, tag=f"shape{name}", **{name: np.zeros(shape)})
            red(f"a {name} of shape {shape} is refused",
                lambda p=bs, c=bcj: load_viability_npz(
                    p, query_manifest=man, label_space=lsp, carrier_json=c), ValueError)

        # the tokenizer provenance the FORWARD RUNNER recorded
        for label, over in (
                ("no tokenizer provenance at all",
                 {"tag": "np", "tokenizer_provenance": None}),
                ("a provenance that disagrees with the label space",
                 {"tag": "wp", "tokenizer_provenance": {
                     "model": "meta-llama/Llama-3.1-8B",
                     "tokenizer_class": "PreTrainedTokenizerFast",
                     "transformers_version": "4.52.3", "revision": None,
                     "is_fast": True, "vocab_size": 128256,
                     "vocab_sha256": "DIFFERENT",
                     "special_tokens_sha256": "s",
                     "backend_tokenizer_sha256": "b"}})):
            bp, _, _, bc = write_run(td, **over)
            red(f"a run with {label} is refused",
                lambda p=bp, c=bc: load_viability_npz(
                    p, query_manifest=man, label_space=lsp, carrier_json=c),
                ValueError)

    print("\n--task actually threads the rebuild through, and its absence "
          "downgrades")
    # ⚠ An earlier version of this section hand-built a DIAGNOSTIC verdict and
    # asserted its shape. That would stay green even if --task were never
    # passed to the validator at all. This exercises the wiring: the SAME
    # forged split is accepted without a task and refused with one.
    import tools.method_a_viability as MV
    with tempfile.TemporaryDirectory() as td:
        npz, man, lsp, carr = write_run(td)
        # The forged artifact is the MANIFEST now. Its validation sets
        # are internally consistent -- every hash agrees -- but they are
        # not what draw_validation_from_train produces from the train
        # split, which is the one property no amount of self-consistency
        # can forge. task_rebuild_inputs is stubbed to report exactly
        # that, so the arms below test the WIRING rather than the draw.
        z, _meta, _g = load_viability_npz(
            npz, query_manifest=man, label_space=lsp, carrier_json=carr, task=None)
        check("WITHOUT a task the forged pool is accepted -- self-consistency "
              "is all that can be checked",
              z["nll"].shape == (2, len(GAMMA_GRID), NS, N_VALIDATION))

        real = MV.task_rebuild_inputs
        MV.task_rebuild_inputs = lambda _t, _m: [
            "seed 42: the manifest's validation set does not rebuild "
            "from the train split"]
        try:
            red("...and WITH a task it is refused, so --task really reaches "
                "the rebuild",
                lambda: load_viability_npz(
                    npz, query_manifest=man, label_space=lsp,
                    carrier_json=carr,
                    task="trec_fine_per_class"), ValueError)

            # ⚠ THE WHOLE CHAIN, unstubbed: main -> run -> loader -> rebuild.
            # There are THREE places the task can be dropped, not two --
            # main's `task=args.task`, run's `task=task`, and the loader's use
            # of it. The two arms above cover the first and the last; deleting
            # `task=task` inside run() would have left all 88 arms green.
            cli = ["--viability-npz", str(npz), "--query-manifest", str(man),
                   "--label-space", str(lsp), "--carrier-json", str(carr), "--no-require-freeze",
                   "--n-boot", "40", "--n-perm", "40"]
            red("main(--task ...) reaches the rebuild through run() and the "
                "loader", lambda: MV.main(cli + ["--task",
                                                 "trec_fine_per_class"]),
                ValueError)
            check("...while the identical call WITHOUT --task completes",
                  MV.main(cli) == 0,
                  "so the difference is the flag, not the artifacts")
        finally:
            MV.task_rebuild_inputs = real
        check("the patch was removed again",
              MV.task_rebuild_inputs is real)
    # ⚠ THROUGH main -> run -> loader, not by hand. Asserting the shape of a
    # hand-built verdict would stay green if the downgrade were deleted from
    # `run` or if `main` stopped passing task=args.task -- the two places the
    # wiring can actually break.
    with tempfile.TemporaryDirectory() as td:
        npz, man, lsp, carr = write_run(td)
        verdict, _extra = MV.run(
            npz, query_manifest=man, label_space=lsp, carrier_json=carr, task=None, freeze=None, n_boot=40, n_perm=40)
        check("run(task=None) really returns DIAGNOSTIC with decision=None",
              verdict["status"] == DIAGNOSTIC and verdict["decision"] is None
              and verdict["provisional_decision"] in DECISIONS)
        check("...and it says the missing task is one of the reasons",
              any("--task" in w for w in verdict["why_diagnostic"]),
              next(w for w in verdict["why_diagnostic"] if "--task" in w)[:58])

        seen = {}
        real_run = MV.run
        MV.run = lambda *a, **k: (seen.update(k) or
                                  ({"status": DIAGNOSTIC, "decision": None,
                                    "provisional_decision": "STOP",
                                    "means": "", "why_diagnostic": []}, {}))
        try:
            MV.main(["--viability-npz", str(npz), "--query-manifest", str(man),
                     "--label-space", str(lsp), "--carrier-json", str(carr),
                     "--task", "trec_fine_per_class", "--no-require-freeze"])
        finally:
            MV.run = real_run
        check("main() forwards --task to run()",
              seen.get("task") == "trec_fine_per_class",
              "dropping `task=args.task` would otherwise go unnoticed")
        check("...and the carrier path with it",
              str(seen.get("carrier_json")) == str(carr),
              "the discovery-split path was checked here too until the "
              "artifact was abolished; asserting on a field that should not "
              "exist would test the fixture's ability to invent one")
        check("the run patch was removed again", MV.run is real_run)

    print("\nthe gate refuses to run before the opponents are frozen, and the "
          "freeze must be VALID")
    red("a missing freeze file stops the gate",
        lambda: require_freeze(Path(tempfile.gettempdir()) / "no_freeze.json"),
        SystemExit)
    with tempfile.TemporaryDirectory() as td:
        # A freeze from the OLD fail-open tool: self-consistent, records every
        # repository as unreadable, pins nothing. A consumer that only asks
        # "has anything drifted" accepts it.
        stale = Path(td) / "baseline_spec_freeze.json"
        from tools.baselines import registry as R
        stale.write_text(json.dumps({
            "stage": "spec",
            "documents": {}, "baselines": {k: {} for k in R.BASELINES},
            "upstream_repos": {n: {"available": False, "why": "old tool"}
                               for n in ("ZeroTuning", "StaICC",
                                         "function_vectors",
                                         "Localizing_TR_TL", "UniBias",
                                         "DeepThinking",
                                         "in-context-mechanism")}}),
            encoding="utf-8")
        red("a freeze recording every repo as unreadable is refused as INVALID, "
            "not merely checked for drift",
            lambda: require_freeze(stale), SystemExit)
        hand = Path(td) / "handmade.json"
        hand.write_text(json.dumps({"looks": "plausible"}), encoding="utf-8")
        red("a hand-authored freeze is refused", lambda: require_freeze(hand),
            SystemExit)

    print()
    if not ok:
        print("FIXTURES FAILED -- do not run the viability gate.")
        return 2
    print("the viability gate holds: the permutation and the interval move the "
          "same thing, an unregistered configuration yields no verdict, and an "
          "npz has to prove it is the registered experiment rather than merely "
          "be self-consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
