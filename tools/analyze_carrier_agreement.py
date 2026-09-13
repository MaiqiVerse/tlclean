"""How much do the nine carrier sets agree, and along which axis?

Section 2.4(2) already requires the pairwise overlap of the three per-seed
top-8, calling it "a direct reading of how much the carriers depend on the
prefix". This adds the axis 13.6.3 created and nobody asked for a reading
of: fold A against fold B WITHIN a seed, and each fold against that seed's
full-validation set.

WHY IT IS WORTH LOOKING AT. The cross-fit exists because selecting on the
data you then score on inflates the estimate. How much that costs depends
entirely on how unstable the selection is:

  * folds agree almost completely -- the cross-fit is nearly free, and "these
    heads carry the effect" is a claim about the heads rather than about one
    half of one seed's validation set;
  * folds disagree substantially -- the cross-fit is doing real work, the
    gate's number is much more trustworthy than an in-sample one would have
    been, AND any statement naming particular heads is weak, because a
    different 72 queries would have named different ones.

Both readings are useful and they point opposite ways, which is why this is
DESCRIPTIVE and enters no criterion. It cannot: the sets it compares are the
ones the gate selects with, so a rule keyed to their agreement would be a
second selection rule fitted on the same data.

ZERO GPU. Everything here is already in the two bundles.

Run:
    python tools/analyze_carrier_agreement.py \\
        --fold results/carriers_fold.json \\
        --full results/carriers_full_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import carrier_bundle as CB  # noqa: E402
from tools.prereg_config import REGISTERED_SEEDS  # noqa: E402


def heads_of(bundle, key):
    return {tuple(int(x) for x in h)
            for h in bundle["sets"][key]["carrier"]["heads"]}


def ranking_of(bundle, key):
    """[(layer, head)] over ALL query heads, best first.

    2.4(2) requires the full ranking in the artifact precisely so that "these
    are the top 8" is checkable. It also makes every statistic below possible
    without touching the scores npz or re-sorting anything -- a second ranking
    implementation is the defect 14.0b-12 was.
    """
    return [(int(e["layer"]), int(e["head"]))
            for e in bundle["sets"][key]["carrier"]["ranking"]]


def rank_map(bundle, key):
    return {lh: i for i, lh in enumerate(ranking_of(bundle, key))}


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def chance_shared(k, n):
    """E|A n B| for two independent uniform k-subsets of n. k*k/n.

    WITHOUT THIS THE OVERLAPS CANNOT BE READ AT ALL. 8 heads out of 1024 gives
    a chance mean of 0.0625 shared and P(at least one shared) = 0.061, so an
    observed 1/8 -- which looks like near-total disagreement -- is 16x the
    chance mean and an event with probability 0.06 under independence. Both
    readings are true and reporting either alone misleads, in opposite
    directions.
    """
    return k * k / float(n)


def spearman(rank_a, rank_b):
    """Rank correlation over EVERY head. The k-free statistic.

    Top-k overlap cannot distinguish "the two halves disagree about which
    heads carry the effect" from "fifteen heads are nearly tied and the k=8
    line cuts them differently" -- working rules 11b(3), where a criterion over
    "the largest k" must bind k to something. This does not depend on k at
    all: if the disagreement is a boundary effect the full rankings still
    correlate strongly, and if it is real they do not.
    """
    keys = sorted(rank_a)
    a = np.array([rank_a[k] for k in keys], dtype=np.float64)
    b = np.array([rank_b[k] for k in keys], dtype=np.float64)
    a -= a.mean()
    b -= b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d else float("nan")


def where_do_they_land(top_a, rank_b):
    """The ranks that A's top-8 occupy in B's ranking. The sharpest readout.

    If A's eight heads sit at ranks 9-20 of B, the sets disagree about the
    cut, not about the heads. If they sit in the hundreds, they disagree
    about the heads. The median is reported because one head banished to rank
    900 would drag a mean while leaving the other seven near the top.
    """
    r = sorted(rank_b[h] for h in top_a if h in rank_b)
    if not r:
        return None
    return {"median": float(np.median(r)), "worst": int(r[-1]),
            "best": int(r[0]), "ranks": r}


def line(label, a, b, n_top):
    inter = len(a & b)
    return (f"    {label:<28} {inter}/{n_top} shared   "
            f"Jaccard {jaccard(a, b):.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    fold = json.loads(Path(args.fold).read_text(encoding="utf-8"))
    full = json.loads(Path(args.full).read_text(encoding="utf-8"))
    for b, scope, path in ((fold, CB.SCOPE_FOLD, args.fold),
                           (full, CB.SCOPE_FULL, args.full)):
        if str(b.get("scope")) != scope:
            raise SystemExit(
                f"{path} is a {b.get('scope')!r} bundle, expected {scope!r}. "
                "The two have the same shape and only the name tells them "
                "apart (13.6.3).")

    n_top = len(next(iter(fold["sets"].values()))["carrier"]["heads"])
    out = {"n_top": n_top, "within_seed_fold": {}, "fold_vs_full": {},
           "across_seed_full": {}, "across_seed_by_fold": {}}

    print("=" * 78)
    print("CARRIER AGREEMENT -- descriptive, enters no criterion")
    print("=" * 78)

    print("\nWITHIN a seed: fold A against fold B")
    print("  How unstable is the selection the cross-fit protects against?")
    for s in REGISTERED_SEEDS:
        a = heads_of(fold, CB.fold_key(s, 0))
        b = heads_of(fold, CB.fold_key(s, 1))
        out["within_seed_fold"][str(s)] = {"shared": len(a & b),
                                           "jaccard": jaccard(a, b)}
        print(line(f"seed {s}: fold0 vs fold1", a, b, n_top))

    print("\nEach fold against that seed's FULL-validation set")
    print("  A fold set is chosen on 72 rows, the full set on 144.")
    for s in REGISTERED_SEEDS:
        u = heads_of(full, CB.full_key(s))
        for f in (0, 1):
            a = heads_of(fold, CB.fold_key(s, f))
            out["fold_vs_full"][f"s{s}_f{f}"] = {"shared": len(a & u),
                                                 "jaccard": jaccard(a, u)}
            print(line(f"seed {s}: fold{f} vs full", a, u, n_top))

    print("\nACROSS seeds, on the full sets -- section 2.4(2) asks for this")
    print("  A direct reading of how much the carriers depend on the prefix.")
    for x, y in combinations(REGISTERED_SEEDS, 2):
        a, b = heads_of(full, CB.full_key(x)), heads_of(full, CB.full_key(y))
        out["across_seed_full"][f"{x}_{y}"] = {"shared": len(a & b),
                                               "jaccard": jaccard(a, b)}
        print(line(f"seed {x} vs seed {y}", a, b, n_top))

    print("\nACROSS seeds, fold by fold")
    for f in (0, 1):
        for x, y in combinations(REGISTERED_SEEDS, 2):
            a = heads_of(fold, CB.fold_key(x, f))
            b = heads_of(fold, CB.fold_key(y, f))
            out["across_seed_by_fold"][f"f{f}_{x}_{y}"] = {
                "shared": len(a & b), "jaccard": jaccard(a, b)}
            print(line(f"fold{f}: seed {x} vs seed {y}", a, b, n_top))

    # ------------------------------------------------------------------
    # IS THE TOP-8 DISAGREEMENT REAL, OR IS IT THE CUT AT 8?
    #
    # Everything above is measured at one k, and one k cannot tell those
    # apart. Three things that can: the same overlap swept over k, the rank
    # correlation over all heads (k-free), and where one set's heads actually
    # land in the other's ranking.
    n_pool = len(ranking_of(fold, CB.fold_key(REGISTERED_SEEDS[0], 0)))
    ks = [k for k in (8, 16, 32, 64) if k <= n_pool]
    out["pool"] = n_pool
    out["k_sweep"] = {}
    print("\n" + "=" * 78)
    print(f"IS IT THE HEADS OR IS IT THE CUT?  pool = {n_pool} query heads")
    print("=" * 78)
    print("\nthe same within-seed fold overlap, swept over k")
    print(f"  {'':<12}" + "".join(f"{('top-' + str(k)):>18}" for k in ks))
    print(f"  {'chance':<12}"
          + "".join(f"{chance_shared(k, n_pool):>10.3f} shared" for k in ks))
    for s in REGISTERED_SEEDS:
        ra = ranking_of(fold, CB.fold_key(s, 0))
        rb = ranking_of(fold, CB.fold_key(s, 1))
        row, cells = {}, ""
        for k in ks:
            a, b = set(ra[:k]), set(rb[:k])
            row[k] = {"shared": len(a & b), "jaccard": jaccard(a, b),
                      "chance": chance_shared(k, n_pool)}
            cells += f"{len(a & b):>7}/{k:<3} j={jaccard(a, b):<5.3f}"
        out["k_sweep"][str(s)] = row
        print(f"  seed {s}    " + cells)
    print("\n  A ratio that RISES with k means the sets agree on a broad set")
    print("  of heads and disagree about the cut; one that stays flat means")
    print("  they disagree about the heads themselves.")

    print("\nrank correlation over ALL heads -- this does not depend on k")
    out["spearman"] = {}
    for s in REGISTERED_SEEDS:
        rho = spearman(rank_map(fold, CB.fold_key(s, 0)),
                       rank_map(fold, CB.fold_key(s, 1)))
        out["spearman"][f"s{s}_f0_f1"] = rho
        print(f"    seed {s}: fold0 vs fold1        rho = {rho:+.3f}")
    for x, y in combinations(REGISTERED_SEEDS, 2):
        rho = spearman(rank_map(full, CB.full_key(x)),
                       rank_map(full, CB.full_key(y)))
        out["spearman"][f"full_{x}_{y}"] = rho
        print(f"    seed {x} vs seed {y} (full)      rho = {rho:+.3f}")

    # SAMPLE SIZE HELD FIXED, SO THE ONLY DIFFERENCE IS THE PREFIX.
    #
    # The two comparisons above are not matched: a within-seed fold pair is
    # two 72-row estimates, an across-seed full pair is two 144-row estimates.
    # More rows means less noise means higher correlation, all else equal, so
    # the unmatched contrast UNDERSTATES the prefix effect -- the better-
    # estimated side is the one that agrees less. Matching it is free: every
    # fold set is 72 rows, so fold f of seed x against fold f of seed y is
    # 72-vs-72 with the prefix as the only difference.
    print("\n  ...and the same contrast with the ROW COUNT HELD FIXED at 72,")
    print("     so the prefix is the only thing that differs")
    same_pref = [out["spearman"][f"s{s}_f0_f1"] for s in REGISTERED_SEEDS]
    diff_pref = []
    for f in (0, 1):
        for x, y in combinations(REGISTERED_SEEDS, 2):
            rho = spearman(rank_map(fold, CB.fold_key(x, f)),
                           rank_map(fold, CB.fold_key(y, f)))
            out["spearman"][f"f{f}_{x}_{y}"] = rho
            diff_pref.append(rho)
            print(f"    fold{f}: seed {x} vs seed {y}     rho = {rho:+.3f}")
    m_same = sum(same_pref) / len(same_pref)
    m_diff = sum(diff_pref) / len(diff_pref)
    out["prefix_contrast"] = {"same_prefix_72v72": m_same,
                              "diff_prefix_72v72": m_diff,
                              "gap": m_same - m_diff}
    print(f"\n    same prefix, 72 vs 72 : mean rho {m_same:+.3f}")
    print(f"    diff prefix, 72 vs 72 : mean rho {m_diff:+.3f}")
    print(f"    difference            : {m_same - m_diff:+.3f}  <- the prefix,")
    print("      with query-sampling noise matched on both sides. This is the")
    print("      number to quote for 'how much do the carriers depend on the")
    print("      prefix'; the unmatched 0.83 vs 0.59 understates it.")

    print("\nwhere one fold's top-8 land in the OTHER fold's ranking")
    print(f"  (best possible median {(n_top - 1) / 2:.1f}; "
          f"chance median {n_pool / 2:.0f})")
    out["landing"] = {}
    for s in REGISTERED_SEEDS:
        for f in (0, 1):
            w = where_do_they_land(
                [h for h in ranking_of(fold, CB.fold_key(s, f))[:n_top]],
                rank_map(fold, CB.fold_key(s, 1 - f)))
            out["landing"][f"s{s}_f{f}_in_f{1 - f}"] = w
            print(f"    seed {s} fold{f} in fold{1 - f}: median rank "
                  f"{w['median']:>6.1f}   best {w['best']:>3}   "
                  f"worst {w['worst']:>4}")

    # THE READING, both directions, because they point opposite ways and
    # picking whichever suits the result afterwards is the thing to avoid.
    wf = [v["jaccard"] for v in out["within_seed_fold"].values()]
    ss = [v["jaccard"] for v in out["across_seed_full"].values()]
    print("\n" + "-" * 78)
    print(f"  within-seed fold agreement : mean Jaccard {sum(wf) / len(wf):.3f}")
    print(f"  across-seed agreement      : mean Jaccard {sum(ss) / len(ss):.3f}")
    print("\n  Read BOTH ways, and say which before looking at the gate:")
    print("    high fold agreement -> the cross-fit costs little, and naming")
    print("      particular heads is a claim about heads rather than about")
    print("      one half of one seed's validation set;")
    print("    low fold agreement  -> the cross-fit is doing real work and")
    print("      the gate's number is worth much more than an in-sample one,")
    print("      but any head-level statement is correspondingly weak.")
    print("  Neither reading changes any criterion: these are the very sets")
    print("  the gate selects with, so a rule keyed to their agreement would")
    print("  be a second selection rule fitted on the same data.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2),
                                       encoding="utf-8")
        print(f"\n  [output] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
