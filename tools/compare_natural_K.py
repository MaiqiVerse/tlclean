"""What is one more demonstration per class worth? ZERO GPU.

WHY THIS IS ITS OWN STEP, AHEAD OF THE INTERVENTION IT SERVES. The proposed
experiment moves section 45's construction up one level: a K=5 receiver whose
carrier heads additionally read an offline memory built from a K=10 bank. Its
readout is `recovered / total`, and

    total = NLL(K=10) - NLL(K=5)

is the DENOMINATOR. Nobody has measured it on this task and model. If the
extra demonstrations are worth little, the ratio has a near-zero denominator
and reports a large fraction of nothing -- the degenerate case this project
already refuses elsewhere. So the denominator is measured first, on two plain
natural forwards, before any carrier or memory machinery is built.

WHAT A GO LOOKS LIKE. There is no registered threshold here; this is
exploratory. The scale to read it against is section 45's own total, -2.777211
nats -- what ALL the demonstrations are worth against none. A K=10-over-K=5
total that is a small fraction of that says the increment is nearly free of
information, and the intervention would be measuring a difference of
differences inside noise.

TWO THINGS THIS DOES NOT DO. It is not the intervention and it says nothing
about whether an offline memory could deliver the increment. And it is not
paired to section 45's numbers: the two runs here render their own prefixes at
their own K, so the query sits at a different position in each, which is a
real difference between the arms and not an artifact to be corrected away.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.analyze_unsafe_test import mcnemar_exact_p  # noqa: E402
from tools.method_a_stats import (cluster_bootstrap_ci,  # noqa: E402
                                  mcnemar_floor_p)
from tools.run_viability_forward import scores_from_logits  # noqa: E402

ARM = "natural"


def load(npz_path):
    """(nll[S,Q], acc[S,Q], query_ids[S,Q], seeds, K, cand[C], gold[S,Q], faults).

    BaselineOutput writes the meta to a SIDECAR json named after the same
    stem, not into the npz (working rules 3.2b lists the three layouts in this
    repo; reading the wrong one gives a silent refusal).
    """
    p = Path(npz_path)
    side = p.with_suffix(".json")
    bad = []
    if not p.is_file():
        return (None,) * 7 + ([f"{p}: not a file. run_natural_readout writes "
                               f"baseline_natural_readout_<mode>.npz into the "
                               f"DIRECTORY given to --out"],)
    if not side.is_file():
        return (None,) * 7 + ([f"{side}: missing; the candidate class order "
                               "and K live there"],)
    z = np.load(p, allow_pickle=False)
    meta = json.loads(side.read_text(encoding="utf-8"))
    for key in ("candidate_logits", "gold_class", "query_ids", "arms"):
        if key not in z:
            return (None,) * 7 + [
                [f"{p}: no {key!r}. Keys: {sorted(z.files)}"]]
    arms = [str(a) for a in z["arms"]]
    if ARM not in arms:
        return (None,) * 7 + [[f"{p}: no {ARM!r} arm, only {arms}"]]
    ai = arms.index(ARM)
    cand = np.asarray([int(c) for c in meta["candidate_classes"]])
    lg = np.asarray(z["candidate_logits"], dtype=np.float64)[ai]   # [S, Q, C]
    gold = np.asarray(z["gold_class"], dtype=np.int64)             # [S, Q]
    if lg.shape[:2] != gold.shape:
        bad.append(f"{p}: logits {lg.shape[:2]} and gold {gold.shape} disagree")
    if lg.shape[2] != cand.size:
        bad.append(f"{p}: {lg.shape[2]} candidate columns but the sidecar "
                   f"names {cand.size} classes")
    if bad:
        return (None,) * 7 + [bad]
    S, Q = gold.shape
    nll = np.empty((S, Q))
    acc = np.empty((S, Q))
    for si in range(S):
        for qi in range(Q):
            nll[si, qi], acc[si, qi], _ = scores_from_logits(
                lg[si, qi], int(gold[si, qi]), cand)
    return (nll, acc, np.asarray(z["query_ids"]).astype(str),
            [int(s) for s in z["seeds"]], int(meta.get("K", -1)),
            cand, gold, [])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True,
                    help="the SMALLER K's baseline_natural_readout_*.npz")
    ap.add_argument("--more", required=True,
                    help="the LARGER K's baseline_natural_readout_*.npz")
    ap.add_argument("--reference-total", type=float, default=-2.777211,
                    help="RESULTS 45's NLL(monolithic) - NLL(K0-offset): what "
                         "ALL the demonstrations are worth, as the scale to "
                         "read this increment against")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    n_b, a_b, id_b, s_b, k_b, cand_b, g_b, bad_b = load(args.base)
    n_m, a_m, id_m, s_m, k_m, cand_m, g_m, bad_m = load(args.more)
    faults = list(bad_b) + list(bad_m)
    if n_b is not None and n_m is not None:
        if s_b != s_m:
            faults.append(f"seeds differ: {s_b} vs {s_m}")
        elif id_b.shape != id_m.shape or not np.array_equal(id_b, id_m):
            faults.append("the two runs scored DIFFERENT queries; a paired "
                          "difference over them would not be paired")
        if not np.array_equal(cand_b, cand_m):
            faults.append(
                f"candidate class spaces differ ({cand_b.size} vs "
                f"{cand_m.size} classes; first mismatch at index "
                f"{int(np.argmax(cand_b[:min(cand_b.size, cand_m.size)] != cand_m[:min(cand_b.size, cand_m.size)])) if cand_b.size == cand_m.size else 0}). "
                "The two NLLs are then normalised over different sets and "
                "their difference is not an effect of K")
        # SAME QUERY, SAME ANSWER. Matching ids with differing gold classes
        # would pass the pairing check above and silently compare two
        # different labellings of the same rows.
        elif g_b.shape == g_m.shape and not np.array_equal(g_b, g_m):
            n_bad = int(np.sum(g_b != g_m))
            faults.append(
                f"{n_bad} of {g_b.size} queries carry a DIFFERENT gold class "
                "in the two files. The ids match, so the pairing looks fine; "
                "the labels do not")
        if k_b == k_m:
            faults.append(f"both files record K = {k_b}; there is no "
                          "increment to measure")
        elif k_m < k_b:
            faults.append(f"--more has K={k_m} and --base has K={k_b}; the "
                          "arguments are the wrong way round")
    if faults:
        print("REFUSING:")
        for f in faults:
            print(f"  - {f}")
        return 1

    print("=" * 78)
    print(f"WHAT K={k_m} BUYS OVER K={k_b} -- natural forwards, "
          "no intervention  [EXPLORATORY]")
    print("=" * 78)
    # The uniform predictor is the line that says whether a row is
    # informative at all (RESULTS 48.1); both files are checked above to use
    # the same candidate space, so one number serves both.
    uniform = float(np.log(cand_b.size))
    print(f"  uniform over {cand_b.size} candidates = ln(n) = "
          f"{uniform:.4f} nats")
    print(f"  {len(s_b)} seeds x {n_b.shape[1]} queries, paired on query_id")

    print(f"\n  {'':<12}{'NLL':>10}{'vs uniform':>12}{'accuracy':>10}")
    for tag, n, a in ((f"K={k_b}", n_b, a_b), (f"K={k_m}", n_m, a_m)):
        print(f"  {tag:<12}{n.mean():>10.4f}{n.mean() - uniform:>+12.4f}"
              f"{a.mean():>10.4f}")

    d = [n_m[i] - n_b[i] for i in range(len(s_b))]
    ids = [list(id_b[i]) for i in range(len(s_b))]
    tot = float(np.mean([x.mean() for x in d]))
    lo, hi = cluster_bootstrap_ci(d, ids)
    print(f"\n  total = NLL(K={k_m}) - NLL(K={k_b}) = {tot:+.6f} nats"
          f"   95 % CI [{lo:+.6f}, {hi:+.6f}]")
    print(f"    {'seed':>6}{'K=' + str(k_b):>10}{'K=' + str(k_m):>10}"
          f"{'diff':>11}")
    for i, s in enumerate(s_b):
        print(f"    {s:>6}{n_b[i].mean():>10.4f}{n_m[i].mean():>10.4f}"
              f"{d[i].mean():>+11.6f}")

    # decisions, paired, with the bound that says whether the test could fire
    b = c = 0
    for i in range(len(s_b)):
        x = a_m[i] > 0.5
        y = a_b[i] > 0.5
        b += int(np.sum(y & ~x))
        c += int(np.sum(~y & x))
    pv, fl = mcnemar_exact_p(b, c), mcnemar_floor_p(c - b)
    n_cells = a_b.size
    print(f"\n  decisions, paired: {c - b:+d} net ({b} broken, {c} fixed) "
          f"of {n_cells}")
    print(f"    exact p {pv:.4f}, and the smallest p ANY split with this net "
          f"could give is {fl:.4f}")
    if fl >= 0.05:
        print("    -> CANNOT be significant: a net under 6 queries never "
              "reaches 0.05")

    print("\n  " + "-" * 74)
    ref = args.reference_total
    if ref < 0 and tot < 0:
        print(f"  the increment is {tot / ref:.1%} of what ALL the "
              f"demonstrations are worth ({ref:+.6f}, RESULTS 45)")
    elif tot >= 0:
        print(f"  ⚠ the increment is NOT negative ({tot:+.6f}): K={k_m} does "
              f"not beat K={k_b} here.")
        print("    A `recovered / total` readout built on this denominator "
              "would be a share of")
        print("    a gain that does not exist, and must not be reported.")
    print("  ⚠ EXPLORATORY: no registered threshold. This measures the "
          "DENOMINATOR of a")
    print("    proposed experiment, not the experiment.")
    print("  ⚠ the two runs render their own prefixes, so the query sits at a "
          "different")
    print("    position in each. That is a real difference between K, not an "
          "artifact.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "K_base": k_b, "K_more": k_m, "seeds": s_b,
            "nll_base": float(n_b.mean()), "nll_more": float(n_m.mean()),
            "acc_base": float(a_b.mean()), "acc_more": float(a_m.mean()),
            "total": tot, "ci": [lo, hi],
            "per_seed_diff": [float(x.mean()) for x in d],
            "decisions": {"net": c - b, "broken": b, "fixed": c,
                          "mcnemar_exact_p": pv, "attainable_floor_p": fl},
            "reference_total": ref,
        }, indent=2), encoding="utf-8")
        print(f"\n  [output] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
