"""Read the STOP-following exploratory test forward. ZERO GPU.

WHAT THIS IS NOT. It is not H4 and not H4b. The viability gate returned STOP
and 13.6.7 ended the method path there; 14.0b-19 records the PI's decision to
run the test forward anyway, and 14.0b-20 records that the lock was opened by
discarding a blocker rather than by satisfying it. Every number below is a
STOP-following exploratory reading and may not enter section 1.1. There is no
`passes` verdict here for that reason -- `decide_aggregate` returns one and
this file does not print it as a decision, because the decision was already
made at the gate.

WHY IT REFUSES AN ORDINARY NPZ. It requires the UNSAFE_ prefix and the
`.bypass.json` written beside it. That is not decoration: a result produced
through the bypass has to keep saying so, and an analyzer that would read any
npz at all is how an exploratory number gets quoted as a confirmatory one six
weeks later. If a real code-stage freeze ever exists, the run does not need
this file -- it needs the ordinary path.

WHAT IT REPORTS. At the ONE frozen gamma the run used:

  * Delta NLL = arm - natural, per (query, seed), aggregated by 13.4's
    clustered test on unique query_id. This is the registered estimand.
  * Delta accuracy and Delta Brier at THAT SAME gamma, descriptive. Read at
    the frozen gamma, never re-selected: 13.6.5's rule exists because
    re-minimising on accuracy picks the WORST-accuracy gamma and reports it
    as the method's.

Expect it to be worse than the in-sample validation number. 13.6.6 forbids
the two substituting for one another, and the gate's out-of-fold -0.010057 is
the honest expectation for anything held out; the full-validation readout's
optimism was measured at 0.0007 nats.

Run:
    python tools/analyze_unsafe_test.py \\
        --npz results/UNSAFE_test_s43/UNSAFE_method_a_test_seed43.npz \\
        --json-out results/UNSAFE_test_s43/UNSAFE_test_readout.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.method_a_stats import decide_aggregate  # noqa: E402
from tools.run_viability_forward import scores_from_logits  # noqa: E402

CAVEAT = ("STOP-following exploratory test reading (14.0b-19, 14.0b-20). "
          "Not H4, not H4b, not admissible in section 1.1.")


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar p for discordant counts (b, c).

    Conditional on b + c discordant pairs, b is Binomial(b + c, 1/2) under
    "the arm changes nothing systematically". Exact rather than the chi-square
    form because the counts here are single digits, where the asymptotic
    version is not usable. b + c == 0 returns 1.0: no discordant pair is no
    evidence of a difference, not a perfect one.
    """
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def cell_guard_faults(path, meta=None):
    """Why this npz may not be read as a bypassed TEST cell. Empty means it may.

    Split out so a second reader over the same files cannot quietly accept
    what this one refuses -- the UNSAFE_ prefix and the bypass record are the
    whole reason an exploratory number stays labelled as one, and a analyzer
    that skipped them would undo that by existing.
    """
    p = Path(path)
    bad = []
    if not p.name.startswith("UNSAFE_"):
        bad.append(f"{p.name} does not start with 'UNSAFE_'. This reader is "
                   "for the bypassed test path only; a result from a real "
                   "code-stage freeze goes through the ordinary analysis, "
                   "and reading it here would attach a caveat it has not "
                   "earned")
    rec_path = Path(str(p) + ".bypass.json")
    if not rec_path.is_file():
        bad.append(f"{rec_path} is missing. Every UNSAFE run writes what the "
                   "lock objected to next to its output; without it there is "
                   "no record of what was not in place when this number was "
                   "produced")
    if bad:
        return bad
    if meta is None:
        meta = json.loads(str(np.load(p, allow_pickle=False)["meta"]))
    if not str(meta.get("split", "")).startswith("test"):
        bad.append(f"{p}: split is {meta.get('split')!r}. On validation "
                   "nothing was bypassed and nothing here applies")
    gammas = [float(g) for g in (meta.get("gammas") or [])]
    if len(gammas) != 1:
        bad.append(f"{p}: {len(gammas)} gammas. 3 freezes ONE gamma* per "
                   "setting before any test prediction, and a grid on test "
                   "is a grid chosen after seeing test")
    return bad


def read_cell(path):
    """(per-row scores, meta, faults) for one UNSAFE test npz.

    Faults are collected rather than raised so that all of them are reported
    at once; a run that has to be relaunched three times to learn three facts
    is the failure mode this project keeps hitting on the server.
    """
    p = Path(path)
    if not p.is_file():
        return None, None, [f"{p} does not exist"]
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    bad = cell_guard_faults(p, meta)
    if bad:
        return None, meta, bad

    cand_cls = np.asarray(z["candidate_classes"])
    rows = []
    for qi, cidx in enumerate(int(c) for c in z["class_idx"]):
        n0, a0, b0 = scores_from_logits(z["logits_natural"][qi], cidx,
                                        cand_cls)
        n1, a1, b1 = scores_from_logits(z["logits_arm"][qi][0], cidx, cand_cls)
        rows.append((str(z["query_id"][qi]), int(z["demo_seed"][qi]),
                     n0, a0, b0, n1, a1, b1))
    return rows, meta, []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", action="append", required=True,
                    help="one per demo seed; repeat the flag")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    cells, faults, metas = {}, [], {}
    for path in args.npz:
        rows, meta, bad = read_cell(path)
        faults += bad
        if bad:
            continue
        seeds = {r[1] for r in rows}
        if len(seeds) != 1:
            faults.append(f"{path}: rows carry demo seeds {sorted(seeds)}. "
                          "13.4 runs test_seed ONE seed per job because each "
                          "seed's 250 are a different draw")
            continue
        sd = seeds.pop()
        if sd in cells:
            faults.append(f"seed {sd} was given twice; the second is {path}")
            continue
        cells[sd] = rows
        metas[sd] = meta
    if faults:
        print("REFUSING TO READ:")
        for f in faults:
            print(f"  - {f}")
        return 2

    seeds = sorted(cells)
    d_nll = [np.array([r[5] - r[2] for r in cells[s]]) for s in seeds]
    ids = [[r[0] for r in cells[s]] for s in seeds]
    gamma = float(metas[seeds[0]]["gammas"][0])

    print("=" * 78)
    print("UNSAFE TEST READOUT -- " + CAVEAT)
    print("=" * 78)
    print(f"  arm        {metas[seeds[0]]['arm']} / "
          f"{metas[seeds[0]]['carrier_impl']}")
    print(f"  gamma      {gamma}   (frozen; the only one this run computed)")
    print(f"  split      {metas[seeds[0]]['split']}")
    print(f"  seeds      {seeds}   ({[len(cells[s]) for s in seeds]} queries)")
    if len(seeds) < 3:
        print(f"\n  ⚠ {len(seeds)} of 3 prefix seeds. T is the average over "
              "the seeds PRESENT,")
        print("  ⚠ and 13.4's estimand is the average over all three. Do not "
              "report this as that.")

    print("\n  Delta NLL = arm - natural, per query. Negative is better.")
    for s in seeds:
        d = np.array([r[5] - r[2] for r in cells[s]])
        print(f"    seed {s}: mean {d.mean():+.6f} nats   "
              f"median {np.median(d):+.6f}   "
              f"frac improved {float((d < 0).mean()):.3f}")

    res = decide_aggregate(d_nll, ids, "decrease")
    lo, hi = res["ci95"]
    print("\n  clustered on unique query_id (13.4's aggregate test)")
    print(f"    T          {res['T']:+.6f} nats")
    print(f"    95% CI     [{lo:+.6f}, {hi:+.6f}]")
    nperm = res["params"]["n_permutations"]
    floor = 1.0 / (nperm + 1)
    at_floor = " -- AT THE RESOLUTION FLOOR" if res["p"] <= floor else ""
    # NOT "%.4f". With 1e6 permutations the smallest attainable p is
    # 1/(n+1) ~ 1e-6, so "0.0000" prints a value the estimator cannot
    # produce and invites reading it as zero.
    print(f"    p          {res['p']:.3e}   (two-sided sign-flip, "
          f"{nperm} permutations; floor {floor:.3e}{at_floor})")
    print(f"    clusters   {res['n_clusters']}  observations "
          f"{res['n_observations']}")
    print("\n    ⚠ p is reported as a descriptive statistic. This is not a "
          "preregistered")
    print("    ⚠ test: 13.6.7's path ended at the gate, so there is no "
          "hypothesis left")
    print("    ⚠ for it to decide. " + CAVEAT)

    print("\n  auxiliaries AT THE SAME FROZEN GAMMA (13.6.5: never "
          "re-selected --")
    print("  re-minimising on accuracy picks the worst-accuracy gamma)")
    aux = {}
    for name, i0, i1 in (("accuracy", 3, 6), ("brier", 4, 7)):
        per_seed = []
        for s in seeds:
            nat = np.array([r[i0] for r in cells[s]])
            arm = np.array([r[i1] for r in cells[s]])
            per_seed.append((float(nat.mean()), float(arm.mean())))
            print(f"    seed {s} {name:8}  natural {nat.mean():.4f}  "
                  f"arm {arm.mean():.4f}  delta {arm.mean() - nat.mean():+.4f}")
        aux[name] = per_seed

    # THE ACCURACY DELTA IS A COUNT OF QUERIES, AND A MEAN DIFFERENCE HIDES
    # THAT. +0.0120 on 250 rows is three queries, and the paired quantity that
    # carries an uncertainty is the DISCORDANT pair (b, c) -- how many the arm
    # broke against how many it fixed. Reporting the mean alone gives a number
    # with no attachable SE (working rules 2.2 rule 10(4)), and a net of +3 built
    # from 40 broken and 43 fixed is a different finding from one built from
    # 0 and 3.
    print("\n  accuracy as PAIRED FLIPS (the net above is their difference)")
    flips = {}
    for s in seeds:
        b = sum(1 for r in cells[s] if r[3] > 0.5 and r[6] < 0.5)   # broke
        c = sum(1 for r in cells[s] if r[3] < 0.5 and r[6] > 0.5)   # fixed
        n = len(cells[s])
        p_mc = mcnemar_exact_p(b, c)
        flips[s] = {"n": n, "broke": b, "fixed": c, "net": c - b,
                    "discordant": b + c, "mcnemar_exact_p": p_mc}
        print(f"    seed {s}: {n} queries -- arm BROKE {b}, FIXED {c}, "
              f"net {c - b:+d} ({(c - b) / n:+.4f})")
        print(f"      discordant {b + c}; McNemar exact two-sided p = "
              f"{p_mc:.4f}")
    # POOLED, because per seed the discordant count is single digits and
    # 2*(1/2)^n is the floor an exact test can reach there: at b+c=3 the
    # smallest attainable two-sided p is 0.25, so that cell CANNOT come out
    # significant however the three fall. Pooling is the only way the
    # accuracy question is answerable at all -- but it is only exact when the
    # pooled pairs are independent, so the repeated-query count is printed
    # beside it rather than assumed away.
    if len(seeds) > 1:
        B = sum(flips[s]["broke"] for s in seeds)
        C = sum(flips[s]["fixed"] for s in seeds)
        n_rep = res["n_repeated_queries"]
        print(f"    POOLED over {len(seeds)} seeds: broke {B}, fixed {C}, "
              f"net {C - B:+d}, discordant {B + C}")
        print(f"      McNemar exact two-sided p = {mcnemar_exact_p(B, C):.4f}"
              f"   (floor at this count: {mcnemar_exact_p(0, B + C):.4f})")
        if n_rep:
            print(f"      ⚠ {n_rep} query_id appear under more than one "
                  "prefix. Those pairs are correlated,")
            print("      ⚠ so the pooled exact p is anti-conservative. The "
                  "per-seed rows above are not.")
        else:
            print("      no query_id repeats across seeds, so the pooled "
                  "pairs are independent and this p is exact")
        flips["pooled"] = {"broke": B, "fixed": C, "net": C - B,
                           "discordant": B + C,
                           "mcnemar_exact_p": mcnemar_exact_p(B, C),
                           "attainable_floor": mcnemar_exact_p(0, B + C),
                           "repeated_query_ids": n_rep}

    print("    the gate's accuracy delta at the NLL-selected gamma was "
          "+0.000000 (13.6.5,")
    print("    read with apply_choice); a cross-fit re-selection there gave "
          "+0.002315 and was wrong.")

    print("\n  reference points, NOT comparisons this run earns:")
    print("    out-of-fold gate (honest expectation)   -0.010057 nats")
    print("    full-validation in sample               optimism 0.0007 nats")
    print("\n  " + CAVEAT)

    out = {"spec": "prereg_method_A.md 14.0b-19 and 14.0b-20",
           "status": CAVEAT,
           "gamma": gamma,
           "seeds": seeds,
           "n_per_seed": {str(s): len(cells[s]) for s in seeds},
           "delta_nll": {"T": res["T"], "p": res["p"],
                         "ci95": [float(lo), float(hi)],
                         "n_clusters": res["n_clusters"],
                         "n_observations": res["n_observations"],
                         "per_seed_means": res["per_seed_means"],
                         "params": res["params"]},
           "auxiliaries_at_frozen_gamma": {
               k: [{"seed": s, "natural": v[0], "arm": v[1],
                    "delta": v[1] - v[0]}
                   for s, v in zip(seeds, vals)] for k, vals in aux.items()},
           "accuracy_flips": {str(k): f for k, f in flips.items()},
           "source_npz": list(args.npz),
           "not_licensed": (
               "13.6.7 ended the method path at the viability gate. Nothing "
               "here is H4, H4b, or admissible in section 1.1. The p-value is "
               "descriptive: there is no preregistered hypothesis left for it "
               "to decide.")}
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  [output] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
