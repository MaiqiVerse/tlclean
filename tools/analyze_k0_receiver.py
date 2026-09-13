"""How much of the demonstrations does the selective memory recover? ZERO GPU.

THE ESTIMAND. Four arms differ only by which heads may see the demo columns,
so the two differences that matter are

    total       NLL(monolithic) - NLL(K0-offset)     what SEEING the demos buys
    recovered   NLL(selective)  - NLL(K0-offset)     what the 8 carrier heads
                                                     seeing them buys

and their ratio is the fraction of the demonstrations' benefit that survives
when only the frozen H_8 may read the memory. Both differences are reported
signed and separately, because the ratio alone hides which of the two moved
(working rules 4).

⚠ `all-head cached K5` IS NOT A METHOD ARM. 13.5.4 makes it the cache-
equivalence GATE: it is the ordinary prompt computed through the cache, so
its number says nothing about the method and is printed only so the gate's
subject stays visible.

⚠ THE NAME IS `K0 receiver + selective K5 latent memory`, not "zero-shot
Method A" (13.5.1). The receiver's text carries no demonstrations, but the
memory is built from a labelled K=5 bank, so this is a zero-demo receiver
with labelled offline memory. The short name is a claim the design does not
support.

⚠ NO EFFICIENCY CLAIM IS AVAILABLE FROM THIS RUN (13.5.4). The mask is
additive over columns that are still computed, so the K=5-length matmuls run
for every head; behaviour and theoretical sparse FLOPs may be reported,
latency and compute may not. The efficiency baseline, when one is measured,
is the full-K5 PREFIX CACHE and not a per-query recompute.

⚠ 13.5.1 places this section after the main line, whose gate returned STOP.
This runs on VALIDATION, so it needs no test-lock decision, but it is not
part of H1-H6 either.

Run:
    python tools/analyze_k0_receiver.py \\
        --npz results/k0_receiver_L31.npz \\
        --json-out results/k0_receiver_readout.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.method_a_stats import (cluster_bootstrap_ci,  # noqa: E402
                                  mcnemar_floor_p)
from tools.analyze_unsafe_test import mcnemar_exact_p  # noqa: E402
from tools.run_k0_receiver import (ARM_ALL, ARM_BASE,  # noqa: E402
                                   ARM_MONO, ARM_SEL, ARMS,
                                   parse_layer_prefix,
                                   parse_tsla_alpha, parse_tsla_family)
from tools.run_viability_forward import scores_from_logits  # noqa: E402
from tools.vector_arms import parse_vector_arm, vector_family_label  # noqa: E402


ROLE_DEFAULTS = (("mono", "ARM_MONO"), ("all", "ARM_ALL"),
                 ("sel", "ARM_SEL"), ("base", "ARM_BASE"))


def roles_of(meta, override=None):
    """The four arms by role: (monolithic, all-head, selective, baseline).

    A file that declares `roles` names them itself; one that does not is a
    run_k0_receiver output and gets that module's names. Reading by role is
    what lets the same estimand be computed for the K=10 increment arm, whose
    arms are the same four things at another K.
    """
    r = dict(meta.get("roles") or {})
    # An explicit override wins over the file, which is how a run made before
    # the probe recorded its roles is read without editing the npz.
    r.update({k: v for k, v in (override or {}).items() if v})
    return (r.get("mono", ARM_MONO), r.get("all", ARM_ALL),
            r.get("sel", ARM_SEL), r.get("base", ARM_BASE))


def load(path, roles=None):
    """(scores, meta, seeds, query ids, classes, ARMS PRESENT, faults).

    The arm list comes from the FILE, not from this module's tuple: a depth
    sweep writes `layer<=L` arms beside the four and iterating the tuple
    would silently drop every one of them.
    """
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    # THE SEEDS MAY BE IN EITHER PLACE. run_k0_receiver records them in the
    # meta json; the K=10 increment probe wrote them as a top-level array.
    # Refusing one of those would be refusing a complete file over where a
    # list of three integers was put.
    if meta.get("seeds") is not None:
        seeds = [int(x) for x in meta["seeds"]]
    elif "seeds" in z:
        seeds = [int(x) for x in np.asarray(z["seeds"]).ravel()]
    elif isinstance(meta.get("per_seed"), dict) and meta["per_seed"]:
        # RECOVERED FROM THE FILE'S OWN STRUCTURE, not guessed. `per_seed` is
        # built inside the probe's `for s in REGISTERED_SEEDS` loop, and a
        # json object preserves insertion order, so its keys are the seeds in
        # exactly the order query_ids and class_idx are stacked in. That
        # ordering is the thing that matters: a set of the right three
        # integers in the wrong order would pair every query with another
        # seed's gold class and nothing would raise.
        seeds = [int(k) for k in meta["per_seed"]]
    else:
        return None, meta, None, None, None, None, [
            f"{path}: no seeds -- not in the meta json, not as an array, and "
            f"no `per_seed` to recover them from. Keys: {sorted(z.files)}; "
            f"meta keys: {sorted(meta)}"]
    qids = np.asarray(z["query_ids"]).astype(str)
    cls = np.asarray(z["class_idx"]).astype(int)
    bad = []
    if qids.shape != cls.shape:
        bad.append(f"query_ids {qids.shape} and class_idx {cls.shape} disagree")
    # THE ARMS THE FILE ACTUALLY HOLDS, not the four this module knows.
    # A depth sweep writes `layer<=L` arms beside them, and iterating the
    # module's own tuple would silently drop every one of them.
    arms = list(meta.get("arms") or ARMS)
    missing = [a for a in roles_of(meta, roles) if a not in arms]
    if missing:
        # Name what IS there. Four arms under other names is the ordinary
        # case for a same-design run at another K, not a broken file.
        missing = [f"{m!r}" for m in missing]
    if missing:
        bad.append(f"the file does not carry the base arms {missing}; the "
                   "gate and the two references are what everything else is "
                   "read against")
    per = {}
    for a in arms:
        for si, s in enumerate(seeds):
            key = f"{a}_seed{s}"
            if key not in z:
                bad.append(f"{key} is missing from the npz")
                continue
            arr = np.asarray(z[key], dtype=np.float64)
            # THE SILENT MISMATCH THIS CHECKS FOR: a --limit run once wrote
            # the sliced logit stacks beside the UNSLICED query ids, so four
            # rows of numbers described 144 queries. Every array in one file
            # has to be about the same queries.
            if arr.shape[0] != qids.shape[1]:
                bad.append(
                    f"{key} holds {arr.shape[0]} rows but the file records "
                    f"{qids.shape[1]} queries for seed {s}; the arrays "
                    "describe different query sets and no readout of them "
                    "means anything")
                continue
            per[(a, s)] = arr
    return per, meta, seeds, qids, cls, arms, bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--json-out")
    # THE FOUR ROLES, for a file that predates the probe recording them.
    # Same design, other names: the K=10 increment run's arms are
    # "full K10 monolithic" / "all-head cached K10" /
    # "selective TL K10 memory" / "K5-offset natural".
    for _r, _d in (("mono", ARM_MONO), ("all", ARM_ALL),
                   ("sel", ARM_SEL), ("base", ARM_BASE)):
        ap.add_argument(f"--arm-{_r}", default=None,
                        help=f"the arm playing the {_r!r} role "
                             f"(file's `roles`, else {_d!r})")
    args = ap.parse_args(argv)

    _roles = {"mono": args.arm_mono, "all": args.arm_all,
              "sel": args.arm_sel, "base": args.arm_base}
    per, meta, seeds, qids, cls, arms, bad = load(args.npz, _roles)
    # Shadowed under new names on purpose: rebinding A_MONO here would make
    # it local to main() and turn the reference on the right-hand side into an
    # unbound-local error.
    A_MONO, A_ALL, A_SEL, A_BASE = roles_of(meta or {}, _roles)
    if bad:
        print("REFUSING TO READ:")
        for b in bad:
            print(f"  - {b}")
        return 2

    # candidate_classes are not in this npz; the label space fixed the column
    # order and class_idx indexes classes, so the gold column is looked up the
    # same way the forward did -- by position in the eligible class list.
    cand_classes = np.array(sorted(set(int(c) for c in cls.ravel())))
    n_cand = next(iter(per.values())).shape[1]
    if cand_classes.size != n_cand:
        # fall back to the full eligible range rather than guessing a subset
        cand_classes = np.arange(n_cand)

    print("=" * 78)
    # THE FILE SAYS WHAT IT IS. This banner was the K=0 arm's name, printed
    # verbatim over a K=10 increment run -- a heading that names a design the
    # file is not.
    print(f"{str(meta.get('arm_name', 'K0 receiver')).upper()} "
          "-- VALIDATION readout")
    print("=" * 78)
    _src = ("meta" if meta.get("seeds") is not None
            else "the npz array" if "seeds" in np.load(args.npz).files
            else "meta['per_seed'] keys (this file records them nowhere else)")
    print(f"  {meta['arm_name']}   seeds {seeds} (from {_src})   "
          f"{qids.shape[1]} queries per seed")
    if meta.get("split") and meta["split"] != "validation":
        print(f"  ⚠⚠ SPLIT {str(meta['split']).upper()} -- "
              f"{meta.get('test_read_decision') or 'no decision recorded'}")

    nll = {}
    acc = {}
    for a in arms:
        for si, s in enumerate(seeds):
            arr = per[(a, s)]
            vals = [scores_from_logits(arr[i], int(cls[si, i]), cand_classes)
                    for i in range(arr.shape[0])]
            nll[(a, s)] = np.array([v[0] for v in vals])
            acc[(a, s)] = np.array([v[1] for v in vals])

    # THE UNIFORM PREDICTOR IS THE LINE THAT DECIDES HOW THESE ROWS READ.
    # A candidate-conditional NLL above ln(n_cand) means the model is doing
    # WORSE than assigning every candidate 1/n -- it is confidently wrong, not
    # merely uninformed. Without this column a row of 4.18 looks like "some
    # way along from 4.45 to 1.67"; with it, 4.18 is still 0.60 nats on the
    # wrong side of a coin, which is why its argmax sits at chance.
    uniform = float(np.log(len(cand_classes)))
    print(f"\n  uniform over {len(cand_classes)} candidates = ln(n) = "
          f"{uniform:.4f} nats. An arm ABOVE this is worse than guessing.")
    print("\n  per arm, mean over all seeds")
    print(f"    {'arm':<24} {'NLL':>10} {'vs uniform':>12} {'accuracy':>10}")
    summary = {}
    for a in arms:
        n = float(np.mean([nll[(a, s)].mean() for s in seeds]))
        c = float(np.mean([acc[(a, s)].mean() for s in seeds]))
        # PER SEED FOR BOTH, not just NLL. The natural/monolithic arm IS
        # the model's own accuracy in the registered 36-candidate readout,
        # and the per-seed spread is what makes it comparable with the test
        # split's (RESULTS 44.4: 0.4520 / 0.6560 / 0.4400). A pooled number
        # alone cannot be put beside those.
        summary[a] = {"nll": n, "accuracy": c,
                      "per_seed_nll": [float(nll[(a, s)].mean())
                                       for s in seeds],
                      "per_seed_accuracy": [float(acc[(a, s)].mean())
                                            for s in seeds]}
        summary[a]["nll_vs_uniform"] = n - uniform
        tag = "   <- GATE, not a method arm" if a == A_ALL else ""
        if n > uniform:
            tag = "   worse than uniform" + tag
        print(f"    {a:<24} {n:>10.4f} {n - uniform:>+12.4f} "
              f"{c:>10.4f}{tag}")

    # ---- the two differences, signed and separate ----
    ids = [list(qids[si]) for si in range(len(seeds))]
    d_tot = [nll[(A_MONO, s)] - nll[(A_BASE, s)] for s in seeds]
    d_rec = [nll[(A_SEL, s)] - nll[(A_BASE, s)] for s in seeds]
    tot = float(np.mean([d.mean() for d in d_tot]))
    rec = float(np.mean([d.mean() for d in d_rec]))
    lo_t, hi_t = cluster_bootstrap_ci(d_tot, ids)
    lo_r, hi_r = cluster_bootstrap_ci(d_rec, ids)
    print("\n  what SEEING the demonstrations is worth, and what the 8 heads "
          "recover")
    print(f"    total     NLL({A_MONO}) - NLL({A_BASE}) = {tot:+.6f} nats"
          f"   95 % CI [{lo_t:+.6f}, {hi_t:+.6f}]")
    print(f"    recovered NLL({A_SEL}) - NLL({A_BASE}) = {rec:+.6f} nats"
          f"   95 % CI [{lo_r:+.6f}, {hi_r:+.6f}]")

    # THE RATIO NEEDS A DENOMINATOR THAT MEANS SOMETHING. If seeing the demos
    # does not help, "the fraction recovered" has nothing to be a fraction of
    # -- and this project has printed 1.97e15 once already for exactly that.
    if tot >= 0:
        print("\n    ⚠ NO FRACTION IS QUOTED: the monolithic arm is not better "
              "than the K0-offset")
        print("      baseline, so there is no benefit for the selective memory "
              "to recover a share of.")
        frac = float("nan")
    else:
        frac = rec / tot
        print(f"\n    recovered fraction {frac:.1%} of the demonstrations' "
              "NLL benefit")
        print("    (a ratio of two MEANS, not a mean of per-query ratios: the "
              "per-query denominator")
        print("     crosses zero and its reciprocal has no finite mean.)")

    print("\n  per seed")
    print(f"    {'seed':>5} {'total':>11} {'recovered':>11} {'fraction':>10}")
    per_seed = {}
    for si, s in enumerate(seeds):
        t, r = float(d_tot[si].mean()), float(d_rec[si].mean())
        f = r / t if t < 0 else float("nan")
        per_seed[str(s)] = {"total": t, "recovered": r, "fraction": f}
        print(f"    {s:>5} {t:>+11.6f} {r:>+11.6f} "
              + (f"{f:>10.1%}" if np.isfinite(f) else f"{'n/a':>10}"))

    print("\n  accuracy per seed (the monolithic row IS the model's own "
          "accuracy in the")
    print("  registered 36-candidate readout -- the arm writes nothing)")
    print(f"    {'arm':<24}" + "".join(f"{s:>10}" for s in seeds)
          + f"{'mean':>10}")
    for a in (A_MONO, A_SEL, A_BASE):
        row = "".join(f"{v:>10.4f}" for v in summary[a]["per_seed_accuracy"])
        print(f"    {a:<24}{row}{summary[a]['accuracy']:>10.4f}")
    print(f"    {'chance (1/n_cand)':<24}"
          + "".join(f"{1.0 / n_cand:>10.4f}" for _ in seeds)
          + f"{1.0 / n_cand:>10.4f}")

    print("\n  " + "-" * 74)
    print(f"  ⚠ `{A_ALL}` above is the cache-equivalence GATE (13.5.4), "
          "not a method arm:")
    print("    it is the ordinary prompt computed through the cache.")
    # The second caveat is about the K=0 DESIGN and does not describe a run
    # whose receiver has demonstrations, so it is printed only for that one.
    if A_BASE == ARM_BASE:
        print("  ⚠ The arm is `K0 receiver + selective K5 latent memory`, "
              "NOT \"zero-shot Method A\"")
        print("    (13.5.1): the receiver carries no demonstrations, but the "
              "memory is built from a")
        print("    labelled K=5 bank.")
    else:
        print(f"  ⚠ `{A_BASE}` is this file's baseline, and it is NOT another "
              "run's condition of")
        print("    the same name: every arm here runs the same token "
              "sequence, so the query and")
        print("    the base demonstrations sit where THAT sequence puts them. "
              "Compare totals by")
        print("    magnitude with a separately rendered run, never bitwise.")
    print("  ⚠ NO efficiency claim is available from this implementation "
          "(13.5.4): the mask is")
    print("    additive over columns that are still computed.")

    # ---- the depth curve, if this run has one ----
    depth = sorted((parse_layer_prefix(a), a) for a in arms
                   if parse_layer_prefix(a) is not None)
    if depth:
        print("\n  DEPTH CURVE -- every head in layers 0..L reads the demo "
              "columns  [EXPLORATORY]")
        print(f"    {'L':>5} {'NLL':>10} {'accuracy':>10} "
              f"{'recovered':>11} {'fraction':>10}   recovered per seed")
        for L, a in depth:
            n = summary[a]["nll"]
            c = summary[a]["accuracy"]
            r = n - summary[A_BASE]["nll"]
            f = r / tot if tot < 0 else float("nan")
            # PER SEED TOO. The curve is not monotone -- shallow prefixes come
            # back WORSE than total blindness -- and a pooled mean cannot say
            # whether that is one seed or all three.
            ps = [summary[a]["per_seed_nll"][i]
                  - summary[A_BASE]["per_seed_nll"][i]
                  for i in range(len(seeds))]
            print(f"    {L:>5} {n:>10.4f} {c:>10.4f} {r:>+11.6f} "
                  + (f"{f:>10.1%}" if np.isfinite(f) else f"{'n/a':>10}")
                  + "   " + "  ".join(f"{v:>+8.4f}" for v in ps)
                  + ("   ALL 3 WORSE THAN BLIND" if all(v > 0 for v in ps)
                     else "   mixed" if any(v > 0 for v in ps) else ""))
        # THE ENDS ARE SELF-CHECKS. A curve whose top does not land on the
        # all-head arm, or whose bottom does not land on the K0-offset one,
        # is not a curve of what it claims.
        hi_L, hi_a = depth[-1]
        lo_L, lo_a = depth[0]
        d_hi = abs(summary[hi_a]["nll"] - summary[A_ALL]["nll"])
        d_lo = abs(summary[lo_a]["nll"] - summary[A_BASE]["nll"])
        print(f"    end check: L={hi_L} vs all-head cached  |dNLL| = "
              f"{d_hi:.2e}" + ("   [PASS]" if d_hi < 1e-6 else
                               "   [FAIL] the top of the curve is not the "
                               "all-head arm"))
        print(f"    end check: L={lo_L} vs K0-offset        |dNLL| = "
              f"{d_lo:.2e}" + ("   [PASS]" if d_lo < 1e-6 else
                               "   [FAIL] the bottom of the curve is not the "
                               "baseline"))
        print("    ⚠ these two are self-checks, not findings: they are the "
              "same computation")
        print("      reached by a different mask, so anything but agreement "
              "is a bug.")

    # ---- accuracy, PAIRED against K0-offset ----
    # The accuracy table above reports levels. Levels cannot say whether an
    # arm that is 4 queries ahead fixed four and broke none or fixed thirty
    # and broke twenty-six, and those are different findings. Every arm is
    # scored on the SAME rows, so the paired counts exist and are free.
    print(f"\n  accuracy PAIRED against `{A_BASE}`, pooled over seeds")
    print("    b = broke a correct answer, c = fixed a wrong one. `floor` is "
          "the smallest p")
    print("    ANY split with this net could give (b = 0); a net under 6 "
          "queries can never")
    print("    reach 0.05, whatever the counts are.")
    print(f"    {'arm':<24}{'net':>6}{'b':>5}{'c':>5}{'exact p':>10}"
          f"{'floor':>9}   verdict")
    acc_pair = {}
    for a in arms:
        if a in (A_BASE, A_MONO):
            continue
        b = c = 0
        for s in seeds:
            x = np.asarray(acc[(a, s)]) > 0.5
            y = np.asarray(acc[(A_BASE, s)]) > 0.5
            b += int(np.sum(y & ~x))
            c += int(np.sum(~y & x))
        pv = mcnemar_exact_p(b, c)
        fl = mcnemar_floor_p(c - b)
        note = ("CANNOT be significant" if fl >= 0.05
                else "significant" if pv < 0.05 else "not significant")
        print(f"    {a:<24}{c - b:>+6}{b:>5}{c:>5}{pv:>10.4f}{fl:>9.4f}   "
              f"{note}")
        acc_pair[a] = {"net": c - b, "broke": b, "fixed": c,
                       "mcnemar_exact_p": pv, "attainable_floor_p": fl,
                       "can_be_significant": fl < 0.05}
    n_cells = len(seeds) * qids.shape[1]
    chance = 1.0 / len(cand_classes)
    sd = float(np.sqrt(n_cells * chance * (1 - chance)))
    print(f"    chance is {chance * n_cells:.1f}/{n_cells} with a binomial SD "
          f"of {sd:.2f} queries, so a")
    print(f"    difference of a few queries is inside the spread of a coin "
          "that never saw a demo.")

    # ---- the TSLA alpha curves, one per FAMILY, if this run has any ----
    # 13.5.4's `TSLA-TL-zero-demo` is one family (a K=5 vector into a K=0
    # receiver); run_k10_increment's `TSLA-K10` / `TSLA-K10inc` are two more
    # (a K=10 vector into the K=5 receiver, the second with the receiver's
    # own contribution subtracted). One curve each, each gated at alpha = 0
    # against the file's own baseline arm, each read beside the selective
    # memory it is the alternative to.
    # The FV / TV baselines (tools/vector_arms) are read here too: one curve
    # per vector, the same alpha = 0 gate, the same comparison. They are
    # written under `vector_alphas`, not `tsla_alphas`, so a reader of the
    # JSON never mistakes one for the other.
    fams, vec_fams = {}, set()
    for a in arms:
        fa = parse_tsla_family(a)
        if fa is not None:
            fams.setdefault(fa[0], []).append((fa[1], a))
            continue
        pv = parse_vector_arm(a)
        if pv is not None:
            lab = vector_family_label(a)
            fams.setdefault(lab, []).append((pv[3], a))
            vec_fams.add(lab)
    tsla_meta = (meta or {}).get("tsla") or {}
    vec_meta = {k: (meta or {}).get(k) or {} for k in ("fv", "tv", "icv", "i2cl")}
    tsla_json, vec_json = {}, {}
    for fam in fams:
        tsla = sorted(fams[fam])
        what = (tsla_meta.get("families") or {}).get(fam)
        if fam in vec_fams:
            method = fam.split("-", 1)[0].lower()          # fv / tv
            what = vec_meta.get(method, {}).get("what")
            print(f"\n  {fam} -- ONE frozen vector on `{A_BASE}`'s mask  "
                  "[EXPLORATORY baseline]")
            if what:
                print(f"    {what}")
            else:
                print("    (the file's meta carries no description of this "
                      "vector)")
        elif fam == "TL-zero-demo":
            print("\n  TSLA-TL-zero-demo -- ONE frozen steering vector, no "
                  "demo columns  [13.5.4]")
            print("    the mask is K0-offset's: this arm sees no "
                  "demonstrations at all, and the")
            print("    vector is the only thing it carries over from having "
                  "seen them.")
        else:
            print(f"\n  TSLA-{fam} -- ONE frozen steering vector on "
                  f"`{A_BASE}`'s mask  [EXPLORATORY]")
            if what:
                print(f"    {what}")
            else:
                print("    (the file's meta carries no description of this "
                      "family)")
        print(f"    {'alpha':>7} {'NLL':>10} {'accuracy':>10} "
              f"{'vs base':>13} {'fraction':>10}   per seed")
        for al, a in tsla:
            n = summary[a]["nll"]
            c = summary[a]["accuracy"]
            r = n - summary[A_BASE]["nll"]
            f = r / tot if tot < 0 else float("nan")
            ps = [summary[a]["per_seed_nll"][i]
                  - summary[A_BASE]["per_seed_nll"][i]
                  for i in range(len(seeds))]
            print(f"    {al:>7.2f} {n:>10.4f} {c:>10.4f} {r:>+13.6f} "
                  + (f"{f:>10.1%}" if np.isfinite(f) else f"{'n/a':>10}")
                  + "   " + "  ".join(f"{v:>+8.4f}" for v in ps)
                  + ("   ALL 3 WORSE" if all(v > 0 for v in ps)
                     else "   mixed" if any(v > 0 for v in ps) else ""))
        # 13.5.4(4): alpha = 0 IS the baseline arm. The probe gated this
        # elementwise on the first query of each seed; here it is checked on
        # every query, which is free and covers the other 143.
        z = [a for al, a in tsla if al == 0.0]
        if z:
            d = max(float(np.abs(nll[(z[0], s_)] - nll[(A_BASE, s_)]).max())
                    for s_ in seeds)
            print(f"    identity check: alpha=0 vs {A_BASE}, max per-query "
                  f"|dNLL| over all {qids.shape[1]} queries = {d:.2e}"
                  + ("   [PASS]" if d == 0.0 else
                     "   [FAIL] alpha=0 adds exactly zero, so this must be "
                     "bitwise 0"))
        elif fam.startswith("TV-"):
            # One gate per TV family set: a replacement that is not
            # installed is the same forward whichever layer the arm names,
            # so the receivers run `a=0` once, on the main family's first
            # layer, and it stands for every TV arm of the file.
            print("    gate: shared with the TV family that carries a=0 "
                  "(nothing is installed at a=0, whichever layer)")
        else:
            print("    ⚠ no alpha=0 arm, so 13.5.4(4)'s identity gate did "
                  "not run here")
        # THE COMPARISON 13.5.4 ASKS FOR. Selective memory reads 8 heads'
        # worth of the real demonstrations; TSLA reads one offline vector.
        # Both are measured against the same baseline arm.
        best_al, best_a = min(tsla, key=lambda t: summary[t[1]]["nll"])
        rb = summary[best_a]["nll"] - summary[A_BASE]["nll"]
        print(f"\n    best alpha IN SAMPLE is {best_al:g}, "
              f"{rb:+.6f} nats against {A_BASE}")
        print(f"    selective memory, for comparison:  {rec:+.6f} nats")
        if tot < 0:
            # The fraction only means anything against a NEGATIVE total: if
            # seeing the demonstrations did not help, "fraction of the gain"
            # has no gain to be a fraction of and the ratio flips sign for a
            # reason that has nothing to do with either arm.
            print(f"    as a fraction of what the demonstrations are worth:  "
                  f"{fam if fam in vec_fams else 'TSLA'} {rb / tot:.1%}   "
                  f"selective {rec / tot:.1%}")
        else:
            print("    ⚠ no fraction reported: the total is not negative, so "
                  "there is no gain to take a fraction of")
        print("    ⚠ the best alpha is chosen ON THESE ROWS; the registered "
              "grid is the whole")
        print("      curve above, and a single alpha picked from it is in "
              "sample.")
        (vec_json if fam in vec_fams else tsla_json)[fam] = {
            f"{al:g}": summary[a] for al, a in tsla}

    out = {"spec": "prereg_method_A.md 13.5.2 / 13.5.4",
           "uniform_nll": uniform,
           "accuracy_paired_vs_k0_offset": acc_pair,
           "tsla_alphas": tsla_json,        # {family: {alpha: summary}}
           "tsla_families": sorted(f for f in fams if f not in vec_fams),
           "vector_alphas": vec_json,       # {FV-K10 / TV-K10 L=14: {alpha: summary}}
           "vector_families": sorted(vec_fams),
           "arm_name": meta["arm_name"], "seeds": seeds,
           "n_queries_per_seed": int(qids.shape[1]),
           "by_arm": summary,
           "total_nll_benefit": {"value": tot, "ci95": [lo_t, hi_t]},
           "recovered_nll_benefit": {"value": rec, "ci95": [lo_r, hi_r]},
           "recovered_fraction": frac,
           "per_seed": per_seed,
           "depth_curve": [{"L": L, "arm": a, "nll": summary[a]["nll"],
                            "accuracy": summary[a]["accuracy"],
                            "recovered": summary[a]["nll"]
                            - summary[A_BASE]["nll"],
                            "recovered_per_seed": [
                                summary[a]["per_seed_nll"][i]
                                - summary[A_BASE]["per_seed_nll"][i]
                                for i in range(len(seeds))],
                            "accuracy_per_seed":
                                summary[a]["per_seed_accuracy"]}
                           for L, a in depth],
           "not_licensed": (
               "all-head cached K5 is the 13.5.4 gate, not a method arm; the "
               "arm is not 'zero-shot Method A' (13.5.1); no latency or "
               "compute claim follows from this implementation (13.5.4).")}
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  [output] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
