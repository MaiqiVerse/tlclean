"""The viability gate's forward pass: six probe runs, one npz. GPU.

WHAT WAS MISSING. method_a_viability READS a viability npz and nothing wrote
one -- `natural_nll` appeared only on the reading side. And the probe takes ONE
`--carrier-json`, a flat {"heads": [...]}, while the gate's array is indexed
e[head, gamma, seed, query] with the head axis meaning "the set selected on
fold h OF THAT SEED". cross_fit evaluates fold f's rows at a[1-f, ...], so
every query has to be scored under BOTH of its seed's head sets.

WHY A DRIVER AND NOT A CHANGE TO THE PROBE. The forward loop already passed
13.4(3)'s GPU gate -- placebo reproducing natural elementwise, per-head writes
leaving the other 31 heads of the layer bitwise unchanged. Teaching it to swap
head sets mid-run would put that behind a new code path for the convenience of
not writing this file. Instead each (seed, fold) cell is an ordinary probe run
against a one-set carrier file extracted from the bundle, and the assembly is
here.

DRIVER AND ASSEMBLY IN ONE FILE, deliberately. The thing that must not go
wrong is the map from (seed, fold) to the head axis index; splitting it across
two tools is precisely the condition that produced 14.0b-12 -- one convention,
two implementations, and no check that could see the difference.

COSTS. 3 seeds x 144 queries x 5 gammas x 2 head sources = 4320 intervened
forwards, plus 432 natural ones (natural writes nothing, so it does not
multiply by gamma or by head set). About 2.1 hours at the 1.6 s/forward
measured in job 835618.

Run (one GPU):
    python tools/run_viability_forward.py \\
        --carrier-bundle results/carriers_fold.json \\
        --query-manifest results/prereg_method_A_query_manifest.json \\
        --label-space results/label_space_llama31.json \\
        --uuid-jsonl data/calibration_trec_fine_per_class_K5_seed42_uuid.jsonl \\
        --out results/method_a_viability_L31.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import carrier_bundle as CB  # noqa: E402
from tools.baselines.forward import candidate_log_probs  # noqa: E402
from tools.discover_carriers import fold_labels  # noqa: E402
from tools.method_a_viability import GAMMA_GRID  # noqa: E402
from tools.prereg_config import (N_VALIDATION,  # noqa: E402
                                 REGISTERED_SEEDS)

CARRIER_IMPL = "gqa_group_v"          # 13.2: the confirmatory L3.1 arm
ARM = "proto_loo"


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scores_from_logits(cand_logits, class_idx, cand_classes):
    """(nll, acc, brier) for one row, candidate-CONDITIONAL (section 1.2).

    The denominator is the candidate set, not the vocabulary. Reusing
    candidate_log_probs rather than writing the log-softmax again is the same
    rule as everywhere else here: one implementation per convention.

    Brier is the multiclass form over the candidate simplex,
    sum_c (p_c - 1[c = gold])^2, which is 1 - 2 p_gold + sum_c p_c^2 and lies
    in [0, 2]. Stated because "Brier" also names the binary p_gold version and
    the two differ by a factor and an offset.
    """
    lp = candidate_log_probs(np.asarray(cand_logits, dtype=np.float64))
    p = np.exp(lp)
    gold = np.flatnonzero(np.asarray(cand_classes) == int(class_idx))
    if gold.size != 1:
        raise SystemExit(
            f"class {class_idx} matches {gold.size} candidates; the label "
            "space must name each eligible class exactly once")
    g = int(gold[0])
    return (-float(lp[g]),
            float(int(np.argmax(lp) == g)),
            float(1.0 - 2.0 * p[g] + float((p * p).sum())))


def one_cell(seed, fold, bundle, args, tmpdir):
    """Run the probe for (seed, fold) and return its npz path."""
    # fold=None means the seed's FULL-validation set: one head source, no
    # cross-fit. The gate cannot read the result and should not.
    key = CB.fold_key(seed, fold) if fold is not None else CB.full_key(seed)
    # THE CUT LIVES IN carrier_bundle, so this runner and run_k0_receiver
    # take it the same way. top_n=None is section 2.4(2)'s frozen set, which
    # is what every registered arm uses.
    heads = CB.heads_for(bundle, seed, args.top_n, key=key)
    cj = Path(tmpdir) / f"carrier_{key}.json"
    cj.write_text(json.dumps({"heads": [[int(l), int(h)] for l, h in heads]}),
                  encoding="utf-8")
    out = Path(args.workdir) / f"viability_{key}.npz"
    if out.is_file() and not args.force:
        print(f"  [skip] {key}: {out} exists (pass --force to redo)")
        return out, cj
    # THIS SEED'S OWN CALIBRATION. The first version passed one --uuid-jsonl
    # for every cell, so the probe could only verify the seed it belonged to
    # and printed "prefix seed 43: unverified (calibration file is for seed
    # 42)" for four of the six cells. The prefix is what every number here
    # depends on and 13.4 wants it bit-identical to the published one, so
    # leaving two thirds of it unchecked is not a detail. All three files
    # exist; the path is derived from the seed.
    from tools.baselines.forward import calibration_path
    cal = calibration_path(args.calibration_dir, args.task, args.K, seed)
    if not Path(cal).is_file():
        raise SystemExit(
            f"{cal} is missing. Each seed's prefix is verified against ITS OWN "
            "calibration prompt; a shared one can only check one of the three.")
    argv = ["--mode", "clean", "--arm", args.arm,
            "--gamma", ",".join(str(g) for g in GAMMA_GRID),
            "--carrier-impl", CARRIER_IMPL, "--carrier-json", str(cj),
            "--demo-seeds", str(seed), "--split", "validation",
            "--query-manifest", args.query_manifest,
            "--label-space", args.label_space,
            "--uuid-jsonl", str(cal),
            "--model", args.model, "--task", args.task, "--K", str(args.K),
            "--allow-kv-group", "--out", str(out)]
    if args.donor_manifest:
        argv += ["--donor-manifest", args.donor_manifest]
    print(f"\n  [cell] seed {seed} fold {fold}: {len(heads)} heads -> {out}")
    from tools.probe_prototype_shrinkage import main as probe_main
    rc = probe_main(argv)
    if rc != 0 or not out.is_file():
        raise SystemExit(f"the probe failed for {key} (rc={rc})")
    return out, cj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--carrier-bundle", required=True)
    # THE ARM IS A PARAMETER, but the DEFAULT is 13.2's confirmatory one so
    # that every existing invocation means exactly what it meant. Anything
    # else is a comparison run and says so in the closing message.
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--top-n", type=int, default=None,
                    help="EXPLORATORY: write at the first N of the bundle's "
                         "ranking instead of section 2.4(2)'s frozen top-8. "
                         "⚠ coverage and DEPTH move together -- the earliest "
                         "carrier is layer 22 at top-8 and 16 at top-16 "
                         "(results/carrier_ranking_profile.json) -- so a "
                         "difference cannot be attributed to head count "
                         "alone. Unlike the K=0 arm, nothing here is "
                         "demo-blind, so the sweep is still readable as "
                         "'how much of the label-row value stream is "
                         "rewritten'")
    ap.add_argument("--donor-manifest",
                    help="EXPLORATORY: frozen donors for --arm "
                         "maxattn_frozen (see tools/exploratory_donor_arms)")
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--label-space", required=True)
    ap.add_argument("--uuid-jsonl", required=True,
                    help="seed 42's file, read for the HEADER only "
                         "(surfaces, token ids, provenance -- all "
                         "seed-independent). Each cell verifies its "
                         "own prefix against its own seed's file, "
                         "taken from --calibration-dir")
    ap.add_argument("--calibration-dir", required=True,
                    help="directory holding all three seeds' "
                         "calibration files; a shared one can only "
                         "verify one of the three prefixes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default="results/viability_cells")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--task", default="trec_fine_per_class")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--carrier-scope", choices=("fold", "full"),
                    default="fold",
                    help="'fold' is the GATE's experiment: six sets, a head "
                         "axis of 2, cross-fitted. 'full' runs the three "
                         "full-validation sets instead -- one head source, no "
                         "cross-fit, gamma chosen in sample. That number is "
                         "OPTIMISTICALLY BIASED and cannot be compared with "
                         "delta_min the way the gate's can; method_a_viability "
                         "refuses it outright because validate_run requires a "
                         "head axis of 2, which is the intended outcome, not "
                         "an obstacle to route around")
    ap.add_argument("--preflight-only", action="store_true",
                    help="check the bundle, the folds and the calibration "
                         "header against the label space, then stop. Zero GPU, "
                         "two seconds, and it is the whole reason not to find "
                         "out 2.1 hours in")
    ap.add_argument("--force", action="store_true",
                    help="redo cells whose npz already exists; without it the "
                         "run resumes, because 2.1 hours is long enough that a "
                         "preemption should not cost all of it")
    args = ap.parse_args(argv)

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    bundle = json.loads(Path(args.carrier_bundle).read_text(encoding="utf-8"))
    qm_sha = sha256_file(args.query_manifest)

    # THE BUNDLE MUST BE THE FOLD ONE, BY NAME. The six fold sets and the
    # three full-validation sets have the same shape, and the full sets are
    # chosen on every query INCLUDING the fold being evaluated -- reading them
    # here would score each fold under heads selected using it, which is the
    # whole thing the cross-fit exists to prevent.
    man = json.loads(Path(args.query_manifest).read_text(encoding="utf-8"))
    vbs = man["validation_by_seed"]
    ids_by_seed = {s: [r["query_id"] for r in vbs[str(s)]]
                   for s in REGISTERED_SEEDS}
    # THE FOLD LABELS ARE RECOMPUTED HERE, from the manifest, with the gate's
    # own function. Passing None makes bundle_faults skip the row check and
    # SAY SO, and an earlier version of this file then filtered that note out
    # -- which would have disabled the one check that can see a bundle cut on
    # the wrong folds. discover_carriers had the same shape of hole: it was
    # bundle_faults' only caller and handed it the labels it had just produced,
    # so discovery was compared against itself (14.0b-12). Recomputing them
    # from the manifest here is what makes this an independent check.
    cls_of = {s: {r["query_id"]: int(r["class_idx"]) for r in vbs[str(s)]}
              for s in REGISTERED_SEEDS}
    folds_by_seed = {s: fold_labels(ids_by_seed[s], cls_of[s])
                     for s in REGISTERED_SEEDS}
    want_scope = (CB.SCOPE_FOLD if args.carrier_scope == "fold"
                  else CB.SCOPE_FULL)
    bad = CB.bundle_faults(bundle, scope=want_scope,
                           validation_by_seed=ids_by_seed,
                           fold_by_seed=folds_by_seed,
                           query_manifest_sha256=qm_sha, model=args.model)
    if bad:
        raise SystemExit("the carrier bundle may not be used here:\n  - "
                         + "\n  - ".join(bad))

    print("=" * 78)
    print("VIABILITY FORWARD -- section 13.6, VALIDATION ONLY")
    print("=" * 78)
    print(f"  gammas {GAMMA_GRID}, seeds {REGISTERED_SEEDS}, "
          f"{N_VALIDATION} queries/seed")
    # THE COUNT IS THE ONE THIS RUN MAKES. It read
    # seeds * N * len(GAMMA_GRID) * 2 -- the *2 is the fold scope's two head
    # sources, which a full-scope run does not have, and gamma=0 reuses the
    # natural forward rather than re-running it. On a full run that printed
    # 4320 where 1728 happen.
    _heads_sources = 2 if args.carrier_scope == "fold" else 1
    _n_int = (len(REGISTERED_SEEDS) * N_VALIDATION
              * (len(GAMMA_GRID) - 1) * _heads_sources)
    print(f"  {_n_int} "
          "intervened forwards + "
          f"{len(REGISTERED_SEEDS) * N_VALIDATION} natural")
    print("  no test query is reachable: --split validation, and load_split "
          "refuses the test splits without a freeze manifest (2.1)")

    # PRE-FLIGHT, BEFORE ANY MODEL IS LOADED. assert_header_matches already
    # exists and already treats a missing field as a failure rather than a
    # skipped check; the probe calls it, but only after the weights are in.
    # For a 2.1-hour job that is the wrong end. It also closes the gap the
    # discovery run reported and could not close: that run printed
    # "[no --label-space] header manifest hash None", because this header
    # predates the manifest-hash field -- but the surfaces and token ids can
    # be compared against the label space directly, which is the thing that
    # actually decides whether every head is projected onto the right row.
    from tools.baselines.forward import load_calibration_header
    from tools.icl_common import run_provenance
    from tools.label_space import assert_header_matches
    header, _first = load_calibration_header(args.uuid_jsonl)
    ls = json.loads(Path(args.label_space).read_text(encoding="utf-8"))
    assert_header_matches(ls, header, path=args.uuid_jsonl, model=args.model)
    print(f"\n  [PASS] calibration header matches the label space: "
          f"{len(ls['label_token_ids'])} classes, model "
          f"{ls['provenance']['model']}")
    print(f"         {args.uuid_jsonl}")
    if args.preflight_only:
        print("\n  --preflight-only: nothing was run. Everything the forward "
              "needs is present and consistent.")
        return 0

    cells = {}
    with tempfile.TemporaryDirectory() as td:
        folds = (0, 1) if args.carrier_scope == "fold" else (None,)
        for s in REGISTERED_SEEDS:
            for f in folds:
                cells[(s, f)], _ = one_cell(s, f, bundle, args, td)

    # ---------------------------------------------------------------- assemble
    ns, nq, ng = len(REGISTERED_SEEDS), N_VALIDATION, len(GAMMA_GRID)
    nh = 2 if args.carrier_scope == "fold" else 1
    nll = np.full((nh, ng, ns, nq), np.nan)
    acc = np.full((nh, ng, ns, nq), np.nan)
    bri = np.full((nh, ng, ns, nq), np.nan)
    nat_nll = np.full((ns, nq), np.nan)
    nat_acc = np.full((ns, nq), np.nan)
    nat_bri = np.full((ns, nq), np.nan)
    qids = np.empty((ns, nq), dtype=object)
    cls = np.full((ns, nq), -1, dtype=np.int64)
    meta0 = None

    print("\nassembling; the head axis is the FOLD the set was selected on")
    for si, s in enumerate(REGISTERED_SEEDS):
        for hi, f in enumerate((0, 1) if args.carrier_scope == "fold"
                               else (None,)):
            z = np.load(cells[(s, f)], allow_pickle=False)
            m = json.loads(str(z["meta"]))
            meta0 = meta0 or m
            rows = [str(q) for q in z["query_id"]]
            if rows != ids_by_seed[s]:
                raise SystemExit(
                    f"seed {s} fold {f}: the probe returned {len(rows)} rows "
                    f"that are not this seed's validation draw in manifest "
                    "order. Every array here is positional.")
            cand_cls = np.asarray(z["candidate_classes"])
            gs = tuple(float(g) for g in z["gammas"])
            if gs != GAMMA_GRID:
                raise SystemExit(f"seed {s} fold {f}: gammas {gs} != "
                                 f"{GAMMA_GRID}")
            for qi, cidx in enumerate(int(c) for c in z["class_idx"]):
                qids[si, qi] = rows[qi]
                cls[si, qi] = cidx
                n, a, b = scores_from_logits(z["logits_natural"][qi], cidx,
                                             cand_cls)
                # NATURAL IS HEAD-INDEPENDENT, and that is an exact check
                # rather than a comment: the two head-source runs of one seed
                # write nothing on the natural path, so they must produce the
                # same number. A difference means the arm leaked into it.
                if hi == 1 and not np.isnan(nat_nll[si, qi]):
                    if abs(nat_nll[si, qi] - n) > 0:
                        raise SystemExit(
                            f"seed {s} query {qi}: natural NLL differs between "
                            f"the two head-source runs ({nat_nll[si, qi]!r} vs "
                            f"{n!r}). Natural writes nothing, so this is not a "
                            "tolerance question -- the intervention reached the "
                            "unintervened forward.")
                nat_nll[si, qi], nat_acc[si, qi], nat_bri[si, qi] = n, a, b
                for gi in range(ng):
                    n2, a2, b2 = scores_from_logits(
                        z["logits_arm"][qi][gi], cidx, cand_cls)
                    nll[hi, gi, si, qi] = n2
                    acc[hi, gi, si, qi] = a2
                    bri[hi, gi, si, qi] = b2
            print(f"  seed {s} "
                  f"{'fold ' + str(f) if f is not None else 'FULL set'}"
                  f": {nq} rows x {ng} gammas")

    for name, arr in (("nll", nll), ("acc", acc), ("brier", bri),
                      ("natural_nll", nat_nll)):
        if not np.all(np.isfinite(arr)):
            raise SystemExit(f"{name} has {int((~np.isfinite(arr)).sum())} "
                             "cells with no value")

    np.savez_compressed(
        args.out, nll=nll, acc=acc, brier=bri,
        natural_nll=nat_nll, natural_acc=nat_acc, natural_brier=nat_bri,
        query_ids=qids.astype(str), class_idx=cls,
        gammas=np.array(GAMMA_GRID, dtype=np.float64),
        seeds=np.array(REGISTERED_SEEDS, dtype=np.int64))
    meta = {
        "model": args.model, "task": args.task, "K": args.K,
        "dtype": (meta0 or {}).get("dtype", "torch.bfloat16"),
        "attn_implementation": (meta0 or {}).get("attn_implementation",
                                                 "eager"),
        "arm": args.arm, "top_n": args.top_n,
        "registered_carrier_set": args.top_n is None,
        "carrier_impl": CARRIER_IMPL,
        # THE BUNDLE, not one extracted set: it is the artifact that pins all
        # six, and load_viability_npz compares this against the --carrier-json
        # the gate is run with.
        "carrier_sha256": sha256_file(args.carrier_bundle),
        "query_manifest_sha256": qm_sha,
        "label_space_sha256": sha256_file(args.label_space),
        "tokenizer_provenance": (meta0 or {}).get("tokenizer_provenance"),
        "spec": "prereg_method_A.md section 13.6",
        # working rules 1.1: the job id has to live in the ARTIFACT. The
        # log carries it only until archiving renames the file, and
        # the ten five-arm top-16/32 runs lost theirs exactly that way
        # because this line was missing.
        "provenance": run_provenance(),
        "carrier_scope": args.carrier_scope,
        "head_axis": ("index h = the set selected on fold h OF THAT SEED "
                      "(13.6.3); cross_fit evaluates fold f at a[1-f]"
                      if args.carrier_scope == "fold" else
                      "ONE head source: this seed's FULL-validation set. No "
                      "cross-fit is possible and none is claimed -- gamma is "
                      "chosen in sample on the same 144 rows the carriers "
                      "were, so the effect is optimistically biased twice "
                      "over and is NOT comparable with delta_min. The gate "
                      "refuses this file."),
        "cells": {f"s{s}_f{f}": str(p) for (s, f), p in cells.items()},
    }
    Path(str(args.out).replace(".npz", ".json")).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n  [output] {args.out}  (sha256 {sha256_file(args.out)[:16]}...)")
    print(f"  [output] {str(args.out).replace('.npz', '.json')}")
    # THE CLOSING LINE DEPENDS ON THE SCOPE, and it used to not. A full-scope
    # npz has a head axis of 1, so method_a_viability refuses it -- pointing a
    # reader at a command that will fail is bad enough, but pointing them at
    # the GATE for a number the gate must not read is worse.
    if args.carrier_scope == "fold":
        print("\nNothing is decided yet. Run tools/method_a_viability.py on "
              "this npz for the verdict;")
        print("its criteria were written before any of these numbers existed "
              "(13.6).")
    else:
        print("\nThis is NOT a gate input. method_a_viability refuses it -- "
              "validate_run requires a")
        print("head axis of 2 and this has 1, because a full-validation set "
              "cannot be cross-fitted.")
        print("The number here is optimistically biased twice over: gamma is "
              "chosen in sample, and")
        print("the carriers were selected on the same 144 rows it is measured "
              "on. Read it with")
        print("tools/analyze_full_validation.py, which says so alongside "
              "every figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
