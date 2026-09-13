"""The shared natural readout every H6 baseline is measured against. GPU.

WHY THIS EXISTS AS ITS OWN RUNNER. Two of the six baselines need natural
candidate logits and nothing else:

  * BC is a post-hoc score transform -- it has no model hook at all, so
    without this it cannot run end to end;
  * every model-side runner needs a natural REFERENCE for the section 11.9
    placebo check, and that reference has to be one forward that all of them
    share. Six copies would be six chances for the prompt, the answer position
    or the candidate slice to drift apart, and a placebo comparison against a
    reference that is not quite the same forward proves nothing.

IT MAY NOT READ A METHOD ARTIFACT, INCLUDING METHOD A'S OWN NATURAL RUN.
It would be tempting to reuse whatever natural logits the Method A probe
already produced -- same model, same prompts, same candidates. Section 0.2
forbids it, and not on a technicality: the H6 arms have to be computable
without the method existing, and a shared natural artifact produced by the
method's own code path is a dependency that would be invisible in the results
table. So this runner builds its prompts from the manifest and the calibration
prefix itself, and `common.refuse_quarantined` will not let it open a probe
output even if someone points it at one.

    python tools/baselines/run_natural_readout.py --mode validation \\
        --model meta-llama/Llama-3.1-8B \\
        --query-manifest results/prereg_method_A_query_manifest.json \\
        --label-space results/label_space_llama31.json \\
        --calibration-dir data/method_a/llama31/ \\
        --out results/baselines/
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.baselines import common  # noqa: E402
from tools.baselines.common import (BaselineOutput, base_parser,  # noqa: E402
                                    load_candidate_space, load_queries,
                                    parse_seeds)
from tools.baselines.forward import (NaturalReadout,  # noqa: E402
                                     build_prefixes)
from tools.label_space import (file_sha256,  # noqa: E402
                               tokenizer_provenance)

NAME = "natural_readout"


def main(argv=None) -> int:
    p = base_parser(NAME, __doc__.split("\n")[0])
    p.add_argument("--task", default="trec_fine_per_class")
    p.add_argument("--K", type=int, default=5)
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    common.require_run_args(args)
    if args.mode == "fit":
        raise SystemExit(
            "the natural readout runs in placebo/validation mode or on one "
            "frozen test seed. 'fit' is the phase in which a method builds "
            "its intervention target, and the natural arm has none -- it is "
            "the untouched reference every other arm is compared against.")
    if not args.calibration_dir:
        raise SystemExit(
            "--calibration-dir is required: the prefix this run renders must "
            "be verified against the calibration prompt the published run "
            "actually used, or a one-character drift in the renderer would "
            "pass silently.")

    seeds = parse_seeds(args.demo_seeds, args.mode)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ls = load_candidate_space(args.label_space, model=args.model,
                              query_manifest=args.query_manifest,
                              tokenizer=tok)
    # PER SEED. Each seed draws its own validation (14.0a) and always drew
    # its own test 250, so there is one load per seed and no shared rows list.
    # The previous single load walked ONE list for all three seeds, which
    # since 14.0a would have scored every seed on seed 42's queries while
    # labelling the output rows with each seed in turn -- a file, not an
    # error.
    rows_by_seed = {
        s: load_queries(args.query_manifest, args.mode, demo_seed=s,
                        freeze_manifest=args.freeze_manifest,
                        # the freeze must pin the label space this run reads,
                        # not merely some label space
                        expected_roles={"label_space": args.label_space})
        for s in seeds}

    print("=" * 78)
    print(f"NATURAL READOUT -- {args.mode}, {args.model}")
    print("=" * 78)
    _sizes = {s: len(rows_by_seed[s]) for s in seeds}
    _distinct = len({q["query_id"] for s in seeds for q in rows_by_seed[s]})
    print(f"  queries per seed {_sizes}, {_distinct} distinct across seeds, "
          f"{len(ls.candidate_token_ids)} candidates")

    prefixes, text_of, checks, render_prompt = build_prefixes(
        args.task, args.K, seeds, args.query_manifest, args.calibration_dir, ls)
    for c in checks:
        print(f"  [PASS] {c}")

    from tools.model_args import load_model_from_args
    model = load_model_from_args(args)
    readout = NaturalReadout(model, tok, ls.candidate_token_ids)

    acc = BaselineOutput(NAME, arms=["natural"], seeds=seeds,
                         queries_by_seed=rows_by_seed,
                         label_space=ls, mode=args.mode)
    for s in seeds:
        rows = rows_by_seed[s]
        for i, q in enumerate(rows):
            text = text_of.get(q["query_id"])
            if text is None:
                raise SystemExit(
                    f"query {q['query_id'][:12]} (class {q['class_idx']}) is "
                    "not in the task's train or test split. The manifest and "
                    "the task have drifted apart; do not guess a text.")
            cand, lse, amax = readout(render_prompt(prefixes[s], text))
            acc.add("natural", s, q["query_id"], cand,
                    full_logsumexp=lse, full_argmax_token=amax)
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                print(f"    seed {s}: [{i + 1}/{len(rows)}]")

    acc.note("task", args.task)
    acc.note("K", args.K)
    acc.note("prefix_checks", checks)
    npz, js = acc.finish(args.out, meta={
        "baseline_spec": "shared natural readout (not an H6 arm)",
        "model": args.model, "task": args.task, "K": args.K,
        "dtype": f"torch.{args.dtype}", "attn_implementation": args.attn,
        "query_manifest": str(args.query_manifest),
        "query_manifest_sha256": file_sha256(args.query_manifest),
        "label_space": str(args.label_space),
        "label_space_sha256": file_sha256(args.label_space),
        # from the LIVE tokenizer, not copied out of the label space -- this
        # is the only evidence a zero-GPU analyser can have that the ids were
        # produced by the tokenizer the label space names
        "tokenizer_provenance": tokenizer_provenance(tok, args.model),
        "calibration_dir": str(args.calibration_dir)})
    print(f"\n  [output] {npz}\n  [output] {js}")
    print("  This is the reference every section 11.9 placebo check compares "
          "against, and the input BC calibrates.")
    return 0


def self_test() -> int:
    """CPU fixtures. The forward itself needs a GPU and is gated on the server;
    what is checkable here is the contract around it."""
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
            print(f"  [PASS] {name}  correctly refused: {str(e)[:70]}...")
            return
        except Exception as e:
            check(name, False, f"raised {type(e).__name__}, wanted {want}")
            return
        check(name, False, "SILENTLY SUCCEEDED -- the gate does not bite")

    import numpy as np
    from tools.baselines.forward import candidate_log_probs

    print("the candidate-conditional readout is normalised over CANDIDATES")
    z = np.array([[2.0, 1.0, 0.0, -1.0], [0.5, 0.5, 0.5, 0.5]])
    lp = candidate_log_probs(z)
    check("rows are log-probabilities over the candidate slice",
          np.allclose(np.exp(lp).sum(axis=-1), 1.0))
    check("a uniform row gives log(1/C) exactly",
          abs(lp[1, 0] - np.log(0.25)) < 1e-12,
          "the denominator is the 4 candidates, not the vocabulary")
    check("adding a constant to a row leaves it unchanged (shift invariance)",
          np.allclose(candidate_log_probs(z + 7.0), lp))

    print("\nmode contracts")
    red("--mode fit is refused: the natural arm has no target to fit",
        lambda: main(
            ["--mode", "fit", "--query-manifest", "x", "--label-space", "x",
             "--out", "x", "--calibration-dir", "x"]), SystemExit)
    red("...and the retired name 'discovery' does not even parse",
        lambda: main(
            ["--mode", "discovery", "--query-manifest", "x",
             "--label-space", "x", "--out", "x", "--calibration-dir", "x"]),
        SystemExit)
    red("a run without --calibration-dir is refused (the prefix would be "
        "unverified)", lambda: main(
            ["--mode", "validation", "--query-manifest", "x",
             "--label-space", "x", "--out", "x"]), SystemExit)

    print()
    if not ok:
        print("FIXTURES FAILED.")
        return 2
    print("the natural readout's contract holds; the forward itself is gated "
          "on the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
