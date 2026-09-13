"""TSLA-TL-on-ICL (adapted): score heads in the label subspace, steer with them.

Section 13.2.4, as corrected the same way 14.0b-14 corrected 13.2.3: PER
PREFIX SEED, from that seed's own validation, no cycling and no single shared
vector.

  * o_qlh is the head's answer-row output IN RESIDUAL SPACE -- its o_proj
    input slice carried through its own W_O block. Not the 128-dimensional
    slice itself: 13.2.4 contracts o against unembedding rows W in R^(36 x d),
    so the two are not interchangeable and only one of them has the right
    length;
  * each head scores mean_c <o, W_y - W_c> / ||o P||, P the label subspace
    projector, over that seed's first 50 validation rows;
  * the top floor(0.03 * 32 * 32) = 30 heads, ties to the smaller (layer,
    head);
  * v_TSLA^(s) = mean over those 50 prompts of the SUM of the selected heads'
    residual-space outputs, injected as alpha * v at the answer row of decoder
    layer 16's output.

The injection operator is fv_hook.inject and the runner's hook is FV's --
identical operator, different layer and a differently built vector. Writing it
twice is what the ZeroTuning rescale did before it was collapsed to one.

TWO PASSES OVER THE 50 PROMPTS, deliberately. Head selection needs the scores
from all 50 before the vector's sum can be taken over the selected heads, and
holding fifty [32, 32, 4096] arrays to avoid a second pass costs a gigabyte to
save 50 forwards per seed. The second pass re-renders the same prompts from
the same prefix, so it is the same quantity, not an approximation of it.

Run:
    sbatch script/lsu1.sh python tools/baselines/run_tsla_tl_on_icl.py \\
        --mode validation \\
        --query-manifest results/prereg_method_A_query_manifest.json \\
        --label-space results/label_space_llama31.json \\
        --calibration-dir data/method_a/llama31 \\
        --out results/baselines
    writes results/baselines/baseline_tsla_tl_on_icl_validation.npz and .json
    (frozen runs add _seed<S>; the stem is built by BaselineOutput.finish)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.baselines import fv_hook as FV  # noqa: E402
from tools.baselines import registry  # noqa: E402
from tools.baselines import tsla_hook as TS  # noqa: E402
from tools.baselines.common import (MODES, BaselineOutput,  # noqa: E402
                                    config_arm_name, load_candidate_space,
                                    load_queries, parse_seeds)
from tools.baselines.forward import build_prefixes  # noqa: E402
from tools.model_args import add_model_arguments, load_model_from_args  # noqa: E402
from tools.baselines.run_fv_on_icl import (inject_hook,  # noqa: E402
                                           per_head_answer_row)
from tools.label_space import file_sha256  # noqa: E402

NAME = "tsla_tl_on_icl"
N_DISCOVERY_PROMPTS = 50          # 13.2.4, not a search parameter


def residual_space_outputs(model, ids, o_by_layer, n_l, n_h, d_h):
    """[layer, head, d]: each head's answer-row contribution to the residual.

    THE UNITS ARE THE POINT. per_head_answer_row returns the o_proj INPUT,
    which is [n_heads * head_dim] and lives in head space. 13.2.4 contracts
    against unembedding rows, so the quantity it means is the head's output
    AFTER its own W_O block -- same vector, different space, and only the
    second one can be dotted with W_y.
    """
    got = per_head_answer_row(model, ids, range(n_l))
    out = np.empty((n_l, n_h, o_by_layer[0].shape[0]), dtype=np.float64)
    for l in range(n_l):
        a = got[l][0].numpy().reshape(n_h, d_h)
        for h in range(n_h):
            out[l, h] = FV.head_slice(o_by_layer[l], h, d_h) @ a[h]
    return out


def vectors_sidecar(vectors, discovery_ids, *, K, mode, model, task,
                    edit_layer, n_discovery_prompts, label_subspace,
                    query_manifest, label_space, extra_classes=None):
    """The sidecar a --discovery-only run writes, in the FULL run's layout.

    run_k0_receiver.find_tsla_vectors looks under ["runner"]["tsla_vectors"]
    first, so a vectors-only file that put them anywhere else would be
    refused by the very consumer this mode exists for. The discovery rows
    are recorded beside the vectors: run_k10_increment's increment family
    subtracts the same heads' output on ITS receiver, averaged over the same
    prompts, and cannot check "same prompts" unless they are on the record.
    """
    from tools.icl_common import run_provenance
    extra = {TS.sidecar_key(c): v for c, v in (extra_classes or {}).items()}
    return {
        "runner": {
            "tsla_vectors": vectors,
            **extra,
            "tsla_classes": ["tl"] + sorted(extra_classes or {}),
            "head_class_note": "tsla_vectors = TL heads (upstream margin_add); "
                               "tsla_vectors_tr = TR heads (upstream "
                               "cossim_norm = ||oP||); tsla_vectors_random = "
                               "a seeded random draw of the same size",
            "tsla_discovery_query_ids": {str(s): [str(q) for q in ids]
                                         for s, ids in discovery_ids.items()},
            "K": int(K), "edit_layer": int(edit_layer),
            "n_discovery_prompts": int(n_discovery_prompts),
            "label_subspace": label_subspace,
            "note": "vectors only: no arm was scored, so there is no npz. "
                    "Consumers: run_k0_receiver --tsla-vectors, "
                    "run_k10_increment --tsla-vectors"},
        "meta": {"baseline": NAME, "discovery_only": True, "mode": mode,
                 "model": model, "task": task, "K": int(K),
                 "query_manifest": str(query_manifest),
                 "query_manifest_sha256": file_sha256(query_manifest),
                 "label_space": str(label_space),
                 "label_space_sha256": file_sha256(label_space),
                 "provenance": run_provenance()}}


def main(argv=None) -> int:
    spec = registry.BASELINES[NAME]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--label-space", required=True)
    ap.add_argument("--calibration-dir", required=True)
    ap.add_argument("--out", required=True,
                    help="DIRECTORY to write into, not a file name. "
                         "The stem is baseline_tsla_tl_on_icl_<mode>"
                         "[_seed<S>] and both .npz and .json are "
                         "written. A path ending in .npz is refused.")
    ap.add_argument("--freeze-manifest")
    add_model_arguments(ap)
    ap.add_argument("--task", default="trec_fine_per_class")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--demo-seeds", default="42,43,44")
    ap.add_argument("--discovery-only", action="store_true",
                    help="build and record the per-seed vectors and score NO "
                         "arm. Writes only the vectors sidecar (see "
                         "--vectors-out), which run_k0_receiver and "
                         "run_k10_increment read through --tsla-vectors. "
                         "Use with --K 10 for the K=10-increment families")
    ap.add_argument("--vectors-out", default=None,
                    help="the sidecar's path in --discovery-only mode "
                         "(default <out>/tsla_vectors_K<K>_<mode>.json)")
    args = ap.parse_args(argv)

    configs = list(spec["configs"])
    seeds = parse_seeds(args.demo_seeds, args.mode)
    if args.mode == "placebo":
        configs = [dict(spec["placebo"])]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ls = load_candidate_space(args.label_space, model=args.model,
                              query_manifest=args.query_manifest,
                              tokenizer=tok)
    rows_by_seed = {
        s: load_queries(args.query_manifest, args.mode, demo_seed=s,
                        freeze_manifest=args.freeze_manifest,
                        expected_roles={"label_space": args.label_space})
        for s in seeds}

    print("=" * 78)
    print(f"TSLA-TL-ON-ICL -- {args.mode}, {args.model}")
    print("=" * 78)
    print(f"  alphas {[c['alpha'] for c in configs]} (GLOBAL); the VECTOR is "
          f"per seed, per_seed_artifacts={spec['per_seed_artifacts']}")

    prefixes, text_of, checks, render_prompt = build_prefixes(
        args.task, args.K, seeds, args.query_manifest, args.calibration_dir,
        ls)
    for c in checks:
        print(f"  [PASS] {c}")

    model = load_model_from_args(args)
    import torch
    n_l = int(model.config.num_hidden_layers)
    n_h = int(model.config.num_attention_heads)
    edit = TS.edit_layer(n_l)    # the upstream's L // 2: 16 on Llama, 14 / 18 on Qwen
    from tools.icl_common import head_dim
    d_h = head_dim(model)      # not hidden_size // n_h (icl_common.head_dim)
    o_by_layer = {l: model.model.layers[l].self_attn.o_proj.weight.detach()
                  .to(torch.float32).cpu().numpy().astype(np.float64)
                  for l in range(n_l)}

    # ---- the label subspace, once: it depends only on the frozen candidates
    W = (model.lm_head.weight.detach()[ls.candidate_token_ids]
         .to(torch.float32).cpu().numpy().astype(np.float64))
    P, diag = TS.label_projector(W)
    bad = TS.projector_faults(P, W)
    if bad:
        raise SystemExit("the label subspace projector is not one:\n  - "
                         + "\n  - ".join(bad))
    k = TS.n_top_heads(n_l, n_h)
    print(f"  label subspace: {W.shape[0]} candidates in {W.shape[1]} dims, "
          f"rank {diag['rank']}")
    print(f"    smallest non-zero singular value "
          f"{diag['smallest_nonzero_singular_value']:.6e}, "
          f"condition number {diag['condition_number']:.6e}")
    print(f"    pinv rule: {diag['rule']}  (rtol {diag['rtol']:.3e})")
    print(f"  [PASS] P is symmetric, idempotent and fixes every label row")
    print(f"  top {k} heads = floor({TS.TOP_FRACTION} x {n_l} x {n_h}); "
          f"edit layer {edit}")

    arms = ["natural"] + [config_arm_name(c) for c in configs]
    acc = BaselineOutput(NAME, arms=arms, seeds=seeds,
                         queries_by_seed=rows_by_seed, label_space=ls,
                         mode=args.mode)
    vectors, discovery_ids, placebo_checked = {}, {}, False
    vectors_by_class = {c: {} for c in TS.HEAD_CLASSES if c != "tl"}
    cls_pos = {int(c): i for i, c in enumerate(ls.eligible_classes)}

    for s in seeds:
        rows = rows_by_seed[s]
        blocks = prefixes[s]
        disc = rows[:N_DISCOVERY_PROMPTS]

        def prompt_ids(r):
            text = text_of.get(r["query_id"])
            if text is None:
                raise SystemExit(
                    f"query {r['query_id'][:12]} is not in the task's splits; "
                    "the manifest and the task have drifted apart")
            return tok(render_prompt(blocks, text),
                       return_tensors="pt").input_ids.to(model.device)

        # ---- pass 1: score every head on this seed's first 50
        print(f"\n  seed {s}: scoring heads on {len(disc)} prompts")
        tot = np.zeros((n_l, n_h), dtype=np.float64)
        tot_tr = np.zeros((n_l, n_h), dtype=np.float64)
        for pi, r in enumerate(disc):
            O = residual_space_outputs(model, prompt_ids(r), o_by_layer,
                                       n_l, n_h, d_h)
            # THE GOLD INDEX IS A POSITION IN THE CANDIDATE BLOCK, not a class
            # id. W's rows are the 36 candidates in eligible_classes order, so
            # passing class_idx directly would index the wrong row whenever the
            # eligible ids are not 0..35 -- and they are scattered over 0..49
            # (working rules 3.10).
            tot += TS.head_scores(O, W, cls_pos[int(r["class_idx"])], P)
            tot_tr += TS.tr_scores(O, P)
            if (pi + 1) % 10 == 0:
                print(f"    {pi + 1}/{len(disc)}", flush=True)
        heads = TS.top_heads(tot / float(len(disc)), k)
        # THE OTHER TWO CLASSES, from the same pass: TR by ||oP|| and a
        # seeded random draw (the upstream's control). Same k, same prompts.
        heads_by = {"tl": heads,
                    "tr": TS.top_heads(tot_tr / float(len(disc)), k),
                    "random": TS.random_heads(n_l, n_h, k, s)}

        # ---- pass 2: the vector, summed within a prompt then meaned across
        print(f"  seed {s}: building the vector from {k} heads")
        per_prompt = {c: np.empty((len(disc), W.shape[1]), dtype=np.float64)
                      for c in heads_by}
        for pi, r in enumerate(disc):
            O = residual_space_outputs(model, prompt_ids(r), o_by_layer,
                                       n_l, n_h, d_h)
            for c, hh in heads_by.items():
                per_prompt[c][pi] = sum(O[l, h] for l, h in hh)
        v_by = {c: TS.steering_vector(per_prompt[c]) for c in heads_by}
        v = v_by["tl"]
        for c in vectors_by_class:
            vectors_by_class[c][s] = {
                "heads": [[int(a), int(b)] for a, b in heads_by[c]],
                "norm": float(np.linalg.norm(v_by[c])),
                "v": [float(x) for x in v_by[c]]}
        # THE VECTOR ITSELF, not just its norm. Recording only `norm` left
        # the artifact unable to reproduce its own injection: re-running the
        # same alphas, or reusing v for 13.5.4's `TSLA-TL-zero-demo` arm,
        # meant redoing both discovery passes. A steering vector IS the
        # frozen object this baseline is, so it belongs on the record --
        # 4096 float64 per seed.
        vectors[s] = {"heads": [[int(a), int(b)] for a, b in heads],
                      "norm": float(np.linalg.norm(v)),
                      "v": [float(x) for x in v]}
        print(f"  seed {s}: top-{k} first five {heads[:5]}")
        print(f"  seed {s}: |v_TSLA| = {np.linalg.norm(v):.4f}")
        print(f"  seed {s}: TR top-{k} first five {heads_by['tr'][:5]}, "
              f"|v_TR| = {np.linalg.norm(v_by['tr']):.4f}; random "
              f"|v_rand| = {np.linalg.norm(v_by['random']):.4f}; TL/TR "
              f"overlap {len(set(heads) & set(heads_by['tr']))}/{k}")
        discovery_ids[s] = [r["query_id"] for r in disc]
        if args.discovery_only:
            continue

        # ---- score every query at natural and at each alpha
        for qi, r in enumerate(rows):
            ids = prompt_ids(r)
            with torch.no_grad():
                nat = model(ids).logits[0, -1].to(torch.float64)
            nat_c = nat[ls.candidate_token_ids].cpu().numpy()
            acc.add("natural", s, r["query_id"], nat_c,
                    full_logsumexp=float(torch.logsumexp(nat, -1)),
                    full_argmax_token=int(torch.argmax(nat)))
            for c in configs:
                h = inject_hook(model, edit, float(c["alpha"]), v)
                try:
                    with torch.no_grad():
                        lg = model(ids).logits[0, -1].to(torch.float64)
                finally:
                    h.remove()
                lg_c = lg[ls.candidate_token_ids].cpu().numpy()
                if float(c["alpha"]) == 0.0:
                    if not np.array_equal(lg_c, nat_c):
                        d = float(np.abs(lg_c - nat_c).max())
                        raise SystemExit(
                            f"seed {s} query {qi}: alpha=0 is not bitwise "
                            f"identical to natural (max |delta| {d:.3e}). The "
                            "registry declares this placebo identity-bitwise; "
                            "a hook that perturbs at 0 makes every other alpha "
                            "uninterpretable.")
                    placebo_checked = True
                acc.add(config_arm_name(c), s, r["query_id"], lg_c,
                        full_logsumexp=float(torch.logsumexp(lg, -1)),
                        full_argmax_token=int(torch.argmax(lg)))
            if (qi + 1) % 40 == 0:
                print(f"    scored {qi + 1}/{len(rows)}", flush=True)

    bad = registry.artifact_seed_faults(
        NAME, {"tsla_vector": list(vectors)}, list(seeds))
    if bad:
        raise SystemExit("; ".join(bad))
    if args.discovery_only:
        # No arm was scored, so BaselineOutput.finish -- which refuses a
        # dense array with unwritten cells -- is not the writer here. The
        # sidecar alone, in the layout the full run's sidecar has.
        out_json = (Path(args.vectors_out) if args.vectors_out
                    else Path(args.out) / f"tsla_vectors_K{args.K}_{args.mode}.json")
        if out_json.suffix != ".json":
            raise SystemExit(f"--vectors-out {out_json}: the sidecar is a "
                             ".json file")
        doc = vectors_sidecar(
            vectors, discovery_ids, K=args.K, mode=args.mode,
            model=args.model, task=args.task, edit_layer=edit,
            n_discovery_prompts=N_DISCOVERY_PROMPTS, label_subspace=diag,
            query_manifest=args.query_manifest, label_space=args.label_space,
            extra_classes=vectors_by_class)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"\n  [output] {out_json}   (vectors only; no arm was scored, "
              "no npz)")
        return 0
    if not placebo_checked and any(float(c["alpha"]) == 0.0 for c in configs):
        raise SystemExit("the alpha=0 identity check never ran")
    acc.note("task", args.task)
    acc.note("K", args.K)
    acc.note("prefix_checks", checks)
    acc.note("configs", configs)
    acc.note("hook_boundary", spec["hook_boundary"])
    acc.note("tsla_vectors", vectors)
    for c, vc in vectors_by_class.items():
        acc.note(TS.sidecar_key(c), vc)
    acc.note("tsla_discovery_query_ids", {str(s): ids for s, ids
                                          in discovery_ids.items()})
    acc.note("label_subspace", diag)
    acc.note("edit_layer", edit)
    acc.note("method", args.method)
    acc.note("n_discovery_prompts", N_DISCOVERY_PROMPTS)
    npz, js = acc.finish(args.out, meta={
        "baseline_spec": spec["arm_name"],
        "model": args.model, "task": args.task, "K": args.K,
        "dtype": "torch.bfloat16", "attn_implementation": "eager",
        "query_manifest": str(args.query_manifest),
        "query_manifest_sha256": file_sha256(args.query_manifest),
        "label_space": str(args.label_space),
        "label_space_sha256": file_sha256(args.label_space),
        "config_axis": spec["config_axis"],
        "selection": spec["selection"],
        "per_seed_artifacts": list(spec["per_seed_artifacts"]),
    })
    print(f"\n  [output] {npz}")
    print(f"  [output] {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
