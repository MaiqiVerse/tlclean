"""FV-on-ICL (adapted): extract one function vector PER PREFIX SEED, inject it.

Section 13.2.3, as corrected by 14.0b-14:

  * for each prefix seed s, take THAT seed's own 144 validation queries under
    THAT seed's own K=5 prefix -- no cycling of seeds across one prompt set,
    which produced a single blended vector and is a different experiment
    (14.0b-7);
  * abar_lh^(s) is head (l,h)'s mean output at the ANSWER ROW, read at the
    o_proj INPUT, which is where the upstream defines it;
  * the CIE uses the first 25 of that seed's validation rows, each rendered
    with the demo DISPLAY LABELS deranged -- query and gold untouched -- and
    scores each head by how much of the gold candidate's FULL-VOCABULARY
    probability it restores when its mean activation is patched in;
  * v_FV^(s) = sum over the CIE top-10 of W_O^(l,h) abar_lh, injected as
    alpha * v at the answer row of decoder layer 9's output.

alpha is GLOBAL: one alpha* chosen by three-seed-mean validation NLL (5(4)).
The vectors are per seed. registry.artifact_seed_faults enforces the second
half of that and would reject one shared vector.

The arithmetic, and the two properties that hold exactly, live in fv_hook and
are tested with no GPU in test_fv.py. Nothing here re-implements them.

COST, STATED BECAUSE IT IS THE REAL CONSTRAINT. The CIE patches ONE head at a
time, so a full sweep is 1024 heads x 25 corrupted prompts x 3 seeds = 76800
patched forwards. Batching packs `--cie-batch` heads OF ONE LAYER into one
forward (13.4's note about doing exactly this), which divides the count but
multiplies the memory: at ~3000 tokens the batch cannot go far. Measure the
first layer's rate before committing to a wall-clock estimate.

⚠ A MUCH CHEAPER EXACT ROUTE EXISTS AND IS NOT IMPLEMENTED HERE. Attention is
causal and the answer row is last, so patching it at layer l changes nothing
at any earlier position: one corrupted forward could be cached and each
(l, h) re-run as a single-token pass through layers l..L with the rest of the
KV reused. That is the standard efficient CIE and it is worth doing -- but it
is a rewrite of the forward loop, and this project has already paid for a
clever unverified one (working rules 3.12). Correct and slow first, with the knob
exposed and the arithmetic printed.

Run:
    sbatch script/lsu1.sh python tools/baselines/run_fv_on_icl.py \\
        --mode validation \\
        --query-manifest results/prereg_method_A_query_manifest.json \\
        --label-space results/label_space_llama31.json \\
        --calibration-dir data/method_a/llama31 \\
        --out results/baselines
    writes results/baselines/baseline_fv_on_icl_validation.npz and .json
    (frozen runs add _seed<S>; the stem is built by BaselineOutput.finish)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.baselines import fv_hook as FV  # noqa: E402
from tools.baselines import registry  # noqa: E402
from tools.baselines.common import (MODES, BaselineOutput,  # noqa: E402
                                    config_arm_name, load_candidate_space,
                                    load_queries, parse_seeds)
from tools.baselines.forward import build_prefixes  # noqa: E402
from tools.label_space import file_sha256  # noqa: E402

NAME = "fv_on_icl"


def per_head_answer_row(model, ids, layers):
    """{layer: [B, n_heads, head_dim]} at the ANSWER ROW, from o_proj INPUT.

    A forward PRE-hook on o_proj, not a hook on the attention module: the
    quantity the upstream defines is the concatenated per-head output just
    before the output projection, and reading it anywhere else measures
    something else that would still have the right shape.
    """
    import torch
    out, handles = {}, []

    def make(l):
        def fn(_mod, args):
            x = args[0]                       # [B, T, n_heads * head_dim]
            out[l] = x[:, -1, :].detach().to(torch.float32).cpu()
        return fn

    for l in layers:
        handles.append(
            model.model.layers[l].self_attn.o_proj
            .register_forward_pre_hook(make(l)))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()
    return out


def answer_logits(model, ids):
    """The last position's logits, float64, [B, V] -- WITHOUT the [B, T, V]
    logits of every position: transformers' `logits_to_keep=1` when the
    model's forward takes it (HF Llama / Qwen), the full call otherwise
    (models/selfExtend). A batch of 8 patched copies of a 3.7k-token prompt
    materialised 8 x 3.7k x 152k float32 logits, 17 GiB, and took the fp32
    Qwen cells down in the exact CIE (jobs 845633 / 845635)."""
    import inspect
    import torch
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        return model(ids, logits_to_keep=1).logits[:, -1, :].to(torch.float64)
    return model(ids).logits[:, -1, :].to(torch.float64)


def patched_logits(model, ids, layer, heads, values, head_dim):
    """Logits with row b's head `heads[b]` replaced by `values[b]`.

    ONE LAYER, MANY HEADS, ONE FORWARD. Every batch row is the same prompt;
    they differ only in which head was overwritten, which is what makes the
    batch dimension a legitimate axis here rather than a way to blur rows.
    """
    import torch
    d = int(head_dim)

    def fn(_mod, args):
        x = args[0].clone()
        for b, h in enumerate(heads):
            x[b, -1, h * d:(h + 1) * d] = values[b].to(x.dtype).to(x.device)
        return (x,) + tuple(args[1:])

    handle = (model.model.layers[layer].self_attn.o_proj
              .register_forward_pre_hook(fn))
    try:
        with torch.no_grad():
            return answer_logits(model, ids)
    finally:
        handle.remove()


def gold_full_prob(logits, token_id):
    """P(gold) under the FULL vocabulary softmax (13.2.3 says full, not
    candidate-conditional; the two are different numbers)."""
    import torch
    return torch.softmax(logits, dim=-1)[..., int(token_id)]


def inject_hook(model, layer, alpha, v):
    """alpha * v added to the answer row of `layer`'s output. Returns a handle.

    alpha = 0 must be bitwise identity, and fv_hook.inject is where that is
    guaranteed and tested; this only carries the tensor into the graph.
    """
    import torch
    vt = torch.as_tensor(np.asarray(v))

    def fn(_mod, _args, output):
        hs = output[0] if isinstance(output, tuple) else output
        add = (alpha * vt).to(hs.dtype).to(hs.device)
        hs = hs.clone()
        hs[:, -1, :] = hs[:, -1, :] + add
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    return model.model.layers[layer].register_forward_hook(fn)


def main(argv=None) -> int:
    spec = registry.BASELINES[NAME]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--label-space", required=True)
    ap.add_argument("--calibration-dir", required=True)
    ap.add_argument("--out", required=True,
                    help="DIRECTORY to write into, not a file name. "
                         "The stem is baseline_fv_on_icl_<mode>"
                         "[_seed<S>] and both .npz and .json are "
                         "written. A path ending in .npz is refused.")
    ap.add_argument("--freeze-manifest")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--task", default="trec_fine_per_class")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--demo-seeds", default="42,43,44")
    ap.add_argument("--cie-prompts", type=int, default=25,
                    help="13.2.3's upstream default. NOT a search parameter")
    ap.add_argument("--cie-batch", type=int, default=8,
                    help="heads of ONE layer per forward. Memory-bound at "
                         "~3000 tokens; raise it only after watching the first "
                         "layer's rate")
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
    print(f"FV-ON-ICL -- {args.mode}, {args.model}")
    print("=" * 78)
    print(f"  alphas {[c['alpha'] for c in configs]} (GLOBAL); the VECTOR is "
          f"per seed, per_seed_artifacts={spec['per_seed_artifacts']}")
    print(f"  edit layer {FV.EDIT_LAYER}, CIE top-{FV.N_CIE_HEADS}, "
          f"{args.cie_prompts} corrupted prompts per seed")

    prefixes, text_of, checks, render_prompt = build_prefixes(
        args.task, args.K, seeds, args.query_manifest, args.calibration_dir,
        ls)
    for c in checks:
        print(f"  [PASS] {c}")

    from tools.probe_prototype_shrinkage import render_demo
    from tools.prereg_task import prefix_demo_rows
    from tools.baselines.forward import _renderers
    manifest_reservation = _renderers()[2]

    from tools.model_loader import load_model
    model = load_model(args.model, method="vanilla",
                       attn_implementation="eager")
    import torch
    n_l = int(model.config.num_hidden_layers)
    n_h = int(model.config.num_attention_heads)
    from tools.icl_common import head_dim
    d_h = head_dim(model)      # not hidden_size // n_h (Qwen3-4B: 128, not 80)
    o_by_layer = {l: model.model.layers[l].self_attn.o_proj.weight.detach()
                  .to(torch.float32).cpu().numpy()
                  for l in range(n_l)}
    print(f"  {n_l} layers x {n_h} heads, head_dim {d_h}; "
          f"{n_l * n_h * args.cie_prompts * len(seeds)} patched forwards at "
          f"batch 1, {args.cie_batch} heads per forward")

    arms = ["natural"] + [config_arm_name(c) for c in configs]
    acc = BaselineOutput(NAME, arms=arms, seeds=seeds,
                         queries_by_seed=rows_by_seed, label_space=ls,
                         mode=args.mode)
    vectors = {}
    placebo_checked = False

    for s in seeds:
        rows = rows_by_seed[s]
        blocks = prefixes[s]
        demos = prefix_demo_rows(args.task, args.K, [s],
                                 excluded_docs=manifest_reservation(
                                     args.query_manifest, demo_seed=s))[s]

        # ---- abar: the mean answer-row per-head output over THIS seed's 144
        print(f"\n  seed {s}: mean activations over {len(rows)} prompts")
        acc_sum = np.zeros((n_l, n_h, d_h), dtype=np.float64)
        for qi, r in enumerate(rows):
            text = text_of.get(r["query_id"])
            if text is None:
                raise SystemExit(
                    f"query {r['query_id'][:12]} is not in the task's splits; "
                    "the manifest and the task have drifted apart")
            ids = tok(render_prompt(blocks, text),
                      return_tensors="pt").input_ids.to(model.device)
            got = per_head_answer_row(model, ids, range(n_l))
            for l in range(n_l):
                acc_sum[l] += got[l][0].numpy().reshape(n_h, d_h)
            if (qi + 1) % 40 == 0:
                print(f"    {qi + 1}/{len(rows)}", flush=True)
        abar = acc_sum / float(len(rows))

        # ---- CIE on corrupted prompts: how much gold probability comes back
        print(f"  seed {s}: CIE over {args.cie_prompts} corrupted prompts")
        cie = np.zeros((n_l, n_h), dtype=np.float64)
        for pi in range(args.cie_prompts):
            r = rows[pi]
            order = FV.corrupted_label_order(len(demos), pi)
            bad_blocks = [render_demo(t, ls.abstract_labels[demos[order[i]][0]])
                          for i, (_c, t) in enumerate(demos)]
            prompt = render_prompt(bad_blocks, text_of[r["query_id"]])
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            gold_tok = int(ls.label_token_ids[int(r["class_idx"])])
            with torch.no_grad():
                base = float(gold_full_prob(
                    answer_logits(model, ids)[0], gold_tok))
            for l in range(n_l):
                for lo in range(0, n_h, args.cie_batch):
                    hs = list(range(lo, min(lo + args.cie_batch, n_h)))
                    vals = [torch.as_tensor(abar[l, h]) for h in hs]
                    lg = patched_logits(model, ids.repeat(len(hs), 1), l, hs,
                                        vals, d_h)
                    p = gold_full_prob(lg, gold_tok).cpu().numpy()
                    cie[l, hs] += (p - base)
            if (pi + 1) % 5 == 0:
                print(f"    {pi + 1}/{args.cie_prompts}", flush=True)
        cie /= float(args.cie_prompts)

        heads = FV.top_cie_heads(cie, k=FV.N_CIE_HEADS)
        v = FV.fv_from_heads(abar, o_by_layer, heads, d_h)
        vectors[s] = {"heads": [[int(a), int(b)] for a, b in heads],
                      "norm": float(np.linalg.norm(v))}
        print(f"  seed {s}: top-{FV.N_CIE_HEADS} {heads}")
        print(f"  seed {s}: |v_FV| = {np.linalg.norm(v):.4f}")

        # ---- score every query at natural and at each alpha
        for qi, r in enumerate(rows):
            ids = tok(render_prompt(blocks, text_of[r["query_id"]]),
                      return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                nat = answer_logits(model, ids)[0]
            nat_c = nat[ls.candidate_token_ids].cpu().numpy()
            acc.add("natural", s, r["query_id"], nat_c,
                    full_logsumexp=float(torch.logsumexp(nat, -1)),
                    full_argmax_token=int(torch.argmax(nat)))
            for c in configs:
                h = inject_hook(model, FV.EDIT_LAYER, float(c["alpha"]), v)
                try:
                    with torch.no_grad():
                        lg = answer_logits(model, ids)[0]
                finally:
                    h.remove()
                lg_c = lg[ls.candidate_token_ids].cpu().numpy()
                # alpha=0 IS the registered placebo, and it is asserted on the
                # cell rather than assumed: adding exact zero changes no bit,
                # so any difference means the hook does something the formula
                # does not say -- and every other alpha is then uninterpretable.
                if float(c["alpha"]) == 0.0 and not np.array_equal(lg_c,
                                                                   nat_c):
                    d = float(np.abs(lg_c - nat_c).max())
                    raise SystemExit(
                        f"seed {s} query {qi}: alpha=0 is not bitwise "
                        f"identical to natural (max |delta| {d:.3e}). The "
                        "registry declares this placebo identity-bitwise.")
                if float(c["alpha"]) == 0.0:
                    placebo_checked = True
                acc.add(config_arm_name(c), s, r["query_id"], lg_c,
                        full_logsumexp=float(torch.logsumexp(lg, -1)),
                        full_argmax_token=int(torch.argmax(lg)))
            if (qi + 1) % 40 == 0:
                print(f"    scored {qi + 1}/{len(rows)}", flush=True)

    # 2.4(4): three vectors, one per prefix seed. One shared vector is a
    # different experiment and this is the check that says so.
    bad = registry.artifact_seed_faults(
        NAME, {"fv_vector": list(vectors)}, list(seeds))
    if bad:
        raise SystemExit("; ".join(bad))
    if not placebo_checked and any(float(c["alpha"]) == 0.0 for c in configs):
        raise SystemExit("the alpha=0 identity check never ran")
    acc.note("task", args.task)
    acc.note("K", args.K)
    acc.note("prefix_checks", checks)
    acc.note("configs", configs)
    acc.note("hook_boundary", spec["hook_boundary"])
    acc.note("fv_vectors", vectors)
    acc.note("edit_layer", FV.EDIT_LAYER)
    acc.note("cie_prompts", args.cie_prompts)
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
