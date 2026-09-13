"""Is the carriers' write eaten downstream, or amplified? GPU. EXPLORATORY.

THE QUESTION. RESULTS 51's selective arm moves the final candidate logits by
some amount. The eight carrier heads are where the intervention acts, and what
they write reaches the readout by two routes: the DIRECT one, their own output
projected onto the unembedding rows, and everything else -- the layers above
them, the final normalisation. If the direct write moves by more than the final
logits do, the rest of the network is absorbing it; if by less, amplifying.

    absorbed = 1 - Delta_final / Delta_direct

`probe_tl_class_attribution` already dumps the same projection, but on ONE
condition. A response is a difference between two, so the direct write is
measured again under the two masks RESULTS 51 compares. Delta_final is already
on disk in that section's npz; this run supplies the missing half.

⚠⚠ THIS RUNS ON THE CACHED PATH, AND THAT IS NOT AN OPTIMISATION -- IT IS
WHAT MAKES IT THE SAME COMPUTATION. RESULTS 51 prefills C_s || D_s with NO
mask, so the demonstrations attend to each other normally, and applies the
visibility mask only while the receiver's own tokens are being run. A single
uncached forward with the mask over every query position would also block
demonstration-to-demonstration attention, and would be measuring the direct
write of a different computation.

The first version of this file did exactly that, and never got far enough to
be wrong: with the query axis spanning the whole 6100-token sequence the masks
are 32 layers x [1, 32, 6100, 6100] float64 = about 305 GB, and slurm killed
it. The cached path's query axis is the receiver's own ~20 tokens.

⚠ THE PER-HEAD PROJECTION IS THE PROJECT'S, NOT A NEW ONE. Section 13's
convention: take the o_proj INPUT, slice it by query head, and project through
that head's o_proj columns and then the unembedding --

    contribution(l, k, c) = weighted_v[l, k] @ (O_k^T W_U[c])

with weighted_v the o_proj input at the answer row, float32, no final
layernorm and no centring. That is term for term what
`diagnostic_forward.extract_per_head_logit_contributions` computes, and
working rules 3.12(3) says a faster path replacing a reference one must carry an
equivalence check against it.

⚠ THAT CHECK ASKS ONE QUESTION AND REPORTS THE OTHER. On the first query of
every seed, this path and the reference are run on the SAME uncached unmasked
forward and must agree to float32 round-off -- that is about the EXTRACTION,
and it is gated. How much the cached path then differs from that forward is a
property of the cache, is reported, and is NOT gated. The first version fused
the two and set the threshold in ulps of the per-head contribution, a quantity
twelve times smaller than the final logits the cache noise was established on;
it fired at 3.80 ulp on values of magnitude 1.6, which is 0.24 ulp at the
magnitude RESULTS 45 measured -- a number that said nothing about either
question.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import carrier_bundle as CB  # noqa: E402
from tools.baselines.common import load_candidate_space  # noqa: E402
from tools.model_args import add_model_arguments, load_model_from_args  # noqa: E402
from tools.baselines.forward import (build_prefixes,  # noqa: E402
                                     calibration_path,
                                     load_calibration_header)
from tools.check_demo_nesting import nesting_faults  # noqa: E402
from tools.icl_common import head_dim, run_provenance  # noqa: E402
from tools.label_space import file_sha256  # noqa: E402
from tools.prereg_config import REGISTERED_SEEDS  # noqa: E402
from tools.run_k0_receiver import _install_rows, prepare_mask_rows  # noqa: E402
from tools.run_k10_increment import (ARM_ALL, ARM_BASE,  # noqa: E402
                                     ARM_SEL, ROLES, arm_names,
                                     base_columns, masks_for)
from tools.split_rows import (add_split_arguments, rows_for,  # noqa: E402
                              split_meta)

ARMS = (ARM_BASE, ARM_SEL, ARM_ALL)


def head_projections(model, carriers, cand, torch):
    """{(l, k): [d_head, n_cand]} = O_k^T W_U[cand], float32 on CPU.

    A function of the WEIGHTS alone, so it is built once and never enters the
    query loop -- the mistake working rules 3.12 records from section 27, where a
    prompt-independent matmul was recomputed 256k times and the GPU sampled at
    0 % utilisation.
    """
    w_u = model.lm_head.weight.detach()
    tgt = w_u[torch.as_tensor(list(cand), dtype=torch.long)]
    tgt = tgt.to(torch.float32).cpu()                    # [n_cand, d_model]
    d_model = int(model.config.hidden_size)
    # icl_common.head_dim, NOT hidden_size // n_heads: Qwen3-4B has hidden
    # 2560, 32 heads and head_dim 128, so the quotient (80) sliced o_proj's
    # input at the wrong columns and the extraction gate refused every
    # candidate on the Q34c36 test read (job 844680, RESULTS 63.9).
    d_head = head_dim(model)
    out = {}
    for l, heads in carriers.items():
        w_o = model.model.layers[l].self_attn.o_proj.weight.detach()
        for k in heads:
            o_k = w_o[:, k * d_head:(k + 1) * d_head].to(torch.float32).cpu()
            out[(l, k)] = o_k.T @ tgt.T                  # [d_head, n_cand]
    return out, d_head


def capture_o_proj(model, layers, store, torch):
    """Pre-hooks on o_proj that keep its input at the ANSWER row.

    The o_proj input is the concatenation of the query heads' outputs, which
    is exactly the `weighted_v` the reference primitive forms as alpha @ V.
    Taking it here rather than recomputing attention is what section 13's
    convention prescribes for per-head readouts.
    """
    handles = []
    for l in layers:
        def fn(_mod, args, _l=l):
            x = args[0]
            store[_l] = x[0, -1, :].detach().to(torch.float32).cpu()
            return None
        handles.append(
            model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(fn))
    return handles


def direct_from_store(store, proj, carriers, d_head, n_cand):
    """[n_cand]: the carriers' summed direct contribution."""
    v = np.zeros(n_cand)
    for l, heads in carriers.items():
        x = store[l]
        for k in heads:
            wv = x[k * d_head:(k + 1) * d_head]
            v += np.asarray((wv @ proj[(l, k)]).numpy(), dtype=np.float64)
    return v


BF16_U = 2.0 ** -8          # unit round-off of bfloat16, 3.906e-3


def bf16_projection_bound(store, proj, carriers, d_head, n_cand, u=BF16_U):
    """[n_cand]: how far the two extractions may differ, computed not chosen.

    THE TWO PATHS READ DIFFERENT THINGS AND BOTH ARE RIGHT. This one takes the
    o_proj input, which is the model's ACTUAL activation and therefore
    bfloat16; the reference recomputes alpha @ V from float32 copies. So they
    differ by the bf16 representation of that vector, propagated through the
    projection:

        |mine_c - ref_c| = |(delta . P)_c| <= u * sum_i |w_i| |P_ic|

    with |delta_i| <= u |w_i|. Both w and P are in hand, so the right-hand
    side is EVALUATED per candidate rather than replaced by a constant. That
    is the `answer_row_faults` pattern -- "a HARD bound, not an estimate" --
    and it is what stops this being a number fitted to a deviation.

    ⚠ TWO EARLIER VERSIONS OF THIS CHECK WERE BOTH TOO TIGHT, and each fired
    on a correct extraction. The first compared in ulps of the per-head
    contribution, a quantity twelve times smaller than the final logits whose
    cache noise the threshold came from. The second used float32 dot-product
    round-off, sqrt(128) * 2^-24, which is the wrong arithmetic: what limits
    the agreement is the bf16 STORAGE of the activation, not the float32
    projection of it. working rules 11b, twice in one probe.
    """
    b = np.zeros(n_cand)
    for l, heads in carriers.items():
        x = np.asarray(store[l].numpy(), dtype=np.float64)
        for k in heads:
            w = np.abs(x[k * d_head:(k + 1) * d_head])
            b += u * (w @ np.abs(np.asarray(proj[(l, k)].numpy(),
                                            dtype=np.float64)))
    return b


def extraction_faults(mine, ref, bound):
    """This path against the reference ON THE SAME FORWARD.

    One question, and it used to be two: the first version compared this
    path's CACHED reading against the reference's UNCACHED one, so "is the
    extraction right?" and "how much does the cache perturb it?" were fused.
    Both sides here come from the SAME uncached unmasked forward, so anything
    structural -- a wrong head slice, a stray layernorm, the wrong answer row
    -- shows up as a difference far beyond what bf16 storage can explain,
    while the representation difference itself is bounded and allowed.
    """
    a = np.asarray(mine, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    bd = np.asarray(bound, dtype=np.float64)
    if a.shape != b.shape or a.shape != bd.shape:
        return [f"this path {a.shape}, the reference {b.shape} and the bound "
                f"{bd.shape} do not agree in shape"]
    err = np.abs(a - b)
    over = np.flatnonzero(err > bd)
    if not over.size:
        return []
    i = int(over[np.argmax(err[over] - bd[over])])
    return [f"{over.size} of {b.size} candidates differ by more than bf16 "
            f"storage of the activation permits; worst is candidate {i}, "
            f"reference {b[i]:+.8f} vs this path {a[i]:+.8f}, error "
            f"{err[i]:.3e} against a bound of {bd[i]:.3e} "
            f"({err[i] / bd[i]:.2f}x). Both were read off the SAME forward, "
            "so this is neither the cache nor the representation -- it is the "
            "extraction"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--carrier-bundle", required=True)
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--label-space", required=True)
    ap.add_argument("--calibration-dir", required=True)
    add_model_arguments(ap)
    ap.add_argument("--task", default="trec_fine_per_class")
    ap.add_argument("--K-base", type=int, default=5)
    ap.add_argument("--K-full", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-all-head", action="store_true",
                    help="drop the all-open reference arm (a third off the "
                         "run time); the two the effect is defined by stay")
    ap.add_argument("--skip-equivalence", action="store_true",
                    help="skip the check against the reference primitive. It "
                         "costs one uncached 6100-token forward per seed and "
                         "is the only thing verifying this faster path")
    ap.add_argument("--out", required=True)
    add_split_arguments(ap)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ls = load_candidate_space(args.label_space, model=args.model,
                              query_manifest=args.query_manifest,
                              tokenizer=tok)
    from tools.build_query_manifest import load_split
    rows_by_seed = {s: rows_for(args.query_manifest, args.split, s,
                                freeze_manifest=args.freeze_manifest,
                                carriers=args.carrier_bundle,
                                label_space=args.label_space)
                    for s in REGISTERED_SEEDS}
    bundle = json.loads(Path(args.carrier_bundle).read_text(encoding="utf-8"))
    if str(bundle.get("scope")) != CB.SCOPE_FULL:
        raise SystemExit(f"{args.carrier_bundle} is a {bundle.get('scope')!r} "
                         "bundle; 13.5.2 names the full-validation carriers")
    # THIS RUN'S NAMES (run_k10_increment.arm_names), shadowing the K=10
    # module constants; the masks and the meta below use these.
    names = arm_names(args.K_base, args.K_full)
    ARM_ALL, ARM_SEL, ARM_BASE = names["all"], names["sel"], names["base"]
    ARMS = (ARM_BASE, ARM_SEL, ARM_ALL)
    arms = [a for a in ARMS if not (args.skip_all_head and a == ARM_ALL)]

    print("=" * 78)
    print("THE CARRIERS' DIRECT WRITE, UNDER THE SAME TWO MASKS -- "
          f"{args.split.upper()}, EXPLORATORY")
    if args.split != "validation":
        print(f"  ⚠⚠ {args.split}: the ONE-SHOT test split, read under "
              "prereg 14.0b-23 (optional/descriptive)")
    print("=" * 78)
    print("  <carrier head output at the answer row, W_U[candidate]>, summed")
    print("  over the carriers. RAW: no final layernorm, no centring.")
    print("  arms:", ", ".join(arms))
    print("  ⚠ CACHED path, matching RESULTS 51: the demonstrations are "
          "prefilled with NO")
    print("  ⚠ mask, so they attend to each other normally, and the mask "
          "gates only the")
    print("  ⚠ receiver's own tokens. One uncached forward would block "
          "demo-to-demo too.")

    prefixes, text_of, checks, render_prompt = build_prefixes(
        args.task, args.K_full, list(REGISTERED_SEEDS), args.query_manifest,
        args.calibration_dir, ls)
    for c in checks:
        print(f"  [PASS] {c}")

    from tools.diagnostic_forward import extract_per_head_logit_contributions
    from tools.model_loader import load_model
    from tools.prereg_task import docs_to_rows, prefix_demo_docs
    from tools.prompt_render import TaskRenderer
    from tools.probe_kappa_matrix import segment_positions
    from tools.probe_prototype_shrinkage import manifest_reservation
    renderer = TaskRenderer(args.task)
    from tools.tl_heads import TLHeadSet
    import torch
    model = load_model_from_args(args)
    n_l = int(model.config.num_hidden_layers)
    n_h = int(model.config.num_attention_heads)
    cand = [int(t) for t in ls.candidate_token_ids]

    direct = {a: {} for a in arms}
    meta_seeds, ran_by_seed, equiv = {}, {}, {}
    for s in REGISTERED_SEEDS:
        blocks = prefixes[s]
        carriers = {}
        for l, h in CB.heads_for(bundle, s, args.top_n):
            carriers.setdefault(int(l), []).append(int(h))
        proj, d_head = head_projections(model, carriers, cand, torch)
        header, _ = load_calibration_header(
            calibration_path(args.calibration_dir, args.task, args.K_full, s))
        rows = rows_by_seed[s][:args.limit] if args.limit else rows_by_seed[s]
        ran_by_seed[s] = rows

        # THE SAME BASE COLUMNS AS RESULTS 51, derived the same way. If the two
        # runs disagreed about which 180 demonstrations are the base, the two
        # halves of the ratio would describe different interventions.
        excl = manifest_reservation(args.query_manifest, demo_seed=s)
        small_docs = prefix_demo_docs(args.task, args.K_base, [s],
                                      excluded_docs=excl)[s]
        large_docs = prefix_demo_docs(args.task, args.K_full, [s],
                                      excluded_docs=excl)[s]
        small = docs_to_rows(args.task, small_docs)
        large = docs_to_rows(args.task, large_docs)
        bad = nesting_faults(small, large, k_small=args.K_base,
                             k_large=args.K_full)
        if bad:
            raise SystemExit(f"seed {s}: " + "; ".join(bad))
        if renderer.build_prefix(large_docs, header["abstract_labels"]) \
                != list(blocks):
            raise SystemExit(
                f"seed {s}: the rebuilt blocks do not reproduce "
                "build_prefixes' output, so the base columns would be chosen "
                "by an index meaning something else")
        small_set = set(small)
        base_idx = [i for i, d in enumerate(large) if d in small_set]

        n_c = len(tok("", add_special_tokens=True).input_ids)
        through = "\n\n".join(blocks) + "\n\n"
        pre_ids = tok(through, return_tensors="pt").input_ids.to(model.device)
        n_pre = int(pre_ids.shape[1])
        n_demo = n_pre - n_c
        with torch.no_grad():
            base_cache = model(pre_ids, use_cache=True).past_key_values
        first = render_prompt(blocks, text_of[rows[0]["query_id"]])
        _i, seg, _lb, _cl = segment_positions(first, tok,
                                              header["abstract_labels"])
        cols = base_columns(seg, n_c, n_demo, base_idx)
        print(f"\n  seed {s}: carriers {carriers}")
        print(f"    C_s {n_c}, D_s {n_demo}; base {cols.size} columns "
              f"({cols.size / n_demo:.1%}), {len(rows)} queries x {len(arms)} "
              "arms")
        meta_seeds[str(s)] = {"n_common": n_c, "n_demo": int(n_demo),
                              "n_base_columns": int(cols.size),
                              "carriers": {str(k): v
                                           for k, v in carriers.items()}}

        for a in arms:
            direct[a][s] = np.zeros((len(rows), len(cand)))
        # the visibility patterns, once per arm per seed; expanded per query
        rows_by_arm = {a: prepare_mask_rows(
                           masks_for(a, n_l, n_h, 1, n_c, n_demo, 1, carriers,
                                     cols, names=names),
                           torch, model.device) for a in arms}
        for qi, r in enumerate(rows):
            full = render_prompt(blocks, text_of[r["query_id"]])
            fid = tok(full, return_tensors="pt").input_ids.to(model.device)
            if not torch.equal(fid[0, :n_pre], pre_ids[0]):
                raise SystemExit(
                    f"seed {s} query {r['query_id'][:12]}: the prompt does not "
                    "begin with its own prefix at the token level")
            suffix = fid[:, n_pre:]
            n_live = int(suffix.shape[1])
            for a in arms:
                store = {}
                handles = _install_rows(model, rows_by_arm[a], n_live, torch)
                handles += capture_o_proj(model, sorted(carriers), store, torch)
                try:
                    base_cache.crop(n_pre)
                    with torch.no_grad():
                        model(suffix, past_key_values=base_cache,
                              use_cache=True)
                finally:
                    for h in handles:
                        h.remove()
                    base_cache.crop(n_pre)
                missing = [l for l in carriers if l not in store]
                if missing:
                    raise SystemExit(
                        f"seed {s} arm {a!r}: no o_proj input captured for "
                        f"layer(s) {missing}. The hook did not fire, so the "
                        "zeros that would have been written are not a "
                        "measurement")
                direct[a][s][qi] = direct_from_store(store, proj, carriers,
                                                     d_head, len(cand))

            if qi == 0 and not args.skip_equivalence:
                # working rules 3.12(3): a faster path replacing a reference one
                # carries an equivalence check. TWO separate things are asked,
                # because conflating them is what made the first version's
                # threshold meaningless.
                heads = TLHeadSet(model_name=args.model,
                                  heads={(l, k) for l, ks in carriers.items()
                                         for k in ks},
                                  source="carriers")
                # (a) THE EXTRACTION, on one uncached unmasked forward that
                #     both sides read. No mask is installed, so this is the
                #     condition the reference primitive itself runs.
                store_u = {}
                hs = capture_o_proj(model, sorted(carriers), store_u, torch)
                try:
                    with torch.no_grad():
                        model(fid)
                finally:
                    for h in hs:
                        h.remove()
                mine_u = direct_from_store(store_u, proj, carriers, d_head,
                                           len(cand))
                bound = bf16_projection_bound(store_u, proj, carriers, d_head,
                                              len(cand))
                ref_d = np.zeros(len(cand))
                contrib = extract_per_head_logit_contributions(
                    model, fid, heads, target_token_ids=cand)
                for _lk, val in contrib.items():
                    ref_d += np.asarray(val.to(torch.float64).cpu().numpy())
                fbad = extraction_faults(mine_u, ref_d, bound)
                if fbad:
                    raise SystemExit(
                        f"seed {s}: the o_proj-input extraction disagrees "
                        "with extract_per_head_logit_contributions on the "
                        "SAME forward:\n  " + "\n  ".join(fbad))
                err = np.abs(mine_u - ref_d)
                frac = float(np.max(err / np.maximum(bound, 1e-30)))
                rel = float(err.max()
                            / max(float(np.abs(ref_d).max()), 1e-30))
                print(f"    [PASS] the extraction matches the reference "
                      f"primitive on one forward: worst candidate is at "
                      f"{frac:.2f} of the bf16 bound ({rel:.2e} relative)")
                # (b) WHAT THE CACHE COSTS, reported and NOT gated. This is a
                #     property of the cached path RESULTS 51 also runs on, not
                #     of the extraction, and it belongs in the record rather
                #     than in a threshold.
                equiv[str(s)] = {"extraction_rel": rel,
                                 "extraction_frac_of_bf16_bound": frac}
                if ARM_ALL in arms:
                    dc = float(np.abs(direct[ARM_ALL][s][0] - mine_u).max())
                    equiv[str(s)]["cache_max_abs"] = dc
                    equiv[str(s)]["cache_rel"] = dc / max(
                        float(np.abs(mine_u).max()), 1e-30)
                    print(f"    (the cached path differs from the uncached "
                          f"one by {dc:.4f} at most, "
                          f"{equiv[str(s)]['cache_rel']:.2%} of scale -- "
                          "reported, not gated)")
            if (qi + 1) % 40 == 0:
                print(f"    {qi + 1}/{len(rows)}", flush=True)

    np.savez_compressed(
        args.out,
        **{f"direct_{a}_seed{s}": direct[a][s] for a in arms for s in direct[a]},
        query_ids=np.array([[r["query_id"] for r in ran_by_seed[s]]
                            for s in REGISTERED_SEEDS]),
        class_idx=np.array([[int(r["class_idx"]) for r in ran_by_seed[s]]
                            for s in REGISTERED_SEEDS]),
        candidate_classes=np.array([int(c) for c in ls.eligible_classes]),
        seeds=np.array(list(REGISTERED_SEEDS), dtype=np.int64),
        meta=json.dumps({
            "spec": "EXPLORATORY: the direct half of RESULTS 51's response",
            "arm_name": "carriers' direct write under the "
                        f"K={args.K_full} increment masks (K={args.K_base} "
                        "receiver)",
            "roles": {r: names[r] for r in ROLES},
            "model": args.model, "method": args.method, "dtype": args.dtype,
            "attn": args.attn, "task": args.task,
            "K": args.K_full, "K_base": args.K_base,
            "arms": list(arms), "top_n": args.top_n,
            **split_meta(args),
            "seeds": [int(x) for x in REGISTERED_SEEDS],
            "scale_convention": "RAW: weighted_v @ (O_k^T W_U[c]) at the "
                                "answer row, summed over carriers, float32, "
                                "no final layernorm and no centring. Matches "
                                "diagnostic_forward's primitive term for term",
            "path": "cached, matching RESULTS 51: unmasked prefill of "
                    "C_s || D_s, mask applied only to the receiver's tokens",
            "equivalence_vs_primitive": equiv,
            "equivalence_note": "extraction_rel compares this path with "
                                "diagnostic_forward's primitive on ONE "
                                "uncached forward, so it is about the "
                                "extraction alone; cache_* is the cached "
                                "path's own difference from that forward, "
                                "reported and not gated",
            "carrier_bundle_sha256": file_sha256(args.carrier_bundle),
            "query_manifest_sha256": file_sha256(args.query_manifest),
            "label_space_sha256": file_sha256(args.label_space),
            "per_seed": meta_seeds, "limit": int(args.limit),
            "provenance": run_provenance(),
            "note": "Delta_direct only. Delta_final is the same arms' "
                    "candidate logits in the RESULTS 51 npz"}))
    print(f"\n  [output] {args.out}")
    print("  next, ZERO GPU:")
    print(f"    python tools/analyze_carrier_absorption.py --direct {args.out} \\")
    print("        --final <the increment run's npz>   "
          f"(--arm-sel {ARM_SEL!r} --arm-base {ARM_BASE!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
