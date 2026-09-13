"""Per-CLASS direct logit attribution of every head, dumped raw. GPU.

WHAT IS MISSING FROM DISK, AND WHY IT HAS TO BE THIS SHAPE. `discover_carriers`
already runs this forward, but `assemble_scores` collapses it onto the gold
token: `scores_seed<s>` is S[q, l, h], one scalar per head. Method B's question
is not about gold alone -- gain steering moves EVERY candidate, and whether a
lambda rescues a query depends on what the substrate contributes to gold
MINUS what it contributes to each competitor. That vector is not recoverable
from a gold-only scalar, so it has to be measured once.

working rules 3.9: dump the raw tensor. dla[q, l, h, c] is 21 MB per seed at
float32, and with it every later question -- which substrate, which cut, which
lambda, which class pair -- is offline. The alternative is a GPU job per
question, which is exactly the mistake that rule records.

THE PRIMITIVE IS THE SHARED ONE. `extract_per_head_logit_contributions` takes
a LIST of target token ids and projects inside the forward, never
materialising u_per_head: <O_k sum_i a_i V_i, W_U[c]> = (sum_i a_i V_i) .
(O_k^T W_U[c]), and O_k^T W_U[c] is a function of the weights alone. Passing
all 36 candidates therefore costs one matmul more, not one forward more.

A SECOND FORWARD PER QUERY IS SPENT ON PURPOSE. The natural candidate logits
m0 already exist in the K=0 receiver's npz as its `full K5 monolithic` arm, so
recomputing them looks wasteful -- but two independent paths to the same
number is the only cross-file gate available here, and this probe's whole
output is read against m0. `tools/analyze_gain_feasibility.py` checks them.

⚠ RAW, NOT LAYER-NORMALISED. Discovery in this project is by raw DLA
(working rules 6: TL-score is authoritative, margin-discovery is reported
alongside), and mixing a normalised readout into a raw ranking is the "raw /
folded / rms" unit confusion the shared loader exists to prevent. The scale
convention is recorded in the meta so a later reader cannot guess wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.baselines.common import load_candidate_space  # noqa: E402
from tools.baselines.forward import build_prefixes  # noqa: E402
from tools.icl_common import run_provenance  # noqa: E402
from tools.label_space import file_sha256  # noqa: E402
from tools.model_args import add_model_arguments, load_model_from_args  # noqa: E402
from tools.prereg_config import REGISTERED_SEEDS  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--label-space", required=True)
    ap.add_argument("--calibration-dir", required=True)
    add_model_arguments(ap)
    ap.add_argument("--task", default="trec_fine_per_class")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help="queries per seed; 0 = all. A limited run is a smoke "
                         "test, not the artifact")
    ap.add_argument("--carrier-bundle", default=None,
                    help="the bundle a later analysis sums the carriers "
                         "from; REQUIRED for --split test_seed, where the "
                         "freeze manifest is held to it (split_rows)")
    from tools.split_rows import add_split_arguments
    add_split_arguments(ap)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ls = load_candidate_space(args.label_space, model=args.model,
                              query_manifest=args.query_manifest,
                              tokenizer=tok)
    from tools.split_rows import rows_for, split_meta
    if args.split != "validation" and not args.carrier_bundle:
        raise SystemExit("--split test_seed needs --carrier-bundle: the test "
                         "lock binds the run to the carriers it serves")
    rows_by_seed = {s: rows_for(args.query_manifest, args.split, s,
                                freeze_manifest=args.freeze_manifest,
                                carriers=args.carrier_bundle,
                                label_space=args.label_space)
                    for s in REGISTERED_SEEDS}
    if args.split != "validation":
        print(f"  ⚠⚠ {args.split}: the ONE-SHOT test split, read under "
              "prereg 14.0b-23 (optional/descriptive)")

    print("=" * 78)
    print("PER-CLASS DIRECT LOGIT ATTRIBUTION, ALL HEADS -- "
          f"{args.split.upper()}")
    print("=" * 78)
    print("  dla[q, l, h, c] = <head (l,h)'s answer-row output, W_U[cand c]>")
    print("  RAW, not layer-normalised (working rules 6). The gold-only collapse "
          "of this is")
    print("  what discover_carriers stores; Method B needs the whole vector.")

    prefixes, text_of, checks, render_prompt = build_prefixes(
        args.task, args.K, list(REGISTERED_SEEDS), args.query_manifest,
        args.calibration_dir, ls)
    for c in checks:
        print(f"  [PASS] {c}")

    from tools.diagnostic_forward import (extract_per_head_logit_contributions,
                                          precompute_head_projections)
    from tools.model_loader import load_model
    from tools.tl_heads import TLHeadSet
    import torch
    model = load_model_from_args(args)
    n_l = int(model.config.num_hidden_layers)
    n_h = int(model.config.num_attention_heads)
    cand = [int(t) for t in ls.candidate_token_ids]
    heads = TLHeadSet(model_name=args.model,
                      heads={(l, h) for l in range(n_l) for h in range(n_h)},
                      source="all heads, for an offline substrate choice")
    # O_k^T W_U[cand] for every head is a function of the weights alone, and
    # the per-query loop below used to rebuild it for all 1024 heads on the
    # CPU (and move every o_proj slice there) for every query: about a
    # minute per query on banking77 in fp32 (job 844652, RESULTS 63.16).
    # Hoisted here once, as discover_carriers does; the per-query work is two
    # contractions per head on the card. The first query of every seed is
    # also scored by the reference path and the two must agree to 1e-3 of
    # the largest value (working rules 3.12 (3)).
    head_proj = precompute_head_projections(model, heads, target_token_ids=cand,
                                            device=model.device)
    print(f"\n  {n_l} layers x {n_h} heads x {len(cand)} candidates")

    dla, m0, qids, cls = {}, {}, {}, {}
    for s in REGISTERED_SEEDS:
        blocks = prefixes[s]
        rows = rows_by_seed[s][:args.limit] if args.limit else rows_by_seed[s]
        qids[s] = [r["query_id"] for r in rows]
        cls[s] = [int(r["class_idx"]) for r in rows]
        a = np.zeros((len(rows), n_l, n_h, len(cand)), dtype=np.float32)
        b = np.zeros((len(rows), len(cand)), dtype=np.float64)
        print(f"\n  seed {s}: {len(rows)} queries")
        for qi, r in enumerate(rows):
            prompt = render_prompt(blocks, text_of[r["query_id"]])
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            contrib = extract_per_head_logit_contributions(
                model, ids, heads, head_proj=head_proj,
                compute_device=model.device)
            if qi == 0:
                ref = extract_per_head_logit_contributions(
                    model, ids, heads, target_token_ids=cand)
                fast_np = np.stack([contrib[(l, h)].to(torch.float64).cpu().numpy()
                                    for l in range(n_l) for h in range(n_h)])
                ref_np = np.stack([ref[(l, h)].to(torch.float64).cpu().numpy()
                                   for l in range(n_l) for h in range(n_h)])
                scale = max(float(np.abs(ref_np).max()), 1e-12)
                diff = float(np.abs(fast_np - ref_np).max())
                print(f"    [{'PASS' if diff <= 1e-3 * scale else 'FAIL'}] seed {s}: "
                      f"hoisted projections == reference path on the first query, "
                      f"all {n_l * n_h} heads x {len(cand)} candidates: max |diff| "
                      f"{diff:.3e} = {diff / scale:.2e} of scale {scale:.3e}")
                if diff > 1e-3 * scale:
                    raise SystemExit("the hoisted projection disagrees with the "
                                     "reference path beyond fp32 reduction-order "
                                     "noise; refusing to write the attribution on it")
            for (l, h), v in contrib.items():
                a[qi, int(l), int(h)] = np.asarray(
                    v.to(torch.float64).cpu().numpy(), dtype=np.float32)
            with torch.no_grad():
                lg = model(ids).logits[0, -1]
            b[qi] = lg[cand].to(torch.float64).cpu().numpy()
            if (qi + 1) % 20 == 0:
                print(f"    {qi + 1}/{len(rows)}", flush=True)
        dla[s], m0[s] = a, b

    np.savez_compressed(
        args.out,
        **{f"dla_seed{s}": dla[s] for s in REGISTERED_SEEDS},
        **{f"m0_seed{s}": m0[s] for s in REGISTERED_SEEDS},
        query_ids=np.array([qids[s] for s in REGISTERED_SEEDS]),
        class_idx=np.array([cls[s] for s in REGISTERED_SEEDS]),
        candidate_classes=np.array([int(c) for c in ls.eligible_classes]),
        candidate_token_ids=np.array(cand),
        seeds=np.array(list(REGISTERED_SEEDS), dtype=np.int64),
        meta=json.dumps({
            "spec": "EXPLORATORY: input to Method B (intervention.md section "
                    "4), not a section 4 arm",
            "model": args.model, "method": args.method,
            "dtype": args.dtype, "attn": args.attn,
            "task": args.task, "K": args.K,
            "n_layers": n_l, "n_heads": n_h,
            **split_meta(args),
            "scale_convention": "RAW direct logit attribution: "
                                "<O_k sum_i a_i V_i, W_U[c]> at the answer "
                                "row, no final layernorm applied and no "
                                "centring. Do not mix with folded or "
                                "rms-divided readouts",
            "limit": int(args.limit),
            "registered_run": args.limit == 0,
            "query_manifest_sha256": file_sha256(args.query_manifest),
            "label_space_sha256": file_sha256(args.label_space),
            "provenance": run_provenance(),
            "note": "m0 is recomputed here rather than read from the K=0 "
                    "receiver's `full K5 monolithic` arm, so the two can be "
                    "compared; analyze_gain_feasibility does that"}))
    print(f"\n  [output] {args.out}")
    print("  next, ZERO GPU:  python tools/analyze_gain_feasibility.py "
          f"--npz {args.out} \\")
    print("      --carrier-bundle results/carriers_full_validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
