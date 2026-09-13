"""Dump the full kernel matrix kappa[query, position] for the TL substrate.

This is the GPU half of temp_task #2 (kappa's query-dependence -- the test that
decides whether our framework is distinguishable from a function vector).

WHY THIS IS COMPUTABLE FROM ONE PASS
------------------------------------
`tasks/trec_fine_task.py::_get_fewshot_examples` caches the few-shot block and
reuses it across every query, and the prompt is [demo block][query]. Two
consequences, both load-bearing:

  1. The demo block occupies the SAME token positions in every prompt, so
     kappa_{q,i} is a complete Q x P matrix (no missing cells) -- a two-factor
     ANOVA needs nothing more than one forward per prompt.

  2. Causal masking means demo-position hidden states cannot see the query.
     Therefore u_i and phi_i^c = <u_i, W_U^c> at demo positions are IDENTICAL
     across queries. The ONLY query-dependent quantity in

         <Delta_TL, W_U^c>|demos  =  sum_i kappa_{q,i} phi_i^c

     is kappa. The per-position votes phi are a FIXED matrix.

(2) is what makes the function-vector question exactly decidable rather than
suggestive: with phi frozen, "kappa is query-invariant" and "the demo-block
output is a fixed vector" are the same statement, and the operational test
(swap kappa_q for its mean, see if the prediction changes) is a closed-form
comparison rather than a new intervention.

Both facts are ASSERTED BY CONSTRUCTION but VERIFIED HERE, because the whole
analysis is void if either fails:
  * pass 0 computes the longest common token prefix over all prompts and
    reports how much of the demo block is actually shared;
  * phi at prefix positions is recomputed on every prompt and checked against
    prompt 0 (max abs deviation reported);
  * Theorem 1 (sum_i kappa_i phi_i^c == <Delta_TL, W_U^c>) is checked per
    prompt as a hook-correctness guard.

WHAT IS WRITTEN
---------------
An .npz that permanently persists the kernel matrix, so every downstream
kappa question (ANOVA, rank-1 fit, recency controls, concentration-vs-K) is
zero-GPU from here on. See tools/analyze_kappa_query_dependence.py.

USAGE (server):
    python tools/probe_kappa_matrix.py \
        --uuid-jsonl data/calibration_trec_fine_per_class_K5_seed42_uuid.jsonl \
        --tl-heads-json results/tl_heads_SE_K5_n250_top8.json \
        --model meta-llama/Llama-2-7b-hf \
        --method selfextend --group-size 4 --neighbor-size 1024 \
        --n-prompts 250 \
        --output results/kappa_matrix_SE_K5_n250_top8.npz
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from tools.diagnostic_forward import diagnostic_forward
from tools.tl_heads import load_heads


# ---------------------------------------------------------------------------
# Prompt structure
# ---------------------------------------------------------------------------

SEG_SPECIAL = -2      # BOS / other special tokens
SEG_QUERY = -1        # the query block (last "\n\n"-delimited block)


def segment_positions(prompt: str, tokenizer, abstract_labels: list[str]):
    """Map every token position to a demo index, the query block, or special.

    Prompt structure (TREC-fine and siblings):
        "Question: <text>\\nType: <label>\\n\\n"   x K*C demos
        "Question: <text>\\nType:"                 query (no label)

    Returns:
        input_ids   : np.int64[n_tokens]
        seg         : np.int32[n_tokens]  demo index >= 0, or SEG_QUERY / SEG_SPECIAL
        is_label    : np.bool_[n_tokens]  True at a demo's label token(s)
        demo_class  : np.int32[n_demos]   class index of each demo (-1 if unparsed)

    Separator tokens ("\\n\\n") are attributed to the block they follow, which
    is the same convention analyze_attention_buckets.py uses for demo_other.
    """
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    ids = np.asarray(enc["input_ids"], dtype=np.int64)

    blocks = prompt.split("\n\n")
    n_blocks = len(blocks)
    block_starts: list[int] = []
    pos = 0
    for blk in blocks:
        block_starts.append(pos)
        pos += len(blk) + 2  # +2 for the "\n\n" separator

    label_to_class = {al: c for c, al in enumerate(abstract_labels)}

    # Per block: (label_char_start, label_char_end, class) for demos; None for query.
    label_ranges: list[tuple[int, int, int] | None] = []
    for bi, blk in enumerate(blocks):
        bs = block_starts[bi]
        if bi == n_blocks - 1:                     # query block: no label
            label_ranges.append(None)
            continue
        nt = blk.find("\nType:")
        if nt == -1:
            label_ranges.append(None)
            continue
        type_end = bs + nt + 1 + len("Type:")      # +1 skips the leading "\n"
        # Label surface is " X": a space then the 1-char abstract label.
        ch = prompt[type_end + 1] if type_end + 1 < len(prompt) else ""
        label_ranges.append((type_end, type_end + 2, label_to_class.get(ch, -1)))

    seg = np.full(len(ids), SEG_SPECIAL, dtype=np.int32)
    is_label = np.zeros(len(ids), dtype=bool)
    for j, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:                      # special token (BOS)
            continue
        bi = bisect.bisect_right(block_starts, s) - 1
        if bi < 0:
            continue
        if bi == n_blocks - 1:
            seg[j] = SEG_QUERY
            continue
        seg[j] = bi
        lr = label_ranges[bi]
        if lr is not None and lr[0] <= s < lr[1]:
            is_label[j] = True

    demo_class = np.array(
        [lr[2] if lr is not None else -1 for lr in label_ranges[:-1]],
        dtype=np.int32,
    )
    return ids, seg, is_label, demo_class


def load_uuid_records(path: Path) -> tuple[dict, list[dict]]:
    header = None
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("header"):
                header = obj
            else:
                out.append(obj)
    if header is None:
        raise ValueError(f"{path} has no header line")
    return header, out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uuid-jsonl", required=True)
    p.add_argument("--tl-heads-json", required=True)
    p.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--method", required=True,   # no default: jobs 841345/842371/843361 ran Llama-3.1 under SelfExtend because the driver passed nothing
                   choices=["vanilla", "selfextend"])
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--neighbor-size", type=int, default=1024)
    p.add_argument("--n-prompts", type=int, default=250)
    p.add_argument("--output", required=True, help="destination .npz")
    args = p.parse_args()

    header, records = load_uuid_records(Path(args.uuid_jsonl))
    records = records[: args.n_prompts]
    label_token_ids = [int(x) for x in header["label_token_ids"]]
    abstract_labels = header["abstract_labels"]
    n_classes = len(label_token_ids)
    n_train_classes = int(header.get("n_train_classes", n_classes))
    print(f"[setup] {len(records)} prompts from {args.uuid_jsonl}")
    print(f"[setup] n_classes={n_classes}  n_train_classes={n_train_classes}  "
          f"K={header.get('K')}")

    tl_heads = load_heads(args.tl_heads_json)
    head_keys = sorted(tl_heads.heads)
    n_heads = len(head_keys)
    print(f"[setup] {n_heads} TL heads from {args.tl_heads_json}")
    print(f"[setup]   layers: {sorted(tl_heads.layers())}")

    from transformers import AutoTokenizer
    from tools.model_loader import load_model
    tok = AutoTokenizer.from_pretrained(args.model)

    # ---- Pass 0 (CPU): how much of the prompt is a SHARED prefix? -----------
    # The entire design assumes the demo block is identical across queries.
    # Measure it instead of trusting it.
    print("\n[pass0] tokenizing all prompts to measure the shared prefix...")
    ids_all = [np.asarray(tok(r["prompt"], add_special_tokens=True).input_ids,
                          dtype=np.int64) for r in records]
    ref_ids = ids_all[0]
    common = len(ref_ids)
    for ids in ids_all[1:]:
        common = min(common, len(ids))
        j = 0
        while j < common and ids[j] == ref_ids[j]:
            j += 1
        common = j

    ref_ids0, seg0, is_label0, demo_class0 = segment_positions(
        records[0]["prompt"], tok, abstract_labels)
    if len(ref_ids0) != len(ref_ids) or not np.array_equal(ref_ids0, ref_ids):
        raise RuntimeError("tokenizer disagreement between pass 0 and segmentation")
    q_start_idx = np.nonzero(seg0 == SEG_QUERY)[0]
    if q_start_idx.size == 0:
        raise RuntimeError("no query block found in prompt 0 -- check the template")
    q_start = int(q_start_idx[0])

    n_demos_total = int(demo_class0.size)
    print(f"[pass0] prompt 0: {len(ref_ids)} tokens, query block starts at {q_start}, "
          f"{n_demos_total} demo blocks")
    print(f"[pass0] longest common token prefix over {len(records)} prompts: {common}")

    if common < q_start:
        # Partial sharing: only the demos fully inside the common prefix are usable.
        usable = int((seg0[:common] >= 0).sum())
        print(f"[pass0] !! WARNING: the demo block is NOT fully shared "
              f"(common={common} < query_start={q_start}).")
        print(f"[pass0]    Only the first {usable} prefix tokens are comparable "
              f"across queries; demos beyond that are dropped from the matrix.")
        print(f"[pass0]    The kappa matrix is still valid on the shared part, but "
              f"report the truncation.")
        P = common
    else:
        P = q_start
        print(f"[pass0] OK: the whole demo block ({P} tokens) is shared across all "
              f"prompts -- kappa is a complete {len(records)} x {P} matrix.")

    seg_prefix = seg0[:P].copy()
    is_label_prefix = is_label0[:P].copy()
    n_demos_in_prefix = int(seg_prefix.max()) + 1 if (seg_prefix >= 0).any() else 0
    demo_class = demo_class0[:n_demos_in_prefix].copy()
    print(f"[pass0] prefix covers {n_demos_in_prefix} demos "
          f"({int((seg_prefix >= 0).sum())} demo tokens, "
          f"{int((seg_prefix == SEG_SPECIAL).sum())} special tokens)")
    if (demo_class < 0).any():
        n_bad = int((demo_class < 0).sum())
        print(f"[pass0] !! WARNING: {n_bad} demo labels did not parse to a class")

    # ---- Model ------------------------------------------------------------
    print(f"\n[setup] loading {args.model} [{args.method}] on {args.device}...")
    t0 = time.time()
    model = load_model(
        args.model, method=args.method, dtype=args.dtype,
        device=args.device, attn_implementation="eager",
        group_size=args.group_size, neighbor_size=args.neighbor_size,
    )
    print(f"[setup] done in {time.time() - t0:.1f}s")

    w_u = model.lm_head.weight.detach()
    w_u_labels = w_u[torch.as_tensor(label_token_ids, dtype=torch.long)]
    w_u_labels = w_u_labels.to(torch.float32).cpu()          # [C, d_model]

    # ---- Storage ----------------------------------------------------------
    Q = len(records)
    kappa_prefix = np.zeros((Q, P, n_heads), dtype=np.float32)
    phi_prefix = np.zeros((P, n_classes, n_heads), dtype=np.float32)   # prompt 0
    proj_query_part = np.zeros((Q, n_classes, n_heads), dtype=np.float32)
    mass_prefix = np.zeros((Q, n_heads), dtype=np.float32)
    mass_query = np.zeros((Q, n_heads), dtype=np.float32)
    mass_special = np.zeros((Q, n_heads), dtype=np.float32)
    delta_tl_proj = np.zeros((Q, n_classes), dtype=np.float32)
    identity_diff = np.zeros(Q, dtype=np.float32)
    phi_dev = np.zeros(Q, dtype=np.float32)
    n_tokens_arr = np.zeros(Q, dtype=np.int32)
    query_len_arr = np.zeros(Q, dtype=np.int32)
    true_class_idx = np.zeros(Q, dtype=np.int32)

    print()
    t_start = time.time()
    for qi, rec in enumerate(records):
        prompt = rec["prompt"]
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(args.device)
        extraction = diagnostic_forward(model, enc.input_ids, tl_heads)
        n_tok = extraction.n_tokens

        ids_q, seg_q, _is_lab_q, _dc_q = segment_positions(prompt, tok, abstract_labels)
        if len(ids_q) != n_tok:
            raise RuntimeError(
                f"prompt {qi}: segmentation length {len(ids_q)} != n_tokens {n_tok}")

        n_tokens_arr[qi] = n_tok
        query_len_arr[qi] = int((seg_q == SEG_QUERY).sum())
        true_class_idx[qi] = int(rec["true_class_idx"])

        pred_full = np.zeros(n_classes, dtype=np.float64)
        for hj, hk in enumerate(head_keys):
            alpha = extraction.alpha_per_head[hk].numpy()            # [n_tok] fp32
            u = extraction.u_per_head[hk]                            # [n_tok, d_model]
            phi = (u @ w_u_labels.T).numpy()                         # [n_tok, C]

            kappa_prefix[qi, :, hj] = alpha[:P]
            mass_prefix[qi, hj] = float(alpha[:P][seg_prefix >= 0].sum())
            mass_special[qi, hj] = float(alpha[:P][seg_prefix == SEG_SPECIAL].sum())
            mass_query[qi, hj] = float(alpha[P:].sum())

            # Everything past the shared prefix, lumped into one vector so the
            # full Theorem-1 sum is reconstructable without storing query-side phi.
            proj_query_part[qi, :, hj] = alpha[P:] @ phi[P:]
            pred_full += alpha @ phi

            if qi == 0:
                phi_prefix[:, :, hj] = phi[:P]
            else:
                dev = float(np.abs(phi[:P] - phi_prefix[:, :, hj]).max())
                phi_dev[qi] = max(phi_dev[qi], dev)

        actual = (extraction.delta_tl @ w_u_labels.T).numpy().astype(np.float64)
        delta_tl_proj[qi] = actual
        identity_diff[qi] = float(np.abs(pred_full - actual).max())

        if (qi + 1) % 10 == 0 or qi == Q - 1:
            el = time.time() - t_start
            print(f"  [{qi+1:4d}/{Q}] n_tok={n_tok:5d}  "
                  f"identity_maxdiff={identity_diff[qi]:.2e}  "
                  f"phi_dev={phi_dev[qi]:.2e}  "
                  f"({el/(qi+1):.2f}s/prompt)")
        del extraction
        torch.cuda.empty_cache()

    # ---- Verification summary --------------------------------------------
    print()
    print("=" * 78)
    print("PREREQUISITE CHECKS (the analysis is void if any of these fails)")
    print("=" * 78)
    print(f"  shared prefix               : {P} of {q_start} demo-block tokens "
          f"({'COMPLETE' if P >= q_start else 'TRUNCATED'})")
    print(f"  phi invariance across queries: max abs dev = {phi_dev.max():.3e}")
    print(f"       (architecturally ~0 by causal masking. In practice GPU kernels tile")
    print(f"        by sequence length, so expect LENGTH-DETERMINISTIC fp noise that")
    print(f"        amplifies with depth. Benign iff same-length prompts give EXACTLY 0")
    print(f"        (checked below); content-dependent dev = broken prefix = STOP.")
    print(f"        Impact on N4 is closed by tools/check_kappa_phi_noise.py.)")
    ref_len = int(n_tokens_arr[0])
    same_len = n_tokens_arr == ref_len
    print(f"  phi dev on same-length prompts (n={int(same_len.sum())}, len={ref_len}): "
          f"max = {phi_dev[same_len].max():.3e}  <- MUST be ~0")
    print(f"  Theorem 1 identity           : max abs diff = {identity_diff.max():.3e} "
          f"(median {np.median(identity_diff):.3e})")
    print(f"  attention mass  demos/special/query (mean over prompts, per head):")
    for hj, hk in enumerate(head_keys):
        print(f"       {str(hk):>10s}  demos={mass_prefix[:, hj].mean():.4f}  "
              f"special={mass_special[:, hj].mean():.4f}  "
              f"query={mass_query[:, hj].mean():.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        kappa_prefix=kappa_prefix,
        phi_prefix=phi_prefix,
        proj_query_part=proj_query_part,
        seg_prefix=seg_prefix,
        is_label_prefix=is_label_prefix,
        demo_class=demo_class,
        mass_prefix=mass_prefix,
        mass_query=mass_query,
        mass_special=mass_special,
        delta_tl_proj=delta_tl_proj,
        identity_diff=identity_diff,
        phi_dev=phi_dev,
        n_tokens=n_tokens_arr,
        query_len=query_len_arr,
        true_class_idx=true_class_idx,
        label_token_ids=np.asarray(label_token_ids, dtype=np.int64),
        head_keys=np.asarray([f"({l},{k})" for (l, k) in head_keys]),
        meta=np.asarray([json.dumps({
            "model": args.model,
            "method": args.method,
            "group_size": args.group_size,
            "neighbor_size": args.neighbor_size,
            "uuid_jsonl": args.uuid_jsonl,
            "tl_heads_json": args.tl_heads_json,
            "tl_heads_source": tl_heads.source,
            "n_prompts": Q,
            "n_classes": n_classes,
            "n_train_classes": n_train_classes,
            "K": header.get("K"),
            "prefix_tokens": int(P),
            "query_block_start_prompt0": int(q_start),
            "prefix_complete": bool(P >= q_start),
            "n_demos_in_prefix": int(n_demos_in_prefix),
            "n_demos_total": int(n_demos_total),
        })]),
    )
    print()
    print(f"[output] wrote {out_path}  "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[next]   python tools/analyze_kappa_query_dependence.py "
          f"--npz {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
