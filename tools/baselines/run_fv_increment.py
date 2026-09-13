"""Function Vectors in the increment setting: one vector PER SEED from the
EXTRA demonstrations, for the receivers to add at the answer row. GPU.

Todd et al. (ICLR 2024) build v_FV from the mean answer-row output of the
heads with the largest causal indirect effect (CIE), summed through each
head's own W_O block, and add it once at the answer row of one layer with
no strength parameter. This is that construction with the project's
information budget (tools/baselines/specs/fv_atp_adapted_spec.md, prereg
14.0b-25):

  * the vector may carry only what the receiver does not already have, so
    the extraction prompts use the EXTRA demonstrations E_s = (K_full draw)
    minus (K_base draw) as their prefix -- 5 per class at 5 -> 10 -- and the
    K_base demonstrations themselves as queries (their labels are the gold;
    no validation or test query is touched). At K_base = 0 there are no base
    demonstrations, so every extra demonstration is the query of a
    LEAVE-ONE-OUT prompt whose prefix is the other extra demonstrations
    (extraction_pairs);
  * abar_lh is the mean over the extraction prompts the model answers
    CORRECTLY (full-vocabulary argmax = the gold token; the paper's rule);
  * the CIE (25 prompts, demo display labels deranged, query and gold
    untouched, metric = full-vocabulary P(gold)) is first approximated for
    every head by attribution patching -- one forward and one backward per
    prompt -- and the top --atp-screen candidates are then re-scored EXACTLY
    by patching, one head at a time; the final head set is the exact top
    --n-heads (the paper's sizes by model: 10 at ~6B, 20 at 7-8B, 50 at 13B,
    100 at 70B -- head_count_for; the AtP is a screen, never the ranking);
  * the edit layer is the paper's L // 3 (11 on 32 layers); alpha = 1 is
    the main arm and the receivers' alpha grid is descriptive.

The AtP backward runs on the answer row alone, the prefix cached under
no_grad (graph_capture: exact under causal attention, one token of graph);
attention runs under sdpa, since nothing in this file reads attention
weights and the eager path would keep [heads, L, L] score matrices.

Output: one JSON sidecar in the layout run_k0_receiver / run_k10_increment
read (`runner.fv_vectors[seed] = {heads, norm, v, layer}`), no npz: no arm
is scored here. write_sidecar is the one place the layout is written, and
the tests build their fixtures by calling it (working rules 3.2b).

    sbatch script/lsu1.sh python tools/baselines/run_fv_increment.py \\
        --query-manifest results/method/L31c36/query_manifest.json \\
        --label-space results/method/L31c36/label_space_L31c36.json \\
        --calibration-dir data/method/L31c36/trec_fine_per_class \\
        --model meta-llama/Llama-3.1-8B --method vanilla --dtype bfloat16 \\
        --K-base 5 --K-full 10 \\
        --out results/method/L31c36/fv_vectors_K10_into_K5.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.baselines import fv_hook as FV  # noqa: E402
from tools.baselines.common import load_candidate_space  # noqa: E402
from tools.baselines.forward import (build_prefixes, calibration_path,  # noqa: E402
                                     load_calibration_header)
from tools.baselines.run_fv_on_icl import (answer_logits,  # noqa: E402
                                           gold_full_prob, patched_logits)
from tools.check_demo_nesting import nesting_faults  # noqa: E402
from tools.icl_common import head_dim, run_provenance  # noqa: E402
from tools.label_space import file_sha256  # noqa: E402
from tools.model_args import (add_model_arguments, load_model_from_args,  # noqa: E402
                              model_meta)
from tools.prereg_task import docs_to_rows, prefix_demo_docs  # noqa: E402
from tools.probe_prototype_shrinkage import manifest_reservation  # noqa: E402
from tools.prompt_render import TaskRenderer  # noqa: E402

NAME = "fv_increment"
REGISTERED_SEEDS = (42, 43, 44)
CIE_PROMPTS = 25          # the upstream default, not a search parameter
# The paper's head counts by model: 10 (GPT-J 6B), 20 (Llama-2 7B), 50 (13B),
# 100 (70B). The cut points BETWEEN those sizes are ours.
HEAD_COUNT_RULE = ((6.5e9, 10), (10.0e9, 20), (20.0e9, 50), (float("inf"), 100))


def edit_layer(n_layers):
    """The paper's layer rule, L // 3: GPT-J 9 of 28, Llama-2-7B 11 of 32."""
    return int(n_layers) // 3


def head_count_for(n_params):
    """The number of CIE heads for a model of `n_params` parameters."""
    for bound, n in HEAD_COUNT_RULE:
        if float(n_params) <= bound:
            return int(n)
    return int(HEAD_COUNT_RULE[-1][1])


def atp_screen_for(n_heads, given=None):
    """Candidates the AtP screen keeps for exact CIE: `given` when the caller
    says, else max(30, ceil(1.5 x n_heads)) -- the spec's 30 for the 10- and
    20-head tiers unchanged, 75 for a 13B model's 50 heads, 150 for 100. A
    fixed 30 could not cover the 50 heads of Qwen3-14B (job 845640)."""
    if given is not None:
        return int(given)
    return max(30, int(math.ceil(1.5 * int(n_heads))))


def extra_demo_docs(task, k_base, k_full, seed, reservation):
    """The K_full draw split into (base docs, extra docs), both in the K_full
    prefix's order, membership decided on (class, text) as the receivers do.
    At k_base = 0 the base is empty and every demonstration is extra."""
    large = prefix_demo_docs(task, k_full, [seed], excluded_docs=reservation)[seed]
    if int(k_base) == 0:
        return [], list(large), large
    small = prefix_demo_docs(task, k_base, [seed], excluded_docs=reservation)[seed]
    small_rows, large_rows = docs_to_rows(task, small), docs_to_rows(task, large)
    bad = nesting_faults(small_rows, large_rows, k_small=k_base, k_large=k_full)
    if bad:
        raise SystemExit(f"seed {seed}: " + "; ".join(bad))
    small_set = set(small_rows)
    base = [(c, d) for (c, d), r in zip(large, large_rows) if r in small_set]
    extra = [(c, d) for (c, d), r in zip(large, large_rows) if r not in small_set]
    if len(base) != len(small_rows):
        raise SystemExit(f"seed {seed}: {len(base)} of {len(large)} blocks matched the "
                         f"{len(small_rows)} base demonstrations (a duplicate (class, text)?)")
    return base, extra, large


def extraction_pairs(base_docs, extra_docs, limit=0):
    """The extraction prompts as [(query (class, doc), prefix docs)].

    K_base > 0: every base demonstration is a query and the extra
    demonstrations are the prefix -- the prefix is exactly the information
    the receiver lacks, and no query is inside its own prefix.
    K_base = 0: there are no base demonstrations, so every extra
    demonstration is the query of a LEAVE-ONE-OUT prompt whose prefix is
    the other extra demonstrations (K_full - 1 in the query's own class).
    Again no query is in its own prefix, and no validation or test query is
    used either way.
    """
    base_docs, extra_docs = list(base_docs), list(extra_docs)
    if not extra_docs:
        raise ValueError("no extra demonstrations: the vector would carry nothing")
    if base_docs:
        pairs = [(q, extra_docs) for q in base_docs]
    else:
        pairs = [(extra_docs[i], extra_docs[:i] + extra_docs[i + 1:])
                 for i in range(len(extra_docs))]
    return pairs[:int(limit)] if limit else pairs


def graph_capture(model, ids, layers, torch):
    """The answer row's logits WITH a graph, and the o_proj inputs of `layers`
    at that row with their gradients retained, so one backward from a scalar
    gives d(scalar)/d(a_lh) for every head at once.

    Every row but the last runs under no_grad into a KV cache; only the LAST
    row runs with the graph. Causal attention makes this exact, not an
    approximation: no earlier row depends on the last one, so the derivative
    of anything at the answer row with respect to the answer row's own o_proj
    input at layer l flows through the answer row's computation alone (its
    later-layer keys and values included, which only it attends to). The
    graph is therefore one token deep -- a prefix prefill plus one token,
    however long the prompt -- where a whole-prompt graph needed ~35 GB at
    3.8k tokens on an 8B model. The cached path differs from a monolithic
    forward by the receivers' measured 1 bf16 ulp, which a screen can carry;
    the exact CIE that ranks the heads runs monolithic forwards.

    The parameters are frozen; the graph exists because the embedding output
    is made to require grad, which puts every later activation on it. Nothing
    is detached in between: detaching layer l's o_proj input would cut the
    path through which lower layers reach the output, and their gradients
    would be silently wrong."""
    caps, handles = {}, []

    def emb(_m, _i, out):
        out.requires_grad_(True)
        return out

    def make(l):
        def fn(_mod, args):
            x = args[0]
            x.retain_grad()
            caps[l] = x
        return fn

    with torch.no_grad():
        past = model(ids[:, :-1], use_cache=True).past_key_values
    handles.append(model.model.embed_tokens.register_forward_hook(emb))
    for l in layers:
        handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make(l)))
    try:
        with torch.enable_grad():
            logits = model(ids[:, -1:], past_key_values=past, use_cache=True).logits[0, -1]
    finally:
        for h in handles:
            h.remove()
    return logits, caps


def atp_from_grads(abar_l, a_l, g_l):
    """First-order CIE for one layer's heads: (abar_lh - a_lh) . dP/da_lh,
    all three [n_heads, head_dim]. Returns [n_heads] float64."""
    abar_l, a_l, g_l = (np.asarray(x, dtype=np.float64) for x in (abar_l, a_l, g_l))
    if not abar_l.shape == a_l.shape == g_l.shape:
        raise ValueError(f"shapes differ: abar {abar_l.shape}, a {a_l.shape}, grad {g_l.shape}")
    return ((abar_l - a_l) * g_l).sum(axis=1)


def atp_scores(model, ids, gold_tok, abar, n_l, n_h, d_h, torch, metric=None):
    """Attribution-patching CIE for every head on one corrupted prompt:
    (abar_lh - a_lh(corrupted)) . dP(gold)/da_lh, P under the full-vocabulary
    softmax (the paper's CIE metric; `metric(logits)` overrides it, for the
    tests). Returns [n_l, n_h] float64."""
    logits, caps = graph_capture(model, ids, range(n_l), torch)
    if metric is None:
        val = torch.softmax(logits.to(torch.float32), dim=-1)[int(gold_tok)]
    else:
        val = metric(logits)
    val.backward()
    score = np.zeros((n_l, n_h), dtype=np.float64)
    for l in range(n_l):
        a = caps[l][0, -1].detach().to(torch.float64).cpu().numpy().reshape(n_h, d_h)
        g = caps[l].grad[0, -1].to(torch.float64).cpu().numpy().reshape(n_h, d_h)
        score[l] = atp_from_grads(abar[l], a, g)
    model.zero_grad(set_to_none=True)
    for l in range(n_l):
        caps[l].grad = None
    return score


def exact_cie_for(model, ids, gold_tok, abar, heads, d_h, batch, torch, metric=None):
    """{(l, h): P(gold | patched) - P(gold | corrupted)} for the given heads,
    one layer's heads packed into a batch (run_fv_on_icl.patched_logits).
    `metric(logits_f64)` defaults to the full-vocabulary P(gold)."""
    if metric is None:
        def metric(lg):
            return gold_full_prob(lg, gold_tok)
    with torch.no_grad():
        base = float(metric(answer_logits(model, ids)[0]))
    # `batch` may be a one-element list: the batch the caller keeps across
    # prompts. On a CUDA OOM the batch is halved and the same heads retried,
    # and the list carries the smaller batch forward (Qwen3-8B fp32 with
    # eight 3.7k-token copies: job 845849). Every head is still scored.
    holder = batch if isinstance(batch, list) else [int(batch)]
    out = {}
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(int(l), []).append(int(h))
    for l, hs in by_layer.items():
        lo = 0
        while lo < len(hs):
            part = hs[lo:lo + holder[0]]
            vals = [torch.as_tensor(abar[l, h]) for h in part]
            try:
                lg = patched_logits(model, ids.repeat(len(part), 1), l, part, vals, d_h)
            except torch.OutOfMemoryError:
                if holder[0] <= 1:
                    raise
                torch.cuda.empty_cache()
                holder[0] = max(1, holder[0] // 2)
                print(f"    exact CIE: out of memory at a batch of {len(part)} patched copies; "
                      f"continuing with {holder[0]}", flush=True)
                continue
            p = metric(lg).cpu().numpy()
            for h, pv in zip(part, p):
                out[(l, h)] = float(pv - base)
            lo += len(part)
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, dtype=np.float64)))
    rb = np.argsort(np.argsort(np.asarray(b, dtype=np.float64)))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def write_sidecar(out, *, vectors, discovery, stats, k_base, k_full, layer, n_heads,
                  head_rule, extraction, model_info, task, seeds, query_manifest,
                  label_space, limit=0, attn="sdpa"):
    """The sidecar the receivers read (--fv-vectors). `vectors[seed]` is
    {heads, norm, v, layer}; the layout is written here and nowhere else."""
    doc = {"runner": {"fv_vectors": {str(s): v for s, v in vectors.items()},
                      "fv_discovery": {str(s): v for s, v in discovery.items()},
                      "fv_stats": {str(s): v for s, v in stats.items()},
                      "K": int(k_full), "K_base": int(k_base), "edit_layer": int(layer),
                      "n_heads": int(n_heads), "head_count_rule": str(head_rule),
                      "cie_prompts": CIE_PROMPTS, "extraction": str(extraction),
                      "hook_boundary": "v_FV = sum over the exact-CIE top-n heads of W_O^(l,h) abar_lh, "
                                       "abar over correctly answered extraction prompts whose prefix is "
                                       "the extra demonstrations; added as alpha * v_FV to the answer row "
                                       "of decoder layer L // 3 (alpha = 1 the main arm)",
                      "note": "vectors only: no arm was scored, so there is no npz. Consumers: "
                              "run_k0_receiver --fv-vectors, run_k10_increment --fv-vectors"},
           "meta": {"baseline": NAME, **dict(model_info), "attn_implementation": attn,
                    "task": str(task), "K": int(k_full), "K_base": int(k_base),
                    "seeds": [int(s) for s in seeds], "limit": int(limit),
                    "query_manifest": str(query_manifest),
                    "query_manifest_sha256": file_sha256(query_manifest),
                    "label_space": str(label_space),
                    "label_space_sha256": file_sha256(label_space),
                    "spec": "tools/baselines/specs/fv_atp_adapted_spec.md (prereg 14.0b-25)",
                    "provenance": run_provenance()}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--label-space", required=True)
    ap.add_argument("--calibration-dir", required=True)
    add_model_arguments(ap)
    ap.add_argument("--task", default="trec_fine_per_class")
    ap.add_argument("--K-base", type=int, required=True)
    ap.add_argument("--K-full", type=int, required=True)
    ap.add_argument("--demo-seeds", default="42,43,44")
    ap.add_argument("--n-heads", type=int, default=None,
                    help="the paper's size rule by default (head_count_for: 10 / 20 / 50 / 100 at "
                         "~6B / 7-8B / 13B / 70B); give a number to override")
    ap.add_argument("--edit-layer", type=int, default=None, help="default: n_layers // 3 (the paper's rule)")
    ap.add_argument("--atp-screen", type=int, default=None,
                    help="candidates kept from the attribution-patching ranking for exact re-scoring; "
                         "default max(30, ceil(1.5 x the head count)): the spec's 30 for the 10- and "
                         "20-head tiers, 75 for the 50 heads of a 13B model, 150 for 100 (a fixed 30 "
                         "refused Qwen3-14B, job 845640)")
    ap.add_argument("--atp-audit", type=int, default=0,
                    help="ALSO score this many random heads exactly and report Spearman / overlap "
                         "between the approximation and the exact CIE (descriptive; 25 forwards each)")
    ap.add_argument("--cie-batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="extraction prompts per seed (0 = all)")
    ap.add_argument("--out", required=True, help="the JSON sidecar")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.demo_seeds.split(",") if s.strip()]
    if not 0 <= args.K_base < args.K_full:
        raise SystemExit(f"--K-base {args.K_base} --K-full {args.K_full}: need 0 <= K_base < K_full")

    from transformers import AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(args.model)
    ls = load_candidate_space(args.label_space, model=args.model,
                              query_manifest=args.query_manifest, tokenizer=tok)
    prefixes, _text_of, checks, _rp = build_prefixes(
        args.task, args.K_full, seeds, args.query_manifest, args.calibration_dir, ls)
    for c in checks:
        print(f"  [PASS] {c}")
    renderer = TaskRenderer(args.task)

    # sdpa for the AtP backward and the batched patched forwards (no [heads, L, L]
    # score matrices); SelfExtend has only its own eager attention (job 845631)
    attn = "eager" if args.method == "selfextend" else "sdpa"
    model = load_model_from_args(args, attn_implementation=attn)
    n_params = 0
    for p_ in model.parameters():
        p_.requires_grad_(False)
        n_params += p_.numel()
    n_l = int(model.config.num_hidden_layers)
    n_h = int(model.config.num_attention_heads)
    d_h = head_dim(model)
    layer = int(args.edit_layer) if args.edit_layer is not None else edit_layer(n_l)
    n_heads = int(args.n_heads) if args.n_heads is not None else head_count_for(n_params)
    head_rule = (f"--n-heads {args.n_heads} (given)" if args.n_heads is not None
                 else f"{n_heads} for {n_params / 1e9:.2f}B parameters (head_count_for)")
    screen = atp_screen_for(n_heads, args.atp_screen)
    if not 0 < screen <= n_l * n_h or n_heads > screen:
        raise SystemExit(f"--atp-screen {screen} must cover the {n_heads} heads and fit "
                         f"the {n_l * n_h} in the model")
    o_by_layer = {l: model.model.layers[l].self_attn.o_proj.weight.detach()
                  .to(torch.float32).cpu().numpy().astype(np.float64) for l in range(n_l)}
    extraction = ("leave-one-out over the extra demonstrations" if args.K_base == 0 else
                  "the base demonstrations as queries, the extra demonstrations as the prefix")
    print("=" * 78)
    print(f"FV, INCREMENT SETTING: K={args.K_full} extra demonstrations into a K={args.K_base} receiver")
    print("=" * 78)
    print(f"  [setup] {args.model} method={args.method} dtype={args.dtype} attn=sdpa; {n_l} x {n_h} "
          f"heads, head_dim {d_h}, {n_params / 1e9:.2f}B parameters")
    print(f"  edit layer {layer} (L // 3); top-{n_heads} by exact CIE ({head_rule}) after an AtP "
          f"screen of {screen}; {CIE_PROMPTS} corrupted prompts; alpha = 1 at the receivers")
    print(f"  extraction prompts: {extraction}")

    vectors, discovery, stats = {}, {}, {}
    for s in seeds:
        t0 = time.time()
        reservation = manifest_reservation(args.query_manifest, demo_seed=s)
        base_docs, extra_docs, large = extra_demo_docs(args.task, args.K_base, args.K_full, s, reservation)
        header, _ = load_calibration_header(calibration_path(args.calibration_dir, args.task, args.K_full, s))
        labels = header["abstract_labels"]
        blocks_full = renderer.build_prefix(large, labels)
        if blocks_full != list(prefixes[s]):
            raise SystemExit(f"seed {s}: the rebuilt K={args.K_full} blocks differ from build_prefixes'")
        pairs = extraction_pairs(base_docs, extra_docs, args.limit)
        print(f"\n  seed {s}: {len(extra_docs)} extra demonstrations, {len(pairs)} extraction prompts "
              f"(prefix of {len(pairs[0][1])} demonstrations each)")

        # ---- abar over the extraction prompts the model answers correctly
        acc_sum = np.zeros((n_l, n_h, d_h), dtype=np.float64)
        n_ok, golds = 0, []
        from tools.baselines.run_fv_on_icl import per_head_answer_row
        for qi, ((c, doc), pre_docs) in enumerate(pairs):
            prompt = renderer.render_prompt(renderer.build_prefix(pre_docs, labels), doc)
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            gold = int(ls.label_token_ids[int(c)])
            with torch.no_grad():
                lg = answer_logits(model, ids)[0]
            correct = int(torch.argmax(lg)) == gold
            golds.append(gold)
            if correct:
                got = per_head_answer_row(model, ids, range(n_l))
                for l in range(n_l):
                    acc_sum[l] += got[l][0].numpy().reshape(n_h, d_h)
                n_ok += 1
            if (qi + 1) % 40 == 0:
                print(f"    {qi + 1}/{len(pairs)} scored, {n_ok} correct so far", flush=True)
        if n_ok == 0:
            raise SystemExit(f"seed {s}: the model answers none of the {len(pairs)} extraction "
                             "prompts correctly; there is nothing to average (the paper averages "
                             "correct prompts only)")
        abar = acc_sum / float(n_ok)
        print(f"  seed {s}: mean activations over {n_ok}/{len(pairs)} correctly answered prompts "
              f"({time.time() - t0:.0f}s)")

        # ---- corrupted prompts: the prefix's display labels deranged
        cie_prompts = []
        for pi in range(min(CIE_PROMPTS, len(pairs))):
            (_c, doc), pre_docs = pairs[pi]
            order = FV.corrupted_label_order(len(pre_docs), pi)
            bad_blocks = [renderer.render_demo(d, labels[int(pre_docs[order[i]][0])])
                          for i, (_c2, d) in enumerate(pre_docs)]
            prompt = renderer.render_prompt(bad_blocks, doc)
            cie_prompts.append((tok(prompt, return_tensors="pt").input_ids.to(model.device), golds[pi]))

        # ---- attribution patching: every head, one forward + backward per prompt
        t1 = time.time()
        atp = np.zeros((n_l, n_h), dtype=np.float64)
        for ids, gold in cie_prompts:
            atp += atp_scores(model, ids, gold, abar, n_l, n_h, d_h, torch)
        atp /= float(len(cie_prompts))
        cand = FV.top_cie_heads(atp, k=screen)
        print(f"  seed {s}: AtP over {len(cie_prompts)} prompts in {time.time() - t1:.0f}s; "
              f"screen {screen} candidates, layers {sorted({l for l, _ in cand})}")

        # ---- exact CIE on the candidates (and on an audit set, if asked)
        t2 = time.time()
        audit = []
        if args.atp_audit:
            rng = np.random.default_rng(int(s))
            pool = [(l, h) for l in range(n_l) for h in range(n_h) if (l, h) not in set(cand)]
            audit = [pool[i] for i in rng.choice(len(pool), size=min(args.atp_audit, len(pool)), replace=False)]
        exact = {lh: 0.0 for lh in list(cand) + audit}
        cie_batch = [int(args.cie_batch)]        # halved in place on an OOM, kept across prompts
        for ids, gold in cie_prompts:
            got = exact_cie_for(model, ids, gold, abar, list(exact), d_h, cie_batch, torch)
            for lh, v in got.items():
                exact[lh] += v
        exact = {lh: v / float(len(cie_prompts)) for lh, v in exact.items()}
        scores = np.full((n_l, n_h), -np.inf)
        for (l, h) in cand:
            scores[l, h] = exact[(l, h)]
        heads = FV.top_cie_heads(np.where(np.isfinite(scores), scores, -1e9), k=n_heads)
        v = FV.fv_from_heads(abar, o_by_layer, heads, d_h)
        rho_cand = spearman([atp[l, h] for l, h in cand], [exact[(l, h)] for l, h in cand])
        print(f"  seed {s}: exact CIE on {len(exact)} heads in {time.time() - t2:.0f}s; "
              f"Spearman(AtP, exact) on the candidates {rho_cand:+.3f}")
        print(f"  seed {s}: top-{n_heads} {heads}")
        print(f"  seed {s}: |v_FV| = {np.linalg.norm(v):.4f}")
        st = {"n_extraction_prompts": len(pairs), "n_correct": n_ok,
              "atp_screen": screen, "cie_batch": int(cie_batch[0]), "spearman_atp_exact_on_candidates": rho_cand,
              "exact_cie_top": {f"{l},{h}": exact[(l, h)] for l, h in heads},
              "atp_top": {f"{l},{h}": float(atp[l, h]) for l, h in cand[:n_heads]}}
        if audit:
            rho_a = spearman([atp[l, h] for l, h in audit], [exact[(l, h)] for l, h in audit])
            st["audit"] = {"n": len(audit), "spearman_atp_exact": rho_a,
                           "exact_top_from_audit_would_enter_top": int(sum(
                               exact[lh] > min(exact[(l, h)] for l, h in heads) for lh in audit))}
            print(f"  seed {s}: audit on {len(audit)} random heads: Spearman {rho_a:+.3f}, "
                  f"{st['audit']['exact_top_from_audit_would_enter_top']} would enter the top-{n_heads}")
        vectors[s] = {"heads": [[int(l), int(h)] for l, h in heads],
                      "norm": float(np.linalg.norm(v)), "v": [float(x) for x in v],
                      "layer": layer}
        queries = [q for q, _p in pairs]
        discovery[s] = {"extraction_queries": [{"class": int(c), "text": r}
                                               for (c, _d), (_c2, r)
                                               in zip(queries, docs_to_rows(args.task, queries))],
                        "extra_demos": docs_to_rows(args.task, extra_docs)}
        stats[s] = st
        torch.cuda.empty_cache()

    write_sidecar(args.out, attn=attn, vectors=vectors, discovery=discovery, stats=stats,
                  k_base=args.K_base, k_full=args.K_full, layer=layer, n_heads=n_heads,
                  head_rule=head_rule, extraction=extraction, model_info=model_meta(args),
                  task=args.task, seeds=seeds, query_manifest=args.query_manifest,
                  label_space=args.label_space, limit=args.limit)
    print(f"\n  [output] {args.out}   (vectors only; no arm was scored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
