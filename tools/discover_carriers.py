"""Section 2.4(2) carrier discovery: one forward pass, nine head sets.

    S_lh = E_q[ <o_gold_qlh, W_U[y_q]> - (1/M) sum_m <o_rand_qlh, W_U[y_q]> ]

Per query, the gold prompt and M random-UUID-label control prompts are run,
each head's output at the answer position is projected onto the gold label's
unembedding row, and the control mean is subtracted. That per-query score is
the whole product of the GPU: the nine sets are nine MEANS of it
(tools/carrier_bundle.build_from_scores), which is why section 13.6.3 says
discovery does not double with the folds.

    2592 forwards = (1 gold + 5 controls) x 144 validation x 3 prefix seeds.

WHICH IMPLEMENTATION, AND WHY NOT THE ONE 2.4(2) NAMES. The preregistration
names tools.tl_score.compute_tl_scores. That wrapper accumulates a mean over
prompts and returns a ranked head list, so it cannot produce the PER-QUERY
scores 13.6.3 needs: the nine sets are nine means of one per-query quantity
over different subsets of rows, and getting them from the wrapper would mean
calling it three times per seed -- tripling the forwards, which is exactly
what 13.6.3 says must not happen. So this uses the primitive that wrapper is
built on, extract_per_head_logit_contributions. The scoring convention is not
reimplemented; only the aggregation differs, and the aggregation is what
13.6.3 changed.

That primitive is also why the forward is affordable. Materialising
u_per_head for 1024 heads is n_tokens x d_model per head -- about 16 GB a
forward at n_tokens ~ 1000, moved to CPU, then 1024 small matmuls in a Python
loop; the first version of this file did exactly that and measured 5.4 s per
forward. But

    <O_k sum_i a_i V_i , W_U[y]> = (sum_i a_i V_i) . (O_k^T W_U[y])

and O_k^T W_U[y] depends on the WEIGHTS ALONE. Projecting first never builds
u. Building it put a prompt-independent quantity inside the prompt loop,
which is working rules 3.12's section 27 mistake repeated.

WHAT IS PURE AND WHAT IS NOT. `variant_labels`, `assemble_scores` and
`fold_labels` take no model and are armed in
tools/test_discover_carriers.py: they are where reproducibility lives -- the
control label sequences must be a function of (prefix_seed, query_id, m) and
nothing else -- and where a defect would be invisible in the output. The
forward loop needs a GPU and is marked as such.

VALIDATION ONLY. This reads `validation`, never a test split, so it needs no
freeze manifest and no clean tree -- choosing carriers is what validation is
for. The run's own commit is stamped into the output regardless, because the
artifact it writes is an input to everything downstream.

Run (server):
    sbatch script/lsu1.sh python tools/discover_carriers.py \\
        --query-manifest results/prereg_method_A_query_manifest.json \\
        --uuid-jsonl data/calibration_L31_seed42.jsonl \\
        --out-scores results/carrier_scores.npz \\
        --out-fold results/carriers_fold.json \\
        --out-full results/carriers_full_validation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import carrier_bundle, carrier_schema  # noqa: E402
from tools.label_space import file_sha256  # noqa: E402
from tools.model_args import (add_model_arguments,  # noqa: E402
                              load_model_from_args)
from tools.prereg_config import (N_ELIGIBLE, N_VALIDATION,  # noqa: E402
                                 REGISTERED_SEEDS)


# ==========================================================================
# pure: the control prompts
# ==========================================================================
def variant_labels(abstract_labels, prefix_seed, query_id, m, *,
                   pool=None):
    """The label sequence for control variant `m` of one query.

    REPRODUCIBLE FROM (prefix_seed, query_id, m) AND NOTHING ELSE. The seed
    comes from carrier_schema.random_uuid_seed rather than being recomputed
    here, so the rule the artifact declares is the rule that ran -- a schema
    can check that a field says "low 32 bits of SHA256(...)" but only a
    shared implementation makes the field true.

    The prefix seed LEADS the hash (2.4(2)): all three seeds discover over
    their own 144 queries, and without it two seeds drawing the same query
    would get bit-identical control prompts, making their control terms
    correlated -- "independent per seed" in name only.

    `pool` is the set of labels to draw from; by default the abstract labels
    themselves, so a control prompt is a PERMUTATION of the real label
    inventory rather than tokens from somewhere else. Drawing from elsewhere
    would confound "this head writes the gold label" with "this head responds
    to unfamiliar tokens".
    """
    labels = list(abstract_labels)
    src = list(pool if pool is not None else labels)
    if not labels or not src:
        raise ValueError("no labels to permute")
    rng = np.random.default_rng(
        carrier_schema.random_uuid_seed(prefix_seed, query_id, m))
    return [src[i] for i in rng.integers(0, len(src), size=len(labels))]


def assemble_scores(gold_proj, rand_proj):
    """S[q, l, h] from the gold and control projections.

    gold_proj is [Q, L, H]; rand_proj is [Q, M, L, H]. The control term is
    the MEAN over M, per query -- not a pooled mean over all queries, which
    would subtract a constant and leave the ranking unchanged while
    describing a different quantity.
    """
    g = np.asarray(gold_proj, dtype=np.float64)
    r = np.asarray(rand_proj, dtype=np.float64)
    if g.ndim != 3:
        raise ValueError(f"gold_proj is {g.ndim}-D, expected [query, layer, "
                         "head]")
    if r.ndim != 4 or r.shape[0] != g.shape[0] or r.shape[2:] != g.shape[1:]:
        raise ValueError(f"rand_proj {r.shape} does not match gold_proj "
                         f"{g.shape}; expected [query, variant, layer, head]")
    if r.shape[1] != carrier_schema.N_RANDOM_VARIANTS:
        raise ValueError(
            f"{r.shape[1]} control variants, section 2.4(2) registers "
            f"M={carrier_schema.N_RANDOM_VARIANTS}")
    return g - r.mean(axis=1)


def fold_labels(query_ids, class_of):
    """The two folds, from the GATE'S OWN function. One implementation.

    This used to be a second copy that claimed in its docstring to be "the
    same rule as method_a_viability.assign_folds" and was not: it ranked each
    class's rows BY POSITION in `query_ids`, while assign_folds ranks them BY
    QUERY_ID. Those agree only when each class's rows already arrive in
    query_id order, and they never do -- draw_validation_from_train sorts each
    class by validation_hash(prefix_seed, class, text), a different hash from
    query_id = content_hash(class, text), and load_split returns that stored
    order unsorted.

    So discovery's fold f and the gate's fold f were different row sets, and
    the head set the gate calls "chosen on the other fold" had been chosen on
    rows overlapping the fold it was scoring -- measured at 0.495 of every
    evaluation row, which is what an independent re-split gives. The cross-fit
    protected the gamma choice (that is selected inside the gate, on the
    gate's own folds) and almost nothing of the head choice.

    The bundle's row check could not catch it: discover_carriers was the only
    caller of bundle_faults and passed THIS function's output as the reference
    it compared against, so discovery was checked against itself.

    Kept as a named wrapper because build_from_scores and the bundle check
    take {seed: [labels]}, not the [S, Q] array assign_folds returns.

    The registered-shape guard is switched OFF here and only here: a --limit
    run is meant to reach build_from_scores with short folds so the bundle
    check refuses the bundle (that refusal is what makes --limit safe). The
    RANKING is not negotiable and is not re-derived.
    """
    from tools.method_a_viability import assign_folds
    qs = [str(q) for q in query_ids]
    cs = [int(class_of[q]) for q in qs]
    return [int(f) for f in assign_folds(np.array([qs]), np.array([cs]),
                                         require_registered_shape=False)[0]]


# ==========================================================================
# the run
# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--uuid-jsonl", required=True)
    ap.add_argument("--label-space")
    add_model_arguments(ap)
    ap.add_argument("--task", default=carrier_schema.TASK)
    ap.add_argument("--K", type=int, default=carrier_schema.K)
    ap.add_argument("--out-scores", required=True)
    ap.add_argument("--out-fold", required=True)
    ap.add_argument("--out-full", required=True)
    ap.add_argument("--carrier-impl", default="gqa_group_v",
                    choices=("mha_v", "gqa_group_v", "gqa_head_realized"))
    ap.add_argument("--limit", type=int, default=0,
                    help="debug only; a limited run is NOT a registered "
                         "discovery and the artifacts say so")
    args = ap.parse_args(argv)

    if Path(args.out_fold).resolve() == Path(args.out_full).resolve():
        raise SystemExit(
            "--out-fold and --out-full name the same file. The six fold sets "
            "and the three full-validation sets must not travel together: "
            "the gate reads the first and must never be able to reach the "
            "second, which is chosen on the very rows it evaluates.")

    from tools.build_query_manifest import load_split
    from tools.icl_common import run_provenance
    from tools.probe_prototype_shrinkage import manifest_reservation
    from tools.prompt_render import TaskRenderer, build_doc_lookup

    qm_sha = hashlib.sha256(
        Path(args.query_manifest).read_bytes()).hexdigest()

    rows_by_seed, class_of = {}, {}
    for s in REGISTERED_SEEDS:
        rows = list(load_split(args.query_manifest, "validation", demo_seed=s))
        if args.limit:
            rows = rows[:args.limit]
        rows_by_seed[s] = rows
        for r in rows:
            class_of[r["query_id"]] = int(r["class_idx"])
    # THE VALIDATION COUNT IS THE MANIFEST'S, not a constant: 144 on TREC's
    # 36 eligible classes x 4, another task's classes x its per-class draw.
    _man = json.loads(Path(args.query_manifest).read_text(encoding="utf-8"))
    n_validation = (len(_man["eligible_classes"])
                    * int(_man["validation_per_class"]))
    if not args.limit:
        for s, rows in rows_by_seed.items():
            if len(rows) != n_validation:
                raise SystemExit(
                    f"seed {s}: {len(rows)} validation queries, the manifest "
                    f"registers {n_validation} ({len(_man['eligible_classes'])} "
                    f"classes x {_man['validation_per_class']})")

    ids_by_seed = {s: [r["query_id"] for r in rows]
                   for s, rows in rows_by_seed.items()}
    folds_by_seed = {s: fold_labels(ids, class_of)
                     for s, ids in ids_by_seed.items()}

    print("=" * 78)
    print(f"CARRIER DISCOVERY -- section 2.4(2), M="
          f"{carrier_schema.N_RANDOM_VARIANTS}")
    print("=" * 78)
    for s in REGISTERED_SEEDS:
        f = folds_by_seed[s]
        print(f"  seed {s}: {len(f)} queries, folds "
              f"{f.count(0)}/{f.count(1)}")
    n_fwd = ((1 + carrier_schema.N_RANDOM_VARIANTS)
             * sum(len(v) for v in ids_by_seed.values()))
    print(f"  {n_fwd} forwards (gold + {carrier_schema.N_RANDOM_VARIANTS} "
          "controls per query). Discovery does NOT double with the folds: "
          "the nine sets are nine means of this one pass (13.6.3).")

    # ---------------------------------------------------------------- GPU
    # THE ONLY PART THAT NEEDS A MODEL. Everything above and below is pure
    # and armed in tools/test_discover_carriers.py.
    import torch

    from tools.diagnostic_forward import (
        extract_per_head_logit_contributions, precompute_head_projections)
    from tools.icl_common import load_jsonl
    from tools.prereg_task import load_task, prefix_demo_docs
    from tools.tl_heads import TLHeadSet

    header, _cal = load_jsonl(Path(args.uuid_jsonl))
    if header is None or not header.get("abstract_labels"):
        raise SystemExit(f"{args.uuid_jsonl}: no header with abstract_labels")
    abstract = header["abstract_labels"]
    # THE HEADER MUST BELONG TO THIS MANIFEST. Everything below reads
    # label_token_ids out of it to pick W_U rows, so the wrong --uuid-jsonl
    # projects every head onto the wrong label and the scores are wrong
    # while looking entirely normal -- the probe checks this and discovery
    # did not. With --label-space the frozen space does the comparison;
    # without one, the header's own manifest hash is compared directly,
    # because a calibration file that names a different manifest was built
    # against different draws.
    if args.label_space:
        from tools.label_space import FrozenLabelSpace, assert_header_matches
        from transformers import AutoTokenizer
        lspace = FrozenLabelSpace.load(
            args.label_space, model=args.model,
            query_manifest=args.query_manifest,
            tokenizer=AutoTokenizer.from_pretrained(args.model))
        assert_header_matches(lspace, header, path=args.uuid_jsonl,
                              model=args.model,
                              query_manifest_sha256=qm_sha)
        print("  calibration header matches the frozen label space")
    else:
        # NO FALLBACK. This branch used to compare the header's own manifest
        # hash "directly", but the comparison was `if got and got != qm_sha`
        # -- so an ABSENT field checked nothing, and every calibration file
        # older than that field opened the door. Job 835618 ran here and
        # printed `header manifest hash None`, which is the check announcing
        # that it did not run.
        #
        # assert_header_matches, twenty lines away, says "A MISSING field is a
        # failure, not a skipped check". Two places in one repository, one
        # each way. This is the one that was wrong.
        #
        # What the weak branch could not see, and the frozen space can: the
        # flat data/calibration_..._uuid.jsonl and
        # data/method_a/llama31/calibration_..._uuid.jsonl carry the SAME 50
        # surfaces and 50 DIFFERENT token ids, and re-deriving them with
        # Llama-3.1's tokenizer in prompt context reproduces the llama31 file.
        # Reading the other one projects every head onto the wrong rows and
        # nothing about the output looks unusual.
        raise SystemExit(
            "--label-space is required. Every head's projection target comes "
            "from the header's label_token_ids, and the only thing that can "
            "show those ids belong to THIS model's tokenizer is the frozen "
            "label space (it records the calibration it was built from, that "
            "file's sha256, and the tokenizer provenance). The old fallback "
            "compared the header's manifest hash only when the header HAD one, "
            "so a file without the field was waved through.\n"
            "  pass: --label-space results/label_space_llama31.json")

    task = load_task(args.task, args.K, REGISTERED_SEEDS[0])
    text_of = build_doc_lookup(task, args.task)
    R = TaskRenderer(args.task)
    model = load_model_from_args(args)
    n_kv = int(getattr(model.config, "num_key_value_heads",
                       model.config.num_attention_heads))
    tok_ids = header.get("label_token_ids")
    if not tok_ids:
        raise SystemExit(f"{args.uuid_jsonl}: header has no label_token_ids")
    # NO w_u here: the primitive takes target_token_ids and does the
    # projection itself, which is what keeps u from ever being built.
    n_l = int(model.config.num_hidden_layers)
    n_h = int(model.config.num_attention_heads)
    # EVERY head, because 2.4(2) ranks all 32x32 and takes the top 8 -- the
    # ranking is the artifact, not just the winners. TLHeadSet is the shape
    # diagnostic_forward wants; there is no all-heads constructor because
    # every other caller passes a discovered subset.
    heads = TLHeadSet(model_name=args.model,
                      heads={(l, h) for l in range(n_l) for h in range(n_h)},
                      source="section 2.4(2): the full ranking, top-8 frozen")

    # THE FAST PATH (working rules 3.12). O_k^T W_U[label] is a function of the
    # weights alone, yet the loop below used to rebuild it for every one of
    # the 1024 heads of every prompt on the CPU -- 25 s a forward with the
    # card idle, hours for a discovery. Hoisted once for every label token
    # and contracted on the device; the first prompt of the run is scored
    # by BOTH paths and they must agree, so the optimisation cannot quietly
    # change the ranking (3.12 (3)).
    label_ids = [int(x) for x in tok_ids]
    head_proj = precompute_head_projections(model, heads,
                                            target_token_ids=label_ids,
                                            device=model.device)
    col_of = {}
    for j, tid in enumerate(label_ids):
        col_of.setdefault(tid, j)
    fast_checked = False
    scores_by_seed = {}
    for s in REGISTERED_SEEDS:
        res = manifest_reservation(args.query_manifest, demo_seed=s)
        demos = prefix_demo_docs(args.task, args.K, [s], excluded_docs=res)[s]
        blocks = R.build_prefix(demos, abstract)
        rows = rows_by_seed[s]
        gold = np.zeros((len(rows), n_l, n_h), dtype=np.float64)
        rand = np.zeros((len(rows), carrier_schema.N_RANDOM_VARIANTS,
                         n_l, n_h), dtype=np.float64)
        for qi, r in enumerate(rows):
            true_id = int(tok_ids[int(r["class_idx"])])
            for m in range(-1, carrier_schema.N_RANDOM_VARIANTS):
                # m = -1 is the GOLD prompt; 0..M-1 are the controls, whose
                # label sequences are a function of (seed, query_id, m) only.
                labs = (abstract if m < 0 else
                        variant_labels(abstract, s, r["query_id"], m))
                prompt = R.render_prompt(R.build_prefix(demos, labs),
                                         text_of[r["query_id"]])
                # THE SHARED PRIMITIVE, not a second implementation. It
                # projects onto W_U[y] inside the forward and never
                # materialises u_per_head: <O_k sum_i a_i V_i, W_U[y]> =
                # (sum_i a_i V_i) . (O_k^T W_U[y]), and O_k^T W_U[y] is a
                # function of the weights alone. Building u first cost ~16 GB
                # a forward and put a prompt-independent quantity inside the
                # prompt loop -- working rules 3.12's section 27 mistake again.
                ids = prompt_ids(prompt, model)
                contrib = extract_per_head_logit_contributions(
                    model, ids, heads, head_proj=head_proj,
                    compute_device=model.device)
                col = col_of[true_id]
                proj = np.zeros((n_l, n_h), dtype=np.float64)
                for (l, h), v in contrib.items():
                    proj[l, h] = float(v[col].item())
                if not fast_checked:
                    # the reference path, once: every head, this prompt
                    ref = extract_per_head_logit_contributions(
                        model, ids, heads, target_token_ids=[true_id])
                    ref_np = np.zeros((n_l, n_h), dtype=np.float64)
                    for (l, h), v in ref.items():
                        ref_np[l, h] = float(v[0].item())
                    scale = max(float(np.abs(ref_np).max()), 1e-12)
                    diff = float(np.abs(ref_np - proj).max())
                    print(f"  [{'PASS' if diff <= 1e-3 * scale else 'FAIL'}] "
                          f"fast path == reference path on the first prompt, "
                          f"all {n_l * n_h} heads: max |diff| {diff:.3e} = "
                          f"{diff / scale:.2e} of scale {scale:.3e}")
                    if diff > 1e-3 * scale:
                        raise SystemExit(
                            "the hoisted projection disagrees with the "
                            "reference path beyond fp32 reduction-order "
                            "noise; refusing to rank on it")
                    fast_checked = True
                if m < 0:
                    gold[qi] = proj
                else:
                    rand[qi, m] = proj
            if (qi + 1) % 20 == 0:
                print(f"    seed {s}: {qi + 1}/{len(rows)}", flush=True)
        scores_by_seed[s] = assemble_scores(gold, rand)

    # ------------------------------------------------------------ artifacts
    prov = run_provenance()
    np.savez_compressed(
        args.out_scores,
        **{f"scores_seed{s}": scores_by_seed[s] for s in REGISTERED_SEEDS},
        **{f"query_ids_seed{s}": np.array(ids_by_seed[s]) for s in REGISTERED_SEEDS},
        **{f"folds_seed{s}": np.array(folds_by_seed[s]) for s in REGISTERED_SEEDS},
        # WHICH CALIBRATION WAS READ. The scores' projection targets come
        # from that file's header and nothing recorded which file it was, so
        # answering "did job 835618 use the right label rows" took a code
        # audit instead of a lookup. Path, file hash and header hash, because
        # a path alone stops being an answer as soon as the file changes.
        meta=json.dumps({"spec": "prereg_method_A.md section 2.4(2)",
                         "model": args.model, "method": args.method,
                         "dtype": args.dtype, "attn": args.attn,
                         "task": args.task, "K": args.K,
                         "query_manifest_sha256": qm_sha,
                         "uuid_jsonl": str(args.uuid_jsonl),
                         "uuid_jsonl_sha256": file_sha256(args.uuid_jsonl),
                         "uuid_header_sha256": hashlib.sha256(
                             json.dumps(header, sort_keys=True,
                                        ensure_ascii=False).encode("utf-8")
                         ).hexdigest(),
                         "label_space": str(args.label_space),
                         "label_space_sha256": file_sha256(args.label_space),
                         "n_random_variants": carrier_schema.N_RANDOM_VARIANTS,
                         "limited": bool(args.limit),
                         "provenance": prov}))
    fold_b, full_b = carrier_bundle.build_from_scores(
        scores_by_seed, ids_by_seed, folds_by_seed, model=args.model,
        query_manifest_sha256=qm_sha, n_layers=n_l, n_attn=n_h, n_kv=n_kv,
        task=args.task, K=args.K, carrier_impl=args.carrier_impl)
    for path, bundle, scope in ((args.out_fold, fold_b,
                                 carrier_bundle.SCOPE_FOLD),
                                (args.out_full, full_b,
                                 carrier_bundle.SCOPE_FULL)):
        bad = carrier_bundle.bundle_faults(
            bundle, scope=scope, validation_by_seed=ids_by_seed,
            fold_by_seed=folds_by_seed, query_manifest_sha256=qm_sha,
            model=args.model, n_validation=n_validation, task=args.task,
            K=args.K, arch=(n_l, n_h, n_kv))
        if bad:
            # A --limit run reaches here BY DESIGN: its folds hold fewer rows
            # than the registered count, so the bundle is refused and not
            # written. That is the intended outcome, not a defect -- a
            # limited run is not a registered discovery and its head sets
            # must not be usable anywhere. The scores npz above IS written,
            # which is what makes --limit useful for exercising the forward
            # loop without producing an artifact anything could pick up.
            note = (f"\n  [expected] --limit {args.limit} was given, so each "
                    f"fold holds {args.limit // 2} rows instead of the "
                    f"registered {n_validation // 2}. The scores WERE written "
                    f"to {args.out_scores}; only the bundles are refused, "
                    "because a limited run is not a registered discovery."
                    if args.limit else "")
            raise SystemExit(f"the {scope} bundle this run built is not "
                             "valid:\n  " + "\n  ".join(bad[:8]) + note)
        Path(path).write_text(json.dumps(bundle, indent=2,
                                         ensure_ascii=False),
                              encoding="utf-8")
        print(f"  [output] {path}  {carrier_bundle.summary(bundle)}")
    print(f"  commit {prov.get('repo_commit', '?')[:12]} "
          f"dirty={prov.get('repo_dirty')}")
    return 0


def prompt_ids(prompt, model):
    """Tokenise with the model's own tokenizer, on its device."""
    import torch
    from transformers import AutoTokenizer
    tok = getattr(prompt_ids, "_tok", None)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model.config._name_or_path)
        prompt_ids._tok = tok
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    return enc.input_ids.to(next(model.parameters()).device)


if __name__ == "__main__":
    raise SystemExit(main())
