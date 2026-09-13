"""
Task Recognition (TR) evaluation.
Following Pan et al. (ACL Findings 2023):
  - Demos use RANDOM labels (uniformly sampled from label space)
  - Evaluation uses GOLD labels
  - Measures: can model recognize task from format/distribution alone?

If Gold ≈ Random → TR-dominated (model doesn't need correct labels)
If Gold >> Random → TL-dominated (model relies on correct label mapping)

Usage:
    # Compare gold vs random on TREC-6
    python test_unified_task_recognition.py \
        --task trec_selfextend --method baseline --num-fewshot 10 50 100

    # Per-class balanced random labels
    python test_unified_task_recognition.py \
        --task trec_fine_per_class --method selfextend --num-fewshot 2 5

    # Compare across datasets
    python test_unified_task_recognition.py \
        --task trec_selfextend clinc150 --method baseline --num-fewshot 50 100
"""
import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import lm_eval
import lm_eval.tasks
import lm_eval.evaluator
from lm_eval.models.huggingface import HFLM

# Reuse model loading and eval infrastructure from test_unified
from test_unified import (
    load_model, build_lm_eval_model, run_eval,
    TASK_DIR, VALID_TASKS, VALID_METHODS, _parse_boundaries,
    ExampleAwareHFLM, AnchorHFLM,
)

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")


def _get_label_field_and_names(task_name):
    """Return (label_field, label_names_list, format_fn) for each task."""
    import sys
    sys.path.insert(0, TASK_DIR)

    base = task_name.replace("_per_class", "")

    if base == "trec_selfextend":
        from trec_selfextend_task import _LABEL_NAMES
        return "coarse_label", _LABEL_NAMES, \
            lambda doc, lab: f"Question: {doc['text']}\nType: {lab}"

    elif base == "trec_fine":
        from trec_fine_task import _FINE_LABELS
        return "fine_label", _FINE_LABELS, \
            lambda doc, lab: f"Question: {doc['text']}\nType: {lab}"

    elif base == "banking77":
        from banking77_task import _build_dataset, _LABEL_NAMES
        if _LABEL_NAMES is None:
            _build_dataset()
            import banking77_task
            return "label", banking77_task._LABEL_NAMES, \
                lambda doc, lab: f"Query: {doc['text']}\nIntent: {lab}"
        return "label", _LABEL_NAMES, \
            lambda doc, lab: f"Query: {doc['text']}\nIntent: {lab}"

    elif base == "clinc150":
        from clinc150_task import _build_dataset, _LABEL_NAMES
        if _LABEL_NAMES is None:
            _build_dataset()
            import clinc150_task
            return "label", clinc150_task._LABEL_NAMES, \
                lambda doc, lab: f"Query: {doc['text']}\nIntent: {lab}"
        return "label", _LABEL_NAMES, \
            lambda doc, lab: f"Query: {doc['text']}\nIntent: {lab}"

    elif base == "dbpedia14":
        from dbpedia14_task import _LABEL_NAMES   # no _format_text there (job 844471); the formatter is inline below
        # dbpedia has title+content
        def fmt(doc, lab):
            title = doc["title"].strip()
            content = doc["content"].strip()
            if len(content) > 800:
                content = content[:800] + "..."
            return f"Title: {title}\nDescription: {content}\nType: {lab}"
        return "label", _LABEL_NAMES, fmt

    elif base == "yahoo_answers":
        from yahoo_answers_task import _LABEL_NAMES, _format_text
        def fmt(doc, lab):
            text = _format_text(doc)
            if len(text) > 1200:
                text = text[:1200] + "..."
            return f"Text: {text}\nTopic: {lab}"
        return "topic", _LABEL_NAMES, fmt

    elif base == "yelp_full":
        from yelp_full_task import _LABEL_NAMES
        def fmt(doc, lab):
            text = doc["text"].strip()
            if len(text) > 1200:
                text = text[:1200] + "..."
            return f"Review: {text}\nRating: {lab}"
        return "label", _LABEL_NAMES, fmt

    elif base == "synthetic_linear":
        # Demos and top-level test docs both expose ``label`` (the demo's
        # ``label_idx`` field was renamed to ``label`` so this works uniformly).
        # The format_fn rebuilds NL text from the demo's feature indices.
        from synthetic_linear_task import (
            DEFAULT_FEATURE_POOL, DEFAULT_TEMPLATE, _LABEL_NAMES, render_example,
        )
        _feature_keys = list(DEFAULT_FEATURE_POOL.keys())
        def fmt(doc, lab):
            return render_example(
                doc["features"], DEFAULT_FEATURE_POOL, lab,
                DEFAULT_TEMPLATE, _feature_keys,
            )
        return "label", list(_LABEL_NAMES), fmt

    elif base == "synthetic_mlp":
        # Dataset 3: integer features rendered numerically ("Feature: 3 7 1 9 2").
        # Demos and test docs share "label" / "features" like Dataset 1.
        from synthetic_mlp_task import _LABEL_NAMES, render_example as _render_mlp
        def fmt(doc, lab):
            return _render_mlp(doc["features"], lab)
        return "label", list(_LABEL_NAMES), fmt

    elif base in ("monk", "monk_r1", "monk_r2", "monk_r3"):
        # Monk: 2 classes (A/B), 6 mixed-arity categorical features rendered
        # via robot-face template. Demo / test doc share "label" + "features".
        # The "_r{N}" variants (rule_id=N) share the same label/feature schema
        # — only the hidden Boolean concept differs.
        from monk_task import _LABEL_NAMES, render_example as _render
        def fmt(doc, lab):
            return _render(doc["features"], lab)
        return "label", list(_LABEL_NAMES), fmt

    elif base in ("monk_bank_r1", "monk_bank_r2", "monk_bank_r3"):
        # The shared-bank form (tasks/monk_bank_task); same schema as monk.
        from monk_task import _LABEL_NAMES, render_example as _render
        def fmt(doc, lab):
            return _render(doc["features"], lab)
        return "label", list(_LABEL_NAMES), fmt

    elif base == "synthetic_linear_bank":
        from synthetic_linear_task import (
            DEFAULT_FEATURE_POOL, DEFAULT_TEMPLATE, _LABEL_NAMES, render_example,
        )
        _feature_keys = list(DEFAULT_FEATURE_POOL.keys())
        def fmt(doc, lab):
            return render_example(doc["features"], DEFAULT_FEATURE_POOL, lab,
                                  DEFAULT_TEMPLATE, _feature_keys)
        return "label", list(_LABEL_NAMES), fmt

    elif base == "synthetic_mlp_bank":
        from synthetic_mlp_task import _LABEL_NAMES, render_example as _render_mlp
        def fmt(doc, lab):
            return _render_mlp(doc["features"], lab)
        return "label", list(_LABEL_NAMES), fmt

    raise ValueError(f"Unknown task: {task_name}")


def _monkey_patch_random_labels(task_obj, task_name, seed):
    """
    Monkey-patch the task's _build_fewshot_prefix to use random labels.
    Only affects demos. doc_to_target and doc_to_choice remain gold.
    """
    label_field, label_names, format_fn = _get_label_field_and_names(task_name)
    n_classes = len(label_names)

    original_build_prefix = task_obj._build_fewshot_prefix

    def random_label_prefix():
        examples = task_obj._get_fewshot_examples()
        if not examples:
            return ""
        rng = random.Random(seed + 12345)  # different from fewshot sampling seed
        demos = []
        for ex in examples:
            # Random label: uniformly sample from label space
            random_label = label_names[rng.randint(0, n_classes - 1)]
            demos.append(format_fn(ex, random_label))
        return "\n\n".join(demos) + "\n\n"

    task_obj._build_fewshot_prefix = random_label_prefix
    return task_obj


def main():
    parser = argparse.ArgumentParser(description="Task Recognition evaluation (random labels)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--task", type=str, nargs="+", default=["trec_selfextend"],
                        choices=VALID_TASKS)
    parser.add_argument("--method", type=str, default="baseline", choices=VALID_METHODS)
    parser.add_argument("--num-fewshot", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2")
    parser.add_argument("--random-seed", type=int, nargs="+", default=[42])
    parser.add_argument("--output-dir", type=str, default="./results")
    # Method-specific params
    parser.add_argument("--rope-factor", type=float, default=None)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--n-parts", type=int, default=-1)
    parser.add_argument("--neighbor-examples", type=int, default=10)
    parser.add_argument("--compress-test", action="store_true")
    parser.add_argument("--offset-mode", type=str, default="dynamic")
    # Structural-Anchor SelfExtend params
    parser.add_argument("--k-marker-tail", type=int, default=4)
    parser.add_argument("--k-marker-head", type=int, default=0)
    parser.add_argument("--disable-anchor", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    rope_factor = args.rope_factor if args.rope_factor is not None else (1.0 if args.method == "dynamic_ntk" else 4.0)
    model, max_length, method_config = load_model(
        args.model, method=args.method, dtype=args.dtype,
        attn_implementation=args.attn_impl, rope_factor=rope_factor,
        group_size=args.group_size, window_size=args.window_size,
        n_parts=args.n_parts, neighbor_examples=args.neighbor_examples,
        compress_test=args.compress_test, offset_mode=args.offset_mode,
        k_marker_tail=args.k_marker_tail, k_marker_head=args.k_marker_head,
        enable_anchor=not args.disable_anchor,
    )
    ea_kw = method_config if args.method == "example_aware_selfextend" else None
    anchor_kw = method_config if args.method == "structural_anchor" else None
    lm = build_lm_eval_model(model, args.model, args.batch_size, max_length,
                              ea_config=ea_kw, anchor_config=anchor_kw)

    all_results = {}

    for task_name in args.task:
        _logger.info(f"\n{'='*70}")
        _logger.info(f"Task Recognition: {task_name}")
        _logger.info(f"{'='*70}")

        for nshot in args.num_fewshot:
            for seed in args.random_seed:
                # ---- RANDOM labels only ----
                # Gold results come from test_unified.py (same model, same demos, same eval)
                # No need to re-run gold here.

                # Normalize limit: 0 / negative / None all mean "full test set"
                effective_limit = args.limit if (args.limit is not None and args.limit > 0) else None

                tm = lm_eval.tasks.TaskManager(include_path=TASK_DIR)
                task_dict = lm_eval.tasks.get_task_dict([task_name], task_manager=tm)
                task_obj = task_dict[task_name]
                task_obj.download()
                task_obj.set_fewshot(num_fewshot=nshot, seed=seed)
                if effective_limit is not None:
                    task_obj.set_config(key="limit", value=effective_limit)

                _monkey_patch_random_labels(task_obj, task_name, seed)

                random_tag = f"{task_name}_random-{nshot}shot-seed{seed}"
                _logger.info(f"\n[RANDOM] {random_tag}")

                results_random = lm_eval.evaluator.evaluate(
                    lm=lm, task_dict={task_name: task_obj},
                    limit=effective_limit, log_samples=True,
                )
                random_acc = results_random["results"].get(task_name, {}).get("acc,none", "N/A")
                all_results[random_tag] = results_random["results"]

                # Save atomic result
                from result_utils import build_result_path, save_result, build_meta, extract_acc
                rpath = build_result_path(args.output_dir, "tr", args.method,
                                          task_name, nshot, seed,
                                          extra={"label_mode": "random"})
                meta = build_meta("tr", args, task_name, nshot, seed, label_mode="random")
                save_result(rpath, meta, extract_acc(results_random, task_name))
                _logger.info(f"  Saved → {rpath}")

                # Try to load corresponding gold result from unified/
                gold_acc = None
                gold_path = build_result_path(args.output_dir, "unified", args.method,
                                              task_name, nshot, seed)
                if os.path.exists(gold_path):
                    import json as _json
                    with open(gold_path) as _f:
                        gold_data = _json.load(_f)
                    gold_acc = gold_data.get("results", {}).get("acc")

                if gold_acc is not None and isinstance(random_acc, float):
                    gap = gold_acc - random_acc
                    _logger.info(f"  Gold={gold_acc:.4f} (from unified)  Random={random_acc:.4f}  "
                                 f"Gap={gap:.4f} ({'TR-dominated' if abs(gap) < 0.05 else 'TL-needed'})")
                elif isinstance(random_acc, float):
                    _logger.info(f"  Random={random_acc:.4f}  (run test_unified.py first for gold comparison)")
                else:
                    _logger.info(f"  Random={random_acc}")

    # Save — merge with existing so separate invocations (e.g. one task per
    # call) accumulate rather than the last call overwriting earlier ones.
    out_path = os.path.join(args.output_dir, f"tr_{args.method}_results.json")
    existing = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.update(all_results)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    _logger.info(f"\nResults saved to {out_path} "
                 f"({len(all_results)} new, {len(existing)} total)")

    # Summary
    _logger.info(f"\n{'='*70}")
    _logger.info(f"Task Recognition Summary")
    _logger.info(f"  Gold results from: {args.output_dir}/unified/")
    _logger.info(f"{'='*70}")
    _logger.info(f"  {'Task':<30} {'Shots':>6} {'Gold':>8} {'Random':>8} {'Gap':>8} {'Verdict'}")
    _logger.info(f"  {'-'*80}")
    for task_name in args.task:
        for nshot in args.num_fewshot:
            golds, randoms = [], []
            for seed in args.random_seed:
                # Random from this run
                r_tag = f"{task_name}_random-{nshot}shot-seed{seed}"
                r = all_results.get(r_tag, {}).get(task_name, {}).get("acc,none")
                if isinstance(r, float):
                    randoms.append(r)
                # Gold from unified results on disk
                from result_utils import build_result_path
                gold_path = build_result_path(args.output_dir, "unified", args.method,
                                              task_name, nshot, seed)
                if os.path.exists(gold_path):
                    with open(gold_path) as _f:
                        gd = json.load(_f)
                    g = gd.get("results", {}).get("acc")
                    if isinstance(g, float):
                        golds.append(g)
            if randoms:
                mr = np.mean(randoms)
                if golds:
                    mg = np.mean(golds)
                    gap = mg - mr
                    verdict = "TR-dom" if abs(gap) < 0.05 else "TL-dep"
                    _logger.info(f"  {task_name:<30} {nshot:>6} {mg:>8.4f} {mr:>8.4f} {gap:>+8.4f} {verdict}")
                else:
                    _logger.info(f"  {task_name:<30} {nshot:>6} {'N/A':>8} {mr:>8.4f} {'N/A':>8} (run unified first)")


if __name__ == "__main__":
    main()