"""
Task Learning (TL) evaluation.
Following Pan et al. (ACL Findings 2023):
  - Demos use ABSTRACT labels (UUID/number/letter mapping)
  - Evaluation also uses ABSTRACT labels
  - Measures: can model learn new input-label mappings from context?

This removes Task Recognition: model can't recognize the task because
the labels are meaningless symbols. Any performance above random chance
must come from learning the input-label mapping in context.

Abstract label types:
  - uuid: 8-char hex strings (e.g., "a3f7c912") — default, needle-in-haystack precedent
  - number: integers (e.g., "1", "2", ...) — Pan et al.
  - letter: uppercase letters / letter combos (e.g., "A", "B", ..., "AA", "AB") — Pan et al.

Usage:
    python test_unified_task_learning.py \
        --task trec_selfextend --method baseline --num-fewshot 10 50 100

    python test_unified_task_learning.py \
        --task clinc150_per_class --method selfextend \
        --num-fewshot 2 5 --label-type uuid

    python test_unified_task_learning.py \
        --task trec_fine --label-type number --num-fewshot 50 100 200
"""
import argparse
import hashlib
import json
import logging
import os
import random
import uuid
from pathlib import Path
from typing import List

import numpy as np
import torch
import lm_eval
import lm_eval.tasks
import lm_eval.evaluator
from lm_eval.models.huggingface import HFLM

from test_unified import (
    load_model, build_lm_eval_model, run_eval,
    TASK_DIR, VALID_TASKS, VALID_METHODS, _parse_boundaries,
    ExampleAwareHFLM, AnchorHFLM,
)

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")


def generate_abstract_labels(n_classes: int, label_type: str, seed: int) -> List[str]:
    """Generate n_classes abstract label strings."""
    rng = random.Random(seed)

    if label_type == "uuid":
        # 8-char hex UUIDs, deterministic from seed
        labels = []
        for i in range(n_classes):
            # Deterministic UUID from seed+index
            h = hashlib.md5(f"{seed}-{i}".encode()).hexdigest()
            labels.append(h[:8])
        return labels

    elif label_type == "number":
        # Shuffled integers: "1", "2", ..., "N"
        labels = [str(i + 1) for i in range(n_classes)]
        rng.shuffle(labels)
        return labels

    elif label_type == "letter":
        # A, B, ..., Z, AA, AB, ..., AZ, BA, ...
        labels = []
        for i in range(n_classes):
            if i < 26:
                labels.append(chr(65 + i))
            else:
                labels.append(chr(65 + i // 26 - 1) + chr(65 + i % 26))
        rng.shuffle(labels)
        return labels

    raise ValueError(f"Unknown label_type: {label_type}")


def _get_task_info(task_name):
    """Return (label_field, original_label_names, format_fn_maker) for each task."""
    import sys
    sys.path.insert(0, TASK_DIR)

    base = task_name.replace("_per_class", "")

    if base == "trec_selfextend":
        from trec_selfextend_task import _LABEL_NAMES
        def make_fmt(mapping):
            return lambda doc, lab: f"Question: {doc['text']}\nType: {lab}"
        return "coarse_label", list(_LABEL_NAMES), make_fmt, \
            lambda doc: f"Question: {doc['text']}\nType:"

    elif base == "trec_fine":
        from trec_fine_task import _FINE_LABELS
        def make_fmt(mapping):
            return lambda doc, lab: f"Question: {doc['text']}\nType: {lab}"
        return "fine_label", list(_FINE_LABELS), make_fmt, \
            lambda doc: f"Question: {doc['text']}\nType:"

    elif base == "banking77":
        from banking77_task import _build_dataset
        _build_dataset()
        import banking77_task
        label_names = list(banking77_task._LABEL_NAMES)
        def make_fmt(mapping):
            return lambda doc, lab: f"Query: {doc['text']}\nIntent: {lab}"
        return "label", label_names, make_fmt, \
            lambda doc: f"Query: {doc['text']}\nIntent:"

    elif base == "clinc150":
        from clinc150_task import _build_dataset
        _build_dataset()
        import clinc150_task
        label_names = list(clinc150_task._LABEL_NAMES)
        def make_fmt(mapping):
            return lambda doc, lab: f"Query: {doc['text']}\nIntent: {lab}"
        return "label", label_names, make_fmt, \
            lambda doc: f"Query: {doc['text']}\nIntent:"

    elif base == "dbpedia14":
        from dbpedia14_task import _LABEL_NAMES
        def make_fmt(mapping):
            def fmt(doc, lab):
                title = doc["title"].strip()
                content = doc["content"].strip()
                if len(content) > 800:
                    content = content[:800] + "..."
                return f"Title: {title}\nDescription: {content}\nType: {lab}"
            return fmt
        def query_fn(doc):
            title = doc["title"].strip()
            content = doc["content"].strip()
            if len(content) > 800:
                content = content[:800] + "..."
            return f"Title: {title}\nDescription: {content}\nType:"
        return "label", list(_LABEL_NAMES), make_fmt, query_fn

    elif base == "yahoo_answers":
        from yahoo_answers_task import _LABEL_NAMES, _format_text
        def make_fmt(mapping):
            def fmt(doc, lab):
                text = _format_text(doc)
                if len(text) > 1200:
                    text = text[:1200] + "..."
                return f"Text: {text}\nTopic: {lab}"
            return fmt
        def query_fn(doc):
            from yahoo_answers_task import _format_text
            text = _format_text(doc)
            if len(text) > 1200:
                text = text[:1200] + "..."
            return f"Text: {text}\nTopic:"
        return "topic", list(_LABEL_NAMES), make_fmt, query_fn

    elif base == "yelp_full":
        from yelp_full_task import _LABEL_NAMES
        def make_fmt(mapping):
            def fmt(doc, lab):
                text = doc["text"].strip()
                if len(text) > 1200:
                    text = text[:1200] + "..."
                return f"Review: {text}\nRating: {lab}"
            return fmt
        def query_fn(doc):
            text = doc["text"].strip()
            if len(text) > 1200:
                text = text[:1200] + "..."
            return f"Review: {text}\nRating:"
        return "label", list(_LABEL_NAMES), make_fmt, query_fn

    elif base == "synthetic_linear":
        # Synthetic data: features are int indices into DEFAULT_FEATURE_POOL,
        # rendered via render_example. Demos use ``features`` and the top-level
        # test doc uses ``query_features`` (separate field; the query word
        # tuple is held out from the demo pool).
        from synthetic_linear_task import (
            DEFAULT_FEATURE_POOL, DEFAULT_TEMPLATE, _LABEL_NAMES, render_example,
        )
        _feature_keys = list(DEFAULT_FEATURE_POOL.keys())
        def make_fmt(mapping):
            def fmt(doc, lab):
                return render_example(
                    doc["features"], DEFAULT_FEATURE_POOL, lab,
                    DEFAULT_TEMPLATE, _feature_keys,
                )
            return fmt
        def query_fn(doc):
            return render_example(
                doc["query_features"], DEFAULT_FEATURE_POOL, None,
                DEFAULT_TEMPLATE, _feature_keys,
            )
        return "label", list(_LABEL_NAMES), make_fmt, query_fn

    elif base == "synthetic_mlp":
        # Dataset 3: integer features, numeric rendering; demos carry
        # "features", the test doc "query_features" (held out from the demos).
        from synthetic_mlp_task import _LABEL_NAMES, render_example as _render_mlp
        def make_fmt(mapping):
            def fmt(doc, lab):
                return _render_mlp(doc["features"], lab)
            return fmt
        def query_fn(doc):
            return _render_mlp(doc["query_features"], None)
        return "label", list(_LABEL_NAMES), make_fmt, query_fn

    elif base in ("monk", "monk_r1", "monk_r2", "monk_r3"):
        # Same label/feature schema across Monk-1/2/3 — only hidden concept differs.
        from monk_task import _LABEL_NAMES, render_example as _render
        def make_fmt(mapping):
            def fmt(doc, lab):
                return _render(doc["features"], lab)
            return fmt
        def query_fn(doc):
            return _render(doc["query_features"], None)
        return "label", list(_LABEL_NAMES), make_fmt, query_fn

    elif base in ("monk_bank_r1", "monk_bank_r2", "monk_bank_r3"):
        # The shared-bank form (tasks/monk_bank_task): ordinary train / test
        # rows, so the query too is rendered from `features`.
        from monk_task import _LABEL_NAMES, render_example as _render
        def make_fmt(mapping):
            def fmt(doc, lab):
                return _render(doc["features"], lab)
            return fmt
        def query_fn(doc):
            return _render(doc["features"], None)
        return "label", list(_LABEL_NAMES), make_fmt, query_fn

    elif base == "synthetic_linear_bank":
        # tasks/synthetic_linear_bank_task: one hidden W per task, the
        # per-prompt task's renderer, the query rendered from `features`.
        from synthetic_linear_task import (
            DEFAULT_FEATURE_POOL, DEFAULT_TEMPLATE, _LABEL_NAMES, render_example,
        )
        _feature_keys = list(DEFAULT_FEATURE_POOL.keys())
        def make_fmt(mapping):
            def fmt(doc, lab):
                return render_example(doc["features"], DEFAULT_FEATURE_POOL, lab,
                                      DEFAULT_TEMPLATE, _feature_keys)
            return fmt
        def query_fn(doc):
            return render_example(doc["features"], DEFAULT_FEATURE_POOL, None,
                                  DEFAULT_TEMPLATE, _feature_keys)
        return "label", list(_LABEL_NAMES), make_fmt, query_fn

    elif base == "synthetic_mlp_bank":
        # tasks/synthetic_mlp_bank_task: one hidden MLP per task.
        from synthetic_mlp_task import _LABEL_NAMES, render_example as _render_mlp
        def make_fmt(mapping):
            def fmt(doc, lab):
                return _render_mlp(doc["features"], lab)
            return fmt
        def query_fn(doc):
            return _render_mlp(doc["features"], None)
        return "label", list(_LABEL_NAMES), make_fmt, query_fn

    raise ValueError(f"Unknown task: {task_name}")


def _monkey_patch_abstract_labels(task_obj, task_name, abstract_labels, seed):
    """
    Monkey-patch the task to use abstract labels everywhere:
    - _build_fewshot_prefix: demos use abstract labels
    - doc_to_target: gold answer mapped to abstract label
    - doc_to_choice: all choices are abstract labels
    """
    label_field, original_labels, make_fmt, query_fn = _get_task_info(task_name)
    n_classes = len(original_labels)
    assert len(abstract_labels) == n_classes, \
        f"Expected {n_classes} abstract labels, got {len(abstract_labels)}"

    # Mapping: original label index → abstract label string
    # (original_labels[i] → abstract_labels[i])
    format_fn = make_fmt(abstract_labels)

    # Patch fewshot prefix
    def abstract_prefix():
        examples = task_obj._get_fewshot_examples()
        if not examples:
            return ""
        demos = []
        for ex in examples:
            label_idx = ex[label_field]
            demos.append(format_fn(ex, abstract_labels[label_idx]))
        return "\n\n".join(demos) + "\n\n"

    task_obj._build_fewshot_prefix = abstract_prefix

    # Patch doc_to_text (uses abstract prefix + original query format)
    def abstract_doc_to_text(doc, *args, **kwargs):
        # Synthetic tasks stash the current doc on the task so their
        # _get_fewshot_examples can return the per-doc demos. Replacing
        # doc_to_text bypasses that; restore it here. No-op for tasks
        # without a _current_doc slot.
        if hasattr(task_obj, "_current_doc"):
            task_obj._current_doc = doc
        prefix = abstract_prefix()
        return prefix + query_fn(doc)

    task_obj.doc_to_text = abstract_doc_to_text

    # Patch doc_to_target: map gold label to abstract
    def abstract_doc_to_target(doc, *args, **kwargs):
        label_idx = doc[label_field]
        return f" {abstract_labels[label_idx]}"

    task_obj.doc_to_target = abstract_doc_to_target

    # Patch doc_to_choice: all abstract labels
    abstract_choices = [f" {al}" for al in abstract_labels]

    def abstract_doc_to_choice(doc, *args, **kwargs):
        return abstract_choices

    task_obj.doc_to_choice = abstract_doc_to_choice

    return task_obj


def main():
    parser = argparse.ArgumentParser(description="Task Learning evaluation (abstract labels)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--task", type=str, nargs="+", default=["trec_selfextend"],
                        choices=VALID_TASKS)
    parser.add_argument("--method", type=str, default="baseline", choices=VALID_METHODS)
    parser.add_argument("--num-fewshot", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--label-type", type=str, default="uuid",
                        choices=["uuid", "number", "letter"],
                        help="Type of abstract labels")
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
        _logger.info(f"Task Learning: {task_name} (label_type={args.label_type})")
        _logger.info(f"{'='*70}")

        # Get task info for abstract label generation
        label_field, original_labels, _, _ = _get_task_info(task_name)
        n_classes = len(original_labels)
        _logger.info(f"  {n_classes} classes, label_type={args.label_type}")

        for nshot in args.num_fewshot:
            for seed in args.random_seed:
                # Generate abstract labels (deterministic from seed)
                abstract_labels = generate_abstract_labels(n_classes, args.label_type, seed)
                _logger.info(f"  Abstract labels (first 5): {abstract_labels[:5]}")

                # Normalize limit: 0 / negative / None all mean "full test set"
                effective_limit = args.limit if (args.limit is not None and args.limit > 0) else None

                # Create and patch task
                tm = lm_eval.tasks.TaskManager(include_path=TASK_DIR)
                task_dict = lm_eval.tasks.get_task_dict([task_name], task_manager=tm)
                task_obj = task_dict[task_name]
                task_obj.download()
                task_obj.set_fewshot(num_fewshot=nshot, seed=seed)
                if effective_limit is not None:
                    task_obj.set_config(key="limit", value=effective_limit)

                _monkey_patch_abstract_labels(task_obj, task_name, abstract_labels, seed)

                tag = f"{task_name}_abstract_{args.label_type}-{nshot}shot-seed{seed}"
                _logger.info(f"\n[ABSTRACT] {tag}")

                results = lm_eval.evaluator.evaluate(
                    lm=lm, task_dict={task_name: task_obj},
                    limit=effective_limit, log_samples=True,
                )
                acc = results["results"].get(task_name, {}).get("acc,none", "N/A")
                all_results[tag] = results["results"]

                # Save atomic result
                from result_utils import build_result_path, save_result, build_meta, extract_acc
                rpath = build_result_path(args.output_dir, "tl", args.method,
                                          task_name, nshot, seed,
                                          extra={"label_type": args.label_type})
                meta = build_meta("tl", args, task_name, nshot, seed,
                                  label_type=args.label_type, n_classes=n_classes)
                save_result(rpath, meta, extract_acc(results, task_name))
                _logger.info(f"  Saved → {rpath}")

                random_chance = 1.0 / n_classes
                if isinstance(acc, float):
                    above_chance = acc - random_chance
                    _logger.info(f"  acc={acc:.4f}  (chance={random_chance:.4f}, "
                                 f"above_chance={above_chance:+.4f}"
                                 f"{' ← TL working!' if above_chance > 0.02 else ' ← near chance'})")
                else:
                    _logger.info(f"  acc={acc}  (chance={random_chance:.4f})")

    # Save — merge with existing so separate invocations (e.g. one task per
    # call) accumulate rather than the last call overwriting earlier ones.
    out_path = os.path.join(args.output_dir, f"tl_{args.method}_{args.label_type}_results.json")
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
    _logger.info(f"Task Learning Summary (label_type={args.label_type})")
    _logger.info(f"{'='*70}")
    _logger.info(f"  {'Task':<30} {'Classes':>7} {'Chance':>7} {'Shots':>6} {'Acc':>8} {'Above':>8}")
    _logger.info(f"  {'-'*75}")

    for task_name in args.task:
        label_field, original_labels, _, _ = _get_task_info(task_name)
        n_classes = len(original_labels)
        chance = 1.0 / n_classes

        for nshot in args.num_fewshot:
            accs = []
            for seed in args.random_seed:
                tag = f"{task_name}_abstract_{args.label_type}-{nshot}shot-seed{seed}"
                a = all_results.get(tag, {}).get(task_name, {}).get("acc,none")
                if isinstance(a, float):
                    accs.append(a)
            if accs:
                mean_acc = np.mean(accs)
                above = mean_acc - chance
                _logger.info(f"  {task_name:<30} {n_classes:>7} {chance:>7.4f} {nshot:>6} "
                             f"{mean_acc:>8.4f} {above:>+8.4f}")


if __name__ == "__main__":
    main()
