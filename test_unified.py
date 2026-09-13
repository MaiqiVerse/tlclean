"""
Unified evaluation script for ICL classification tasks.
Supports: trec_selfextend (TREC 6-class), trec_fine (50-class), banking77 (77-class), clinc150 (150-class)
Supports: baseline (vanilla), dynamic NTK, selfextend, example-aware selfextend

Now supports multiple random seeds for measuring ordering sensitivity.

Usage:
    # Single seed (backward compatible)
    python test_unified.py --task banking77 --method baseline --num-fewshot 0 10 50 100 200 400

    # Multiple seeds
    python test_unified.py --task banking77 --method baseline --num-fewshot 100 200 400 --random-seed 42 123 456

    # 5 seeds for ordering sensitivity experiment
    python test_unified.py --task clinc150 --method example_aware_selfextend \
        --num-fewshot 100 200 300 --random-seed 42 123 456 789 1024

    # Dynamic NTK on TREC-fine
    python test_unified.py --task trec_fine --method dynamic_ntk --rope-factor 1.0

    # SelfExtend on CLINC150
    python test_unified.py --task clinc150 --method selfextend --group-size 8 --window-size 1024

    # All tasks baseline comparison
    python test_unified.py --task trec_selfextend trec_fine banking77 clinc150 --method baseline
"""
import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import lm_eval
import lm_eval.tasks
import lm_eval.evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

TASK_DIR = str(Path(__file__).parent / "tasks")

VALID_TASKS = ["trec_selfextend", "trec_fine", "banking77", "clinc150",
               "dbpedia14", "yahoo_answers", "yelp_full",
               "trec_selfextend_per_class", "trec_fine_per_class",
               "banking77_per_class", "clinc150_per_class",
               "dbpedia14_per_class", "yahoo_answers_per_class", "yelp_full_per_class",
               "synthetic_linear", "synthetic_linear_per_class",
               "monk", "monk_per_class",
               "monk_per_class_r1", "monk_per_class_r2", "monk_per_class_r3",
               "monk_bank_r1_per_class", "monk_bank_r2_per_class",
               "monk_bank_r3_per_class", "synthetic_linear_bank_per_class",
               "synthetic_mlp_bank_per_class"]
VALID_METHODS = ["baseline", "dynamic_ntk", "selfextend", "yarn", "linear",
                 "example_aware_selfextend", "structural_anchor"]

DELIMITER = "\n\n"


def load_model(
    model_name: str,
    method: str = "baseline",
    dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    device_map: str = "auto",
    # Dynamic NTK params
    rope_factor: float = 1.0,
    # SelfExtend params
    group_size: int = 8,
    window_size: int = 1024,
    # Example-Aware SelfExtend params
    n_parts: int = -1,
    neighbor_examples: int = 10,
    compress_test: bool = False,
    offset_mode: str = "dynamic",
    # Structural-Anchor params
    k_marker_tail: int = 4,
    k_marker_head: int = 0,
    enable_anchor: bool = True,
):
    torch_dtype = getattr(torch, dtype)

    if method == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation, device_map=device_map,
        )
        max_length = 5000 * 100

    elif method == "yarn":
        config = AutoConfig.from_pretrained(model_name)
        config.rope_scaling = {"type": "yarn", "factor": rope_factor}
        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation, device_map=device_map,
        )
        max_length = 5000 * 100

    elif method == "dynamic_ntk":
        config = AutoConfig.from_pretrained(model_name)
        config.rope_scaling = {"type": "dynamic", "factor": rope_factor,
                                "original_max_position_embeddings": getattr(config, "max_position_embeddings", 4096),
                                "attention_factor": None,
                                "beta_fast": 32, "beta_slow": 1}
        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation, device_map=device_map,
        )
        max_length = 50000 * 100

    elif method == "linear":
        config = AutoConfig.from_pretrained(model_name)
        config.rope_scaling = {"type": "linear", "factor": rope_factor}
        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation, device_map=device_map,
        )
        max_length = 5000 * 100

    elif method == "selfextend":
        if attn_implementation == "flash_attention_2":
            from models.selfExtend.llama2_flash import LlamaForCausalLM
        else:
            from models.selfExtend.llama2 import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation, device_map=device_map,
        )
        for layer in model.model.layers:
            layer.self_attn.group_size = group_size
            layer.self_attn.neighbor_size = window_size

        se_max_pos = getattr(model.config, "max_position_embeddings", 4096)
        max_length = (se_max_pos - window_size) * group_size + window_size
        max_length = max_length * 100

    elif method == "example_aware_selfextend":
        from models.selfextendExample.llama2_example_aware_v3 import LlamaForCausalLM
        from models.selfextendExample.example_aware_selfextend_v3 import (
            ExampleAwareSelfExtendConfig,
        )

        model = LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation, device_map=device_map,
        )

        ea_config = ExampleAwareSelfExtendConfig(
            n_parts=n_parts,
            neighbor_examples=neighbor_examples,
            compress_test=compress_test,
            offset_mode=offset_mode,
        )
        model.set_example_aware_config(ea_config)

        max_length = 5000 * 100

        return model, max_length, ea_config

    elif method == "structural_anchor":
        from models.selfExtendAnchor.llama2_anchor import LlamaForCausalLM
        from models.selfExtendAnchor.anchor_config import AnchorConfig

        model = LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation, device_map=device_map,
        )

        anchor_config = AnchorConfig(
            group_size=group_size,
            window_size=window_size,
            k_marker_tail=k_marker_tail,
            k_marker_head=k_marker_head,
            enable_anchor=enable_anchor,
        )
        model.set_anchor_config(anchor_config)

        sas_max_pos = getattr(model.config, "max_position_embeddings", 4096)
        max_length = (sas_max_pos - window_size) * group_size + window_size
        max_length = max_length * 100

        return model, max_length, anchor_config

    else:
        raise ValueError(f"Unknown method: {method}")

    return model, max_length, None


def _parse_boundaries(tokenizer, input_ids_1d):
    """Parse example boundaries from tokenized input_ids."""
    text = tokenizer.decode(input_ids_1d, skip_special_tokens=False)
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors=None)
    offsets = encoding["offset_mapping"]

    bos_token = tokenizer.bos_token or ""
    bos_offset = len(bos_token) if text.startswith(bos_token) else 0
    text_no_bos = text[bos_offset:]

    parts = text_no_bos.split(DELIMITER)
    char_starts = []
    pos = bos_offset
    for i, part in enumerate(parts):
        if part:
            char_starts.append(pos)
        pos += len(part)
        if i < len(parts) - 1:
            pos += len(DELIMITER)

    if not char_starts:
        return [0], [len(input_ids_1d)]

    first_content = 0
    for tid, (cs, ce) in enumerate(offsets):
        if cs != ce:
            first_content = tid
            break

    split_indices = [first_content]
    for ex_idx in range(1, len(char_starts)):
        target_char = char_starts[ex_idx]
        split_tok = len(offsets)
        for tid in range(split_indices[-1], len(offsets)):
            cs, ce = offsets[tid]
            if cs == ce:
                continue
            if cs >= target_char:
                split_tok = tid
                break
        split_indices.append(split_tok)

    example_starts = []
    example_lengths = []
    for i in range(len(split_indices)):
        tok_start = split_indices[i]
        tok_end = split_indices[i + 1] if i + 1 < len(split_indices) else len(input_ids_1d)
        example_starts.append(tok_start)
        example_lengths.append(tok_end - tok_start)

    return example_starts, example_lengths


class ExampleAwareHFLM(HFLM):
    """HFLM subclass that injects example boundaries into EA model forward."""

    def __init__(self, ea_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ea_config = ea_config

    def _model_call(self, inps, attn_mask=None, labels=None):
        if inps.shape[0] == 1:
            input_ids_1d = inps[0]
            if self.tokenizer.pad_token_id is not None:
                real_len = (input_ids_1d != self.tokenizer.pad_token_id).sum().item()
            else:
                real_len = input_ids_1d.shape[0]
            real_ids = input_ids_1d[:real_len]

            example_starts, example_lengths = _parse_boundaries(self.tokenizer, real_ids)

            with torch.no_grad():
                output = self.model(
                    input_ids=inps, attention_mask=attn_mask,
                    example_starts=example_starts, example_lengths=example_lengths,
                    ea_config=self.ea_config,
                )
            return output.logits
        else:
            _logger.warning("Batch size > 1, falling back to standard forward")
            with torch.no_grad():
                return self.model(input_ids=inps, attention_mask=attn_mask).logits


class AnchorHFLM(HFLM):
    """HFLM subclass that forwards example boundaries + AnchorConfig to the
    SAS model so it can build canonical-anchor far-path positions per prompt."""

    def __init__(self, anchor_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_config = anchor_config

    def _model_call(self, inps, attn_mask=None, labels=None):
        if inps.shape[0] == 1:
            input_ids_1d = inps[0]
            if self.tokenizer.pad_token_id is not None:
                real_len = (input_ids_1d != self.tokenizer.pad_token_id).sum().item()
            else:
                real_len = input_ids_1d.shape[0]
            real_ids = input_ids_1d[:real_len]

            example_starts, example_lengths = _parse_boundaries(self.tokenizer, real_ids)

            with torch.no_grad():
                output = self.model(
                    input_ids=inps, attention_mask=attn_mask,
                    example_starts=example_starts, example_lengths=example_lengths,
                    anchor_config=self.anchor_config,
                )
            return output.logits
        else:
            _logger.warning("Batch size > 1, falling back to standard forward")
            with torch.no_grad():
                return self.model(input_ids=inps, attention_mask=attn_mask).logits


def build_lm_eval_model(model, model_name: str, batch_size: int = 1,
                        max_length: int = None, ea_config=None,
                        anchor_config=None) -> HFLM:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {}
    if max_length is not None:
        kwargs["max_length"] = max_length

    if ea_config is not None:
        return ExampleAwareHFLM(
            ea_config=ea_config,
            pretrained=model, tokenizer=tokenizer,
            batch_size=batch_size, **kwargs,
        )

    if anchor_config is not None:
        return AnchorHFLM(
            anchor_config=anchor_config,
            pretrained=model, tokenizer=tokenizer,
            batch_size=batch_size, **kwargs,
        )

    return HFLM(
        pretrained=model, tokenizer=tokenizer,
        batch_size=batch_size, **kwargs,
    )


def run_eval(
    lm: HFLM,
    task_name: str,
    num_fewshot: int = 0,
    limit: Optional[int] = 250,
    random_seed: int = 42,
) -> dict:
    # limit=0 or limit<0 means "no limit" (full test set)
    if limit is not None and limit <= 0:
        limit = None

    tm = lm_eval.tasks.TaskManager(include_path=TASK_DIR)
    task_dict = lm_eval.tasks.get_task_dict([task_name], task_manager=tm)
    task = task_dict[task_name]

    task.download()
    task.set_fewshot(num_fewshot=num_fewshot, seed=random_seed)
    if limit is not None:
        task.set_config(key="limit", value=limit)

    results = lm_eval.evaluator.evaluate(
        lm=lm, task_dict={task_name: task},
        limit=limit, log_samples=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Unified ICL evaluation")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--task", type=str, nargs="+", default=["trec_fine", "banking77", "clinc150"],
                        choices=VALID_TASKS, help="Task(s) to evaluate")
    parser.add_argument("--method", type=str, default="baseline",
                        choices=VALID_METHODS, help="Method to use")
    parser.add_argument("--num-fewshot", type=int, nargs="+",
                        default=[0, 10, 50, 100, 200, 300, 400])
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2",
                        choices=["eager", "flash_attention_2", "sdpa"])
    parser.add_argument("--random-seed", type=int, nargs="+", default=[42],
                        help="Random seed(s) for fewshot sampling. "
                             "Multiple seeds enable ordering sensitivity measurement.")
    parser.add_argument("--output-dir", type=str, default="./results")
    # Dynamic NTK params
    parser.add_argument("--rope-factor", type=float, default=None)
    # SelfExtend params
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=1024)
    # Example-Aware SelfExtend params
    parser.add_argument("--n-parts", type=int, default=-1,
                        help="n_parts for EA. Use -1 for auto")
    parser.add_argument("--neighbor-examples", type=int, default=10,
                        help="Number of neighbor examples for EA")
    parser.add_argument("--compress-test", action="store_true",
                        help="Compress TEST tokens in EA")
    parser.add_argument("--offset-mode", type=str, default="dynamic",
                        choices=["dynamic", "avg", "max"],
                        help="Offset mode for EA")
    # Structural-Anchor SelfExtend params
    parser.add_argument("--k-marker-tail", type=int, default=4,
                        help="# of tokens at END of each example treated as markers")
    parser.add_argument("--k-marker-head", type=int, default=0,
                        help="# of tokens at START of each example treated as markers")
    parser.add_argument("--disable-anchor", action="store_true",
                        help="If set with --method structural_anchor, run as plain SE (ablation)")
    args = parser.parse_args()

    print(f"Running with args: {args}")

    multi_seed = len(args.random_seed) > 1
    if multi_seed:
        _logger.info(f"Multi-seed mode: {len(args.random_seed)} seeds = {args.random_seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- per-seed results: tag → {task_name: {metric: value}} ----
    all_results = {}

    # ---- aggregated across seeds: (task, nshot) → [acc_seed1, acc_seed2, ...] ----
    agg_results = defaultdict(list)

    # Load model once
    _logger.info(f"Loading model: {args.model} (method={args.method}, attn={args.attn_impl})")
    if args.method == "dynamic_ntk" and args.rope_factor is None:
        rope_factor = 1.0
    else:
        rope_factor = args.rope_factor if args.rope_factor is not None else 4.0
    model, max_length, method_config = load_model(
        args.model, method=args.method, dtype=args.dtype,
        attn_implementation=args.attn_impl,
        rope_factor=rope_factor,
        group_size=args.group_size, window_size=args.window_size,
        n_parts=args.n_parts, neighbor_examples=args.neighbor_examples,
        compress_test=args.compress_test, offset_mode=args.offset_mode,
        k_marker_tail=args.k_marker_tail, k_marker_head=args.k_marker_head,
        enable_anchor=not args.disable_anchor,
    )
    _logger.info(f"  max_length = {max_length}")

    ea_kw, anchor_kw = None, None
    if args.method == "example_aware_selfextend":
        ea_kw = method_config
    elif args.method == "structural_anchor":
        anchor_kw = method_config
    lm = build_lm_eval_model(model, args.model, args.batch_size, max_length,
                              ea_config=ea_kw, anchor_config=anchor_kw)

    for task_name in args.task:
        _logger.info(f"\n{'='*70}")
        _logger.info(f"Task: {task_name}")
        is_per_class = task_name.endswith("_per_class")
        if is_per_class:
            _logger.info(f"  Mode: k-per-class (--num-fewshot = k per class)")
        _logger.info(f"{'='*70}")

        for nshot in args.num_fewshot:
            for seed in args.random_seed:
                tag = f"{task_name}_{args.method}-{nshot}shot-seed{seed}"
                _logger.info(f"\nRunning {tag}")

                results = run_eval(lm, task_name, num_fewshot=nshot,
                                 limit=args.limit, random_seed=seed)

                task_results = results["results"].get(task_name, {})
                acc = task_results.get("acc,none", task_results.get("acc", "N/A"))
                stderr = task_results.get("acc_stderr,none", "")
                _logger.info(f"[{tag}] acc = {acc}  (stderr={stderr})")

                all_results[tag] = results["results"]

                # Save atomic result file
                from result_utils import build_result_path, save_result, build_meta, extract_acc
                rpath = build_result_path(args.output_dir, "unified", args.method,
                                          task_name, nshot, seed)
                meta = build_meta("unified", args, task_name, nshot, seed)
                save_result(rpath, meta, extract_acc(results, task_name))
                _logger.info(f"  Saved → {rpath}")

                # Collect for aggregation
                if isinstance(acc, (int, float)):
                    agg_results[(task_name, nshot)].append(acc)

            # Log per-shot aggregation immediately if multi-seed
            if multi_seed:
                key = (task_name, nshot)
                accs = agg_results[key]
                if accs:
                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
                    _logger.info(f"  [{task_name} {nshot}-shot] "
                                 f"mean={mean_acc:.4f} ± {std_acc:.4f}  "
                                 f"(n={len(accs)}, min={min(accs):.4f}, max={max(accs):.4f})")

    # ---- Build aggregated summary ----
    agg_summary = {}
    for (task_name, nshot), accs in agg_results.items():
        key = f"{task_name}_{args.method}-{nshot}shot"
        agg_summary[key] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "min": float(min(accs)),
            "max": float(max(accs)),
            "n_seeds": len(accs),
            "seeds": args.random_seed[:len(accs)],
            "per_seed": accs,
        }

    # ---- Save ----
    # Merge with existing bulk file so separate invocations (e.g. one task per
    # call) accumulate, rather than the last call overwriting earlier ones.
    out_path = os.path.join(args.output_dir, f"unified_{args.method}_results.json")
    existing = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.update(all_results)  # new keys win on collision (re-run with same tag)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    _logger.info(f"\nPer-seed results saved to {out_path} "
                 f"({len(all_results)} new, {len(existing)} total)")

    if multi_seed:
        agg_path = os.path.join(args.output_dir, f"unified_{args.method}_aggregated.json")
        existing_agg = {}
        if os.path.exists(agg_path):
            try:
                with open(agg_path) as f:
                    existing_agg = json.load(f)
                if not isinstance(existing_agg, dict):
                    existing_agg = {}
            except (json.JSONDecodeError, OSError):
                existing_agg = {}
        existing_agg.update(agg_summary)
        with open(agg_path, "w") as f:
            json.dump(existing_agg, f, indent=2, default=str)
        _logger.info(f"Aggregated results saved to {agg_path} "
                     f"({len(agg_summary)} new, {len(existing_agg)} total)")

    # ---- Summary Table ----
    _logger.info(f"\n{'='*70}")
    _logger.info(f"Summary (method={args.method}, seeds={args.random_seed})")
    _logger.info(f"{'='*70}")

    for task_name in args.task:
        _logger.info(f"\n  Task: {task_name}")

        if multi_seed:
            _logger.info(f"  {'Shots':>8}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'Per-seed accuracies'}")
            _logger.info(f"  {'-'*80}")
        else:
            _logger.info(f"  {'Shots':>8}  {'Accuracy':>10}")
            _logger.info(f"  {'-'*20}")

        for nshot in args.num_fewshot:
            key = (task_name, nshot)
            accs = agg_results.get(key, [])

            if not accs:
                _logger.info(f"  {nshot:>8}  {'N/A':>10}")
                continue

            if multi_seed:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
                per_seed_str = "  ".join(f"{a:.3f}" for a in accs)
                _logger.info(f"  {nshot:>8}  {mean_acc:>8.4f}  {std_acc:>8.4f}  "
                             f"{min(accs):>8.4f}  {max(accs):>8.4f}  [{per_seed_str}]")
            else:
                _logger.info(f"  {nshot:>8}  {accs[0]:>10.4f}")


if __name__ == "__main__":
    main()