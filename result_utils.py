"""
Shared result I/O for test_unified*.py scripts.

Design:
  1. Each run saves ONE atomic JSON file with a unique name encoding all params
  2. merge_results.py reads all files in a directory and produces summary tables

File naming: {script}_{method}_{task}_{shots}shot_seed{seed}[_extra].json
Directory:   {output_dir}/{script}/

JSON structure per file:
{
    "meta": {
        "script": "unified" | "tr" | "tl",
        "method": "baseline",
        "task": "trec_selfextend",
        "num_fewshot": 100,
        "seed": 42,
        "model": "meta-llama/Llama-2-7b-hf",
        "timestamp": "2026-04-20T10:30:00",
        "args": { ... full argparse namespace ... },
        # TR-specific
        "label_mode": "gold" | "random",
        # TL-specific
        "label_type": "uuid" | "number" | "letter",
    },
    "results": {
        "acc": 0.888,
        "acc_stderr": 0.020,
        ... other metrics from lm_eval ...
    }
}
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np


def build_result_path(output_dir, script, method, task, num_fewshot, seed, extra=None):
    """
    Build unique output path.
    
    Args:
        output_dir: base output directory
        script: "unified", "tr", "tl"
        method: "baseline", "selfextend", etc.
        task: "trec_selfextend", "clinc150_per_class", etc.
        num_fewshot: int
        seed: int
        extra: optional dict of extra keys, e.g. {"label_mode": "random"} or {"label_type": "uuid"}
    
    Returns:
        Path to JSON file
    """
    subdir = os.path.join(output_dir, script)
    os.makedirs(subdir, exist_ok=True)

    parts = [script, method, task, f"{num_fewshot}shot", f"seed{seed}"]
    if extra:
        for k, v in sorted(extra.items()):
            parts.append(f"{k}={v}")

    fname = "_".join(parts) + ".json"
    return os.path.join(subdir, fname)


def save_result(path, meta, results):
    """Save a single result atomically."""
    data = {
        "meta": meta,
        "results": results,
    }
    # Write to temp file then rename (atomic on same filesystem)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def build_meta(script, args, task, num_fewshot, seed, **extra_fields):
    """Build metadata dict from argparse namespace."""
    meta = {
        "script": script,
        "method": getattr(args, "method", "baseline"),
        "task": task,
        "num_fewshot": num_fewshot,
        "seed": seed,
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "args": {k: v for k, v in vars(args).items()
                 if k not in ("output_dir",)},
    }
    meta.update(extra_fields)
    return meta


def extract_acc(lm_eval_results, task_name):
    """Extract accuracy and stderr from lm_eval results dict."""
    task_res = lm_eval_results.get("results", {}).get(task_name, {})
    acc = task_res.get("acc,none", task_res.get("acc", None))
    stderr = task_res.get("acc_stderr,none", task_res.get("acc_stderr", None))
    return {"acc": acc, "acc_stderr": stderr, "raw": task_res}


# =========================================================================
# Merging / loading utilities
# =========================================================================

def load_all_results(result_dir):
    """Load all JSON result files from a directory (recursively)."""
    results = []
    for root, dirs, files in os.walk(result_dir):
        for f in sorted(files):
            if f.endswith(".json") and not f.endswith(".tmp"):
                path = os.path.join(root, f)
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                    if "meta" in data and "results" in data:
                        data["_path"] = path
                        results.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass
    return results


def group_results(results, group_keys=("script", "method", "task", "num_fewshot")):
    """Group results by specified meta keys. Returns dict of key_tuple → list of results."""
    groups = defaultdict(list)
    for r in results:
        key = tuple(r["meta"].get(k) for k in group_keys)
        groups[key] = groups.get(key, [])
        groups[key].append(r)
    return dict(groups)


def summarize_group(group):
    """Compute mean/std/min/max for a group of results."""
    accs = [r["results"]["acc"] for r in group
            if isinstance(r["results"].get("acc"), (int, float))]
    if not accs:
        return None
    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "min": float(min(accs)),
        "max": float(max(accs)),
        "n": len(accs),
        "seeds": [r["meta"]["seed"] for r in group],
        "per_seed": accs,
    }
