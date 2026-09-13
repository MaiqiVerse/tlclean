"""Do a calibration file's prompts fit the model's context? CPU, tokenizer only.

Before a cell is queued: a K=10 banking77 prefix is 770 demonstrations and a
Llama-2 window is 4096 tokens, and the place that discovers this otherwise is
an hour into a GPU job. Reads the calibration jsonl, tokenises every prompt
with the model's own tokenizer, and compares the longest against the window:

    vanilla     config.max_position_embeddings
    SelfExtend  group_size * (window - neighbor_size) + neighbor_size,
                the furthest position the grouped attention can address
                (the SE paper's extension; group 4 / neighbour 1024 on a
                4096 window gives 13312)

Exit 0 when every prompt fits, 1 otherwise; --json-out records the numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def se_window(base_window, group_size, neighbor_size):
    return int(group_size) * (int(base_window) - int(neighbor_size)) + int(neighbor_size)


def prompt_lengths(path, tokenizer, limit=None):
    """[tokens per prompt] over the file's non-header rows."""
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if i == 0 and row.get("header"):
                continue
            text = row.get("prompt")
            if text is None:
                continue
            out.append(len(tokenizer(text, add_special_tokens=True).input_ids))
            if limit and len(out) >= limit:
                break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--path", required=True, help="calibration jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--selfextend", action="store_true")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--neighbor-size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0,
                    help="prompts to measure (0 = all; prefixes are shared, "
                         "so a few dozen already bound the maximum)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    base = int(getattr(cfg, "max_position_embeddings", 0) or 0)
    if base <= 0:
        raise SystemExit(f"{args.model}: config has no max_position_embeddings")
    window = (se_window(base, args.group_size, args.neighbor_size)
              if args.selfextend else base)
    lens = prompt_lengths(args.path, tok, args.limit or None)
    if not lens:
        raise SystemExit(f"{args.path}: no prompt rows")
    mx, mean = max(lens), sum(lens) / len(lens)
    fits = mx <= window
    print(f"  {args.path}")
    print(f"    prompts {len(lens)}, tokens max {mx}, mean {mean:.0f}; "
          f"window {window} ({'SelfExtend' if args.selfextend else 'vanilla'}"
          f", base {base})  -> {'fits' if fits else 'DOES NOT FIT'}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"path": args.path, "model": args.model, "n": len(lens),
             "max_tokens": mx, "mean_tokens": mean, "window": window,
             "base_window": base, "selfextend": bool(args.selfextend),
             "fits": fits}, indent=2), encoding="utf-8")
    return 0 if fits else 1


if __name__ == "__main__":
    raise SystemExit(main())
