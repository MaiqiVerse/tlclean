"""ONE home for the model tags every driver spells. MTAG -> model, method.

Two shell scripts and two tools used to carry their own `case` tables of
L31c36 / SEc36; a third model would have had to be added in four places and
would have been added in three. This table is the only place, and the CLI
below is how a shell script reads it:

    MODEL=$(python tools/model_tags.py L31c36 --field model)
    SE=$(python tools/model_tags.py SEc36 --field se)      # runner flags
    GATE=$(python tools/model_tags.py SEc36 --field gate)  # calibration_prompt_length flags
    python tools/model_tags.py --list

| MTAG   | model                    | method                              |
|--------|--------------------------|-------------------------------------|
| L31c36 | meta-llama/Llama-3.1-8B  | vanilla                             |
| SEc36  | meta-llama/Llama-2-7b-hf | SelfExtend, group 4, neighbour 1024 |
| L2c36  | meta-llama/Llama-2-7b-hf | vanilla (4096 window)               |
| Q25c36 | Qwen/Qwen2.5-7B          | vanilla                             |
| Q3c36  | Qwen/Qwen3-8B-Base       | vanilla                             |
| Q34c36 | Qwen/Qwen3-4B-Base       | vanilla (float32)                   |
| Q314c36| Qwen/Qwen3-14B-Base      | vanilla (bfloat16; see the table)   |

The `c36` suffix is the corrected 36-eligible-class TREC split every existing
artifact name carries; a non-TREC task folds its own tag in after it
(tools/rediscover_status.TASK_TAGS), so the base tag stays the model's name.

A tag not in the table is accepted only with an explicit MODEL (and METHOD)
override -- `resolve("mytag", model=...)` -- so a smoke on a cached
checkpoint does not have to be registered here, and a typo of a registered
tag cannot silently become a new model.
"""

from __future__ import annotations

import argparse
import json
import sys

MODEL_TAGS = {
    "L31c36": {"model": "meta-llama/Llama-3.1-8B", "method": "vanilla"},
    "SEc36": {"model": "meta-llama/Llama-2-7b-hf", "method": "selfextend",
              "group_size": 4, "neighbor_size": 1024},
    "L2c36": {"model": "meta-llama/Llama-2-7b-hf", "method": "vanilla"},
    # float32: under HF's eager attention Qwen2 in bf16 disagrees with itself
    # between one forward and prefill+continue by up to 13 bf16 ulp on the
    # label logits (Llama-2 / 3.1: 1 ulp; both ~1e-5 in fp32;
    # tools/check_two_path_noise.py), and at the time the receivers'
    # cache-equivalence gate asked for 1. Measured again on 2026-09-13
    # (RESULTS 63.17, K=10 TREC prompt): the noise is the eager kernel's --
    # Qwen2-7B bf16 is 6 ulp under eager and 4 under sdpa, Qwen3-8B-Base
    # bf16 is 2 ulp under sdpa -- so under the measured gate (14.0b-24) and
    # ATTN=sdpa both could run in bf16 (DTYPE=bfloat16 ATTN=sdpa on the
    # driver). The defaults stay float32 because the finished Q25c36 / Q3c36
    # cells are float32 eager; a rerun in bf16 is a new cell, not a resume.
    "Q25c36": {"model": "Qwen/Qwen2.5-7B", "method": "vanilla",
               "dtype": "float32"},
    "Q3c36": {"model": "Qwen/Qwen3-8B-Base", "method": "vanilla",
              "dtype": "float32"},
    # the Qwen3 size ladder around the 8B entry. 4B follows the family's
    # float32 setting (16 GB of weights). 14B cannot be float32 on a 40 GB
    # card (59 GB of weights), so it is registered in bfloat16; the cell's
    # precheck (tools/check_two_path_noise.py) measures that dtype's two-path
    # noise at the cell's largest K and the receivers' cache gate is set to
    # it (prereg 14.0b-24: max(1, measured)); the only refusal left is an
    # argmax flip between the two paths, which no gate covers.
    "Q34c36": {"model": "Qwen/Qwen3-4B-Base", "method": "vanilla",
               "dtype": "float32"},
    "Q314c36": {"model": "Qwen/Qwen3-14B-Base", "method": "vanilla",
                "dtype": "bfloat16"},
}
METHODS = ("vanilla", "selfextend")
DTYPES = ("bfloat16", "float32")


def resolve(mtag, model=None, method=None, group_size=None, neighbor_size=None,
            dtype=None):
    """The spec for `mtag`: the table's entry, or an explicit override.

    Overrides replace the table's fields; a tag absent from the table needs
    at least `model`. The result always carries model / method / group_size /
    neighbor_size (the SE parameters are present but unused under vanilla).
    """
    base = dict(MODEL_TAGS.get(mtag) or {})
    if not base and not model:
        raise KeyError(
            f"unknown MTAG {mtag!r}; registered: {sorted(MODEL_TAGS)}. An "
            "unregistered tag needs an explicit model (MODEL=... in the "
            "drivers, model=... here).")
    spec = {"mtag": mtag, "model": base.get("model"),
            "method": base.get("method", "vanilla"),
            "group_size": base.get("group_size", 4),
            "neighbor_size": base.get("neighbor_size", 1024),
            "dtype": base.get("dtype", "bfloat16")}
    if model:
        spec["model"] = model
    if method:
        spec["method"] = method
    if group_size is not None:
        spec["group_size"] = int(group_size)
    if neighbor_size is not None:
        spec["neighbor_size"] = int(neighbor_size)
    if dtype:
        spec["dtype"] = dtype
    if spec["method"] not in METHODS:
        raise ValueError(f"method {spec['method']!r}; expected one of {METHODS}")
    if spec["dtype"] not in DTYPES:
        raise ValueError(f"dtype {spec['dtype']!r}; expected one of {DTYPES}")
    return spec


def se_flags(spec):
    """The runners' flags (tools/model_args): empty under vanilla."""
    if spec["method"] != "selfextend":
        return ""
    return (f"--method selfextend --group-size {spec['group_size']} "
            f"--neighbor-size {spec['neighbor_size']}")


def gate_flags(spec):
    """tools/calibration_prompt_length's flags: empty under vanilla."""
    if spec["method"] != "selfextend":
        return ""
    return (f"--selfextend --group-size {spec['group_size']} "
            f"--neighbor-size {spec['neighbor_size']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mtag", nargs="?")
    ap.add_argument("--field", choices=("model", "method", "dtype", "se", "gate",
                                        "json"),
                    default="json")
    ap.add_argument("--dtype", default=None, choices=DTYPES)
    ap.add_argument("--model", default=None, help="override / unregistered tag")
    ap.add_argument("--method", default=None, choices=METHODS)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--neighbor-size", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for k, v in MODEL_TAGS.items():
            print(f"{k:8s} {v['model']:28s} {v['method']} {v.get('dtype', 'bfloat16')}"
                  + (f" (group {v['group_size']}, neighbour {v['neighbor_size']})"
                     if v["method"] == "selfextend" else ""))
        return 0
    if not args.mtag:
        ap.error("an MTAG is required unless --list")
    try:
        spec = resolve(args.mtag, model=args.model, method=args.method,
                       group_size=args.group_size, neighbor_size=args.neighbor_size,
                       dtype=args.dtype)
    except (KeyError, ValueError) as e:
        print(f"[model_tags] {e}", file=sys.stderr)
        return 1
    if args.field == "json":
        print(json.dumps(spec))
    elif args.field == "se":
        print(se_flags(spec))
    elif args.field == "gate":
        print(gate_flags(spec))
    else:
        print(spec[args.field])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
