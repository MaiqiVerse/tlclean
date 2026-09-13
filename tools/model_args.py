"""The model arguments every GPU runner takes, and the one load call.

Seven runners spelled `load_model(args.model, method="vanilla",
attn_implementation="eager")` -- a SelfExtend model could not reach any of
them, and Llama-2's 4096 window puts the K=10 level out of reach without it.
This module gives them the same four flags the rediscovery chain's probes
already take (`--method --group-size --neighbor-size` on top of `--model`),
one load, and one meta block, so a driver passes tools/model_tags' `se`
string to every runner alike.

    from tools.model_args import add_model_arguments, load_model_from_args, model_meta
    add_model_arguments(ap)                         # --model --method --group-size --neighbor-size
    model = load_model_from_args(args)              # eager attention, as before
    meta.update(model_meta(args))                   # {"model", "method", ...}
"""

from __future__ import annotations

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
METHODS = ("vanilla", "selfextend")
DTYPES = ("bfloat16", "float32")
ATTNS = ("eager", "sdpa")


def add_model_arguments(ap, default_model=DEFAULT_MODEL):
    ap.add_argument("--model", default=default_model)
    ap.add_argument("--method", choices=METHODS, default="vanilla",
                    help="selfextend = Llama-2 + SelfExtend (models/selfExtend); "
                         "the receivers' additive visibility mask enters the "
                         "attention module the same way under both")
    ap.add_argument("--group-size", type=int, default=4,
                    help="SelfExtend group size (ignored under vanilla)")
    ap.add_argument("--neighbor-size", type=int, default=1024,
                    help="SelfExtend neighbour window (ignored under vanilla)")
    ap.add_argument("--dtype", choices=DTYPES, default="bfloat16",
                    help="bfloat16 (the Llama runs) or float32. Under eager "
                         "attention Qwen2 in bf16 disagrees with itself between "
                         "one forward and prefill+continue by up to 13 bf16 ulp "
                         "(Llama: 1), which is why the Qwen tags default to "
                         "float32; under sdpa it is ~1 ulp (RESULTS 63.17; "
                         "tools/check_two_path_noise.py measures it)")
    ap.add_argument("--attn", choices=ATTNS, required=True,
                    help="the attention kernel the model is loaded with: eager "
                         "(HF's, a [heads, L, L] float32 softmax in memory; every "
                         "registered run) or sdpa (torch's fused kernels, no score "
                         "matrix, RESULTS 63.17). A run's numbers depend on it, so "
                         "it is spelled every time (working rules 3.13) and recorded in "
                         "the meta; SelfExtend is eager only")
    return ap


def load_model_from_args(args, attn_implementation=None):
    """The cell's kernel is `args.attn`; a runner that must use another one
    for a reason of its own (run_i2cl_increment calibrates under sdpa)
    passes it explicitly and records it."""
    from tools.model_loader import load_model
    return load_model(args.model, method=args.method,
                      dtype=getattr(args, "dtype", "bfloat16"),
                      attn_implementation=attn_implementation or args.attn,
                      group_size=int(args.group_size),
                      neighbor_size=int(args.neighbor_size))


def model_meta(args):
    """What to record: the method and the kernel matter as much as the checkpoint."""
    out = {"model": args.model, "method": args.method,
           "dtype": getattr(args, "dtype", "bfloat16"), "attn": args.attn}
    if args.method == "selfextend":
        out["group_size"] = int(args.group_size)
        out["neighbor_size"] = int(args.neighbor_size)
    return out


def describe(args):
    if args.method == "selfextend":
        return (f"{args.model} + SelfExtend (group {args.group_size}, "
                f"neighbour {args.neighbor_size}, {args.attn})")
    return f"{args.model} (vanilla, {args.attn})"
