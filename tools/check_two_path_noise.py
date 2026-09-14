"""Does this model agree with itself between ONE forward and PREFILL + CONTINUE?
GPU, plain transformers, no receiver code. Run it before the first method
cell on a new model, in the dtype the cell will use.

The receivers (run_k0_receiver, run_k10_increment) prefill the demonstrations
into the KV cache and run the receiver's tokens against it; their
cache-equivalence gate (13.5.4(2)) demands that this reproduces the ordinary
one-forward prompt to 1 bf16 ulp on every label logit. On Llama-2 and
Llama-3.1 it does. On Qwen2-7B under HF's eager attention it does NOT in
bf16 -- up to 13 ulp (0.8 of a logit), argmax flips -- while in float32 both
differ by ~1e-5; under torch's sdpa kernels the same model in bf16 is back
at ~1 ulp (RESULTS 63.17: the eager kernel's bf16 QK^T is the noise, not
the construction). So the noise is a property of (model, dtype, kernel),
and this check runs under the kernel the cell will use (--attn, spelled
every time; tools/model_tags gives the Qwen tags float32 as their default).

    python tools/check_two_path_noise.py --model Qwen/Qwen2.5-7B --attn sdpa \\
        --calibration data/method/Q25c36/trec_fine_per_class/calibration_trec_fine_per_class_K5_seed42_uuid.jsonl \\
        --dtype bfloat16 float32

The prompt is the calibration file's first prompt (the label tokens come from
its header); the live tail is cut at q = 1, 12 and 40 tokens, which brackets
what a receiver runs (the query block). Exit 1 if, in the last dtype listed,
any candidate differs by more than --max-ulp (default 1) -- the gate's own
criterion -- so a driver can refuse a dtype before spending a cell on it.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def ulp_bf16(x):
    x = abs(float(x))
    if x == 0:
        return 2.0 ** -133
    e = int(np.floor(np.log2(x)))
    return 2.0 ** (e - 7)


def top2_margin_ulp(v):
    """The gap between the two largest label logits, in bf16 ulp at the
    largest one: how far the argmax is from a tie on that path."""
    s = np.sort(np.asarray(v, dtype=np.float64))[::-1]
    if s.size < 2:
        return float("inf")
    return float((s[0] - s[1]) / ulp_bf16(float(s[0])))


def flip_verdict(by_q):
    """The argmax flips of a run, split into REAL disagreements and TIES.
    A flip on a tail is a tie when, on either path, the top-two margin of
    the label logits is within that tail's own noise (its max ulp, floored
    at 1, the smallest bf16 step): the argmax is not defined there at this
    precision, so the two paths cannot be said to disagree on it (prereg
    14.0b-29; the synthetic linear bank at K=2 sits one bf16 step from a
    six-way tie, job 846377). Entries without margins (older json) count
    as real, as they always did."""
    real, ties = [], []
    for q, v in by_q.items():
        if v.get("argmax_same", True):
            continue
        noise = max(1.0, float(v.get("max_ulp", 0.0)))
        m = min(float(v.get("margin_mono_ulp", np.inf)),
                float(v.get("margin_cont_ulp", np.inf)))
        (ties if m <= noise + 1e-9 else real).append(q)
    return real, ties


def measure(model_name, calibration, dtype, method="vanilla", group_size=4,
            neighbor_size=1024, qs=(1, 12, 40), attn="eager", n_prompts=1):
    """The two paths on the first `n_prompts` prompts of the calibration, each
    split at every live length in `qs`. One prompt was the sample until
    2026-09-13: the receivers then read hundreds, and the worst site over
    900 prompts is not the worst over 3 (L31c36 x clinc150 bf16: 3 ulp on
    one prompt per seed, 5 ulp on seed 43's 300 validation prompts, job
    845647). `by_q` is keyed "<prompt>:<q>" when n_prompts > 1 and stays
    "<q>" for one prompt, so cache_gate_from_noise's max over the values
    reads both."""
    import torch
    from transformers import AutoTokenizer
    from tools.model_loader import load_model
    rows = []
    with open(calibration, encoding="utf-8") as f:
        header = json.loads(f.readline())
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= max(1, int(n_prompts)):
                break
    if not rows:
        raise SystemExit(f"{calibration}: no prompt rows after the header")
    cand = [int(t) for t in header["label_token_ids"]]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = load_model(model_name, method=method, dtype=dtype,
                       attn_implementation=attn, group_size=group_size,
                       neighbor_size=neighbor_size)
    out = {"model": model_name, "dtype": dtype, "attn": attn, "method": method,
           "n_tokens": None, "n_candidates": len(cand), "n_prompts": len(rows), "by_q": {}}
    with torch.no_grad():
        for pi, row in enumerate(rows):
            ids = tok(row["prompt"], return_tensors="pt").input_ids.to(model.device)
            T = int(ids.shape[1])
            if out["n_tokens"] is None:
                out["n_tokens"] = T
            mono = model(ids, use_cache=False).logits[0, -1, cand].float().cpu().numpy()
            for q in qs:
                pre, live = ids[:, :T - q], ids[:, T - q:]
                o = model(pre, use_cache=True)
                cont = model(live, past_key_values=o.past_key_values,
                             use_cache=True).logits[0, -1, cand].float().cpu().numpy()
                d = np.abs(cont - mono)
                ulps = d / np.array([ulp_bf16(m) for m in mono])
                w = int(d.argmax())
                key = q if len(rows) == 1 else f"{pi}:{q}"
                out["by_q"][key] = {"max_abs": float(d.max()), "max_ulp": float(ulps.max()),
                                    "n_over_1ulp": int((ulps > 1.0 + 1e-9).sum()),
                                    "argmax_same": bool(cont.argmax() == mono.argmax()),
                                    "worst_mono": float(mono[w]), "worst_cont": float(cont[w]),
                                    # how far each path's argmax is from a tie (flip_verdict)
                                    "margin_mono_ulp": top2_margin_ulp(mono),
                                    "margin_cont_ulp": top2_margin_ulp(cont),
                                    "n_tokens": T}
            del o
    del model
    torch.cuda.empty_cache()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibration", required=True,
                    help="a calibration jsonl; its first prompt and label tokens are used")
    ap.add_argument("--dtype", nargs="+", default=["bfloat16", "float32"])
    ap.add_argument("--method", choices=("vanilla", "selfextend"), default="vanilla")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--neighbor-size", type=int, default=1024)
    ap.add_argument("--attn", choices=("eager", "sdpa"), required=True,
                    help="the kernel the cell's receivers will run under; the noise "
                         "belongs to (model, dtype, kernel), so it is measured under "
                         "the same one and recorded in the json")
    ap.add_argument("--max-ulp", type=float, default=1.0,
                    help="the receivers' gate: every candidate within this many bf16 ulp")
    ap.add_argument("--require-argmax", action="store_true",
                    help="also FAIL if the two paths' argmax over the label logits differs on "
                         "any live tail -- the categorical part of the check, which a measured "
                         "gate (prereg 14.0b-24) never relaxes")
    ap.add_argument("--json-out")
    ap.add_argument("--n-prompts", type=int, default=1,
                    help="prompts of the calibration to measure (the first N; each at every "
                         "live length). 1 = the pre-2026-09-13 sample; the driver passes GATE_N")
    args = ap.parse_args(argv)
    sys.path.insert(0, ".")
    results = []
    for dt in args.dtype:
        r = measure(args.model, args.calibration, dt, args.method,
                    args.group_size, args.neighbor_size, attn=args.attn,
                    n_prompts=args.n_prompts)
        results.append(r)
        print(f"{args.model}  {dt}  {args.attn}  {r['n_prompts']} prompt(s), first {r['n_tokens']} tokens, "
              f"{r['n_candidates']} label logits")
        items = list(r["by_q"].items())
        shown = items if len(items) <= 12 else sorted(items, key=lambda kv: -kv[1]["max_ulp"])[:12]
        for q, v in shown:
            print(f"   live q={str(q):>6}: max|diff| {v['max_abs']:.4f} = {v['max_ulp']:.2f} bf16-ulp, "
                  f"{v['n_over_1ulp']}/{r['n_candidates']} > 1 ulp, argmax same "
                  f"{v['argmax_same']}, worst {v['worst_mono']:.4f} vs {v['worst_cont']:.4f}")
        if len(items) > 12:
            hist = {}
            for _q, v in items:
                b = int(np.ceil(v["max_ulp"] - 1e-9))
                hist[b] = hist.get(b, 0) + 1
            print(f"   ({len(items)} (prompt, live length) pairs; the 12 worst shown; "
                  f"worst-ulp histogram {dict(sorted(hist.items()))})")
    last = results[-1]
    worst = max(v["max_ulp"] for v in last["by_q"].values())
    real, ties = flip_verdict(last["by_q"])
    last["flips"] = {"real": [str(q) for q in real], "ties": [str(q) for q in ties]}
    ok = worst <= args.max_ulp and not (args.require_argmax and real)
    print(f"\n  {'[PASS]' if ok else '[FAIL]'} {last['dtype']}: worst {worst:.2f} ulp against "
          f"the gate's {args.max_ulp:g}"
          + (f"; argmax differs on live tail(s) {real}" if real else "; argmax agrees on every tail")
          + (f"; a tie within the noise on tail(s) {ties} (the top-two margin on a path is within "
             "that tail's ulp: the argmax is not defined there at this precision, prereg 14.0b-29)"
             if ties else "")
          + ("" if ok else
             " -- the two paths disagree beyond what this run allows; under prereg 14.0b-24 the "
             "receivers' gate is the noise measured at the cell's largest K, but an argmax flip "
             "is not a noise level and no gate covers it"))
    if args.json_out:
        from pathlib import Path
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  [output] {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
