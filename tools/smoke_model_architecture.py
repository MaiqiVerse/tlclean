"""GPU smoke: does this repo's capture / DLA path hold on a given architecture?

Run it before the first real job on any model that is not Llama-3.1-8B. The
SAME identities are computed on the reference (Llama-3.1-8B, the architecture
every number in RESULTS was produced on) and on the candidates, so a candidate
is read against the reference's bf16-level numbers rather than a made-up
tolerance (working rules 3.11). One short synthetic-MLP prompt (K=2 per class,
~220 tokens) is enough: every identity is exact up to rounding, or wrong.

    python tools/smoke_model_architecture.py \\
        --models meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B Qwen/Qwen3-8B-Base

Identities (all must hold if the capture is right on the architecture):
  1. attention_capture.CaptureContext rows are attention weights (verify_ran).
  2a. per layer: sum_k alpha_k @ u_k == that layer's o_proj output at the answer
      row (alpha capture, V capture incl. any v_proj bias, o_proj slicing).
  2b. resid[last] == resid[embed] + sum_l attn_out_l + sum_l mlp_out_l
      (pre-norm bookkeeping over the whole stack; hook placement).
  2c. final norm + lm_head on the captured last row == the answer logits of
      the same forward (model.model.norm / lm_head names and semantics).
  3. the all-heads DLA primitive (extract_per_head_logit_contributions) summed
      over heads == <sum_l attn_out_l, W_U[c]> -- the paper's primitive.

2026-09-09 reference numbers (bf16, RTX A6000): Llama-3.1-8B layer_recon_rel
2.5e-3..2.8e-3, resid_bookkeeping_rel 2.9e-3, logits_closure 0.0,
allheads_dla_closure_rel 1.7e-4; Qwen2-7B-Instruct (the Qwen2.5 code path,
QKV bias) 1.3e-3..2.4e-3 / 2.3e-3 / 0.0 / 1.3e-4; Qwen3-4B-Instruct-2507 (q/k
norm, tied embeddings, head_dim 128 != hidden/H) 1.9e-3..2.7e-3 / 2.2e-3 /
0.0 / 2.8e-4 -- after icl_common.head_dim replaced hidden_size // n_heads.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

DEFAULT_MODELS = ["meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-7B", "Qwen/Qwen3-8B-Base"]
SUMMARY_KEYS = ("labels_single_token_in_context", "capture_rows_sum_dev",
                "layer_recon_rel", "resid_bookkeeping_rel",
                "logits_closure_maxabs_cand", "allheads_dla_closure_rel",
                "model_pred", "gold", "peak_gpu_GB")


def build_prompt():
    import synthetic_mlp_task as M
    t = M.SyntheticMLP(n_queries=1)
    t.set_fewshot(num_fewshot=2, seed=11)
    doc = t.dataset["test"][0]
    return t.doc_to_text(doc), t.doc_to_target(doc).strip()


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def row_hook(store, key):
    """Answer-row (last position) of a module's output, fp32 on CPU."""
    def hook(_m, _i, out):
        x = out[0] if isinstance(out, tuple) else out
        store[key] = x[0, -1].detach().float().cpu().numpy()
    return hook


def run(name, prompt, gold):
    import torch
    from transformers import AutoTokenizer
    from tools.model_loader import load_model
    from tools.tl_heads import TLHeadSet
    from tools.label_space import first_token_in_context
    from tools.attention_capture import CaptureContext
    from tools.diagnostic_forward import (diagnostic_forward,
                                          extract_per_head_logit_contributions)
    R = {}
    tok = AutoTokenizer.from_pretrained(name)
    R["n_special_prefix_tokens"] = len(tok("", add_special_tokens=True).input_ids)
    cand = [first_token_in_context(tok, prompt, c) for c in "ABCDEF"]
    alone = [tok(" " + c, add_special_tokens=False).input_ids for c in "ABCDEF"]
    R["labels_single_token_in_context"] = bool(
        len(set(cand)) == 6 and all(len(s) == 1 and s[0] == c
                                    for s, c in zip(alone, cand)))
    R["cand_ids"] = cand

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = load_model(name, method="vanilla", attn_implementation="eager",
                       device="cuda")
    R["load_s"] = round(time.time() - t0, 1)
    cfg = model.config
    attn0 = model.model.layers[0].self_attn
    L, H = int(cfg.num_hidden_layers), int(cfg.num_attention_heads)
    KV = int(getattr(cfg, "num_key_value_heads", H))
    R["arch"] = dict(attn=type(attn0).__name__, L=L, H=H, KV=KV,
                     hidden=int(cfg.hidden_size),
                     head_dim_true=int(attn0.o_proj.in_features // H),
                     hidden_over_H=int(cfg.hidden_size) // H,
                     qkv_bias=attn0.v_proj.bias is not None,
                     qk_norm=hasattr(attn0, "q_norm"),
                     attn_impl=cfg._attn_implementation,
                     tied=bool(getattr(cfg, "tie_word_embeddings", False)))
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    T = int(ids.shape[1])
    R["n_tokens"] = T
    layers3 = [0, L // 2, L - 1]

    # 1. the attention-capture adapter
    store = {}
    with CaptureContext(model, layers3, store) as cc, torch.no_grad():
        model(ids, use_cache=False)
    cc.verify_ran()
    R["capture_rows_sum_dev"] = max(float(np.abs(store[l].sum(axis=1) - 1).max())
                                    for l in layers3)
    R["capture_rows_shape_ok"] = all(store[l].shape == (H, T) for l in layers3)

    # 2. diagnostic_forward on three whole layers, with o_proj / mlp rows hooked
    attn_out, mlp_out, hs = {}, {}, []
    for l in range(L):
        hs.append(model.model.layers[l].self_attn.o_proj
                  .register_forward_hook(row_hook(attn_out, l)))
        hs.append(model.model.layers[l].mlp
                  .register_forward_hook(row_hook(mlp_out, l)))
    heads3 = TLHeadSet(model_name=name,
                       heads={(l, h) for l in layers3 for h in range(H)},
                       source="smoke: three whole layers")
    try:
        ext = diagnostic_forward(model, ids, heads3, capture_resid=True)
    finally:
        for h in hs:
            h.remove()
    R["layer_recon_rel"] = {}
    for l in layers3:
        rec = sum((ext.alpha_per_head[(l, k)] @ ext.u_per_head[(l, k)]).numpy()
                  for k in range(H))
        R["layer_recon_rel"][l] = round(rel(rec, attn_out[l]), 6)
    resid = ext.resid_at_answer.numpy()
    book = resid[0] + sum(attn_out[l] for l in range(L)) \
        + sum(mlp_out[l] for l in range(L))
    R["resid_bookkeeping_rel"] = round(rel(book, resid[-1]), 6)
    with torch.no_grad():
        last = torch.as_tensor(resid[-1]).to(model.device, model.dtype)
        lg = model.lm_head(model.model.norm(last[None]))[0].float().cpu().numpy()
    m0 = ext.answer_logits.numpy()[cand]
    R["logits_closure_maxabs_cand"] = float(np.abs(lg[cand] - m0).max())

    # 3. the all-heads DLA primitive against the hooked attention total
    all_heads = TLHeadSet(model_name=name,
                          heads={(l, h) for l in range(L) for h in range(H)},
                          source="smoke: all heads")
    hs = [model.model.layers[l].self_attn.o_proj
          .register_forward_hook(row_hook(attn_out, l)) for l in range(L)]
    try:
        contrib = extract_per_head_logit_contributions(
            model, ids, all_heads, target_token_ids=cand)
    finally:
        for h in hs:
            h.remove()
    dla = np.zeros((L, H, 6))
    for (l, h), v in contrib.items():
        dla[l, h] = v.numpy()
    W = model.lm_head.weight.detach()[torch.as_tensor(cand)].float().cpu().numpy()
    attn_total = sum(attn_out[l] for l in range(L))
    R["allheads_dla_closure_rel"] = round(rel(dla.sum(axis=(0, 1)), W @ attn_total), 6)
    R["model_pred"] = "ABCDEF"[int(m0.argmax())]
    R["gold"] = gold
    R["direct_write_argmax_allheads"] = "ABCDEF"[int(dla.sum(axis=(0, 1)).argmax())]
    R["peak_gpu_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    del model
    torch.cuda.empty_cache()
    return R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="reference first, then the candidates")
    args = ap.parse_args(argv)
    import torch
    prompt, gold = build_prompt()
    print(f"prompt: {len(prompt)} chars, gold {gold!r}, head: {prompt[:60]!r}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    out = {}
    for name in args.models:
        print("=" * 78)
        print(name)
        print("=" * 78, flush=True)
        try:
            out[name] = run(name, prompt, gold)
            for k, v in out[name].items():
                print(f"  {k:32s} {v}")
        except Exception as e:                       # noqa: BLE001
            out[name] = {"ERROR": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
            torch.cuda.empty_cache()
        print(flush=True)
    print("SUMMARY (read each candidate against the first, reference, line)")
    for name, R in out.items():
        if "ERROR" in R:
            print(f"  {name}: {R['ERROR']}")
        else:
            print(f"  {name}: " + ", ".join(f"{k}={R[k]}" for k in SUMMARY_KEYS))
    return 1 if any("ERROR" in R for R in out.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
