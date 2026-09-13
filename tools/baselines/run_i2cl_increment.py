"""Implicit In-Context Learning (Li et al., ICLR 2025) in the increment
setting: context vectors from the EXTRA demonstrations and 4 L calibrated
scalars per seed, for the receivers to steer every layer with. GPU: a
calibration of --epochs epochs over |E_s| pseudo-queries.

The construction follows the upstream code (LzVv123456/I2CL at c2c6dd9,
wrapper.py, configs/config_i2cl.py), which settles the spec's
[VERIFY-REPO] items (tools/baselines/specs/i2cl_adapted_spec.md, prereg
14.0b-25):

  * context vectors: every extra demonstration (its rendering WITH the
    display label) is one forward; at every decoder layer the self_attn
    output and the mlp output at the LAST token are taken; the mean over
    the demonstrations gives cv[attn][l], cv[mlp][l] ('split_demon',
    'tok_pos' last, 'post_fuse' mean, 'layer' all, modules attn + mlp);
  * injection ('linear', 'all'): on every layer's self_attn and mlp
    outputs, every position, out <- beta * out + lambda * cv, with
    (lambda, beta) per (layer, module) initialised at (0.1, 1.0) -- 4 L
    scalars (vector_arms.i2cl_module_hooks);
  * calibration: the pseudo-queries are the extra demonstrations
    themselves, rendered as queries after the K_base prefix (no prefix at
    K_base = 0, the upstream's zero-shot recipe); the loss is the FULL
    vocabulary cross-entropy at the answer position with the demonstration's
    label token as gold; AdamW lr 0.01 wd 1e-3, --epochs 100, the upstream's
    5 % warm-up then cosine schedule; in train mode every hooked output gets
    randn * ||out|| * 0.001 (noisy self-calibration, on the module output
    as the code does). Two adaptations for our sizes: the batch is --grad-bs
    (8, accumulated one query at a time: |E_s| = 180 here against the
    upstream's ~30 demonstrations, so the step count stays comparable), and
    the injected PREFIX cache is recomputed under no_grad every
    --refresh-steps optimisation steps rather than carried through the graph
    (the whole-prompt graph of a 3.8k-token prefix does not fit a 40 GB
    card); the coefficients still receive the gradient through every
    position of the pseudo-query itself;
  * the receivers' `I2CL-K<full> a=1` arm rebuilds the prefix cache with
    the calibrated hooks once per seed and runs the query under them, on
    ARM_BASE's mask; `a=0` installs nothing and is the bitwise gate. There
    is no grid: the strengths are the calibrated coefficients.

Output: an npz (cv_attn_seed<s>, cv_mlp_seed<s> [n_layers, d], coef_seed<s>
[n_layers, 2, 2], meta json) -- the arrays are too big for JSON -- no arm is
scored here. write_sidecar is the single writer.

    sbatch script/lsu1.sh python tools/baselines/run_i2cl_increment.py \\
        --query-manifest results/method/L31c36/query_manifest.json \\
        --label-space results/method/L31c36/label_space_L31c36.json \\
        --calibration-dir data/method/L31c36/trec_fine_per_class \\
        --model meta-llama/Llama-3.1-8B --method vanilla --dtype bfloat16 \\
        --K-base 5 --K-full 10 \\
        --out results/method/L31c36/i2cl_vectors_K10_into_K5.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.baselines.common import load_candidate_space  # noqa: E402
from tools.baselines.forward import (build_prefixes, calibration_path,  # noqa: E402
                                     load_calibration_header)
from tools.baselines.run_fv_increment import extra_demo_docs  # noqa: E402
from tools.icl_common import run_provenance  # noqa: E402
from tools.label_space import file_sha256  # noqa: E402
from tools.model_args import (add_model_arguments, load_model_from_args,  # noqa: E402
                              model_meta)
from tools.prereg_task import docs_to_rows  # noqa: E402
from tools.probe_prototype_shrinkage import manifest_reservation  # noqa: E402
from tools.prompt_render import TaskRenderer  # noqa: E402
from tools.vector_arms import I2CL_MODULES, i2cl_module, i2cl_module_hooks  # noqa: E402

NAME = "i2cl_increment"
INIT_COEF = (0.1, 1.0)          # the upstream's init_value for 'linear': (lambda, beta)


def upstream_lr_lambda(total_steps, warmup_steps):
    """The upstream LambdaLR: linear warm-up, then cosine to 0 over the
    total (wrapper.calibrate_strength); step 0 has factor 0."""
    def f(step):
        if warmup_steps <= 0:
            return (1 + math.cos(math.pi * step / max(1, total_steps))) / 2
        if step > warmup_steps:
            return min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2
        return step / warmup_steps
    return f


def capture_module_last(model, ids, torch):
    """{module: [n_layers, d] float32}: every layer's self_attn and mlp
    OUTPUT at the last token of one forward (the upstream extract_latent
    with tok_pos 'last')."""
    n_l = len(model.model.layers)
    store = {m: [None] * n_l for m in I2CL_MODULES}
    handles = []

    def make(l, m):
        def fn(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            store[m][l] = hs[0, -1, :].detach().to(torch.float32).cpu().numpy()
        return fn

    for l in range(n_l):
        for m in I2CL_MODULES:
            handles.append(i2cl_module(model, l, m).register_forward_hook(make(l, m)))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()
    return {m: np.stack(store[m]) for m in I2CL_MODULES}


def write_sidecar(out, *, cv, coef, losses, pseudo, config, n_layers, k_base, k_full, model_info,
                  task, seeds, query_manifest, label_space):
    """The npz the receivers read (--i2cl-vectors): arrays per seed plus a
    0-d `meta` json (the receivers' npz convention)."""
    arrays = {}
    for s in cv:
        arrays[f"cv_attn_seed{s}"] = np.asarray(cv[s]["attn"], dtype=np.float32)
        arrays[f"cv_mlp_seed{s}"] = np.asarray(cv[s]["mlp"], dtype=np.float32)
        arrays[f"coef_seed{s}"] = np.asarray(coef[s], dtype=np.float32)
    meta = {"baseline": NAME, **dict(model_info), "task": str(task),
            "K": int(k_full), "K_base": int(k_base), "n_layers": int(n_layers),
            "modules": list(I2CL_MODULES), "coef_layout": "[layer, module (attn, mlp), (lambda, beta)]",
            "init_coef": list(INIT_COEF), "config": dict(config),
            "losses": {str(s): [float(x) for x in v] for s, v in losses.items()},
            "final_coef_summary": {str(s): {"lambda_mean": float(np.asarray(coef[s])[:, :, 0].mean()),
                                            "beta_mean": float(np.asarray(coef[s])[:, :, 1].mean()),
                                            "lambda_min": float(np.asarray(coef[s])[:, :, 0].min()),
                                            "lambda_max": float(np.asarray(coef[s])[:, :, 0].max())}
                                   for s in coef},
            "pseudo_queries": {str(s): v for s, v in pseudo.items()},
            "seeds": [int(s) for s in seeds],
            "query_manifest": str(query_manifest),
            "query_manifest_sha256": file_sha256(query_manifest),
            "label_space": str(label_space),
            "label_space_sha256": file_sha256(label_space),
            "hook_boundary": "on every decoder layer's self_attn and mlp outputs, every position: "
                             "out <- beta * out + lambda * cv (4 L calibrated scalars); the prefix "
                             "cache is rebuilt with the hooks per seed; cv = mean over the extra "
                             "demonstrations' last-token module outputs",
            "spec": "tools/baselines/specs/i2cl_adapted_spec.md (prereg 14.0b-25); upstream "
                    "LzVv123456/I2CL c2c6dd9",
            "provenance": run_provenance()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--label-space", required=True)
    ap.add_argument("--calibration-dir", required=True)
    add_model_arguments(ap)
    ap.add_argument("--task", default="trec_fine_per_class")
    ap.add_argument("--K-base", type=int, required=True)
    ap.add_argument("--K-full", type=int, required=True)
    ap.add_argument("--demo-seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=100, help="the upstream's 100")
    ap.add_argument("--grad-bs", type=int, default=8,
                    help="pseudo-queries per optimisation step, accumulated one at a time (upstream 2 "
                         "on ~30 demonstrations; 8 on 180 keeps the step count comparable)")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--noise-scale", type=float, default=0.001)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of the epochs, the upstream's 0.05")
    ap.add_argument("--refresh-steps", type=int, default=10,
                    help="rebuild the injected prefix cache (no_grad) every this many optimisation steps")
    ap.add_argument("--limit", type=int, default=0, help="extra demonstrations used per seed (0 = all)")
    ap.add_argument("--out", required=True, help="the npz sidecar")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.demo_seeds.split(",") if s.strip()]
    if not 0 <= args.K_base < args.K_full:
        raise SystemExit(f"--K-base {args.K_base} --K-full {args.K_full}: need 0 <= K_base < K_full")
    if args.epochs < 1 or args.grad_bs < 1:
        raise SystemExit("--epochs and --grad-bs must be at least 1")

    from transformers import AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(args.model)
    ls = load_candidate_space(args.label_space, model=args.model,
                              query_manifest=args.query_manifest, tokenizer=tok)
    prefixes, _text_of, checks, _rp = build_prefixes(
        args.task, args.K_full, seeds, args.query_manifest, args.calibration_dir, ls)
    for c in checks:
        print(f"  [PASS] {c}")
    renderer = TaskRenderer(args.task)

    # sdpa for the calibration's backward (no [heads, L, L] score matrices kept);
    # the receivers' eager kernel differs by the measured ulp, which a
    # calibration can carry
    attn = "eager" if args.method == "selfextend" else "sdpa"   # SelfExtend: its own eager attention only
    model = load_model_from_args(args, attn_implementation=attn)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    n_l = int(model.config.num_hidden_layers)
    config = {"epochs": int(args.epochs), "grad_bs": int(args.grad_bs), "lr": float(args.lr),
              "wd": float(args.wd), "noise_scale": float(args.noise_scale), "warmup": float(args.warmup),
              "refresh_steps": int(args.refresh_steps), "optim": "adamW", "inject": "linear/all",
              "attn_implementation": attn, "limit": int(args.limit)}
    print("=" * 78)
    print(f"I2CL, INCREMENT SETTING: K={args.K_full} extra demonstrations into a K={args.K_base} receiver")
    print("=" * 78)
    print(f"  [setup] {args.model} method={args.method} dtype={args.dtype} attn=sdpa; {n_l} layers; "
          f"{config}")

    cvs, coefs, losses, pseudo = {}, {}, {}, {}
    for s in seeds:
        t0 = time.time()
        reservation = manifest_reservation(args.query_manifest, demo_seed=s)
        base_docs, extra_docs, large = extra_demo_docs(args.task, args.K_base, args.K_full, s, reservation)
        header, _ = load_calibration_header(calibration_path(args.calibration_dir, args.task, args.K_full, s))
        labels = header["abstract_labels"]
        if renderer.build_prefix(large, labels) != list(prefixes[s]):
            raise SystemExit(f"seed {s}: the rebuilt K={args.K_full} blocks differ from build_prefixes'")
        use = extra_docs[:args.limit] if args.limit else extra_docs

        # ---- context vectors: mean over the extra demonstrations (with label)
        acc = {m: np.zeros((n_l, model.config.hidden_size), dtype=np.float64) for m in I2CL_MODULES}
        for c, doc in use:
            ids = tok(renderer.render_demo(doc, labels[int(c)]), return_tensors="pt").input_ids.to(model.device)
            got = capture_module_last(model, ids, torch)
            for m in I2CL_MODULES:
                acc[m] += got[m]
        cv = {m: (acc[m] / float(len(use))).astype(np.float32) for m in I2CL_MODULES}
        print(f"\n  seed {s}: context vectors from {len(use)} extra demonstrations ({time.time() - t0:.0f}s); "
              f"|cv attn| by layer {np.linalg.norm(cv['attn'], axis=1).mean():.2f} mean, |cv mlp| "
              f"{np.linalg.norm(cv['mlp'], axis=1).mean():.2f} mean")

        # ---- pseudo-queries: the extra demonstrations as queries after the K_base prefix
        blocks_base = renderer.build_prefix(base_docs, labels)
        through = ("\n\n".join(blocks_base) + "\n\n") if blocks_base else ""
        pre_ids = tok(through, return_tensors="pt").input_ids.to(model.device) if through else None
        n_pre = int(pre_ids.shape[1]) if pre_ids is not None else 0
        queries = []
        for c, doc in use:
            fid = tok(renderer.render_prompt(blocks_base, doc), return_tensors="pt").input_ids.to(model.device)
            if pre_ids is not None and not torch.equal(fid[0, :n_pre], pre_ids[0]):
                raise SystemExit(f"seed {s}: a pseudo-query's tokenisation does not begin with the "
                                 "prefix's own; the boundary does not exist at the token level")
            queries.append((fid[:, n_pre:] if pre_ids is not None else fid, int(ls.label_token_ids[int(c)])))
        pseudo[s] = [{"class": int(c), "text": r} for (c, _d), (_c2, r) in zip(use, docs_to_rows(args.task, use))]

        # ---- calibration (the upstream loop, accumulated one query at a time)
        coef = torch.nn.Parameter(torch.tensor([[list(INIT_COEF)] * len(I2CL_MODULES)] * n_l,
                                               dtype=torch.float32, device=model.device))
        opt = torch.optim.AdamW([coef], lr=args.lr, weight_decay=args.wd)
        steps_per_epoch = max(1, len(queries) // args.grad_bs)
        total_steps = args.epochs * steps_per_epoch
        warmup_steps = int(args.warmup * args.epochs * steps_per_epoch)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, upstream_lr_lambda(total_steps, warmup_steps))
        rng = np.random.default_rng(int(s))
        handles = i2cl_module_hooks(model, cv, coef, torch, train=True, noise_scale=args.noise_scale)
        # The injected prefix is kept as DETACHED per-layer (k, v) tensors and
        # every pseudo-query gets a fresh DynamicCache built on them: the
        # forward's torch.cat leaves the prefix tensors untouched, so no graph
        # survives from one query to the next (cropping a cache that a
        # backward had run through raised "backward through the graph a
        # second time" on the smoke).
        from transformers import DynamicCache
        prefix_legacy, step, ep_losses = None, 0, []
        t1 = time.time()
        try:
            for ep in range(args.epochs):
                order = rng.permutation(len(queries))
                ep_loss = []
                for b in range(steps_per_epoch):
                    if pre_ids is not None and (prefix_legacy is None or step % args.refresh_steps == 0):
                        with torch.no_grad():
                            _c = model(pre_ids, use_cache=True).past_key_values
                            prefix_legacy = tuple((k.detach(), v.detach()) for k, v in _c.to_legacy_cache())
                            del _c
                    opt.zero_grad(set_to_none=True)
                    batch = order[b * args.grad_bs:(b + 1) * args.grad_bs]
                    tot = 0.0
                    for qi in batch:
                        suffix, gold = queries[qi]
                        with torch.enable_grad():
                            if prefix_legacy is not None:
                                cache = DynamicCache.from_legacy_cache(prefix_legacy)
                                lg = model(suffix, past_key_values=cache, use_cache=True).logits[0, -1]
                                del cache
                            else:
                                lg = model(suffix).logits[0, -1]
                            loss = torch.nn.functional.cross_entropy(
                                lg.to(torch.float32).unsqueeze(0),
                                torch.tensor([gold], device=lg.device)) / float(len(batch))
                            loss.backward()
                        tot += float(loss.detach()) * len(batch)
                        del lg, loss
                    opt.step()
                    sched.step()
                    step += 1
                    ep_loss.append(tot / float(len(batch)))
                ep_losses.append(float(np.mean(ep_loss)))
                if (ep + 1) % 10 == 0 or ep == 0:
                    print(f"    epoch {ep + 1}/{args.epochs}: loss {ep_losses[-1]:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                          f"lambda mean {float(coef.detach()[:, :, 0].mean()):.4f}  beta mean "
                          f"{float(coef.detach()[:, :, 1].mean()):.4f}  ({time.time() - t1:.0f}s)", flush=True)
        finally:
            for h in handles:
                h.remove()
        coef_np = coef.detach().to(torch.float32).cpu().numpy()
        cvs[s], coefs[s], losses[s] = cv, coef_np, ep_losses
        print(f"  seed {s}: calibrated {coef_np.size} scalars in {time.time() - t1:.0f}s; loss "
              f"{ep_losses[0]:.4f} -> {ep_losses[-1]:.4f}; lambda in [{coef_np[:, :, 0].min():.3f}, "
              f"{coef_np[:, :, 0].max():.3f}], beta in [{coef_np[:, :, 1].min():.3f}, {coef_np[:, :, 1].max():.3f}]")
        del prefix_legacy
        torch.cuda.empty_cache()

    write_sidecar(args.out, cv=cvs, coef=coefs, losses=losses, pseudo=pseudo, config=config, n_layers=n_l,
                  k_base=args.K_base, k_full=args.K_full, model_info=model_meta(args), task=args.task,
                  seeds=seeds, query_manifest=args.query_manifest, label_space=args.label_space)
    print(f"\n  [output] {args.out}   (vectors and coefficients only; no arm was scored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
