"""In-Context Vectors (Liu, Ye, Xing & Zou, ICML 2024) in the increment
setting: one direction PER SEED from the EXTRA demonstrations, for the
receivers to steer every layer with. GPU, 2 x |E_s| short forwards per seed.

The construction follows the upstream code (shengliu66/ICV at b187c63,
tasks/base.py obtain_icv, utils/pca.py, utils/llm_layers.py ICVLayer),
which settles the spec's [VERIFY-REPO] items
(tools/baselines/specs/icv_adapted_spec.md, prereg 14.0b-25):

  * for every extra demonstration i two texts are encoded SEPARATELY, each
    with BOS and no other demonstration: x_i = the query rendering up to the
    label slot, x_i y_i = the demonstration with its display label. h(.) is
    the last token's [embedding output; every decoder layer's output],
    flattened ((L + 1) d); Delta_i = h(x_i y_i) - h(x_i);
  * the direction is the upstream's: a CENTRED rank-1 PCA of the Delta_i
    (SVD of Delta - mean, sklearn's svd_flip sign) PLUS the mean Delta --
    `(components_.sum(0) + mean_)` -- reshaped to (L + 1) rows with the
    embedding row dropped (`icv[1:]`). ||mean Delta|| is far above 1 in
    practice, so the direction is dominated by the mean difference; the
    sidecar records both norms and their cosine;
  * at inference the upstream ICVLayer sits after every decoder layer's MLP
    and edits EVERY position: with x the MLP output and v_l the layer's
    segment, x <- normalize(normalize(x) + lam (1 + max(0, cos(x, -v_l)))
    v_l / ||v_l||) ||x|| (vector_arms.icv_transform). The prefix therefore
    has to be re-encoded with the hooks for each lambda; the receivers keep
    one prefix cache per lambda and the arms `ICV-K<full> a=<lambda>` run on
    ARM_BASE's mask; a=0 installs nothing and is the bitwise gate;
  * lambda is the one continuous choice: the spec's grid {0.05, 0.1, 0.2,
    0.4, 0.8} (the upstream's --lam default is 0.8, its README example 0.1)
    is run on validation, tools/select_icv_lambda.py picks by validation
    NLL (ties to the smaller lambda), and the test read runs that lambda.

Output: one JSON sidecar (`runner.icv_vectors[seed]` = [n_layers][d]),
no npz: no arm is scored here. write_sidecar is the single writer.

    sbatch script/lsu1.sh python tools/baselines/run_icv_increment.py \\
        --query-manifest results/method/L31c36/query_manifest.json \\
        --label-space results/method/L31c36/label_space_L31c36.json \\
        --calibration-dir data/method/L31c36/trec_fine_per_class \\
        --model meta-llama/Llama-3.1-8B --method vanilla --dtype bfloat16 \\
        --K-base 5 --K-full 10 \\
        --out results/method/L31c36/icv_vectors_K10_into_K5.json
"""

from __future__ import annotations

import argparse
import json
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
from tools.vector_arms import ICV_LAMBDAS  # noqa: E402

NAME = "icv_increment"


def icv_direction(deltas):
    """The upstream direction from the difference vectors D [k, n]:
    centred rank-1 PCA (SVD of D - mean, sklearn's svd_flip sign: the
    component's sign makes the largest-|score| sample's score positive) PLUS
    the mean difference. Returns (direction, pc1, mean, explained variance
    ratio of pc1), all float64."""
    d = np.asarray(deltas, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] < 2:
        raise ValueError(f"deltas is {d.shape}; need [k >= 2, n]")
    mean = d.mean(axis=0)
    z = d - mean
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    # svd_flip: the sign of each component from the largest |u| entry of its column
    sign = np.sign(u[np.argmax(np.abs(u[:, 0])), 0]) or 1.0
    pc1 = vt[0] * sign
    ratio = float(s[0] ** 2 / (s ** 2).sum()) if (s ** 2).sum() > 0 else float("nan")
    return pc1 + mean, pc1, mean, ratio


def capture_last_token(model, ids, torch):
    """[n_layers + 1, d] float32: the embedding output then every decoder
    layer's output at the LAST token (the upstream tracer's `hidden`)."""
    store, handles = {}, []

    def make(i):
        def fn(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            store[i] = hs[0, -1, :].detach().to(torch.float32).cpu().numpy()
        return fn

    handles.append(model.model.embed_tokens.register_forward_hook(make(0)))
    for l, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make(l + 1)))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()
    return np.stack([store[i] for i in range(len(model.model.layers) + 1)])


def write_sidecar(out, *, vectors, stats, pairs_used, lambda_grid, n_layers, k_base, k_full,
                  model_info, task, seeds, query_manifest, label_space):
    """The sidecar the receivers read (--icv-vectors): `vectors[seed]` is
    [n_layers][d] (the embedding row already dropped)."""
    doc = {"runner": {"icv_vectors": {str(s): v for s, v in vectors.items()},
                      "icv_stats": {str(s): v for s, v in stats.items()},
                      "pairs": {str(s): v for s, v in pairs_used.items()},
                      "lambda_grid": [float(x) for x in lambda_grid],
                      "n_layers": int(n_layers), "K": int(k_full), "K_base": int(k_base),
                      "hook_boundary": "upstream ICVLayer after every decoder layer's MLP, every "
                                       "position: x <- normalize(normalize(x) + lam (1 + max(0, "
                                       "cos(x, -v_l))) v_l/||v_l||) ||x||; the prefix cache is "
                                       "rebuilt per lambda with the hooks; direction = centred "
                                       "PC1 (svd_flip sign) + mean Delta over the extra "
                                       "demonstrations' (x, xy) last-token differences, embedding "
                                       "row dropped",
                      "note": "vectors only: no arm was scored, so there is no npz. Consumers: "
                              "run_k0_receiver --icv-vectors, run_k10_increment --icv-vectors, "
                              "tools/select_icv_lambda.py"},
           "meta": {"baseline": NAME, **dict(model_info), "task": str(task),
                    "K": int(k_full), "K_base": int(k_base),
                    "seeds": [int(s) for s in seeds],
                    "query_manifest": str(query_manifest),
                    "query_manifest_sha256": file_sha256(query_manifest),
                    "label_space": str(label_space),
                    "label_space_sha256": file_sha256(label_space),
                    "spec": "tools/baselines/specs/icv_adapted_spec.md (prereg 14.0b-25); upstream "
                            "shengliu66/ICV b187c63",
                    "provenance": run_provenance()}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc), encoding="utf-8")
    return doc


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
    ap.add_argument("--limit", type=int, default=0, help="extra demonstrations used per seed (0 = all)")
    ap.add_argument("--out", required=True, help="the JSON sidecar")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.demo_seeds.split(",") if s.strip()]
    if not 0 <= args.K_base < args.K_full:
        raise SystemExit(f"--K-base {args.K_base} --K-full {args.K_full}: need 0 <= K_base < K_full")

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

    model = load_model_from_args(args)      # the receivers' kernel (eager)
    n_l = int(model.config.num_hidden_layers)
    print("=" * 78)
    print(f"IN-CONTEXT VECTORS, INCREMENT SETTING: K={args.K_full} extra demonstrations into a "
          f"K={args.K_base} receiver")
    print("=" * 78)
    print(f"  [setup] {args.model} method={args.method} dtype={args.dtype}; {n_l} layers; lambda grid "
          f"{list(ICV_LAMBDAS)} (0 = gate); direction = centred PC1 + mean Delta (upstream)")

    vectors, stats, pairs_used = {}, {}, {}
    for s in seeds:
        t0 = time.time()
        reservation = manifest_reservation(args.query_manifest, demo_seed=s)
        _base_docs, extra_docs, large = extra_demo_docs(args.task, args.K_base, args.K_full, s, reservation)
        header, _ = load_calibration_header(calibration_path(args.calibration_dir, args.task, args.K_full, s))
        labels = header["abstract_labels"]
        if renderer.build_prefix(large, labels) != list(prefixes[s]):
            raise SystemExit(f"seed {s}: the rebuilt K={args.K_full} blocks differ from build_prefixes'")
        use = extra_docs[:args.limit] if args.limit else extra_docs
        deltas = []
        for c, doc in use:
            x = renderer.render_query(doc)
            xy = renderer.render_demo(doc, labels[int(c)])
            if not xy.startswith(x):
                raise SystemExit(f"seed {s}: the demonstration rendering does not begin with the query "
                                 f"rendering, so 'x' and 'x y' are not a pair:\n  x  = {x!r}\n  xy = {xy!r}")
            hx = capture_last_token(model, tok(x, return_tensors="pt").input_ids.to(model.device), torch)
            hxy = capture_last_token(model, tok(xy, return_tensors="pt").input_ids.to(model.device), torch)
            deltas.append((hxy - hx).reshape(-1))
        d = np.stack(deltas)
        direction, pc1, mean, ratio = icv_direction(d)
        rows = direction.reshape(n_l + 1, -1)[1:]           # the upstream's icv[1:]
        vectors[s] = [[float(x) for x in r] for r in rows.astype(np.float32)]
        nm, npc = float(np.linalg.norm(mean)), float(np.linalg.norm(pc1))
        st = {"n_pairs": int(d.shape[0]), "norm_mean_delta": nm, "norm_pc1": npc,
              "cos_pc1_mean": float(pc1 @ mean / (npc * nm)) if nm > 0 and npc > 0 else float("nan"),
              "explained_variance_ratio_pc1": ratio,
              "segment_norms": [float(np.linalg.norm(r)) for r in rows]}
        stats[s] = st
        pairs_used[s] = [{"class": int(c), "text": r} for (c, _d), (_c2, r)
                         in zip(use, docs_to_rows(args.task, use))]
        print(f"\n  seed {s}: {st['n_pairs']} (x, xy) pairs from the extra demonstrations "
              f"({time.time() - t0:.0f}s); ||mean Delta|| {nm:.2f}, PC1 explains {ratio:.3f} of the "
              f"centred variance, cos(PC1, mean) {st['cos_pc1_mean']:+.3f}")
        print("    direction segment norms by layer: " +
              " ".join(f"{v:.2f}" for v in st["segment_norms"]))

    write_sidecar(args.out, vectors=vectors, stats=stats, pairs_used=pairs_used,
                  lambda_grid=ICV_LAMBDAS, n_layers=n_l, k_base=args.K_base, k_full=args.K_full,
                  model_info=model_meta(args), task=args.task, seeds=seeds,
                  query_manifest=args.query_manifest, label_space=args.label_space)
    print(f"\n  [output] {args.out}   (vectors only; no arm was scored; lambda is chosen on validation "
          "by tools/select_icv_lambda.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
