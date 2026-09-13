"""Task Vectors (Hendel, Geva & Globerson, Findings of EMNLP 2023) in the
increment setting: one theta per (seed, candidate layer) from the EXTRA
demonstrations, for the receivers to REPLACE the answer row with. GPU, a
handful of forwards per seed.

The original: demonstrations S plus a dummy query x'; the hidden state at
the separator (the token before the answer) at layer L is theta_L(S). At
inference the zero-shot prompt [x_q, ->] has that position's layer-L state
overwritten by theta and the forward continues. Here, with the project's
information budget (tools/baselines/specs/task_vector_adapted_spec.md,
prereg 14.0b-25):

  * S is the extra demonstrations E_s (the K_full draw minus the K_base
    draw) and the dummy query is a K_base demonstration's TEXT -- outside
    S, no validation or test query; at K_base = 0 it is leave-one-out over
    E_s. These are run_fv_increment.extraction_pairs, the same prompts FV
    extracts from;
  * the MAIN theta is ONE dummy query, one forward (the paper's section 3.1
    recipe, no mean). The mean over the first --m-desc (5) dummy queries is
    written beside it as the descriptive `TV-K<full>m5` family, with the
    cosine between the two per layer (the paper's section 4 stability);
  * the candidate layers are {8, 11, 14, 17, 20} on 32 layers -- the spec's
    <= 5 configurations -- and the same fractions of the depth elsewhere
    (vector_arms.tv_candidate_layers). The receivers run every candidate on
    validation; tools/select_tv_layer.py picks the layer by validation
    accuracy, ties to the smaller layer, and the test read runs that one;
  * the receivers' `TV-K<full> L=<layer> a=1` arm replaces the answer row of
    decoder layer <layer>'s OUTPUT (0-based, the module this runner hooks)
    on the K_base receiver's cached forward, ARM_BASE's mask; `a=0`
    installs nothing and is the bitwise gate. A task vector has no
    strength: there is no alpha grid.

Output: one JSON sidecar (`runner.tv_vectors[seed][layer]`,
`runner.tv_vectors_m5[seed][layer]`, `runner.candidate_layers`), no npz: no
arm is scored here. write_sidecar is the single writer; the tests build
their fixtures by calling it (working rules 3.2b).

    sbatch script/lsu1.sh python tools/baselines/run_tv_increment.py \\
        --query-manifest results/method/L31c36/query_manifest.json \\
        --label-space results/method/L31c36/label_space_L31c36.json \\
        --calibration-dir data/method/L31c36/trec_fine_per_class \\
        --model meta-llama/Llama-3.1-8B --method vanilla --dtype bfloat16 \\
        --K-base 5 --K-full 10 \\
        --out results/method/L31c36/tv_vectors_K10_into_K5.json
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
from tools.baselines.run_fv_increment import (extra_demo_docs,  # noqa: E402
                                              extraction_pairs)
from tools.icl_common import run_provenance  # noqa: E402
from tools.label_space import file_sha256  # noqa: E402
from tools.model_args import (add_model_arguments, load_model_from_args,  # noqa: E402
                              model_meta)
from tools.prereg_task import docs_to_rows  # noqa: E402
from tools.probe_prototype_shrinkage import manifest_reservation  # noqa: E402
from tools.prompt_render import TaskRenderer  # noqa: E402
from tools.vector_arms import parse_layer_list, tv_candidate_layers  # noqa: E402

NAME = "tv_increment"
M_DESC = 5


def capture_rows(model, ids, layers, torch):
    """{layer: the answer-row OUTPUT of decoder layer `layer`, float64} from
    one forward. Captured by a forward hook on the very module the
    receivers' replace_hook edits, so the two cannot be off by one."""
    store, handles = {}, []

    def make(l):
        def fn(_mod, _args, out):
            hs = out[0] if isinstance(out, tuple) else out
            store[l] = hs[0, -1, :].detach().to(torch.float64).cpu().numpy()
        return fn

    for l in layers:
        handles.append(model.model.layers[int(l)].register_forward_hook(make(int(l))))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()
    return store


def theta_stats(main, mean):
    """{layer: {norm_main, norm_m5, cos_main_m5}} -- how far the single
    dummy query's theta sits from the mean over several (the paper's
    stability claim, section 4, as a number per layer)."""
    out = {}
    for l in main:
        a, b = np.asarray(main[l], dtype=np.float64), np.asarray(mean[l], dtype=np.float64)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        out[int(l)] = {"norm_main": na, "norm_m5": nb,
                       "cos_main_m5": float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")}
    return out


def write_sidecar(out, *, vectors, vectors_m5, dummies, stats, layers, m_desc, n_prefix,
                  k_base, k_full, model_info, task, seeds, query_manifest, label_space):
    """The sidecar the receivers read (--tv-vectors). `vectors[seed][layer]`
    is theta (a list of floats); the layout is written here and nowhere else."""
    doc = {"runner": {"tv_vectors": {str(s): {str(l): v for l, v in d.items()} for s, d in vectors.items()},
                      "tv_vectors_m5": {str(s): {str(l): v for l, v in d.items()}
                                        for s, d in vectors_m5.items()},
                      "candidate_layers": [int(l) for l in layers],
                      "m_desc": int(m_desc), "n_prefix_demos": int(n_prefix),
                      "dummy_queries": {str(s): v for s, v in dummies.items()},
                      "tv_stats": {str(s): {str(l): v for l, v in d.items()} for s, d in stats.items()},
                      "K": int(k_full), "K_base": int(k_base),
                      "hook_boundary": "theta_L = the answer-row OUTPUT of decoder layer L (0-based) on "
                                       "[extra demonstrations || dummy query]; the receivers REPLACE the "
                                       "answer row of layer L's output with it (vector_arms.replace_hook); "
                                       "tv_vectors = one dummy query (the paper's recipe), tv_vectors_m5 = "
                                       "the mean over m_desc dummy queries (descriptive)",
                      "note": "vectors only: no arm was scored, so there is no npz. Consumers: "
                              "run_k0_receiver --tv-vectors, run_k10_increment --tv-vectors, "
                              "tools/select_tv_layer.py"},
           "meta": {"baseline": NAME, **dict(model_info), "task": str(task),
                    "K": int(k_full), "K_base": int(k_base),
                    "seeds": [int(s) for s in seeds],
                    "query_manifest": str(query_manifest),
                    "query_manifest_sha256": file_sha256(query_manifest),
                    "label_space": str(label_space),
                    "label_space_sha256": file_sha256(label_space),
                    "spec": "tools/baselines/specs/task_vector_adapted_spec.md (prereg 14.0b-25)",
                    "provenance": run_provenance()}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
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
    ap.add_argument("--layers", default="",
                    help="comma list of candidate layers (0-based decoder layers whose output is "
                         "replaced); default: vector_arms.tv_candidate_layers -- 8,11,14,17,20 on 32")
    ap.add_argument("--m-desc", type=int, default=M_DESC,
                    help="dummy queries averaged for the descriptive m5 family (the main theta is "
                         "always the first one alone)")
    ap.add_argument("--out", required=True, help="the JSON sidecar")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.demo_seeds.split(",") if s.strip()]
    if not 0 <= args.K_base < args.K_full:
        raise SystemExit(f"--K-base {args.K_base} --K-full {args.K_full}: need 0 <= K_base < K_full")
    if args.m_desc < 1:
        raise SystemExit(f"--m-desc {args.m_desc}: at least 1")

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
    layers = parse_layer_list(args.layers) or tv_candidate_layers(n_l)
    bad = [l for l in layers if not 0 <= l < n_l]
    if bad:
        raise SystemExit(f"--layers {bad}: outside 0..{n_l - 1}")
    print("=" * 78)
    print(f"TASK VECTORS, INCREMENT SETTING: K={args.K_full} extra demonstrations into a K={args.K_base} receiver")
    print("=" * 78)
    print(f"  [setup] {args.model} method={args.method} dtype={args.dtype}; {n_l} layers; candidate layers "
          f"{layers} (0-based outputs); main theta = one dummy query, m{args.m_desc} = the mean over "
          f"{args.m_desc}")

    vectors, vectors_m5, dummies, stats = {}, {}, {}, {}
    n_prefix = None
    for s in seeds:
        t0 = time.time()
        reservation = manifest_reservation(args.query_manifest, demo_seed=s)
        base_docs, extra_docs, large = extra_demo_docs(args.task, args.K_base, args.K_full, s, reservation)
        header, _ = load_calibration_header(calibration_path(args.calibration_dir, args.task, args.K_full, s))
        labels = header["abstract_labels"]
        if renderer.build_prefix(large, labels) != list(prefixes[s]):
            raise SystemExit(f"seed {s}: the rebuilt K={args.K_full} blocks differ from build_prefixes'")
        pairs = extraction_pairs(base_docs, extra_docs)
        m = min(int(args.m_desc), len(pairs))
        thetas = []
        for (_c, doc), pre_docs in pairs[:m]:
            prompt = renderer.render_prompt(renderer.build_prefix(pre_docs, labels), doc)
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            thetas.append(capture_rows(model, ids, layers, torch))
            n_prefix = len(pre_docs)
        main_t = thetas[0]
        mean_t = {l: np.mean([t[l] for t in thetas], axis=0) for l in layers}
        st = theta_stats(main_t, mean_t)
        vectors[s] = {l: [float(x) for x in main_t[l]] for l in layers}
        vectors_m5[s] = {l: [float(x) for x in mean_t[l]] for l in layers}
        dummies[s] = [{"class": int(c), "text": r}
                      for (c, _d), (_c2, r) in zip([q for q, _p in pairs[:m]],
                                                   docs_to_rows(args.task, [q for q, _p in pairs[:m]]))]
        stats[s] = st
        print(f"\n  seed {s}: {len(extra_docs)} extra demonstrations as S, dummy query = "
              f"{'a base demonstration' if base_docs else 'one extra demonstration, left out of S'}; "
              f"{m} forwards ({time.time() - t0:.0f}s)")
        for l in layers:
            print(f"    layer {l:>2}: |theta| {st[l]['norm_main']:.3f}   |mean_{m}| {st[l]['norm_m5']:.3f}   "
                  f"cos {st[l]['cos_main_m5']:.4f}")

    write_sidecar(args.out, vectors=vectors, vectors_m5=vectors_m5, dummies=dummies, stats=stats,
                  layers=layers, m_desc=min(int(args.m_desc), max(len(v) for v in dummies.values())),
                  n_prefix=n_prefix, k_base=args.K_base, k_full=args.K_full,
                  model_info=model_meta(args), task=args.task, seeds=seeds,
                  query_manifest=args.query_manifest, label_space=args.label_space)
    print(f"\n  [output] {args.out}   (vectors only; no arm was scored; the layer is chosen on "
          "validation by tools/select_tv_layer.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
