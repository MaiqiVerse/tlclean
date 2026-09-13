"""The Task Vector layer, chosen on validation. ZERO GPU.

Hendel et al. pick the layer on a development set; the spec (tools/baselines/
specs/task_vector_adapted_spec.md) keeps that as the one discrete choice
(<= 5 configurations, the same rule the method's gamma and TSLA's grid
follow): among the candidate layers the receivers ran on validation, take
the highest validation ACCURACY of the main family `TV-K<full> L=<layer>
a=1`; ties go to the smaller layer. The test read then runs that layer
alone (`--tv-layers`). The descriptive m5 family is tabulated, never chosen.

    python tools/select_tv_layer.py --readout results/method/L31c36/k10_increment_into_K5_validation_readout.json \\
        --K-full 10 --json-out results/method/L31c36/tv_layer_K10_into_K5.json
    python tools/select_tv_layer.py --read results/method/L31c36/tv_layer_K10_into_K5.json      # prints the layer
    python tools/select_tv_layer.py --first-layer results/method/L31c36/tv_vectors_K10_into_K5.json
                                                                        # the sidecar's first candidate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.vector_arms import tv_family  # noqa: E402


def layer_table(readout, k_full, suffix=""):
    """{layer: {accuracy, nll}} for one TV family from a receiver readout's
    `vector_alphas` (analyze_k0_receiver). Refuses by name."""
    fam_re = re.compile(rf"^TV-{re.escape(tv_family(k_full, suffix))} L=(\d+)$")
    va = readout.get("vector_alphas")
    if not isinstance(va, dict):
        raise SystemExit(f"the readout has no 'vector_alphas' (keys: {sorted(readout)}); "
                         "analyze_k0_receiver writes it from 2026-09-13 on")
    table = {}
    for fam, curve in va.items():
        m = fam_re.match(str(fam))
        if not m:
            continue
        row = (curve or {}).get("1")
        if row is None:
            raise SystemExit(f"{fam}: no a=1 entry (has {sorted(curve or {})})")
        table[int(m.group(1))] = {"accuracy": float(row["accuracy"]), "nll": float(row["nll"])}
    return table


def select_layer(table):
    """(layer, best accuracy): the highest accuracy, ties to the SMALLER
    layer -- a rule, so that two runs with the same table choose the same."""
    if not table:
        raise ValueError("empty table")
    best = max(r["accuracy"] for r in table.values())
    return min(l for l, r in table.items() if r["accuracy"] == best), best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--readout", help="analyze_k0_receiver --json-out of the VALIDATION receiver run")
    ap.add_argument("--K-full", type=int, default=None)
    ap.add_argument("--json-out")
    ap.add_argument("--read", help="print the layer a --json-out file chose")
    ap.add_argument("--first-layer", help="print the first candidate layer of a run_tv_increment sidecar")
    args = ap.parse_args(argv)
    if args.read:
        doc = json.loads(Path(args.read).read_text(encoding="utf-8"))
        if "selected_layer" not in doc:
            raise SystemExit(f"{args.read}: no 'selected_layer' (keys: {sorted(doc)})")
        print(int(doc["selected_layer"]))
        return 0
    if args.first_layer:
        doc = json.loads(Path(args.first_layer).read_text(encoding="utf-8"))
        layers = (doc.get("runner") or {}).get("candidate_layers") or []
        if not layers:
            raise SystemExit(f"{args.first_layer}: no runner.candidate_layers "
                             f"(runner keys: {sorted(doc.get('runner') or {})})")
        print(int(layers[0]))
        return 0
    if not (args.readout and args.K_full is not None and args.json_out):
        raise SystemExit("need --readout, --K-full and --json-out (or --read / --first-layer)")
    readout = json.loads(Path(args.readout).read_text(encoding="utf-8"))
    table = layer_table(readout, args.K_full)
    if not table:
        raise SystemExit(f"{args.readout}: no `TV-K{args.K_full} L=<layer>` family "
                         f"(vector_families: {readout.get('vector_families')}); the receiver did not "
                         "run with --tv-vectors")
    m5 = layer_table(readout, args.K_full, "m5")
    sel, best = select_layer(table)
    n = int(readout.get("n_queries_per_seed", 0)) * len(readout.get("seeds") or [])
    print("=" * 78)
    print(f"TASK VECTOR LAYER ON VALIDATION -- K={args.K_full}, {n} queries, main family "
          f"(one dummy query){' and m5 (mean of five)' if m5 else ''}")
    print("=" * 78)
    print(f"    {'layer':>5} {'accuracy':>10} {'NLL':>10}" + (f"   {'m5 acc':>8} {'m5 NLL':>8}" if m5 else ""))
    for l in sorted(table):
        r = table[l]
        line = f"    {l:>5} {r['accuracy']:>10.4f} {r['nll']:>10.4f}"
        if l in m5:
            line += f"   {m5[l]['accuracy']:>8.4f} {m5[l]['nll']:>8.4f}"
        print(line + ("   <- chosen" if l == sel else ""))
    print(f"\n  chosen: layer {sel} (accuracy {best:.4f}; ties go to the smaller layer)")
    print("  ⚠ chosen ON validation: the selected layer's validation number is in sample; the test "
          "read is the one that counts")
    out = {"family": f"TV-{tv_family(args.K_full)}", "selected_layer": int(sel),
           "rule": "highest validation accuracy of the main family; ties to the smaller layer",
           "table": {str(l): table[l] for l in sorted(table)},
           "table_m5": {str(l): m5[l] for l in sorted(m5)},
           "n_queries": n, "readout": str(args.readout), "K": int(args.K_full)}
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"  [output] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
