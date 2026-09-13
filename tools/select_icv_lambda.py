"""The in-context vector's lambda, chosen on validation. ZERO GPU.

The spec (tools/baselines/specs/icv_adapted_spec.md) makes lambda the one
continuous choice, picked on validation by the mean candidate NLL over the
grid {0.05, 0.1, 0.2, 0.4, 0.8} (<= 5 configurations); ties go to the
smaller lambda. The test read then runs that lambda alone (`--icv-lambdas`).

    python tools/select_icv_lambda.py --readout results/method/L31c36/k10_increment_into_K5_validation_readout.json \\
        --K-full 10 --json-out results/method/L31c36/icv_lambda_K10_into_K5.json
    python tools/select_icv_lambda.py --read results/method/L31c36/icv_lambda_K10_into_K5.json   # prints lambda
    python tools/select_icv_lambda.py --first-lambda results/method/L31c36/icv_vectors_K10_into_K5.json
                                                                        # the sidecar grid's smallest lambda > 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def lambda_table(readout, k_full):
    """{lambda: {accuracy, nll}} for the `ICV-K<full>` family from a receiver
    readout's `vector_alphas`, lambda = 0 (the gate) excluded."""
    va = readout.get("vector_alphas")
    if not isinstance(va, dict):
        raise SystemExit(f"the readout has no 'vector_alphas' (keys: {sorted(readout)})")
    curve = va.get(f"ICV-K{int(k_full)}")
    if not isinstance(curve, dict):
        return {}
    out = {}
    for key, row in curve.items():
        lam = float(key)
        if lam == 0.0:
            continue
        out[lam] = {"accuracy": float(row["accuracy"]), "nll": float(row["nll"])}
    return out


def select_lambda(table):
    """(lambda, best NLL): the lowest validation NLL, ties to the SMALLER
    lambda."""
    if not table:
        raise ValueError("empty table")
    best = min(r["nll"] for r in table.values())
    return min(l for l, r in table.items() if r["nll"] == best), best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--readout", help="analyze_k0_receiver --json-out of the VALIDATION receiver run")
    ap.add_argument("--K-full", type=int, default=None)
    ap.add_argument("--json-out")
    ap.add_argument("--read", help="print the lambda a --json-out file chose")
    ap.add_argument("--first-lambda", help="print the smallest lambda > 0 of a run_icv_increment sidecar's grid")
    args = ap.parse_args(argv)
    if args.read:
        doc = json.loads(Path(args.read).read_text(encoding="utf-8"))
        if "selected_lambda" not in doc:
            raise SystemExit(f"{args.read}: no 'selected_lambda' (keys: {sorted(doc)})")
        print(f"{float(doc['selected_lambda']):g}")
        return 0
    if args.first_lambda:
        doc = json.loads(Path(args.first_lambda).read_text(encoding="utf-8"))
        grid = [float(x) for x in ((doc.get("runner") or {}).get("lambda_grid") or []) if float(x) > 0]
        if not grid:
            raise SystemExit(f"{args.first_lambda}: no positive runner.lambda_grid "
                             f"(runner keys: {sorted(doc.get('runner') or {})})")
        print(f"{min(grid):g}")
        return 0
    if not (args.readout and args.K_full is not None and args.json_out):
        raise SystemExit("need --readout, --K-full and --json-out (or --read / --first-lambda)")
    readout = json.loads(Path(args.readout).read_text(encoding="utf-8"))
    table = lambda_table(readout, args.K_full)
    if not table:
        raise SystemExit(f"{args.readout}: no `ICV-K{args.K_full}` family with a lambda > 0 "
                         f"(vector_families: {readout.get('vector_families')}); the receiver did not "
                         "run with --icv-vectors")
    sel, best = select_lambda(table)
    n = int(readout.get("n_queries_per_seed", 0)) * len(readout.get("seeds") or [])
    print("=" * 78)
    print(f"IN-CONTEXT VECTOR LAMBDA ON VALIDATION -- K={args.K_full}, {n} queries")
    print("=" * 78)
    print(f"    {'lambda':>7} {'NLL':>10} {'accuracy':>10}")
    for l in sorted(table):
        r = table[l]
        print(f"    {l:>7g} {r['nll']:>10.4f} {r['accuracy']:>10.4f}" + ("   <- chosen" if l == sel else ""))
    print(f"\n  chosen: lambda {sel:g} (NLL {best:.4f}; ties go to the smaller lambda)")
    print("  ⚠ chosen ON validation: in sample; the test read is the one that counts")
    out = {"family": f"ICV-K{int(args.K_full)}", "selected_lambda": float(sel),
           "rule": "lowest validation NLL over the grid, lambda = 0 excluded; ties to the smaller lambda",
           "table": {f"{l:g}": table[l] for l in sorted(table)},
           "n_queries": n, "readout": str(args.readout), "K": int(args.K_full)}
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"  [output] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
