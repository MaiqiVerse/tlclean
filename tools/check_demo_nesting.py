"""Is the K=10 demonstration draw a superset of the K=5 one? ZERO GPU.

WHY THIS RUNS BEFORE THE INTERVENTION IS BUILT. The proposed arm is a K=5
receiver whose carrier heads additionally read the K=10 increment, and its
baseline arm is "everyone sees only the K=5 part". That baseline is only the
registered K=5 condition if the K=5 demonstrations are literally the ones
inside the K=10 prompt. If the two draws merely overlap, the arm is comparing
against a set nothing else in the project has measured.

WHAT THE CODE SAYS, AND WHY THAT IS NOT ENOUGH. `_get_fewshot_examples` seeds
one Random, walks the classes in sorted order, shuffles each class's candidate
list and takes the first k. The shuffle does not depend on k, so per class the
first five of the K=10 selection should BE the K=5 selection. That is a
behavioural claim about a draw with a deduplication step, an allowed-class
filter and a validation reservation in front of it, and working rules 3.2 is
explicit that consuming an artifact means reading the producer rather than
reasoning about it. So it is measured.

WHAT IS EXPECTED TO DIFFER, AND WHY IT MATTERS ANYWAY. The final
`rng.shuffle(demos)` runs on 180 items at K=5 and 360 at K=10, so the ORDER
differs even when the set nests. The K=5 demonstrations are therefore
SCATTERED through the K=10 prompt, not sitting in a contiguous prefix -- which
means the visibility mask for the proposed arm cannot be a split point and has
to be a per-column set. Better to learn that here than from a mask that
silently opens the wrong 180 columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.prereg_config import REGISTERED_SEEDS  # noqa: E402


def nesting_faults(small, large, *, k_small, k_large):
    """Faults for one seed's pair of (class, text) demonstration lists."""
    bad = []
    s_set, l_set = set(small), set(large)
    if len(s_set) != len(small):
        bad.append(f"K={k_small} draw holds {len(small) - len(s_set)} "
                   "duplicate (class, text) pairs, so 'set' is the wrong word "
                   "for it")
    if len(l_set) != len(large):
        bad.append(f"K={k_large} draw holds {len(large) - len(l_set)} "
                   "duplicates")
    missing = s_set - l_set
    if missing:
        bad.append(
            f"{len(missing)} of the {len(s_set)} K={k_small} demonstrations "
            f"are NOT in the K={k_large} draw (e.g. class "
            f"{sorted(missing)[0][0]}). The two draws overlap rather than "
            "nest, so an arm that opens 'the K=5 part' of the K=10 prompt "
            "would be opening a set the project has never measured")
    return bad


def order_report(small, large):
    """Where the small draw's items sit inside the large one.

    Returns (positions, is_contiguous_prefix). The proposed mask can be a
    split point only if the answer is a contiguous prefix.
    """
    idx = {d: i for i, d in enumerate(large)}
    pos = sorted(idx[d] for d in small if d in idx)
    return pos, pos == list(range(len(pos)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--task", default="trec_fine_per_class")
    ap.add_argument("--k-small", type=int, default=5)
    ap.add_argument("--k-large", type=int, default=10)
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    from tools.prereg_task import prefix_demo_rows
    from tools.probe_prototype_shrinkage import manifest_reservation

    print("=" * 78)
    print(f"IS THE K={args.k_large} DRAW A SUPERSET OF K={args.k_small}?  "
          "[ZERO GPU]")
    print("=" * 78)

    out = {"task": args.task, "k_small": args.k_small,
           "k_large": args.k_large, "per_seed": {}}
    faults = []
    for s in REGISTERED_SEEDS:
        # THE SAME RESERVATION BOTH TIMES. Each seed reserved its own 144
        # before its demonstrations were drawn (14.0a); using a different
        # exclusion for the two K would change the candidate pool and the
        # nesting question would be about the exclusion, not about k.
        excl = manifest_reservation(args.query_manifest, demo_seed=s)
        small = prefix_demo_rows(args.task, args.k_small, [s],
                                 excluded_docs=excl)[s]
        large = prefix_demo_rows(args.task, args.k_large, [s],
                                 excluded_docs=excl)[s]
        bad = nesting_faults(small, large,
                             k_small=args.k_small, k_large=args.k_large)
        pos, contig = order_report(small, large)
        faults += [f"seed {s}: {b}" for b in bad]
        print(f"\n  seed {s}: {len(small)} vs {len(large)} demonstrations")
        print(f"    nests: {'YES' if not bad else 'NO'}")
        if pos:
            print(f"    the K={args.k_small} demonstrations sit at positions "
                  f"{pos[:6]}...{pos[-3:]} of {len(large)}")
            print(f"    contiguous prefix: {'YES' if contig else 'NO'}"
                  + ("" if contig else "   -> the mask must be a per-column "
                                       "SET, not a split point"))
        out["per_seed"][str(s)] = {"n_small": len(small), "n_large": len(large),
                                   "nests": not bad,
                                   "contiguous_prefix": bool(contig),
                                   "positions": pos}

    print("\n" + "=" * 78)
    if faults:
        print("  REFUSING: the draws do not nest")
        for f in faults:
            print(f"    - {f}")
        return 1
    contigs = [v["contiguous_prefix"] for v in out["per_seed"].values()]
    print(f"  [PASS] every K={args.k_small} demonstration is in the "
          f"K={args.k_large} draw, all {len(REGISTERED_SEEDS)} seeds")
    if not any(contigs):
        print("  ⚠ but NONE of them is a contiguous prefix: the final shuffle "
              "runs on a list")
        print("    whose length depends on k, so the order differs even where "
              "the set nests.")
        print("  ⟹ the proposed arm's visibility mask must open a SET of "
              "columns, not a span.")
    elif all(contigs):
        print("  and each is a contiguous prefix, so a split point suffices.")
    else:
        print(f"  ⚠ MIXED: {sum(contigs)} of {len(contigs)} seeds are "
              "contiguous. Do not build a mask that is right for some seeds.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2),
                                       encoding="utf-8")
        print(f"\n  [output] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
