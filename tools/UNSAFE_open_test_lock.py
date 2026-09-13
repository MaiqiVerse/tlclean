"""UNSAFE: open the test lock with fewer preconditions than 13.4(4) registers.

WHAT THIS IS FOR. The viability gate returned STOP, 13.6.7 terminates the path
there, and the PI decided after being told twice to run the L3.1 test forward
anyway; that decision and its reasons are recorded in 14.0b-19, written BEFORE
any test number existed. This script is the last mechanical step of it.

WHAT IS UNSAFE ABOUT IT, PRECISELY -- one thing, not a general loosening:

    5(4) freezes one gamma* per setting/implementation and 13.4(4) freezes all
    three before ANY test forward, because the lock is not per split: a freeze
    that opens test_seed opens test_common too. Only gamma*_group was ever
    measured. gamma*_L2 (the L2 mechanism setting) and gamma*_head (Method
    A-head) were never run.

HOW THE GAP IS MADE UNUSABLE RATHER THAN HIDDEN. The two unmeasured settings
are written as a SENTINEL OUTSIDE THE REGISTERED GRID. load_frozen_gammas
requires each key to be a number, so the file parses and the L3.1-group run --
which reads only gamma_group -- proceeds. But the probe refuses any gamma not
in GAMMA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0), so a run whose arm maps to one of
those keys CANNOT use the sentinel: it fails at the point of use, before doing
anything. A plain 0.0 would have been the dangerous choice, because 0.0 is
itself a meaningful decision -- the identity operator -- and nobody could tell
a placeholder from a frozen one.

EVERYTHING IT WRITES SAYS SO. The gammas artifact and the freeze manifest both
carry an `unsafe` block naming what was not verified, and the file names carry
an UNSAFE_ prefix, so a result traced back to them cannot be mistaken for one
produced under 13.4(4).

WHAT IT DOES NOT DO. It does not touch the production lock, the probe, or any
registered criterion. Nothing here is imported by another tool. Delete this
file and the project is exactly as it was.

Usage (the acknowledgement is required and is not a formality):
    python tools/UNSAFE_open_test_lock.py \\
        --gamma-group 1.0 \\
        --query-manifest results/prereg_method_A_query_manifest.json \\
        --carriers results/carriers_full_validation.json \\
        --label-space results/label_space_llama31.json \\
        --spec-freeze results/baseline_spec_freeze_v2.json \\
        --out-dir results \\
        --i-am-bypassing-13-4-4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.build_query_manifest import (GAMMA_KEYS,  # noqa: E402
                                        write_freeze_manifest)
from tools.prototype_targets import GAMMA_GRID  # noqa: E402

# Outside GAMMA_GRID on purpose: present enough for load_frozen_gammas,
# unusable at the point a run would need it.
SENTINEL = -1.0

UNSAFE = {
    "bypassed": "prereg_method_A.md 13.4(4) -- 'Method A 两版与全部 baseline "
                "validation' before the freeze, and 5(4)'s three gamma*",
    "measured": ["gamma_group"],
    "unmeasured": ["gamma_L2", "gamma_head"],
    "why_unmeasured": {
        "gamma_L2": "the L2 mechanism setting was never run on validation",
        "gamma_head": "Method A-head was never run on validation; the "
                      "viability gate returned STOP first",
    },
    "sentinel": SENTINEL,
    "sentinel_rationale": (
        f"outside the registered grid {list(GAMMA_GRID)}, so the probe's "
        "'unregistered gamma' check refuses any run that tries to use it. "
        "A placeholder of 0.0 would have been indistinguishable from a frozen "
        "decision, because 0.0 is the identity operator and a legitimate "
        "choice."),
    "authorised_by": "prereg_method_A.md 14.0b-19, recorded before any test "
                     "number existed",
    "what_this_does_not_license": (
        "Nothing about H4 or H4b. 13.6.7's path ended at the gate and running "
        "afterwards does not revive it. Results are 'STOP-following "
        "exploratory test readings' and may not enter 1.1."),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gamma-group", type=float, required=True)
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--carriers", required=True)
    ap.add_argument("--label-space", required=True)
    ap.add_argument("--spec-freeze", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--i-am-bypassing-13-4-4", action="store_true")
    args = ap.parse_args(argv)

    if not getattr(args, "i_am_bypassing_13_4_4"):
        raise SystemExit(
            "refusing without --i-am-bypassing-13-4-4.\n"
            "This writes a freeze that opens the ONE-SHOT test split with two "
            "of three gamma* never measured. 14.0b-19 records the decision; "
            "the flag records that whoever ran it knew which one they were "
            "carrying out.")

    if args.gamma_group in (None,) or args.gamma_group not in GAMMA_GRID:
        raise SystemExit(
            f"--gamma-group {args.gamma_group} is not in the registered grid "
            f"{list(GAMMA_GRID)}. The one value this file DOES claim to have "
            "measured has to be a real frozen choice, or there is nothing left "
            "that is honest about it.")

    out = Path(args.out_dir)
    gpath = out / "UNSAFE_frozen_gammas.json"
    fpath = out / "UNSAFE_freeze_manifest.json"

    doc = {k: (float(args.gamma_group) if k == "gamma_group" else SENTINEL)
           for k in GAMMA_KEYS}
    doc["spec"] = ("prereg_method_A.md 5(4): one frozen gamma* per "
                   "setting/implementation, three-seed-mean validation NLL")
    doc["unsafe"] = UNSAFE
    gpath.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    write_freeze_manifest(
        fpath,
        query_manifest=args.query_manifest,
        carriers=args.carriers,
        label_space=args.label_space,
        gammas=str(gpath),
        spec_freeze=args.spec_freeze,
    )
    # The manifest's own schema has no room for this, so it is appended after
    # write_freeze_manifest rather than smuggled into a role. The hashes it
    # computed are untouched.
    man = json.loads(fpath.read_text(encoding="utf-8"))
    man["unsafe"] = UNSAFE
    fpath.write_text(json.dumps(man, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print("=" * 78)
    print("UNSAFE TEST LOCK OPENED")
    print("=" * 78)
    print(f"  gamma_group  {args.gamma_group}   (measured, "
          "results/method_a_full_selection.json)")
    for k in UNSAFE["unmeasured"]:
        print(f"  {k:12} {SENTINEL}  SENTINEL -- {UNSAFE['why_unmeasured'][k]}")
    print(f"\n  the sentinel is outside {list(GAMMA_GRID)}, so any run that "
          "tries to USE it is refused")
    print("  by the probe's registered-grid check. It opens the lock; it "
          "cannot supply a value.")
    print(f"\n  [output] {gpath}")
    print(f"  [output] {fpath}")
    print("\n  NOT verified, and recorded as such in both files:")
    print("    - Method A-head validation (13.4(4) 'Method A 两版')")
    print("    - the L2 mechanism setting")
    print("    - all six baseline validations (the gate said STOP)")
    print("\n  ⚠ This freeze opens test_common as well as test_seed. The lock "
          "is not per split.")
    print("  ⚠ Results from it are STOP-following EXPLORATORY readings. They "
          "are not H4, not H4b,")
    print("  ⚠ and may not enter section 1.1. See 14.0b-19.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
