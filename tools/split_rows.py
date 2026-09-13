"""The rows a receiver run scores: validation, or the ONE-SHOT test split.

ONE PLACE for the three receiver runners (run_k0_receiver,
run_k10_increment, probe_carrier_direct_response) to load their query rows,
so the test lock is consulted the same way by all of them and none of them
can reach `test_seed` by a private route.

    validation   per-seed, from TRAIN, open (sections 2.4, 14.0a)
    test_seed    one demo seed's 250 draws; refused by load_split unless a
                 freeze manifest that binds THIS run's carriers and label
                 space still hashes correctly (sections 2.1, 13.4(4)). This
                 checkout has no code-stage spec freeze, so the only way
                 through is tools/UNSAFE_run_test_forward.py, which discards
                 exactly that one blocker and records having done so.

The 2026-09-09 decision that reads test for the section 13.5 constructions
is prereg 14.0b-23: Method A's path ended at the viability gate (13.6.7) and
its STOP-following test reading (14.0b-19) confirmed it; the selective TL
memory constructions are the paper's method from here, their test readings
are optional/descriptive under 13.5.4, and their configuration -- the
full-validation carriers; TSLA read at the upstream's own alpha = 1, the
grid descriptive -- was frozen on validation before any test row was loaded.
"""

from __future__ import annotations

SPLITS = ("validation", "test_seed")


def expected_roles_for(split, *, carriers, label_space):
    """What the freeze must bind for a test read; nothing for validation.

    `carriers` is the path the run actually loads (the bundle). The lock
    compares it with the freeze's `carriers` role, so a freeze written for a
    per-seed extract cannot open a run that reads the bundle, and vice versa.
    """
    if split == "validation":
        return None
    return {"carriers": str(carriers), "label_space": str(label_space)}


def rows_for(query_manifest, split, seed, *, freeze_manifest=None,
             carriers=None, label_space=None):
    """The rows one seed scores under `split`, through load_split's lock."""
    from tools.build_query_manifest import load_split
    if split not in SPLITS:
        raise ValueError(f"split {split!r}: expected one of {SPLITS}")
    if split != "validation" and (carriers is None or label_space is None):
        raise ValueError("a test read must say which carriers and label "
                         "space it loads, so the freeze can be held to them")
    return list(load_split(
        query_manifest, split, demo_seed=seed,
        freeze_manifest=freeze_manifest,
        expected_roles=expected_roles_for(split, carriers=carriers,
                                          label_space=label_space)))


def add_split_arguments(ap):
    """--split and --freeze-manifest, worded once."""
    ap.add_argument("--split", choices=SPLITS, default="validation",
                    help="validation (default) or the ONE-SHOT test_seed "
                         "split. test_seed needs --freeze-manifest and, in "
                         "this checkout, tools/UNSAFE_run_test_forward.py "
                         "(prereg 14.0b-23)")
    ap.add_argument("--freeze-manifest", default=None,
                    help="the freeze manifest that binds this run's carriers "
                         "and label space; required for --split test_seed")


def split_meta(args):
    """What the artifact records about which rows it scored."""
    return {"split": args.split,
            "freeze_manifest": (str(args.freeze_manifest)
                                if args.freeze_manifest else None),
            "test_read_decision": ("prereg_method_A.md 14.0b-23: optional/"
                                   "descriptive test reading of a 13.5 "
                                   "construction, configuration frozen on "
                                   "validation" if args.split != "validation"
                                   else None)}
