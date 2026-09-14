"""UNSAFE: run one Method A test forward without a code-stage spec freeze.

WHY THIS EXISTS, AND WHY IT IS NOT `UNSAFE_open_test_lock.py`. That file
writes the two artifacts the lock reads (the gammas and the freeze manifest)
and it works: `freeze_gamma_path` and `load_frozen_gammas` accept them. It
does NOT open the lock, because `freeze_manifest_blockers` asks one more
question -- is the spec freeze it points at at the CODE stage -- and the
answer here is no:

    results/baseline_spec_freeze_v2.json is a 'spec'-stage freeze. Section
    2.1 requires the analysis code frozen before any test prediction, and
    only the code stage records that

and `code_stage_blockers()` returns eight reasons why this checkout cannot be
frozen at the code stage: four runners with no `runner_commit`, the frozen
carriers file, the 6.1 verification report, ZeroTuning's adaptation diff, and
the 5(4) validation selection.

THE ROUTE NOT TAKEN. `build_freeze("code")` would produce a record, and
`validate_freeze(rec, require_baselines_ready=False)` tolerates
NOT-IMPLEMENTED opponents, so a code-stage freeze written here would not lie
about the baselines. It would still fail on `artifacts`: all four
code-stage artifacts are absent, so the record hashes nothing and
`validate_freeze` says so. Making them exist means writing a validation
selection and a verification report for six baselines that were never run --
i.e. fabricating the exact documents whose whole job is to be trustworthy.
A forged freeze is worse than an open bypass, because it is indistinguishable
from a real one forever after.

WHAT THIS DOES INSTEAD. It wraps ONE named function --
`build_query_manifest.freeze_manifest_blockers`, the function the test lock
consults -- and discards from its verdict exactly ONE blocker, the spec-stage
one quoted above. The whitelist is fail-closed:

  * every OTHER blocker is returned unchanged, so the run still refuses on
    drift, on a role the freeze does not pin, on a freeze built for different
    carriers, on an unparseable gamma artifact;
  * the discarded blocker, and `code_stage_blockers()` in full, are printed
    and written to a bypass record next to the output;
  * if the production message this whitelist keys on ever changes, the marker
    is no longer found in the source and the tool refuses to run at all
    rather than silently matching nothing.

NOTHING FALSE IS WRITTEN TO `results/`. No freeze claims a stage it is not
at, no baseline is recorded READY, no artifact is invented. The bypass lives
in this file, under this name, for the length of one run.

WHAT IT DOES NOT LICENSE. 13.6.7 ended the method path at the viability
gate; running afterwards does not revive it. Anything produced through here
is a STOP-following exploratory test reading -- not H4, not H4b, not for
section 1.1. See 14.0b-19.

SINCE 2026-09-09 (prereg 14.0b-23) THE SAME BYPASS SERVES THE SECTION 13.5
RECEIVERS. Method A's path ended at the viability gate and its STOP-following
test reading confirmed it; the selective TL memory constructions are the
paper's method from here, and 13.5.4 lets them be read on test as optional/
descriptive once their configuration is frozen on validation. `--runner`
picks which main() runs behind the bypass -- the whitelist, the record and
every refusal are unchanged, and the record names the runner and the
decision it was run under.

Usage (everything after `--` goes to the chosen runner):

    python tools/UNSAFE_run_test_forward.py --i-am-bypassing-13-4-4 -- \\
        --arm proto_shrink --carrier-impl gqa_group_v --allow-kv-group \\
        --split test_seed --demo-seeds 43 --gamma 1.0 \\
        --freeze-manifest results/UNSAFE_freeze_manifest.json \\
        --out results/UNSAFE_method_a_test_seed43.json ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# The exact sentence freeze_manifest_blockers emits for a non-code freeze.
# Keyed on a distinctive middle span rather than the whole message, because
# the path and the stage name are interpolated into it.
SPEC_STAGE_MARKER = ("-stage freeze. Section 2.1 requires the analysis code "
                     "frozen before any test prediction, and only the code "
                     "stage records that")

# Where that sentence must still live. If it does not, the whitelist below
# matches nothing -- which is SAFE (everything stays fatal) but useless, and
# silently useless is how a bypass turns into a mystery.
SOURCE = REPO / "tools" / "build_query_manifest.py"

# Which main() runs behind the bypass. Module paths, imported only after the
# bypass is installed, so a runner's own imports cannot consult the lock
# before the record exists.
RUNNERS = {
    "prototype": ("tools.probe_prototype_shrinkage",
                  "Method A's viability forward (14.0b-19)"),
    "k10_increment": ("tools.run_k10_increment",
                      "a K receiver whose carriers read the increment to a "
                      "larger K (RESULTS 51 / 55; 14.0b-23)"),
    "k0_receiver": ("tools.run_k0_receiver",
                    "the K=0 receiver with selective K=5 memory (RESULTS "
                    "45 / 48; 14.0b-23)"),
    "carrier_direct": ("tools.probe_carrier_direct_response",
                       "the carriers' direct write under the increment "
                       "masks (RESULTS 52 / 53; 14.0b-23)"),
    "tl_class_attribution": ("tools.probe_tl_class_attribution",
                             "the native all-heads DLA on the test rows: "
                             "the carriers' direct-write ceiling (RESULTS "
                             "53.6; 14.0b-23)"),
}


def marker_faults():
    """Why this whitelist cannot be trusted against the current source."""
    try:
        src = SOURCE.read_text(encoding="utf-8")
    except OSError as e:
        return [f"{SOURCE}: unreadable -- {e}"]
    # The source wraps the sentence across lines AND across adjacent string
    # literals, so collapsing whitespace is not enough -- the quote pair has
    # to go too, or the marker never matches and this check reports drift
    # that is not there.
    flat = re.sub(r'"\s*"', "", " ".join(src.split()))
    if " ".join(SPEC_STAGE_MARKER.split()) not in flat:
        return [f"{SOURCE} no longer contains the spec-stage message this "
                "bypass whitelists. It was rewritten, and a whitelist that "
                "matches nothing would make every blocker fatal without "
                "saying why. Re-read freeze_manifest_blockers and update "
                "SPEC_STAGE_MARKER."]
    return []


def is_the_accepted_blocker(b) -> bool:
    """ONLY the spec-stage one -- or its SPEC_FREEZE=none form, which
    carries the same sentence (prereg 14.0b-30). Everything else stays
    fatal."""
    return isinstance(b, str) and SPEC_STAGE_MARKER in b


def install_bypass(record):
    """Patch freeze_manifest_blockers to drop one blocker, keeping the rest.

    Returns nothing; `record` is filled in place so the caller can write it
    even if the probe later fails.
    """
    import tools.build_query_manifest as bqm
    real = bqm.freeze_manifest_blockers

    def wrapper(path, *, expected_roles=None):
        bad = real(path, expected_roles=expected_roles)
        dropped = [b for b in bad if is_the_accepted_blocker(b)]
        kept = [b for b in bad if not is_the_accepted_blocker(b)]
        record["calls"].append({
            "freeze_manifest": str(path),
            "expected_roles": {k: str(v)
                               for k, v in (expected_roles or {}).items()},
            "blockers_discarded": dropped,
            "blockers_kept": kept,
        })
        if dropped:
            print(f"  [UNSAFE] discarded {len(dropped)} blocker(s) from "
                  f"{path}:")
            for b in dropped:
                print(f"    - {b}")
        if kept:
            print(f"  [UNSAFE] KEPT {len(kept)} blocker(s); this bypass does "
                  "not cover them, so the run will refuse:")
            for b in kept:
                print(f"    - {b}")
        return kept

    bqm.freeze_manifest_blockers = wrapper
    return real


def out_path_of(rest):
    """The --out the probe was given, or None."""
    for i, a in enumerate(rest):
        if a == "--out" and i + 1 < len(rest):
            return rest[i + 1]
        if a.startswith("--out="):
            return a.split("=", 1)[1]
    return None


def split_of(rest):
    for i, a in enumerate(rest):
        if a == "--split" and i + 1 < len(rest):
            return rest[i + 1]
        if a.startswith("--split="):
            return a.split("=", 1)[1]
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="everything after -- is passed to probe_prototype_shrinkage")
    ap.add_argument("--i-am-bypassing-13-4-4", action="store_true")
    ap.add_argument("--runner", choices=sorted(RUNNERS), default="prototype",
                    help="which main() runs behind the bypass; the record "
                         "names it")
    ap.add_argument("--bypass-record", default=None,
                    help="where to write the record of what was discarded; "
                         "defaults to <out>.bypass.json")
    ap.add_argument("--self-test", action="store_true")
    args, rest = ap.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]

    if args.self_test:
        return self_test()

    if not getattr(args, "i_am_bypassing_13_4_4"):
        raise SystemExit(
            "refusing without --i-am-bypassing-13-4-4.\n"
            "This runs a forward on the ONE-SHOT test split with no "
            "code-stage freeze behind it. 14.0b-19 records the decision; the "
            "flag records that whoever ran it knew which one they were "
            "carrying out.")

    faults = marker_faults()
    if faults:
        raise SystemExit("REFUSING TO RUN:\n  " + "\n  ".join(faults))

    split = split_of(rest)
    if not split or not split.startswith("test"):
        raise SystemExit(
            f"--split {split!r}: this wrapper exists only to get past the "
            "TEST lock. On validation the probe needs no bypass -- run it "
            "directly, so that nothing produced on validation carries an "
            "UNSAFE provenance it did not need.")

    out = out_path_of(rest)
    if not out:
        raise SystemExit("the probe needs --out, and this wrapper needs to "
                         "know it to name the bypass record")
    if not Path(out).name.startswith("UNSAFE_"):
        raise SystemExit(
            f"--out {out}: the basename must start with 'UNSAFE_'. A result "
            "produced through this file must be unmistakable in a directory "
            "listing, three months from now, to someone who never read this "
            "docstring.")

    from tools.baselines.freeze_baseline_spec import code_stage_blockers
    record = {
        "kind": "unsafe_test_lock_bypass",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "tools/UNSAFE_run_test_forward.py",
        "bypassed": (
            "prereg_method_A.md 2.1 / 13.4(4): the test splits require a "
            "CODE-stage spec freeze, and this checkout has none"),
        "whitelist": (
            "exactly one blocker -- the 'is a <stage>-stage freeze' message. "
            "Every other blocker (drift, role binding, unparseable gammas) is "
            "returned unchanged and refuses the run."),
        "code_stage_blockers_at_run_time": list(code_stage_blockers()),
        "runner": args.runner,
        "runner_module": RUNNERS[args.runner][0],
        "runner_what": RUNNERS[args.runner][1],
        "not_licensed": (
            "13.6.7 ended the method path at the viability gate. Results from "
            "this run are STOP-following exploratory test readings: not H4, "
            "not H4b, not for section 1.1. See 14.0b-19."
            if args.runner == "prototype" else
            "prereg 14.0b-23: an optional/descriptive test reading of a "
            "section 13.5 construction, configuration frozen on validation "
            "(full-validation carriers; TSLA read at the upstream's own "
            "alpha = 1, the grid descriptive). Not "
            "H1-H6, no Holm family, no claim of beating a registered "
            "baseline; the test split was already read once under 14.0b-19."),
        "probe_argv": list(rest),
        "calls": [],
    }
    rec_path = Path(args.bypass_record or (str(out) + ".bypass.json"))

    print("=" * 78)
    print("UNSAFE TEST FORWARD -- no code-stage freeze")
    print("=" * 78)
    print(f"  runner     {args.runner}  ({RUNNERS[args.runner][1]})")
    print(f"  split      {split}")
    print(f"  out        {out}")
    print(f"  record     {rec_path}")
    print(f"\n  code_stage_blockers() says this checkout cannot be frozen at "
          f"the code stage, {len(record['code_stage_blockers_at_run_time'])} "
          "reasons:")
    for b in record["code_stage_blockers_at_run_time"]:
        print(f"    - {b}")
    print("\n  None of them is fixed by running this. They are recorded so a "
          "reader of the")
    print("  output knows exactly what was not in place when the number was "
          "produced.")
    print("=" * 78)

    install_bypass(record)
    try:
        import importlib
        rc = importlib.import_module(RUNNERS[args.runner][0]).main(rest)
    finally:
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec_path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"\n  [output] {rec_path}")

    print("\n  ⚠ STOP-following exploratory test reading. Not H4, not "
          "H4b, not for 1.1.")
    return rc


# ----------------------------------------------------------------- self test

def self_test() -> int:
    """Fixtures only -- no model, no GPU, no real artifacts.

    The arm that MUST be red is the third: a freeze whose registered file has
    been edited produces a drift blocker, which is not on the whitelist, so
    the bypass has to leave the lock shut. Without such an arm the whitelist
    would be untested in the only direction that matters.
    """
    import tempfile

    import tools.build_query_manifest as bqm
    from tools.test_probe_queries import write_freeze, write_manifest

    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {extra}" if extra and not cond else ""))

    check("the whitelisted message still exists in production",
          not marker_faults(), str(marker_faults()))

    with tempfile.TemporaryDirectory() as td:
        man = write_manifest(td)
        spec_freeze = write_freeze(td, man, stage="spec")

        # 1. WITHOUT the bypass the spec-stage freeze is refused.
        try:
            bqm.load_split(man, "test_seed", demo_seed=42,
                           freeze_manifest=spec_freeze)
            check("a spec-stage freeze is refused without the bypass", False,
                  "it opened the lock")
        except PermissionError as e:
            check("a spec-stage freeze is refused without the bypass",
                  "stage freeze" in str(e), str(e)[:140])

        # 2. WITH the bypass it opens, and the discarded blocker is recorded.
        record = {"calls": []}
        real = install_bypass(record)
        try:
            rows = bqm.load_split(man, "test_seed", demo_seed=42,
                                  freeze_manifest=spec_freeze)
            check("the bypass opens test_seed", len(rows) > 0)
            check("the discarded blocker is recorded",
                  record["calls"] and record["calls"][0]["blockers_discarded"],
                  json.dumps(record["calls"][:1])[:200])
            check("nothing else was discarded",
                  all(len(c["blockers_discarded"]) == 1
                      for c in record["calls"]))

            # 3. THE RED ARM. Edit a file the freeze registers: that is drift,
            #    which is not whitelisted, so the lock must stay shut.
            doc = json.loads(Path(spec_freeze).read_text(encoding="utf-8"))
            victim = Path(doc["roles"]["carriers"])
            victim.write_bytes(victim.read_bytes() + b" ")
            try:
                bqm.load_split(man, "test_seed", demo_seed=42,
                               freeze_manifest=spec_freeze)
                check("a tampered artifact still re-locks under the bypass",
                      False, "the bypass swallowed a drift blocker")
            except PermissionError as e:
                # The refusal has to be the DRIFT one. If it were the
                # spec-stage message the whitelist would simply have missed
                # it this time, which is a different bug wearing the same
                # green tick.
                check("a tampered artifact still re-locks under the bypass",
                      "may not open the test lock" in str(e)
                      and SPEC_STAGE_MARKER not in str(e), str(e)[:160])
            check("the kept blocker is recorded too",
                  any(c["blockers_kept"] for c in record["calls"]))
        finally:
            bqm.freeze_manifest_blockers = real

        # 4. After restoring, the lock is shut again.
        try:
            bqm.load_split(man, "test_seed", demo_seed=42,
                           freeze_manifest=spec_freeze)
            check("the patch is not permanent", False, "still open")
        except PermissionError:
            check("the patch is not permanent", True)

    # 5. Argument guards.
    for argv, why in (
            (["--i-am-bypassing-13-4-4", "--", "--split", "validation",
              "--out", "results/UNSAFE_x.json"], "validation is refused"),
            (["--i-am-bypassing-13-4-4", "--", "--split", "test_seed",
              "--out", "results/x.json"], "an un-prefixed --out is refused"),
            ([" --", "--split", "test_seed"], "no acknowledgement is refused"),
    ):
        try:
            main(argv)
            check(why, False, "it ran")
        except SystemExit as e:
            check(why, e.code not in (0, None), str(e)[:120])

    print("\n  " + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
