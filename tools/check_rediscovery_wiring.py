"""Does every artifact a later phase READS get WRITTEN by an earlier one?

ZERO GPU, and it runs no experiment: it resolves the shell variables in the
rediscovery drivers under one cell's environment and compares the names.

WHY. The standalone probe drivers build their input names from ${MTAG}, ${K}
and ${NQ}, while rediscover_cell.sh writes the head sets and the accuracy json
that they read. Those two spellings agreed only because I compared them by
hand, and one of them was already wrong: llama31.sh writes
`check_model_accuracy_<tag>_K<k>_top8_n<nq>.json` while every probe driver
reads `check_model_accuracy_K<k>_top8_<mtag>_n<nq>.json`. A cell wired that
way runs for an hour, writes a file nobody opens, and the next phase aborts on
a missing input -- or worse, finds a stale one from another cell.

The check is textual because the names are: it extracts each driver's
assignments, expands them under the cell environment, and reports any
`results/...` path a driver READS that no phase of the cell WRITES.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("script")

# The later phases, in the order rediscover_cell.sh runs them.
PROBES = ["probe_kappa_query_dependence.sh", "probe_kappa_position_vs_content.sh",
          "probe_component_budget.sh", "probe_contrast_budget.sh",
          "probe_offset_sweep.sh", "probe_pair_mode_control.sh",
          "probe_demo_causal.sh", "probe_demo_causal_vpatch.sh",
          "probe_label_repair.sh", "probe_substrate_size_curve.sh"]


def resolve(path, env, argv=()):
    """Every `results/...` literal in the file, with the variables expanded.

    Runs the file's ASSIGNMENTS ONLY, in a shell, then echoes each candidate
    string. Expanding by hand would mean reimplementing shell substitution,
    which is how a checker ends up disagreeing with the thing it checks.
    """
    text = Path(path).read_text(encoding="utf-8")
    assigns = [l for l in text.split("\n")
               if re.match(r'^\s*(: "\$\{[A-Z_]+=|[A-Z_][A-Za-z0-9_]*=)', l)
               # `$(` and backticks would run commands; `;;` and a bare `)`
               # are case-branch syntax, which is a parse error once the
               # surrounding `case` is filtered out -- and a parse error means
               # the whole extraction silently yields nothing, which is how
               # this reported "0 paths" for a file full of them.
               and "$(" not in l and "`" not in l
               and not l.rstrip().endswith(";;")]
    cands = sorted(set(re.findall(r'"(results/[^"]*)"', text)
                       + re.findall(r'"(\$\{[A-Za-z_][^"]*)"', text)))
    if not cands:
        return set()
    prog = "\n".join(assigns) + "\n" + "\n".join(
        f'printf "%s\\n" "{c}"' for c in cands)
    # POSITIONAL ARGS MATTER. rediscover_cell.sh takes MTAG/K/SEED as $1..$3,
    # and scanning its assignments without them resolves MTAG_BASE to the
    # empty string -- every path then carries a tag no driver uses, and the
    # checker reports twelve faults that are its own.
    r = subprocess.run(["bash", "-c", prog, "bash", *argv],
                       capture_output=True, text=True,
                       env={**os.environ, **env})
    return {l for l in r.stdout.split("\n")
            if l.startswith("results/") and "*" not in l}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mtag-base", default="L31c36")
    ap.add_argument("--K", default="5")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--nq", default="250")
    ap.add_argument("--task", default="trec_fine_per_class",
                    help="any tools/rediscover_status.TASK_TAGS task; its "
                         "tag is folded into MTAG exactly as the cell does")
    args = ap.parse_args(argv)

    # THE TAG AND THE CALIBRATION DIRECTORY COME FROM ONE PLACE. The cell
    # script asks rediscover_status for the tag through a command
    # substitution, which `resolve` deliberately does not run, so the value
    # is supplied here from the same table instead.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.rediscover_status import calib_dir, task_tag
    tag = task_tag(args.task)
    mtag = f"{args.mtag_base}{tag}s{args.seed}"
    from tools.model_tags import resolve as resolve_tag, se_flags
    spec = resolve_tag(args.mtag_base)
    env = {"MODEL": spec["model"], "SE": se_flags(spec), "MTAG": mtag,
           "K": args.K, "NQ": args.nq, "SEED": args.seed,
           "TASK": args.task, "TASKTAG": tag,
           "CALIB_DIR": str(calib_dir(args.mtag_base, args.task))}

    print("=" * 78)
    print(f"REDISCOVERY WIRING -- cell {mtag} K={args.K}  [ZERO GPU]")
    print("=" * 78)

    produced = resolve(SCRIPT / "rediscover_cell.sh", env,
                       argv=(args.mtag_base, args.K, args.seed))
    print(f"\n  rediscover_cell.sh names {len(produced)} results/ paths")
    for p in sorted(produced):
        print(f"    {p}")

    bad = []
    print("\n  what each later driver resolves:")
    for name in PROBES:
        p = SCRIPT / name
        if not p.is_file():
            bad.append(f"{name}: not found")
            continue
        got = resolve(p, env)
        # An input is a path the driver does NOT create: heuristically, the
        # head sets and the accuracy json. Reported rather than guessed at.
        shared = got & produced
        inputs = {g for g in got
                  if "tl_heads_" in g or "check_model_accuracy" in g
                  or "s0_baseline" in g}
        # A path with an empty segment came from a variable set inside a
        # loop (a slice size, a head tag), which this scan cannot see. That
        # is a limit of the checker, not a fault in the wiring, and calling
        # it one would train the reader to ignore the output.
        unresolved = {g for g in inputs if "__" in g or g.endswith("_.json")}
        missing = inputs - produced - unresolved
        flag = "" if not missing else "   <-- READS WHAT NOTHING WRITES"
        if unresolved and not missing:
            flag = f"   ({len(unresolved)} name(s) set in a loop, not checked)"
        print(f"    {name:<38} {len(got):>2} paths, "
              f"{len(shared)} shared{flag}")
        for m in sorted(missing):
            print(f"        {m}")
            bad.append(f"{name} reads {m}, which no phase of the cell writes")

    print()
    if bad:
        print(f"  {len(bad)} wiring fault(s):")
        for b in bad:
            print(f"    - {b}")
        return 1
    print("  every head set and accuracy json a later driver reads is one an "
          "earlier phase writes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
