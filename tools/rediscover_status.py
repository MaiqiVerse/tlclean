"""Where did a rediscovery cell get to, and what is the next command? ZERO GPU.

A cell (rediscover_cell.sh MTAG K SEED) is seven phases of GPU work, hours
each, and a job can die at any step: time limit, OOM, a probe's own refusal.
The chain's `[ -f ]` guards let a re-run skip what exists, but "exists" is
not "finished" -- a JSON cut off mid-write exists -- and nothing told the
operator which phase to resume from. This reads the truth from three places
and prints one plan:

  artifacts   every file the cell writes, per phase, VALIDATED: a json must
              parse, a jsonl must hold the expected number of rows, an npz
              must load. `--fix` moves a broken one aside so the guard
              rebuilds it.
  ledger      results/rediscovery_ledger_<CELL>.jsonl, one line per step the
              cell script completed (or refused), with the job id -- the
              record that survives a renamed log.
  log         `--log slurm-<jobid>.out`: the last phase banner reached, the
              last step, and the failure the scheduler or Python printed.

The resume command is the cell script itself with the phase list cut at the
first phase that is not complete; nothing is re-derived here, the names are
the cell script's (mirrored in `cell_names`, and the test checks the mirror
against the script).

Also the ONE home of the task tag: `--task-tag TASK` prints the short tag the
cell script folds into MTAG for a non-TREC task, so the shell and this tool
cannot disagree on a name.

Run:
    python tools/rediscover_status.py --mtag-base L31c36 --K 5 --seed 42
    python tools/rediscover_status.py --mtag-base L31c36 --K 5 --seed 42 \\
        --log logs/L31c36_K5_s42_rediscover.out --fix
    python tools/rediscover_status.py --check results/x.json          # guard
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PHASES = ("discover", "forward", "causal", "kernel", "budget", "demo",
          "substrate")
SLICES = (16, 32, 48, 64, 128)
DEFAULT_TASK = "trec_fine_per_class"

# The tasks the chain may run (tasks/*_per_class), and the tag folded into
# MTAG for each. TREC keeps the bare tag so every existing L31c36 / SEc36
# name is unchanged. trec_selfextend is excluded on purpose (its own line).
TASK_TAGS = {
    "trec_fine_per_class": "",
    "banking77_per_class": "_bk77",
    "clinc150_per_class": "_cl150",
    "dbpedia14_per_class": "_dbp14",
    "yahoo_answers_per_class": "_yahoo",
    "yelp_full_per_class": "_yelp",
    "monk_per_class": "_monk",
    "monk_per_class_r1": "_monkr1",
    "monk_per_class_r2": "_monkr2",
    "monk_per_class_r3": "_monkr3",
    "synthetic_linear_per_class": "_synl",
    "synthetic_mlp_per_class": "_synm",
    # the shared-bank forms (tasks/shared_bank_task.py); method line only
    "monk_bank_r1_per_class": "_monkb1",
    "monk_bank_r2_per_class": "_monkb2",
    "monk_bank_r3_per_class": "_monkb3",
    "synthetic_linear_bank_per_class": "_synlb",
    "synthetic_mlp_bank_per_class": "_synmb",
}

# The later phases' drivers, as rediscover_cell.sh runs them.
DRIVERS = {
    "kernel": ["probe_kappa_query_dependence.sh",
               "probe_kappa_position_vs_content.sh",
               "analyze_c2_violations.sh",
               "analyze_anchor_identity.sh"],
    "budget": ["probe_component_budget.sh", "probe_contrast_budget.sh",
               "probe_offset_sweep.sh", "probe_pair_mode_control.sh"],
    "demo": ["probe_demo_causal.sh", "probe_demo_causal_vpatch.sh",
             "probe_label_repair.sh"],
    "substrate": ["probe_substrate_size_curve.sh"],
}
INPUT_MARKERS = ("tl_heads_", "check_model_accuracy", "s0_baseline")
FAILURE_PATTERNS = (
    r"Traceback \(most recent call last\)", r"^\S*Error: ", r"slurmstepd",
    r"CANCELLED", r"DUE TO TIME LIMIT", r"oom-kill", r"Out of memory",
    r"OutOfMemoryError", r"Killed", r"\[abort\]", r"REFUSING",
)


def task_tag(task):
    if task not in TASK_TAGS:
        raise SystemExit(f"task {task!r}: not one the rediscovery chain runs; "
                         f"known: {sorted(TASK_TAGS)}")
    return TASK_TAGS[task]


def calib_dir(mtag_base, task):
    """Per model, and per task below it: label surfaces are shared within a
    (model, task) and must not be across tasks."""
    base = Path("data/rediscovery") / mtag_base
    return base if task == DEFAULT_TASK else base / task


def cell_names(mtag_base, K, seed, nq=250, task=DEFAULT_TASK):
    """Every name rediscover_cell.sh computes, spelled ONCE more here.

    The test compares this against the script's own resolution
    (check_rediscovery_wiring.resolve), so the two cannot drift apart
    silently.
    """
    K, seed, nq = int(K), int(seed), int(nq)
    mtag = f"{mtag_base}{task_tag(task)}s{seed}"
    cell = f"{mtag}_K{K}_n{nq}"
    cd = calib_dir(mtag_base, task)
    n = {
        "MTAG": mtag, "CELL": cell, "TASK": task, "CALIB_DIR": str(cd),
        "UUID": str(cd / f"calibration_{task}_K{K}_seed{seed}_uuid.jsonl"),
        "RANDU": str(cd / f"calibration_{task}_K{K}_seed{seed}_random_uuid.jsonl"),
        "HJ": f"results/tl_heads_{mtag}_K{K}_n{nq}_top8.json",
        "RANKING": f"results/tl_heads_{mtag}_K{K}_n{nq}_top8.ranking.jsonl",
        "IDENT": f"results/identity_{mtag}_K{K}_n{nq}_top8",
        "MODEL_JSON": f"results/check_model_accuracy_K{K}_top8_{mtag}_n{nq}.json",
        "EP": f"results/error_prediction_{mtag}_K{K}_n{nq}_top8.json",
        "MS": f"results/magnitude_share_{mtag}_K{K}_n{nq}_top8.json",
        "S0": f"results/s0_baseline_{mtag}_K{K}_n{nq}.json",
        "B1": f"results/causal_ablation_B1_{mtag}_K{K}_n{nq}.json",
        "E1": f"results/tl_heads_{mtag}_K{K}_n{nq}_L11_L16_from_top64.json",
        "LEDGER": f"results/rediscovery_ledger_{cell}.jsonl",
    }
    n["SLICE_HEADS"] = {s: f"results/tl_heads_{mtag}_K{K}_n{nq}_top{s}.json"
                        for s in SLICES}
    n["SLICE_F1"] = {s: f"results/causal_ablation_F1_{mtag}_K{K}_n{nq}_top{s}.json"
                     for s in SLICES}
    return n


def core_artifacts(names, nq):
    """{phase: [(path, kind, rows)]} for the phases the cell script runs
    itself. `rows` is the row count a jsonl must reach."""
    a = {
        "discover": [(names["HJ"], "json", None),
                     (names["RANKING"], "jsonl", 1)]
                    + [(names["SLICE_HEADS"][s], "json", None) for s in SLICES],
        "forward": [(names["IDENT"] + "/per_prompt.jsonl", "jsonl", int(nq)),
                    (names["MODEL_JSON"], "json", None),
                    (names["EP"], "json", None),
                    (names["MS"], "json", None)],
        "causal": [(names["S0"], "json", None), (names["B1"], "json", None)]
                  + [(names["SLICE_F1"][s], "json", None) for s in SLICES]
                  + [(names["E1"], "json", None)],
    }
    return a


def driver_artifacts(names, K, nq, seed, model, se):
    """{phase: [(path, kind, None)]} for the env-driven drivers, resolved
    the way check_rediscovery_wiring does -- by running their assignments
    under the cell's environment -- minus the inputs they read."""
    from tools.check_rediscovery_wiring import resolve
    env = {"MODEL": model, "SE": se, "MTAG": names["MTAG"], "K": str(K),
           "NQ": str(nq), "SEED": str(seed), "CALIB_DIR": names["CALIB_DIR"],
           "TASK": names["TASK"]}
    out = {}
    for phase, drivers in DRIVERS.items():
        rows = []
        for d in drivers:
            p = Path("script") / d
            if not p.is_file():
                continue
            for path in sorted(resolve(p, env)):
                if any(m in path for m in INPUT_MARKERS):
                    continue
                if "__" in path or path.endswith("_.json"):
                    continue                    # set inside a loop; unseen
                kind = ("npz" if path.endswith(".npz")
                        else "jsonl" if path.endswith(".jsonl") else "json")
                rows.append((path, kind, None))
        out[phase] = rows
    return out


def artifact_state(path, kind, rows=None):
    """('ok' | 'missing' | 'invalid', detail). Existence is not completion."""
    p = Path(path)
    if not p.exists():
        return "missing", ""
    try:
        if kind == "json":
            json.loads(p.read_text(encoding="utf-8"))
            return "ok", ""
        if kind == "jsonl":
            n = 0
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        json.loads(line)
                        n += 1
            if rows is not None and n < int(rows):
                return "invalid", f"{n} rows, expected at least {rows}"
            if n == 0:
                return "invalid", "empty"
            return "ok", f"{n} rows"
        if kind == "npz":
            import numpy as np
            with np.load(p, allow_pickle=False) as z:
                keys = list(z.files)
            return ("ok", f"{len(keys)} arrays") if keys else ("invalid", "no arrays")
        if p.stat().st_size == 0:
            return "invalid", "empty file"
        return "ok", ""
    except Exception as e:                     # noqa: BLE001 -- any parse failure
        return "invalid", f"{type(e).__name__}: {str(e)[:80]}"


def move_aside(path):
    """Rename a broken artifact so the cell's guard rebuilds it; nothing is
    deleted."""
    p = Path(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = p.with_name(p.name + f".invalid-{stamp}")
    p.rename(dst)
    return dst


def read_ledger(path, last=8):
    p = Path(path)
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                rows.append({"raw": line[:120]})
    return rows[-last:]


def read_log(path):
    """The last phase banner, the last step line, and the first failure."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    last_phase = last_step = None
    failure = None
    done = False
    for i, l in enumerate(lines):
        m = re.match(r"^── (\w+):", l)
        if m:
            last_phase = m.group(1)
        if re.match(r"^\[(skip|build|output|done)\]|^\s*\[output\]|^\[ok\]", l):
            last_step = l.strip()
        if failure is None and any(re.search(p, l) for p in FAILURE_PATTERNS):
            failure = (i + 1, l.strip()[:160])
        if re.search(r"cell \S+ done", l):
            done = True
    return {"last_phase": last_phase, "last_step": last_step,
            "failure": failure, "done": done, "n_lines": len(lines)}


def plan(mtag_base, K, seed, nq, task, model, se):
    names = cell_names(mtag_base, K, seed, nq, task)
    arts = core_artifacts(names, nq)
    arts.update(driver_artifacts(names, K, nq, seed, model, se))
    states = {ph: [(p, k, r) + artifact_state(p, k, r) for p, k, r in arts.get(ph, [])]
              for ph in PHASES}
    first_incomplete = next((ph for ph in PHASES
                             if any(st != "ok" for _p, _k, _r, st, _d in states[ph])),
                            None)
    return names, states, first_incomplete


def resume_command(mtag_base, K, seed, task, first_incomplete):
    if first_incomplete is None:
        return None
    phases = " ".join(PHASES[PHASES.index(first_incomplete):])
    env = "" if task == DEFAULT_TASK else f"TASK={task} "
    return (f"{env}sbatch script/lsu1.sh bash script/rediscover_cell.sh "
            f"{mtag_base} {K} {seed} \"{phases}\"")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mtag-base", default="L31c36")
    ap.add_argument("--K", default="5")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--nq", default="250")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--log", default=None, help="a slurm .out of this cell")
    ap.add_argument("--fix", action="store_true",
                    help="move every INVALID artifact aside (never delete)")
    ap.add_argument("--check", default=None,
                    help="validate ONE artifact and exit 0/1 (the cell "
                         "script's guard); with --fix it is moved aside "
                         "when invalid")
    ap.add_argument("--rows", type=int, default=None,
                    help="with --check on a jsonl: rows it must reach")
    ap.add_argument("--task-tag", default=None,
                    help="print the tag folded into MTAG for this task")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    if args.task_tag is not None:
        print(task_tag(args.task_tag))
        return 0

    if args.check is not None:
        p = args.check
        kind = ("npz" if p.endswith(".npz") else "jsonl" if p.endswith(".jsonl")
                else "json" if p.endswith(".json") else "file")
        st, detail = artifact_state(p, kind, args.rows)
        if st == "invalid" and args.fix:
            dst = move_aside(p)
            print(f"[invalid] {p}: {detail} -> moved to {dst}")
            return 1
        print(f"[{st}] {p}" + (f": {detail}" if detail else ""))
        return 0 if st == "ok" else 1

    if args.task not in TASK_TAGS:
        task_tag(args.task)                     # raises with the known list
    import os
    from tools.model_tags import resolve, se_flags
    try:
        spec = resolve(args.mtag_base, model=os.environ.get("MODEL"),
                       method=os.environ.get("METHOD"))
    except (KeyError, ValueError) as e:
        raise SystemExit(f"[rediscover_status] {e}")
    model, se = spec["model"], se_flags(spec)
    names, states, first = plan(args.mtag_base, args.K, args.seed, args.nq,
                                args.task, model, se)

    print("=" * 78)
    print(f"REDISCOVERY STATUS -- cell {names['CELL']}  task {args.task}  "
          "[ZERO GPU]")
    print("=" * 78)
    for f_ in (names["UUID"], names["RANDU"]):
        st, d = artifact_state(f_, "jsonl", 1)
        print(f"  calibration  [{st}] {f_}" + (f"  {d}" if d else ""))

    fixed = []
    print(f"\n  {'phase':<10} {'artifact':<70} state")
    for ph in PHASES:
        rows = states[ph]
        if not rows:
            print(f"  {ph:<10} (no artifact names resolved; driver not found?)")
            continue
        for p, k, r, st, d in rows:
            mark = {"ok": "ok", "missing": "MISSING", "invalid": "INVALID"}[st]
            print(f"  {ph:<10} {p:<70} {mark}" + (f"  {d}" if d else ""))
            if st == "invalid" and args.fix:
                fixed.append((p, str(move_aside(p))))
    if fixed:
        print("\n  moved aside (the guard will rebuild them):")
        for p, dst in fixed:
            print(f"    {p} -> {dst}")

    ledger = read_ledger(names["LEDGER"])
    if ledger:
        print(f"\n  ledger {names['LEDGER']} (last {len(ledger)}):")
        for e in ledger:
            print("    " + json.dumps(e, ensure_ascii=False)[:150])
    else:
        print(f"\n  no ledger at {names['LEDGER']} (written by cells run after "
              "2026-09-09)")

    log = None
    if args.log:
        log = read_log(args.log)
        print(f"\n  log {args.log}: {log['n_lines']} lines")
        print(f"    last phase banner : {log['last_phase']}")
        print(f"    last step line    : {log['last_step']}")
        if log["failure"]:
            print(f"    first failure     : line {log['failure'][0]}: "
                  f"{log['failure'][1]}")
        print(f"    cell done banner  : {log['done']}")

    cmd = resume_command(args.mtag_base, args.K, args.seed, args.task, first)
    print()
    if first is None:
        print("  every phase is complete; nothing to resume.")
    else:
        print(f"  first incomplete phase: {first}")
        print("  resume with:")
        print(f"      {cmd}")
        print("  (phases before it are skipped by their guards; artifacts "
              "marked INVALID are")
        print("   rebuilt only after --fix moved them aside, or the guard "
              "will keep them)")

    if args.json_out:
        doc = {"cell": names["CELL"], "task": args.task,
               "first_incomplete": first, "resume": cmd,
               "phases": {ph: [{"path": p, "kind": k, "state": st, "detail": d}
                               for p, k, r, st, d in states[ph]]
                          for ph in PHASES},
               "fixed": fixed, "log": log, "ledger_tail": ledger}
        Path(args.json_out).write_text(json.dumps(doc, indent=2,
                                                  ensure_ascii=False),
                                       encoding="utf-8")
        print(f"\n  [output] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
