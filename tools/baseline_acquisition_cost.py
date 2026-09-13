"""What each method pays to OBTAIN its object on validation (the acquisition
side of the cost, beside bench/bench_cost.py's inference side): the model
forwards it takes, counted from the runners' recipes, and the seconds the
runners printed in a cell's log. ZERO GPU.

    python tools/baseline_acquisition_cost.py --classes 5 --levels "0:5 2:5" --vpc 4 \\
        --log logs/L31_yelp_method_cell_baselines_pass1.out [--ledger results/method/L31c36_yelp/ledger.jsonl]

COUNTS (per seed and per level, K_base -> K_full; C eligible classes,
N_x = (K_full - K_base) * C extra demonstrations, n_v = VPC * C validation
queries, `long` = a prompt of about the K_full demonstrations, `short` = a
demonstration or a query alone, `cached` = the K_base prefix cached plus
the query):

  ours (selective TL memory, section 13.5)
      discover   n_v long forwards with per-head capture at K_disc, ONCE PER
                 CELL (the folds reuse the per-prompt scores; discover_carriers)
      build      1 long forward per (seed, K_full): the memory
      select     nothing (top-8 by rank; top-30 is the exploratory cut)
  TSLA (run_tsla_tl_on_icl)
      discover   n_v long forwards at K_full plus the per-head projections
                 (float64 on the CPU; the slow part on a shared host)
      select     nothing registered (alpha = 1; the alpha grid is descriptive)
  FV (run_fv_increment: mean activations, AtP screen, exact CIE)
      extract    N_x long forwards (leave-one-out extraction prompts)
      screen     25 corrupted prompts, forward + backward each
      CIE        25 x |candidates| long forwards (|candidates| = max(30, 1.5 x heads))
      select     nothing (alpha = 1; top-|heads| by exact CIE)
  TV (run_tv_increment)
      extract    5 long forwards (1 main theta + 4 for the descriptive m5 mean)
      select     5 candidate layers x n_v cached forwards on validation
                 (+ 5 m5 arms, descriptive), tools/select_tv_layer
  ICV (run_icv_increment)
      extract    2 N_x short forwards (x and xy per demonstration), then a PCA
      select     5 lambdas x (1 K_base prefix re-encode under the hooks +
                 n_v cached forwards), tools/select_icv_lambda
  I2CL (run_i2cl_increment)
      extract    N_x short forwards (the context vectors)
      calibrate  epochs x N_x cached forward + backward passes (the
                 pseudo-queries after the K_base prefix), the prefix cache
                 refreshed every --refresh-steps steps
      select     nothing (the strengths are the calibrated coefficients)
  every arm      its validation read: n_v cached forwards (the receivers)

MEASURED: the runners print their own seconds ("(741s)", "in 77s",
"5 forwards (45s)", "calibrated 128 scalars in 250s"); --log collects them per
level and seed. --ledger turns the driver's ledger.jsonl (one `start`
event per step) into step durations. Wall-clock depends on the card,
the dtype and the prompt length; the counts do not.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

N_CORRUPTED = 25          # run_fv_increment: corrupted prompts for AtP and the exact CIE
TV_LAYERS = 5             # vector_arms.tv_candidate_layers: at most five
ICV_LAMBDAS = 5           # select_icv_lambda: {0.05, 0.1, 0.2, 0.4, 0.8}
I2CL_EPOCHS = 100         # the upstream's 100


def fv_candidates(fv_heads):
    """The AtP screen's candidate count for FV's head budget (20 on the ~8B
    tier, 50 on 13B; run_fv_increment.head_count_for): max(30, 1.5 x heads)."""
    from tools.baselines.run_fv_increment import atp_screen_for
    return atp_screen_for(fv_heads)


def counts(classes, k_base, k_full, vpc, fv_heads, epochs=I2CL_EPOCHS):
    nx = (k_full - k_base) * classes
    nv = vpc * classes
    cand = fv_candidates(fv_heads)
    rows = [
        ("ours", f"{nv} long (once per cell) + 1 long (memory)", "0", f"{nv} cached / arm"),
        ("TSLA", f"{nv} long + per-head projections (CPU)", "0", f"{nv} cached / arm"),
        ("FV", f"{nx} long + {N_CORRUPTED} fwd+bwd + {N_CORRUPTED * cand} long (CIE, {cand} cand.)", "0",
         f"{nv} cached / arm"),
        ("TV", "5 long", f"{TV_LAYERS} layers x {nv} cached = {TV_LAYERS * nv}", f"{nv} cached / arm"),
        ("ICV", f"{2 * nx} short", f"{ICV_LAMBDAS} x ({nv} cached + 1 prefix) = {ICV_LAMBDAS * (nv + 1)}",
         f"{nv} cached / arm"),
        ("I2CL", f"{nx} short + {epochs} x {nx} = {epochs * nx} cached fwd+bwd", "0", f"{nv} cached / arm"),
    ]
    return nx, nv, rows


# ------------------------------------------------------------------ the log
LEVEL_RE = re.compile(r"^(FV|TASK VECTORS|IN-CONTEXT VECTORS|I2CL), INCREMENT SETTING: K=(\d+) extra demonstrations into a K=(\d+) receiver")
SEED_RE = re.compile(r"^\s+seed (\d+): (.*)$")
SECS = [
    ("FV", "mean", re.compile(r"mean activations over (\d+)/(\d+) correctly answered prompts \((\d+)s\)")),
    ("FV", "AtP", re.compile(r"AtP over (\d+) prompts in (\d+)s")),
    ("FV", "CIE", re.compile(r"exact CIE on (\d+) heads in (\d+)s")),
    ("TV", "theta", re.compile(r"(\d+) forwards \((\d+)s\)")),
    ("ICV", "pairs", re.compile(r"\((\d+)s\); \|\|mean Delta\|\|")),
    ("I2CL", "cv", re.compile(r"context vectors from (\d+) extra demonstrations \((\d+)s\)")),
    ("I2CL", "calib", re.compile(r"calibrated (\d+) scalars in (\d+)s")),
]
# the epoch lines ("epoch 10/100: ... (28s)") carry the CUMULATIVE seconds and are
# printed every ten epochs; the runner's own total is the "calibrated ... in Ns" line
NAMES = {"FV": "FV", "TASK VECTORS": "TV", "IN-CONTEXT VECTORS": "ICV", "I2CL": "I2CL"}
REQUIRED = {"FV": ("mean", "AtP", "CIE"), "TV": ("theta",), "ICV": ("pairs",), "I2CL": ("cv", "calib")}


def parse_log(path):
    """{(method, k_base, k_full): {seed: {part: seconds}}} from a cell's .out."""
    out = {}
    cur = None
    seed = None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = LEVEL_RE.match(raw)
        if m:
            cur = (NAMES[m.group(1)], int(m.group(3)), int(m.group(2)))
            out.setdefault(cur, {})
            seed = None
            continue
        if cur is None:
            continue
        m = SEED_RE.match(raw)
        if m:
            seed = int(m.group(1))
            body = m.group(2)
            for meth, part, rx in SECS:
                if meth != cur[0]:
                    continue
                mm = rx.search(body)
                if mm:
                    out[cur].setdefault(seed, {})[part] = int(mm.groups()[-1])
            continue
    return out


def print_log(parsed):
    print("\nmeasured seconds per seed (the runners' own timings)")
    print(f"  {'level':<12} {'method':<6} {'seed':>4}  parts -> total")
    totals = {}
    for (meth, kb, kf), seeds in sorted(parsed.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        for s, parts in sorted(seeds.items()):
            shown = dict(parts)
            tot = sum(shown.values())
            done = all(k in parts for k in REQUIRED[meth])
            if done:
                totals.setdefault((meth, kb, kf), []).append(tot)
            print(f"  K={kb}->{kf:<7} {meth:<6} {s:>4}  {shown} -> {tot} s"
                  + ("" if done else "  (INCOMPLETE: the run stopped here; not in the mean)"))
    print("\n  per level, mean over seeds:")
    for (meth, kb, kf), ts in sorted(totals.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        print(f"    K={kb}->{kf:<6} {meth:<6} {sum(ts) / len(ts):8.0f} s / seed  ({len(ts)} seeds)")


def print_ledger(path):
    """Step durations per job: a step lasts from its `start` to the next
    step's `start` in the SAME job (or to the job's `cell done`); a job's
    last step without a `done` is open (the job aborted or was cut). A step
    of a few seconds was skipped (its artifacts existed)."""
    evs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_job = {}
    for e in evs:
        by_job.setdefault(str(e.get("job", "")), []).append(e)
    print(f"\nledger {path}: step durations per job (seconds; a few seconds = skipped)")
    for job, es in by_job.items():
        cell = es[0].get("cell", "")
        marks = [(e["step"], e["event"], datetime.strptime(e["utc"], "%Y-%m-%dT%H:%M:%SZ"))
                 for e in es if e.get("event") in ("start", "done") and e["step"] != "cell"]
        done = [datetime.strptime(e["utc"], "%Y-%m-%dT%H:%M:%SZ") for e in es
                if e.get("event") == "done" and e["step"] == "cell"]
        starts = [(s, t) for s, ev, t in marks if ev == "start"]
        if not starts:
            continue
        print(f"  job {job or '(none)'}  {cell}  {es[0]['utc']}")
        for (s, t0), (_s1, t1) in zip(starts, starts[1:]):
            print(f"    {s:<10} {(t1 - t0).total_seconds():8.0f}")
        s, t0 = starts[-1]
        if done:
            print(f"    {s:<10} {(done[-1] - t0).total_seconds():8.0f}   (to the job's done)")
        else:
            print(f"    {s:<10}     open   (no done event: aborted, cut, or still running)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--classes", type=int, required=True, help="eligible classes")
    ap.add_argument("--levels", default="0:5 2:5", help="K_base:K_full pairs")
    ap.add_argument("--vpc", type=int, default=4, help="validation queries per class per seed")
    ap.add_argument("--fv-heads", type=int, default=20,
                    help="FV's head budget (20 on the ~8B tier, 50 on 13B); the AtP screen keeps max(30, 1.5 x it)")
    ap.add_argument("--epochs", type=int, default=I2CL_EPOCHS, help="I2CL calibration epochs")
    ap.add_argument("--log", help="a cell's .out log: the runners' printed seconds")
    ap.add_argument("--ledger", help="the cell's ledger.jsonl: step durations")
    args = ap.parse_args(argv)

    print("=" * 100)
    print(f"acquisition cost per seed: {args.classes} classes, validation {args.vpc}/class "
          f"(n_v = {args.vpc * args.classes}), FV {args.fv_heads} heads, I2CL {args.epochs} epochs")
    print("=" * 100)
    for lv in args.levels.split():
        kb, kf = (int(x) for x in lv.split(":"))
        nx, nv, rows = counts(args.classes, kb, kf, args.vpc, args.fv_heads, args.epochs)
        print(f"\nlevel K={kb}->{kf}: N_x = {nx} extra demonstrations, n_v = {nv} validation queries")
        print(f"  {'method':<6} {'obtain (forwards)':<62} {'choose on validation':<44} {'read / arm'}")
        for name, obt, sel, read in rows:
            print(f"  {name:<6} {obt:<62} {sel:<44} {read}")
    if args.log:
        print_log(parse_log(args.log))
    if args.ledger:
        print_ledger(args.ledger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
