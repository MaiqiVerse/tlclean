"""Where a method cell got to. ZERO GPU.

script/method_cell.sh runs the section-13.5 constructions on one (model, task)
cell as a sequence of steps, each leaving artifacts under
results/method/<cell>/. This tool answers three questions the shell cannot
answer well on its own:

  --check PATH [--rows N] [--fix]   is this artifact COMPLETE (parses; a jsonl
                                    has >= N rows; an npz has arrays; a
                                    receiver npz was not a --limit run unless
                                    --limit was asked for)? --fix moves a
                                    broken one aside so the cell rebuilds it.
  --levels "0:5 2:5 5:10"           parse a LEVELS spec; print one level per
                                    line as "base full", refusing a pair that
                                    is not base < full.
  --ks "0:5 2:5 5:10" [--kdisc 5]   the distinct K the calibration must build.
  --task-check TASK                 refuse a task the method line cannot host
                                    (generated tasks carry their demonstrations
                                    inside each query; multi-field tasks are
                                    fine).
  --fits LENGTH_JSON...             exit 0 iff every window record says fits.

    python tools/method_status.py --check results/method/L31c36/k0_receiver_K5_validation.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def artifact_state(path, rows=None, limit=None, needs_arms=()):
    """('ok' | 'missing' | 'invalid', detail). Existence is not completion.

    `needs_arms`: arm names a receiver npz must carry (meta.arms). A cell
    whose sidecars now provide a baseline arm the npz predates is not done
    at that level; the driver moves the npz aside and reruns the receiver."""
    p = Path(path)
    if not p.exists():
        return "missing", ""
    try:
        if p.suffix == ".json":
            json.loads(p.read_text(encoding="utf-8"))
            return "ok", ""
        if p.suffix == ".jsonl":
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
        if p.suffix == ".npz":
            import numpy as np
            with np.load(p, allow_pickle=False) as z:
                keys = list(z.files)
                meta = None
                if "meta" in keys:
                    # Two producer conventions: the receivers store
                    # `meta=json.dumps(...)` (a 0-d array), the probes
                    # `meta=np.array([json.dumps(...)])` (1-d). Indexing [0]
                    # on the 0-d form raised, meta came back None, and the
                    # --limit check silently never ran on a receiver npz
                    # (found 2026-09-13 by the --needs-arm smoke).
                    try:
                        arr = z["meta"]
                        raw = arr.item() if arr.ndim == 0 else arr.reshape(-1)[0]
                        meta = json.loads(str(raw))
                    except Exception:            # noqa: BLE001
                        meta = None
            if not keys:
                return "invalid", "no arrays"
            if meta is not None and limit is not None:
                got = int(meta.get("limit", 0) or 0)
                if got != int(limit):
                    return "invalid", (f"meta.limit={got}, this run asks for "
                                       f"limit={limit} (a smoke artifact is "
                                       "not a full one and vice versa)")
            if needs_arms:
                have = list((meta or {}).get("arms") or [])
                if meta is None:
                    return "invalid", (f"no readable meta, so the arms "
                                       f"{list(needs_arms)} cannot be confirmed")
                lack = [a for a in needs_arms if a not in have]
                if lack:
                    return "invalid", (f"lacks arm(s) {lack}: the cell's sidecars "
                                       "now provide them, so this level is rerun "
                                       f"(present: {len(have)} arms)")
            return "ok", f"{len(keys)} arrays"
        if p.stat().st_size == 0:
            return "invalid", "empty file"
        return "ok", ""
    except Exception as e:                     # noqa: BLE001
        return "invalid", f"{type(e).__name__}: {str(e)[:80]}"


def move_aside(path):
    p = Path(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = p.with_name(p.name + f".invalid-{stamp}")
    p.rename(dst)
    return dst


def parse_levels(spec):
    """'0:5 2:5 5:10' -> [(0, 5), (2, 5), (5, 10)], base < full."""
    out = []
    for tok in str(spec).split():
        if ":" not in tok:
            raise ValueError(f"level {tok!r}: expected base:full")
        b, f = tok.split(":", 1)
        b, f = int(b), int(f)
        if not 0 <= b < f:
            raise ValueError(f"level {tok!r}: need 0 <= base < full")
        if (b, f) in out:
            raise ValueError(f"level {tok!r} repeated")
        out.append((b, f))
    if not out:
        raise ValueError("no levels")
    return out


def calibration_ks(levels, kdisc):
    ks = {int(kdisc)}
    for b, f in levels:
        ks.add(f)
        if b > 0:
            ks.add(b)
    return sorted(ks)


def task_faults(task):
    from tools.prereg_task import ALL_ELIGIBLE, DOC_FIELDS
    if task not in DOC_FIELDS:
        return [f"unknown task {task!r}; known: {sorted(DOC_FIELDS)}"]
    if task in ALL_ELIGIBLE:
        from tools.prereg_task import BANK_FORM
        return [f"{task}: a generated task carries its demonstrations inside "
                "each query (a per-prompt hidden function), so there is no "
                "shared demonstration bank for the constructions to read; the "
                "rediscovery chain covers it, the method line does not. Its "
                f"shared-bank form is {BANK_FORM[task]}: one fixed concept, a "
                "train bank and a test pool (tasks/shared_bank_task.py)"]
    if not task.endswith("_per_class"):
        return [f"{task}: the constructions read K demonstrations PER CLASS; "
                f"use {task}_per_class"]
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="with --check on a receiver npz: the meta.limit this "
                         "run expects (0 for a full run)")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--needs-arm", action="append", default=[],
                    help="with --check on a receiver npz: an arm meta.arms must "
                         "list (repeatable); e.g. 'FV-K10 a=1' once the cell's "
                         "FV sidecar exists")
    ap.add_argument("--levels")
    ap.add_argument("--ks")
    ap.add_argument("--kdisc", type=int, default=5)
    ap.add_argument("--task-check")
    ap.add_argument("--fits", nargs="*")
    args = ap.parse_args(argv)
    if args.check:
        st, detail = artifact_state(args.check, rows=args.rows, limit=args.limit,
                                    needs_arms=tuple(args.needs_arm))
        if st == "invalid" and args.fix:
            dst = move_aside(args.check)
            print(f"[invalid] {args.check}: {detail}; moved to {dst.name}")
            return 1
        print(f"[{st}] {args.check}" + (f": {detail}" if detail else ""))
        return 0 if st == "ok" else 1
    if args.levels is not None and args.ks is None:
        try:
            for b, f in parse_levels(args.levels):
                print(f"{b} {f}")
        except ValueError as e:
            print(f"[method_status] LEVELS: {e}", file=sys.stderr)
            return 1
        return 0
    if args.ks is not None:
        try:
            print(" ".join(str(k) for k in calibration_ks(parse_levels(args.ks), args.kdisc)))
        except ValueError as e:
            print(f"[method_status] LEVELS: {e}", file=sys.stderr)
            return 1
        return 0
    if args.task_check:
        bad = task_faults(args.task_check)
        for b in bad:
            print(f"[method_status] {b}", file=sys.stderr)
        return 1 if bad else 0
    if args.fits is not None:
        ok = True
        for p in args.fits:
            try:
                d = json.loads(Path(p).read_text(encoding="utf-8"))
            except Exception as e:               # noqa: BLE001
                print(f"[fits] {p}: unreadable ({type(e).__name__})")
                ok = False
                continue
            if not d.get("fits"):
                print(f"[fits] {p}: max {d.get('max_tokens')} tokens > window "
                      f"{d.get('window')}")
                ok = False
        return 0 if ok else 1
    ap.error("nothing to do")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
