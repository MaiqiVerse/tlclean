"""Is this calibration file the one it is supposed to be? ZERO GPU.

WHY A SEPARATE TOOL FOR THREE LINES. Driver scripts skip a step when its
output file EXISTS, and a run interrupted partway leaves a file that exists.
The K=10 render in `script/measure_K10_headroom.sh` aborted at the
--reuse-labels guard on 2026-09-08; had it aborted a few lines later there
would now be a truncated jsonl that the loop skips and every downstream step
reads as the artifact. "Exists" is not a check -- what the header SAYS is.

Exit 0 when the header matches, non-zero with the mismatch named, so a shell
can use it directly:

    if [ -f "$f" ] && python tools/check_calibration_header.py \\
           --path "$f" --K 10 --seed 42; then ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def header_faults(path, *, K=None, seed=None, task=None, model=None,
                  n_classes=None):
    p = Path(path)
    if not p.is_file():
        return [f"{p}: not a file"]
    try:
        with p.open(encoding="utf-8") as f:
            line = f.readline()
        if not line.strip():
            return [f"{p}: empty first line, so there is no header. A file "
                    "that exists is not a file that finished"]
        head = json.loads(line)
    except Exception as e:                                  # noqa: BLE001
        return [f"{p}: header is not readable json ({type(e).__name__}: {e}). "
                "A truncated write looks exactly like this"]
    if not isinstance(head, dict) or not head.get("header"):
        return [f"{p}: first line is not a header record; keys "
                f"{sorted(head) if isinstance(head, dict) else type(head).__name__}"]
    bad = []
    for name, want in (("K", K), ("seed", seed), ("task", task),
                       ("model", model), ("n_classes", n_classes)):
        if want is None:
            continue
        got = head.get(name)
        try:
            same = int(got) == int(want)
        except (TypeError, ValueError):
            same = got == want
        if not same:
            bad.append(f"{p}: header says {name}={got!r}, wanted {want!r}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--path", required=True)
    ap.add_argument("--K", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--task")
    ap.add_argument("--model")
    ap.add_argument("--n-classes", type=int)
    ap.add_argument("--quiet", action="store_true",
                    help="say nothing on success; a driver loop calling this "
                         "per file does not need a line per file")
    args = ap.parse_args(argv)
    bad = header_faults(args.path, K=args.K, seed=args.seed, task=args.task,
                        model=args.model, n_classes=args.n_classes)
    if bad:
        for b in bad:
            print(b, file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"[ok] {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
