"""How much of the carriers' write survives to the readout? ZERO GPU.

Pairs `probe_carrier_direct_response`'s direct logit attribution with the same
arms' FINAL candidate logits from the RESULTS 51 npz, and regresses one on the
other through the origin:

    Delta_final = slope * Delta_direct

slope 1 means what the carriers wrote arrives intact; below 1 means the rest
of the network absorbs part of it; above 1 means it is amplified. This is the
project's own idiom -- section 31 measured 1.0006 written against 0.4216
surviving to the final residual stream, on another model and another design.

⚠⚠ CENTRING IS THE WHOLE QUESTION HERE, NOT A DETAIL (working rules 10b). A shift
applied equally to all 36 candidates cancels in the softmax and changes no
probability, so the RAW slope counts a component the readout cannot see. The
centred slope -- per query, the mean over candidates removed -- is the one
that describes what the intervention does to the answer. Both are reported,
labelled, and never averaged together.

⚠ THE RATIO CAN BE UNDEFINED, AND THAT IS A RESULT. If the carriers' direct
write barely moves between the two masks, the denominator is ~0 and the slope
is noise divided by noise. The two sides are always reported separately in
nats so the reader can see that before reading any ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_k10_increment import ARM_BASE, ARM_SEL  # noqa: E402


BUILD_DIRECT = """    sbatch script/lsu1.sh python tools/probe_carrier_direct_response.py \\
        --carrier-bundle results/carriers_full_validation.json \\
        --query-manifest results/prereg_method_A_query_manifest.json \\
        --label-space results/label_space_llama31.json \\
        --calibration-dir data/method_a/llama31 \\
        --out results/carrier_direct_L31.npz"""

BUILD_FINAL = """    sbatch script/lsu1.sh python tools/run_k10_increment.py \\
        --carrier-bundle results/carriers_full_validation.json \\
        --query-manifest results/prereg_method_A_query_manifest.json \\
        --label-space results/label_space_llama31.json \\
        --calibration-dir data/method_a/llama31 \\
        --out results/k10_increment_L31.npz"""


def load_pair(direct_npz, final_npz, arm_sel, arm_base):
    """((d_sel, d_base), (f_sel, f_base), qids, seeds, cand, faults)."""
    # NAME THE MISSING FILE AND WHAT MAKES IT (working rules 3.6). This reader is
    # the second half of a two-step: the GPU probe has to finish first, and a
    # bare FileNotFoundError does not say which of the two is missing or how
    # to get it.
    for path, what, how in ((direct_npz, "the carriers' direct write",
                             BUILD_DIRECT),
                            (final_npz, "the final candidate logits "
                                        "(RESULTS 51's run)", BUILD_FINAL)):
        if not Path(path).is_file():
            return (None,) * 5 + ([
                f"{path}: not a file. It holds {what}, and is written by\n"
                f"{how}"],)
    dz = np.load(direct_npz, allow_pickle=False)
    fz = np.load(final_npz, allow_pickle=False)
    dm = json.loads(str(dz["meta"]))
    fm = json.loads(str(fz["meta"]))
    bad = []
    seeds = [int(x) for x in (dm.get("seeds") or [])]
    if not seeds:
        bad.append(f"{direct_npz}: no seeds in its meta")
    fseeds = ([int(x) for x in fm["seeds"]] if fm.get("seeds") is not None
              else [int(k) for k in (fm.get("per_seed") or {})])
    if seeds and fseeds and seeds != fseeds:
        bad.append(f"seeds differ: {seeds} vs {fseeds}")
    if not np.array_equal(np.asarray(dz["query_ids"]).astype(str),
                          np.asarray(fz["query_ids"]).astype(str)):
        bad.append("the two files scored different queries; a per-cell "
                   "comparison over them would not be paired")
    d, f = {}, {}
    for tag, arm in (("sel", arm_sel), ("base", arm_base)):
        for s in seeds:
            dk, fk = f"direct_{arm}_seed{s}", f"{arm}_seed{s}"
            if dk not in dz:
                bad.append(f"{direct_npz}: no {dk!r}. Keys: {sorted(dz.files)}")
            if fk not in fz:
                bad.append(f"{final_npz}: no {fk!r}. Keys: {sorted(fz.files)}")
        if bad:
            continue
        d[tag] = np.stack([np.asarray(dz[f"direct_{arm}_seed{s}"],
                                      dtype=np.float64) for s in seeds])
        f[tag] = np.stack([np.asarray(fz[f"{arm}_seed{s}"],
                                      dtype=np.float64) for s in seeds])
    if bad:
        return None, None, None, None, None, bad
    return (d, f, np.asarray(dz["query_ids"]).astype(str), seeds,
            np.asarray(dz["candidate_classes"]), [])


def slope_through_origin(x, y):
    """(slope, R^2). Through the origin: at zero write there is zero change."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    sxx = float((x * x).sum())
    if sxx <= 0:
        return float("nan"), float("nan")
    b = float((x * y).sum() / sxx)
    ss_res = float(((y - b * x) ** 2).sum())
    ss_tot = float((y * y).sum())
    return b, (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def centre(a):
    """Per query, remove the mean over candidates.

    That component is invisible to the readout: adding the same number to
    every candidate leaves every probability unchanged.
    """
    a = np.asarray(a, dtype=np.float64)
    return a - a.mean(axis=-1, keepdims=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--direct", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--arm-sel", default=ARM_SEL)
    ap.add_argument("--arm-base", default=ARM_BASE)
    ap.add_argument("--min-rms", type=float, default=1e-6,
                    help="refuse the slope when the direct side's centred RMS "
                         "is below this: a ratio whose denominator is noise "
                         "is noise")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    d, f, qids, seeds, cand, bad = load_pair(args.direct, args.final,
                                             args.arm_sel, args.arm_base)
    if bad:
        print("REFUSING TO READ:")
        for b in bad:
            print(f"  - {b}")
        return 1

    dd = d["sel"] - d["base"]          # [S, Q, C] change in the DIRECT write
    df = f["sel"] - f["base"]          # [S, Q, C] change in the FINAL logits

    print("=" * 78)
    print("DOES THE CARRIERS' WRITE SURVIVE TO THE READOUT?  [ZERO GPU]")
    print("=" * 78)
    print(f"  {len(seeds)} seeds x {dd.shape[1]} queries x {dd.shape[2]} "
          "candidates")
    print(f"  arms: {args.arm_sel!r} minus {args.arm_base!r}")

    print("\n  how big each side is, in nats (RMS over cells)")
    print(f"    {'':<26}{'raw':>12}{'centred':>12}")
    for tag, a in (("Delta_direct", dd), ("Delta_final", df)):
        print(f"    {tag:<26}{float(np.sqrt((a ** 2).mean())):>12.6f}"
              f"{float(np.sqrt((centre(a) ** 2).mean())):>12.6f}")

    rms_c = float(np.sqrt((centre(dd) ** 2).mean()))
    if rms_c < args.min_rms:
        print(f"\n  ⚠ REFUSING the slope: the direct side's centred RMS is "
              f"{rms_c:.3e}, below {args.min_rms:.0e}.")
        print("    The carriers' direct write barely differs between the two "
              "masks, so a ratio")
        print("    against it divides noise by noise. The two sides above are "
              "the result.")
        return 0

    print("\n  Delta_final = slope * Delta_direct, through the origin")
    print(f"    {'':<26}{'slope':>10}{'R^2':>10}   per seed")
    out = {"arms": [args.arm_sel, args.arm_base], "seeds": seeds}
    for tag, x, y in (("raw", dd, df), ("CENTRED", centre(dd), centre(df))):
        b, r2 = slope_through_origin(x, y)
        per = [slope_through_origin(x[i], y[i])[0] for i in range(len(seeds))]
        print(f"    {tag:<26}{b:>10.4f}{r2:>10.4f}   "
              + "  ".join(f"{v:.4f}" for v in per))
        out[f"slope_{tag.lower()}"] = {"slope": b, "r2": r2, "per_seed": per}

    b_c = out["slope_centred"]["slope"]
    print(f"\n  ⟹ of what the carriers write into the readout direction, "
          f"{b_c:.1%} arrives.")
    if b_c < 1.0:
        print(f"    The rest of the network absorbs {1 - b_c:.1%} of it.")
    else:
        print(f"    The rest of the network amplifies it by {b_c - 1:.1%}.")
    print("\n  " + "-" * 74)
    print("  ⚠⚠ THE CENTRED ROW IS THE ONE THAT DESCRIBES THE ANSWER "
          "(working rules 10b).")
    print("     A shift applied equally to all candidates cancels in the "
          "softmax, so the raw")
    print("     slope counts a component no probability can see. The two are "
          "never averaged.")
    print("  ⚠ This is a DIRECT-path accounting, not a causal decomposition: "
          "the carriers")
    print("    also change what every later layer reads, and that route is "
          "inside Delta_final")
    print("    rather than separated from it.")

    if args.json_out:
        out["rms"] = {"direct_raw": float(np.sqrt((dd ** 2).mean())),
                      "direct_centred": rms_c,
                      "final_raw": float(np.sqrt((df ** 2).mean())),
                      "final_centred": float(np.sqrt((centre(df) ** 2).mean()))}
        Path(args.json_out).write_text(json.dumps(out, indent=2),
                                       encoding="utf-8")
        print(f"\n  [output] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
