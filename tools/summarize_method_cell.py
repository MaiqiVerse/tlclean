"""The tables of one method cell, from its readouts. ZERO GPU.

    python tools/summarize_method_cell.py --cell results/method/L31c36 \\
        --levels "0:5 2:5 5:10" --split validation --json-out results/method/L31c36/summary_validation.json

Reads what script/method_cell.sh left in the cell:
  <prefix>k0_receiver_K<f>_<split>_readout.json               (tools/analyze_k0_receiver)
  <prefix>k<f>_increment_into_K<b>_<split>_readout.json       (same)
  <prefix>direct_write_accuracy_K<b>base_<split>.json         (tools/analyze_direct_write_accuracy)
  <prefix>carrier_absorption_K<b>base_<split>.json            (tools/analyze_carrier_absorption)
  natural_K<b>_vs_K<f>.json                                   (tools/compare_natural_K; validation only)
with <prefix> = "UNSAFE/UNSAFE_" on the test split. Keys are the producers'
(their names are asserted against the producers' sources in
tools/test_summarize_method_cell.py). Three tables:

  1. the levels side by side: total, recovered, fraction, decisions
  2. accuracy at alpha = 1 (the upstream protocol) for every arm present
  3. the carriers' direct write beside the model's own decisions

A missing level is a row of dashes, never an error: the cell may have skipped
it (window) or not reached it yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.method_status import parse_levels  # noqa: E402

READOUT_KEYS = ("by_arm", "accuracy_paired_vs_k0_offset", "total_nll_benefit",
                "recovered_nll_benefit", "recovered_fraction", "per_seed",
                "n_queries_per_seed", "seeds")
PAIR_KEYS = ("net", "broke", "fixed", "mcnemar_exact_p", "attainable_floor_p")
DIRECT_KEYS = ("rows", "model", "paired", "chance", "n")
ABSORB_KEYS = ("slope_raw", "slope_centred")


def arm_names(kb, kf):
    """The four arms' names (tools/run_k10_increment.arm_names)."""
    try:
        from tools.run_k10_increment import arm_names as _an
        n = _an(kb, kf)
        return n["mono"], n["all"], n["sel"], n["base"]
    except Exception:                        # noqa: BLE001 -- fall back to the spelling
        return (f"full K{kf} monolithic", f"all-head cached K{kf}",
                f"selective TL K{kf} memory", f"K{kb}-offset natural")


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def need(d, keys, what):
    lack = [k for k in keys if k not in d]
    if lack:
        raise SystemExit(f"{what}: lacks {lack}; keys present {sorted(d)}")


def level_files(cell, kb, kf, split, variant=""):
    """`variant` is the driver's artifact prefix of a carrier-count cut
    ("top30_": the first 30 heads of the bundle's ranking, TSLA's head count;
    method_cell.sh TOPN); "" is the registered top-8 arm. The natural readouts
    are shared by every variant."""
    cell = Path(cell)
    if split == "validation":
        pre, sub = variant, cell
    else:
        pre, sub = "UNSAFE_" + variant, cell / "UNSAFE"
    if kb == 0:
        readout = sub / f"{pre}k0_receiver_K{kf}_{split}_readout.json"
        direct = absorb = None
    else:
        readout = sub / f"{pre}k{kf}_increment_into_K{kb}_{split}_readout.json"
        direct = sub / f"{pre}direct_write_accuracy_K{kb}base_{split}.json"
        absorb = sub / f"{pre}carrier_absorption_K{kb}base_{split}.json"
    natural = cell / f"natural_K{kb}_vs_K{kf}.json" if kb > 0 else None
    return readout, direct, absorb, natural


def summarise_level(cell, kb, kf, split, variant=""):
    readout_p, direct_p, absorb_p, natural_p = level_files(cell, kb, kf, split, variant)
    out = {"K_base": kb, "K_full": kf, "split": split, "variant": variant,
           "files": {"readout": str(readout_p), "direct": str(direct_p) if direct_p else None,
                     "absorption": str(absorb_p) if absorb_p else None,
                     "natural": str(natural_p) if natural_p else None},
           "present": False}
    r = load_json(readout_p)
    if r is None:
        return out
    need(r, READOUT_KEYS, str(readout_p))
    mono, allh, sel, base = arm_names(kb, kf)
    by = r["by_arm"]
    pair = r["accuracy_paired_vs_k0_offset"]
    n_per_seed = int(r["n_queries_per_seed"])
    n = n_per_seed * len(r["seeds"])
    out.update({
        "present": True, "n": n, "arms": {"mono": mono, "all": allh, "sel": sel, "base": base},
        "total": r["total_nll_benefit"]["value"], "total_ci95": r["total_nll_benefit"]["ci95"],
        "recovered": r["recovered_nll_benefit"]["value"],
        "recovered_ci95": r["recovered_nll_benefit"]["ci95"],
        "fraction": r["recovered_fraction"], "per_seed": r["per_seed"],
        "accuracy": {a: {"acc": by[a]["accuracy"], "n_correct": int(round(by[a]["accuracy"] * n)),
                         "nll": by[a]["nll"]} for a in by},
        "decisions": {a: {k: pair[a][k] for k in PAIR_KEYS} for a in pair},
    })
    # alpha = 1 rows: every TSLA / FV / TV arm ending in " a=1"
    out["alpha1_arms"] = sorted(a for a in by if a.endswith(" a=1"))
    # the arms chosen ON VALIDATION by the selection tools (TV's layer, ICV's
    # lambda), when the cell has made the choice; on validation they are in
    # sample, on test they are the arms that ran
    chosen = {}
    for name, fname, key, fmt in (
            ("tv", f"tv_layer_K{kf}_into_K{kb}.json", "selected_layer",
             lambda v: f"TV-K{kf} L={int(v)} a=1"),
            ("icv", f"icv_lambda_K{kf}_into_K{kb}.json", "selected_lambda",
             lambda v: f"ICV-K{kf} a={float(v):g}")):
        sel = load_json(Path(cell) / fname)
        if sel is not None and key in sel:
            arm = fmt(sel[key])
            chosen[name] = {"arm": arm, "present": arm in by, "file": fname}
    out["chosen_arms"] = chosen
    nat = load_json(natural_p) if natural_p else None
    if nat is not None and "total" in nat:
        out["natural_total"] = nat["total"]
        out["natural_ci"] = nat.get("ci")
    d = load_json(direct_p) if direct_p else None
    if d is not None:
        need(d, DIRECT_KEYS, str(direct_p))
        out["direct"] = {
            "rows": {lab: {conv: {k: d["rows"][lab][conv][k] for k in ("accuracy", "n_correct", "n")}
                           for conv in ("uncentred", "centred") if conv in d["rows"][lab]}
                     for lab in d["rows"]},
            "model": {lab: {k: d["model"][lab][k] for k in ("accuracy", "n_correct", "n")}
                      for lab in d["model"]},
            "paired_reference": d["paired"]["reference"],
            "paired": {lab: {conv: {k: d["paired"]["rows"][lab][conv][k]
                                    for k in ("net", "b", "c", "mcnemar_exact_p", "attainable_floor_p")}
                             for conv in d["paired"]["rows"][lab]}
                       for lab in d["paired"]["rows"]},
            "chance": d["chance"], "n": d["n"]}
    ab = load_json(absorb_p) if absorb_p else None
    if ab is not None:
        need(ab, ABSORB_KEYS, str(absorb_p))
        out["absorption"] = {k: {"slope": ab[k]["slope"], "r2": ab[k]["r2"],
                                 "per_seed": ab[k]["per_seed"]} for k in ABSORB_KEYS}
    return out


MODEL_NAMES = {"meta-llama/Llama-3.1-8B": "Llama-3.1-8B",
               "meta-llama/Llama-2-7b-hf": "Llama-2-7B",
               "Qwen/Qwen2.5-7B": "Qwen2.5-7B",
               "Qwen/Qwen3-8B-Base": "Qwen3-8B",
               "Qwen/Qwen3-4B-Base": "Qwen3-4B",
               "Qwen/Qwen3-14B-Base": "Qwen3-14B"}
TASK_NAMES = {"trec_fine_per_class": "TREC-fine", "banking77_per_class": "banking77",
              "clinc150_per_class": "clinc150", "dbpedia14_per_class": "dbpedia14",
              "yahoo_answers_per_class": "yahoo", "yelp_full_per_class": "yelp",
              "monk_bank_r1_per_class": "Monk-1", "monk_bank_r2_per_class": "Monk-2",
              "monk_bank_r3_per_class": "Monk-3",
              "synthetic_linear_bank_per_class": "synthetic linear",
              "synthetic_mlp_bank_per_class": "synthetic MLP"}


def display_model(model, method=None):
    name = MODEL_NAMES.get(model or "", model or "?")
    return name + (" + SelfExtend" if method == "selfextend" else "")


def ceiling_file(cell, kf, split):
    """The native K_full ceiling the driver writes for this split."""
    cell = Path(cell)
    if split == "validation":
        return cell / f"direct_write_ceiling_K{kf}.json"
    return cell / "UNSAFE" / f"UNSAFE_direct_write_ceiling_K{kf}_test_seed.json"


def paper_rows(cell, levels_spec, split, model=None, method=None, task=None, topn="30"):
    """One LaTeX row per increment level (K_base > 0) in the paper table's
    column order -- Natural | All heads | upper bound 8 / N heads | TSLA TL /
    TR | Ours 8 / N heads -- from the registered readout (model argmax),
    the split's ceiling json (uncentred direct write on the native K_full
    prompt) and the two direct-write jsons (registered and top<N>: the
    selective arm's uncentred direct write). A piece that is not there
    prints as '?'; the larger of the two Ours values is bold. A comment line
    follows with the four baselines' main-config accuracies when the cell
    ran them."""
    def pct(x):
        return "?" if x is None else f"{100 * float(x):.2f}"

    def acc_of(L, arm):
        cell_ = L["accuracy"].get(arm) if L.get("present") else None
        return None if cell_ is None else cell_["acc"]

    def direct_of(L, arm):
        d = L.get("direct") if L.get("present") else None
        if not d:
            return None
        return d["rows"].get(arm, {}).get("uncentred", {}).get("accuracy")

    out, ext = [], []
    name, tname = display_model(model, method), TASK_NAMES.get(task or "", task or "?")
    for kb, kf in parse_levels(levels_spec):
        if kb == 0:
            continue
        L = summarise_level(cell, kb, kf, split, "")
        LN = summarise_level(cell, kb, kf, split, f"top{topn}_")
        lvl = f"${kb}{{\\to}}{kf}$"
        if not L["present"]:
            out.append(f"% {name} & {tname} & {lvl} & (no {split} readout for this level yet)")
            ext.append(f"% {name} & {tname} & {lvl} & (no {split} readout for this level yet)")
            continue
        base, mono, sel = L["arms"]["base"], L["arms"]["mono"], L["arms"]["sel"]
        ceil = load_json(ceiling_file(cell, kf, split)) or {}
        crow = ceil.get("rows", {})
        ub8 = crow.get(f"carriers, natural K={kf}", {}).get("uncentred", {}).get("accuracy")
        ubn = crow.get(f"top-{topn} by ranking, natural K={kf}", {}).get("uncentred", {}).get("accuracy")
        o8, on = direct_of(L, sel), direct_of(LN, sel)
        ours = [pct(o8), pct(on)]
        if o8 is not None and on is not None:
            i = 1 if float(on) >= float(o8) else 0
            ours[i] = f"\\textbf{{{ours[i]}}}"
        tl, tr = acc_of(L, f"TSLA-K{kf} a=1"), acc_of(L, f"TSLA-K{kf}tr a=1")
        out.append(f"{name} & {tname} & {lvl} & {pct(acc_of(L, base))} & {pct(acc_of(L, mono))} & "
                   f"{pct(ub8)} & {pct(ubn)} & {pct(tl)} & {pct(tr)} & {ours[0]} & {ours[1]} \\\\")
        # the four baselines at their main configs (model argmax on the same
        # queries): FV and I2CL at alpha = 1, TV at the layer and ICV at the
        # lambda chosen on validation (test reads carry those arms alone)
        base_acc = {"FV": acc_of(L, f"FV-K{kf} a=1"), "I2CL": acc_of(L, f"I2CL-K{kf} a=1")}
        cfg = {}
        for key, label in (("tv", "TV"), ("icv", "ICV")):
            c = (L.get("chosen_arms") or {}).get(key)
            base_acc[label] = acc_of(L, c["arm"]) if c and c.get("present") else None
            if c and c.get("present"):
                cfg[label] = c["arm"]
        ours_m = [pct(acc_of(L, sel)), pct(acc_of(LN, sel))]
        ext.append(f"{name} & {tname} & {lvl} & {pct(acc_of(L, base))} & {pct(base_acc['FV'])} & "
                   f"{pct(base_acc['TV'])} & {pct(base_acc['ICV'])} & {pct(base_acc['I2CL'])} & "
                   f"{pct(tl)} & {pct(tr)} & {pct(acc_of(L, mono))} & {ours_m[0]} & {ours_m[1]} & "
                   f"{ours[0]} & {ours[1]} \\\\")
        if cfg:
            ext.append("%   chosen on validation: " + "; ".join(f"{k} = {v}" for k, v in cfg.items()))
    return out, ext


def fmt_dec(dc):
    if dc is None:
        return "—"
    return (f"{dc['net']:+d} ({dc['broke']}/{dc['fixed']}) p={dc['mcnemar_exact_p']:.3g}"
            + ("" if dc["attainable_floor_p"] < 0.05 else " (floor)"))


def render(levels, split, variant=""):
    print("=" * 100)
    print(f"METHOD CELL SUMMARY -- {split}"
          + (f" -- variant {variant!r}: the selective arm reads the first "
             f"{variant.strip('top_') or '?'} heads of the bundle's ranking (exploratory cut); "
             "the TSLA and natural arms are the same as in the registered tables"
             if variant else ""))
    print("=" * 100)
    print("\n1. the levels side by side (nats; fraction = recovered / total, two means)")
    print(f"{'level':>10} | {'n':>5} | {'total (native)':>22} | {'recovered [CI]':>28} | "
          f"{'fraction':>9} | {'selective net (b/f) p':>26} | {'all-head net':>14}")
    for L in levels:
        tag = f"K{L['K_base']}->{L['K_full']}"
        if not L["present"]:
            print(f"{tag:>10} | {'—':>5} | {'— (not run / skipped)':>22} |")
            continue
        nat = f" ({L['natural_total']:.3f})" if "natural_total" in L else ""
        tot = f"{L['total']:.3f}{nat}"
        rec = f"{L['recovered']:.3f} [{L['recovered_ci95'][0]:.3f}, {L['recovered_ci95'][1]:.3f}]"
        sel = L["decisions"].get(L["arms"]["sel"])
        allh = L["decisions"].get(L["arms"]["all"])
        print(f"{tag:>10} | {L['n']:>5} | {tot:>22} | {rec:>28} | {L['fraction'] * 100:>8.1f}% | "
              f"{fmt_dec(sel):>26} | {(str(allh['net']) if allh else '—'):>14}")
        ps = L["per_seed"]
        print(f"{'':>10}   per seed: " + "  ".join(
            f"{s}: {ps[s]['fraction'] * 100:.1f}%" for s in sorted(ps)))
    print("\n2. accuracy, alpha = 1 (argmax over the label tokens; n correct / n)")
    for L in levels:
        if not L["present"]:
            continue
        tag = f"K{L['K_base']}->{L['K_full']}"
        base = L["arms"]["base"]
        base_acc = L["accuracy"][base]["acc"]
        chosen = {c["arm"] for c in L.get("chosen_arms", {}).values() if c.get("present")}
        rows = ([L["arms"]["mono"], L["arms"]["sel"], base] + L["alpha1_arms"]
                + sorted(a for a in chosen if a not in L["alpha1_arms"]))
        print(f"  {tag}  (baseline {base!r} = {base_acc:.4f})")
        for a in rows:
            if a not in L["accuracy"]:
                continue
            acc = L["accuracy"][a]
            dc = L["decisions"].get(a)
            print(f"    {a:<30} {acc['acc']:.4f} ({acc['n_correct']}/{L['n']})  "
                  f"{(acc['acc'] - base_acc) * 100:+.1f} pt   {fmt_dec(dc) if a != base else ''}"
                  + ("   <- chosen on validation" if a in chosen else ""))
    print("\n3. the carriers' direct write beside the model (uncentred | centred | model; n correct)")
    for L in levels:
        if not L["present"] or "direct" not in L:
            continue
        tag = f"K{L['K_base']}->{L['K_full']}"
        d = L["direct"]
        print(f"  {tag}  chance {d['chance']:.4f}, N {d['n']}")
        for lab in (L["arms"]["base"], L["arms"]["sel"], L["arms"]["all"]):
            if lab not in d["rows"]:
                continue
            u = d["rows"][lab].get("uncentred", {})
            c = d["rows"][lab].get("centred", {})
            m = d["model"].get(lab, {})
            pu = d["paired"].get(lab, {}).get("uncentred")
            net = f"  direct net {pu['net']:+d} ({pu['b']}/{pu['c']}) p={pu['mcnemar_exact_p']:.3g}" if pu else ""
            print(f"    {lab:<30} {u.get('accuracy', float('nan')):.4f} ({u.get('n_correct', '—')}) | "
                  f"{c.get('accuracy', float('nan')):.4f} ({c.get('n_correct', '—')}) | "
                  f"{m.get('accuracy', float('nan')):.4f} ({m.get('n_correct', '—')}){net}")
        if "absorption" in L:
            ab = L["absorption"]
            print(f"    absorption slope: raw {ab['slope_raw']['slope']:.4f} (R2 {ab['slope_raw']['r2']:.3f}), "
                  f"centred {ab['slope_centred']['slope']:.4f} (R2 {ab['slope_centred']['r2']:.3f})")
    print("\n  ⚠ centred and uncentred are two readouts (working rules 10b); the model's column "
          "is read against the uncentred one only. Test-split numbers are optional/descriptive "
          "(prereg 14.0b-23).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--levels", default="0:5 2:5 5:10")
    ap.add_argument("--split", choices=("validation", "test_seed"), default="validation")
    ap.add_argument("--json-out")
    ap.add_argument("--variant", default="",
                    help="artifact prefix of a carrier-count cut the driver ran "
                         "beside the registered arm (method_cell.sh TOPN), e.g. "
                         "'top30_'; default '' = the registered top-8 arm")
    ap.add_argument("--model", default=None, help="HF id, for the paper rows' model column")
    ap.add_argument("--method", default=None, help="vanilla / selfextend, for the paper rows")
    ap.add_argument("--task", default=None, help="task name, for the paper rows' task column")
    ap.add_argument("--topn", default="30",
                    help="the carrier-count cut whose artifacts fill the 'N heads' columns of "
                         "the paper rows (default 30 = the driver's TOPN)")
    args = ap.parse_args(argv)
    levels = [summarise_level(args.cell, kb, kf, args.split, args.variant)
              for kb, kf in parse_levels(args.levels)]
    render(levels, args.split, args.variant)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"cell": args.cell, "split": args.split, "variant": args.variant,
             "levels": levels}, indent=2),
            encoding="utf-8")
        print(f"\n  [output] {args.json_out}")
    if not args.variant:
        # the registered run prints the paper rows once, reading the top-N
        # artifacts itself, so a test job's log ends with the row(s) to paste
        rows_a, rows_b = paper_rows(args.cell, args.levels, args.split, args.model, args.method,
                                    args.task, args.topn)
        print(f"\n4. the paper table's rows ({args.split}; accuracy %)")
        if args.split == "validation":
            print("   ⚠ validation rows are in sample; the paper's rows come from test_seed")
        print(f"   A (figs/tables/accuracy_comparison_multi.tex): Natural | All heads | upper bound 8 / "
              f"{args.topn} heads (the native K_full ceiling's uncentred direct write) | TSLA TL / TR "
              f"| Ours 8 / {args.topn} heads (the selective arm's uncentred direct write)")
        for row in rows_a:
            print(row)
        print(f"   B (with the four baselines; every column but the last two is the model's argmax on "
              f"the same queries): Natural | FV | TV | ICV | I2CL | TSLA TL | TSLA TR | All heads | "
              f"Ours 8 / {args.topn} heads, model argmax | Ours 8 / {args.topn} heads, direct write")
        for row in rows_b:
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
