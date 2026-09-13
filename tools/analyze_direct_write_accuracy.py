"""What does the carriers' DIRECT write alone decide? ZERO GPU.

Takes the carrier heads' per-candidate direct logit attribution as if it were
the readout -- argmax over the 36 candidates, compared with the gold class.
This is section 27's TL-alone accuracy, computed on whichever conditions the
input files hold, in ONE table so the conditions can be read against each
other:

    K5-offset natural          the K=5 receiver, no memory        (--direct)
    selective TL K10 memory    the carriers read a K=10 memory    (--direct)
    all-head cached K10        every head reads it                (--direct)
    carriers, natural K=10     the SAME heads on a native K=10    (--dla)
                               prompt: the CEILING of what the
                               increment's direct path could reach

The last row is the one the others are measured against. The increment
construction can at best make the carriers do what they do natively at K=10;
what it actually delivers on the direct path is the distance from its row to
that one, and the paired block at the end prints it as decisions.

⚠⚠ CENTRING IS A READOUT CONVENTION AND BOTH ARE REPORTED (working rules 10b).
Subtracting each class's mean over queries removes the query-INDEPENDENT class
bias, and on this project's own measurement that is not a detail: section 27
went from 0.2480 to 0.4120 on all 1024 heads by doing it. The two numbers
answer different questions --

    uncentred   what the direct write decides as it stands
    centred     what it decides once a query-independent preference for some
                classes is taken out

-- and section 10b exists because a centred 0.4120 was once put beside the
model's uncentred 0.4280, manufacturing a comparison that was not there. So
this prints both, labels both, pairs rows only within a convention, and
refuses to subtract one from the other.

⚠ THE MODEL'S OWN ACCURACY IS NOT A CENTRED QUANTITY. It is printed for scale
in its own block, per arm, with the convention stated, and may be read against
the UNCENTRED column only.

Two input layouts, because both files exist, and both may be given at once:

    --direct   probe_carrier_direct_response: already summed over carriers,
               one array per arm, under the K=10 increment masks
    --dla      probe_tl_class_attribution: dla[q, l, h, c] on a natural
               K prompt, summed here over a carrier set given by
               --carrier-bundle (the frozen set; --top-n cuts and --all-heads
               are extra rows from the same dump)

When both are given, the two files are checked to have scored the same
queries in the same order with the same candidate set -- a paired count over
two files that do not agree on that is not paired.

    --centre-from FILE [FILE ...]
               the centring mean estimated on ANOTHER split: the same row
               (arm, or carrier cut of a per-head dump) in a file of that
               split, per seed the class mean over its queries, subtracted
               from the analysed rows. A third column, `centred_ref`, beside
               the transductive `centred` one (whose mean is taken over the
               analysed queries themselves). The reference must be disjoint
               in query ids from the analysed file -- a mean over the same
               queries is the transductive column again -- and must agree
               on seeds and candidates. A row without a counterpart in any
               reference shows n/a in that column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import carrier_bundle as CB  # noqa: E402
from tools.analyze_unsafe_test import mcnemar_exact_p  # noqa: E402
from tools.method_a_stats import mcnemar_floor_p  # noqa: E402

# "top-16 by ranking, natural K=10" is 31 characters and a cell with n/N =
# 155/432 is 26; a label that fills its column exactly glues onto the cell.
LABEL_W, CELL_W = 34, 28


def centre_over_queries(a):
    """Per class, remove its mean over queries.

    The cross-query mean of a class is what that class gets regardless of what
    was asked, so removing it is what "query-conditional" means for this
    readout. Section 27's convention, applied to the carriers' sum rather than
    per head; the choice is stated wherever the number is.
    """
    a = np.asarray(a, dtype=np.float64)
    return a - a.mean(axis=0, keepdims=True)


CONVENTIONS = (("uncentred", lambda x: x), ("centred", centre_over_queries))


def hits(scores, gold_col):
    """[Q] bool: argmax over candidates equals the gold column."""
    s = np.asarray(scores, dtype=np.float64)
    g = np.asarray(gold_col, dtype=np.int64)
    return np.argmax(s, axis=1) == g


def accuracy(scores, gold_col):
    """(accuracy, n_correct, n): argmax over candidates against the gold."""
    h = hits(scores, gold_col)
    return float(h.mean()), int(h.sum()), int(h.size)


def se(p, n):
    return float(np.sqrt(max(p * (1 - p), 0.0) / n)) if n else float("nan")


def paired(h_row, h_ref):
    """(b, c): b = ref right & row wrong, c = row right & ref wrong."""
    a = np.asarray(h_row, dtype=bool)
    r = np.asarray(h_ref, dtype=bool)
    return int(np.sum(r & ~a)), int(np.sum(a & ~r))


def load_centre_refs(paths, common, bundle_path, top_ns, all_heads):
    """{row label: {seed: class mean [C]}} from files of another split.

    Each file is read by its own kind (a probe_carrier_direct_response npz
    carries `direct_<arm>_seed<s>` arrays; anything else is a per-head dump
    cut with the same bundle / --top-n / --all-heads as the analysed rows),
    rows are matched to the analysed ones by label. Refused when the seeds
    or the candidate classes differ, or when a query id of the reference is
    also in the analysed file: the point of the flag is a mean that never
    saw the queries the readout is scored on.
    """
    mu, used, bad = {}, [], []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        if any(k.startswith("direct_") for k in z.files):
            rr, cc, b = load_direct(p)
        elif not bundle_path:
            rr, cc, b = None, None, [f"{p}: a per-head dump needs --carrier-bundle "
                                      "to be cut like the analysed rows"]
        else:
            rr, cc, _K, _m0, b = load_dla(p, bundle_path, top_ns, all_heads)
        if b:
            bad += b
            continue
        if cc["seeds"] != common["seeds"]:
            bad.append(f"{p}: seeds {cc['seeds']} vs the analysed {common['seeds']}")
            continue
        if not np.array_equal(cc["candidate_classes"], common["candidate_classes"]):
            bad.append(f"{p}: candidate classes differ from the analysed file's")
            continue
        overlap = [(s, len(set(map(str, cc["query_ids"][si])) & set(map(str, common["query_ids"][si]))))
                   for si, s in enumerate(cc["seeds"])]
        if any(n for _s, n in overlap):
            bad.append(f"{p}: query ids shared with the analysed file "
                       f"{[(s, n) for s, n in overlap if n]}; the centring mean must come "
                       "from queries the readout is not scored on (that is the "
                       "transductive `centred` column, printed anyway)")
            continue
        for r in rr:
            mu[r["label"]] = {s: np.asarray(r["per_seed"][s], dtype=np.float64).mean(axis=0)
                              for s in cc["seeds"]}
            used.append({"row": r["label"], "file": str(p),
                         "n_queries_per_seed": {str(s): int(r["per_seed"][s].shape[0]) for s in cc["seeds"]}})
    return mu, used, bad


ALIGN_KEYS = ("query_ids", "class_idx", "candidate_classes")


def _common(z, *, strict=True):
    """The arrays two files must agree on.

    The two probes write all of them. run_k10_increment writes `query_ids`
    and `class_idx` but keeps `seeds` in its meta only and has no
    `candidate_classes` array (its np.savez_compressed call in main; the
    test parses it rather than citing a line number that moves), so
    for --final `strict=False` takes what is there and `alignment_faults`
    compares the keys both sides hold -- and says which. A gate that quietly
    compared nothing on the real file would be the working rules 3.4 gate no
    input can turn red.
    """
    # Seeds: the array, else the meta's `seeds`, else the keys of the meta's
    # `per_seed`. The last is where results/k10_increment_L31.npz (job 839609)
    # keeps them -- it predates run_k10_increment writing `seeds` into its
    # meta, and analyze_k0_receiver recovers them the same way. The first
    # version of this stopped at the meta's `seeds` and refused that file
    # with "seeds differ: [42, 43, 44] vs []" -- the artifact on disk was made
    # by an older producer than the one whose savez was read (3.2b again).
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    if "seeds" in z:
        seeds, src = [int(x) for x in z["seeds"]], "the seeds array"
    elif meta.get("seeds"):
        seeds, src = [int(x) for x in meta["seeds"]], "meta['seeds']"
    elif isinstance(meta.get("per_seed"), dict) and meta["per_seed"]:
        seeds, src = [int(k) for k in meta["per_seed"]], "meta['per_seed'] keys"
    else:
        seeds, src = [], "nowhere"
    out = {"seeds": seeds, "seeds_source": src}
    for k in ALIGN_KEYS:
        if k in z:
            out[k] = (np.asarray(z[k]).astype(str) if k == "query_ids"
                      else np.asarray(z[k], dtype=np.int64))
        elif strict:
            raise KeyError(f"no {k!r}; keys are {sorted(z.files)}")
    return out


def load_direct(path):
    """Rows {label, source, per_seed{seed: [Q, C]}}, plus the common arrays."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    seeds = [int(x) for x in (meta.get("seeds") or [])]
    arms = [str(a) for a in (meta.get("arms") or [])]
    if not seeds or not arms:
        return None, None, [
            f"{path}: meta records seeds={seeds!r} arms={arms!r}; both are "
            f"needed. Keys: {sorted(z.files)}"]
    rows = []
    for a in arms:
        per = {}
        for s in seeds:
            k = f"direct_{a}_seed{s}"
            if k not in z:
                return None, None, [f"{path}: no {k!r}. Keys: {sorted(z.files)}"]
            per[s] = np.asarray(z[k], dtype=np.float64)
        rows.append({"label": a, "source": str(path), "per_seed": per,
                     "kind": "direct"})
    common = _common(z)
    if common["seeds"] != seeds:
        return None, None, [f"{path}: meta seeds {seeds} but the seeds array "
                            f"says {common['seeds']}"]
    return rows, common, []


def load_dla(path, bundle_path, top_ns=(), all_heads=False):
    """The same row shape from the per-head dump, summed over head sets.

    The frozen carrier set is always the first row and the one every other
    row is paired against. `--top-n` cuts slice the bundle's own ranking
    (exploratory, same score, different cut); `--all-heads` sums every head,
    which is section 27's whole-model TL-alone readout at this K.
    """
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    K = meta.get("K")
    if K is None:
        return None, None, None, None, [
            f"{path}: meta has no 'K'; the row cannot be named. Meta keys: "
            f"{sorted(meta)}"]
    common = _common(z)
    seeds = common["seeds"]
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    sets = [("carriers", None)] + [(f"top-{n} by ranking", int(n))
                                   for n in (top_ns or ())]
    rows = {lab: {"label": f"{lab}, natural K={K}", "source": str(path),
                  "per_seed": {}, "kind": "dla", "top_n": n}
            for lab, n in sets}
    if all_heads:
        rows["all"] = {"label": None, "source": str(path), "per_seed": {},
                       "kind": "dla", "top_n": "all"}
    m0 = {}
    for s in seeds:
        k = f"dla_seed{s}"
        if k not in z:
            return None, None, None, None, [
                f"{path}: no {k!r}. Keys: {sorted(z.files)}"]
        d = np.asarray(z[k], dtype=np.float64)          # [Q, L, H, C]
        if d.ndim != 4:
            return None, None, None, None, [
                f"{path}: {k} is {d.shape}, expected [Q, L, H, C]"]
        for lab, n in sets:
            try:
                hh = sorted({(int(l), int(h))
                             for l, h in CB.heads_for(bundle, s, n)})
            except (KeyError, ValueError) as e:
                return None, None, None, None, [
                    f"{bundle_path}: seed {s}, {lab}: {e}"]
            rows[lab]["per_seed"][s] = d[:, [l for l, _ in hh],
                                         [h for _, h in hh], :].sum(axis=1)
            rows[lab]["heads"] = len(hh)
        if all_heads:
            rows["all"]["per_seed"][s] = d.sum(axis=(1, 2))
            rows["all"]["label"] = (f"all {d.shape[1] * d.shape[2]} heads, "
                                    f"natural K={K}")
        mk = f"m0_seed{s}"
        if mk not in z:
            return None, None, None, None, [
                f"{path}: no {mk!r}. Keys: {sorted(z.files)}"]
        m0[s] = np.asarray(z[mk], dtype=np.float64)
    return list(rows.values()), common, int(K), m0, []


def alignment_faults(a, b, what_a, what_b):
    """(faults, compared): the same queries, in order, on one candidate set,
    or nothing below is paired. `compared` names the keys both sides held."""
    bad, compared = [], [f"seeds (from {b.get('seeds_source', '?')})"]
    for side, what in ((a, what_a), (b, what_b)):
        if not side["seeds"]:
            bad.append(f"{what}: seeds found nowhere (looked at the array, "
                       "meta['seeds'] and meta['per_seed'])")
    if not bad and a["seeds"] != b["seeds"]:
        bad.append(f"seeds differ: {what_a} {a['seeds']} vs {what_b} "
                   f"{b['seeds']}")
    for k in ALIGN_KEYS:
        if k not in a or k not in b:
            continue
        compared.append(k)
        x, y = a[k], b[k]
        if x.shape != y.shape:
            bad.append(f"{k}: {what_a} is {x.shape}, {what_b} is {y.shape}")
        elif not np.array_equal(x, y):
            n = int(np.sum(x != y))
            bad.append(f"{k}: {n} of {x.size} entries differ between {what_a} "
                       f"and {what_b}; the rows would not be paired")
    if len(compared) < 2:
        bad.append(f"{what_a} and {what_b} share none of {ALIGN_KEYS}; "
                   "nothing says they scored the same queries")
    return bad, compared


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--direct", help="probe_carrier_direct_response npz")
    ap.add_argument("--dla", help="probe_tl_class_attribution npz")
    ap.add_argument("--carrier-bundle", help="required with --dla")
    ap.add_argument("--top-n", type=int, nargs="*", default=[],
                    help="extra --dla rows cut from the bundle's ranking")
    ap.add_argument("--all-heads", action="store_true",
                    help="an extra --dla row summing every head")
    ap.add_argument("--final", help="a run's candidate logits: the model's "
                                    "own accuracy per same-named arm, and the "
                                    "natural forward the --dla m0 is checked "
                                    "against")
    ap.add_argument("--final-arm", default=None,
                    help="which --final arm is the natural forward at the "
                         "--dla file's K (default: 'full K<K> monolithic')")
    ap.add_argument("--pair-against", default=None,
                    help="the row every other row is paired with (default: "
                         "the first --dla row, the ceiling; without one, "
                         "the first row)")
    ap.add_argument("--json-out")
    ap.add_argument("--centre-from", nargs="*", default=[],
                    help="file(s) of ANOTHER split holding the same rows; per row and "
                         "seed their class mean over queries is the centring vector of "
                         "a third column, `centred_ref` (see the module docstring)")
    args = ap.parse_args(argv)

    if not (args.direct or args.dla):
        print("give --direct and/or --dla", file=sys.stderr)
        return 2
    if args.dla and not args.carrier_bundle:
        print("--dla needs --carrier-bundle: the per-head dump has no "
              "carrier set of its own", file=sys.stderr)
        return 2
    for p in (args.direct, args.dla, args.carrier_bundle, args.final):
        if p and not Path(p).is_file():
            print(f"REFUSING: {p}: not a file", file=sys.stderr)
            return 1

    rows, common, bad = [], None, []
    K = m0 = None
    if args.direct:
        r, common, bad = load_direct(args.direct)
        rows += r or []
    if args.dla and not bad:
        r, c2, K, m0, bad = load_dla(args.dla, args.carrier_bundle,
                                     args.top_n, args.all_heads)
        if not bad:
            if common is not None:
                fb, _ = alignment_faults(common, c2, args.direct, args.dla)
                bad += fb
            else:
                common = c2
            rows += r or []
    if bad:
        print("REFUSING TO READ:")
        for b in bad:
            print(f"  - {b}")
        return 1

    seeds = common["seeds"]
    cand = common["candidate_classes"]
    col = {int(c): i for i, c in enumerate(cand)}
    gold = np.array([[col[int(c)] for c in common["class_idx"][si]]
                     for si in range(len(seeds))])
    chance = 1.0 / len(cand)
    n_all = int(gold.size)

    print("=" * 78)
    print("WHAT THE CARRIERS' DIRECT WRITE DECIDES ON ITS OWN  [ZERO GPU]")
    print("=" * 78)
    for p in (args.direct, args.dla):
        if p:
            print(f"  {p}")
    print(f"  {len(seeds)} seeds x {gold.shape[1]} queries, "
          f"{len(cand)} candidates; chance {chance:.4f} "
          f"= {chance * n_all:.1f}/{n_all}")
    if args.dla:
        print(f"  --dla rows: the natural K={K} prompt; `carriers` is the "
              "frozen set every registered arm uses")

    mu_ref, ref_used, ref_bad = {}, [], []
    if args.centre_from:
        for p in args.centre_from:
            if not Path(p).is_file():
                ref_bad.append(f"{p}: not a file")
        if not ref_bad:
            mu_ref, ref_used, ref_bad = load_centre_refs(
                args.centre_from, common, args.carrier_bundle, args.top_n, args.all_heads)
        if ref_bad:
            print("REFUSING --centre-from:")
            for b in ref_bad:
                print(f"  - {b}")
            return 1
        print("  --centre-from: the `centred_ref` column subtracts, per row and seed, "
              "the class mean of the same row over ANOTHER split's queries")
        for u in ref_used:
            print(f"    {u['row']:<{LABEL_W}} <- {u['file']}  "
                  f"(queries per seed {u['n_queries_per_seed']})")
        missing = [r["label"] for r in rows if r["label"] not in mu_ref]
        if missing:
            print(f"    no reference row for {missing}: n/a in that column")

    out = {"direct": args.direct, "dla": args.dla, "seeds": seeds,
           "chance": chance, "n": n_all, "rows": {},
           "centre_from": ref_used}
    ceiling = next((r for r in rows if r["kind"] == "dla"), None)
    if args.pair_against:
        ref = next((r for r in rows if r["label"] == args.pair_against), None)
        if ref is None:
            print(f"REFUSING: no row named {args.pair_against!r}; the rows "
                  f"are {[r['label'] for r in rows]}")
            return 1
    else:
        ref = ceiling if ceiling is not None else (
            rows[0] if len(rows) > 1 else None)

    third = f"{'CENTRED (ref mean)':>{CELL_W}}" if mu_ref else ""
    print(f"\n  {'row':<{LABEL_W}}{'UNCENTRED':>{CELL_W}}"
          f"{'CENTRED':>{CELL_W}}{third}")
    print(f"  {'':<{LABEL_W}}{'acc +/- SE      n/N':>{CELL_W}}"
          f"{'acc +/- SE      n/N':>{CELL_W}}"
          + (f"{'acc +/- SE      n/N':>{CELL_W}}" if mu_ref else ""))
    for r in rows:
        rec, cells = {"source": r["source"], "kind": r["kind"]}, []
        if r.get("heads") is not None:
            rec["heads"] = r["heads"]
        r["hits"] = {}
        # the conventions: the two of CONVENTIONS on the row's own queries,
        # plus the reference-mean one when --centre-from holds this row
        convs = [(tag, (lambda x, s, f=fn: f(x))) for tag, fn in CONVENTIONS]
        if r["label"] in mu_ref:
            convs.append(("centred_ref",
                          lambda x, s, m=mu_ref[r["label"]]: np.asarray(x, dtype=np.float64) - m[s][None, :]))
        for tag, fn in convs:
            h = np.concatenate([hits(fn(r["per_seed"][s], s), gold[si])
                                for si, s in enumerate(seeds)])
            r["hits"][tag] = h
            p, k, n = float(h.mean()), int(h.sum()), int(h.size)
            rec[tag] = {"accuracy": p, "n_correct": k, "n": n, "se": se(p, n),
                        "per_seed": [float(hits(fn(r["per_seed"][s], s),
                                                gold[si]).mean())
                                     for si, s in enumerate(seeds)]}
            cells.append(f"{p:.4f} +/- {se(p, n):.4f}  {k}/{n}")
        if mu_ref and len(cells) == 2:
            cells.append("n/a")
        mark = "   <- ceiling" if r is ceiling and len(rows) > 1 else ""
        print(f"  {r['label']:<{LABEL_W}}" + "".join(f"{c:>{CELL_W}}" for c in cells) + mark)
        out["rows"][r["label"]] = rec

    # The model's own decisions, UNCENTRED, per arm where the same arm exists
    # on both sides: the direct write under one mask belongs beside what the
    # model decided under THAT mask. The --dla rows come with the natural
    # logits of the forward the attribution was read from, so their row needs
    # no second file.
    model = {}
    if args.final:
        fz = np.load(args.final, allow_pickle=False)
        fb, compared = alignment_faults(common, _common(fz, strict=False),
                                        "the inputs", args.final)
        if fb:
            print("\n  REFUSING the model rows from --final:")
            for b in fb:
                print(f"    - {b}")
            return 1
        print(f"\n  [PASS] {args.final} scored the same queries as the "
              f"inputs (compared: {', '.join(compared)})")
        for r in rows:
            if r["kind"] != "direct":
                continue
            a = r["label"]
            if all(f"{a}_seed{s}" in fz for s in seeds):
                h = np.concatenate([hits(np.asarray(fz[f"{a}_seed{s}"],
                                                    dtype=np.float64),
                                         gold[si])
                                    for si, s in enumerate(seeds)])
                model[a] = {"accuracy": float(h.mean()), "n_correct":
                            int(h.sum()), "n": int(h.size), "centred": False,
                            "source": args.final}
    if m0 is not None:
        h = np.concatenate([hits(m0[s], gold[si])
                            for si, s in enumerate(seeds)])
        model[f"natural K={K}"] = {
            "accuracy": float(h.mean()), "n_correct": int(h.sum()),
            "n": int(h.size), "centred": False,
            "source": f"m0 in {args.dla} (the same forward the DLA is from)"}
    if model:
        print("\n  the MODEL's own accuracy (UNCENTRED; read against the "
              "uncentred column only)")
        for a, m in model.items():
            print(f"    {a:<{LABEL_W - 2}}{m['accuracy']:.4f} +/- "
                  f"{se(m['accuracy'], m['n']):.4f}  "
                  f"{m['n_correct']}/{m['n']}   [{m['source']}]")
    elif args.final:
        print(f"\n  ⚠ {args.final}: none of "
              f"{[r['label'] for r in rows if r['kind'] == 'direct']} is in "
              "it; the model's own accuracy is not shown for the direct rows")
    out["model"] = model

    # CROSS-FILE GATE, the idiom of analyze_gain_feasibility: the DLA probe's
    # own natural logits against the natural arm of a run made by other code
    # on the same prompt. Same weights, same eager attention, same uncached
    # forward, so the two are expected to be BITWISE equal (RESULTS 50 found
    # 0.000e+00 at K=5); anything else is a different forward, not rounding.
    gates = {}
    if m0 is not None and args.final:
        arm = args.final_arm or f"full K{K} monolithic"
        fz = np.load(args.final, allow_pickle=False)
        if all(f"{arm}_seed{s}" in fz for s in seeds):
            print(f"\n  the --dla file's natural logits vs {arm!r} in "
                  f"{args.final}")
            for si, s in enumerate(seeds):
                nat = np.asarray(fz[f"{arm}_seed{s}"], dtype=np.float64)
                if nat.shape != m0[s].shape:
                    print(f"    seed {s}: {nat.shape} vs {m0[s].shape}; "
                          "different query sets, not compared")
                    gates[str(s)] = {"shape_mismatch": True}
                    continue
                d = float(np.abs(nat - m0[s]).max())
                gates[str(s)] = {"max_abs": d, "bitwise": d == 0.0}
                print(f"    seed {s}: max |diff| {d:.3e}"
                      + ("   [PASS] bitwise" if d == 0.0 else
                         "   ⚠ two code paths, one prompt: any difference is "
                         "a different forward, not rounding"))
        else:
            print(f"\n  ⚠ {args.final}: no {arm!r} arm, so the --dla file's "
                  "natural logits were not cross-checked. Keys: "
                  f"{sorted(fz.files)[:8]}...")
            gates["missing_arm"] = arm
    out["gates"] = gates

    # PAIRED, against the ceiling row, within a convention only. b and c are
    # decisions, the floor is the smallest p ANY split with that net could
    # give (b = 0), so a p of 1.0000 can be told apart from a test that could
    # not have fired.
    if ref is not None and len(rows) > 1:
        print(f"\n  paired against `{ref['label']}`"
              + (" (the ceiling)" if ref is ceiling else "")
              + ", pooled over seeds")
        print("  (b = reference right & row wrong, c = row right & reference "
              "wrong; net = c - b)")
        print(f"    {'row':<{LABEL_W}}{'conv':>10}{'acc':>8}{'delta':>8}"
              f"{'b':>5}{'c':>5}{'net':>6}{'exact p':>9}{'floor':>8}")
        print("    " + "-" * (LABEL_W + 59))
        out["paired"] = {"reference": ref["label"],
                         "reference_is_ceiling": ref is ceiling, "rows": {}}
        for r in rows:
            if r is ref:
                continue
            cell = {}
            for tag in [t for t, _fn in CONVENTIONS] + (
                    ["centred_ref"] if "centred_ref" in r["hits"] and "centred_ref" in ref["hits"] else []):
                hr, hc = r["hits"][tag], ref["hits"][tag]
                b, c = paired(hr, hc)
                a, racc = float(hr.mean()), float(hc.mean())
                p = mcnemar_exact_p(b, c)
                fl = mcnemar_floor_p(c - b)
                print(f"    {r['label']:<{LABEL_W}}{tag:>10}{a:>8.4f}"
                      f"{a - racc:>+8.4f}{b:>5}{c:>5}{c - b:>+6}{p:>9.4f}"
                      f"{fl:>8.4f}")
                cell[tag] = {"accuracy": a, "delta": a - racc, "b": b, "c": c,
                             "net": c - b, "mcnemar_exact_p": p,
                             "attainable_floor_p": fl,
                             "can_be_significant": fl < 0.05}
            out["paired"]["rows"][r["label"]] = cell

    print("\n  " + "-" * 74)
    print("  ⚠⚠ THE TWO COLUMNS ARE DIFFERENT READOUTS, NOT AN ESTIMATE AND A "
          "CORRECTION.")
    print("     Centring removes each class's mean over queries -- the part "
          "of the write that")
    print("     does not depend on what was asked. Section 27 moved 0.2480 to "
          "0.4120 by it.")
    print("  ⚠⚠ A CENTRED NUMBER MAY NOT BE PUT BESIDE AN UNCENTRED ONE "
          "(working rules 10b). The")
    print("     model's rows are UNCENTRED and belong beside the uncentred "
          "column only.")
    if mu_ref:
        print("  ⚠ `centred_ref` subtracts a mean estimated on the --centre-from "
              "split's queries (the")
        print("     same row, same seed); `centred` subtracts the mean over the "
              "analysed queries themselves.")
    print("  ⚠ This is the DIRECT path alone: the carriers also change what "
          "later layers read,")
    print("    and none of that is in these numbers.")
    if ceiling is not None:
        print("  ⚠ The ceiling is what THESE heads write natively at that K. "
              "It bounds the increment's")
        print("    direct path, not TL-alone accuracy in general (--all-heads "
              "is a different ceiling).")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2),
                                       encoding="utf-8")
        print(f"\n  [output] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
