"""Can 8 heads reading an offline K=10 memory deliver the increment? GPU.

SECTION 45's CONSTRUCTION, MOVED UP ONE LEVEL. There, a receiver with NO
demonstrations had eight carrier heads read an offline K=5 memory; it recovered
9.5 % of the negative log likelihood and touched one decision in 432. RESULTS
48.1 says why that was hard to read: both arms sat on the wrong side of
ln(36), so the arm never entered the region where an argmax could cross.

Here the receiver keeps its K=5 prompt and the carriers additionally read the
INCREMENT -- the other five demonstrations per class. Three things that made
the K=0 version uninformative do not apply:

  * the layers below the carriers are not blind, because the prompt supplies
    the K=5 demonstrations (RESULTS 47: the demonstrations' work happens below
    layer 22, and the carriers are all at 22 and above);
  * the baseline is 1.67 nats, far on the good side of uniform, where
    decisions do move;
  * the carriers' queries are formed from a normal residual stream rather than
    a zero-demonstration one (RESULTS 45.5b).

AND THE DENOMINATOR IS MEASURED, not assumed. RESULTS 49: going from five
demonstrations per class to ten is worth -0.497398 nats, CI [-0.583057,
-0.414041], same sign on all three seeds, and +49 decisions of 432. That is
what this arm is trying to deliver a share of.

FOUR ARMS, DIFFERING ONLY IN WHO MAY SEE THE INCREMENT COLUMNS. The base
columns -- the registered K=5 demonstrations -- are open to every head in
every arm.

    full K10 monolithic     everyone, no cache. The cache-equivalence gate's
                            reference; it must not share the cached path.
    all-head cached K10     everyone, through the cache. A GATE, not a method
                            arm: it is the ordinary K=10 prompt.
    selective TL K10 memory ONLY the eight carriers. This is the arm.
    K5-offset natural       nobody. The receiver sees exactly the registered
                            K=5 demonstrations, at their K=10 positions.

⚠ `K5-offset natural` IS NOT RESULTS 45's K=5 CONDITION. Every arm here runs
the full K=10 token sequence, so the query sits at about 6100 tokens rather
than 3050 and the base demonstrations sit at their K=10 positions. That is the
same `-offset` convention 13.5.2 already uses for the K=0 baseline, and it is
why this file's total is compared with section 49's for MAGNITUDE and not
bitwise.

THE BASE IS A SET OF COLUMNS, NOT A PREFIX. tools/check_demo_nesting.py
settled it on all three seeds: the K=5 demonstrations ARE inside the K=10 draw,
and none of them forms a contiguous prefix, because the task shuffles the whole
list at the end and the list length depends on k. Seed 42's sit at positions
2, 3, 10, 11, 14, 20, ... 356, 358, 359 of 360. Membership is decided on
(class, text) pairs, never on position.

PARAMETERISED BY --K-base / --K-full. The K=5 receiver + K=10 increment above
is the registered example and the module constants name its arms; any nested
pair runs through the same code with its own names (`arm_names`), so
K=2 -> 5 -- a K=2 receiver whose carriers read the other three demonstrations
per class -- is the same experiment one level DOWN. That one asks whether the
mechanism is already at work where accuracy may not move (script/
k5_increment_into_K2.sh); the nesting of the two draws is checked first,
exactly as it was for 5 -> 10.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import carrier_bundle as CB  # noqa: E402
from tools.baselines.common import load_candidate_space  # noqa: E402
from tools.baselines.forward import (build_prefixes,  # noqa: E402
                                     calibration_path,
                                     load_calibration_header)
from tools.check_demo_nesting import nesting_faults  # noqa: E402
from tools.icl_common import head_dim, run_provenance  # noqa: E402
from tools.model_args import add_model_arguments, load_model_from_args  # noqa: E402
from tools.k0_memory import mask_faults, visibility_mask  # noqa: E402
from tools.label_space import file_sha256  # noqa: E402
from tools.receiver_append import (load_for_append, merged_meta,  # noqa: E402
                                   plan_append, prefill_out, save_receiver_npz)
from tools.prereg_config import REGISTERED_SEEDS  # noqa: E402
from tools.baselines import fv_hook as FV  # noqa: E402
from tools.baselines.run_fv_on_icl import inject_hook  # noqa: E402
from tools.baselines.run_tsla_tl_on_icl import N_DISCOVERY_PROMPTS  # noqa: E402
from tools.baselines.tsla_hook import edit_layer as tsla_edit_layer  # noqa: E402
from tools.run_k0_receiver import (TSLA_ALPHAS, _install, _install_rows,  # noqa: E402
                                   argmax_swap_note, prepare_mask_rows,
                                   cache_equivalence_faults,
                                   cache_equivalence_report,
                                   format_report, load_tsla_vectors,
                                   parse_tsla_family, tsla_arm)
from tools.split_rows import (add_split_arguments, rows_for,  # noqa: E402
                              split_meta)
from tools.vector_arms import (alpha_zero_faults, fv_arm, fv_arms,  # noqa: E402
                               i2cl_arm, i2cl_module_hooks, icv_arm,
                               icv_arms, icv_hooks, install_vector_arm,
                               load_fv_sidecar, load_i2cl_sidecar,
                               load_icv_sidecar, load_tv_sidecar,
                               parse_float_list, parse_layer_list,
                               parse_vector_arm, tv_arm, tv_arms, tv_family)

def arm_names(k_base, k_full):
    """The four arms' names for one (receiver K, increment K) pair.

    role  mono   everyone, no cache -- the cache-equivalence reference
          all    everyone, cached  -- the GATE, not a method arm
          sel    only the carriers -- the arm
          base   nobody            -- the receiver at its offset positions
    """
    kb, kf = int(k_base), int(k_full)
    return {"mono": f"full K{kf} monolithic",
            "all": f"all-head cached K{kf}",
            "sel": f"selective TL K{kf} memory",
            "base": f"K{kb}-offset natural"}


ROLES = ("mono", "all", "sel", "base")
_K10 = arm_names(5, 10)
ARM_MONO = _K10["mono"]
ARM_ALL = _K10["all"]
ARM_SEL = _K10["sel"]
ARM_BASE = _K10["base"]
ARMS = (ARM_MONO, ARM_ALL, ARM_SEL, ARM_BASE)
assert ARMS == ("full K10 monolithic", "all-head cached K10",
                "selective TL K10 memory", "K5-offset natural")

# TSLA STEERING IN THIS SETTING (--tsla-vectors): the K=10 twin of 13.5.4's
# `TSLA-TL-zero-demo`. There a vector built from the K=5 prompt was added to
# a K=0 receiver; here a vector built from the K=10 prompt is added to the
# K=5 receiver, at the answer row of decoder layer 16, scaled by alpha, on
# ARM_BASE's mask -- the increment columns stay closed to every head, and the
# vector is the only thing that carries the increment. Two families from ONE
# head set (the 30 TSLA-TL heads selected on the K=10 prompt):
#   K10     v_K10 exactly as run_tsla_tl_on_icl builds it at K=10
#   K10inc  v_K10 - v_base, v_base being the same heads' summed answer-row
#           output on THIS receiver's own base forward, averaged over the
#           same discovery prompts: only what seeing the increment CHANGES,
#           so the receiver's own contribution is not added a second time
# (at K=0 the best alpha was 0.25, which is what double counting looks like).
def tsla_family_names(k_full, cls="tl"):
    """(full, inc) family names for one increment K and head class:
    `K10` / `K10inc` for TL (unchanged), `K10tr` / `K10trinc`,
    `K10rand` / `K10randinc`."""
    tag = {"tl": "", "tr": "tr", "random": "rand"}[cls]
    return f"K{int(k_full)}{tag}", f"K{int(k_full)}{tag}inc"


TSLA_CLASS_WHAT = {
    "tl": "the 30 TSLA-TL heads (upstream margin_add = mean_c <o, W_y - W_c> "
          "/ ||oP||)",
    "tr": "the 30 TSLA-TR heads (upstream cossim_norm = ||oP||: how much a "
          "head writes into the label subspace) -- the class the upstream "
          "recommends for fixed-label classification",
    "random": "30 heads drawn at random, seeded by the demo seed (the "
              "upstream's control)",
}


TSLA_FAMILY_FULL, TSLA_FAMILY_INC = tsla_family_names(10)
TSLA_FAMILIES = (TSLA_FAMILY_FULL, TSLA_FAMILY_INC)


def masks_for(arm, n_layers, n_heads, n_query, n_common, n_demo, n_live,
              carriers, base_cols, names=None):
    """One additive mask per layer; only WHO SEES THE INCREMENT differs.

    `names` is the arm_names() dict of the run; None means the K=10 names.
    """
    n = names or _K10
    out = {}
    for l in range(n_layers):
        if arm == n["all"]:
            allowed = list(range(n_heads))
        elif arm == n["sel"]:
            allowed = carriers.get(l, [])
        elif arm == n["base"]:
            allowed = []
        elif parse_tsla_family(arm) is not None:
            # A steering arm carries its offline information in the vector,
            # so the increment stays closed exactly as in ARM_BASE. Named
            # rather than left to fall through, so a future arm cannot
            # inherit this mask silently (run_k0_receiver does the same).
            allowed = []
        elif parse_vector_arm(arm) is not None:
            # FV / TV (tools/vector_arms): the same kind of arm as TSLA --
            # one offline vector, the increment closed to every head.
            allowed = []
        else:
            raise ValueError(f"{arm!r} has no mask; {n['mono']!r} runs "
                             "uncached")
        out[l] = visibility_mask(n_heads, n_query, n_common, n_demo, n_live,
                                 allowed, base_cols=base_cols)
    return out


def base_columns(seg, n_common, n_demo, base_demo_idx):
    """Demo-span column indices belonging to the base demonstrations.

    `seg` is segment_positions' per-token demo index over the FULL prompt.
    Returns indices in [0, n_demo), i.e. already relative to the demo span.
    """
    seg = np.asarray(seg)
    c, d = int(n_common), int(n_demo)
    pos = np.nonzero(np.isin(seg, list(base_demo_idx)))[0]
    if pos.size == 0:
        raise SystemExit("no token belongs to a base demonstration; the "
                         "segment map and the demo list disagree")
    if pos.min() < c or pos.max() >= c + d:
        raise SystemExit(
            f"base demonstration tokens run from {int(pos.min())} to "
            f"{int(pos.max())}, outside the demo span [{c}, {c + d}). The "
            "prefix cuts and the segment map describe different prompts")
    return pos - c


def tsla_arms(families=TSLA_FAMILIES, alphas=TSLA_ALPHAS):
    """One arm per (family, alpha); alpha = 0 in EVERY family, as its gate."""
    return [tsla_arm(a, fam) for fam in families for a in alphas]


def tsla_sidecar_extras(path, seeds, cls="tl"):
    """({seed: [(layer, head)]}, {seed: discovery query ids}) from the sidecar,
    for one head class (`cls`: tl, tr or random -- tsla_hook.sidecar_key).

    load_tsla_vectors gives the vectors; the increment family also needs the
    HEAD SET (v_base is those heads' output on this receiver) and the rows
    the K=10 mean was taken over (v_base must average the SAME prompts, or
    the difference subtracts two different means). Refused by name when
    absent, with the keys that are there (working rules 3.6).
    """
    from tools.baselines.tsla_hook import sidecar_key
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    runner = doc.get("runner") or {}
    vecs = runner.get(sidecar_key(cls)) or {}
    if not vecs:
        raise SystemExit(
            f"{path}: no {sidecar_key(cls)!r} under 'runner' (keys there: "
            f"{sorted(runner)}); a sidecar written before the TR/random "
            "classes carries TL only -- re-run run_tsla_tl_on_icl "
            "--discovery-only")
    ids = runner.get("tsla_discovery_query_ids")
    if ids is None:
        raise SystemExit(
            f"{path}: no 'tsla_discovery_query_ids' under 'runner' (keys "
            f"there: {sorted(runner)}). The increment family needs the rows "
            "the K=10 mean was taken over; re-run run_tsla_tl_on_icl "
            "--discovery-only, which records them.")
    heads, rows = {}, {}
    for s in seeds:
        cell = vecs.get(str(s), vecs.get(s))
        if not isinstance(cell, dict) or "heads" not in cell:
            keys = (sorted(cell) if isinstance(cell, dict)
                    else type(cell).__name__)
            raise SystemExit(f"{path}: seed {s} has no 'heads' (has {keys})")
        heads[s] = [(int(l), int(h)) for l, h in cell["heads"]]
        r = ids.get(str(s), ids.get(s))
        if not r:
            raise SystemExit(f"{path}: no discovery query ids for seed {s}; "
                             f"seeds present: {sorted(ids)}")
        rows[s] = [str(x) for x in r]
    return heads, rows


def discovery_rows_faults(sidecar_ids, run_ids, seed, n_expected):
    """The K=10 vector's discovery rows must be THIS run's first rows, in
    order and in number -- otherwise v_K10 and v_base average different
    prompts and their difference is not an increment."""
    a = [str(x) for x in sidecar_ids]
    b = [str(x) for x in run_ids]
    if len(a) != int(n_expected):
        return [f"seed {seed}: the sidecar averaged {len(a)} discovery "
                f"prompts; v_base here averages {int(n_expected)}"]
    if len(b) < len(a):
        return [f"seed {seed}: this run offers {len(b)} rows, the sidecar "
                f"averaged {len(a)}"]
    b = b[:len(a)]
    if a == b:
        return []
    i = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    return [f"seed {seed}: discovery row {i} is {a[i][:12]} in the sidecar "
            f"and {b[i][:12]} here; v_base would average different prompts "
            "than v_K10"]


def heads_residual_sum(store, o_by_layer, heads, d_head):
    """[d_model]: sum over `heads` of W_O^(l,h) @ (o_proj input slice of h).

    The quantity run_tsla_tl_on_icl sums for v_TSLA -- each head's
    answer-row output carried through its own W_O block -- read here off a
    cached, masked forward instead of an uncached natural one. `store` is
    {layer: o_proj input at the answer row}, as capture_o_proj leaves it.
    """
    d = int(d_head)
    out = None
    for l, h in heads:
        x = store[l]
        x = np.asarray(x.numpy() if hasattr(x, "numpy") else x,
                       dtype=np.float64)
        c = FV.head_slice(o_by_layer[l], h, d) @ x[h * d:(h + 1) * d]
        out = c if out is None else out + c
    if out is None:
        raise ValueError("no heads to sum")
    return out


def vector_report(v_full, v_base):
    """Norms of v_K10, v_base and their difference, and the angle between."""
    a = np.asarray(v_full, dtype=np.float64)
    b = np.asarray(v_base, dtype=np.float64)
    inc = a - b
    na, nb, ni = (float(np.linalg.norm(a)), float(np.linalg.norm(b)),
                  float(np.linalg.norm(inc)))
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")
    return {"norm_full": na, "norm_base": nb, "norm_inc": ni,
            "cosine_full_base": cos,
            "inc_over_full": ni / na if na > 0 else float("nan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--carrier-bundle", required=True)
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--label-space", required=True)
    ap.add_argument("--calibration-dir", required=True)
    add_model_arguments(ap)
    ap.add_argument("--task", default="trec_fine_per_class")
    ap.add_argument("--K-base", type=int, default=5)
    ap.add_argument("--K-full", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tsla-classes", default="tl",
                    help="comma list of head classes to steer with, from "
                         "tl,tr,random; two families per class "
                         "(tsla_family_names)")
    ap.add_argument("--tsla-vectors", default=None,
                    help="run_tsla_tl_on_icl --K 10 --discovery-only's "
                         "sidecar. Adds the K10 and K10inc steering "
                         f"families at alphas {list(TSLA_ALPHAS)}, on "
                         "ARM_BASE's mask, alpha=0 gated bitwise")
    ap.add_argument("--fv-vectors", default=None,
                    help="run_fv_increment's sidecar for THIS (K_base, K_full): adds the "
                         f"`FV-K<full> a=<alpha>` arms at alphas {list(TSLA_ALPHAS)} -- alpha * "
                         "v_FV at the answer row of the sidecar's layer, on ARM_BASE's mask, "
                         "alpha=0 gated bitwise (Todd et al.; alpha = 1 the main arm)")
    ap.add_argument("--tv-vectors", default=None,
                    help="run_tv_increment's sidecar for THIS (K_base, K_full): adds "
                         "`TV-K<full> L=<layer> a=1` -- the answer row of decoder layer <layer>'s "
                         "output REPLACED by theta -- for every candidate layer (or --tv-layers), on "
                         "ARM_BASE's mask, plus one a=0 gate (Hendel et al.; no strength grid)")
    ap.add_argument("--tv-layers", default="",
                    help="comma list restricting the TV arms to these candidate layers (the test "
                         "read passes the one tools/select_tv_layer.py chose on validation)")
    ap.add_argument("--tv-m5", action="store_true",
                    help="also run the descriptive `TV-K<full>m5` family (theta = the mean over five "
                         "dummy queries) at the same layers; validation only")
    ap.add_argument("--icv-vectors", default=None,
                    help="run_icv_increment's sidecar for THIS (K_base, K_full): adds "
                         "`ICV-K<full> a=<lambda>` -- the upstream ICVLayer on every layer's MLP "
                         "output, every position, the prefix cache rebuilt per lambda with the "
                         "hooks -- for the sidecar's grid (or --icv-lambdas), on ARM_BASE's mask, "
                         "plus the a=0 gate (Liu et al.)")
    ap.add_argument("--icv-lambdas", default="",
                    help="comma list restricting the ICV arms to these grid lambdas (the test read "
                         "passes the one tools/select_icv_lambda.py chose on validation)")
    ap.add_argument("--i2cl-vectors", default=None,
                    help="run_i2cl_increment's npz for THIS (K_base, K_full): adds `I2CL-K<full> "
                         "a=1` -- the calibrated linear injection on every layer's self_attn and mlp "
                         "outputs, every position, the prefix cache rebuilt with the hooks once per "
                         "seed -- on ARM_BASE's mask, plus the a=0 gate (Li et al.)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true",
                    help="if --out exists and is this run's file (same model, kernel, "
                         "dtype, task, K, head cut, split, rows, bundle), read only the "
                         "arms it lacks and write the union; the stored arms are kept "
                         "bit for bit (tools/receiver_append)")
    ap.add_argument("--cache-gate-ulp", type=float, default=1.0,
                    help="13.5.4(2) cache-equivalence bound in bf16 ulp (run_k0_receiver.MAX_ULP): 1 by "
                         "derivation; above 1 only with --cache-gate-source naming the independent "
                         "measurement it is the worst ulp of (check_two_path_noise, largest K, this dtype)")
    ap.add_argument("--cache-gate-source", default="",
                    help="where a --cache-gate-ulp above 1 was measured; written to the meta")
    add_split_arguments(ap)
    args = ap.parse_args(argv)
    if args.cache_gate_ulp > 1.0 and not args.cache_gate_source:
        raise SystemExit(f"--cache-gate-ulp {args.cache_gate_ulp:g} is above the derived 1: name the "
                         "measurement it comes from with --cache-gate-source")
    if not 0 <= args.K_base < args.K_full:
        raise SystemExit(f"--K-base {args.K_base} must be below --K-full "
                         f"{args.K_full}: the base has to nest in the full "
                         "draw and leave an increment")

    # THIS RUN'S NAMES, bound before anything else in main reads them. The
    # module constants are the K=10 instance; every use below is of these.
    names = arm_names(args.K_base, args.K_full)
    ARM_MONO, ARM_ALL, ARM_SEL, ARM_BASE = (names[r] for r in ROLES)
    ARMS = (ARM_MONO, ARM_ALL, ARM_SEL, ARM_BASE)
    from tools.baselines.tsla_hook import HEAD_CLASSES
    tsla_classes = [c.strip() for c in args.tsla_classes.split(",")
                    if c.strip()]
    bad_c = [c for c in tsla_classes if c not in HEAD_CLASSES]
    if bad_c:
        raise SystemExit(f"--tsla-classes {bad_c}: expected from {HEAD_CLASSES}")
    fams = {c: tsla_family_names(args.K_full, c) for c in tsla_classes}
    TSLA_FAMILIES = tuple(f for c in tsla_classes for f in fams[c])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ls = load_candidate_space(args.label_space, model=args.model,
                              query_manifest=args.query_manifest,
                              tokenizer=tok)
    from tools.build_query_manifest import load_split
    # THE ROWS SCORED come from --split, through the test lock; the rows the
    # TSLA increment vector's v_base is averaged over are ALWAYS validation's
    # first fifty -- that is where the sidecar's vector was built, and a test
    # read must not re-derive any part of its configuration on test rows.
    disc_by_seed = {s: list(load_split(args.query_manifest, "validation",
                                       demo_seed=s))
                    for s in REGISTERED_SEEDS}
    rows_by_seed = {s: rows_for(args.query_manifest, args.split, s,
                                freeze_manifest=args.freeze_manifest,
                                carriers=args.carrier_bundle,
                                label_space=args.label_space)
                    for s in REGISTERED_SEEDS}
    bundle = json.loads(Path(args.carrier_bundle).read_text(encoding="utf-8"))
    if str(bundle.get("scope")) != CB.SCOPE_FULL:
        raise SystemExit(f"{args.carrier_bundle} is a {bundle.get('scope')!r} "
                         "bundle; 13.5.2 names the full-validation carriers")

    tsla_v, tsla_heads, tsla_rows = {}, {}, {}
    if args.tsla_vectors:
        for c in tsla_classes:
            tsla_v[c] = load_tsla_vectors(args.tsla_vectors, REGISTERED_SEEDS,
                                          c)
            tsla_heads[c], tsla_rows = tsla_sidecar_extras(
                args.tsla_vectors, REGISTERED_SEEDS, c)
        for s in REGISTERED_SEEDS:
            # The sidecar averaged min(50, the split's rows) prompts, and so
            # does v_base below: a task with fewer than 50 validation rows
            # per seed (yelp: 5 classes x 4 = 20) is not a fault. The fixed
            # 50 refused yelp's test read (job 844784, RESULTS 63.10).
            bad = discovery_rows_faults(
                tsla_rows[s], [r["query_id"] for r in disc_by_seed[s]], s,
                min(N_DISCOVERY_PROMPTS, len(disc_by_seed[s])))
            if bad:
                raise SystemExit("; ".join(bad))
    fv_v = (load_fv_sidecar(args.fv_vectors, REGISTERED_SEEDS,
                            k_base=args.K_base, k_full=args.K_full)
            if args.fv_vectors else None)
    tv, tv_layers = {}, []
    if args.tv_vectors:
        # `tv_cand`, never `cand`: `cand` is the label-token list (run_k0_receiver
        # had it shadowed by the candidate layers once; the gate caught it)
        tv_cand, main_fam = load_tv_sidecar(args.tv_vectors, REGISTERED_SEEDS,
                                            k_base=args.K_base, k_full=args.K_full)
        want = parse_layer_list(args.tv_layers)
        bad_l = [l for l in want if l not in tv_cand]
        if bad_l:
            raise SystemExit(f"--tv-layers {bad_l}: not among the sidecar's candidate "
                             f"layers {tv_cand}")
        tv_layers = want or list(tv_cand)
        tv[tv_family(args.K_full)] = main_fam
        if args.tv_m5:
            tv[tv_family(args.K_full, "m5")] = load_tv_sidecar(
                args.tv_vectors, REGISTERED_SEEDS, k_base=args.K_base,
                k_full=args.K_full, key="tv_vectors_m5")[1]
    icv_v, icv_lams = None, []
    if args.icv_vectors:
        grid, icv_v = load_icv_sidecar(args.icv_vectors, REGISTERED_SEEDS,
                                       k_base=args.K_base, k_full=args.K_full)
        want = parse_float_list(args.icv_lambdas)
        bad_lam = [l for l in want if l not in grid]
        if bad_lam:
            raise SystemExit(f"--icv-lambdas {bad_lam}: not in the sidecar's grid {grid}")
        icv_lams = [l for l in (want or grid) if l > 0]
        if not icv_lams:
            raise SystemExit("--icv-lambdas: no lambda above 0 (0 is the gate, run always)")
    run_arms = (list(ARMS) + (tsla_arms(TSLA_FAMILIES) if tsla_v else [])
                + (fv_arms(args.K_full, TSLA_ALPHAS) if fv_v else [])
                + (tv_arms(tv_layers, args.K_full,
                           families=[""] + (["m5"] if args.tv_m5 else []))
                   if tv else [])
                + (icv_arms(args.K_full, [0.0] + icv_lams) if icv_v is not None else []))
    i2cl_v = None
    if args.i2cl_vectors:
        _, i2cl_v = load_i2cl_sidecar(args.i2cl_vectors, REGISTERED_SEEDS,
                                      k_base=args.K_base, k_full=args.K_full)
        run_arms += [i2cl_arm(args.K_full, on=False), i2cl_arm(args.K_full)]

    # --append: an existing file of this run's identity is kept and only the
    # arms it lacks are read (tools/receiver_append); anything else is refused
    todo, all_arms, old_meta, old_arrays = list(run_arms), list(run_arms), None, None
    if args.append and Path(args.out).is_file():
        _ran = {s: (rows_by_seed[s][:args.limit] if args.limit else rows_by_seed[s])
                for s in REGISTERED_SEEDS}
        old_meta, old_arrays = load_for_append(args.out, {
            "model": args.model, "method": args.method, "dtype": args.dtype,
            "attn": args.attn, "task": args.task, "K": args.K_full, "K_base": args.K_base,
            "top_n": args.top_n, "split": args.split, "limit": int(args.limit),
            "seeds": list(REGISTERED_SEEDS),
            "carrier_bundle_sha256": file_sha256(args.carrier_bundle),
            "query_manifest_sha256": file_sha256(args.query_manifest),
            "label_space_sha256": file_sha256(args.label_space)}, _ran)
        todo, all_arms = plan_append(old_meta, run_arms)
        print(f"  --append: {args.out} holds {len(old_meta['arms'])} arms; "
              f"this run adds {len(todo)}: {todo}")
        if not todo:
            print("  --append: nothing to add; the file stands as it is")
            return 0

    print("=" * 78)
    print(f"K={args.K_full} INCREMENT INTO A K={args.K_base} RECEIVER "
          f"-- {args.split.upper()}, EXPLORATORY")
    if args.split != "validation":
        print(f"  ⚠⚠ {args.split}: the ONE-SHOT test split, read under "
              "prereg 14.0b-23 (optional/descriptive; configuration frozen "
              "on validation)")
    print("=" * 78)
    for a in run_arms:
        print(f"    {a}")
    if tsla_v:
        print(f"  TSLA steering from {args.tsla_vectors}: a K={args.K_full} "
              f"vector at the answer row of the mid-stack layer (n_layers // 2), on "
              f"{ARM_BASE!r}'s mask.")
        for c in tsla_classes:
            print(f"    {fams[c][0]:<10} v_K{args.K_full} from "
                  f"{TSLA_CLASS_WHAT[c]}, selected at K={args.K_full}")
            print(f"    {fams[c][1]:<10} v_K{args.K_full} - v_base, the same "
                  "heads' output on THIS receiver over the same "
                  f"{N_DISCOVERY_PROMPTS} prompts subtracted")
        print("    [PASS] the sidecar's discovery rows are this run's first "
              f"{N_DISCOVERY_PROMPTS}, in order, every seed")
    if fv_v:
        print(f"  FV steering from {args.fv_vectors}: alpha * v_FV at the answer "
              f"row of the sidecar's layer, on {ARM_BASE!r}'s mask (Todd et al.; "
              "alpha = 1 the main arm, the grid descriptive)")
        for s_ in REGISTERED_SEEDS:
            print(f"    seed {s_}: layer {fv_v[s_][0]}, |v_FV| "
                  f"{np.linalg.norm(fv_v[s_][1]):.4f}")
    if tv:
        print(f"  TV replacement from {args.tv_vectors}: the answer row of decoder "
              f"layer L's output replaced by theta_L (Hendel et al.), on "
              f"{ARM_BASE!r}'s mask; layers {tv_layers}, families {sorted(tv)}; "
              f"{tv_arm(tv_layers[0], args.K_full, on=False)!r} is the gate")
    if icv_v is not None:
        print(f"  ICV steering from {args.icv_vectors}: the upstream ICVLayer on every "
              f"layer's MLP output, every position (Liu et al.), on {ARM_BASE!r}'s "
              f"mask; lambdas {icv_lams}, one prefix cache per lambda; "
              f"{icv_arm(0.0, args.K_full)!r} is the gate")
    if i2cl_v is not None:
        print(f"  I2CL from {args.i2cl_vectors}: the calibrated linear injection on every "
              "layer's self_attn and mlp outputs, every position (Li et al.), on "
              f"{ARM_BASE!r}'s mask; one hooked prefix cache per seed; "
              f"{i2cl_arm(args.K_full, on=False)!r} is the gate")
        for s_ in REGISTERED_SEEDS:
            c_ = i2cl_v[s_]["coef"]
            print(f"    seed {s_}: lambda mean {c_[:, :, 0].mean():.4f} [{c_[:, :, 0].min():.3f}, "
                  f"{c_[:, :, 0].max():.3f}], beta mean {c_[:, :, 1].mean():.4f}")
    print("  the arms differ ONLY in who may see the INCREMENT columns; the "
          "base ones are")
    print("  open to every head in every arm.")
    print(f"  ⚠ {ARM_BASE!r} is NOT section 45's K={args.K_base} condition: "
          "every arm runs the full")
    print(f"  ⚠ K={args.K_full} sequence, so the query sits later and the base "
          "demonstrations sit at")
    print(f"  ⚠ their K={args.K_full} positions. Compare with a natively "
          "rendered run by magnitude, not bitwise.")

    prefixes, text_of, checks, render_prompt = build_prefixes(
        args.task, args.K_full, list(REGISTERED_SEEDS), args.query_manifest,
        args.calibration_dir, ls)
    for c in checks:
        print(f"  [PASS] {c}")

    # WHICH BLOCKS ARE THE BASE. Decided on (class, text), never on position.
    from tools.prereg_task import docs_to_rows, prefix_demo_docs
    from tools.probe_prototype_shrinkage import manifest_reservation
    from tools.prompt_render import TaskRenderer
    renderer = TaskRenderer(args.task)
    from tools.probe_kappa_matrix import segment_positions
    base_idx = {}
    for s in REGISTERED_SEEDS:
        excl = manifest_reservation(args.query_manifest, demo_seed=s)
        small_docs = prefix_demo_docs(args.task, args.K_base, [s],
                                      excluded_docs=excl)[s]
        large_docs = prefix_demo_docs(args.task, args.K_full, [s],
                                      excluded_docs=excl)[s]
        small = docs_to_rows(args.task, small_docs)
        large = docs_to_rows(args.task, large_docs)
        bad = nesting_faults(small, large, k_small=args.K_base,
                             k_large=args.K_full)
        if bad:
            raise SystemExit(f"seed {s}: " + "; ".join(bad))
        header, _ = load_calibration_header(
            calibration_path(args.calibration_dir, args.task, args.K_full, s))
        # THE INDEX CORRESPONDENCE IS ASSERTED, NOT ASSUMED. build_prefixes
        # renders the blocks itself; if `large` is not the same list in the
        # same order, every base column would be off and nothing downstream
        # would notice.
        rebuilt = renderer.build_prefix(large_docs, header["abstract_labels"])
        if rebuilt != list(prefixes[s]):
            raise SystemExit(
                f"seed {s}: rebuilding the blocks from prefix_demo_rows does "
                f"not reproduce build_prefixes' output ({len(rebuilt)} vs "
                f"{len(prefixes[s])} blocks, first difference at index "
                f"{next(i for i, (a, b) in enumerate(zip(rebuilt, prefixes[s])) if a != b)}). "
                "The base columns would be chosen by an index that means "
                "something else.")
        small_set = set(small)
        base_idx[s] = [i for i, d in enumerate(large) if d in small_set]
        if len(base_idx[s]) != len(small):
            raise SystemExit(
                f"seed {s}: {len(base_idx[s])} blocks matched {len(small)} "
                "base demonstrations; a duplicate (class, text) would do this")
        print(f"  [PASS] seed {s}: {len(base_idx[s])} of {len(large)} blocks "
              f"are the registered K={args.K_base} demonstrations, at "
              f"positions {base_idx[s][:4]}...{base_idx[s][-2:]}")

    import torch
    model = load_model_from_args(args)
    n_l = int(model.config.num_hidden_layers)
    tsla_edit = tsla_edit_layer(n_l)
    n_h = int(model.config.num_attention_heads)
    cand = list(ls.candidate_token_ids)
    if len(cand) < 2:
        raise SystemExit(f"the label space has {len(cand)} candidates; nothing to read out")
    d_h = head_dim(model)      # not hidden_size // n_h (icl_common.head_dim)
    o_by_layer = {}          # o_proj weights, float64, only the layers needed

    out = {a: {} for a in all_arms}
    if old_arrays is not None:
        prefill_out(out, old_arrays, [a for a in all_arms if a not in todo],
                    list(REGISTERED_SEEDS))
    meta_seeds, ran_by_seed = {}, {}
    for s in REGISTERED_SEEDS:
        blocks = prefixes[s]
        carriers = {}
        for l, h in CB.heads_for(bundle, s, args.top_n):
            carriers.setdefault(int(l), []).append(int(h))
        header, _ = load_calibration_header(
            calibration_path(args.calibration_dir, args.task, args.K_full, s))
        rows = rows_by_seed[s][:args.limit] if args.limit else rows_by_seed[s]
        ran_by_seed[s] = rows
        through = "\n\n".join(blocks) + "\n\n"
        n_c = len(tok("", add_special_tokens=True).input_ids)
        pre_ids = tok(through, return_tensors="pt").input_ids.to(model.device)
        n_pre = int(pre_ids.shape[1])
        n_demo = n_pre - n_c
        with torch.no_grad():
            base_cache = model(pre_ids, use_cache=True).past_key_values
        # ICV edits every position, so its prefix is a DIFFERENT prefix: one
        # cache per lambda, prefilled with the hooks installed (the spec's
        # "the injected prefix KV once per seed"); I2CL likewise. They are
        # built ONE AT A TIME, each in its own pass over the queries after
        # the natural-prefix arms, and freed after it. Keeping every steered
        # cache resident put the K=10 prefix of an MHA model (Llama-2: about
        # 4 GB per cache, five lambdas) beside the weights and the base cache
        # and OOMed SEc36's k10 on a 40 GB card (job 845845). The forwards
        # are the same as before -- the same hooks on the same ids build the
        # same cache, and every arm continues it with the same suffix -- so
        # the logits do not change; only the order of the arms does.
        def build_icv_cache(_lam):
            _h = icv_hooks(model, icv_v[s], _lam, torch)
            try:
                with torch.no_grad():
                    return model(pre_ids, use_cache=True).past_key_values
            finally:
                for _x in _h:
                    _x.remove()

        def build_i2cl_cache():
            _h = i2cl_module_hooks(model, i2cl_v[s]["cv"], i2cl_v[s]["coef"], torch)
            try:
                with torch.no_grad():
                    return model(pre_ids, use_cache=True).past_key_values
            finally:
                for _x in _h:
                    _x.remove()

        def _steered(arm):
            """('ICV', lam) / ('I2CL',) for an arm that continues a hooked
            prefix; None for every arm that continues the natural one (the
            a=0 arms included: 0 * v adds nothing, they ARE base forwards)."""
            _pv = parse_vector_arm(arm)
            if _pv is None or _pv[3] == 0.0 or _pv[0] not in ("ICV", "I2CL"):
                return None
            return ("ICV", _pv[3]) if _pv[0] == "ICV" else ("I2CL",)
        todo_base = [a for a in todo if _steered(a) is None]
        steered_passes = []
        for _a in todo:
            _g = _steered(_a)
            if _g is not None and _g not in [g for g, _ in steered_passes]:
                steered_passes.append((_g, [b for b in todo if _steered(b) == _g]))

        first = render_prompt(blocks, text_of[rows[0]["query_id"]])
        _ids, seg, _lab, _cls = segment_positions(first, tok,
                                                  header["abstract_labels"])
        cols = base_columns(seg, n_c, n_demo, base_idx[s])
        # the visibility patterns, once per arm per seed (k0_memory
        # .mask_prefix_rows); the query loops expand them on the card
        rows_by_arm = {a: prepare_mask_rows(
                           masks_for(a, n_l, n_h, 1, n_c, n_demo, 1, carriers,
                                     cols, names=names),
                           torch, model.device)
                       for a in todo if a != ARM_MONO}
        frac = cols.size / n_demo
        print(f"\n  seed {s}: carriers {carriers}")
        print(f"    C_s {n_c}, D_s {n_demo} tokens; the base is {cols.size} "
              f"of them ({frac:.1%})")
        if not 0 < cols.size < n_demo:
            raise SystemExit(
                f"seed {s}: the base covers {cols.size} of {n_demo} demo "
                "columns. Nothing or everything means there is no increment "
                "and the arms collapse onto one another")
        meta_seeds[str(s)] = {"n_common": n_c, "n_demo": int(n_demo),
                              "n_base_columns": int(cols.size),
                              "base_block_idx": base_idx[s],
                              "carriers": {str(k): v
                                           for k, v in carriers.items()}}

        # THE MASK IS CHECKED ONCE PER SEED, and in two halves because the
        # layers are two kinds. mask_faults refuses an empty allowed set by
        # design -- "that is the K0-offset baseline, not the selective-memory
        # arm" -- which is right when it is asked about the whole arm and
        # wrong for the 22 layers below the carriers, where base-only IS the
        # intended mask. So carrier layers go through mask_faults, and
        # carrier-less ones are asserted to equal the baseline arm's mask,
        # which is the same statement without the misfire.
        m_sel = masks_for(ARM_SEL, n_l, n_h, 1, n_c, n_demo, 1, carriers, cols,
                          names=names)
        m_base = masks_for(ARM_BASE, n_l, n_h, 1, n_c, n_demo, 1, carriers,
                           cols, names=names)
        n_carrier_layers = 0
        for l, mm in m_sel.items():
            if carriers.get(l):
                n_carrier_layers += 1
                f = mask_faults(mm, n_demo, carriers[l], n_common=n_c,
                                base_cols=cols)
                if f:
                    raise SystemExit(f"seed {s} layer {l}: " + "; ".join(f[:3]))
            elif not np.array_equal(mm, m_base[l]):
                raise SystemExit(
                    f"seed {s} layer {l} has no carrier, so the selective arm "
                    "must be bitwise the baseline arm there, and it is not")
        print(f"    [PASS] the selective mask opens the base to every head and "
              f"the increment to the carriers only ({n_carrier_layers} carrier "
              f"layers checked, {n_l - n_carrier_layers} carrier-free layers "
              "identical to the baseline arm)")

        # v_base FOR THE INCREMENT FAMILY: the K=10-selected heads' summed
        # answer-row output on this receiver's own base forward (cached,
        # increment closed), averaged over the sidecar's discovery rows.
        # Same operator as the K=10 vector -- o_proj input through the
        # head's own W_O block -- on the forward the vector will be added to.
        tsla_vec = {}
        if tsla_v and any(parse_tsla_family(a) is not None for a in todo):
            from tools.probe_carrier_direct_response import capture_o_proj
            heads_by = {c: tsla_heads[c][s] for c in tsla_classes}
            layers_s = sorted({l for hh in heads_by.values() for l, _ in hh})
            for l in layers_s:
                if l not in o_by_layer:
                    o_by_layer[l] = (model.model.layers[l].self_attn.o_proj
                                     .weight.detach().to(torch.float32).cpu()
                                     .numpy().astype(np.float64))
            disc = (disc_by_seed[s][:args.limit] if args.limit
                    else disc_by_seed[s])[:N_DISCOVERY_PROMPTS]
            acc_v = {c: None for c in tsla_classes}
            for r in disc:
                fid = tok(render_prompt(blocks, text_of[r["query_id"]]),
                          return_tensors="pt").input_ids.to(model.device)
                suffix = fid[:, n_pre:]
                n_live = int(suffix.shape[1])
                store = {}
                handles = _install_rows(model, rows_by_arm[ARM_BASE], n_live,
                                        torch)
                handles += capture_o_proj(model, layers_s, store, torch)
                try:
                    base_cache.crop(n_pre)
                    with torch.no_grad():
                        model(suffix, past_key_values=base_cache,
                              use_cache=True)
                finally:
                    for h in handles:
                        h.remove()
                    base_cache.crop(n_pre)
                # ONE capture, one residual sum per class: the head sets
                # differ, the forward does not.
                for c in tsla_classes:
                    part = heads_residual_sum(store, o_by_layer, heads_by[c],
                                              d_h)
                    acc_v[c] = part if acc_v[c] is None else acc_v[c] + part
            meta_seeds[str(s)]["tsla"] = {}
            for c in tsla_classes:
                v_base = acc_v[c] / float(len(disc))
                v_full = np.asarray(tsla_v[c][s], dtype=np.float64)
                tsla_vec[fams[c][0]] = v_full
                tsla_vec[fams[c][1]] = v_full - v_base
                rep = vector_report(v_full, v_base)
                rep["n_base_prompts"] = int(len(disc))
                rep["heads"] = [[int(l), int(h)] for l, h in heads_by[c]]
                meta_seeds[str(s)]["tsla"][c] = rep
                print(f"    TSLA[{c}]: {len(heads_by[c])} heads in layers "
                      f"{sorted({l for l, _ in heads_by[c]})}; "
                      f"|v_K{args.K_full}| {rep['norm_full']:.4f}, |v_base| "
                      f"{rep['norm_base']:.4f}, cos "
                      f"{rep['cosine_full_base']:.4f}, "
                      f"|v_K{args.K_full} - v_base| {rep['norm_inc']:.4f} "
                      f"({rep['inc_over_full']:.1%} of |v_K{args.K_full}|), "
                      f"over {len(disc)} prompts")
            if args.limit and len(disc) < N_DISCOVERY_PROMPTS:
                print(f"    ⚠ --limit: v_base averages {len(disc)} prompts, "
                      f"not {N_DISCOVERY_PROMPTS}; a smoke, not the artifact")
            elif len(disc) < N_DISCOVERY_PROMPTS:
                print(f"    note: v_base averages {len(disc)} prompts -- every "
                      "discovery row this split has (fewer than "
                      f"{N_DISCOVERY_PROMPTS}), the same rows the sidecar averaged")

        for qi, r in enumerate(rows):
            full = render_prompt(blocks, text_of[r["query_id"]])
            fid = tok(full, return_tensors="pt").input_ids.to(model.device)
            if not torch.equal(fid[0, :n_pre], pre_ids[0]):
                raise SystemExit(
                    f"seed {s} query {r['query_id'][:12]}: the prompt does not "
                    "begin with its own prefix at the token level")
            suffix = fid[:, n_pre:]
            n_live = int(suffix.shape[1])
            for arm in todo_base:
                if arm == ARM_MONO:
                    with torch.no_grad():
                        lg = model(fid).logits[0, -1]
                    out[arm].setdefault(s, []).append(
                        lg[cand].to(torch.float64).cpu().numpy())
                    continue
                handles = _install_rows(model, rows_by_arm[arm], n_live, torch)
                fa = parse_tsla_family(arm)
                if fa is not None:
                    # A second hook on a different module: the mask is a
                    # pre-hook on self_attn, this a forward hook on the
                    # decoder layer, so they compose (run_k0_receiver's
                    # arrangement). hs[:, -1, :] is the answer row because
                    # the cached forward runs only the receiver's tokens.
                    handles.append(inject_hook(model, tsla_edit, fa[1],
                                               tsla_vec[fa[0]]))
                # FV / TV / ICV: the sidecar's operator; [] at a=0
                handles += install_vector_arm(model, arm, s, fv_v, tv,
                                              inject_hook, torch, icv=icv_v,
                                              i2cl=i2cl_v)
                # every arm of this pass continues the natural prefix; the
                # ICV / I2CL arms at a nonzero strength continue THEIR OWN
                # hooked prefix and run in the steered passes below
                cache = base_cache
                try:
                    cache.crop(n_pre)
                    with torch.no_grad():
                        lg = model(suffix, past_key_values=cache,
                                   use_cache=True).logits[0, -1]
                finally:
                    for h in handles:
                        h.remove()
                    cache.crop(n_pre)
                out[arm].setdefault(s, []).append(
                    lg[cand].to(torch.float64).cpu().numpy())
            if qi == 0:
                rep = cache_equivalence_report(out[ARM_ALL][s][0],
                                               out[ARM_MONO][s][0])
                print(f"    13.5.4(2) cache equivalence, seed {s}:")
                for line in format_report(rep):
                    print(f"      {line}")
                _note = argmax_swap_note(out[ARM_ALL][s][0],
                                         out[ARM_MONO][s][0],
                                         max_ulp=args.cache_gate_ulp)
                if _note:
                    print(f"      ⚠ {_note}")
                gbad = cache_equivalence_faults(out[ARM_ALL][s][0],
                                                out[ARM_MONO][s][0],
                                                max_ulp=args.cache_gate_ulp)
                if gbad:
                    raise SystemExit(f"seed {s}: cache equivalence FAILED\n  "
                                     + "\n  ".join(gbad))
                print("    [PASS] 13.5.4(2) cache equivalence")
                # 13.5.4(4), per family: alpha = 0 must reproduce ARM_BASE
                # ELEMENTWISE. 0.0 * v is exactly zero and adding it changes
                # no bit, so this is an identity, not a tolerance.
                for fam in (TSLA_FAMILIES if tsla_v else ()):
                    _z = tsla_arm(0.0, fam)
                    _a = np.asarray(out[_z][s][0])
                    _b = np.asarray(out[ARM_BASE][s][0])
                    if not np.array_equal(_a, _b):
                        _d = np.abs(_a - _b)
                        raise SystemExit(
                            f"seed {s}: 13.5.4(4) FAILED -- {_z!r} does not "
                            f"reproduce {ARM_BASE!r} elementwise "
                            f"({int((_d > 0).sum())} of {_d.size} candidates "
                            f"differ, max {_d.max():.3e}). alpha=0 adds "
                            "exactly zero, so this is an identity: a "
                            "difference means the injection hook is doing "
                            "something the formula does not say.")
                    print(f"    [PASS] 13.5.4(4) {_z!r} reproduces "
                          f"{ARM_BASE!r} elementwise")
                for _z in (([fv_arm(0.0, args.K_full)] if fv_v else [])
                           + ([tv_arm(tv_layers[0], args.K_full, on=False)]
                              if tv else [])
                           + ([icv_arm(0.0, args.K_full)]
                              if icv_v is not None else [])
                           + ([i2cl_arm(args.K_full, on=False)]
                              if i2cl_v is not None else [])):
                    _f = alpha_zero_faults(out[_z][s][0], out[ARM_BASE][s][0],
                                           _z, ARM_BASE)
                    if _f:
                        raise SystemExit(f"seed {s}: 13.5.4(4) FAILED -- {_f}")
                    print(f"    [PASS] 13.5.4(4) {_z!r} reproduces "
                          f"{ARM_BASE!r} elementwise")
            if (qi + 1) % 40 == 0:
                print(f"    {qi + 1}/{len(rows)}", flush=True)

        # the steered arms: one hooked prefix cache at a time, its arms over
        # every query (the same suffix forwards the single loop above ran),
        # then the cache is freed before the next one is built
        for _g, _arms in steered_passes:
            pass_cache = (build_icv_cache(_g[1]) if _g[0] == "ICV"
                          else build_i2cl_cache())
            print(f"    steered pass {_g}: {len(_arms)} arm(s)", flush=True)
            for qi, r in enumerate(rows):
                full = render_prompt(blocks, text_of[r["query_id"]])
                fid = tok(full, return_tensors="pt").input_ids.to(model.device)
                suffix = fid[:, n_pre:]
                n_live = int(suffix.shape[1])
                for arm in _arms:
                    handles = _install_rows(model, rows_by_arm[arm], n_live, torch)
                    handles += install_vector_arm(model, arm, s, fv_v, tv,
                                                  inject_hook, torch, icv=icv_v,
                                                  i2cl=i2cl_v)
                    try:
                        pass_cache.crop(n_pre)
                        with torch.no_grad():
                            lg = model(suffix, past_key_values=pass_cache,
                                       use_cache=True).logits[0, -1]
                    finally:
                        for h in handles:
                            h.remove()
                        pass_cache.crop(n_pre)
                    out[arm].setdefault(s, []).append(
                        lg[cand].to(torch.float64).cpu().numpy())
                if (qi + 1) % 40 == 0:
                    print(f"    {_g} {qi + 1}/{len(rows)}", flush=True)
            del pass_cache
            torch.cuda.empty_cache()

    meta_doc = {
            "spec": "EXPLORATORY: RESULTS 45's construction one level up",
            "cache_gate": {"max_ulp": float(args.cache_gate_ulp),
                           "source": args.cache_gate_source or "the derived 1 ulp (14.0b-22)"},
            "arm_name": f"K={args.K_full} increment into a K={args.K_base} "
                        "receiver",
            "model": args.model, "method": args.method, "task": args.task,
            "dtype": args.dtype, "attn": args.attn,
            "K": args.K_full, "K_base": args.K_base,
            "arms": list(all_arms), "top_n": args.top_n,
            # THE SEEDS AND THE ROLES, in the meta and not only as arrays.
            # analyze_k0_receiver reads both from here; writing `seeds` only
            # as an npz key cost a KeyError on a complete file, and leaving
            # the roles out would have had it look for the K=0 spellings of
            # four arms this run deliberately names differently.
            "seeds": [int(x) for x in REGISTERED_SEEDS],
            "roles": {"mono": ARM_MONO, "all": ARM_ALL,
                      "sel": ARM_SEL, "base": ARM_BASE},
            "registered_arm": args.top_n is None,
            "carrier_bundle_sha256": file_sha256(args.carrier_bundle),
            "query_manifest_sha256": file_sha256(args.query_manifest),
            "label_space_sha256": file_sha256(args.label_space),
            **split_meta(args),
            "per_seed": meta_seeds, "limit": int(args.limit),
            "tsla": ({"source": str(args.tsla_vectors),
                      "edit_layer": int(tsla_edit),
                      "alphas": [float(a) for a in TSLA_ALPHAS],
                      "n_discovery_prompts": int(N_DISCOVERY_PROMPTS),
                      "mask": "ARM_BASE's: the increment columns closed to "
                              "every head; the vector is the only carrier",
                      "classes": list(tsla_classes),
                      "families": {
                          **{fams[c][0]:
                             f"v_K{args.K_full}: run_tsla_tl_on_icl's vector "
                             f"from {TSLA_CLASS_WHAT[c]}, built on the natural "
                             f"K={args.K_full} prompt and added at the answer "
                             f"row of layer {tsla_edit} on the "
                             f"K={args.K_base} receiver"
                             for c in tsla_classes},
                          **{fams[c][1]:
                             f"v_K{args.K_full} - v_base: the same "
                             f"{c} heads' summed answer-row output on this "
                             "receiver's base forward, averaged over the "
                             "same discovery prompts, subtracted -- only "
                             "what seeing the increment changes"
                             for c in tsla_classes}},
                      "spec": "EXPLORATORY: the K=10 twin of 13.5.4's "
                              "TSLA-TL-zero-demo; not a registered arm"}
                     if tsla_v else None),
            "fv": ({"source": str(args.fv_vectors),
                    "family": f"FV-K{args.K_full}",
                    "alphas": [float(a) for a in TSLA_ALPHAS],
                    "per_seed": {str(s_): {"layer": int(fv_v[s_][0]),
                                           "norm": float(np.linalg.norm(fv_v[s_][1]))}
                                 for s_ in REGISTERED_SEEDS},
                    "mask": "ARM_BASE's: the increment columns closed to "
                            "every head; the vector is the only carrier",
                    "what": "Todd et al.'s function vector, built by "
                            "run_fv_increment from the extra demonstrations of "
                            f"the K={args.K_full} draw (the K={args.K_base} "
                            "demonstrations as extraction queries, exact CIE "
                            "after an attribution-patching screen) and added as "
                            "alpha * v_FV at the answer row of layer L // 3; "
                            "alpha = 1 is the main arm, the grid descriptive",
                    "spec": "EXPLORATORY baseline (prereg 14.0b-25, "
                            "tools/baselines/specs/fv_atp_adapted_spec.md); "
                            "not a registered arm"}
                   if fv_v else None),
            "tv": ({"source": str(args.tv_vectors),
                    "families": sorted(tv), "layers": [int(l) for l in tv_layers],
                    "gate": tv_arm(tv_layers[0], args.K_full, on=False),
                    "per_seed": {str(s_): {str(l): float(np.linalg.norm(
                        tv[tv_family(args.K_full)][s_][l])) for l in tv_layers}
                                 for s_ in REGISTERED_SEEDS},
                    "mask": "ARM_BASE's: the increment columns closed to "
                            "every head; the vector is the only carrier",
                    "what": "Hendel et al.'s task vector: theta_L is the "
                            "answer-row output of decoder layer L on one "
                            "forward over the extra demonstrations of the "
                            f"K={args.K_full} draw plus a dummy query (a "
                            f"K={args.K_base} demonstration's text), built by "
                            "run_tv_increment; the arm REPLACES the answer row "
                            "of layer L's output on this receiver's cached "
                            "forward. The layer is chosen on validation among "
                            "the candidates by tools/select_tv_layer.py (ties "
                            "to the smaller layer); the m5 family is the mean "
                            "over five dummy queries, descriptive",
                    "spec": "EXPLORATORY baseline (prereg 14.0b-25, "
                            "tools/baselines/specs/task_vector_adapted_spec.md); "
                            "not a registered arm"}
                   if tv else None),
            "icv": ({"source": str(args.icv_vectors),
                     "family": f"ICV-K{args.K_full}",
                     "lambdas": [float(l) for l in icv_lams],
                     "gate": icv_arm(0.0, args.K_full),
                     "mask": "ARM_BASE's: the increment columns closed to "
                             "every head; the direction is the only carrier, "
                             "and the prefix cache is rebuilt per lambda with "
                             "the hooks",
                     "what": "Liu et al.'s in-context vector, built by "
                             "run_icv_increment from the extra demonstrations "
                             f"of the K={args.K_full} draw ((x, xy) last-token "
                             "differences; centred PC1 + mean, the upstream's "
                             "formula) and applied as the upstream ICVLayer "
                             "on every decoder layer's MLP output at every "
                             "position; lambda is chosen on validation by "
                             "tools/select_icv_lambda.py (NLL, ties to the "
                             "smaller lambda)",
                     "spec": "EXPLORATORY baseline (prereg 14.0b-25, "
                             "tools/baselines/specs/icv_adapted_spec.md); "
                             "not a registered arm"}
                    if icv_v is not None else None),
            "i2cl": ({"source": str(args.i2cl_vectors),
                      "family": f"I2CL-K{args.K_full}",
                      "gate": i2cl_arm(args.K_full, on=False),
                      "per_seed": {str(s_): {
                          "lambda_mean": float(i2cl_v[s_]["coef"][:, :, 0].mean()),
                          "beta_mean": float(i2cl_v[s_]["coef"][:, :, 1].mean())}
                          for s_ in REGISTERED_SEEDS},
                      "mask": "ARM_BASE's: the increment columns closed to "
                              "every head; the context vectors are the only "
                              "carrier, and the prefix cache is rebuilt with "
                              "the calibrated hooks per seed",
                      "what": "Li et al.'s implicit in-context learning, "
                              "built by run_i2cl_increment from the extra "
                              f"demonstrations of the K={args.K_full} draw "
                              "(mean last-token self_attn / mlp outputs) with "
                              "4 L scalars calibrated on the extra "
                              "demonstrations as pseudo-queries after the "
                              f"K={args.K_base} prefix (noisy "
                              "self-calibration); applied as out <- beta * "
                              "out + lambda * cv on every layer's self_attn "
                              "and mlp outputs at every position",
                      "spec": "EXPLORATORY baseline (prereg 14.0b-25, "
                              "tools/baselines/specs/i2cl_adapted_spec.md); "
                              "not a registered arm"}
                     if i2cl_v is not None else None),
            "provenance": run_provenance(),
            "note": "the base columns are the registered K=5 demonstrations, "
                    "chosen by (class, text) membership and SCATTERED through "
                    "the K=10 prompt; K5-offset natural is not section 45's "
                    "K=5 condition because every arm runs the full K=10 "
                    "sequence"}
    if old_meta is not None:
        meta_doc = merged_meta(old_meta, meta_doc, todo)
    save_receiver_npz(args.out, out, all_arms, ran_by_seed, list(REGISTERED_SEEDS), meta_doc)
    print(f"\n  [output] {args.out}")
    print("  read with tools/analyze_k0_receiver.py: `total` is "
          f"{ARM_MONO} - {ARM_BASE}, which is what the increment is")
    print("  worth here, and `recovered` is what the eight carriers deliver "
          "of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
