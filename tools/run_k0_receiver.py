"""13.5's K=0 receiver with selective K=5 latent memory. GPU.

THE ARMS DIFFER ONLY BY A MASK, which is what makes the comparison exact.

Prefill C_s || D_s once, keep the cache, then run R_s(q) against it. The
receiver's tokens then occupy their ORIGINAL positions from the full K=5
prompt automatically -- there is no position bookkeeping to get wrong, because
this IS the K=5 forward. What changes is only which heads may look at the demo
columns:

  full K5 monolithic    the ordinary forward, NO cache -- the reference
  all-head cached K5    every head reads the cache -- a CACHE EQUIVALENCE
                        GATE, not a method arm (13.5.4), and 13.5.4(2) blocks
                        the section unless it matches the monolithic arm to
                        within 1 % of that query's inter-class logit range
                        with identical candidate argmax
  selective TL K5 mem   ONLY the frozen H_8 read it -- 13.5.2's arm
  K0-offset natural     no head does -- the baseline / empty-memory placebo

Same weights, same cache, same positions, same softmax. Every difference
between the three is attributable to visibility and to nothing else, which is
what 13.5.2 wants the K0-offset baseline for: it stops a virtual-position
effect being counted as a memory effect.

`K0-native` -- C_s || R_s(q) renumbered contiguously -- is a SEPARATE run
without the demo cache, and 13.5.2 keeps it as an extra reference only.

⚠ THE NAME IS NOT `zero-shot Method A`. 13.5.1 fixes the arm as `K0 receiver
+ selective K5 latent memory`: the receiver's text carries no demonstrations,
but the memory is built from a labelled K=5 bank, so this is a zero-demo
receiver with labelled offline memory and not zero-shot learning. Writing the
short name is a claim the design does not support.

⚠ EFFICIENCY MAY NOT BE CLAIMED FROM THIS IMPLEMENTATION. 13.5.4's last
paragraph: a first version that uses dense padding/masking still runs the
K=5-length matmuls for every head, so it may report behaviour and theoretical
sparse FLOPs and NOT latency or compute savings. This is exactly that kind --
the mask is additive over columns that are still computed. And the efficiency
baseline, when one is measured, is the full-K5 PREFIX CACHE, not a per-query
recompute, because standard inference may cache the fixed prefix too.

⚠ 13.5.1 also puts this section AFTER the main line, whose gate returned STOP.
Running it is a decision for the PI and belongs in section 14 before any
optional test forward; this file runs on VALIDATION, which needs no such
decision.

Run:
    sbatch script/lsu1.sh python tools/run_k0_receiver.py \\
        --carrier-bundle results/carriers_full_validation.json \\
        --query-manifest results/prereg_method_A_query_manifest.json \\
        --label-space results/label_space_llama31.json \\
        --calibration-dir data/method_a/llama31 \\
        --out results/k0_receiver_L31.npz
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import carrier_bundle as CB  # noqa: E402
from tools.baselines.common import load_candidate_space  # noqa: E402
from tools.baselines.forward import build_prefixes  # noqa: E402
from tools.k0_decomposition import prefix_cuts  # noqa: E402
from tools.k0_memory import mask_pad_for, visibility_mask  # noqa: E402
from tools.icl_common import run_provenance  # noqa: E402
from tools.model_args import add_model_arguments, load_model_from_args  # noqa: E402
from tools.label_space import file_sha256  # noqa: E402
from tools.prereg_config import REGISTERED_SEEDS  # noqa: E402
from tools.receiver_append import (load_for_append, merged_meta,  # noqa: E402
                                   plan_append, prefill_out, save_receiver_npz)
from tools.vector_arms import (alpha_zero_faults, fv_arm, fv_arms,  # noqa: E402
                               i2cl_arm, i2cl_module_hooks, icv_arm,
                               icv_arms, icv_hooks, install_vector_arm,
                               load_fv_sidecar, load_i2cl_sidecar,
                               load_icv_sidecar, load_tv_sidecar,
                               parse_float_list, parse_layer_list,
                               parse_vector_arm, tv_arm, tv_arms, tv_family)

# 13.5.4's names, verbatim. "K5-natural" was my own and it hid a role: the
# all-heads-read-the-cache arm is a CACHE EQUIVALENCE GATE, not a method arm,
# and the thing it is gated against is the monolithic forward with no cache at
# all. Two different objects, and 13.5.4 names them apart.
ARM_MONO = "full K5 monolithic"          # ordinary text ICL, no cache
ARM_ALL = "all-head cached K5"           # gate, not a method arm
ARM_SEL = "selective TL K5 memory"       # 13.5.2's main arm
ARM_BASE = "K0-offset natural"           # baseline / empty-memory placebo
ARMS = (ARM_MONO, ARM_ALL, ARM_SEL, ARM_BASE)

# EXPLORATORY layer-prefix family (RESULTS 45.4b). `layer<=L` lets EVERY head
# in layers 0..L read the demo columns and blocks every layer above. It varies
# DEPTH with head count held at "all", which is what the head-count sweep
# cannot do: the ranking's earliest carrier drops from layer 22 at top-8 to 16
# at top-16 and 6 at top-128, so "more heads" and "earlier layers" move
# together there (results/carrier_ranking_profile.json).
#
# THE CURVE IS SELF-ANCHORED. L >= n_layers-1 is ARM_ALL by construction and
# L < 0 is ARM_BASE, both already measured -- so the two ends need no new
# reference and a run that disagrees with them at the ends is wrong.
# 13.5.4's `TSLA-TL-zero-demo`: the K=0 receiver sees NO demo columns and
# instead gets ONE frozen TL residual vector, injected at the answer row of
# layer 16 (13.2.4's fixed edit layer) scaled by alpha. The mask is exactly
# ARM_BASE's -- the vector is the only offline information the arm has, which
# is what the 13.5.4 table means by "一个冻结 TL residual vector".
#
# 13.5.4(4) MAKES alpha=0 A GATE: it must reproduce `K0-offset natural`
# ELEMENTWISE. That is attainable and is not a tolerance question -- adding
# exactly 0.0 * v changes no bit of a finite float -- so the arm is run at
# alpha=0 and compared bitwise before any other alpha is read.
TSLA_ARM_PREFIX = "TSLA-TL-zero-demo a="
TSLA_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)      # the registry's configs, 13.2.4

LAYER_PREFIX_PREFIX = "layer<="


def layer_prefix_arm(L):
    return f"{LAYER_PREFIX_PREFIX}{int(L)}"


# WHERE THE VECTORS LIVE IN THE SIDECAR, read off the producer rather than
# guessed. run_tsla_tl_on_icl calls `acc.note("tsla_vectors", vectors)`;
# BaselineOutput.note stores into self._extra; finish() writes that under the
# key "runner". So the path is doc["runner"]["tsla_vectors"], and it is NOT
# "notes" -- which is what I looked for, and is why a correct artifact was
# refused. The older spellings stay accepted because refusing a file that
# does carry the vectors is the failure being fixed, not a purity to defend.
TSLA_NOTE_PATHS = (("runner", "tsla_vectors"),      # the producer's layout
                   ("notes", "tsla_vectors"),       # never written; kept
                   ("tsla_vectors",))               # ...so is a bare top level


def note_paths_for(key):
    """The sidecar paths a class's vectors may live at. TL keeps the three
    historical spellings; another class has its own key under `runner`."""
    if key == "tsla_vectors":
        return TSLA_NOTE_PATHS
    return (("runner", key), (key,))


def find_tsla_vectors(doc, key="tsla_vectors"):
    """The per-seed vector map, or None with the paths that were tried."""
    for path in note_paths_for(key):
        cur = doc
        for key in path:
            cur = cur.get(key) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, dict) and cur:
            return cur
    return None


def load_tsla_vectors(path, seeds, cls="tl"):
    """{seed: vector} of one head class from a run_tsla_tl_on_icl sidecar.

    `cls` is tl (the original `tsla_vectors` key), tr or random -- the two
    the runner has written since 2026-09-09 under `tsla_vectors_<cls>`. A
    sidecar made before that carries TL only, and asking it for TR is
    refused by name rather than answered with TL.

    Refuses by name at every step: 3.6 wants the ACTUAL keys in the message,
    because the caller cannot see the file and neither can I.
    """
    from tools.baselines.tsla_hook import sidecar_key
    key = sidecar_key(cls)
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"--tsla-vectors {path}: not a file. The artifact is written by\n"
            "    sbatch script/lsu1.sh python "
            "tools/baselines/run_tsla_tl_on_icl.py \\\n"
            "        --mode validation ... --out results/baselines\n"
            "and is named results/baselines/"
            "baseline_tsla_tl_on_icl_validation.json -- note the `baseline_` "
            "prefix, which `--out` does NOT control (--out is the DIRECTORY).")
    doc = json.loads(p.read_text(encoding="utf-8"))
    vecs = find_tsla_vectors(doc, key)
    if vecs is None:
        tried = " or ".join(".".join(t) for t in note_paths_for(key))
        raise SystemExit(
            f"{path}: no {cls} steering vectors at {tried}. Top-level keys: "
            f"{sorted(doc)}; under 'runner': "
            f"{sorted(doc.get('runner') or {})}. If 'runner' is empty this is "
            "not a TSLA sidecar; if it holds only 'tsla_vectors' it predates "
            "the TR/random classes -- re-run run_tsla_tl_on_icl "
            "--discovery-only, which writes all three.")
    out = {}
    for s_ in seeds:
        cell = vecs.get(str(s_), vecs.get(s_))
        if cell is None:
            raise SystemExit(
                f"{path}: no entry for seed {s_}. Seeds present: "
                f"{sorted(vecs)}.")
        if not isinstance(cell, dict) or "v" not in cell:
            keys = sorted(cell) if isinstance(cell, dict) else type(cell).__name__
            raise SystemExit(
                f"{path}: seed {s_} has no `v`, only {keys}. Runs made before "
                "2026-09-08 recorded `norm` alone, and a norm cannot "
                "reproduce the injection -- that artifact must be re-made.")
        out[s_] = np.asarray(cell["v"], dtype=np.float64)
    return out


TSLA_ARM_RE = re.compile(r"^TSLA-(?P<family>\S+) a=(?P<alpha>[-+0-9.eE]+)$")


def tsla_arm(alpha, family="TL-zero-demo"):
    """`TSLA-<family> a=<alpha>`; the default family is 13.5.4's zero-demo arm."""
    return f"TSLA-{family} a={float(alpha):g}"


def parse_tsla_family(arm):
    """(family, alpha) for any TSLA steering arm, or None if it is not one.

    One grammar for every TSLA arm this project runs: 13.5.4's
    `TSLA-TL-zero-demo a=0.25` (a K=5 vector into a K=0 receiver) and
    run_k10_increment's `TSLA-K10 a=...` / `TSLA-K10inc a=...` (a K=10 vector
    into the K=5 receiver). The reader groups arms by family and draws one
    alpha curve per family; the mask builders ask only "is a TSLA arm",
    because every family runs on its file's baseline mask.
    """
    m = TSLA_ARM_RE.match(str(arm))
    if not m:
        return None
    return m.group("family"), float(m.group("alpha"))


def parse_tsla_alpha(arm):
    """alpha for any TSLA steering arm, or None if it is not one."""
    fa = parse_tsla_family(arm)
    return None if fa is None else fa[1]


def parse_layer_prefix_spec(spec):
    """The L values from a --layer-prefix string. Never raises on a name.

    `none` / `base` MEAN -1, and they exist because argparse refuses a value
    beginning with "-" unless it matches its negative-number regex: "-1" alone
    is accepted, "-1,4,8" is read as an option and the run dies with "expected
    one argument". `--layer-prefix=-1,4,8` works, but a flag whose correct use
    depends on remembering that is a flag people get wrong once each.
    """
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in ("none", "base", "-"):
            out.append(-1)
            continue
        try:
            out.append(int(tok))
        except ValueError:
            raise SystemExit(
                f"--layer-prefix: {tok!r} is neither an integer nor "
                "'none'/'base' (which mean 'no layer sees the demos', the "
                "K0-offset end of the curve)")
    return out


def parse_layer_prefix(arm):
    """L for a layer-prefix arm, or None if it is not one."""
    if not str(arm).startswith(LAYER_PREFIX_PREFIX):
        return None
    return int(str(arm)[len(LAYER_PREFIX_PREFIX):])


def heads_by_layer(bundle, seed, top_n=None):
    """{layer: [query heads]} for this seed's carriers.

    The cut itself lives in `carrier_bundle.heads_for` -- one implementation
    for both runners. `top_n=None` is the frozen set 13.5.2 names; anything
    else is an EXPLORATORY slice of the same ranking and the caller has to
    say so.
    """
    out = {}
    for l, h in CB.heads_for(bundle, seed, top_n):
        out.setdefault(int(l), []).append(int(h))
    return {l: sorted(v) for l, v in sorted(out.items())}


def masks_for(arm, n_layers, n_heads, n_query, n_common, n_demo, n_live,
              carriers):
    """One additive mask per layer. The ONLY thing that differs between arms.

    THE WIDTH IS n_common + n_demo + n_live, which is the model's own
    cache_len + q_len. It was n_demo + n_live, as if the cache began at D_s;
    it begins at C_s, and the smoke run died on the mismatch rather than
    running with a misaligned mask, which is the better of the two failures.

    THIS WIDTH IS THE KEYS', AND THE MODEL'S 4D MASK CAN BE ONE WIDER: with a
    DynamicCache and no 2D attention_mask, transformers builds it at past + q
    + 1 and eager slices it back. `mask_pad_for` rules on the gap at the hook
    rather than widening anything here, so this stays the true column count.
    """
    out = {}
    L = parse_layer_prefix(arm)
    for l in range(n_layers):
        if arm == ARM_ALL:
            allowed = list(range(n_heads))
        elif arm == ARM_SEL:
            allowed = carriers.get(l, [])
        elif L is not None:
            # every head up to L, nobody above -- depth varied, head count
            # held at "all"
            allowed = list(range(n_heads)) if l <= L else []
        elif parse_tsla_alpha(arm) is not None:
            # 13.5.4: the TSLA arm's only offline information is the frozen
            # vector, so the demo columns are closed to everyone exactly as in
            # ARM_BASE. Named rather than left to fall through the else, so
            # that adding a future arm cannot silently inherit this mask.
            allowed = []
        elif parse_vector_arm(arm) is not None:
            # FV / TV (tools/vector_arms): the same kind of arm as TSLA --
            # one offline vector, the demo columns closed to every head.
            allowed = []
        else:                                   # ARM_BASE
            allowed = []
        if allowed == list(range(n_heads)):
            continue                            # nothing to mask at this layer
        # `_closed` is visibility_mask with nobody allowed, so it is not a
        # second implementation: allowed=[] would trip its own guard, and
        # writing the zeros here keeps C_s open by construction.
        out[l] = (visibility_mask(n_heads, n_query, n_common, n_demo, n_live,
                                  allowed) if allowed
                  else _closed(n_heads, n_query, n_common, n_demo, n_live))
    return out


def _closed(n_heads, n_query, n_common, n_demo, n_live):
    """Every head blocked from D_s, C_s and the live columns left open."""
    m = np.zeros((1, n_heads, n_query, n_common + n_demo + n_live),
                 dtype=np.float64)
    m[0, :, :, n_common:n_common + n_demo] = -np.inf
    return m


def bf16_ulp(x):
    """The spacing of bf16 values at magnitude |x|. 0 where x is 0.

    bf16 keeps 7 stored mantissa bits, so a value in [2^e, 2^(e+1)) sits on a
    grid of step 2^(e-7): 0.0078125 near 1, 0.125 in [16, 32), 0.25 in
    [32, 64). This is the resolution the model's own logits are quantised to,
    and it is what a difference between two float paths has to be measured
    against -- a "1 % of the range" tolerance means nothing if 1 % of the
    range is smaller than one grid step.
    """
    a = np.abs(np.asarray(x, dtype=np.float64))
    out = np.zeros_like(a)
    nz = a > 0
    out[nz] = np.exp2(np.floor(np.log2(a[nz])) - 7)
    return out


def cache_equivalence_report(cached, mono):
    """Everything needed to tell ROUNDING from a broken cache. Never raises.

    THE MAX ALONE CANNOT TELL THEM APART (working rules 2.2 rule 10(3): on bf16
    the max is a weak discriminator, and three earlier comparisons all landed
    on exactly 1.250e-01 = a power of two). What separates the two worlds:

      * a max error of ONE ulp, with most candidates identical and the rest
        differing by a grid step, is two reduction orders on the same
        arithmetic -- unavoidable, since the cached path chunks the sequence
        3018 + q and the monolithic one does it in a single pass;
      * a max error of many ulps, or a large mean, or a moved argmax, is the
        cache reading something else.

    So this reports the error IN ULPS at the site where it is largest, the
    ulp histogram over all candidates, the mean, and the logit magnitudes the
    ulp depends on -- the quantity the failing run never printed and without
    which neither reading can be ruled out.
    """
    c = np.asarray(cached, dtype=np.float64)
    m = np.asarray(mono, dtype=np.float64)
    err = np.abs(c - m)
    i = int(np.argmax(err))
    ulp = bf16_ulp(m)
    # where a logit is 0 the local ulp is 0; count those separately rather
    # than dividing by it
    with np.errstate(divide="ignore", invalid="ignore"):
        in_ulps = np.where(ulp > 0, err / np.where(ulp > 0, ulp, 1.0), np.nan)
    finite = in_ulps[np.isfinite(in_ulps)]
    hist = {}
    for name, sel in (("0", finite < 0.5), ("1", (finite >= 0.5) & (finite < 1.5)),
                      ("2", (finite >= 1.5) & (finite < 2.5)),
                      (">2", finite >= 2.5)):
        hist[name] = int(sel.sum())
    rng = float(m.max() - m.min())
    return {
        "n_candidates": int(m.size),
        "argmax_agrees": bool(int(np.argmax(c)) == int(np.argmax(m))),
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "n_differing": int((err > 0).sum()),
        "inter_class_range": rng,
        "frac_of_range": float(err.max() / rng) if rng > 0 else float("nan"),
        "worst_site": {"index": i,
                       "monolithic_logit": float(m[i]),
                       "cached_logit": float(c[i]),
                       "bf16_ulp_here": float(ulp[i]),
                       "err_in_ulps": (float(err[i] / ulp[i])
                                       if ulp[i] > 0 else float("nan"))},
        "ulp_histogram": hist,
        "logit_abs": {"min": float(np.abs(m).min()),
                      "median": float(np.median(np.abs(m))),
                      "max": float(np.abs(m).max())},
        "tolerance_in_ulps_at_worst_site": (
            float(0.01 * rng / ulp[i]) if ulp[i] > 0 else float("nan")),
    }


def argmax_swap_gap(cached, mono):
    """(gap, i_cached, i_mono): how far apart the two argmax candidates are.

    Measured on the MONOLITHIC arm, which is the reference. The gap is what
    says whether an argmax flip was avoidable: two candidates within the
    numeric error of each other cannot be ordered reliably by either reduction
    order, so which one comes top is not a property of the cache.
    """
    c = np.asarray(cached, dtype=np.float64)
    m = np.asarray(mono, dtype=np.float64)
    ic, im = int(np.argmax(c)), int(np.argmax(m))
    return float(abs(m[ic] - m[im])), ic, im


MAX_ULP = 1.0            # see the derivation in cache_equivalence_faults


def ulp_agreement_holds(cached, mono, max_ulp=MAX_ULP):
    """Is every candidate within `max_ulp` grid steps? The precondition.

    The near-tie exemption below only makes sense while this is true. If the
    cache is off by five nats somewhere, an argmax flip is a CONSEQUENCE of
    that, not two reduction orders failing to order a tie -- and excusing it
    would let a broken cache pick the friendlier of the two explanations.
    """
    c = np.asarray(cached, dtype=np.float64)
    m = np.asarray(mono, dtype=np.float64)
    ulp = bf16_ulp(m)
    err = np.abs(c - m)
    return not np.any(np.where(ulp > 0, err > max_ulp * ulp, err > 0))


def argmax_tie_bound(mono, ic, im, max_ulp=MAX_ULP):
    """How close two candidates must be for the order to be undetermined.

    THE BOUND IS THE INDEPENDENTLY JUSTIFIED ONE, not the error observed on
    this row. Using the observed error would let a broken cache excuse itself:
    the worse it is, the wider the window in which an argmax flip counts as a
    tie -- working rules 11's "a tolerance may not equal the deviation it is
    explaining away", which is exactly the mistake this line replaced.

    Two reduction orders agree to within `max_ulp` grid steps; taking the
    coarser of the two candidates' grids is the widest that bound can be.
    """
    m = np.asarray(mono, dtype=np.float64)
    return float(max_ulp) * float(max(bf16_ulp(np.array([m[ic]]))[0],
                                      bf16_ulp(np.array([m[im]]))[0]))


def argmax_swap_note(cached, mono, max_ulp=MAX_ULP):
    """A line about an argmax flip that the arithmetic does not determine.

    Returns None when the argmaxes agree, or when they disagree by more than
    the measured error -- that case is a FAULT and belongs to
    `cache_equivalence_faults`, not here. Separate channels on purpose: a
    "not fatal" string returned in the fault list would still stop the run.
    """
    c = np.asarray(cached, dtype=np.float64)
    m = np.asarray(mono, dtype=np.float64)
    if int(np.argmax(c)) == int(np.argmax(m)):
        return None
    if not ulp_agreement_holds(c, m, max_ulp):
        return None          # a fault, not a tie; see ulp_agreement_holds
    gap, ic, im = argmax_swap_gap(c, m)
    bound = argmax_tie_bound(m, ic, im, max_ulp)
    if gap > bound:
        return None
    return (f"NEAR-TIE argmax swap: cached {ic} vs monolithic {im}, and the "
            f"monolithic gap between them is {gap:.6f}, within the "
            f"{max_ulp:g}-ulp agreement two reduction orders can be asked for "
            f"({bound:.6f} at this magnitude). Neither order determines which "
            "of those two is top, so this row cannot carry the argmax half of "
            "the gate. The ulp half is what decides it.")


def format_report(rep):
    """The report as printable lines. Kept apart so the dump and the print
    cannot disagree about what was measured."""
    w = rep["worst_site"]
    return [
        f"candidates {rep['n_candidates']}, differing {rep['n_differing']}, "
        f"argmax agrees: {rep['argmax_agrees']}",
        f"max |diff| {rep['max_abs_err']:.4e}   mean |diff| "
        f"{rep['mean_abs_err']:.4e}   inter-class range "
        f"{rep['inter_class_range']:.4f}",
        f"worst site: monolithic {w['monolithic_logit']:+.4f} vs cached "
        f"{w['cached_logit']:+.4f}",
        f"  bf16 ulp there {w['bf16_ulp_here']:.6f}  ->  the error is "
        f"{w['err_in_ulps']:.2f} ULP",
        f"  and 1 % of the range is {rep['tolerance_in_ulps_at_worst_site']:.2f}"
        " ULP -- under 1.0 means the criterion demands BITWISE equality",
        f"ulp histogram over candidates: 0 ulp {rep['ulp_histogram']['0']}, "
        f"1 ulp {rep['ulp_histogram']['1']}, 2 ulp {rep['ulp_histogram']['2']}, "
        f">2 ulp {rep['ulp_histogram']['>2']}",
        f"|logit| min {rep['logit_abs']['min']:.3f} median "
        f"{rep['logit_abs']['median']:.3f} max {rep['logit_abs']['max']:.3f}",
    ]




def cache_equivalence_faults(cached, mono, max_ulp=MAX_ULP, tol_frac=0.01):
    """13.5.4(2): the cached path must reproduce the monolithic one.

    Both are RAW candidate logits for one query. Two conditions, and the
    section is blocked unless both hold:

      * the candidate argmax is identical -- categorical, needs no tolerance;
      * every candidate is within `max_ulp` of the monolithic value AT ITS OWN
        MAGNITUDE, i.e. the two paths landed on the same or adjacent
        representable bf16 numbers.

    ⚠ THE REGISTERED TOLERANCE WAS UNATTAINABLE, and this is the re-derivation
    (deviation recorded in prereg 14.0b-22). 13.5.4(2) wrote "under 1 % of
    that query's inter-class logit range". Measured on the first validation
    query of seed 42: the range is 3.7500, so the tolerance is 0.0375 -- while
    the candidate logits sit at |logit| 18.6 to 22.4, where the bf16 grid step
    is 0.125. The registered tolerance is therefore **0.30 of ONE GRID STEP**,
    and a tolerance below one step admits only bitwise-equal values. The two
    paths chunk the sequence differently (prefill 3018 then 20, versus 3038 in
    one pass), so their attention matmuls have different shapes and different
    float reduction orders; bitwise equality is not obtainable and demanding
    it is working rules 11b's dead criterion, instance 1, exactly.

    WHY ULPS AND WHY 1. The quantity that decides whether a difference is
    real is the resolution the model computes in, not the spread of one
    query's classes -- which is what the old denominator was reaching for and
    could not deliver, since it has no relation to the arithmetic. One ulp
    means "the same or the next representable number", which is the STRONGEST
    agreement two different reduction orders can be asked for. It is
    attainable: the same measurement gives 27 candidates at 0 ulp, 9 at
    exactly 1 ulp, none above. And it discriminates: a cache reading the wrong
    columns moves values by many ulps (the world-B fixture in
    test_k0_cache_gate gives 8+), not by one.

    ⚠ FALSIFICATION, and what NOT to do about it. A legitimate run could in
    principle exceed 1 ulp -- a longer prefix means a longer reduction. If one
    ever does, that is a finding to investigate and record, NOT a threshold to
    raise. Raising it after seeing it fail is precisely the tolerance-fitted-
    to-the-deviation error working rules 11 was written for.

    The old fraction-of-range figure is still REPORTED by
    `cache_equivalence_report`, so the registered number stays visible next to
    the one that replaced it.
    """
    c = np.asarray(cached, dtype=np.float64)
    m = np.asarray(mono, dtype=np.float64)
    if c.shape != m.shape:
        return [f"cached {c.shape} and monolithic {m.shape} differ in shape"]
    bad = []
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(m))):
        bad.append("candidate logits contain non-finite values")
        return bad
    ulp = bf16_ulp(m)
    err = np.abs(c - m)
    # THE ARGMAX CHECK, AND THE QUESTION IT HAS TO ASK FIRST (working rules 11b).
    # Two reduction orders agree to within one grid step -- the bound enforced
    # below, which the measurement keeps meeting. If the top TWO candidates
    # are themselves within that much of each other, a correct cache can put
    # either on top, and demanding a particular one demands something the
    # arithmetic does not determine. So the flip is fatal only when the two
    # swapped candidates are FARTHER apart than the error measured on this
    # row -- which a cache reading the wrong columns always is, because it
    # moves values by many ulps (the world-B fixture gives 8+).
    #
    # ⚠ NOT a tolerance fitted to a deviation (working rules 11). The bound is the
    # same 1-ulp agreement justified independently above; what is new is
    # asking whether the compared quantity can resolve the difference at all.
    # The non-fatal case is reported by `argmax_swap_note`, in its own
    # channel, because a "not fatal" line inside this list would still stop
    # the run.
    if int(np.argmax(c)) != int(np.argmax(m)):
        gap, ic, im = argmax_swap_gap(c, m)
        bound = argmax_tie_bound(m, ic, im, max_ulp)
        # TWO conditions, not one. The candidates must be too close to order
        # AND the numbers must otherwise agree. A cache off by five nats can
        # also happen to swap two near-tied candidates, and excusing that
        # would let it choose the friendlier explanation of its own failure.
        wide = gap > bound
        agrees = ulp_agreement_holds(c, m, max_ulp)
        if wide or not agrees:
            # WHICH of the two conditions fired, because they mean different
            # things. Sharing one sentence for both produced a message that
            # said a gap of 0.100 was "LARGER than 0.125".
            why = (f"the monolithic gap between them is {gap:.6f}, larger "
                   f"than the {max_ulp:g}-ulp agreement two reduction orders "
                   f"can be asked for ({bound:.6f} at this magnitude), and a "
                   "correct cache cannot reorder two candidates that far "
                   "apart" if wide else
                   f"the gap between them ({gap:.6f}) is inside the "
                   f"{max_ulp:g}-ulp window, but the per-candidate agreement "
                   "does NOT hold (see the ulp fault below), so this flip is "
                   "a consequence of that disagreement and not two reduction "
                   "orders failing to order a tie")
            bad.append(f"candidate argmax differs: cached {ic} vs monolithic "
                       f"{im}. {why}. The cache is not reproducing the prompt")
    # where the monolithic logit is exactly 0 the local ulp is 0 and no
    # multiple of it is defined; such an entry must simply be equal.
    over = np.flatnonzero(np.where(ulp > 0, err > max_ulp * ulp, err > 0))
    if over.size:
        i = int(over[np.argmax(err[over] / np.where(ulp[over] > 0,
                                                    ulp[over], 1.0))])
        n_ulp = err[i] / ulp[i] if ulp[i] > 0 else float("inf")
        rng = float(m.max() - m.min())
        bad.append(
            f"{over.size} of {m.size} candidates differ by more than "
            f"{max_ulp:g} ulp; the worst is candidate {i}, monolithic "
            f"{m[i]:+.4f} vs cached {c[i]:+.4f} = {n_ulp:.2f} ulp at a grid "
            f"step of {ulp[i]:.6f}"
            + (f" (and {err.max() / rng:.3%} of the inter-class range "
               f"{rng:.4f}, the figure 13.5.4(2) registered)" if rng > 0 else "")
            + ". Every other arm rides the same cache, so none of their "
              "numbers can be read")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--carrier-bundle", required=True)
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--label-space", required=True)
    ap.add_argument("--calibration-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true",
                    help="if --out exists and is this run's file (same model, kernel, "
                         "dtype, task, K, head cut, split, rows, bundle), read only the "
                         "arms it lacks and write the union; the stored arms are kept "
                         "bit for bit (tools/receiver_append)")
    ap.add_argument("--cache-gate-ulp", type=float, default=MAX_ULP,
                    help="13.5.4(2) cache-equivalence bound in bf16 ulp: 1 by derivation "
                         "(cache_equivalence_faults). Above 1 ONLY with --cache-gate-source "
                         "naming the independent measurement it is the worst ulp of "
                         "(tools/check_two_path_noise on this cell's largest K, this dtype): "
                         "the construction is held to the model's own precision, never to a "
                         "value a failure suggested (prereg 14.0b-24)")
    ap.add_argument("--cache-gate-source", default="",
                    help="where a --cache-gate-ulp above 1 was measured; written to the meta")
    from tools.split_rows import add_split_arguments
    add_split_arguments(ap)
    add_model_arguments(ap)
    ap.add_argument("--task", default="trec_fine_per_class")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--tsla-classes", default="tl",
                    help="comma list of head classes to steer with, from "
                         "tl,tr,random (tsla_hook.HEAD_CLASSES); one family "
                         "`TSLA-<CLASS>-zero-demo` per class, TL keeping "
                         "13.5.4's spelling")
    ap.add_argument("--tsla-vectors",
                    help="13.5.4's `TSLA-TL-zero-demo`: the JSON written "
                         "beside run_tsla_tl_on_icl's npz, whose "
                         "notes.tsla_vectors carries one steering vector per "
                         "seed. Adds one arm per registered alpha "
                         f"{list(TSLA_ALPHAS)}, all on ARM_BASE's mask. "
                         "13.5.4(4)'s alpha=0 elementwise gate runs first")
    ap.add_argument("--fv-vectors", default=None,
                    help="run_fv_increment's sidecar built at --K-base 0 for "
                         "this --K: adds the `FV-K<K> a=<alpha>` arms at "
                         f"alphas {list(TSLA_ALPHAS)} -- alpha * v_FV at the "
                         "answer row of the sidecar's layer, on ARM_BASE's "
                         "mask, alpha=0 gated bitwise (Todd et al.; alpha = 1 "
                         "the main arm)")
    ap.add_argument("--tv-vectors", default=None,
                    help="run_tv_increment's sidecar built at --K-base 0 for "
                         "this --K: adds `TV-K<K> L=<layer> a=1` -- the answer "
                         "row of decoder layer <layer>'s output REPLACED by "
                         "theta -- for every candidate layer (or --tv-layers), "
                         "on ARM_BASE's mask, plus one a=0 gate (Hendel et al.)")
    ap.add_argument("--tv-layers", default="",
                    help="comma list restricting the TV arms to these candidate "
                         "layers (the test read passes the one "
                         "tools/select_tv_layer.py chose on validation)")
    ap.add_argument("--tv-m5", action="store_true",
                    help="also run the descriptive `TV-K<K>m5` family (theta = "
                         "the mean over five dummy queries); validation only")
    ap.add_argument("--icv-vectors", default=None,
                    help="run_icv_increment's sidecar built at --K-base 0 for "
                         "this --K: adds `ICV-K<K> a=<lambda>` -- the upstream "
                         "ICVLayer on every layer's MLP output, every position, "
                         "the prefix cache rebuilt per lambda with the hooks -- "
                         "for the sidecar's grid (or --icv-lambdas), on "
                         "ARM_BASE's mask, plus the a=0 gate (Liu et al.)")
    ap.add_argument("--icv-lambdas", default="",
                    help="comma list restricting the ICV arms to these grid "
                         "lambdas (the test read passes the one "
                         "tools/select_icv_lambda.py chose on validation)")
    ap.add_argument("--i2cl-vectors", default=None,
                    help="run_i2cl_increment's npz built at --K-base 0 for this "
                         "--K: adds `I2CL-K<K> a=1` -- the calibrated linear "
                         "injection on every layer's self_attn and mlp "
                         "outputs, every position, the prefix cache rebuilt "
                         "with the hooks once per seed -- on ARM_BASE's mask, "
                         "plus the a=0 gate (Li et al.)")
    ap.add_argument("--layer-prefix", default="",
                    help="EXPLORATORY depth sweep (RESULTS 45.4b): comma-"
                         "separated L values. Every head in layers 0..L reads "
                         "the demo columns, every layer above is blocked. Its "
                         "ends reproduce the all-head and K0-offset arms by "
                         "construction, so the curve needs no outside "
                         "reference. Use this rather than --top-n: the "
                         "ranking's earliest carrier drops from layer 22 at "
                         "top-8 to 6 at top-128, so a head-count sweep is "
                         "confounded with a depth sweep. Write 'none' "
                         "for the no-layer end -- argparse rejects a bare "
                         "leading '-1,...' unless you use --layer-prefix=...")
    ap.add_argument("--top-n", type=int, default=None,
                    help="EXPLORATORY head-count sweep: use the first N of "
                         "the bundle's own ranking instead of the frozen "
                         "top-8 that 13.5.2 names. ⚠ adding heads probably "
                         "also adds EARLIER layers, so a head-count sweep and "
                         "a layer-depth sweep are confounded on this data -- "
                         "run tools/describe_carrier_ranking.py first")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug only; never a registered run")
    args = ap.parse_args(argv)
    if args.cache_gate_ulp > MAX_ULP and not args.cache_gate_source:
        raise SystemExit(f"--cache-gate-ulp {args.cache_gate_ulp:g} is above the derived "
                         f"{MAX_ULP:g}: name the measurement it comes from with "
                         "--cache-gate-source (tools/check_two_path_noise on this "
                         "cell's largest K, in this dtype)")

    bundle = json.loads(Path(args.carrier_bundle).read_text(encoding="utf-8"))
    if str(bundle.get("scope")) != CB.SCOPE_FULL:
        raise SystemExit(
            f"{args.carrier_bundle} is a {bundle.get('scope')!r} bundle. "
            "13.5.2 names section 2.4's frozen top-8, which is the "
            "full-validation set; the fold sets are the gate's and each was "
            "chosen on half the rows.")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ls = load_candidate_space(args.label_space, model=args.model,
                              query_manifest=args.query_manifest,
                              tokenizer=tok)
    from tools.split_rows import rows_for, split_meta
    rows_by_seed = {s: rows_for(args.query_manifest, args.split, s,
                                freeze_manifest=args.freeze_manifest,
                                carriers=args.carrier_bundle,
                                label_space=args.label_space)
                    for s in REGISTERED_SEEDS}
    if args.split != "validation":
        print(f"  ⚠⚠ {args.split}: the ONE-SHOT test split, read under "
              "prereg 14.0b-23 (optional/descriptive)")

    print("=" * 78)
    print("K0 RECEIVER + SELECTIVE K5 LATENT MEMORY (13.5) -- VALIDATION")
    print("=" * 78)
    print("  arms differ ONLY by which heads may see the demo columns:")
    for a in ARMS:
        print(f"    {a}")
    print("  same weights, same cache, same positions, same softmax denominator")
    if args.top_n is not None:
        print(f"\n  ⚠⚠ EXPLORATORY: --top-n {args.top_n} is NOT 13.5.2's arm, "
              "which names section 2.4(2)'s")
        print("  ⚠⚠ frozen top-8. This slices the bundle's own ranking to a "
              "different cut. Adding")
        print("  ⚠⚠ heads probably also adds EARLIER layers, so this sweep is "
              "confounded with a")
        print("  ⚠⚠ layer-depth sweep -- see the layer profile before reading "
              "any trend.")

    prefixes, text_of, checks, render_prompt = build_prefixes(
        args.task, args.K, list(REGISTERED_SEEDS), args.query_manifest,
        args.calibration_dir, ls)
    for c in checks:
        print(f"  [PASS] {c}")

    model = load_model_from_args(args)
    import torch
    n_l = int(model.config.num_hidden_layers)
    n_h = int(model.config.num_attention_heads)
    cand = list(ls.candidate_token_ids)

    Ls = parse_layer_prefix_spec(args.layer_prefix)
    # ONE FAMILY PER HEAD CLASS. TL keeps 13.5.4's spelling; TR and random
    # are the upstream's other two vectors (tsla_hook.HEAD_CLASSES).
    tsla_v, tsla_fam = {}, {}
    if args.tsla_vectors:
        from tools.baselines.tsla_hook import HEAD_CLASSES
        classes = [c.strip() for c in args.tsla_classes.split(",") if c.strip()]
        bad_c = [c for c in classes if c not in HEAD_CLASSES]
        if bad_c:
            raise SystemExit(f"--tsla-classes {bad_c}: expected from "
                             f"{HEAD_CLASSES}")
        for c in classes:
            tsla_v[c] = load_tsla_vectors(args.tsla_vectors, REGISTERED_SEEDS,
                                          c)
            tsla_fam[c] = {"tl": "TL-zero-demo", "tr": "TR-zero-demo",
                           "random": "RAND-zero-demo"}[c]
            print(f"\n  TSLA zero-demo [{c}] -> family {tsla_fam[c]!r}: "
                  f"{len(tsla_v[c])} steering vectors from "
                  f"{args.tsla_vectors}")
            print("  |v| per seed: "
                  + ", ".join(f"{k}={np.linalg.norm(v):.4f}"
                              for k, v in sorted(tsla_v[c].items())))
    cls_of_family = {fam: c for c, fam in tsla_fam.items()}
    fv_v = (load_fv_sidecar(args.fv_vectors, REGISTERED_SEEDS, k_base=0,
                            k_full=args.K)
            if args.fv_vectors else None)
    if fv_v:
        print(f"\n  FV [Todd et al.] -> family 'FV-K{args.K}': alpha * v_FV at "
              f"the answer row of the sidecar's layer, on {ARM_BASE!r}'s mask, "
              f"from {args.fv_vectors}")
        print("  layer / |v| per seed: "
              + ", ".join(f"{k}={fv_v[k][0]}/{np.linalg.norm(fv_v[k][1]):.4f}"
                          for k in sorted(fv_v)))
    tv, tv_layers = {}, []
    if args.tv_vectors:
        # `tv_cand`, not `cand`: `cand` is the label-token list above, and a
        # first draft shadowed it with the candidate LAYERS -- five "label
        # logits" at token ids 8..20, which the cache-equivalence gate then
        # refused (31 ulp). The guard below the arm list keeps that from
        # coming back.
        tv_cand, main_fam = load_tv_sidecar(args.tv_vectors, REGISTERED_SEEDS,
                                            k_base=0, k_full=args.K)
        want = parse_layer_list(args.tv_layers)
        bad_l = [l for l in want if l not in tv_cand]
        if bad_l:
            raise SystemExit(f"--tv-layers {bad_l}: not among the sidecar's "
                             f"candidate layers {tv_cand}")
        tv_layers = want or list(tv_cand)
        tv[tv_family(args.K)] = main_fam
        if args.tv_m5:
            tv[tv_family(args.K, "m5")] = load_tv_sidecar(
                args.tv_vectors, REGISTERED_SEEDS, k_base=0, k_full=args.K,
                key="tv_vectors_m5")[1]
        print(f"\n  TV [Hendel et al.] -> families {sorted(tv)}: the answer row "
              f"of decoder layer L's output replaced by theta_L, on "
              f"{ARM_BASE!r}'s mask; layers {tv_layers}; "
              f"{tv_arm(tv_layers[0], args.K, on=False)!r} is the gate; from "
              f"{args.tv_vectors}")
    icv_v, icv_lams = None, []
    if args.icv_vectors:
        grid, icv_v = load_icv_sidecar(args.icv_vectors, REGISTERED_SEEDS,
                                       k_base=0, k_full=args.K)
        want = parse_float_list(args.icv_lambdas)
        bad_lam = [l for l in want if l not in grid]
        if bad_lam:
            raise SystemExit(f"--icv-lambdas {bad_lam}: not in the sidecar's "
                             f"grid {grid}")
        icv_lams = [l for l in (want or grid) if l > 0]
        if not icv_lams:
            raise SystemExit("--icv-lambdas: no lambda above 0 (0 is the gate, "
                             "run always)")
        print(f"\n  ICV [Liu et al.] -> family 'ICV-K{args.K}': the upstream "
              "ICVLayer on every layer's MLP output, every position, on "
              f"{ARM_BASE!r}'s mask; lambdas {icv_lams}, one prefix cache per "
              f"lambda; {icv_arm(0.0, args.K)!r} is the gate; from "
              f"{args.icv_vectors}")
    run_arms = (list(ARMS) + [layer_prefix_arm(L) for L in Ls]
                + [tsla_arm(a, tsla_fam[c]) for c in tsla_v
                   for a in TSLA_ALPHAS]
                + (fv_arms(args.K, TSLA_ALPHAS) if fv_v else [])
                + (tv_arms(tv_layers, args.K,
                           families=[""] + (["m5"] if args.tv_m5 else []))
                   if tv else [])
                + (icv_arms(args.K, [0.0] + icv_lams) if icv_v is not None else []))
    i2cl_v = None
    if args.i2cl_vectors:
        _, i2cl_v = load_i2cl_sidecar(args.i2cl_vectors, REGISTERED_SEEDS,
                                      k_base=0, k_full=args.K)
        run_arms += [i2cl_arm(args.K, on=False), i2cl_arm(args.K)]
        print(f"\n  I2CL [Li et al.] -> family 'I2CL-K{args.K}': the calibrated "
              "linear injection on every layer's self_attn and mlp outputs, "
              f"every position, on {ARM_BASE!r}'s mask; one hooked prefix cache "
              f"per seed; {i2cl_arm(args.K, on=False)!r} is the gate; from "
              f"{args.i2cl_vectors}")
        for s_ in REGISTERED_SEEDS:
            c_ = i2cl_v[s_]["coef"]
            print(f"    seed {s_}: lambda mean {c_[:, :, 0].mean():.4f} "
                  f"[{c_[:, :, 0].min():.3f}, {c_[:, :, 0].max():.3f}], beta mean "
                  f"{c_[:, :, 1].mean():.4f}")
    # --append: an existing file of this run's identity is kept and only the
    # arms it lacks are read (tools/receiver_append); anything else is refused
    todo, all_arms, old_meta, old_arrays = list(run_arms), list(run_arms), None, None
    if args.append and Path(args.out).is_file():
        _ran = {s: (rows_by_seed[s][:args.limit] if args.limit else rows_by_seed[s])
                for s in REGISTERED_SEEDS}
        old_meta, old_arrays = load_for_append(args.out, {
            "model": args.model, "method": args.method, "dtype": args.dtype,
            "attn": args.attn, "task": args.task, "K": args.K,
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

    if list(cand) != list(ls.candidate_token_ids):
        raise SystemExit(f"internal: the candidate list ({len(cand)} entries) is no "
                         f"longer the label space ({len(ls.candidate_token_ids)}); "
                         "a local variable shadowed `cand`")
    if Ls:
        print(f"\n  ⚠ EXPLORATORY depth sweep: {len(Ls)} layer-prefix arms "
              f"L = {Ls}")
        print("  every head in layers 0..L reads the demo columns, every "
              "layer above is blocked.")
        print("  ⚠ THE ENDS ARE SELF-CHECKS, NOT RESULTS. An L at or above "
              "the last layer must")
        print("  ⚠ reproduce `all-head cached K5`, and a negative L must "
              "reproduce `K0-offset")
        print("  ⚠ natural`. Include both in the sweep and check them -- a "
              "curve whose ends do")
        print("  ⚠ not land on arms already measured is not a curve of "
              "anything.")
    out = {a: {} for a in all_arms}
    if old_arrays is not None:
        prefill_out(out, old_arrays, [a for a in all_arms if a not in todo],
                    list(REGISTERED_SEEDS))
    meta_seeds = {}
    # WHAT EACH SEED ACTUALLY RAN. `--limit` slices a local `rows`, and the
    # npz used to write the UNSLICED rows_by_seed alongside the sliced logit
    # stacks -- 4 rows of logits against 144 query ids, silently. The arrays
    # in one file have to describe the same queries.
    ran_by_seed = {}
    for s in REGISTERED_SEEDS:
        blocks = prefixes[s]
        carriers = heads_by_layer(bundle, s, args.top_n)
        common = ""                     # this renderer has no header before D
        through = "\n\n".join(blocks) + "\n\n"
        rows = rows_by_seed[s][:args.limit] if args.limit else rows_by_seed[s]
        ran_by_seed[s] = rows
        print(f"\n  seed {s}: carriers {carriers}")

        # ONE offline prefill of C_s || D_s per seed, reused by every query.
        n_c, n_d = None, None
        first = render_prompt(blocks, text_of[rows[0]["query_id"]])
        n_c, n_d = prefix_cuts(tok, first, [common, through]) if common else (
            len(tok("", add_special_tokens=True).input_ids),
            len(tok(through, add_special_tokens=True).input_ids))
        pre_ids = tok(through, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            base_cache = model(pre_ids, use_cache=True).past_key_values
        # ICV edits every position, so its prefix is a DIFFERENT prefix: one
        # cache per lambda, prefilled with the hooks installed
        icv_caches = {}
        for _lam in [l for l in icv_lams if icv_arm(l, args.K) in todo]:
            _h = icv_hooks(model, icv_v[s], _lam, torch)
            try:
                with torch.no_grad():
                    icv_caches[_lam] = model(pre_ids, use_cache=True).past_key_values
            finally:
                for _x in _h:
                    _x.remove()
        i2cl_cache = None
        if i2cl_v is not None and i2cl_arm(args.K) in todo:
            _h = i2cl_module_hooks(model, i2cl_v[s]["cv"], i2cl_v[s]["coef"], torch)
            try:
                with torch.no_grad():
                    i2cl_cache = model(pre_ids, use_cache=True).past_key_values
            finally:
                for _x in _h:
                    _x.remove()
        n_demo = int(pre_ids.shape[1]) - n_c
        # the visibility patterns, once per arm per seed (k0_memory
        # .mask_prefix_rows); the query loop expands them on the card
        rows_by_arm = {a: prepare_mask_rows(
                           masks_for(a, n_l, n_h, 1, n_c, n_demo, 1, carriers),
                           torch, model.device)
                       for a in todo if a != ARM_MONO}
        meta_seeds[str(s)] = {"n_common": int(n_c), "n_demo": int(n_demo),
                              "carriers": {str(k): v
                                           for k, v in carriers.items()}}
        print(f"    C_s {n_c} tokens, D_s {n_demo} tokens, "
              f"{len(rows)} queries x {len(todo)} arms")

        for qi, r in enumerate(rows):
            full = render_prompt(blocks, text_of[r["query_id"]])
            fid = tok(full, return_tensors="pt").input_ids.to(model.device)
            if not torch.equal(fid[0, :int(pre_ids.shape[1])], pre_ids[0]):
                raise SystemExit(
                    f"seed {s} query {r['query_id'][:12]}: the prompt's "
                    "tokenisation does not begin with C_s || D_s's own. The "
                    "boundary does not exist at the token level and rounding "
                    "it would move text between D_s and R_s(q).")
            suffix = fid[:, int(pre_ids.shape[1]):]
            n_live = int(suffix.shape[1])
            for arm in todo:
                if arm == ARM_MONO:
                    # NO CACHE AT ALL. This is the reference 13.5.4(2) gates
                    # the cached arm against, so it must not share the cached
                    # path -- comparing a path with itself proves nothing.
                    with torch.no_grad():
                        lg = model(fid).logits[0, -1]
                    out[arm].setdefault(s, []).append(
                        lg[cand].to(torch.float64).cpu().numpy())
                    continue
                handles = _install_rows(model, rows_by_arm[arm], n_live, torch)
                # THE INJECTION IS A SECOND HOOK ON A DIFFERENT MODULE: the
                # mask is a pre-hook on self_attn, this is a forward hook on
                # the decoder layer, so they compose rather than fight. And
                # `hs[:, -1, :]` is the answer position here because the
                # cached forward runs only the receiver's own tokens.
                _fa = parse_tsla_family(arm)
                if _fa is not None:
                    from tools.baselines.run_fv_on_icl import inject_hook
                    from tools.baselines.tsla_hook import edit_layer
                    handles.append(inject_hook(
                        model, edit_layer(n_l), _fa[1],
                        tsla_v[cls_of_family[_fa[0]]][s]))
                cache = base_cache
                _pv = parse_vector_arm(arm)
                if _pv is not None:
                    # FV / TV / ICV: the sidecar's operator; [] at a=0, so
                    # that arm IS the base arm's forward
                    from tools.baselines.run_fv_on_icl import inject_hook
                    handles += install_vector_arm(model, arm, s, fv_v, tv,
                                                  inject_hook, torch, icv=icv_v,
                                                  i2cl=i2cl_v)
                    # an ICV / I2CL arm continues ITS OWN prefix (hooked at
                    # every position); every other arm continues the natural
                    if _pv[3] != 0.0 and _pv[0] == "ICV":
                        cache = icv_caches[_pv[3]]
                    elif _pv[3] != 0.0 and _pv[0] == "I2CL":
                        cache = i2cl_cache
                try:
                    cache.crop(int(pre_ids.shape[1]))
                    with torch.no_grad():
                        lg = model(suffix, past_key_values=cache,
                                   use_cache=True).logits[0, -1]
                finally:
                    for h in handles:
                        h.remove()
                    cache.crop(int(pre_ids.shape[1]))
                # RAW candidate logits, not log-probs. 13.5.4(2)'s gate is
                # on "candidate logit error", and candidate_log_probs
                # subtracts each arm's OWN logsumexp, so a log-prob difference
                # is not a logit difference. working rules 3.9 also wants the raw
                # quantity on disk so a different readout needs no new GPU.
                out[arm].setdefault(s, []).append(
                    lg[cand].to(torch.float64).cpu().numpy())
            if qi == 0:
                # 13.5.4(2), ON ONE VALIDATION QUERY PER SEED as written. This
                # is a GATE, not an estimand: it says whether the cache
                # plumbing reproduces the prompt, and every other arm rides
                # that plumbing. It blocks rather than warns.
                rep = cache_equivalence_report(out[ARM_ALL][s][0],
                                               out[ARM_MONO][s][0])
                # 13.5.4(4): alpha = 0 must reproduce K0-offset natural
                # ELEMENTWISE. Adding exactly 0.0 * v changes no bit of a
                # finite float, so this is an identity and not a tolerance --
                # the same standing the sham arm has in 11 gate 3.
                for _fam in tsla_fam.values():
                    _z = tsla_arm(0.0, _fam)
                    if _z not in out or s not in out[_z]:
                        continue
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
                for _z in (([fv_arm(0.0, args.K)] if fv_v else [])
                           + ([tv_arm(tv_layers[0], args.K, on=False)]
                              if tv else [])
                           + ([icv_arm(0.0, args.K)]
                              if icv_v is not None else [])
                           + ([i2cl_arm(args.K, on=False)]
                              if i2cl_v is not None else [])):
                    _f = alpha_zero_faults(out[_z][s][0], out[ARM_BASE][s][0],
                                           _z, ARM_BASE)
                    if _f:
                        raise SystemExit(f"seed {s}: 13.5.4(4) FAILED -- {_f}")
                    print(f"    [PASS] 13.5.4(4) {_z!r} reproduces "
                          f"{ARM_BASE!r} elementwise")
                _note = argmax_swap_note(out[ARM_ALL][s][0],
                                         out[ARM_MONO][s][0],
                                         max_ulp=args.cache_gate_ulp)
                if _note:
                    print(f"    ⚠ {_note}")
                gbad = cache_equivalence_faults(out[ARM_ALL][s][0],
                                                out[ARM_MONO][s][0],
                                                max_ulp=args.cache_gate_ulp)
                print(f"    13.5.4(2) cache equivalence, seed {s}:")
                for line in format_report(rep):
                    print(f"      {line}")
                if gbad:
                    # THE RAW VECTORS GO TO DISK BEFORE RAISING. The gate had
                    # printed one number and died, so answering "is that one
                    # ulp or a broken cache" cost another GPU run. Both
                    # candidate logit vectors are a few hundred bytes;
                    # everything anyone can ask afterwards is offline
                    # (working rules 3.9).
                    dump = Path(str(args.out) + f".gate_failure_seed{s}.npz")
                    dump.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        dump,
                        cached=np.asarray(out[ARM_ALL][s][0]),
                        monolithic=np.asarray(out[ARM_MONO][s][0]),
                        candidate_token_ids=np.asarray(cand),
                        query_id=np.array([str(r["query_id"])]),
                        demo_seed=np.array([int(s)]),
                        report=json.dumps(rep))
                    raise SystemExit(
                        f"seed {s}: 13.5.4(2) cache-equivalence gate FAILED, "
                        "so this section is blocked:\n  - "
                        + "\n  - ".join(gbad)
                        + f"\n\n  the two candidate logit vectors are in "
                          f"{dump}; the diagnostic above says whether this is "
                          "the bf16 grid or the cache, and it needs no GPU to "
                          "re-read")
                print("    [PASS] 13.5.4(2) cache equivalence")
            if (qi + 1) % 40 == 0:
                print(f"    {qi + 1}/{len(rows)}", flush=True)

    meta_doc = {
            **split_meta(args),
            "cache_gate": {"max_ulp": float(args.cache_gate_ulp),
                           "source": args.cache_gate_source or "the derived 1 ulp (14.0b-22)"},
            "tsla": ({"source": str(args.tsla_vectors),
                      "classes": sorted(tsla_v),
                      "alphas": [float(a) for a in TSLA_ALPHAS],
                      "families": {
                          tsla_fam[c]: {
                              "tl": "13.5.4's TSLA-TL-zero-demo: the 30 "
                                    "TSLA-TL heads (upstream margin_add) on "
                                    "the K=5 prompt, their summed answer-row "
                                    "output averaged over 50 prompts, added "
                                    "at layer 16 of the K=0 receiver",
                              "tr": "the same construction from the 30 "
                                    "TSLA-TR heads (upstream cossim_norm = "
                                    "||oP||, how much a head writes into the "
                                    "label subspace) -- the class the "
                                    "upstream recommends for fixed-label "
                                    "classification",
                              "random": "the same construction from 30 "
                                        "heads drawn at random (seeded by "
                                        "the demo seed), the upstream's "
                                        "control"}[c]
                          for c in tsla_v}}
                     if tsla_v else None),
            "fv": ({"source": str(args.fv_vectors),
                    "family": f"FV-K{args.K}",
                    "alphas": [float(a) for a in TSLA_ALPHAS],
                    "per_seed": {str(s_): {"layer": int(fv_v[s_][0]),
                                           "norm": float(np.linalg.norm(fv_v[s_][1]))}
                                 for s_ in REGISTERED_SEEDS},
                    "mask": "ARM_BASE's: the demo columns closed to every "
                            "head; the vector is the only carrier",
                    "what": "Todd et al.'s function vector, built by "
                            f"run_fv_increment at K_base 0 from the K={args.K} "
                            "demonstrations (each one the query of a "
                            "leave-one-out extraction prompt, exact CIE after "
                            "an attribution-patching screen) and added as "
                            "alpha * v_FV at the answer row of layer L // 3; "
                            "alpha = 1 is the main arm, the grid descriptive",
                    "spec": "EXPLORATORY baseline (prereg 14.0b-25, "
                            "tools/baselines/specs/fv_atp_adapted_spec.md); "
                            "not a registered arm"}
                   if fv_v else None),
            "tv": ({"source": str(args.tv_vectors),
                    "families": sorted(tv), "layers": [int(l) for l in tv_layers],
                    "gate": tv_arm(tv_layers[0], args.K, on=False),
                    "per_seed": {str(s_): {str(l): float(np.linalg.norm(
                        tv[tv_family(args.K)][s_][l])) for l in tv_layers}
                                 for s_ in REGISTERED_SEEDS},
                    "mask": "ARM_BASE's: the demo columns closed to every "
                            "head; the vector is the only carrier",
                    "what": "Hendel et al.'s task vector: theta_L is the "
                            "answer-row output of decoder layer L on one "
                            f"forward over the K={args.K} demonstrations minus "
                            "one, that one's text the dummy query "
                            "(leave-one-out), built by run_tv_increment at "
                            "K_base 0; the arm REPLACES the answer row of "
                            "layer L's output on the K=0 receiver's forward. "
                            "The layer is chosen on validation among the "
                            "candidates by tools/select_tv_layer.py (ties to "
                            "the smaller layer); the m5 family is the mean "
                            "over five dummy queries, descriptive",
                    "spec": "EXPLORATORY baseline (prereg 14.0b-25, "
                            "tools/baselines/specs/task_vector_adapted_spec.md); "
                            "not a registered arm"}
                   if tv else None),
            "icv": ({"source": str(args.icv_vectors),
                     "family": f"ICV-K{args.K}",
                     "lambdas": [float(l) for l in icv_lams],
                     "gate": icv_arm(0.0, args.K),
                     "mask": "ARM_BASE's: the demo columns closed to every "
                             "head; the direction is the only carrier, and "
                             "the prefix cache is rebuilt per lambda with the "
                             "hooks",
                     "what": "Liu et al.'s in-context vector, built by "
                             "run_icv_increment at K_base 0 from the "
                             f"K={args.K} demonstrations ((x, xy) last-token "
                             "differences; centred PC1 + mean, the upstream's "
                             "formula) and applied as the upstream ICVLayer on "
                             "every decoder layer's MLP output at every "
                             "position of the K=0 receiver's forward; lambda "
                             "is chosen on validation by "
                             "tools/select_icv_lambda.py (NLL, ties to the "
                             "smaller lambda)",
                     "spec": "EXPLORATORY baseline (prereg 14.0b-25, "
                             "tools/baselines/specs/icv_adapted_spec.md); "
                             "not a registered arm"}
                    if icv_v is not None else None),
            "i2cl": ({"source": str(args.i2cl_vectors),
                      "family": f"I2CL-K{args.K}",
                      "gate": i2cl_arm(args.K, on=False),
                      "per_seed": {str(s_): {
                          "lambda_mean": float(i2cl_v[s_]["coef"][:, :, 0].mean()),
                          "beta_mean": float(i2cl_v[s_]["coef"][:, :, 1].mean())}
                          for s_ in REGISTERED_SEEDS},
                      "mask": "ARM_BASE's: the demo columns closed to every "
                              "head; the context vectors are the only "
                              "carrier, and the prefix cache is rebuilt with "
                              "the calibrated hooks per seed",
                      "what": "Li et al.'s implicit in-context learning, "
                              "built by run_i2cl_increment at K_base 0 from "
                              f"the K={args.K} demonstrations (mean last-token "
                              "self_attn / mlp outputs) with 4 L scalars "
                              "calibrated on the same demonstrations as "
                              "zero-shot pseudo-queries (noisy "
                              "self-calibration, the upstream's recipe); "
                              "applied as out <- beta * out + lambda * cv on "
                              "every layer's self_attn and mlp outputs at "
                              "every position of the K=0 receiver's forward",
                      "spec": "EXPLORATORY baseline (prereg 14.0b-25, "
                              "tools/baselines/specs/i2cl_adapted_spec.md); "
                              "not a registered arm"}
                     if i2cl_v is not None else None),
            "spec": "prereg_method_A.md section 13.5.2",
            # working rules 1.1: see run_viability_forward -- the log loses
            # the job id at archiving time, the artifact must not.
            "provenance": run_provenance(),
            "arm_name": "K0 receiver + selective K5 latent memory",
            "model": args.model, "method": args.method,
            "dtype": args.dtype, "attn": args.attn,
            "task": args.task, "K": args.K,
            "arms": list(all_arms), "layer_prefix": Ls,
            "seeds": [int(s) for s in REGISTERED_SEEDS],
            "carrier_bundle": str(args.carrier_bundle),
            "carrier_bundle_sha256": file_sha256(args.carrier_bundle),
            "query_manifest_sha256": file_sha256(args.query_manifest),
            "label_space_sha256": file_sha256(args.label_space),
            "top_n": args.top_n,
            "registered_arm": args.top_n is None,
            "per_seed": meta_seeds,
            "limit": int(args.limit),
            "n_queries_per_seed": {str(k): len(v)
                                   for k, v in ran_by_seed.items()},
            "note": "arms differ ONLY by the visibility mask over the demo "
                    "columns; positions, cache and softmax denominator are "
                    "identical, which is what makes K0-offset the right "
                    "baseline (13.5.2)"}
    if old_meta is not None:
        meta_doc = merged_meta(old_meta, meta_doc, todo)
    save_receiver_npz(args.out, out, all_arms, ran_by_seed, list(REGISTERED_SEEDS), meta_doc)
    print(f"\n  [output] {args.out}")
    print("  13.5.4(2)'s cache-equivalence gate ran on the first query of "
          "every seed and passed;")
    print("  it is a GATE, not a result -- `all-head cached K5` IS the "
          "ordinary prompt computed")
    print("  through the cache, so its number says nothing about the method.")
    return 0


def _install(model, masks, n_demo, torch):
    """Add each layer's demo-column mask inside its attention.

    The reference installer: one full [1, heads, n_query, n_cols] mask per
    layer, moved to the card in the hook. _install_rows below is the fast
    path the receivers run (the pattern built once per arm per seed and
    expanded on the card); tools/test_mask_rows.py pins the two to the same
    attention_mask bit for bit."""
    handles = []
    for l, m in masks.items():
        t = torch.as_tensor(m)

        def fn(_mod, args, kwargs, _t=t):
            am = kwargs.get("attention_mask")
            add = _t.to(model.device)
            if am is not None:
                # The model's 4D mask can be ONE column wider than the keys
                # (transformers builds it at past + q + 1 for a DynamicCache
                # when no 2D attention_mask is given, and eager slices it back
                # to key_states.shape[-2]). mask_pad_for rules on the gap:
                # 0 or 1 is that padding, anything else is a real
                # misalignment and raises. Zeros on the RIGHT, so the C_s and
                # D_s columns keep their indices.
                pad = mask_pad_for(add.shape[-1], am.shape[-1])
                if pad:
                    add = torch.nn.functional.pad(add, (0, pad))
                add = add.to(am.dtype)
            kwargs["attention_mask"] = add if am is None else am + add
            return args, kwargs

        handles.append(model.model.layers[l].self_attn
                       .register_forward_pre_hook(fn, with_kwargs=True))
    return handles


def prepare_mask_rows(masks, torch, device):
    """{layer: [heads, n_prefix] float64 tensor on `device`} from a masks dict
    built at n_query = 1, n_live = 1 (k0_memory.mask_prefix_rows): what
    _install_rows expands per query. Built once per arm per seed."""
    from tools.k0_memory import mask_prefix_rows
    return {l: torch.as_tensor(r).to(device)
            for l, r in mask_prefix_rows(masks).items()}


def _install_rows(model, rows, n_live, torch):
    """The fast twin of _install: each layer's mask is rebuilt on the card
    from its prefix pattern (`rows`, prepare_mask_rows) for this query's
    n_live -- the pattern repeated over the query rows, n_live zero columns
    appended, cast to the model's mask dtype BEFORE the expansion (the same
    elementwise values as casting after), then the same pad rule and the
    same `am + add`. No per-query numpy mask, no host-to-device copy."""
    from tools.k0_memory import expand_mask
    handles = []
    for l, r in rows.items():

        def fn(_mod, args, kwargs, _r=r):
            am = kwargs.get("attention_mask")
            if am is None:
                add = expand_mask(_r, n_live, torch)
            else:
                add = expand_mask(_r, n_live, torch, dtype=am.dtype)
                pad = mask_pad_for(add.shape[-1], am.shape[-1])
                if pad:
                    add = torch.nn.functional.pad(add, (0, pad))
            kwargs["attention_mask"] = add if am is None else am + add
            return args, kwargs

        handles.append(model.model.layers[l].self_attn
                       .register_forward_pre_hook(fn, with_kwargs=True))
    return handles


if __name__ == "__main__":
    raise SystemExit(main())
