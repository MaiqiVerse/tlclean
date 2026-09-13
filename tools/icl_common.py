"""Shared loaders and conventions for every analysis script in this repo.

WHY THIS EXISTS
---------------
Each analysis script used to re-implement "load the standard artifacts and apply
the standard conventions" from scratch, and the implementations diverged --
which is how identical settings produced different logic and, twice, wrong
results:

  * CLASS SPACE. analyze_error_prediction correctly argmaxed over all 50 label
    tokens; the two kappa scripts sliced [:n_train], assuming the 30 train
    classes are the index prefix 0..29. They are NOT -- TREC-fine's train
    classes are scattered across 0..49, so C2 compared 17 of 30 classes
    (272 = 17x16 pairs instead of 870) and kappa-voting silently dropped every
    vote for a class with index >= 30.
  * ARTIFACT KEYS. check_model_accuracy writes its per-prompt list under
    "per_prompt"; a script guessed "per_records" and crashed on the string keys.

Both were one-line conventions that should exist in exactly one place. If a
loader you need is missing, ADD IT HERE rather than inlining it -- that is the
whole point of the module.

CONVENTIONS THIS MODULE ENCODES
-------------------------------
1. Train-class set = the set of classes that actually appear among the demos.
   Never `range(n_train)`, never `[:n_train]`.
2. The saved `model_pred_class` from check_model_accuracy is an argmax over ALL
   label tokens (not restricted to train classes); any agreement statistic
   against it must live in the full class space.
3. `per_prompt.jsonl` from exp_identity_verification carries a HEADER whose
   `label_token_ids` gives the PROJECTION order of the `predicted` list, which
   is NOT class order. Translate through the calibration header before
   comparing to class indices. (A 2026-05-16 indexing bug from exactly this
   confusion invalidated a whole round of TL-alone numbers; see RESULTS 5 row 7.)
4. Three different normalizations float around and must never be mixed in one
   table -- see NORMALIZATION_NOTE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Conventions worth stating in prose, quoted by scripts in their output
# ---------------------------------------------------------------------------

NORMALIZATION_NOTE = (
    "Three distinct units appear in this project and must not be mixed in one "
    "table: (a) RAW projection <u, W_U[c]> (what phi in the kappa npz is); "
    "(b) GAMMA-FOLDED <u, gamma*W_U[dir]> (final-RMSNorm weight folded in); "
    "(c) NORMALIZED (b)/rms(x) with rms frozen per prompt -- the only one whose "
    "components sum exactly to the model's logit (probe_component_budget's "
    "convention). rms is a positive scalar so it never changes a sign; gamma "
    "sits inside the inner product and can."
)

# TREC-fine coarse-category boundary classes: the last fine class of each coarse
# group, derived from tasks/trec_fine_task.py's _FINE_LABELS order (not assumed).
TREC_FINE_BOUNDARY_CLASSES = (1, 23, 27, 31, 36, 49)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_jsonl(path) -> tuple[dict | None, list[dict]]:
    """Return (header, rows) for our header-plus-records jsonl files."""
    header, rows = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("header"):
                header = obj
            else:
                rows.append(obj)
    return header, rows


def load_calibration(path) -> dict:
    """Header + per-prompt fields of a data/calibration_*_uuid.jsonl.

    Returns dict with:
      label_token_ids : list[int]  -- CLASS order (index c -> token for class c)
      n_classes       : int        -- size of the label vocabulary (e.g. 50)
      n_train_classes : int        -- header's count (e.g. 30) -- a COUNT, not a range
      true_class      : np.int32[n_prompts]
      abstract_labels : list[str] | None
      tok2class       : dict[int, int]
    """
    header, rows = load_jsonl(path)
    if header is None:
        raise SystemExit(f"{path}: no header line")
    label_token_ids = [int(x) for x in header["label_token_ids"]]
    return {
        "label_token_ids": label_token_ids,
        "n_classes": len(label_token_ids),
        "n_train_classes": int(header.get("n_train_classes", len(label_token_ids))),
        "true_class": np.asarray([int(r["true_class_idx"]) for r in rows],
                                 dtype=np.int32),
        "abstract_labels": header.get("abstract_labels"),
        "tok2class": {t: c for c, t in enumerate(label_token_ids)},
        "header": header,
    }


def train_class_set(demo_class=None, true_class=None) -> np.ndarray:
    """THE train-class set as a sorted index array -- the scattered truth.

    Prefer demo_class (classes that actually have demonstrations). Falls back to
    the observed true classes. NEVER use range(n_train) or a [:n_train] slice:
    on TREC-fine the 30 train classes sit at scattered indices within 0..49, and
    assuming a prefix silently drops classes (see module docstring).
    """
    if demo_class is not None:
        vals = {int(c) for c in np.asarray(demo_class).ravel() if int(c) >= 0}
        if vals:
            return np.asarray(sorted(vals))
    if true_class is not None:
        vals = {int(c) for c in np.asarray(true_class).ravel() if int(c) >= 0}
        if vals:
            return np.asarray(sorted(vals))
    raise ValueError("train_class_set: need demo_class or true_class")


def argmax_in_set(scores: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """argmax restricted to a scattered column set, mapped back to class indices.

    scores: [n, n_classes]; cols: sorted index array from train_class_set.
    """
    return np.asarray(cols)[scores[:, np.asarray(cols)].argmax(axis=1)]


def load_model_accuracy(path) -> dict[int, dict]:
    """per-prompt records from check_model_accuracy.py --output, keyed by prompt_idx.

    That script writes {"summary": ..., "per_prompt": [...]}. Older/other shapes
    are accepted; an unknown shape reports the ACTUAL top-level keys instead of
    dying on a string index.
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        rows = obj
    elif "per_prompt" in obj:
        rows = obj["per_prompt"]
    elif "per_records" in obj:
        rows = obj["per_records"]
    else:
        raise SystemExit(f"{path}: no per-prompt list found "
                         f"(top-level keys: {sorted(obj.keys())})")
    return {int(r["prompt_idx"]): r for r in rows}


def load_projections(per_prompt_path, class_order) -> tuple[np.ndarray, list[int]]:
    """Per-class TL projections from exp_identity_verification's per_prompt.jsonl.

    Returns (proj[n_prompts, n_classes] in CLASS order, prompt_idx list).
    Entries with no projection column are -inf.

    The header's label_token_ids is the PROJECTION order of `predicted`, which is
    not class order -- translating through it is mandatory (see docstring note 3).
    A legacy file without a header is refused rather than guessed.
    """
    header, rows = load_jsonl(per_prompt_path)
    if header is None:
        raise SystemExit(f"{per_prompt_path}: no header -- projection order "
                         f"unknown, refusing to guess (see RESULTS section 5 row 7)")
    tok2class = {t: c for c, t in enumerate(class_order)}
    proj_idx_to_class = [tok2class.get(int(t), -1) for t in header["label_token_ids"]]
    n_classes = len(class_order)
    out = np.full((len(rows), n_classes), -np.inf)
    idxs = []
    for r_i, r in enumerate(rows):
        pred = np.asarray(r["predicted"], dtype=np.float64)
        for j, c in enumerate(proj_idx_to_class):
            if 0 <= c < n_classes:
                out[r_i, c] = pred[j]
        idxs.append(int(r.get("prompt_idx", r_i)))
    if not np.isfinite(out).any():
        raise SystemExit(f"{per_prompt_path}: no projection column maps into the "
                         f"class space -- label orders inconsistent")
    return out, idxs


def load_budget(path) -> dict:
    """probe_component_budget.py --output JSON, with legacy tolerance.

    Adds derived flags so callers degrade gracefully instead of crashing on the
    pre-`--pair-mode` outputs (whose pairing IS model top1/top2 by construction):
      pair_mode      : str | None
      has_per_record : per_record_raw + rms_per_prompt present
      has_pair_idx   : pair_idx_per_prompt present
      is_contrast    : both arms present
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    obj["_pair_mode"] = obj.get("pair_mode")
    obj["_has_per_record"] = ("per_record_raw" in obj and "rms_per_prompt" in obj)
    obj["_has_pair_idx"] = "pair_idx_per_prompt" in obj
    obj["_is_contrast"] = bool(obj.get("contrast")) and \
        obj.get("per_record_raw_random") is not None
    return obj


def budget_series(payload: dict, arm: str, key: str, channel: str) -> np.ndarray:
    """Normalized per-prompt series from a budget payload.

    arm: "uuid" (per_record_raw / rms_per_prompt) or "random"
    (per_record_raw_random / rms_per_prompt_random). Division by the frozen
    per-prompt rms is convention (c) in NORMALIZATION_NOTE -- the additive one.
    """
    if arm == "uuid":
        raw, rms = payload["per_record_raw"], payload["rms_per_prompt"]
    elif arm == "random":
        raw, rms = payload["per_record_raw_random"], payload["rms_per_prompt_random"]
    else:
        raise ValueError(f"arm must be 'uuid' or 'random', got {arm!r}")
    return np.asarray(raw[key][channel], dtype=np.float64) / np.asarray(rms, dtype=np.float64)


def load_kappa_npz(path) -> dict:
    """probe_kappa_matrix.py npz + its json meta, with the train-class set derived.

    Keys: kappa[Q,P,H], phi[P,C,H] (frozen), seg, is_label, demo_class,
    true_class, query_len, delta_tl_proj, head_keys, meta, train_cols.
    """
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))
    demo_class = z["demo_class"]
    return {
        "kappa": z["kappa_prefix"].astype(np.float64),
        "phi": z["phi_prefix"].astype(np.float64),
        "proj_query_part": z["proj_query_part"].astype(np.float64),
        "seg": z["seg_prefix"],
        "is_label": z["is_label_prefix"].astype(bool),
        "demo_class": demo_class,
        "true_class": z["true_class_idx"],
        "query_len": z["query_len"],
        "n_tokens": z["n_tokens"],
        "phi_dev": z["phi_dev"],
        "identity_diff": z["identity_diff"],
        "delta_tl_proj": z["delta_tl_proj"].astype(np.float64),
        "head_keys": [str(x) for x in z["head_keys"]],
        "meta": meta,
        "train_cols": train_class_set(demo_class=demo_class),
    }


def head_dim(model) -> int:
    """Per-head width of the attention projections, read off the weights.

    `hidden_size // num_attention_heads` is what every probe used to compute,
    and it is the Llama convention: 4096 / 32 = 128 on Llama-3.1-8B. It is
    WRONG on any model whose config sets head_dim on its own -- Qwen3-4B has
    hidden 2560, 32 heads and head_dim 128, so the quotient (80) makes the
    v_proj view in diagnostic_forward fail, and would slice o_proj at the
    wrong columns anywhere it did not. o_proj's input width is the structural
    truth (n_heads * head_dim by construction), so it is read from there and
    cross-checked against config.head_dim when the config carries one.
    Identical to the old quotient on every model where the two agree
    (Llama-2 / Llama-3.1, Qwen2.5-7B, Qwen3-8B).
    """
    cfg = model.config
    n_h = int(cfg.num_attention_heads)
    o_in = int(model.model.layers[0].self_attn.o_proj.in_features)
    if o_in % n_h:
        raise ValueError(
            f"o_proj.in_features {o_in} is not a multiple of "
            f"num_attention_heads {n_h}")
    d = o_in // n_h
    cfg_d = getattr(cfg, "head_dim", None)
    if cfg_d is not None and int(cfg_d) != d:
        raise ValueError(
            f"config.head_dim {cfg_d} disagrees with o_proj: "
            f"{o_in} / {n_h} = {d}")
    return d


def mcnemar_exact(a_ok, b_ok) -> dict:
    """Exact two-sided McNemar on paired binary outcomes (no scipy).

    b01 = a wrong & b right, b10 = a right & b wrong; under the null each
    discordant pair is a fair coin, so p is the two-sided binomial tail.
    The same statistic analyze_label_repair.mcnemar and
    analyze_substrate_curve.mcnemar_exact compute; it lives here so the next
    analyzer imports it instead of writing another copy (working rules 2.6.2).
    `floor` is the smallest p this many discordant pairs can produce,
    2^(1-n): a non-significant p above the floor is a null result, one at
    the floor could not have been significant with these pairs.
    """
    from math import comb
    a = np.asarray(a_ok, dtype=bool)
    b = np.asarray(b_ok, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"paired outcomes differ in shape: {a.shape} vs {b.shape}")
    b01 = int((~a & b).sum())
    b10 = int((a & ~b).sum())
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "n_discordant": 0, "p": 1.0, "floor": 1.0}
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return {"b01": b01, "b10": b10, "n_discordant": n,
            "p": float(min(1.0, 2.0 * tail)),
            "floor": float(min(1.0, 2.0 ** (1 - n)))}


def run_provenance() -> dict:
    """Where this run came from, stamped into every GPU artifact's meta.

    Only sbatch jobs have a job id -- a zero-GPU `python tools/analyze_*.py`
    never touches the scheduler, so asking for one is a category error (and I
    made it). Artifacts that DO come from a job should carry the id themselves
    rather than depending on someone remembering it: the log gets renamed on
    archiving and the id is lost at that moment, which is exactly why
    working rules 1.1 wants both recorded.

    THE COMMIT IS THE POINT. A job id says which run produced a file; the
    commit says which CODE did, which is what reproducing it needs. One
    40-character string covers every module -- including the ones no
    hand-written list would have named, tasks/ and script/ among them -- so
    nothing here hashes source files. `repo_dirty` is recorded because a
    commit id on a dirty tree is a lie, and a lie is worse than an absence.

    Returns {} of strings; missing keys mean the run was not under slurm.
    """
    import os
    import socket
    import subprocess
    keys = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST")
    out = {k.lower(): os.environ[k] for k in keys if os.environ.get(k)}
    out["host"] = socket.gethostname()
    if "slurm_job_id" in out:
        out["expected_log"] = f"slurm-{out['slurm_job_id']}.out"
    root = str(Path(__file__).resolve().parent.parent)
    try:
        h = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        # TRACKED MODIFICATIONS ONLY. Plain `status --porcelain` counts
        # UNTRACKED files, and data/ and results/ are untracked BY DESIGN --
        # 2.5.1 puts them on the server and never in git -- so the flag was
        # `true` on every server run whatever the code was doing. A flag that
        # cannot be false distinguishes nothing, and this one was read as
        # evidence that job 835618's commit did not identify its code
        # (it does not, but for the other reason: a real edit to
        # experiments/causal_ablation.py). working rules 11b: a criterion nothing
        # can satisfy is as broken as one nothing can fail.
        st = subprocess.run(["git", "-C", root, "status", "--porcelain",
                             "--untracked-files=no"],
                            capture_output=True, text=True, timeout=30)
        un = subprocess.run(["git", "-C", root, "status", "--porcelain",
                             "--untracked-files=all"],
                            capture_output=True, text=True, timeout=30)
        if h.returncode == 0:
            out["repo_commit"] = h.stdout.strip()
            out["repo_dirty"] = str(bool(st.stdout.strip())).lower()
            # WHICH files, not just whether: "dirty" sends someone looking and
            # says nothing about where. Truncated because the point is to name
            # the edit, not to embed a diff.
            if st.stdout.strip():
                # ln[2:].strip(), not ln[3:]: the status field is two
                # characters but the separator is not reliably one, and
                # slicing a fixed width lost the first letter of the path.
                out["repo_modified"] = ", ".join(
                    ln[2:].strip() for ln in st.stdout.strip().splitlines()[:8])
            # Recorded separately so the untracked state stays VISIBLE without
            # being conflated with a code change. On the server this is never
            # empty and that is correct, not a fault.
            if un.returncode == 0:
                out["repo_untracked_count"] = str(
                    sum(1 for ln in un.stdout.splitlines()
                        if ln.startswith("??")))
        else:
            out["repo_commit"] = f"UNAVAILABLE: {h.stderr.strip()[:80]}"
    except (OSError, subprocess.SubprocessError) as e:
        out["repo_commit"] = f"UNAVAILABLE: {type(e).__name__}: {e}"
    return out


# The demos the published 4-demo run drew on seed 42 with D=148 (RESULTS 29/30,
# experiment_report/2026-08-20 and 2026-08-22). Asserted whenever those exact
# conditions recur, so a numpy change to Generator.choice cannot silently move
# the pinned prefix and destroy the reproduction gate below.
PINNED_DEMOS_SEED42_D148 = (12, 64, 96, 112)


def draw_perturbations(demo_blocks, demo_class0, abstract_labels,
                       n_demos, n_alts, seed, n_demos_pinned=4, ids_only=False):
    """The demo-relabelling perturbation set: (demo_ids, perts).

    perts[p] = (demo, c_old, c_new, relabelled_block_text), the label swap being
    the only edit -- everything before the final "\nType:" is preserved.

    BOTH ARMS MUST CALL THIS. The prompt-edit arm and the V-patch arm are
    compared perturbation by perturbation, so a difference between two copies of
    the draw would show up as a physical finding. It lived as two copies.

    SIZE-NESTED BY CONSTRUCTION. `rng.choice(D, size=k, replace=False)` gives
    unrelated sets for different k, so simply raising n_demos would share
    nothing with the published 4-demo run and leave the rewrite ungated. The
    first `n_demos_pinned` demos are therefore drawn from `default_rng(seed)`
    with the same call order as that run, their alternatives consume the same
    generator in the same sequence, and only then do the extra demos come from
    the independent stream `default_rng([seed, 1])`. Consequences:

      * n_demos == n_demos_pinned reduces to the original code path exactly;
      * for any larger n_demos, perturbations 0 .. 2*n_demos_pinned-1 ARE the
        published ones, so their slopes/weights must come back unchanged. That
        is the only check this enlargement has against a silent regression, and
        it is a live one: the numbers are already published.

    perts is ordered [pinned demos in ascending id][extra demos in ascending
    id], each demo contributing its n_alts alternatives contiguously, so
    p // n_alts still indexes the demo list and p % n_alts the alternative.
    """
    demo_class0 = np.asarray(demo_class0)
    # ids_only lets the weight predictor reproduce the SAME demo selection with
    # no prompt text in hand. It must go through this function rather than
    # re-deriving the draw, or the prediction would describe a different run.
    D = demo_class0.size if ids_only else len(demo_blocks)
    if demo_class0.size != D:
        raise SystemExit(f"draw_perturbations: {demo_class0.size} demo classes "
                         f"vs {D} demo blocks")
    train_cols = np.asarray(sorted({int(c) for c in demo_class0 if c >= 0}))
    if train_cols.size < n_alts + 1:
        raise SystemExit(f"draw_perturbations: {train_cols.size} demo classes "
                         f"cannot supply {n_alts} alternatives")
    n_pinned = min(int(n_demos_pinned), int(n_demos))
    if n_demos > D:
        raise SystemExit(f"draw_perturbations: asked for {n_demos} demos but "
                         f"the prompt has {D}")

    def _alts(gen, ids):
        out = []
        for d in ids:
            d = int(d)
            c_old = int(demo_class0[d])
            others = train_cols[train_cols != c_old]
            if ids_only:
                for c_new in gen.choice(others, size=n_alts, replace=False):
                    out.append((d, c_old, int(c_new), None))
                continue
            blk = demo_blocks[d]
            cut = blk.rfind("\nType:")
            if cut == -1:
                raise SystemExit(f"demo {d}: no label marker to swap")
            for c_new in gen.choice(others, size=n_alts, replace=False):
                out.append((d, c_old, int(c_new),
                            blk[:cut] + "\nType: " + abstract_labels[int(c_new)]))
        return out

    rng = np.random.default_rng(seed)
    pinned = np.sort(rng.choice(D, size=n_pinned, replace=False))
    if seed == 42 and D == 148 and n_pinned == 4:
        got = tuple(int(x) for x in pinned)
        if got != PINNED_DEMOS_SEED42_D148:
            raise SystemExit(
                f"draw_perturbations: the pinned draw is {got}, not "
                f"{PINNED_DEMOS_SEED42_D148}. Generator.choice changed, so the "
                f"published 4-demo run can no longer be reproduced as a prefix "
                f"and the enlargement would be ungated. Pin numpy or drop the "
                f"nesting deliberately.")
    perts = _alts(rng, pinned)          # same generator, same order as before
    demo_ids = pinned
    if n_demos > n_pinned:
        pool = np.setdiff1d(np.arange(D), pinned)
        gen2 = np.random.default_rng([seed, 1])
        extra = np.sort(gen2.choice(pool, size=n_demos - n_pinned,
                                    replace=False))
        perts += _alts(gen2, extra)
        demo_ids = np.concatenate([pinned, extra])
    return demo_ids, perts


def demo_onehots(seg, is_label):
    """(onehot[P,D], onehot_label[P,D]) mapping prefix positions to their demo.

    seg[p] = demo index of position p, or <0 for non-demo positions (BOS, query).
    Columns of onehot_label keep only that demo's LABEL-token positions.
    """
    seg = np.asarray(seg)
    is_label = np.asarray(is_label).astype(bool)
    P = seg.shape[0]
    D = int(seg.max()) + 1 if seg.size and seg.max() >= 0 else 0
    demo_mask = seg >= 0
    onehot = np.zeros((P, D))
    onehot[np.arange(P)[demo_mask], seg[demo_mask]] = 1.0
    lab_pos = is_label & demo_mask
    onehot_lab = np.zeros((P, D))
    onehot_lab[np.arange(P)[lab_pos], seg[lab_pos]] = 1.0
    return onehot, onehot_lab


def demo_vote_matrices(phi, seg, is_label):
    """Per-demo per-class vote matrices v_demo[D,C] and v_lab[D,C].

    v[d, c] = mean over demo d's positions of the head-summed vote phi[p, c, :].
    'label' restricts the average to the demo's label-token positions. This is
    the quantity tl.tex Condition 1 (C2) is stated over, so the two scripts that
    measure C2 must build it IDENTICALLY -- hence it lives here rather than in
    either of them (the two kappa scripts having each rolled their own class
    space is what produced the 272-vs-870 bug).

    phi: [P, C, H] frozen prefix votes. Positions with no tokens give 0, not nan.
    """
    phi = np.asarray(phi, dtype=np.float64)
    onehot, onehot_lab = demo_onehots(seg, is_label)
    phi_agg = phi.sum(axis=2)                                   # [P, C]
    sizes = onehot.sum(axis=0)                                  # [D]
    lab_sizes = onehot_lab.sum(axis=0)
    v_demo = (onehot.T @ phi_agg) / np.maximum(sizes[:, None], 1)
    v_lab = (onehot_lab.T @ phi_agg) / np.maximum(lab_sizes[:, None], 1)
    return v_demo, v_lab


def c2_contrast_matrix(v, demo_class, train_cols):
    """C2 contrast matrix M[i,j] = E_{d: class(d)=c_i}[ v[d,c_i] - v[d,c_j] ].

    Rows/cols are indexed by position in train_cols (the SCATTERED class set),
    not by raw class index. Diagonal is 0 and excluded from the pair count.
    Condition 1 holds iff every off-diagonal entry is > 0.
    """
    v = np.asarray(v, dtype=np.float64)
    demo_class = np.asarray(demo_class)
    cols = np.asarray(train_cols)
    n = cols.size
    # mean vote of class-c_i demos for every class: V[i, :] over the full C axis
    V = np.stack([v[demo_class == int(c)].mean(axis=0) for c in cols])   # [n, C]
    own = V[np.arange(n), cols]                                          # [n]
    M = own[:, None] - V[:, cols]                                        # [n, n]
    np.fill_diagonal(M, 0.0)
    return M, V, own


if __name__ == "__main__":
    print(__doc__)
    print("\nNORMALIZATION_NOTE:\n  " + NORMALIZATION_NOTE)
    print(f"\nTREC_FINE_BOUNDARY_CLASSES = {TREC_FINE_BOUNDARY_CLASSES}")
