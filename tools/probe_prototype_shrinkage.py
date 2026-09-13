"""Method A: LOO class-prototype V shrinkage on clean prompts. Sections 3, 4, 11.

A NEW probe rather than another condition bolted onto `probe_label_repair.py`.
That script's business is corrupting labels and repairing them; Method A's is a
clean prompt with no corruption anywhere, and the two share only the patch
plumbing. Section 11 asks for the separation by name.

    python tools/probe_prototype_shrinkage.py \\
        --mode clean --arm proto_loo --gamma 0,0.25,0.5,0.75,1 \\
        --carrier-impl gqa_group_v --carrier-json results/tl_heads_L31_...json \\
        --demo-seeds 42 --split test_seed \\
        --query-manifest results/prereg_method_A_query_manifest.json \\
        --uuid-jsonl data/calibration_trec_fine_per_class_K5_seed42_uuid.jsonl \\
        --allow-kv-group --out results/method_a_group_proto_loo.npz

THE FIXED-SOURCE RULE IS STRUCTURAL, NOT A CONVENTION (section 3, gate 5).

Every v0, mu and Delta comes from ONE natural forward of that query. When an
early layer is patched, later layers see a changed hidden state and would
produce different values if anything recomputed them -- and "the prototype the
model would have had, had it already been steered" is a different operator with
a different meaning. Section 3 says this experiment does not run it.

A comment saying so would not survive contact with a refactor, so `FixedSource`
owns the natural cache, exposes it read-only, and counts writes after sealing.
Nothing downstream can reach a live cache because nothing downstream is given
one. `--self-test` exercises that with a stub whose values change under
patching, and checks the targets do not move.

TWO SAMPLING REGIMES, AND ONE RUN CANNOT SPAN SEEDS. The L3.1 method main table
follows PCW's evaluation setting -- fixed test size 250, multiple random seeds --
drawing 250 queries per demo seed independently, so seed 42's query set and seed
43's are different questions. `--split test_seed` therefore takes exactly one
seed. The three runs are combined afterwards by ONE aggregate clustered test on
the mean of the per-seed means, clustering on unique query_id because the draws
overlap (tools/method_a_stats.decide_aggregate); never by averaging "the same
query" across seeds, and never by requiring each seed to be significant on its
own. `--split test_common` keeps the single shared set for the L2 mechanism
setting and the section 7 reconciliation, where holding the query fixed while
the seed varies IS the comparison.

THE DECISION SPACE IS THE TRAIN CLASSES (section 6.1). Candidate tokens come
from the manifest, which selects them out of the calibration header's existing
mapping by train-class id. The task is closed-set over the classes with training
demonstrations and gold support; the rest are outside the decision space by
definition of the task, not because anything makes them unreachable.

WHAT EACH CARRIER IMPLEMENTATION MEANS (sections 3.1, 3.2). They are not three
ways to do one thing; two of them are different interventions.

  mha_v             L2 / Llama-2-7B. MHA, so every attention head owns a v_proj
                    slot and the write is genuinely per head.
  gqa_group_v       L3.1 confirmatory. Writes a whole KV group, which by
                    construction moves all four query heads reading it. Report
                    as KV-group-selective; never call it per-head.
  gqa_head_realized L3.1 secondary. Overwrites individual query-head views after
                    repeat_kv, so group-mates keep their natural values. It is a
                    path intervention, not a V patch -- see tools/realized_write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.exploratory_donor_arms import ARMS as DONOR_ARMS  # noqa: E402
from tools.exploratory_donor_arms import \
    written_rows_from_donor  # noqa: E402
from tools.prototype_targets import (ARMS, GAMMA_GRID,  # noqa: E402
                                     LayerWrite, dose_scale, group_members,
                                     raw_delta, written_label_rows,
                                     written_position_rows)

# EXPLORATORY, and kept in a separate name so that every `arm in ARMS` test in
# this file keeps meaning "one of section 4's eleven". A run of one of these
# says so in its own meta; nothing about H1-H6 reads them.
ALL_ARMS = tuple(ARMS) + tuple(DONOR_ARMS)

# Which frozen gamma* governs which arm on TEST. The three settings of
CARRIER_IMPLS = ("mha_v", "gqa_group_v", "gqa_head_realized")

# THE THREE GAMMA* ARE PER SETTING/IMPLEMENTATION, NOT PER ARM NAME.
#
# 5(4): "L2 raw LOO、L3.1 Method A-group 与 Method A-head 各自的完整 gamma 网格
# ... 每个 setting/implementation 独立以三 seed 平均 candidate NLL 最小者冻结
# gamma*_L2、gamma*_group、gamma*_head". All three run RAW LOO; what
# distinguishes them is the write implementation, which is exactly
# CARRIER_IMPLS.
#
# This was a dict keyed on ARM, and the keys had been guessed from the names:
# `proto_global` got gamma_group because both say "glob/group", and
# `matched_heads` got gamma_head because both say "head". But "group" in
# gamma*_group means the KV GROUP that L3.1 writes as a unit, and "head" means
# the per-head realised write -- neither is an arm. Both of those arms are
# CONTROLS (4's table: "平均化本身", "head/group 定位"), while the main method
# `proto_loo` was mapped to gamma_L2 and would therefore have been checked
# against another setting's number on the test path. That is the one place a
# mistake cannot be taken back.
GAMMA_KEY_FOR_IMPL = {
    "mha_v": "gamma_L2",              # L2 exact-mechanism setting
    "gqa_group_v": "gamma_group",     # L3.1, whole-KV-group write
    "gqa_head_realized": "gamma_head",  # L3.1, per-query-head realised write
}

# Arms that write nothing: their only defensible gamma is 0.0, whatever the
# setting. Listed rather than inferred so that an unknown arm is an error and
# not a silent zero.
ZERO_WRITE_ARMS = ("natural", "sham")


def gamma_key_for(arm, carrier_impl):
    """(key, why) -- which frozen gamma* a test run must use.

    Returns (None, None) for an arm that writes nothing, (key, None) for one
    that does, and (None, reason) when the question cannot be answered.
    Composed as a reason rather than raised so it joins the caller's problem
    list, like freeze_gamma_path.
    """
    if arm not in ARMS:
        return None, (f"arm {arm!r} is not one of {ARMS}, so no registered "
                      "gamma* applies and a test run cannot be checked")
    if arm in ZERO_WRITE_ARMS:
        return None, None
    if carrier_impl not in GAMMA_KEY_FOR_IMPL:
        return None, (
            f"carrier implementation {carrier_impl!r} has no registered "
            f"gamma* (expected one of {sorted(GAMMA_KEY_FOR_IMPL)}). 5(4) "
            "freezes one gamma* per setting/implementation, so without it the "
            "run cannot say which frozen number it is claiming to use")
    return GAMMA_KEY_FOR_IMPL[carrier_impl], None
MODES = ("clean", "corrupt-targeting")
LABEL_ROW_ARMS = tuple(a for a in ARMS if a not in
                       ("natural", "sham", "pos_raw", "pos_dose"))


# ==========================================================================
# fixed source
# ==========================================================================
class FixedSource:
    """The natural forward's values, sealed. Section 3's fixed-source rule.

    Keyed by (layer, carrier), where the carrier is the KV SLOT -- under MHA
    that is the attention head, under GQA the KV group. All three
    implementations read the same natural values; they differ only in who is
    made to see the write.

    `add` collects, `seal()` closes it, and every later read is read-only.
    Adding after sealing raises rather than silently replacing a target with a
    live-recomputed one -- the failure this class exists to make impossible.
    """

    def __init__(self, label_rows, labels, demo_ids, donors=None):
        self._v = {}
        self._sealed = False
        # {(layer, carrier): [n_demos] donor index}. Only the exploratory
        # donor arms read it; a registered arm with donors present behaves
        # exactly as it does without them, which `written` enforces by
        # branching on the ARM rather than on whether donors exist.
        self.donors = dict(donors or {})
        self.label_rows = np.asarray(label_rows, dtype=np.int64)
        self.labels = np.asarray(labels)
        self.demo_ids = list(demo_ids)
        self.n_refresh_attempts = 0
        if not (self.label_rows.size == self.labels.size
                == len(self.demo_ids)):
            raise ValueError(
                f"{self.label_rows.size} label rows, {self.labels.size} labels, "
                f"{len(self.demo_ids)} demo ids -- these index the same demos "
                "and must agree")

    def add(self, layer, carrier, v_label_rows):
        if self._sealed:
            self.n_refresh_attempts += 1
            raise RuntimeError(
                f"layer {layer} carrier {carrier}: the fixed source is sealed. "
                "Section 3 requires every v0/mu/Delta to come from the ONE "
                "natural forward; writing here would substitute values "
                "recomputed from an already-patched hidden state, which is a "
                "different operator (live-recomputed prototype) that this "
                "preregistration does not run.")
        v = np.asarray(v_label_rows, dtype=np.float64)
        if v.ndim != 2 or v.shape[0] != self.label_rows.size:
            raise ValueError(f"layer {layer} carrier {carrier}: expected "
                             f"[{self.label_rows.size}, d_head], got {v.shape}")
        self._v[(int(layer), int(carrier))] = v

    def seal(self):
        if not self._v:
            raise RuntimeError("nothing captured: the natural forward produced "
                               "no values, so every arm would be a no-op")
        self._sealed = True
        return self

    @property
    def sealed(self):
        return self._sealed

    def keys(self):
        return sorted(self._v)

    def layers(self):
        return sorted({l for l, _ in self._v})

    def v0(self, layer, carrier=0):
        if not self._sealed:
            raise RuntimeError("read before seal(): the natural forward is not "
                               "finished, so these values are not the source")
        key = (int(layer), int(carrier))
        if key not in self._v:
            raise KeyError(f"no natural value captured for layer {layer} "
                           f"carrier {carrier}; captured {self.keys()[:6]}...")
        return self._v[key].copy()             # copy: callers must not mutate

    def written(self, layer, arm, gamma, *, setting, demo_seed, carrier=0):
        """The value each demo's label row receives at this (layer, carrier).

        THE BRANCH IS ON THE ARM, not on whether donors were supplied. A
        registered arm run with a donor map present must produce exactly what
        it produces without one, or the eleven would quietly change meaning
        whenever the exploratory plumbing happened to be loaded.
        """
        if arm in DONOR_ARMS:
            key = (int(layer), int(carrier))
            if key not in self.donors:
                raise KeyError(
                    f"arm {arm!r} needs a donor for layer {layer} carrier "
                    f"{carrier} and none was supplied. Filling it in here "
                    "from whatever is at hand is exactly how the frozen "
                    "variant would silently become the dynamic one; the "
                    f"donors given cover {sorted(self.donors)[:6]}")
            return written_rows_from_donor(gamma, self.v0(layer, carrier),
                                           self.donors[key])
        return written_label_rows(arm, gamma, self.v0(layer, carrier),
                                  self.labels, self.demo_ids, setting=setting,
                                  demo_seed=demo_seed, layer=layer,
                                  carrier=carrier)

    def delta(self, layer, carrier=0):
        return raw_delta(self.v0(layer, carrier), self.labels)


# ==========================================================================
# prompts
# ==========================================================================
def render_demo(text, label):
    """One demonstration block, in the task's own format.

    `tasks/trec_fine_task.py` renders "Question: {text}\\nType: {label}" and
    joins blocks with a blank line; the abstract-label variant substitutes the
    label only (experiments/data_calibration.patched_abstract_labels). The
    format is reproduced here rather than driven through the task object
    because the runner needs prompts for the manifest's queries, which are not
    the shuffled 250 the calibration file happens to hold -- and
    `verify_prefix` below checks the reproduction against a real calibration
    prompt instead of trusting it.
    """
    return f"Question: {text}\nType: {label}"


def render_prompt(prefix_blocks, query_text):
    return "\n\n".join(list(prefix_blocks) + [f"Question: {query_text}\nType:"])


def build_prefix(demo_pairs, abstract_labels):
    """The fixed prefix's blocks, in the task's demo order."""
    return [render_demo(text, abstract_labels[c]) for c, text in demo_pairs]


def build_text_lookup(task, task_name):
    """content_hash -> raw text, over BOTH splits.

    Validation queries are drawn from TRAIN and test queries from TEST, so a
    lookup built on one split cannot resolve the other. Building it on the test
    split alone made `--split validation` fail with "not found in the task's
    test split", which is true and useless.
    """
    from tools.prereg_ids import content_hash
    from tools.prereg_task import test_rows, train_rows

    out = {}
    for c, t, _src in list(train_rows(task, task_name)) + \
            list(test_rows(task, task_name)):
        out[content_hash(int(c), t)] = t
    return out


def manifest_reservation(manifest, *, demo_seed=None):
    """The (class, text) pairs the manifest reserved for validation.

    The demonstrations must be drawn from train MINUS these, exactly as
    data_calibration was told to. Rebuilding the prefix without the reservation
    produces a different demo set from the one in the calibration files -- the
    prompt would differ from the published one and `verify_prefix` is the only
    thing that would notice.
    """
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    vbs = manifest.get("validation_by_seed")
    if vbs is None:
        raise SystemExit(
            "the query manifest has no 'validation_by_seed', so the "
            "reservation that shaped the demonstrations cannot be "
            "reconstructed. A manifest with a single 'validation' list "
            "predates section 14.0a and describes a draw that no longer "
            "exists. Rebuild it with tools/build_query_manifest.py.")
    if demo_seed is None:
        raise SystemExit(
            "manifest_reservation needs a demo_seed. Each seed reserved its "
            "OWN validation before its demonstrations were drawn (14.0a), so "
            "there is no single reservation to return. The union is NOT a "
            "safe default: prefix_demo_rows takes one excluded_docs per call, "
            "so a union would be applied to every seed, remove three times as "
            "much from each demo pool, and yield prefixes matching no "
            f"published calibration file. Available: {sorted(vbs)}")
    if str(demo_seed) not in vbs:
        raise SystemExit(
            f"the manifest has no validation for seed {demo_seed}; "
            f"it carries {sorted(vbs)}")
    return {(int(v["class_idx"]), v["source_text"])
            for v in vbs[str(demo_seed)]}


def verify_prefix(prefix_blocks, calibration_prompt):
    """Check the rendered prefix against a real calibration prompt.

    The calibration file stores whole rendered prompts, so its demo blocks are
    exactly what the model saw in the published runs. Splitting one on the blank
    line and dropping the trailing query block gives those blocks back, and they
    must equal what `build_prefix` produced. This is the gate that catches a
    drift between this renderer and the task's -- a missing space, a different
    separator, a label rendered from the wrong table -- none of which would
    raise anywhere else.
    """
    got = calibration_prompt.split("\n\n")[:-1]
    if list(prefix_blocks) == got:
        return True, f"{len(got)} demo blocks match the calibration prompt"
    if len(prefix_blocks) != len(got):
        return False, (f"rendered {len(prefix_blocks)} demo blocks, the "
                       f"calibration prompt has {len(got)}")
    for i, (a, b) in enumerate(zip(prefix_blocks, got)):
        if a != b:
            return False, (f"demo block {i} differs.\n  rendered:    {a!r}\n"
                           f"  calibration: {b!r}")
    return False, "blocks compare unequal but no difference found"


# ==========================================================================
# carriers
# ==========================================================================
def load_carriers(path, carrier_impl, n_attn_heads, n_kv_heads):
    """Frozen carriers as {layer: [query heads]}, plus their KV groups.

    Refuses a partial KV group for the two native paths, because v_proj has no
    per-query-head slot there and a partial selection would silently widen.
    `gqa_head_realized` accepts any subset -- that is the point of it.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    heads = raw.get("heads", raw if isinstance(raw, list) else None)
    if heads is None:
        raise SystemExit(f"{path}: no 'heads' list (keys: {sorted(raw)})")
    by_layer = {}
    for lk in heads:
        l, k = int(lk[0]), int(lk[1])
        if not 0 <= k < n_attn_heads:
            raise SystemExit(f"{path}: head {k} out of range for "
                             f"{n_attn_heads} query heads")
        by_layer.setdefault(l, set()).add(k)
    by_layer = {l: sorted(v) for l, v in sorted(by_layer.items())}
    if not by_layer:
        raise SystemExit(f"{path}: empty carrier set")

    width = n_attn_heads // n_kv_heads
    groups = {l: sorted({k // width for k in ks}) for l, ks in by_layer.items()}
    if carrier_impl == "gqa_group_v":
        # the confirmatory L3.1 arm writes whole groups; record what that drags
        dragged = {l: sorted({q for g in groups[l]
                              for q in range(g * width, (g + 1) * width)}
                             - set(ks))
                   for l, ks in by_layer.items()}
        return by_layer, groups, dragged
    if carrier_impl == "mha_v" and width != 1:
        raise SystemExit(
            f"--carrier-impl mha_v needs MHA, but this model has "
            f"{n_attn_heads} query heads over {n_kv_heads} KV heads "
            f"({width} per group). Use gqa_group_v (section 3.1) or "
            "gqa_head_realized (section 3.2).")
    return by_layer, groups, {l: [] for l in by_layer}


# ==========================================================================
# building the writes
# ==========================================================================
def build_writes(source, arm, gamma, carriers_by_layer, *, setting, demo_seed,
                 carrier_impl, n_attn_heads, n_kv_heads, d_head, natural_raw):
    """What each layer receives, in the form its carrier implementation wants.

    Returns (native, realized):
      native   {layer: (rows, vals, heads)} for the v_proj patch path, where
               `vals` is full v_proj width and only the selected columns are
               read (see diagnostic_forward's hook);
      realized {layer: LayerWrite} for the post-repeat_kv path.

    Exactly one of the two is populated. The targets are identical either way --
    they come from the same sealed source and the same arm arithmetic -- so the
    two paths differ only in which query heads end up reading them, which is the
    distinction sections 3.1 and 3.2 draw.
    """
    native, realized = {}, {}
    width = n_attn_heads // n_kv_heads
    rows = source.label_rows
    for layer, heads in sorted(carriers_by_layer.items()):
        slots = sorted({h // width for h in heads})       # KV slots to target
        targets = {g: source.written(layer, arm, gamma, setting=setting,
                                     demo_seed=demo_seed, carrier=g)
                   for g in slots}
        if carrier_impl == "gqa_head_realized":
            # only the SELECTED query heads read the new value; their group
            # mates keep the natural one (section 3.2)
            realized[layer] = LayerWrite(
                rows, {int(h): targets[h // width] for h in sorted(heads)})
        else:
            # a whole KV slot is rewritten, so every query head sharing it moves
            vals = np.array(natural_raw[layer], dtype=np.float64, copy=True)
            vals = vals[rows]
            for g, t in targets.items():
                vals[:, g * d_head:(g + 1) * d_head] = t
            touched = sorted({h for g in slots
                              for h in group_members(g, n_attn_heads,
                                                     n_kv_heads)})
            native[layer] = (rows, vals, touched)
    return native, realized


def dragged_heads(carriers_by_layer, n_attn_heads, n_kv_heads):
    """Query heads a whole-group write moves that were not selected.

    Section 13.4(3) requires Method A-group to record these. They are not a
    side effect to be minimised -- under GQA they are what the operator IS --
    so they are written into the output rather than mentioned in a log line.
    """
    width = n_attn_heads // n_kv_heads
    out = {}
    for layer, heads in sorted(carriers_by_layer.items()):
        slots = sorted({h // width for h in heads})
        members = {h for g in slots
                   for h in group_members(g, n_attn_heads, n_kv_heads)}
        out[int(layer)] = sorted(members - set(heads))
    return out


# ==========================================================================
# one query
# ==========================================================================
def donors_for_query(arm, *, model, by_layer, n_attn, n_kv, ids, label_rows,
                     demo_classes, demo_ids, frozen=None, seed=None):
    """{(layer, kv_slot): [n_demos] donor index} for the exploratory arms.

    THE CARRIER KEY IS ALWAYS THE KV SLOT. `build_writes` asks the source for
    `carrier=g` with g = h // width under BOTH implementations -- the realized
    path still reads `targets[h // width]` -- so a donor map keyed on query
    heads would silently miss every lookup.

    frozen  the donor was chosen once from validation-averaged attention and
            is read, never recomputed. A missing cell raises: filling it from
            this query would be the dynamic arm under the frozen name.
    dynamic one capture forward on THIS query, then argmax within each class.
            The extra forward is why the dynamic arm costs ~1/6 more than the
            others; it is not shared with the natural forward because that one
            runs inside run_one_query, which is deliberately model-free.
    """
    width = int(n_attn) // int(n_kv)
    slots = {int(l): sorted({int(h) // width for h in hs})
             for l, hs in by_layer.items()}
    if arm == "maxattn_frozen":
        cell = (frozen or {}).get(str(seed))
        if not isinstance(cell, dict):
            raise SystemExit(f"the donor manifest has no seed {seed}")
        out = {}
        for l, gs in sorted(slots.items()):
            for g in gs:
                key = f"{l}:{g}"
                if key not in cell:
                    raise SystemExit(
                        f"the donor manifest has no entry for layer {l} "
                        f"carrier {g} of seed {seed}. Recomputing it here "
                        "would make this the dynamic arm.")
                d = np.asarray(cell[key], dtype=np.int64)
                if d.size != len(demo_ids):
                    raise SystemExit(
                        f"frozen donor {key} has {d.size} entries for "
                        f"{len(demo_ids)} demos; the manifest was built for a "
                        "different prefix")
                out[(l, g)] = d
        return out

    from tools.attention_capture import CaptureContext, label_row_attention
    from tools.exploratory_donor_arms import max_attention_donor
    import torch
    store = {}
    ctx = CaptureContext(model, sorted(slots), store)
    with ctx:
        with torch.no_grad():
            model(input_ids=torch.as_tensor(ids).reshape(1, -1)
                  .to(model.device))
    ctx.verify_ran()
    out = {}
    for l, gs in sorted(slots.items()):
        for g in gs:
            qh = list(range(g * width, (g + 1) * width))
            a = label_row_attention(store, l, label_rows, heads=qh)
            out[(l, g)] = max_attention_donor(demo_classes, a, demo_ids)
    return out


def run_one_query(forward, prompt_ids, label_rows, demo_classes, demo_ids,
                  carriers_by_layer, arm, gammas, *, setting, demo_seed,
                  carrier_impl, n_attn_heads, n_kv_heads, d_head,
                  donors=None):
    """Natural forward, then one patched forward per gamma. Model-agnostic.

    `forward(input_ids, patch_v=..., realized=...)` returns
    (v_raw_by_layer, answer_logits). Keeping the model behind that one callable
    is what lets the whole orchestration -- sealing the source, building the
    writes, the natural/patched ordering -- be tested on CPU with no model at
    all, which is where the ordering bugs would actually live.

    Returns {"natural": logits, "gamma": {g: logits}, "dragged": {...}}.
    """
    natural_raw, natural_logits = forward(prompt_ids)
    # DONORS ARE DATA, computed by the caller. Keeping the attention capture
    # out of here is what lets this whole orchestration stay testable on CPU
    # with no model, which is where the ordering bugs live.
    source = FixedSource(label_rows, demo_classes, demo_ids, donors=donors)
    width = n_attn_heads // n_kv_heads
    for layer, heads in sorted(carriers_by_layer.items()):
        if layer not in natural_raw:
            raise KeyError(f"the natural forward captured no v_proj output for "
                           f"layer {layer}; captured {sorted(natural_raw)}")
        raw = np.asarray(natural_raw[layer], dtype=np.float64)
        for g in sorted({h // width for h in heads}):
            source.add(layer, g, raw[label_rows, g * d_head:(g + 1) * d_head])
    source.seal()        # nothing may be recomputed from a patched state now

    out = {"natural": natural_logits,
           "gamma": {},
           "dragged": dragged_heads(carriers_by_layer, n_attn_heads,
                                    n_kv_heads)}
    for gamma in gammas:
        if arm == "natural" or (gamma == 0 and arm != "sham"):
            # gamma = 0 is the natural forward by construction (section 3), so
            # it is not re-run; sham still executes the patch branch, which is
            # the whole point of it (section 11 gate 3).
            out["gamma"][gamma] = natural_logits
            continue
        native, realized = build_writes(
            source, arm, gamma, carriers_by_layer, setting=setting,
            demo_seed=demo_seed, carrier_impl=carrier_impl,
            n_attn_heads=n_attn_heads, n_kv_heads=n_kv_heads, d_head=d_head,
            natural_raw=natural_raw)
        _raw, logits = forward(prompt_ids, patch_v=native or None,
                               realized=realized or None)
        out["gamma"][gamma] = logits
    return out


# ==========================================================================
# the model side
# ==========================================================================
def queries_for_run(manifest_path, split, seeds, *, freeze_manifest=None,
                   limit=None, text_of=None, expected_roles=None):
    """(queries_by_seed, faults) for one run. NO model, NO tokenizer.

    Split out because three defects lived in this logic in three consecutive
    rounds and none was catchable: the only way in was main(), and main()
    needs a GPU.

      * one load reused for every seed, so three blocks all came from seed
        42 while being labelled 42, 43 and 44;
      * a single-seed alias kept "for the census" that also fed the
        missing-query check, so 43's and 44's own absences surfaced as a
        KeyError deep in the forward loop;
      * --limit slicing that alias while the loop read the mapping, so the
        limit had no effect.

    THE SPLITS DIFFER, and this is where that is decided.

      validation   drawn per seed (14.0a); a run spans all three
      test_seed    drawn per seed (2.2), but `validate()` restricts a run to
                   ONE seed, since each seed's 250 are a different question
      test_common  ONE shared set that load_split returns whatever demo_seed
                   is passed, because holding the queries fixed under three
                   prefixes is what the L2 comparison needs

    The per-seed rule is PAIRING, not distinctness. "the three results must
    differ" is both too weak and too strong: a loop that returns 42's rows for
    43, 43's for 44 and 44's for 42 passes it while every output is labelled
    with the wrong seed, and nothing in an independent draw forbids two seeds
    from coinciding. So each seed's rows are compared, in order, against the
    manifest's OWN entry for that seed -- read here from the JSON rather than
    through load_split, so the two paths have to agree.

    `expected_roles` is forwarded to the test lock so the freeze is bound to
    the CARRIERS AND LABEL SPACE this run loads, not merely to some valid
    freeze. main() supplies them; a freeze that pins different ones is not
    evidence about this run.
    """
    from tools.build_query_manifest import load_split
    faults = []
    by_seed = {}
    for sd in seeds:
        by_seed[sd] = list(load_split(manifest_path, split, demo_seed=sd,
                                      freeze_manifest=freeze_manifest,
                                      expected_roles=expected_roles))
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if split == "test_common":
        # PAIRED AGAINST THE MANIFEST, not merely against each other. "the
        # three agree" is satisfied by three empty lists, and by three copies
        # of the wrong set; test_common is defined as every pool row, so that
        # is what it is compared to. load_split ignoring demo_seed here is the
        # point of the split -- the same queries under three prefixes is what
        # makes the L2 cross-seed comparison possible -- so the identity across
        # seeds is checked too.
        want = [e["query_id"] for e in man["entries"] if e.get("role") == "pool"]
        for sd in seeds:
            got = [q["query_id"] for q in by_seed[sd]]
            if got != want:
                where = next((i for i, (a, b) in enumerate(zip(got, want))
                              if a != b), min(len(got), len(want)))
                faults.append(
                    f"split test_common seed {sd}: got {len(got)} queries, the "
                    f"manifest has {len(want)} pool rows, first difference at "
                    f"index {where}. test_common is the WHOLE eligible test "
                    "split; a short or reordered set is a different experiment")
        ids0 = [q["query_id"] for q in by_seed[seeds[0]]]
        for sd in seeds[1:]:
            if [q["query_id"] for q in by_seed[sd]] != ids0:
                faults.append(
                    f"split test_common returned different queries for seeds "
                    f"{seeds[0]} and {sd}; it is defined as ONE shared set "
                    "and the L2 comparison depends on that")
    else:
        key = "validation_by_seed" if split == "validation" else "test_by_seed"
        # NO "manifest has no such seed" branch here: load_split reads this
        # same key first and raises ValueError, so the branch would be dead
        # code -- structurally unreachable, not merely hard to reach.
        want_all = man.get(key) or {}
        for sd in seeds:
            want = [q["query_id"] if isinstance(q, dict) else q
                    for q in want_all[str(sd)]]
            got = [q["query_id"] for q in by_seed[sd]]
            if got != want:
                where = next((i for i, (a, b) in enumerate(zip(got, want))
                              if a != b), min(len(got), len(want)))
                owner = next((o for o in want_all
                              if [q["query_id"] if isinstance(q, dict) else q
                                  for q in want_all[o]] == got), None)
                faults.append(
                    f"split {split} seed {sd}: got {len(got)} queries, the "
                    f"manifest records {len(want)}, first difference at "
                    f"index {where}"
                    + (f" -- these are seed {owner}'s rows, so the seeds are "
                       "permuted and every output would carry the wrong "
                       "label" if owner is not None else ""))
    if text_of is not None:
        missing = [(sd, q["query_id"]) for sd in seeds
                   for q in by_seed[sd] if q["query_id"] not in text_of]
        if missing:
            faults.append(
                f"{len(missing)} manifest queries were not found in either "
                f"split by content hash (first: {missing[:2]}). The manifest "
                "and the task disagree about the data; do not proceed.")
    if limit:
        by_seed = {sd: rows[:limit] for sd, rows in by_seed.items()}
    return by_seed, faults


class TorchForwardRunner:
    """One forward with v_proj capture and, optionally, a write.

    Deliberately NOT `diagnostic_forward._run_capture_forward`. That function
    also monkey-patches `self_attn.forward` to capture attention weights, which
    this measurement does not need, and which would have to coexist with the
    realized-write adapter's re-routing of the attention interface. Two
    interacting interventions on the same module, one of them unnecessary, is
    not worth the risk. The GQA column arithmetic IS reused --
    `v_patch_columns` carries the partial-group refusal -- so the part that is
    subtle is shared and only the plumbing is local.
    """

    def __init__(self, model, layers, *, allow_kv_group=False,
                 capture_o_proj=False):
        self.model = model
        self.layers = sorted(int(l) for l in layers)
        self.allow_kv_group = allow_kv_group
        # Per-head outputs are taken from o_proj's INPUT, which is
        # [B, N, d_model] = the heads concatenated, reshaped to [B, N, H, d_h].
        # Section 11 fixes that boundary, and it is the only place a per-query
        # head's contribution exists as a separate object -- which is what
        # "the unselected group-mates did not move" has to be checked on.
        self.capture_o_proj = capture_o_proj
        self.o_proj_in = {}
        cfg = model.config
        self.n_attn_heads = int(cfg.num_attention_heads)
        self.n_kv_heads = int(getattr(cfg, "num_key_value_heads",
                                      cfg.num_attention_heads))
        self.d_head = int(cfg.hidden_size) // self.n_attn_heads

    def __call__(self, input_ids, *, patch_v=None, realized=None):
        import torch
        from tools.diagnostic_forward import v_patch_columns
        from tools.realized_write import RealizedWriteContext

        captured, handles = {}, []

        def make_hook(layer_idx):
            def hook(_module, _inp, output):
                is_tensor = isinstance(output, torch.Tensor)
                out = output if is_tensor else output[0]
                raw = out[0]                       # [n_tokens, n_kv*d_head]
                if patch_v and layer_idx in patch_v:
                    pos, vals, heads = patch_v[layer_idx]
                    cols = v_patch_columns(heads, self.n_attn_heads,
                                           self.n_kv_heads, self.d_head,
                                           allow_kv_group=self.allow_kv_group)
                    raw = raw.clone()
                    v = torch.as_tensor(vals, device=raw.device, dtype=raw.dtype)
                    if cols is None:
                        raw[pos] = v
                    elif cols.numel():
                        p = torch.as_tensor(pos, dtype=torch.long,
                                            device=raw.device).reshape(-1)
                        c = cols.to(raw.device)
                        raw[p[:, None], c[None, :]] = v[:, c]
                    captured[layer_idx] = raw.detach().float().cpu().numpy()
                    out = raw.unsqueeze(0)
                    return out if is_tensor else (out,) + tuple(output[1:])
                captured[layer_idx] = raw.detach().float().cpu().numpy()
                return None
            return hook

        for l in self.layers:
            handles.append(self.model.model.layers[l].self_attn.v_proj
                           .register_forward_hook(make_hook(l)))

        if self.capture_o_proj:
            self.o_proj_in = {}

            def make_pre(layer_idx):
                def pre(_module, inputs):
                    x = inputs[0]
                    self.o_proj_in[layer_idx] = (
                        x[0].detach().float().cpu().numpy())
                    return None
                return pre

            for l in self.layers:
                handles.append(self.model.model.layers[l].self_attn.o_proj
                               .register_forward_pre_hook(make_pre(l)))
        try:
            ids = torch.as_tensor(input_ids, dtype=torch.long).reshape(1, -1)
            ids = ids.to(next(self.model.parameters()).device)
            if realized:
                with RealizedWriteContext(self.model, realized):
                    with torch.no_grad():
                        res = self.model(input_ids=ids)
            else:
                with torch.no_grad():
                    res = self.model(input_ids=ids)
        finally:
            for h in handles:
                h.remove()
        logits = res.logits[0, -1, :].detach().float().cpu().numpy()
        return captured, logits


# ==========================================================================
# section 13.4(3): the Method A half, on one real validation query
# ==========================================================================
def parse_gate_carriers(spec):
    """"8:4,5,6,7;16:20" -> {8: [4,5,6,7], 16: [20]}"""
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        layer, heads = part.split(":")
        out[int(layer)] = sorted({int(h) for h in heads.split(",")})
    if not out:
        raise ValueError(f"no carriers parsed from {spec!r}")
    return out


def gpu_gate(args):  # noqa: C901
    """Run the Method A equivalence gates against real weights.

    Section 13.4(3) wants this alongside the five model-side placebos; only the
    Method A half is here, so this closes part of that step and is reported as
    such.

    It deliberately drives the RUNNER's own functions -- TorchForwardRunner,
    build_writes, verify_prefix -- rather than a parallel implementation. A gate
    that exercises different code than the experiment proves nothing about the
    experiment.

    THE CARRIERS ARE ENGINEERING CARRIERS. Section 2.4's discovery has not run,
    so there is no frozen carrier set yet and this gate must not pretend to use
    one. `--gate-carriers` names them explicitly and they are recorded as
    engineering-only; the gate tests plumbing, not which heads matter.
    """
    import torch
    from tools.build_query_manifest import load_split
    from tools.icl_common import load_jsonl, run_provenance
    from tools.model_loader import load_model
    from tools.prereg_ids import content_hash
    from tools.prereg_task import doc_fields, load_task, prefix_demo_rows
    from tools.probe_kappa_matrix import segment_positions
    from transformers import AutoTokenizer

    ok = True
    findings = []

    def check(name, cond, detail=""):
        nonlocal ok
        line = f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}"
                                                             if detail else "")
        print(line, flush=True)
        findings.append({"check": name, "pass": bool(cond), "detail": detail})
        ok = ok and bool(cond)

    seed = int(str(args.demo_seeds).split(",")[0])
    carriers = parse_gate_carriers(args.gate_carriers)
    print("=" * 78)
    print("METHOD A GPU GATE (prereg 13.4(3), Method A half only)")
    print("=" * 78)
    print(f"  model {args.model} / {args.method} / bf16 / eager")
    print(f"  engineering carriers {carriers}  (NOT discovered carriers)")

    # The seed this gate runs on. Validation is drawn per prefix seed
    # (14.0a), so a seed-free load returned seed 42's queries whatever seed
    # the rest of the gate used -- and the prefix a few lines below IS built
    # from `seed`, so the query and the prompt came from different draws.
    queries = load_split(args.query_manifest, "validation", demo_seed=seed)
    q = queries[int(args.gate_query_index)]
    header, cal_rows = load_jsonl(Path(args.uuid_jsonl))
    abstract_labels = header["abstract_labels"]

    task = load_task(args.task, args.K, seed)
    text_of = build_text_lookup(task, args.task)
    if q["query_id"] not in text_of:
        raise SystemExit(f"validation query {q['query_id'][:12]} not found in "
                         "either split by content hash")

    reserved = manifest_reservation(args.query_manifest, demo_seed=seed)
    blocks = build_prefix(
        prefix_demo_rows(args.task, args.K, [seed],
                         excluded_docs=reserved)[seed],
        abstract_labels)
    if int(header.get("seed", -1)) == seed and cal_rows:
        okp, why = verify_prefix(blocks, cal_rows[0]["prompt"])
        check("the rendered prefix matches a real calibration prompt", okp, why)
    else:
        check("prefix verification available for this seed", False,
              f"calibration file is for seed {header.get('seed')}, gate uses "
              f"{seed}; run the gate on the calibration seed")

    prompt = render_prompt(blocks, text_of[q["query_id"]])
    tok = AutoTokenizer.from_pretrained(args.model)
    ids, seg, is_lab, demo_cls = segment_positions(prompt, tok, abstract_labels)
    rows = np.array([np.where((seg == d) & is_lab)[0][0]
                     for d in range(demo_cls.size)], dtype=np.int64)
    check("exactly one label row per demo",
          all(int(((seg == d) & is_lab).sum()) == 1
              for d in range(demo_cls.size)),
          f"{demo_cls.size} demos, {len(ids)} tokens")

    model = load_model(args.model, method=args.method,
                       attn_implementation="eager")
    n_attn = int(model.config.num_attention_heads)
    n_kv = int(getattr(model.config, "num_key_value_heads", n_attn))
    d_head = int(model.config.hidden_size) // n_attn
    width = n_attn // n_kv
    check("the model is GQA, as sections 3.1/3.2 assume", width > 1,
          f"{n_attn} query heads over {n_kv} KV heads ({width} per group)")

    runner = TorchForwardRunner(model, sorted(carriers), allow_kv_group=True,
                               capture_o_proj=True)
    demo_ids = [f"d{i:04d}" for i in range(demo_cls.size)]
    nat_raw, nat_logits = runner(ids)
    nat_o = {l: v.copy() for l, v in runner.o_proj_in.items()}

    src = FixedSource(rows, demo_cls, demo_ids)
    for layer, heads in sorted(carriers.items()):
        raw = np.asarray(nat_raw[layer], dtype=np.float64)
        for g in sorted({h // width for h in heads}):
            src.add(layer, g, raw[rows, g * d_head:(g + 1) * d_head])
    src.seal()

    def run(arm, gamma, impl, carr):
        native, realized = build_writes(
            src, arm, gamma, carr, setting="L3.1", demo_seed=seed,
            carrier_impl=impl, n_attn_heads=n_attn, n_kv_heads=n_kv,
            d_head=d_head, natural_raw=nat_raw)
        raw, log = runner(ids, patch_v=native or None,
                          realized=realized or None)
        return raw, log, dict(runner.o_proj_in)

    # ---- placebos: both implementations must reproduce natural -------------
    print("\nequivalence gates (sham must reproduce natural elementwise)")
    _r, sham_g, _o = run("sham", 1.0, "gqa_group_v", carriers)
    check("Method A-group sham reproduces natural ELEMENTWISE",
          np.array_equal(sham_g, nat_logits),
          f"max |diff| {np.abs(sham_g - nat_logits).max():.3e}")
    _r, sham_h, _o = run("sham", 1.0, "gqa_head_realized", carriers)
    check("Method A-head sham reproduces natural ELEMENTWISE",
          np.array_equal(sham_h, nat_logits),
          f"max |diff| {np.abs(sham_h - nat_logits).max():.3e}")

    # ---- the group write, and where it is allowed to land ------------------
    print("\nMethod A-group: the write hits the expected tensor boundary")
    g_raw, g_log, g_o = run("proto_loo", 1.0, "gqa_group_v", carriers)
    check("the arm is live (logits move)", not np.array_equal(g_log, nat_logits),
          f"max |diff| {np.abs(g_log - nat_logits).max():.4f}")
    # Boundary exactness can only hold at the FIRST patched layer. Section 3.1
    # says so explicitly: "the first affected layer's Q/K and attention must be
    # bitwise unchanged; attention drift in later layers caused by the early
    # residual change is the model's response to this native per-layer
    # intervention". Applying the exactness criterion to every layer of a
    # multi-layer patch demands something no correct implementation can deliver
    # -- working rules rule 11b. Each layer is therefore checked IN ISOLATION.
    for layer, heads in sorted(carriers.items()):
        slots = sorted({h // width for h in heads})
        solo_raw, _sl, _so = run("proto_loo", 1.0, "gqa_group_v",
                                 {layer: heads})
        diff = np.asarray(solo_raw[layer]) != np.asarray(nat_raw[layer])
        want = np.zeros_like(diff)
        for g in slots:
            want[np.ix_(rows, range(g * d_head, (g + 1) * d_head))] = True
        contained = bool((diff <= want).all())
        moved_rows = sorted(set(np.flatnonzero(diff.any(axis=1)).tolist()))
        check(f"layer {layer} patched ALONE: v_proj changed exactly at the "
              f"label rows of KV slot(s) {slots}",
              contained and bool(diff.any()),
              f"{int(diff.sum())} entries changed"
              # "within", not "=": the criterion is containment. A written
              # value can round to the natural one in bf16, so slightly fewer
              # than rows x cols entries differ and that is not a miss.
              + (f" within the {len(rows)} x {len(slots) * d_head} = "
                 f"{len(rows) * len(slots) * d_head} target entries"
                 if contained else
                 f", but {int((diff & ~want).sum())} of them fall OUTSIDE the "
                 f"target; rows touched {moved_rows[:4]}"
                 f"{'...' if len(moved_rows) > 4 else ''}"))

    # With every carrier patched at once, the later layers MUST drift: their
    # v_proj reads an already-changed residual stream. Asserting it rather than
    # tolerating it is the difference between a documented model response and
    # an unnoticed leak.
    first = min(carriers)
    later = [l for l in sorted(carriers) if l != first]
    if later:
        d_first = np.asarray(g_raw[first]) != np.asarray(nat_raw[first])
        w_first = np.zeros_like(d_first)
        for g in sorted({h // width for h in carriers[first]}):
            w_first[np.ix_(rows, range(g * d_head, (g + 1) * d_head))] = True
        check(f"multi-layer patch: the FIRST carrier layer ({first}) is still "
              "exact", bool((d_first <= w_first).all()),
              "nothing upstream has perturbed it")
        for l in later:
            d_l = np.asarray(g_raw[l]) != np.asarray(nat_raw[l])
            beyond = int((d_l.any(axis=1)).sum()) - len(rows)
            check(f"...and layer {l} DOES drift beyond the label rows, which "
                  "section 3.1 calls the model's response",
                  beyond > 0,
                  f"{int(d_l.any(axis=1).sum())} of {d_l.shape[0]} rows moved; "
                  "an early patch changes the residual stream every later "
                  "token reads")
    dragged = dragged_heads(carriers, n_attn, n_kv)
    check("GQA heads dragged in by the group write are recorded",
          isinstance(dragged, dict),
          f"{ {l: v for l, v in dragged.items() if v} }")

    # ---- the head write: group-mates must not move -------------------------
    print("\nMethod A-head: unselected group-mates stay bitwise unchanged")
    one_layer = sorted(carriers)[0]
    one_head = sorted(carriers[one_layer])[0]
    grp = one_head // width
    mates = [h for h in range(grp * width, (grp + 1) * width) if h != one_head]
    solo = {one_layer: [one_head]}
    for layer, heads in sorted(carriers.items()):
        if layer != one_layer:
            solo[layer] = heads
    h_raw, h_log, h_o = run("proto_loo", 1.0, "gqa_head_realized",
                            {one_layer: [one_head]})
    check("the arm is live (logits move)", not np.array_equal(h_log, nat_logits),
          f"max |diff| {np.abs(h_log - nat_logits).max():.4f}")
    check(f"layer {one_layer}: v_proj is BITWISE unchanged -- the write is "
          "after repeat_kv, so it cannot appear here",
          np.array_equal(np.asarray(h_raw[one_layer]),
                         np.asarray(nat_raw[one_layer])),
          "this is what separates 3.2 from 3.1 at the tensor level")
    ho = h_o[one_layer].reshape(-1, n_attn, d_head)
    no = nat_o[one_layer].reshape(-1, n_attn, d_head)
    check(f"layer {one_layer}: head {one_head}'s own contribution DID change",
          not np.array_equal(ho[:, one_head], no[:, one_head]),
          f"max |diff| {np.abs(ho[:, one_head] - no[:, one_head]).max():.4f}")
    check(f"layer {one_layer}: its group-mates {mates} are BITWISE unchanged",
          all(np.array_equal(ho[:, m], no[:, m]) for m in mates),
          "section 3.2's defining property, on real weights")
    others = [h for h in range(n_attn) if h != one_head]
    check("...and so is every other head in the layer",
          np.array_equal(ho[:, others], no[:, others]),
          f"{len(others)} heads unchanged")
    g_o_l = g_o[one_layer].reshape(-1, n_attn, d_head)
    check("...whereas the GROUP write did move those same group-mates, so the "
          "check discriminates",
          not all(np.array_equal(g_o_l[:, m], no[:, m]) for m in mates))

    out = {"spec": "prereg_method_A.md 13.4(3), Method A half",
           "model": args.model, "method": args.method,
           "query_id": q["query_id"], "demo_seed": seed,
           "engineering_carriers": {str(k): v for k, v in carriers.items()},
           "carriers_are_discovered": False,
           "gqa_dragged_heads": {str(k): v for k, v in dragged.items()},
           "n_tokens": int(len(ids)), "n_demos": int(demo_cls.size),
           "findings": findings, "all_pass": bool(ok),
           "provenance": run_provenance()}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\n  [output] {args.out}")
    print()
    if not ok:
        print("VERDICT: FAILED -- 13.4(3) is not satisfied for Method A.")
        return 2
    print("VERDICT: PASS -- Method A-group and Method A-head reproduce natural "
          "under sham, hit the expected tensor boundary when live, and the "
          "head write leaves its group-mates bitwise unchanged. This is the "
          "Method A HALF of 13.4(3); the five model-side placebos still need "
          "the baseline runners.")
    return 0


# ==========================================================================
# self-test: section 11 gate 5
# ==========================================================================
def self_test() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {detail}" if detail else ""))
        ok = ok and bool(cond)

    print("gate 5  fixed source: early patches must not move later targets")
    rng = np.random.default_rng(7)
    labels = np.array([5, 5, 9, 9, 7, 7])
    ids = [f"d{i}" for i in range(6)]
    rows = np.array([2, 6, 11, 15, 19, 24])
    natural = {l: rng.standard_normal((6, 8)) for l in (3, 10, 17)}

    fs = FixedSource(rows, labels, ids)
    for l, v in natural.items():
        fs.add(l, 0, v)
    fs.seal()
    before = {l: fs.written(l, "proto_loo", 1.0, setting="L2", demo_seed=42,
                            carrier=0) for l in natural}

    # a later forward, with layer 3 patched, produces DIFFERENT values at
    # layers 10 and 17. This is what a live-recomputed prototype would consume.
    perturbed = {l: v + rng.standard_normal(v.shape) * 0.5
                 for l, v in natural.items() if l > 3}
    check("the stub really does move (the arm can discriminate)",
          all(not np.allclose(perturbed[l], natural[l]) for l in perturbed))
    for l, v in perturbed.items():
        try:
            fs.add(l, 0, v)
            check(f"layer {l} refresh refused", False, "it was accepted")
        except RuntimeError:
            pass
    check("a sealed source refuses every refresh",
          fs.n_refresh_attempts == len(perturbed),
          f"{fs.n_refresh_attempts} attempts, all raised")
    after = {l: fs.written(l, "proto_loo", 1.0, setting="L2", demo_seed=42,
                           carrier=0) for l in natural}
    check("targets at layers 10 and 17 are unchanged after patching layer 3",
          all(np.array_equal(before[l], after[l]) for l in natural))
    check("...and they equal the NATURAL-source targets, not the perturbed ones",
          all(np.array_equal(
              after[l], written_label_rows("proto_loo", 1.0, natural[l], labels,
                                           ids, setting="L2", demo_seed=42,
                                           layer=l, carrier=0))
              for l in natural))
    live = {l: written_label_rows("proto_loo", 1.0, perturbed[l], labels, ids,
                                  setting="L2", demo_seed=42, layer=l,
                                  carrier=0) for l in perturbed}
    check("...which differ from what a live-recomputed prototype would write",
          all(not np.array_equal(after[l], live[l]) for l in perturbed),
          "so the check distinguishes the two operators")

    check("v0() returns a copy, so a caller cannot mutate the source",
          (lambda a: (a.__setitem__((0, 0), 1e9),
                      np.array_equal(fs.v0(3), natural[3]))[1])(fs.v0(3)))
    try:
        FixedSource(rows, labels, ids).v0(3, 0)
        check("reading before seal is refused", False, "it was allowed")
    except RuntimeError:
        check("reading before seal is refused", True)
    try:
        FixedSource(rows, labels, ids).seal()
        check("sealing an empty source is refused", False, "it was allowed")
    except RuntimeError:
        check("sealing an empty source is refused", True)
    try:
        FixedSource(rows[:3], labels, ids)
        check("mismatched demo bookkeeping is refused", False, "allowed")
    except ValueError:
        check("mismatched demo bookkeeping is refused", True)

    print("\n  position arms use the LABEL row's displacement, not their own")
    d = fs.delta(10)
    pos_v0 = rng.standard_normal((6, 8))
    pr = written_position_rows("pos_raw", 1.0, pos_v0, d, cast_bf16=False)
    check("pos_raw injects the label-row Delta at the frozen position",
          np.abs((pr - pos_v0) - d).max() < 1e-12)
    s = dose_scale(np.full(6, 0.3), np.full(6, 0.1))
    pd = written_position_rows("pos_dose", 1.0, pos_v0, d, dose=s,
                               cast_bf16=False)
    check("pos_dose scales it by the attention ratio",
          np.abs((pd - pos_v0) - 3.0 * d).max() < 1e-9, "s = 0.3/0.1 = 3")

    # ======================================================================
    print("\nprompt rendering, checked against a real calibration prompt")
    # ======================================================================
    labels = ["ZZ", "QQ", "WW"]
    pairs = [(0, "who wrote hamlet"), (2, "how far is the moon"),
             (1, "what is a quark")]
    blocks = build_prefix(pairs, labels)
    check("a demo block is rendered in the task's format",
          blocks[0] == "Question: who wrote hamlet\nType: ZZ", repr(blocks[0]))
    check("the label comes from the class's entry, not the demo's position",
          blocks[1].endswith("Type: WW") and blocks[2].endswith("Type: QQ"))
    prompt = render_prompt(blocks, "when did rome fall")
    check("the query block carries no label and ends at 'Type:'",
          prompt.endswith("Question: when did rome fall\nType:"))
    check("blocks are separated by a blank line, as the task joins them",
          prompt.count("\n\n") == 3)
    okp, why = verify_prefix(blocks, prompt)
    check("the prefix verifier accepts a matching calibration prompt", okp, why)
    for label, spoilt in (
            ("a changed separator",
             prompt.replace("Type: ZZ", "Type:ZZ")),
            ("a dropped demo", "\n\n".join(prompt.split("\n\n")[1:])),
            ("a relabelled demo", prompt.replace("Type: WW", "Type: QQ"))):
        bad_ok, bad_why = verify_prefix(blocks, spoilt)
        check(f"...and rejects {label}", not bad_ok, bad_why.split("\n")[0])

    # ======================================================================
    print("\nforward-loop orchestration (stub model, CPU)")
    # ======================================================================
    # 32 query heads over 8 KV groups, like L3.1. The stub's logits are a
    # deterministic function of the value rows the model would read, so a write
    # that never reaches the forward cannot pass unnoticed.
    N_ATTN, N_KV, D_HEAD, N_TOK, VOCAB = 32, 8, 4, 40, 60
    RAW_W = N_KV * D_HEAD
    base = np.arange(N_TOK * RAW_W, dtype=np.float64).reshape(N_TOK, RAW_W)
    base = (base % 7) - 3.0
    rows_s = np.array([4, 9, 14, 19, 24, 29], dtype=np.int64)
    cls_s = np.array([5, 5, 9, 9, 7, 7])
    ids_s = [f"d{i}" for i in range(6)]
    carriers = {3: [4, 5, 6, 7], 11: [20]}          # group 1; group 5
    calls = []

    def stub(input_ids, patch_v=None, realized=None):
        raw = {l: base.copy() for l in carriers}
        if patch_v:
            for l, (pos, vals, heads) in patch_v.items():
                raw[l][np.asarray(pos)] = np.asarray(vals)
        if realized:
            for l, w in realized.items():
                for h, v in w.values.items():
                    g = h // (N_ATTN // N_KV)
                    raw[l][np.asarray(w.rows),
                           g * D_HEAD:(g + 1) * D_HEAD] = np.asarray(v)
        calls.append({"patch_v": patch_v, "realized": realized})
        z = np.zeros(VOCAB)
        # A ROW-WEIGHTED sum, not a plain one. Sum_i mu_{-i} = Sum_i v_i
        # exactly, so a plain sum is invariant to a leave-one-out write and the
        # stub would report "no change" for a write that landed perfectly.
        # (The fixture caught that; it is a real identity, not a stub artefact.)
        wgt = np.arange(1, rows_s.size + 1, dtype=np.float64)[:, None]
        z[:8] = [float((raw[l][rows_s] * wgt).sum())
                 for l in sorted(carriers)] * 4
        return raw, z

    calls.clear()
    res = run_one_query(stub, np.arange(N_TOK), rows_s, cls_s, ids_s, carriers,
                        "proto_loo", [0.0, 0.5, 1.0], setting="L3.1",
                        demo_seed=42, carrier_impl="gqa_group_v",
                        n_attn_heads=N_ATTN, n_kv_heads=N_KV, d_head=D_HEAD)
    check("the natural forward runs first and exactly once",
          calls[0]["patch_v"] is None and calls[0]["realized"] is None)
    check("gamma=0 reuses the natural forward instead of re-running it",
          len(calls) == 3 and np.array_equal(res["gamma"][0.0], res["natural"]),
          f"{len(calls)} forwards for 3 gammas: natural + 0.5 + 1.0")
    check("gamma>0 actually changes the logits",
          not np.array_equal(res["gamma"][1.0], res["natural"])
          and not np.array_equal(res["gamma"][0.5], res["gamma"][1.0]))
    check("the write lands on the label rows only",
          all(np.array_equal(calls[1]["patch_v"][3][0], rows_s)
              for _ in (0,)))
    check("a whole-group write covers the group's four query heads",
          calls[1]["patch_v"][3][2] == [4, 5, 6, 7])
    check("...and the dragged heads are recorded, not just implied",
          res["dragged"] == {3: [], 11: [21, 22, 23]},
          f"{res['dragged']} -- selecting head 20 alone moves 21-23 too")

    # calls = [natural, gamma 0.5, gamma 1.0]; compare like with like
    res_native_vals = calls[2]["patch_v"][3][1]

    # the realized path writes the same targets to different readers
    calls.clear()
    res_h = run_one_query(stub, np.arange(N_TOK), rows_s, cls_s, ids_s,
                          carriers, "proto_loo", [1.0], setting="L3.1",
                          demo_seed=42, carrier_impl="gqa_head_realized",
                          n_attn_heads=N_ATTN, n_kv_heads=N_KV, d_head=D_HEAD)
    check("the realized path uses LayerWrite, not a v_proj patch",
          calls[1]["patch_v"] is None and calls[1]["realized"] is not None)
    check("...writing exactly the selected query heads",
          sorted(calls[1]["realized"][11].values) == [20]
          and sorted(calls[1]["realized"][3].values) == [4, 5, 6, 7])
    vals3 = calls[1]["realized"][3].values
    check("...and the four heads of a full group all receive the SAME target",
          all(np.array_equal(vals3[4], vals3[h]) for h in (5, 6, 7)),
          "section 3.2's target is indexed by KV group, not by query head")
    g3 = 4 // (N_ATTN // N_KV)          # heads 4-7 read KV group 1
    native_t = np.asarray(res_native_vals)[:, g3 * D_HEAD:(g3 + 1) * D_HEAD]
    check("...identical to what the native group patch wrote, so the two "
          "implementations differ only in who reads it",
          np.allclose(vals3[4], native_t))

    calls.clear()
    run_one_query(stub, np.arange(N_TOK), rows_s, cls_s, ids_s, carriers,
                  "sham", [1.0], setting="L3.1", demo_seed=42,
                  carrier_impl="gqa_group_v", n_attn_heads=N_ATTN,
                  n_kv_heads=N_KV, d_head=D_HEAD)
    check("sham still executes the patch branch (gate 3)",
          len(calls) == 2 and calls[1]["patch_v"] is not None)
    check("...and writes the natural values, so nothing changes",
          np.allclose(calls[1]["patch_v"][3][1], base[rows_s]))

    calls.clear()
    nat = run_one_query(stub, np.arange(N_TOK), rows_s, cls_s, ids_s, carriers,
                        "natural", [0.0, 1.0], setting="L3.1", demo_seed=42,
                        carrier_impl="gqa_group_v", n_attn_heads=N_ATTN,
                        n_kv_heads=N_KV, d_head=D_HEAD)
    check("the natural arm never patches at all",
          len(calls) == 1 and np.array_equal(nat["gamma"][1.0], nat["natural"]))

    try:
        run_one_query(stub, np.arange(N_TOK), rows_s, cls_s, ids_s, {99: [0]},
                      "proto_loo", [1.0], setting="L3.1", demo_seed=42,
                      carrier_impl="gqa_group_v", n_attn_heads=N_ATTN,
                      n_kv_heads=N_KV, d_head=D_HEAD)
        check("a carrier layer the forward never captured is refused", False)
    except KeyError:
        check("a carrier layer the forward never captured is refused", True)

    print()
    if not ok:
        print("SELF-TEST FAILED -- section 13.4(2) blocks GPU work until green.")
        return 2
    print("gate 5 and the forward-loop orchestration pass. The fixed source is "
          "sealed after the natural forward and refuses refreshes; the natural "
          "forward runs once and gamma=0 reuses it; sham executes the patch "
          "branch; and the group write's dragged heads are recorded rather "
          "than implied.")
    return 0


# ==========================================================================
# CLI
# ==========================================================================
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=MODES, default="clean")
    ap.add_argument("--arm", choices=ALL_ARMS)
    ap.add_argument("--donor-manifest",
                    help="frozen donors for --arm maxattn_frozen, written by "
                         "tools/probe_donor_concentration.py --write-donors. "
                         "EXPLORATORY; not one of section 4's eleven arms")
    ap.add_argument("--gamma", default=",".join(str(g) for g in GAMMA_GRID))
    ap.add_argument("--carrier-impl", choices=CARRIER_IMPLS)
    ap.add_argument("--carrier-json")
    ap.add_argument("--demo-seeds", default="42,43,44")
    ap.add_argument("--discovery-manifest")
    ap.add_argument("--query-manifest")
    ap.add_argument("--label-space",
                    help="frozen per-model label space "
                         "(tools/build_label_space.py). Candidate token ids are "
                         "tokenizer-specific and no longer live in the query "
                         "manifest; L2 uses the Llama-2 space, the L3.1 method "
                         "main table the Llama-3.1 one.")
    ap.add_argument("--freeze-manifest")
    ap.add_argument("--split", default="validation",
                    choices=("validation", "test_common", "test_seed"),
                    help="validation = each seed's own 144, drawn per "
                         "prefix seed since 14.0a (gamma* lives here); "
                         "test_common = all remaining, L2 mechanism setting; "
                         "test_seed = this demo seed's 250 PCW draws, L3.1 "
                         "method main table and H6")
    ap.add_argument("--allow-kv-group", action="store_true")
    ap.add_argument("--capture-resid", action="store_true")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--method", default="vanilla", choices=("vanilla",
                                                            "selfextend"))
    ap.add_argument("--task", default="trec_fine_per_class")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--uuid-jsonl",
                    help="calibration jsonl; its HEADER supplies "
                         "abstract_labels and label_token_ids. Only the header "
                         "is read -- the rows store rendered prompts indexed by "
                         "a shuffled doc_idx, which is not the manifest's index "
                         "space (see tools/prereg_task.py).")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug only; never used for a registered run")
    ap.add_argument("--self-test", action="store_true",
                    help="section 11 gate 5 plus the forward-loop "
                         "orchestration; no model and no GPU")
    ap.add_argument("--gpu-gate", action="store_true",
                    help="section 13.4(3), Method A half: one validation query "
                         "on real weights. Needs a GPU and the model.")
    ap.add_argument("--gate-carriers", default="8:4,5,6,7;16:20",
                    help="engineering carriers for --gpu-gate, "
                         "'layer:head,head;layer:head'. NOT discovered "
                         "carriers -- section 2.4 discovery has not run.")
    ap.add_argument("--gate-query-index", type=int, default=0,
                    help="which validation query to use for --gpu-gate")
    return ap


def validate(args):
    """Refuse impossible or unregistered combinations before loading a model."""
    problems = []
    gammas = [float(g) for g in str(args.gamma).split(",") if g != ""]
    unregistered = [g for g in gammas if g not in GAMMA_GRID]
    if unregistered:
        problems.append(f"gamma values {unregistered} are not on the "
                        f"preregistered grid {list(GAMMA_GRID)}")
    for name in ("arm", "carrier_impl", "carrier_json", "query_manifest",
                 "label_space", "uuid_jsonl", "out"):
        if not getattr(args, name):
            problems.append(f"--{name.replace('_', '-')} is required")
    if args.arm in DONOR_ARMS:
        # ⚠ VALIDATION ONLY. These are not section 4 arms, so nothing about
        # them is frozen before test and a test forward with one would be an
        # unregistered arm on a one-shot split. Refused rather than warned.
        if args.split != "validation":
            problems.append(
                f"--arm {args.arm} is EXPLORATORY (not one of section 4's "
                f"eleven) and may only run on validation; --split "
                f"{args.split} would put an unregistered arm on a one-shot "
                "split")
        if args.arm == "maxattn_frozen" and not args.donor_manifest:
            problems.append(
                "--arm maxattn_frozen needs --donor-manifest: the donor is "
                "chosen ONCE from validation-averaged attention, and without "
                "the file it would be recomputed per query, which is the "
                "dynamic arm under the frozen name")
        if args.arm == "maxattn_dynamic" and args.donor_manifest:
            problems.append(
                "--arm maxattn_dynamic must not be given --donor-manifest; "
                "its donor comes from each query's own attention and a frozen "
                "file would silently override that")
    if args.arm in ("pos_raw", "pos_dose") and not args.discovery_manifest:
        problems.append("the position arms need --discovery-manifest: section "
                        "4.2 freezes p_{l,i} from the VALIDATION natural "
                        "forward, so it cannot be chosen at run time")
    if args.carrier_impl == "gqa_group_v" and not args.allow_kv_group:
        problems.append(
            "--carrier-impl gqa_group_v writes whole KV groups, so it moves "
            "every query head in the group. Pass --allow-kv-group to say that "
            "is intended, and report the result as KV-group-selective")
    if args.split.startswith("test") and not args.freeze_manifest:
        problems.append(f"--split {args.split} requires --freeze-manifest "
                        "(section 2.1's test lock; see "
                        "tools/build_query_manifest.load_split)")
    # ON TEST, GAMMA COMES FROM THE FREEZE. The lock pinned a gamma artifact
    # by hash and nothing read it, while --gamma still supplied the value --
    # so after the freeze one could change gamma* or sweep the entire grid on
    # test with the lock verifying clean. Section 3 says gamma* is ONE frozen
    # number per setting; that is enforced here, against the file the freeze
    # registers.
    elif args.split.startswith("test"):
        from tools.build_query_manifest import (attribution_blockers,
                                                freeze_gamma_path,
                                                load_frozen_gammas)
        # The freeze records a commit rather than source hashes, so the run
        # has to be at a commit for that to mean anything.
        problems += attribution_blockers()
        gpath, why = freeze_gamma_path(args.freeze_manifest)
        if gpath is None:
            problems.append(f"--split {args.split}: {why}")
        else:
            frozen, gfaults = load_frozen_gammas(gpath)
            problems += gfaults
            key, why_key = gamma_key_for(args.arm, args.carrier_impl)
            if why_key:
                problems.append(f"--split {args.split}: {why_key}")
            elif not gfaults:
                # `None` means the arm writes nothing, so its only defensible
                # gamma is 0.0 -- not "any gamma is fine".
                want = 0.0 if key is None else frozen[key]
                what = "0.0 (this arm writes nothing)" if key is None \
                    else f"the frozen {key}={want}"
                if len(gammas) != 1:
                    problems.append(
                        f"--split {args.split} takes exactly ONE gamma, "
                        f"{what}; {len(gammas)} were given. Section 3 freezes "
                        "gamma* before any test prediction, so a grid on test "
                        "is a grid chosen after seeing test")
                elif gammas[0] != want:
                    problems.append(
                        f"--gamma {gammas[0]} is not {what}"
                        + (f", recorded in {gpath}" if key else ""))
    if args.split == "test_seed" and len(str(args.demo_seeds).split(",")) > 1:
        problems.append(
            "--split test_seed runs ONE demo seed at a time: each seed has its "
            "own 250 queries, so a single run cannot span seeds without "
            "silently concatenating unrelated query sets. Launch one job per "
            "seed and combine with the aggregate clustered test "
            "(tools/method_a_stats.decide_aggregate), which clusters on unique "
            "query_id because the draws overlap")
    if args.mode == "corrupt-targeting" and args.arm == "natural":
        problems.append("corrupt-targeting with the natural arm writes nothing "
                        "and measures nothing")
    return problems, gammas


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if args.gpu_gate:
        missing = [f"--{n.replace('_', '-')}" for n in
                   ("query_manifest", "uuid_jsonl") if not getattr(args, n)]
        if missing:
            print("REFUSING TO RUN: --gpu-gate needs " + ", ".join(missing))
            return 2
        return gpu_gate(args)

    problems, gammas = validate(args)
    if problems:
        print("REFUSING TO RUN:")
        for p in problems:
            print(f"  - {p}")
        return 2

    from tools.build_query_manifest import load_split

    seeds = [int(s) for s in args.demo_seeds.split(",")]
    manifest = json.loads(Path(args.query_manifest).read_text(encoding="utf-8"))
    if manifest.get("candidate_space"):
        raise SystemExit(
            f"{args.query_manifest} still carries 'candidate_space'. Candidate "
            "token ids are tokenizer-specific and were removed from this "
            "model-independent file on 2026-09-01; rebuild it and pass "
            "--label-space instead.")
    # The tokenizer is loaded FIRST so the label space can be checked against
    # the live object, not merely against a model string. Two checkouts of the
    # same model can tokenize differently, and a name comparison would not see
    # it -- which is the whole reason provenance hashes exist.
    from transformers import AutoTokenizer

    from tools.label_space import FrozenLabelSpace
    tok = AutoTokenizer.from_pretrained(args.model)
    lspace = FrozenLabelSpace.load(args.label_space, model=args.model,
                                   query_manifest=args.query_manifest,
                                   tokenizer=tok)
    cand = {"n_candidates": len(lspace.eligible_classes),
            "classes": lspace.eligible_classes,
            "token_ids": lspace.candidate_token_ids}

    # Loaded through queries_for_run so the same code an arm drives is the
    # code that runs. The missing check and --limit are applied there too,
    # once text_of exists.
    # The freeze is bound to the carriers and label space THIS run loads. A
    # valid freeze built against other artifacts says nothing about this run,
    # and the runner reads args.carrier_json / args.label_space regardless of
    # what the freeze pins.
    _roles = {"carriers": args.carrier_json, "label_space": args.label_space}
    queries_by_seed, _qfaults = queries_for_run(
        args.query_manifest, args.split, seeds,
        freeze_manifest=args.freeze_manifest, expected_roles=_roles)
    if _qfaults:
        raise SystemExit("REFUSING TO RUN:\n  " + "\n  ".join(_qfaults))
    # NO single-seed alias. One held "for the census" here and reached the
    # missing-query check and --limit; the same shape reached the
    # disjointness assertions in build_query_manifest. A convenience alias
    # for one seed's data has no safe scope in code that handles three.
    print("=" * 78)
    print(f"METHOD A  arm={args.arm}  carrier={args.carrier_impl}  "
          f"mode={args.mode}")
    print("=" * 78)
    _n_by = {sd: len(queries_by_seed[sd]) for sd in seeds}
    _n_distinct = len({q["query_id"] for sd in seeds
                       for q in queries_by_seed[sd]})
    # THREE splits, three different relationships to the seed, and the
    # census has to say which. "every split is drawn per seed" replaced one
    # wrong claim with another: load_split IGNORES demo_seed for test_common
    # and returns the single shared pool, which is the whole point of that
    # split -- the same queries under three prefixes is what makes the
    # cross-seed comparison possible.
    if args.split == "test_common":
        print(f"  split test_common: ONE shared set of {_n_distinct} queries, "
              f"scored under {len(seeds)} prefixes. Not a per-seed draw: "
              "load_split ignores demo_seed here, and holding the queries "
              "fixed is what the L2 cross-seed comparison needs.")
    elif args.split == "validation":
        print(f"  split validation: {_n_by} per seed, {_n_distinct} distinct "
              f"across {len(seeds)} seeds -- each seed draws its own 144 "
              "(14.0a), and the overlap is why inference clusters on "
              "query_id.")
    else:
        print(f"  split {args.split}: {_n_by} per seed, {_n_distinct} "
              f"distinct across {len(seeds)} seeds -- PCW, each seed draws "
              "its own 250 independently (2.2).")
    print(f"  candidate space: {lspace.summary()}")
    print(f"  gamma grid {gammas}")

    if args.arm in ("pos_raw", "pos_dose"):
        raise SystemExit(
            "the position arms are not implemented. Section 4.2 freezes "
            "p_{l,i} from the VALIDATION natural forward -- the non-label "
            "token with the highest mean answer attention, per layer per demo "
            "-- and that file does not exist yet. Running them against "
            "positions chosen at run time would be a different arm.")
    if args.mode == "corrupt-targeting":
        raise SystemExit(
            "--mode corrupt-targeting is not implemented. It is the G0/G1 "
            "targeting condition (section 5: demo seed 42, fixed f=30 "
            "corruption), whose corruption set and wrong-label assignment are "
            "specified there and not here; guessing them would produce a "
            "plausible run of the wrong experiment.")

    import torch
    from tools.icl_common import load_jsonl, run_provenance
    from tools.model_loader import load_model
    from tools.prereg_task import load_task, prefix_demo_rows
    from tools.probe_kappa_matrix import segment_positions

    header, cal_rows = load_jsonl(Path(args.uuid_jsonl))
    if header is None:
        raise SystemExit(f"{args.uuid_jsonl}: no header line")
    abstract_labels = header.get("abstract_labels")
    if not abstract_labels:
        raise SystemExit(
            f"{args.uuid_jsonl}: header has no abstract_labels. The runner "
            "inherits the label mapping rather than re-deriving it, because "
            "re-discovering labels would move the label tokens the frozen "
            "carriers were found against.")

    # The header this run actually loaded must be the one the frozen label
    # space was built from. Passing the wrong --uuid-jsonl is easy and quiet:
    # the file parses, the prompts render, and the only thing wrong is that its
    # surfaces map to different ids than the space says they do.
    from tools.label_space import assert_header_matches
    assert_header_matches(
        lspace, header, path=args.uuid_jsonl, model=args.model,
        query_manifest_sha256=hashlib.sha256(
            Path(args.query_manifest).read_bytes()).hexdigest())
    print("  calibration header matches the label space "
          "(surfaces, token ids, model, provenance, manifest hash)")
    cand_tokens = list(lspace.candidate_token_ids)
    cand_classes = list(lspace.eligible_classes)

    seed = seeds[0] if args.split == "test_seed" else None
    demo_seeds = [seed] if seed is not None else seeds

    task = load_task(args.task, args.K, demo_seeds[0])
    # BOTH splits: validation queries come from train, test queries from test
    text_of = build_text_lookup(task, args.task)
    # The missing check and --limit, on the same code path an arm drives.
    if args.limit:
        print(f"  [debug] --limit {args.limit}: NOT a registered run")
    queries_by_seed, _qfaults = queries_for_run(
        args.query_manifest, args.split, demo_seeds,
        freeze_manifest=args.freeze_manifest, limit=args.limit,
        text_of=text_of, expected_roles=_roles)
    if _qfaults:
        raise SystemExit("REFUSING TO RUN:\n  " + "\n  ".join(_qfaults))

    # Demonstrations come from train MINUS the validation reservation, exactly
    # as the calibration files were built. Without this the prefix rendered
    # here is a different demo set from the published one.
    prefixes, _res = {}, {}
    for _s in demo_seeds:
        _res[_s] = manifest_reservation(args.query_manifest, demo_seed=_s)
        prefixes[_s] = prefix_demo_rows(args.task, args.K, [_s],
                                        excluded_docs=_res[_s])[_s]
    print("  validation reservation per seed: "
          + ", ".join(f"seed {s}: {len(_res[s])}" for s in demo_seeds)
          + " train docs withheld from that seed's demonstrations")
    model = load_model(args.model, method=args.method,
                       attn_implementation="eager")
    n_attn = int(model.config.num_attention_heads)
    n_kv = int(getattr(model.config, "num_key_value_heads", n_attn))
    d_head = int(model.config.hidden_size) // n_attn
    _frozen_donors = None
    if args.donor_manifest:
        _frozen_donors = json.loads(
            Path(args.donor_manifest).read_text(encoding="utf-8"))["donors"]
        print(f"  frozen donors: {args.donor_manifest} "
              f"({len(_frozen_donors)} seeds)  [EXPLORATORY arm]")
    by_layer, groups, dragged = load_carriers(args.carrier_json,
                                              args.carrier_impl, n_attn, n_kv)
    print(f"  carriers: {sum(len(v) for v in by_layer.values())} heads over "
          f"layers {sorted(by_layer)}")
    if args.carrier_impl == "gqa_group_v":
        print(f"  GQA heads dragged in by whole-group writes: "
              f"{ {l: v for l, v in dragged.items() if v} }")

    runner = TorchForwardRunner(model, sorted(by_layer),
                                allow_kv_group=args.allow_kv_group)
    results, prefix_checks = [], {}

    for s in demo_seeds:
        blocks = build_prefix(prefixes[s], abstract_labels)
        # the renderer is checked against a real calibration prompt, for the
        # seed that file was built with; other seeds have no such file
        if int(header.get("seed", -1)) == int(s) and cal_rows:
            okp, why = verify_prefix(blocks, cal_rows[0]["prompt"])
            prefix_checks[str(s)] = why
            print(f"  prefix seed {s}: {'OK' if okp else 'MISMATCH'} -- {why}")
            if not okp:
                raise SystemExit(
                    "the rendered prefix does not match the calibration "
                    "prompt, so this runner would present the model with "
                    "different text than the published runs. Fix render_demo "
                    "before going further.")
        else:
            prefix_checks[str(s)] = "no calibration file for this seed"
            print(f"  prefix seed {s}: unverified (calibration file is for "
                  f"seed {header.get('seed')})")

        for qi, q in enumerate(queries_by_seed[s]):
            prompt = render_prompt(blocks, text_of[q["query_id"]])
            ids, seg, is_lab, demo_cls = segment_positions(prompt, tok,
                                                           abstract_labels)
            rows = np.array([np.where((seg == d) & is_lab)[0][0]
                             for d in range(demo_cls.size)], dtype=np.int64)
            n_lab = [int(((seg == d) & is_lab).sum())
                     for d in range(demo_cls.size)]
            if set(n_lab) != {1}:
                raise SystemExit(
                    f"query {q['query_id'][:12]}: demos with "
                    f"{sorted(set(n_lab))} label rows; exactly one is required "
                    "(single-token abstract labels).")
            _ids = [f"d{i:04d}" for i in range(demo_cls.size)]
            _donors = None
            if args.arm in DONOR_ARMS:
                _donors = donors_for_query(
                    args.arm, model=model, by_layer=by_layer, n_attn=n_attn,
                    n_kv=n_kv, ids=ids, label_rows=rows,
                    demo_classes=demo_cls, demo_ids=_ids,
                    frozen=_frozen_donors, seed=s)
            res = run_one_query(
                runner, ids, rows, demo_cls, _ids,
                by_layer, args.arm, gammas, setting="L3.1", demo_seed=s,
                carrier_impl=args.carrier_impl, n_attn_heads=n_attn,
                n_kv_heads=n_kv, d_head=d_head, donors=_donors)
            results.append({
                "query_id": q["query_id"], "class_idx": int(q["class_idx"]),
                "demo_seed": int(s),
                "natural": res["natural"][cand_tokens],
                "arm": np.stack([res["gamma"][g][cand_tokens] for g in gammas])})
            if (qi + 1) % 25 == 0:
                print(f"    seed {s}: {qi + 1}/{len(queries_by_seed[s])}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "spec": "prereg_method_A.md sections 3, 4, 11",
        "arm": args.arm, "carrier_impl": args.carrier_impl, "mode": args.mode,
        "split": args.split, "gammas": gammas, "demo_seeds": demo_seeds,
        "candidate_classes": cand_classes, "candidate_token_ids": cand_tokens,
        "label_space": args.label_space,
        "label_space_sha256": hashlib.sha256(
            Path(args.label_space).read_bytes()).hexdigest(),
        "tokenizer_provenance": lspace.data["provenance"],
        "carriers": {str(k): v for k, v in by_layer.items()},
        "gqa_dragged_heads": {str(k): v for k, v in dragged.items()},
        "prefix_verification": prefix_checks,
        "allow_kv_group": bool(args.allow_kv_group),
        "inputs": {
            "query_manifest": args.query_manifest,
            "query_manifest_sha256": hashlib.sha256(
                Path(args.query_manifest).read_bytes()).hexdigest(),
            "carrier_json": args.carrier_json,
            "carrier_json_sha256": hashlib.sha256(
                Path(args.carrier_json).read_bytes()).hexdigest(),
            "uuid_jsonl": args.uuid_jsonl,
            "uuid_jsonl_sha256": hashlib.sha256(
                Path(args.uuid_jsonl).read_bytes()).hexdigest()},
        "provenance": run_provenance(),
    }
    np.savez_compressed(
        out,
        query_id=np.array([r["query_id"] for r in results]),
        class_idx=np.array([r["class_idx"] for r in results]),
        demo_seed=np.array([r["demo_seed"] for r in results]),
        gammas=np.array(gammas),
        candidate_token_ids=np.array(cand_tokens),
        candidate_classes=np.array(cand_classes),
        logits_natural=np.stack([r["natural"] for r in results]),
        logits_arm=np.stack([r["arm"] for r in results]),
        meta=json.dumps(meta, ensure_ascii=False))
    print(f"\n  [output] {out}  ({len(results)} (query, seed) rows x "
          f"{len(gammas)} gamma x {len(cand_tokens)} candidates)")
    print(f"  sha256 {hashlib.sha256(Path(out).read_bytes()).hexdigest()[:16]}"
          "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
