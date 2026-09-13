"""The one natural forward every H6 baseline reads from. Torch imported lazily.

WHY THIS IS SHARED AND NOT PER-RUNNER. Section 1.1 requires every baseline to
use the same prompt constructor and the same candidate-logit extraction as the
method; six copies of that would be six chances to drift, and this project has
already paid for exactly that mistake once (working rules 2.6.2: two scripts each
carrying their own idea of the class space, and writing different answers).

So the prompt renderers are IMPORTED from the method probe rather than
reimplemented. That is not a section 0.2 violation: 0.2 forbids a baseline from
consuming the project's method ARTIFACTS -- TL-head rankings, kappa dumps,
prototype caches -- because the H6 comparison is only meaningful if the
baselines do not know what the method found. A prompt renderer is not a
finding; sharing it is what makes the comparison paired at all.

TWO KINDS OF BASELINE, AND ONLY ONE NEEDS A MODEL HOOK.

  * Score transforms (BC) are post-hoc: they read natural candidate logits and
    return calibrated ones. They need no forward of their own, and giving them
    one would just be another place for the prompt to drift.
  * Interventions (ZeroTuning, FV, TSLA, UniBias, Deep-Thinking) change the
    computation, so they subclass the forward and are handed the same prompts.

`NaturalReadout` therefore produces an artifact -- natural candidate logits per
(seed, query) -- that the transforms consume and the interventions compare
against for gate P2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.baselines.common import read_json  # noqa: E402


def _renderers():
    """The method's own prompt renderers, imported not copied (see docstring).

    Task-aware since the method line generalised beyond TREC: the prefix is
    rebuilt through tools.prompt_render, which formats through the same
    per-task table experiments/data_calibration.py renders with, and
    verify_prefix still compares the result with the calibration prompt.
    """
    from tools.probe_prototype_shrinkage import (manifest_reservation,
                                                 verify_prefix)
    from tools.prompt_render import TaskRenderer, build_doc_lookup
    return TaskRenderer, build_doc_lookup, manifest_reservation, verify_prefix


def calibration_path(calib_dir, task, k, seed, arm="uuid"):
    return Path(calib_dir) / f"calibration_{task}_K{k}_seed{seed}_{arm}.jsonl"


def load_calibration_header(path):
    """The header line of a calibration jsonl, and its first prompt.

    The header is what `assert_header_matches` re-checks against the label
    space, and the first prompt is what `verify_prefix` re-checks the rendered
    prefix against. Neither is optional: a baseline that rendered a slightly
    different prefix would still produce plausible numbers.
    """
    header, first = None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if header is None:
                header = rec
                continue
            first = rec
            break
    if header is None:
        raise ValueError(f"{path}: empty file, no header")
    return header, first


def build_prefixes(task_name, k, seeds, query_manifest, calib_dir,
                   label_space):
    """One verified prefix per demo seed, plus the query-document lookup.

    Every prefix is checked against the calibration prompt that the published
    run actually used. Section 1.1 says all queries share one fixed prefix per
    seed; if this renderer and the task's renderer ever disagree by a space,
    nothing downstream raises -- the numbers just quietly describe a different
    prompt.

    Returns (prefixes, doc_lookup, checks, render_prompt): `doc_lookup` maps a
    manifest query id to the task's DOCUMENT and `render_prompt(blocks, doc)`
    renders it in the task's own format, so a runner never touches a
    template. On TREC this is byte-identical to the old text renderer
    (tools/test_prompt_render.py).
    """
    from tools.label_space import assert_header_matches, file_sha256
    from tools.prereg_task import load_task, prefix_demo_docs

    (TaskRenderer, build_doc_lookup, manifest_reservation,
     verify_prefix) = _renderers()
    renderer = TaskRenderer(task_name)

    man = read_json(query_manifest)
    qm_sha = file_sha256(query_manifest)
    seeds = [int(s) for s in seeds]

    # ONE CALL PER SEED, each with its OWN reservation. `prefix_demo_docs`
    # takes a single `excluded_docs` and applies it to every seed it is given,
    # which was right while one validation set shaped all three prefixes.
    # Since 14.0a each seed reserved its own validation rows before its
    # demonstrations were drawn, so a single call -- with any reservation,
    # that seed's or the union's -- would rebuild prefixes that no published
    # calibration file matches. `verify_prefix` below would catch it, but only
    # after the model is loaded and only as an opaque mismatch.
    #
    # `load_task` inside it sets `_allowed_classes` to the eligible classes by
    # default (36 on TREC, which with K=5 is what yields the 180 demos), and
    # this is the same call data_calibration made -- the point being that the
    # prefix has to be bit-identical to the published one, not merely similar.
    docs_by_seed = {}
    for seed in seeds:
        reserved_s = manifest_reservation(man, demo_seed=seed)
        docs_by_seed[seed] = prefix_demo_docs(
            task_name, k, [seed], excluded_docs=reserved_s)[seed]

    prefixes, checks, doc_lookup = {}, [], None
    for seed in seeds:
        p = calibration_path(calib_dir, task_name, k, seed)
        header, first = load_calibration_header(p)
        assert_header_matches(label_space.data, header, path=str(p),
                              model=label_space.model,
                              query_manifest_sha256=qm_sha)
        if doc_lookup is None:
            doc_lookup = build_doc_lookup(load_task(task_name, k, seed),
                                          task_name)
        blocks = renderer.build_prefix(docs_by_seed[seed],
                                       header["abstract_labels"])
        ok, detail = verify_prefix(blocks, first["prompt"])
        if not ok:
            raise RuntimeError(f"seed {seed}: rendered prefix does not "
                               f"reproduce the calibration prompt: {detail}")
        prefixes[seed] = blocks
        checks.append(f"seed {seed}: {detail}")
    return prefixes, doc_lookup, checks, renderer.render_prompt


class NaturalReadout:
    """Natural candidate logits for (seed, query). The stock path, no hooks.

    Also records `full_logsumexp` and `full_argmax_token` -- see common.py for
    why a bare candidate slice is not enough to store.
    """

    def __init__(self, model, tokenizer, candidate_token_ids):
        self.model = model
        self.tok = tokenizer
        self.cand = list(int(t) for t in candidate_token_ids)

    def __call__(self, prompt):
        import torch
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(
            self.model.device)
        with torch.no_grad():
            out = self.model(ids)
        logits = out.logits[0, -1].to(torch.float64)
        lse = torch.logsumexp(logits, dim=-1)
        cand = logits[self.cand]
        return (cand.cpu().numpy(), float(lse.item()),
                int(torch.argmax(logits).item()))


def candidate_log_probs(candidate_logits):
    """log-softmax RESTRICTED to the candidate set (section 1.2).

    This is the candidate-CONDITIONAL readout: the denominator is the 36
    candidates, not the vocabulary. Reported separately from `full_logsumexp`,
    which keeps the full-space quantity recoverable -- the two are different
    numbers and mixing them is working rules 10b's error in another costume.
    """
    z = np.asarray(candidate_logits, dtype=np.float64)
    m = z.max(axis=-1, keepdims=True)
    e = np.exp(z - m)
    return z - m - np.log(e.sum(axis=-1, keepdims=True))
