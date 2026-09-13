"""The one place the preregistration's splits are read from the task. Needs lm_eval.

WHY THIS EXISTS. `data/calibration_*.jsonl` stores a RENDERED PROMPT STRING per
query -- `prompt`, `true_label_token_id`, `doc_idx`, `true_class_idx`, `task`,
`K`, `label_scheme`. The demonstrations are inside that string. There is no
`demos` list, no per-demo class, and no separate query text field.

That matters because two preregistered rules are stated over (class, raw text)
pairs and cannot be evaluated on a rendered prompt at all:

  * section 2.4(1) step 2 excludes any (class, text) appearing in the three
    fixed demonstration prefixes -- discovery draws from the SAME train split
    the demos come from, so without this a demo could be its own discovery
    query;
  * section 2.1(1) enumerates train-class queries of the native TEST split.

Both need the task object: `task.dataset["train"]`, `task.test_docs()`, and
`task._get_fewshot_examples()` after `set_fewshot(K, seed)`. So this module
builds the task exactly the way `experiments/data_calibration.py` does -- same
TaskManager, same include path, same download/set_fewshot order -- and every
producer imports it instead of re-deriving the setup. working rules 2.6.2: one
convention, one definition.

TREC-fine docs are `{"text": str, "fine_label": int}` (tasks/trec_fine_task.py),
and the per-class variant draws K demos PER CLASS, so every displayed class has
N_c = K >= 2 and section 3's leave-one-out mean is always defined. The field
names are looked up rather than hardcoded, and an unknown task fails loudly
rather than silently reading the wrong column.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# doc schema per task family. Extend deliberately -- a wrong guess here reads a
# different column and every hash downstream is quietly wrong.
# (text field or fields, train label field) per task, read off each task
# class. A tuple of text fields is joined for the dedup key; a non-string
# value (the generated tasks' feature lists) is stringified.
DOC_FIELDS = {
    "trec_fine": ("text", "fine_label"),
    "trec_fine_per_class": ("text", "fine_label"),
    "trec_selfextend": ("text", "fine_label"),
    "trec_selfextend_per_class": ("text", "fine_label"),
    "banking77": ("text", "label"),
    "banking77_per_class": ("text", "label"),
    "clinc150": ("text", "label"),
    "clinc150_per_class": ("text", "label"),
    "dbpedia14": (("title", "content"), "label"),
    "dbpedia14_per_class": (("title", "content"), "label"),
    "yahoo_answers": (("question_title", "question_content", "best_answer"),
                      "topic"),
    "yahoo_answers_per_class": (("question_title", "question_content",
                                 "best_answer"), "topic"),
    "yelp_full": ("text", "label"),
    "yelp_full_per_class": ("text", "label"),
    "monk": ("features", "label"),
    "monk_per_class": ("features", "label"),
    "monk_per_class_r1": ("features", "label"),
    "monk_per_class_r2": ("features", "label"),
    "monk_per_class_r3": ("features", "label"),
    "synthetic_linear": ("features", "label"),
    "synthetic_linear_per_class": ("features", "label"),
    "synthetic_mlp": ("features", "label"),
    "synthetic_mlp_per_class": ("features", "label"),
}

# The generated tasks keep their query label under another key than their
# train rows do, and their classes are fixed by construction -- every class
# is present in both splits with a known count -- so the count rule is not
# applied to them: every class is eligible.
TEST_LABEL_FIELDS = {t: "query_label_idx" for t in DOC_FIELDS
                     if t.startswith(("monk", "synthetic_"))}
ALL_ELIGIBLE = {t for t in DOC_FIELDS
                if t.startswith(("monk", "synthetic_"))}


def test_label_field(task_name: str) -> str:
    return TEST_LABEL_FIELDS.get(task_name, doc_fields(task_name)[1])


def text_key(doc, text_f):
    """The dedup key's text part: joined when several fields, stringified
    when the value is not a string (a feature list)."""
    fields = text_f if isinstance(text_f, tuple) else (text_f,)
    return "\n".join(str(doc.get(f, "")) for f in fields)


def doc_fields(task_name: str):
    if task_name not in DOC_FIELDS:
        raise SystemExit(
            f"unknown task {task_name!r}: this module refuses to guess which "
            f"columns hold the text and the label. Known: {sorted(DOC_FIELDS)}. "
            "Add it here after reading the task class, not at the call site.")
    return DOC_FIELDS[task_name]


def load_task(task_name: str, k: int, seed: int, n_queries: int | None = None,
              restrict_to_eligible: bool = True):
    """Build the task exactly as data_calibration.py does, then set the prefix.

    `restrict_to_eligible` narrows the demonstration classes to
    `eligible_classes()` -- the classes present in BOTH splits. Pass False to
    get the raw task, e.g. to inspect what the restriction removed.
    """
    import lm_eval
    from test_unified import TASK_DIR

    tm = lm_eval.tasks.TaskManager(include_path=TASK_DIR)
    task = lm_eval.tasks.get_task_dict([task_name], task_manager=tm)[task_name]
    if n_queries is not None and hasattr(task, "_n_queries"):
        # must precede download(): some tasks hash their prompt cache on it
        task._n_queries = n_queries
    task.download()
    if restrict_to_eligible:
        task._allowed_classes = set(eligible_classes(task, task_name))
    task.set_fewshot(num_fewshot=k, seed=seed)
    return task


from tools.prereg_config import MIN_TRAIN_PER_CLASS  # noqa: E402,F401


def unique_train_counts(task, task_name):
    """Per-class counts over DISTINCT (class, text) train rows.

    TREC repeats some questions verbatim. Counting raw rows would let a class
    clear the minimum on duplicates it cannot actually supply, because both the
    validation draw and the demo draw deduplicate.
    """
    text_f, label_f = doc_fields(task_name)
    seen, counts = set(), {}
    for d in task.dataset["train"]:
        key = (int(d[label_f]), text_key(d, text_f))
        if key in seen:
            continue
        seen.add(key)
        counts[key[0]] = counts.get(key[0], 0) + 1
    return counts


def eligible_classes(task, task_name, min_train=MIN_TRAIN_PER_CLASS):
    """The preregistered class space. Three conditions, all on counts alone.

      1. occurs in TRAIN  -- otherwise the class has no demonstration;
      2. occurs in TEST   -- otherwise no query can ever be about it, so its
                             demos only lengthen the prompt and its candidate
                             logit can never be correct;
      3. at least `min_train` training examples -- the class has to supply four
                             validation queries AND K demonstrations per seed
                             out of the same split.

    TREC's train split covers all 50 fine classes, its 500-question test set
    reaches 42, and six of those 42 fall below the count threshold, leaving 36.

    WHY A COUNT AND NOT "WHAT THE DEMO DRAW LEFT OVER". The obvious filter --
    keep a class if enough examples survive the three prefix draws -- is
    circular: `_get_fewshot_examples` shuffles class by class in sorted order,
    so removing a class changes the RNG consumption and hence every other
    class's draw, which changes the filter. A threshold on the raw count has no
    such feedback.

    WHY 15. Validation is reserved first (4 per class), so a class needs
    4 + K = 9 at an absolute minimum. 15 leaves at least 11 for the demo draw,
    so each seed picks 5 of 11 and the three prefixes genuinely differ; at 9
    every seed would be forced onto the same 5 and that class would contribute
    no seed-to-seed variation at all.

    This is a property of the EXPERIMENT, not of TREC, which is why it lives
    here: the task stays a faithful loader of all 50 classes.
    """
    _text_f, label_f = doc_fields(task_name)
    counts = unique_train_counts(task, task_name)
    if task_name in ALL_ELIGIBLE:
        return sorted(int(c) for c in counts)
    te = {int(d[test_label_field(task_name)]) for d in task.test_docs()}
    both = sorted(set(counts) & te)
    keep = [c for c in both if counts[c] >= min_train]
    if not keep:
        raise SystemExit(f"{task_name}: no class meets all three conditions")
    return keep


def eligibility_report(task, task_name, min_train=MIN_TRAIN_PER_CLASS):
    """Why each class was kept or dropped. For the manifest's audit trail."""
    _text_f, label_f = doc_fields(task_name)
    counts = unique_train_counts(task, task_name)
    te = {int(d[test_label_field(task_name)]) for d in task.test_docs()}
    return {"min_train": min_train,
            "n_train_classes": len(counts),
            "n_test_classes": len(te),
            "train_only": sorted(set(counts) - te),
            "below_min_train": sorted(
                {c: counts[c] for c in sorted(set(counts) & te)
                 if counts[c] < min_train}.items()),
            "eligible": eligible_classes(task, task_name, min_train)}


def row_text(doc, text_fields):
    """The text a row reports: the raw field of a single-field task, or the
    fields of a multi-field task joined with a newline. tasks/per_class_draw
    .row_text is the same rule on the task side of the validation
    reservation, and tools/test_prompt_render pins the two equal."""
    fields = text_fields if isinstance(text_fields, tuple) else (text_fields,)
    if len(fields) == 1:
        return doc[fields[0]]
    return "\n".join(str(doc.get(f, "")) for f in fields)


def train_docs_of(task, task_name):
    """(class_idx, row_text, doc) over the native TRAIN split."""
    text_f, label_f = doc_fields(task_name)
    return [(int(d[label_f]), row_text(d, text_f), d)
            for d in task.dataset["train"]]


def test_docs_of(task, task_name):
    """(class_idx, row_text, doc) over the native TEST split, in test_docs()
    order. A generated task keeps its query under other keys
    (query_features / query_label_idx); those are read here so the row
    exists, though the method line refuses such tasks upstream."""
    text_f, label_f = doc_fields(task_name)
    if task_name in TEST_LABEL_FIELDS:
        text_f, label_f = "query_features", TEST_LABEL_FIELDS[task_name]
    return [(int(d[label_f]), row_text(d, text_f), d) for d in task.test_docs()]


def docs_to_rows(task_name, docs):
    """[(class_idx, doc)] -> [(class_idx, row_text)], the pair the manifest
    hashes and the nesting check compares."""
    text_f, _label_f = doc_fields(task_name)
    return [(int(c), row_text(d, text_f)) for c, d in docs]


def train_rows(task, task_name):
    """(class_idx, raw_text, source_id) over the native TRAIN split."""
    return [(c, t, f"train:{i}")
            for i, (c, t, _d) in enumerate(train_docs_of(task, task_name))]


def test_rows(task, task_name):
    """(class_idx, raw_text, native_index) over the native TEST split.

    `native_index` is the position in `test_docs()` and NOTHING ELSE. In
    particular it is NOT the calibration files' `doc_idx`: that field indexes a
    list which `experiments/data_calibration.py` has already filtered to train
    classes, shuffled with `random.Random(seed)` and truncated to `--n-queries`
    (lines 533-567). The two live in different index spaces, and an integer join
    between them would land on the wrong query without raising.

    Cross-file identity must therefore go through the CONTENT HASH --
    `tools.prereg_ids.content_hash(class_idx, raw_text)`, which is also the
    manifest's `query_id` -- or through another stable identifier. Never through
    a row number.
    """
    return [(c, t, i)
            for i, (c, t, _d) in enumerate(test_docs_of(task, task_name))]


def train_classes(task, task_name):
    """Classes with at least one TRAIN example.

    NOT the preregistered decision space -- use `eligible_classes()` for that.
    This is the wider set (all 50 on TREC), kept for diagnostics and for
    reporting what the eligibility restriction removed.

    Never range(n): these are scattered through 0..49 and a prefix assumption
    silently drops classes (working rules rule 10).
    """
    _text_f, label_f = doc_fields(task_name)
    return sorted({int(d[label_f]) for d in task.dataset["train"]})


def test_classes(task, task_name):
    """Classes that occur in the TEST split -- the ones a query can be about."""
    _text_f, label_f = doc_fields(task_name)
    return sorted({int(d[label_f]) for d in task.test_docs()})


def draw_validation_from_train(train_rows, eligible, excluded_hashes,
                               prefix_seed, per_class=4):
    """`per_class` validation queries for every eligible class, from TRAIN.

    WHY TRAIN. Validation used to be carved out of the 500-query test split,
    which made it zero-sum: every tuning query cost a test query. It funds three
    gamma*, the matched non-TL carriers, the matched non-label positions and six
    baseline configurations, so it wants to be generous exactly where it was
    most expensive. Train has 5452 rows and no such competition.

    WHY PER CLASS. A hash-ordered block is class-imbalanced by construction and
    can miss classes outright; UniBias (13.2.5) needs the full class support and
    every baseline's configuration is chosen on this set. Equal representation
    is worth more here than matching the test split's class frequencies, which
    a tuning set does not need to do.

    THE COST, WHICH IS REAL AND NOT HIDDEN. gamma* is then chosen on the TRAIN
    distribution and applied to the TEST distribution, and TREC's two splits do
    differ -- eight classes occur in train and never in test. This is an
    EFFICIENCY question, not a validity one: no test information leaks, so the
    test result stays clean; the selected gamma* may simply be a little off the
    test-optimal. Recorded in section 2.1 rather than left implicit.

    Deterministic: ordered by validation_hash(prefix_seed, ...), so the draw
    is reproducible from the data and the seed alone, with no RNG state.

    PER SEED (section 14.0a). Each prefix seed gets its own 144. Sharing one
    set made the three configurations selected from the same sample, so their
    errors were correlated and the between-seed spread -- the thing the seeds
    exist to measure -- omitted validation-sampling noise.

    ⚠ Only the ORDER depends on the seed. `query_id` stays
    content_hash(class, text), so a text drawn under two seeds is one query
    with two cells, which is what 13.6.4's clustering needs it to be.
    """
    from tools.prereg_ids import content_hash, validation_hash

    eligible = sorted({int(c) for c in eligible})
    buckets = {}
    n_excluded = n_dup = 0
    seen_hashes = set()
    for class_idx, raw_text, source_id in train_rows:
        c = int(class_idx)
        if c not in set(eligible):
            continue
        h = content_hash(c, raw_text)
        if h in excluded_hashes:
            n_excluded += 1          # a demonstration, or an already-used query
            continue
        if h in seen_hashes:
            n_dup += 1               # TREC repeats some questions verbatim
            continue
        seen_hashes.add(h)
        buckets.setdefault(c, []).append(
            {"query_id": h, "class_idx": c, "source_id": source_id,
             # kept so the caller can hand the reservation to the task, which
             # matches on (class, text) rather than on our hash
             "source_text": raw_text,
             "_ord": validation_hash(prefix_seed, c, raw_text)})

    picked, thin = [], {}
    for c in eligible:
        rows = sorted(buckets.get(c, []), key=lambda r: r["_ord"])
        if len(rows) < per_class:
            thin[c] = len(rows)
        picked.extend(rows[:per_class])
    for r in picked:
        r.pop("_ord")
    assert len({r["query_id"] for r in picked}) == len(picked), \
        "duplicate validation query survived deduplication"
    return picked, {"n_excluded_by_hash": n_excluded,
                    "n_duplicate_train_rows": n_dup,
                    "classes_short_of_quota": thin,
                    "n_classes": len(eligible), "per_class": per_class}


def train_class_candidate_tokens(header, train_class_ids):
    """The candidate label tokens: the train classes' tokens, not the full set.

    The task as preregistered is CLOSED-SET over the classes that have training
    demonstrations, and gold support falls only on those classes. The remaining
    ones are therefore outside the decision space by definition of the task.

    That is a statement about the task, not about the model or the
    intervention: a V write changes logits across the whole vocabulary, and
    nothing here claims those classes are unreachable. They are simply not
    candidates.

    The 50-entry mapping is read from the calibration header UNCHANGED and the
    thirty are SELECTED out of it by class id. Re-discovering or renumbering
    abstract labels would move the label tokens, and the frozen carriers were
    discovered against the existing ones -- so the mapping is inherited and only
    the candidate subset narrows.

    Returns (class_ids, token_ids), aligned and sorted by class id.
    """
    ids = [int(x) for x in header["label_token_ids"]]
    classes = sorted(int(c) for c in train_class_ids)
    bad = [c for c in classes if not 0 <= c < len(ids)]
    if bad:
        raise ValueError(
            f"train class ids {bad} fall outside the header's {len(ids)}-entry "
            "label_token_ids. The header and the task disagree about the class "
            "numbering; fix that before selecting a candidate set.")
    toks = [ids[c] for c in classes]
    if len(set(toks)) != len(toks):
        dup = sorted({t for t in toks if toks.count(t) > 1})
        raise ValueError(
            f"candidate token ids are not distinct: {dup}. Two train classes "
            "share a label token, so the argmax over candidates could never "
            "separate them.")
    return classes, toks


def prefix_demo_docs(task_name, k, seeds, n_queries=None,
                     excluded_docs=None):
    """Every demonstration shown under each seed: {seed: [(class, doc), ...]}.
    The DOCUMENTS, so a task-aware renderer can format them; prefix_demo_rows
    below is the (class, text) view the manifest hashes.

    `set_fewshot` clears the cached draw, so the seeds can be walked on one task
    object. Returns the raw pairs; hashing is the caller's job, so that the
    exclusion uses the same content hash the manifest keys on.
    """
    text_f, label_f = doc_fields(task_name)
    out = {}
    task = load_task(task_name, k, int(seeds[0]), n_queries)
    if excluded_docs:
        # the validation reservation; demos are drawn from what is left
        task._excluded_docs = {(int(c), t) for c, t in excluded_docs}
    for s in seeds:
        task.set_fewshot(num_fewshot=k, seed=int(s))
        demos = task._get_fewshot_examples()
        if not demos:
            raise SystemExit(
                f"seed {s}: the task returned no demonstrations for K={k}. "
                "Section 2.4(1) requires the three fixed prefixes to be "
                "excluded from the discovery pool; an empty prefix would make "
                "that exclusion vacuous and let a demo be its own discovery "
                "query.")
        out[int(s)] = [(int(d[label_f]), d) for d in demos]
    return out


def prefix_demo_rows(task_name, k, seeds, n_queries=None,
                     excluded_docs=None):
    """Every demonstration shown under each seed: {seed: [(class, text), ...]}.
    The same draw as prefix_demo_docs, reported as the pair the manifest
    hashes on (row_text, so a multi-field task joins its fields)."""
    return {s: docs_to_rows(task_name, v)
            for s, v in prefix_demo_docs(task_name, k, seeds, n_queries,
                                         excluded_docs).items()}
