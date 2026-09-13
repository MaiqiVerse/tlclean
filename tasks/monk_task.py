"""Monk's Problem ICL task (Thrun et al. 1991).

Binary classification on 6 categorical attributes with mixed arity (3, 3, 2,
3, 4, 2 — total 432 combinations). The hidden Boolean concept is one of
Monk-1 / Monk-2 / Monk-3 (selectable via ``--rule-id``):

- Monk-1: ``(a1 == a2) OR (a5 == 1)``                    — disjunction
- Monk-2: ``EXACTLY TWO of {a1=1, a2=1, ..., a6=1}``     — count (XOR-like)
- Monk-3: ``(a5=3 AND a4=1) OR (a5/=4 AND a2/=3)``       — disjunction

Source: UCI ML Repository dataset 70 (``tasks/monk/monks-{1,2,3}.test``
holds the full 432-combination universe with ground-truth labels).

Per-prompt sampling:
  - sample ``k_per_class`` demos from each class (``balance=balanced``) or
    ``2 * k_per_class`` demos under the rule's natural class distribution
    (``balance=natural``)
  - sample 1 query from the *unused* combinations
  - query class is balanced 50/50 across prompts under ``balance=balanced``,
    and follows natural distribution under ``balance=natural``

Cache: hyperparameter-hashed under HF datasets root via
``tools/synthetic_cache.py``, identical layout to ``synthetic_linear``.

Labels: 2-class (A / B) — single-token, single-letter, projection-friendly
under the kernel-retrieval framework.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make tools/ and (if not installed) repo root importable
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

import datasets  # type: ignore
from lm_eval.api.task import ConfigurableTask  # type: ignore

from tools.synthetic_cache import get_or_generate


# ---------------------------------------------------------------------------
# Attribute value mappings (per UCI monks.names)
# ---------------------------------------------------------------------------

A1_VALUES = ["round", "square", "octagon"]                # a1 ∈ {1, 2, 3}
A2_VALUES = ["round", "square", "octagon"]                # a2 ∈ {1, 2, 3}
A3_VALUES = ["is smiling", "is not smiling"]              # a3 ∈ {1, 2}
A4_VALUES = ["sword", "balloon", "flag"]                  # a4 ∈ {1, 2, 3}
A5_VALUES = ["red", "yellow", "green", "blue"]            # a5 ∈ {1, 2, 3, 4}
A6_VALUES = ["wears a tie", "does not wear a tie"]        # a6 ∈ {1, 2}

ATTR_ARITY = [3, 3, 2, 3, 4, 2]   # for one-hot encoding (17-dim total)
N_ATTRS = 6

# Records the attribute name + text values (mirrors synthetic_linear's
# DEFAULT_FEATURE_POOL — kept in cache for debug / reproducibility).
MONK_FEATURE_POOL: dict[str, list[str]] = {
    "a1": A1_VALUES, "a2": A2_VALUES, "a3": A3_VALUES,
    "a4": A4_VALUES, "a5": A5_VALUES, "a6": A6_VALUES,
}

DEFAULT_TEMPLATE = (
    "The robot has a {a1} head, a {a2} body, {a3_phrase}, "
    "holds a {a4}, wears a {a5} jacket, and {a6_phrase}.\n"
    "Category: {label}"
)

DEFAULT_LABEL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DEFAULT_N_CLASSES: int = 2  # Monk is binary
_LABEL_NAMES: list[str] = list(DEFAULT_LABEL_CHARS[:_DEFAULT_N_CLASSES])

FEWSHOT_DELIMITER = "\n\n"

MONK_DATA_DIR = _HERE.parent / "monk"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_example(
    features: list[int],
    label: str | None,
    template: str = DEFAULT_TEMPLATE,
) -> str:
    """Render a single (features, label) pair. ``features`` is a length-6
    list of 1-based ints matching the UCI monks file format. ``label=None``
    produces a query stub (ends after ``"Category: "``)."""
    a1, a2, a3, a4, a5, a6 = features
    fields = {
        "a1": A1_VALUES[a1 - 1],
        "a2": A2_VALUES[a2 - 1],
        "a3_phrase": A3_VALUES[a3 - 1],
        "a4": A4_VALUES[a4 - 1],
        "a5": A5_VALUES[a5 - 1],
        "a6_phrase": A6_VALUES[a6 - 1],
        "label": label if label is not None else "",
    }
    text = template.format(**fields)
    if label is None:
        text = text.rstrip()
    return text


def _format_example_for_unified(doc: dict[str, Any], label: str) -> str:
    """Render a demo dict (with 'features' key) using ``label``. Called by
    test_unified_*.py random/abstract patches."""
    return render_example(doc["features"], label)


def query_stub_for_unified(doc: dict[str, Any]) -> str:
    """Render the query stub from a top-level test doc."""
    return render_example(doc["query_features"], None)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_monk_file(path: Path) -> list[dict[str, Any]]:
    """Parse a UCI Monk .train / .test file into a list of ``{"features",
    "label"}`` records. File rows are ``<class> <a1> <a2> <a3> <a4> <a5> <a6> <id>``."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Monk dataset file not found: {path}. "
            f"Expected UCI dataset 70 files under {MONK_DATA_DIR}."
        )
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            parts = raw.split()
            if len(parts) < 8:
                continue
            rows.append({
                "features": [int(x) for x in parts[1:7]],
                "label": int(parts[0]),
            })
    return rows


def _features_key(features: list[int]) -> tuple[int, ...]:
    return tuple(features)


def load_monk_data(rule_id: int, verbose: bool = True) -> dict[str, Any]:
    """Load .train (sampled subset) and held-out (= .test − .train) split by class.

    Strict (P2): demos must come from ``.train``; query must come from
    ``.test − .train``. No augmentation — k_per_class is bounded by .train's
    per-class counts (see ``get_max_k_per_class``).
    """
    train_rows = _load_monk_file(MONK_DATA_DIR / f"monks-{rule_id}.train")
    test_rows = _load_monk_file(MONK_DATA_DIR / f"monks-{rule_id}.test")
    if len(test_rows) != 432:
        raise RuntimeError(
            f"Expected 432 rows in monks-{rule_id}.test, got {len(test_rows)}."
        )

    train_keys = {_features_key(r["features"]) for r in train_rows}
    heldout_rows = [
        r for r in test_rows if _features_key(r["features"]) not in train_keys
    ]

    train_by_class: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for r in train_rows:
        train_by_class[r["label"]].append(
            {"features": list(r["features"]), "label": r["label"]}
        )
    heldout_by_class: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for r in heldout_rows:
        heldout_by_class[r["label"]].append(
            {"features": list(r["features"]), "label": r["label"]}
        )

    if verbose:
        print(f"[monk_data] rule={rule_id}  "
              f"train: {len(train_by_class[0])}/{len(train_by_class[1])}  "
              f"heldout: {len(heldout_by_class[0])}/{len(heldout_by_class[1])}")

    return {
        "train_by_class": train_by_class,
        "heldout_by_class": heldout_by_class,
        "n_train": len(train_rows),
        "n_heldout": len(heldout_rows),
    }


def get_max_k_per_class(rule_id: int, balance: str = "balanced") -> int:
    """Largest ``k_per_class`` supported by strict (.train-only) sampling.

      - balanced: ``min(|train_c0|, |train_c1|)``
      - natural:  ``|train| // 2``

    Per-rule values for Monk-1/2/3 with .train as released by UCI dataset 70:
      balanced -> {1: 62, 2: 64, 3: 60}
      natural  -> {1: 62, 2: 84, 3: 61}
    """
    data = load_monk_data(rule_id, verbose=False)
    t0 = len(data["train_by_class"][0])
    t1 = len(data["train_by_class"][1])
    if balance == "balanced":
        return min(t0, t1)
    elif balance == "natural":
        return (t0 + t1) // 2
    raise ValueError(f"Unknown balance mode: {balance!r}")


def get_max_n_queries(rule_id: int, balance: str = "balanced") -> int:
    """Largest ``n_queries`` supported by *unique* (no-replacement) query sampling
    from the held-out pool ``.test − .train``.

      - balanced: ``2 * min(|heldout_c0|, |heldout_c1|)`` (enforced 50/50 split)
      - natural:  ``|heldout|``

    Per-rule values for Monk-1/2/3 (UCI dataset 70):
      balanced -> {1: 308, 2: 156, 3: 284}
      natural  -> {1: 308, 2: 263, 3: 310}
    """
    data = load_monk_data(rule_id, verbose=False)
    h0 = len(data["heldout_by_class"][0])
    h1 = len(data["heldout_by_class"][1])
    if balance == "balanced":
        return 2 * min(h0, h1)
    elif balance == "natural":
        return h0 + h1
    raise ValueError(f"Unknown balance mode: {balance!r}")


def _resolve_k(k_spec: str | int, rule_id: int, balance: str) -> int:
    """Resolve ``--K`` CLI value. ``'maximum'`` → per-rule cap; else int."""
    if isinstance(k_spec, str) and k_spec.lower() == "maximum":
        return get_max_k_per_class(rule_id, balance)
    return int(k_spec)


# ---------------------------------------------------------------------------
# Per-prompt sampling
# ---------------------------------------------------------------------------

def _allocate_unique_queries(
    rng: np.random.Generator,
    data: dict[str, Any],
    n_queries: int,
    balance: str,
) -> list[dict[str, Any]]:
    """Pre-allocate ``n_queries`` *unique* queries (no replacement) from the
    held-out pool. Length-``n_queries`` ordered list of ``{features, label}``.

    Under ``balanced``: query class follows ``prompt_idx % 2`` (50/50 split,
    deterministic). Under ``natural``: random permutation across the union of
    both classes' held-out rows.
    """
    heldout = data["heldout_by_class"]
    if balance == "balanced":
        n_c0 = (n_queries + 1) // 2   # prompt_idx even
        n_c1 = n_queries // 2          # prompt_idx odd
        idx0 = rng.permutation(len(heldout[0]))[:n_c0]
        idx1 = rng.permutation(len(heldout[1]))[:n_c1]
        out: list[dict[str, Any]] = []
        for prompt_idx in range(n_queries):
            if prompt_idx % 2 == 0:
                row = heldout[0][int(idx0[prompt_idx // 2])]
            else:
                row = heldout[1][int(idx1[prompt_idx // 2])]
            out.append(row)
        return out
    elif balance == "natural":
        all_h = heldout[0] + heldout[1]
        idxs = rng.permutation(len(all_h))[:n_queries]
        return [all_h[int(i)] for i in idxs]
    raise ValueError(f"Unknown balance mode: {balance!r}")


def _sample_demos(
    rng: np.random.Generator,
    data: dict[str, Any],
    k_per_class: int,
    balance: str,
) -> list[dict[str, Any]]:
    """Sample one prompt's demos from .train. Same row may appear in multiple
    prompts (no across-prompt deduplication) — model has no cross-prompt
    state, so this is fine. Within a prompt: distinct rows."""
    train = data["train_by_class"]
    demos: list[dict[str, Any]] = []
    if balance == "balanced":
        for c in (0, 1):
            avail = train[c]
            idxs = rng.permutation(len(avail))[:k_per_class]
            for i in idxs:
                row = avail[int(i)]
                demos.append({"features": list(row["features"]), "label": c})
    elif balance == "natural":
        train_all = train[0] + train[1]
        n_total = 2 * k_per_class
        idxs = rng.permutation(len(train_all))[:n_total]
        for i in idxs:
            row = train_all[int(i)]
            demos.append({"features": list(row["features"]),
                          "label": int(row["label"])})
    else:
        raise ValueError(f"Unknown balance mode: {balance!r}")
    # Shuffle so demos are class-interleaved (matches TREC_per_class convention).
    perm = rng.permutation(len(demos))
    return [demos[int(i)] for i in perm]


def _generate_prompts(
    *,
    rule_id: int,
    n_queries: int,
    k_per_class: int,
    balance: str,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    data = load_monk_data(rule_id)

    cap_k = get_max_k_per_class(rule_id, balance)
    if k_per_class > cap_k:
        raise RuntimeError(
            f"Monk-{rule_id} ({balance}): k_per_class={k_per_class} exceeds "
            f"cap={cap_k}. Use --K maximum or a smaller K."
        )
    cap_q = get_max_n_queries(rule_id, balance)
    if n_queries > cap_q:
        raise RuntimeError(
            f"Monk-{rule_id} ({balance}): n_queries={n_queries} exceeds the "
            f"unique-query pool of size {cap_q}. "
            f"Lower --n-queries or switch --balance "
            f"({'natural' if balance == 'balanced' else 'balanced'} cap = "
            f"{get_max_n_queries(rule_id, 'natural' if balance == 'balanced' else 'balanced')})."
        )

    queries = _allocate_unique_queries(rng, data, n_queries, balance)

    prompts: list[dict[str, Any]] = []
    for prompt_idx, q in enumerate(queries):
        demos = _sample_demos(rng, data, k_per_class, balance)
        prompts.append({
            "prompt_idx": prompt_idx,
            "demos": demos,
            "query_features": list(q["features"]),
            "query_label_idx": int(q["label"]),
        })
    return prompts


def _generator_fn(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Cache-callable generator. Returns (feature_pool, prompts)."""
    prompts = _generate_prompts(
        rule_id=config["rule_id"],
        n_queries=config["n_queries"],
        k_per_class=config["k_per_class"],
        balance=config["balance"],
        seed=config["seed"],
    )
    return MONK_FEATURE_POOL, prompts


def generate(
    config: dict[str, Any],
    *,
    force_regenerate: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Cache-aware load-or-generate via ``tools.synthetic_cache``."""
    return get_or_generate(
        task_name="monk_per_class",
        config=config,
        generator_fn=_generator_fn,
        generator_module="tasks.monk_task",
        generator_version="1",
        force_regenerate=force_regenerate,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# lm_eval ConfigurableTask wrapper
# ---------------------------------------------------------------------------

class Monk(ConfigurableTask):
    """lm_eval base task for the Monk binary classifier.

    Hyperparameters (all overridable at __init__):
      - rule_id ∈ {1, 2, 3}     (default 1)
      - balance ∈ {balanced, natural}  (default 'balanced')
      - k_per_class             (default 20)
      - n_queries               (default 250)
      - labels                  (default 'A-Z' chars, first n_classes used)

    ``set_fewshot(num_fewshot, seed)`` reinterprets ``num_fewshot`` as
    ``k_per_class`` (mirroring SyntheticLinear / TRECFinePerClass).
    """

    VERSION = 1.0
    DATASET_NAME = "monk"

    DEFAULT_RULE_ID = 1
    DEFAULT_BALANCE = "balanced"
    DEFAULT_N_CLASSES = _DEFAULT_N_CLASSES   # 2
    DEFAULT_K_PER_CLASS = 20
    DEFAULT_N_QUERIES = 250
    DEFAULT_SEED = 42

    def __init__(self, **kwargs):
        self._rule_id = kwargs.pop("rule_id", self.DEFAULT_RULE_ID)
        self._balance = kwargs.pop("balance", self.DEFAULT_BALANCE)
        self._n_classes = kwargs.pop("n_classes", self.DEFAULT_N_CLASSES)
        self._k_per_class = kwargs.pop("k_per_class", self.DEFAULT_K_PER_CLASS)
        self._n_queries = kwargs.pop("n_queries", self.DEFAULT_N_QUERIES)
        self._labels_str = kwargs.pop("labels", DEFAULT_LABEL_CHARS)

        self._num_fewshot = self._k_per_class
        self._fewshot_seed = self.DEFAULT_SEED
        self._fewshot_examples = None
        self._current_doc: dict[str, Any] | None = None
        self._feature_pool: dict[str, list[str]] = MONK_FEATURE_POOL
        self._labels: list[str] = list(self._labels_str[: self._n_classes])

        kwargs.pop("task", None)
        super().__init__(
            config={
                "task": "monk",
                "dataset_path": "monk",
                "test_split": "test",
                "training_split": "train",
                "output_type": "multiple_choice",
                "doc_to_text": "",
                "doc_to_target": "",
                "doc_to_choice": "",
                "fewshot_delimiter": FEWSHOT_DELIMITER,
                "target_delimiter": "",
                "metric_list": [
                    {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
                ],
                "num_fewshot": 0,  # demos baked into each test doc
            },
            **kwargs,
        )

    # ---- fewshot ----------------------------------------------------------

    def set_fewshot(self, num_fewshot: int, seed: int = 42):
        """``num_fewshot`` ≡ ``k_per_class`` (mirrors TRECFinePerClass)."""
        regen = (num_fewshot != self._k_per_class) or (seed != self._fewshot_seed)
        self._k_per_class = num_fewshot
        self._num_fewshot = num_fewshot
        self._fewshot_seed = seed
        if regen:
            self._fewshot_examples = None
            self.download()

    def _get_fewshot_examples(self):
        """Return demos baked into the current test doc."""
        if self._current_doc is None:
            return []
        return list(self._current_doc.get("demos", []))

    def _build_fewshot_prefix(self) -> str:
        examples = self._get_fewshot_examples()
        if not examples:
            return ""
        demos = [
            render_example(ex["features"], self._labels[ex["label"]])
            for ex in examples
        ]
        return FEWSHOT_DELIMITER.join(demos) + FEWSHOT_DELIMITER

    # ---- dataset build ----------------------------------------------------

    def download(self, dataset_kwargs=None):
        cache_cfg = {
            "schema_version": 1,
            "task": "monk_per_class",
            "rule_id": self._rule_id,
            "balance": self._balance,
            "n_classes": self._n_classes,
            "k_per_class": self._k_per_class,
            "n_queries": self._n_queries,
            "feature_pool": MONK_FEATURE_POOL,
            "seed": self._fewshot_seed,
        }
        result = generate(cache_cfg)
        self._feature_pool = result["feature_pool"]
        prompts = result["prompts"]
        self._labels = list(self._labels_str[: self._n_classes])

        test_rows = [
            {
                "prompt_idx": int(p["prompt_idx"]),
                "demos": [
                    {"features": list(d["features"]), "label": int(d["label"])}
                    for d in p["demos"]
                ],
                "query_features": list(p["query_features"]),
                "label": int(p["query_label_idx"]),
                "query_label_idx": int(p["query_label_idx"]),
                "text": query_stub_for_unified(
                    {"query_features": p["query_features"]}
                ),
            }
            for p in prompts
        ]

        train_rows = [
            {"label": c, "text": "", "demos": [], "query_features": [],
             "query_label_idx": c}
            for c in range(self._n_classes)
        ]

        self.dataset = datasets.DatasetDict({
            "train": datasets.Dataset.from_list(train_rows),
            "test": datasets.Dataset.from_list(test_rows),
        })

    # ---- prompt rendering -------------------------------------------------

    def doc_to_text(self, doc, *args, **kwargs) -> str:
        self._current_doc = doc
        prefix = self._build_fewshot_prefix()
        query_stub = render_example(doc["query_features"], None)
        return prefix + query_stub

    def doc_to_target(self, doc, *args, **kwargs) -> str:
        return f" {self._labels[doc['query_label_idx']]}"

    def doc_to_choice(self, doc, *args, **kwargs):
        return [f" {label}" for label in self._labels]


# ---------------------------------------------------------------------------
# CLI for standalone inspection / smoke testing
# ---------------------------------------------------------------------------

def _build_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "monk_per_class",
        "rule_id": args.rule_id,
        "balance": args.balance,
        "n_classes": 2,
        "k_per_class": args.k_per_class,
        "n_queries": args.n_queries,
        "feature_pool": MONK_FEATURE_POOL,
        "seed": args.seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Monk Dataset 2 (binary classification on UCI Monk universe). "
                    "Demos sampled from .train only; queries from .test − .train. "
                    "First run caches; subsequent identical runs hit cache."
    )
    parser.add_argument("--rule-id", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--balance", default="balanced",
                        choices=["balanced", "natural"])
    parser.add_argument("--K", "--k-per-class", dest="k_per_class",
                        default="20",
                        help="k_per_class as int, or 'maximum' to use the "
                             "per-rule .train cap (balanced: Monk-1=62, "
                             "Monk-2=64, Monk-3=60; natural: 62/84/61).")
    parser.add_argument("--n-queries", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", default=DEFAULT_LABEL_CHARS)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--print-example", action="store_true")
    parser.add_argument("--print-max-k", action="store_true",
                        help="print the per-(rule, balance) k_per_class cap and exit")
    parser.add_argument("--print-max-n-queries", action="store_true",
                        help="print the per-(rule, balance) unique-query cap and exit")
    args = parser.parse_args()

    if args.print_max_k:
        print(get_max_k_per_class(args.rule_id, args.balance))
        return 0
    if args.print_max_n_queries:
        print(get_max_n_queries(args.rule_id, args.balance))
        return 0

    args.k_per_class = _resolve_k(args.k_per_class, args.rule_id, args.balance)
    print(f"[setup] k_per_class = {args.k_per_class} "
          f"(rule={args.rule_id}, balance={args.balance})")

    config = _build_cli_config(args)
    result = generate(config, force_regenerate=args.force_regenerate)
    prompts = result["prompts"]
    labels = list(args.labels[:2])
    print(f"[monk_task] rule_id={args.rule_id} balance={args.balance} "
          f"{len(prompts)} prompts ready (cache hash "
          f"{result['meta'].get('cache_hash', '?')})")

    # Class balance sanity
    q_classes = [p["query_label_idx"] for p in prompts]
    print(f"[monk_task] query class distribution: "
          f"0={q_classes.count(0)} 1={q_classes.count(1)} "
          f"(n={len(q_classes)})")

    if args.print_example:
        p = prompts[0]
        demos_text = [
            render_example(d["features"], labels[d["label"]])
            for d in p["demos"]
        ]
        query_text = render_example(p["query_features"], None)
        rendered = FEWSHOT_DELIMITER.join(demos_text) + FEWSHOT_DELIMITER + query_text
        print()
        print("--- example rendered prompt (prompt_idx=0) ---")
        print(rendered)
        print("--- end ---")
        print(f"\n(true label: {labels[p['query_label_idx']]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
