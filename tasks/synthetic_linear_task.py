"""Synthetic Dataset 1: NL-rendered linear classifier (lm_eval ConfigurableTask).

Per-prompt hidden function::

    f(x) = argmax_c (W @ one_hot(x))_c

where ``W ~ N(0, sigma)`` is sampled fresh per prompt, ``x`` is a ``D``-dim
categorical feature vector with each entry in ``{0, ..., V-1}``.

Each example renders to a single natural-language sentence::

    He is a fan of {sport}, eats {food}, drinks {drink}, lives in {city}, and likes {color}.
    Category: {label}

Architectural note: unlike TREC-style tasks where demos are sampled from a
separate train pool, each synthetic test doc carries its OWN per-prompt random
``W`` and its OWN ``k_per_class * n_classes`` pre-sampled demos (because demos
from a different prompt's ``W`` would be meaningless). The lm_eval fewshot
machinery is therefore bypassed: ``_get_fewshot_examples`` returns the *current*
doc's demos (set by ``doc_to_text`` immediately before ``_build_fewshot_prefix``
is called). Monkey-patching ``_build_fewshot_prefix`` (as done by the
``patched_random_labels`` family in ``experiments/data_calibration.py``) still
works because the patch operates on the per-doc demo list, not a global pool.

Generation is cached via ``tools.synthetic_cache``: identical hyperparameters →
load from cache; new hyperparameters → regenerate. Cache root piggy-backs the
HuggingFace ``datasets`` cache directory.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import datasets
import numpy as np

# Allow standalone execution as ``python tasks/synthetic_linear_task.py ...``
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lm_eval.api.task import ConfigurableTask

from tools.synthetic_cache import get_or_generate


# ---------------------------------------------------------------------------
# Defaults (task-intrinsic; matches extend_dataset_plan.md §1)
# ---------------------------------------------------------------------------

DEFAULT_FEATURE_POOL: dict[str, list[str]] = {
    "sport": [
        "football", "basketball", "tennis", "swimming", "cycling",
        "baseball", "hockey", "golf", "boxing", "skiing",
    ],
    "food": [
        "beef", "chicken", "fish", "pizza", "rice",
        "pasta", "sushi", "salad", "soup", "bread",
    ],
    "drink": [
        "water", "coffee", "tea", "milk", "juice",
        "beer", "wine", "cola", "lemonade", "smoothie",
    ],
    "city": [
        "Tokyo", "Paris", "London", "NewYork", "Sydney",
        "Berlin", "Beijing", "Mumbai", "Cairo", "Toronto",
    ],
    "color": [
        "red", "blue", "green", "yellow", "purple",
        "orange", "black", "white", "pink", "brown",
    ],
}

DEFAULT_TEMPLATE = (
    "He is a fan of {sport}, eats {food}, drinks {drink}, "
    "lives in {city}, and likes {color}.\n"
    "Category: {label}"
)

DEFAULT_LABEL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"""Per ``paper_draft.md`` §3.1: abstract single-token labels (A–Z)."""

# Default number of classes — kept at module scope so ``_LABEL_NAMES`` (used by
# test_unified hooks that don't have a task instance) can be sliced to the right
# length at import time. If you instantiate SyntheticLinear with a different
# n_classes you must also override _LABEL_NAMES (or bump this constant) so
# data_calibration.py sees the correct label cardinality.
_DEFAULT_N_CLASSES: int = 6

FEWSHOT_DELIMITER = "\n\n"

# Module-level public name (referenced by ``_get_label_field_and_names`` etc.
# for label-field discovery — see "Integration with test_unified" note below).
# Sliced to _DEFAULT_N_CLASSES so downstream callers don't see 20 phantom
# classes that have no test/train examples.
_LABEL_NAMES: list[str] = list(DEFAULT_LABEL_CHARS[:_DEFAULT_N_CLASSES])


# ---------------------------------------------------------------------------
# Generation internals (sample W, balanced demos, query)
# ---------------------------------------------------------------------------

def _sample_weights(
    rng: np.random.Generator,
    n_classes: int,
    encoded_dim: int,
    dist: str,
    sigma: float,
) -> np.ndarray:
    """Sample weight matrix ``W ∈ R^{n_classes × encoded_dim}``."""
    if dist == "normal":
        return rng.normal(loc=0.0, scale=sigma, size=(n_classes, encoded_dim))
    if dist == "uniform":
        return rng.uniform(low=-sigma, high=sigma, size=(n_classes, encoded_dim))
    if dist == "sparse":
        # Density-0.3 ternary {-1, 0, 1}; ``sigma`` is unused
        z = rng.uniform(0.0, 1.0, size=(n_classes, encoded_dim))
        s = rng.choice([-1.0, 1.0], size=(n_classes, encoded_dim))
        return np.where(z < 0.3, s, 0.0)
    raise ValueError(f"unknown weight distribution: {dist!r}")


def _one_hot_encode(features: list[int], n_values: int) -> np.ndarray:
    D = len(features)
    out = np.zeros(D * n_values, dtype=np.float64)
    for d, v in enumerate(features):
        out[d * n_values + v] = 1.0
    return out


def _sample_balanced_demos_and_query(
    rng: np.random.Generator,
    W: np.ndarray,
    n_features: int,
    n_values: int,
    n_classes: int,
    k_per_class: int,
    pool_size: int = 5000,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Sample ``k_per_class`` demos per class + 1 unseen query under hidden W.

    Returns (demos, query) on success, or None if W produces a too-unbalanced
    class distribution (caller should resample W).
    """
    seen: set[tuple[int, ...]] = set()
    by_class: dict[int, list[tuple[int, ...]]] = {c: [] for c in range(n_classes)}

    for _ in range(pool_size):
        features = tuple(int(v) for v in rng.integers(0, n_values, size=n_features))
        if features in seen:
            continue
        seen.add(features)
        x = _one_hot_encode(list(features), n_values)
        by_class[int(np.argmax(W @ x))].append(features)

    if any(len(by_class[c]) < k_per_class + 1 for c in range(n_classes)):
        return None

    demos: list[dict[str, Any]] = []
    used: set[tuple[int, ...]] = set()
    for c in range(n_classes):
        rng.shuffle(by_class[c])
        for features in by_class[c][:k_per_class]:
            # Schema: {"features": [int,...], "label": int}. Using "label"
            # (matches lm_eval convention and our train/test row schema) so
            # data_calibration.py's `label_field = "label"` lookup works on
            # both demo entries and top-level docs.
            demos.append({"features": list(features), "label": c})
            used.add(features)

    # Shuffle demos so classes are interleaved, matching the TREC_per_class
    # convention (rng.shuffle(demos) in trec_fine_per_class_task.py). Without
    # this the prompt has all A's, then all B's, etc.; the model would learn
    # positional structure (recency bias toward the last class) rather than
    # the input→label mapping.
    rng.shuffle(demos)

    query_class = int(rng.integers(0, n_classes))
    remaining = [f for f in by_class[query_class] if f not in used]
    if not remaining:
        return None
    return demos, {"features": list(remaining[0]), "label": query_class}


def _generate_prompts(
    *,
    n_queries: int,
    n_features: int,
    n_values: int,
    n_classes: int,
    k_per_class: int,
    weight_dist: str,
    weight_sigma: float,
    seed: int,
    pool_size: int = 5000,
    max_W_resamples: int = 50,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    encoded_dim = n_features * n_values

    prompts: list[dict[str, Any]] = []
    for prompt_idx in range(n_queries):
        record = None
        for _ in range(max_W_resamples):
            W = _sample_weights(rng, n_classes, encoded_dim, weight_dist, weight_sigma)
            r = _sample_balanced_demos_and_query(
                rng, W, n_features, n_values, n_classes, k_per_class, pool_size=pool_size,
            )
            if r is not None:
                demos, query = r
                record = {
                    "prompt_idx": prompt_idx,
                    "hidden_W": W.tolist(),
                    "demos": demos,
                    "query_features": query["features"],
                    "query_label_idx": query["label"],
                }
                break
        if record is None:
            raise RuntimeError(
                f"prompt {prompt_idx}: rejection sampling failed after "
                f"{max_W_resamples} W resamples"
            )
        prompts.append(record)
    return prompts


def _generator_fn(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Cache-callable generator: produces (feature_pool, prompts)."""
    feature_pool = config.get("feature_pool", DEFAULT_FEATURE_POOL)
    if len(feature_pool) != config["n_features"]:
        raise ValueError(
            f"feature_pool has {len(feature_pool)} features; "
            f"config requests {config['n_features']}"
        )
    for fk, fv in feature_pool.items():
        if len(fv) != config["n_values"]:
            raise ValueError(
                f"feature_pool[{fk!r}] has {len(fv)} values; "
                f"config requests {config['n_values']}"
            )
    prompts = _generate_prompts(
        n_queries=config["n_queries"],
        n_features=config["n_features"],
        n_values=config["n_values"],
        n_classes=config["n_classes"],
        k_per_class=config["k_per_class"],
        weight_dist=config["weight_dist"],
        weight_sigma=config["weight_sigma"],
        seed=config["seed"],
    )
    return feature_pool, prompts


def generate(
    config: dict[str, Any],
    *,
    force_regenerate: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Top-level: cache-aware load-or-generate via ``tools.synthetic_cache``."""
    return get_or_generate(
        task_name="synthetic_linear_per_class",
        config=config,
        generator_fn=_generator_fn,
        generator_module="tasks.synthetic_linear_task",
        generator_version="1",
        force_regenerate=force_regenerate,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Rendering (used by both the task class and the standalone CLI)
# ---------------------------------------------------------------------------

def render_example(
    features: list[int],
    feature_pool: dict[str, list[str]],
    label: str | None,
    template: str = DEFAULT_TEMPLATE,
    feature_keys: list[str] | None = None,
) -> str:
    """Render a single (features, label) pair. ``label=None`` produces a query
    stub (ends after ``"Category: "``)."""
    if feature_keys is None:
        feature_keys = list(feature_pool.keys())
    fields = {k: feature_pool[k][v] for k, v in zip(feature_keys, features)}
    fields["label"] = label if label is not None else ""
    text = template.format(**fields)
    if label is None:
        text = text.rstrip()
    return text


# ---------------------------------------------------------------------------
# Format function used by ``test_unified_*`` helpers (random / uuid patches)
# ---------------------------------------------------------------------------

def _format_example_for_unified(
    doc: dict[str, Any],
    label: str,
    feature_pool: dict[str, list[str]] = DEFAULT_FEATURE_POOL,
    template: str = DEFAULT_TEMPLATE,
) -> str:
    """Render a doc (with 'features' field) with the given label string.

    Used by ``test_unified_*`` monkey-patches to render demos with random or
    abstract labels. Note: ``doc`` here is a single demo entry from
    ``self.dataset["test"][...]["demos"]``, not a top-level test doc.
    """
    feature_keys = list(feature_pool.keys())
    return render_example(doc["features"], feature_pool, label, template, feature_keys)


def query_stub_for_unified(
    doc: dict[str, Any],
    feature_pool: dict[str, list[str]] = DEFAULT_FEATURE_POOL,
    template: str = DEFAULT_TEMPLATE,
) -> str:
    """Render the query-stub portion (no label) of a top-level test doc."""
    feature_keys = list(feature_pool.keys())
    return render_example(doc["query_features"], feature_pool, None, template, feature_keys)


# ---------------------------------------------------------------------------
# lm_eval ConfigurableTask wrapper
# ---------------------------------------------------------------------------

class SyntheticLinear(ConfigurableTask):
    """lm_eval base task for the synthetic linear classifier.

    Hyperparameters with sensible defaults; override at __init__ time via kwargs::

        task = SyntheticLinear(n_features=5, n_values=10, n_classes=6,
                                weight_dist="normal", weight_sigma=1.0,
                                n_queries=250, k_per_class=20)

    ``set_fewshot(num_fewshot, seed)`` reinterprets ``num_fewshot`` as
    ``k_per_class`` (matching ``TRECFinePerClass`` convention); a different
    ``(k_per_class, seed)`` triggers regeneration via cache.
    """

    VERSION = 1.0
    DATASET_NAME = "synthetic_linear"

    DEFAULT_N_FEATURES = 5
    DEFAULT_N_VALUES = 10
    DEFAULT_N_CLASSES = _DEFAULT_N_CLASSES  # synced with module-level _LABEL_NAMES
    DEFAULT_N_QUERIES = 250
    DEFAULT_WEIGHT_DIST = "normal"
    DEFAULT_WEIGHT_SIGMA = 1.0
    DEFAULT_K_PER_CLASS = 20
    DEFAULT_SEED = 42

    def __init__(self, **kwargs):
        self._n_features = kwargs.pop("n_features", self.DEFAULT_N_FEATURES)
        self._n_values = kwargs.pop("n_values", self.DEFAULT_N_VALUES)
        self._n_classes = kwargs.pop("n_classes", self.DEFAULT_N_CLASSES)
        self._n_queries = kwargs.pop("n_queries", self.DEFAULT_N_QUERIES)
        self._weight_dist = kwargs.pop("weight_dist", self.DEFAULT_WEIGHT_DIST)
        self._weight_sigma = kwargs.pop("weight_sigma", self.DEFAULT_WEIGHT_SIGMA)
        self._k_per_class = kwargs.pop("k_per_class", self.DEFAULT_K_PER_CLASS)
        self._labels_str = kwargs.pop("labels", DEFAULT_LABEL_CHARS)

        self._num_fewshot = self._k_per_class
        self._fewshot_seed = self.DEFAULT_SEED
        self._fewshot_examples = None  # bypassed; demos baked into each test doc
        self._current_doc: dict[str, Any] | None = None
        self._feature_pool: dict[str, list[str]] = DEFAULT_FEATURE_POOL
        self._labels: list[str] = list(self._labels_str[: self._n_classes])

        kwargs.pop("task", None)
        super().__init__(
            config={
                "task": "synthetic_linear",
                "dataset_path": "synthetic_linear",
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
                "num_fewshot": 0,  # fewshot is baked into each test doc
            },
            **kwargs,
        )

    # -- fewshot API (matches TRECFinePerClass convention) ------------------

    def set_fewshot(self, num_fewshot: int, seed: int = 42):
        """Reinterprets ``num_fewshot`` as ``k_per_class``. Triggers regeneration
        if the (k_per_class, seed) combination differs from the current cache."""
        regen = (num_fewshot != self._k_per_class) or (seed != self._fewshot_seed)
        self._k_per_class = num_fewshot
        self._num_fewshot = num_fewshot
        self._fewshot_seed = seed
        if regen:
            self._fewshot_examples = None
            self.download()

    def _get_fewshot_examples(self):
        """Return the demos from the *current* test doc (set by ``doc_to_text``).

        Each demo is a dict ``{"features": [...], "label": int}``. Patches
        in ``data_calibration.py`` that render demos with random labels operate
        on this list via ``_build_fewshot_prefix``. The key is ``"label"`` (not
        ``"label_idx"``) so ``label_field = "label"`` works uniformly on demos,
        train rows, and test docs.
        """
        if self._current_doc is None:
            return []
        return list(self._current_doc.get("demos", []))

    def _build_fewshot_prefix(self) -> str:
        examples = self._get_fewshot_examples()
        if not examples:
            return ""
        feature_keys = list(self._feature_pool.keys())
        demos = [
            render_example(
                ex["features"], self._feature_pool,
                self._labels[ex["label"]],
                DEFAULT_TEMPLATE, feature_keys,
            )
            for ex in examples
        ]
        return FEWSHOT_DELIMITER.join(demos) + FEWSHOT_DELIMITER

    # -- dataset build -------------------------------------------------------

    def download(self, dataset_kwargs=None):
        cache_cfg = {
            # schema_version is part of the cache hash so renaming the demo
            # key (label_idx -> label) auto-invalidates older caches. Bump
            # this whenever the on-disk prompts.jsonl schema changes.
            "schema_version": 2,
            "task": "synthetic_linear_per_class",
            "n_features": self._n_features,
            "n_values": self._n_values,
            "n_classes": self._n_classes,
            "k_per_class": self._k_per_class,
            "n_queries": self._n_queries,
            "weight_dist": self._weight_dist,
            "weight_sigma": self._weight_sigma,
            "feature_pool": DEFAULT_FEATURE_POOL,
            "seed": self._fewshot_seed,
        }
        result = generate(cache_cfg)
        self._feature_pool = result["feature_pool"]
        prompts = result["prompts"]
        self._labels = list(self._labels_str[: self._n_classes])

        # Test rows: one per prompt, carrying demos + query
        test_rows = [
            {
                "prompt_idx": int(p["prompt_idx"]),
                "demos": [
                    {"features": list(d["features"]), "label": int(d["label"])}
                    for d in p["demos"]
                ],
                "query_features": list(p["query_features"]),
                # Mirror under `label` for compatibility with downstream
                # `train_classes` discovery in data_calibration.py
                "label": int(p["query_label_idx"]),
                "query_label_idx": int(p["query_label_idx"]),
                # Convenience textfield (so data_calibration's format_fn closures
                # have something to read if they expect doc["text"])
                "text": query_stub_for_unified(
                    {"query_features": p["query_features"]},
                    self._feature_pool,
                ),
            }
            for p in prompts
        ]

        # Train rows: synthetic placeholder so data_calibration.py's
        # `train_classes_set` detection passes (it scans dataset["train"] for
        # which class indices have any example). One placeholder per class.
        train_rows = [
            {"label": c, "text": "", "demos": [], "query_features": [], "query_label_idx": c}
            for c in range(self._n_classes)
        ]

        self.dataset = datasets.DatasetDict({
            "train": datasets.Dataset.from_list(train_rows),
            "test": datasets.Dataset.from_list(test_rows),
        })

    # -- prompt rendering ---------------------------------------------------

    def doc_to_text(self, doc, *args, **kwargs) -> str:
        # Stash current doc so _get_fewshot_examples / _build_fewshot_prefix
        # operate on this doc's demos (rather than a global pool).
        self._current_doc = doc
        prefix = self._build_fewshot_prefix()
        feature_keys = list(self._feature_pool.keys())
        query_stub = render_example(
            doc["query_features"], self._feature_pool, None,
            DEFAULT_TEMPLATE, feature_keys,
        )
        return prefix + query_stub

    def doc_to_target(self, doc, *args, **kwargs) -> str:
        return f" {self._labels[doc['query_label_idx']]}"

    def doc_to_choice(self, doc, *args, **kwargs):
        return [f" {label}" for label in self._labels]


# ---------------------------------------------------------------------------
# CLI for standalone inspection / smoke testing
# ---------------------------------------------------------------------------

def _build_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.feature_pool == "default":
        feature_pool = DEFAULT_FEATURE_POOL
    else:
        with open(args.feature_pool, encoding="utf-8") as f:
            feature_pool = json.load(f)
    return {
        "schema_version": 2,
        "task": "synthetic_linear_per_class",
        "n_features": args.n_features,
        "n_values": args.n_values,
        "n_classes": args.n_classes,
        "k_per_class": args.k_per_class,
        "n_queries": args.n_queries,
        "weight_dist": args.weight_dist,
        "weight_sigma": args.weight_sigma,
        "feature_pool": feature_pool,
        "seed": args.seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Dataset 1 (NL linear classifier). "
                    "First run caches; subsequent runs with identical "
                    "hyperparameters load from cache."
    )
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--n-values", type=int, default=10)
    parser.add_argument("--n-classes", type=int, default=6)
    parser.add_argument("--k-per-class", type=int, default=20)
    parser.add_argument("--n-queries", type=int, default=250)
    parser.add_argument("--weight-dist", default="normal",
                        choices=["normal", "uniform", "sparse"])
    parser.add_argument("--weight-sigma", type=float, default=1.0)
    parser.add_argument("--feature-pool", default="default",
                        help="'default' or path to a custom feature-pool JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", default=DEFAULT_LABEL_CHARS)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--print-example", action="store_true")
    args = parser.parse_args()

    config = _build_cli_config(args)
    result = generate(config, force_regenerate=args.force_regenerate)
    prompts = result["prompts"]
    feature_pool = result["feature_pool"]
    labels = list(args.labels[: args.n_classes])

    print(f"[synthetic_linear_task] {len(prompts)} prompts ready "
          f"(cache hash {result['meta'].get('cache_hash', '?')})")

    if args.print_example:
        p = prompts[0]
        feature_keys = list(feature_pool.keys())
        demos_text = [
            render_example(d["features"], feature_pool, labels[d["label"]],
                           DEFAULT_TEMPLATE, feature_keys)
            for d in p["demos"]
        ]
        query_text = render_example(p["query_features"], feature_pool, None,
                                    DEFAULT_TEMPLATE, feature_keys)
        rendered = FEWSHOT_DELIMITER.join(demos_text) + FEWSHOT_DELIMITER + query_text
        print()
        print("--- example rendered prompt (prompt_idx=0) ---")
        print(rendered)
        print("--- end ---")
        print(f"\n(true label: {labels[p['query_label_idx']]})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
