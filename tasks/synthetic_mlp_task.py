"""Synthetic Dataset 3: numeric MLP classifier with ReLU (lm_eval ConfigurableTask).

extend_dataset_plan.md section 3. Per-prompt hidden function::

    f(x) = argmax_c ( W2 @ relu(W1 @ x + b1) + b2 )_c

with ``x`` a ``D``-vector of integers in ``{feature_min, ..., feature_max}``
(default ``{0, ..., 9}^5``, so 100,000 combinations against 120 demos), a
hidden layer of ``H`` units, ``C`` classes, Xavier-scaled weights
``N(0, 1/sqrt(fan_in))`` and small biases ``N(0, 0.1)`` -- all resampled per
prompt. The MLP reads the digits CENTRED, ``x - (min + max) / 2`` (see
DEFAULT_INPUT_CENTER for why the raw digits do not work); the rendering is
the raw digits. Each example renders numerically::

    Feature: 3 7 1 9 2
    Type: A

Every digit is a single token in both Llama-2 and Llama-3.1 (checked in
tools/test_synthetic_mlp.py against the cached tokenizers), which is what
the plan asked to verify before launch.

Same architecture as synthetic_linear_task: each test doc carries its OWN
hidden function and its OWN pre-balanced demos, the lm_eval fewshot pool is
bypassed (``_get_fewshot_examples`` returns the current doc's demos), and
generation is cached by hyperparameter hash via ``tools.synthetic_cache``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import datasets
import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lm_eval.api.task import ConfigurableTask

from tools.synthetic_cache import get_or_generate


# ---------------------------------------------------------------------------
# Defaults (extend_dataset_plan.md section 3 / 4.3)
# ---------------------------------------------------------------------------

DEFAULT_N_FEATURES = 5
DEFAULT_FEATURE_MIN = 0
DEFAULT_FEATURE_MAX = 9
DEFAULT_HIDDEN_DIM = 10
DEFAULT_N_CLASSES = 6
DEFAULT_K_PER_CLASS = 20
DEFAULT_N_QUERIES = 250
DEFAULT_WEIGHT_INIT = "xavier"
DEFAULT_BIAS_SIGMA = 0.1
DEFAULT_ACTIVATION = "relu"
# THE HIDDEN FUNCTION SEES CENTRED INPUTS, x - (min + max) / 2. The plan
# writes x in {0..9}^5 as the input, and the RENDERING is exactly that; but a
# random Xavier MLP on raw digits (mean 4.5 on every coordinate) is dominated
# by the shared direction W1 @ 4.5, so one or two classes own almost the
# whole cube and the balanced draw of 20 demos per class fails for most
# weight draws (seed 8 found no admissible MLP in 50). Centring removes the
# shared term and leaves the plan's function class, parameter count and
# rendering untouched; the offset is stored with the weights so a record
# reproduces its own labels. --no-input-center gives the raw version.
DEFAULT_INPUT_CENTER = True
DEFAULT_POOL_SIZE = 6000
DEFAULT_MAX_RESAMPLES = 200
DEFAULT_SEED = 42

DEFAULT_TEMPLATE = "Feature: {features}\nType: {label}"
DEFAULT_LABEL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DEFAULT_N_CLASSES = DEFAULT_N_CLASSES
FEWSHOT_DELIMITER = "\n\n"
TASK_NAME = "synthetic_mlp_per_class"
SCHEMA_VERSION = 1

# Module-level names the unified hooks read (test_unified_task_*.py), as
# synthetic_linear_task exposes them.
_LABEL_NAMES: list[str] = list(DEFAULT_LABEL_CHARS[:_DEFAULT_N_CLASSES])


# ---------------------------------------------------------------------------
# The hidden function
# ---------------------------------------------------------------------------

def sample_mlp(rng, n_features, hidden_dim, n_classes, weight_init, bias_sigma):
    """{W1 [H, D], b1 [H], W2 [C, H], b2 [C]} for one prompt."""
    if weight_init == "xavier":
        s1, s2 = 1.0 / np.sqrt(n_features), 1.0 / np.sqrt(hidden_dim)
        W1 = rng.normal(0.0, s1, size=(hidden_dim, n_features))
        W2 = rng.normal(0.0, s2, size=(n_classes, hidden_dim))
    elif weight_init == "normal":
        W1 = rng.normal(0.0, 1.0, size=(hidden_dim, n_features))
        W2 = rng.normal(0.0, 1.0, size=(n_classes, hidden_dim))
    elif weight_init == "uniform":
        W1 = rng.uniform(-1.0, 1.0, size=(hidden_dim, n_features))
        W2 = rng.uniform(-1.0, 1.0, size=(n_classes, hidden_dim))
    else:
        raise ValueError(f"unknown weight_init {weight_init!r}")
    b1 = rng.normal(0.0, bias_sigma, size=hidden_dim)
    b2 = rng.normal(0.0, bias_sigma, size=n_classes)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def mlp_logits(mlp, x, activation=DEFAULT_ACTIVATION):
    """[C] for one integer feature vector; the plan's f before the argmax."""
    W1, b1, W2, b2 = (np.asarray(mlp[k], dtype=np.float64)
                      for k in ("W1", "b1", "W2", "b2"))
    xc = np.asarray(x, dtype=np.float64) - float(mlp.get("offset", 0.0))
    h = W1 @ xc + b1
    if activation == "relu":
        h = np.maximum(h, 0.0)
    elif activation == "tanh":
        h = np.tanh(h)
    else:
        raise ValueError(f"unknown activation {activation!r}")
    return W2 @ h + b2


def mlp_class(mlp, x, activation=DEFAULT_ACTIVATION):
    return int(np.argmax(mlp_logits(mlp, x, activation)))


def mlp_classes(mlp, X, activation=DEFAULT_ACTIVATION):
    """[N] classes for a [N, D] batch -- the same arithmetic as mlp_logits,
    row-wise, and the test checks the two agree point by point."""
    W1, b1, W2, b2 = (np.asarray(mlp[k], dtype=np.float64)
                      for k in ("W1", "b1", "W2", "b2"))
    Xc = np.asarray(X, dtype=np.float64) - float(mlp.get("offset", 0.0))
    H = Xc @ W1.T + b1
    if activation == "relu":
        H = np.maximum(H, 0.0)
    elif activation == "tanh":
        H = np.tanh(H)
    else:
        raise ValueError(f"unknown activation {activation!r}")
    return np.argmax(H @ W2.T + b2, axis=1)


# ---------------------------------------------------------------------------
# Generation: balanced demos + one unseen query per prompt
# ---------------------------------------------------------------------------

def _sample_balanced_demos_and_query(rng, mlp, *, n_features, feature_min,
                                     feature_max, n_classes, k_per_class,
                                     activation, pool_size=5000):
    """``k_per_class`` demos per class and one held-out query, all distinct
    feature vectors, under one hidden MLP. None when the MLP's class
    distribution cannot supply k+1 members of every class from the pool --
    the caller resamples the weights, exactly as synthetic_linear does."""
    # The pool is drawn and classified in one shot: [pool, D] integers,
    # deduplicated by row, pushed through the MLP as a matrix product. A
    # per-point loop here was 6000 small matmuls per weight draw.
    pool = rng.integers(feature_min, feature_max + 1, size=(pool_size, n_features))
    pool = np.unique(pool, axis=0)
    rng.shuffle(pool)
    cls = mlp_classes(mlp, pool, activation)
    by_class: dict[int, list[tuple[int, ...]]] = {c: [] for c in range(n_classes)}
    for feats, c in zip(pool.tolist(), cls.tolist()):
        by_class[int(c)].append(tuple(int(v) for v in feats))
    if any(len(by_class[c]) < k_per_class + 1 for c in range(n_classes)):
        return None
    demos, used = [], set()
    for c in range(n_classes):
        rng.shuffle(by_class[c])
        for feats in by_class[c][:k_per_class]:
            demos.append({"features": list(feats), "label": c})
            used.add(feats)
    rng.shuffle(demos)                   # interleave the classes (TREC convention)
    query_class = int(rng.integers(0, n_classes))
    remaining = [f for f in by_class[query_class] if f not in used]
    if not remaining:
        return None
    return demos, {"features": list(remaining[0]), "label": query_class}


def _generate_prompts(*, n_queries, n_features, feature_min, feature_max,
                      hidden_dim, n_classes, k_per_class, weight_init,
                      bias_sigma, activation, seed,
                      input_center=DEFAULT_INPUT_CENTER,
                      pool_size=DEFAULT_POOL_SIZE,
                      max_resamples=DEFAULT_MAX_RESAMPLES):
    rng = np.random.default_rng(seed)
    offset = (feature_min + feature_max) / 2.0 if input_center else 0.0
    prompts = []
    for prompt_idx in range(n_queries):
        record = None
        for _ in range(max_resamples):
            mlp = sample_mlp(rng, n_features, hidden_dim, n_classes,
                             weight_init, bias_sigma)
            mlp["offset"] = offset
            r = _sample_balanced_demos_and_query(
                rng, mlp, n_features=n_features, feature_min=feature_min,
                feature_max=feature_max, n_classes=n_classes,
                k_per_class=k_per_class, activation=activation,
                pool_size=pool_size)
            if r is not None:
                demos, query = r
                record = {
                    "prompt_idx": prompt_idx,
                    "hidden_mlp": {k: np.asarray(v).tolist()
                                   for k, v in mlp.items()},
                    "demos": demos,
                    "query_features": query["features"],
                    "query_label_idx": query["label"],
                }
                break
        if record is None:
            raise RuntimeError(
                f"prompt {prompt_idx}: no MLP in {max_resamples} draws gave "
                f"{k_per_class + 1} members of every class from a pool of "
                f"{pool_size}; lower hidden_dim or k_per_class")
        prompts.append(record)
    return prompts


def _generator_fn(config):
    """Cache-callable generator: (feature descriptor, prompts)."""
    feature_pool = {"kind": "integer", "n_features": config["n_features"],
                    "min": config["feature_min"], "max": config["feature_max"]}
    prompts = _generate_prompts(
        n_queries=config["n_queries"], n_features=config["n_features"],
        feature_min=config["feature_min"], feature_max=config["feature_max"],
        hidden_dim=config["hidden_dim"], n_classes=config["n_classes"],
        k_per_class=config["k_per_class"], weight_init=config["weight_init"],
        bias_sigma=config["bias_sigma"], activation=config["activation"],
        seed=config["seed"], input_center=config["input_center"])
    return feature_pool, prompts


def cache_config(**over):
    """The full hyperparameter set, defaults filled -- the cache hash."""
    cfg = {"schema_version": SCHEMA_VERSION, "task": TASK_NAME,
           "n_features": DEFAULT_N_FEATURES, "feature_min": DEFAULT_FEATURE_MIN,
           "feature_max": DEFAULT_FEATURE_MAX, "hidden_dim": DEFAULT_HIDDEN_DIM,
           "n_classes": DEFAULT_N_CLASSES, "k_per_class": DEFAULT_K_PER_CLASS,
           "n_queries": DEFAULT_N_QUERIES, "weight_init": DEFAULT_WEIGHT_INIT,
           "bias_sigma": DEFAULT_BIAS_SIGMA, "activation": DEFAULT_ACTIVATION,
           "input_center": DEFAULT_INPUT_CENTER, "seed": DEFAULT_SEED}
    unknown = set(over) - set(cfg)
    if unknown:
        raise ValueError(f"unknown hyperparameters {sorted(unknown)}")
    cfg.update(over)
    return cfg


def generate(config, *, force_regenerate=False, verbose=True):
    """Cache-aware load-or-generate (tools.synthetic_cache)."""
    return get_or_generate(task_name=TASK_NAME, config=config,
                           generator_fn=_generator_fn,
                           generator_module="tasks.synthetic_mlp_task",
                           generator_version="1",
                           force_regenerate=force_regenerate, verbose=verbose)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_example(features, label, template=DEFAULT_TEMPLATE):
    """``Feature: 3 7 1 9 2\\nType: A``; ``label=None`` gives the query stub
    ending after ``Type:``."""
    text = template.format(features=" ".join(str(int(v)) for v in features),
                           label=label if label is not None else "")
    return text.rstrip() if label is None else text


def _format_example_for_unified(doc, label, template=DEFAULT_TEMPLATE):
    return render_example(doc["features"], label, template)


def query_stub_for_unified(doc, template=DEFAULT_TEMPLATE):
    return render_example(doc["query_features"], None, template)


# ---------------------------------------------------------------------------
# lm_eval ConfigurableTask wrapper
# ---------------------------------------------------------------------------

class SyntheticMLP(ConfigurableTask):
    """lm_eval base task for the synthetic MLP classifier.

    ``set_fewshot(num_fewshot, seed)`` reinterprets ``num_fewshot`` as
    ``k_per_class`` (the per-class convention); a different (k, seed) pair
    regenerates through the cache.
    """

    VERSION = 1.0
    DATASET_NAME = "synthetic_mlp"

    DEFAULT_N_FEATURES = DEFAULT_N_FEATURES
    DEFAULT_FEATURE_MIN = DEFAULT_FEATURE_MIN
    DEFAULT_FEATURE_MAX = DEFAULT_FEATURE_MAX
    DEFAULT_HIDDEN_DIM = DEFAULT_HIDDEN_DIM
    DEFAULT_N_CLASSES = DEFAULT_N_CLASSES
    DEFAULT_N_QUERIES = DEFAULT_N_QUERIES
    DEFAULT_K_PER_CLASS = DEFAULT_K_PER_CLASS
    DEFAULT_WEIGHT_INIT = DEFAULT_WEIGHT_INIT
    DEFAULT_BIAS_SIGMA = DEFAULT_BIAS_SIGMA
    DEFAULT_ACTIVATION = DEFAULT_ACTIVATION
    DEFAULT_SEED = DEFAULT_SEED

    def __init__(self, **kwargs):
        self._n_features = kwargs.pop("n_features", self.DEFAULT_N_FEATURES)
        self._feature_min = kwargs.pop("feature_min", self.DEFAULT_FEATURE_MIN)
        self._feature_max = kwargs.pop("feature_max", self.DEFAULT_FEATURE_MAX)
        self._hidden_dim = kwargs.pop("hidden_dim", self.DEFAULT_HIDDEN_DIM)
        self._n_classes = kwargs.pop("n_classes", self.DEFAULT_N_CLASSES)
        self._n_queries = kwargs.pop("n_queries", self.DEFAULT_N_QUERIES)
        self._k_per_class = kwargs.pop("k_per_class", self.DEFAULT_K_PER_CLASS)
        self._weight_init = kwargs.pop("weight_init", self.DEFAULT_WEIGHT_INIT)
        self._bias_sigma = kwargs.pop("bias_sigma", self.DEFAULT_BIAS_SIGMA)
        self._activation = kwargs.pop("activation", self.DEFAULT_ACTIVATION)
        self._input_center = kwargs.pop("input_center", DEFAULT_INPUT_CENTER)
        self._labels_str = kwargs.pop("labels", DEFAULT_LABEL_CHARS)

        self._num_fewshot = self._k_per_class
        self._fewshot_seed = self.DEFAULT_SEED
        self._fewshot_examples = None
        self._current_doc: dict[str, Any] | None = None
        self._labels: list[str] = list(self._labels_str[: self._n_classes])

        kwargs.pop("task", None)
        super().__init__(
            config={
                "task": "synthetic_mlp",
                "dataset_path": "synthetic_mlp",
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
                "num_fewshot": 0,
            },
            **kwargs,
        )

    # -- fewshot API --------------------------------------------------------

    def set_fewshot(self, num_fewshot: int, seed: int = 42):
        regen = (num_fewshot != self._k_per_class) or (seed != self._fewshot_seed)
        self._k_per_class = num_fewshot
        self._num_fewshot = num_fewshot
        self._fewshot_seed = seed
        if regen:
            self._fewshot_examples = None
            self.download()

    def _get_fewshot_examples(self):
        if self._current_doc is None:
            return []
        return list(self._current_doc.get("demos", []))

    def _build_fewshot_prefix(self) -> str:
        examples = self._get_fewshot_examples()
        if not examples:
            return ""
        demos = [render_example(ex["features"], self._labels[ex["label"]])
                 for ex in examples]
        return FEWSHOT_DELIMITER.join(demos) + FEWSHOT_DELIMITER

    # -- dataset build -------------------------------------------------------

    def _cache_config(self):
        return cache_config(
            n_features=self._n_features, feature_min=self._feature_min,
            feature_max=self._feature_max, hidden_dim=self._hidden_dim,
            n_classes=self._n_classes, k_per_class=self._k_per_class,
            n_queries=self._n_queries, weight_init=self._weight_init,
            bias_sigma=self._bias_sigma, activation=self._activation,
            input_center=self._input_center, seed=self._fewshot_seed)

    def download(self, dataset_kwargs=None):
        result = generate(self._cache_config())
        prompts = result["prompts"]
        self._labels = list(self._labels_str[: self._n_classes])
        test_rows = [
            {
                "prompt_idx": int(p["prompt_idx"]),
                "demos": [{"features": list(d["features"]), "label": int(d["label"])}
                          for d in p["demos"]],
                "query_features": list(p["query_features"]),
                "label": int(p["query_label_idx"]),
                "query_label_idx": int(p["query_label_idx"]),
                "text": query_stub_for_unified(
                    {"query_features": p["query_features"]}),
            }
            for p in prompts
        ]
        # One placeholder train row per class, so class discovery over
        # dataset["train"] sees every class (the same device as Dataset 1).
        train_rows = [
            {"label": c, "text": "", "demos": [], "query_features": [],
             "query_label_idx": c}
            for c in range(self._n_classes)
        ]
        self.dataset = datasets.DatasetDict({
            "train": datasets.Dataset.from_list(train_rows),
            "test": datasets.Dataset.from_list(test_rows),
        })

    # -- prompt rendering ---------------------------------------------------

    def doc_to_text(self, doc, *args, **kwargs) -> str:
        self._current_doc = doc
        return self._build_fewshot_prefix() + render_example(
            doc["query_features"], None)

    def doc_to_target(self, doc, *args, **kwargs) -> str:
        return f" {self._labels[doc['query_label_idx']]}"

    def doc_to_choice(self, doc, *args, **kwargs):
        return [f" {label}" for label in self._labels]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Dataset 3 (numeric MLP classifier); "
                    "identical hyperparameters load from cache.")
    parser.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    parser.add_argument("--feature-min", type=int, default=DEFAULT_FEATURE_MIN)
    parser.add_argument("--feature-max", type=int, default=DEFAULT_FEATURE_MAX)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--n-classes", type=int, default=DEFAULT_N_CLASSES)
    parser.add_argument("--k-per-class", type=int, default=DEFAULT_K_PER_CLASS)
    parser.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES)
    parser.add_argument("--weight-init", default=DEFAULT_WEIGHT_INIT,
                        choices=["xavier", "normal", "uniform"])
    parser.add_argument("--bias-sigma", type=float, default=DEFAULT_BIAS_SIGMA)
    parser.add_argument("--activation", default=DEFAULT_ACTIVATION,
                        choices=["relu", "tanh"])
    parser.add_argument("--no-input-center", action="store_true",
                        help="feed the MLP the raw digits instead of "
                             "x - (min+max)/2; most weight draws are then "
                             "too unbalanced to supply k demos per class")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--labels", default=DEFAULT_LABEL_CHARS)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--print-example", action="store_true")
    args = parser.parse_args()

    config = cache_config(
        n_features=args.n_features, feature_min=args.feature_min,
        feature_max=args.feature_max, hidden_dim=args.hidden_dim,
        n_classes=args.n_classes, k_per_class=args.k_per_class,
        n_queries=args.n_queries, weight_init=args.weight_init,
        bias_sigma=args.bias_sigma, activation=args.activation,
        input_center=not args.no_input_center, seed=args.seed)
    result = generate(config, force_regenerate=args.force_regenerate)
    prompts = result["prompts"]
    labels = list(args.labels[: args.n_classes])
    print(f"[synthetic_mlp_task] {len(prompts)} prompts ready "
          f"(cache hash {result['meta'].get('cache_hash', '?')})")
    if args.print_example:
        p = prompts[0]
        demos = [render_example(d["features"], labels[d["label"]]) for d in p["demos"]]
        print("\n--- example rendered prompt (prompt_idx=0) ---")
        print(FEWSHOT_DELIMITER.join(demos) + FEWSHOT_DELIMITER
              + render_example(p["query_features"], None))
        print(f"--- end ---\n(true label: {labels[p['query_label_idx']]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
