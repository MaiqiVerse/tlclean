"""Synthetic linear classifier as a shared-bank task: ONE hidden weight
matrix per task instance, a balanced demonstration bank and a disjoint,
balanced test pool over the same 10^5 feature universe.

The hidden function is synthetic_linear_task's, unchanged::

    y(x) = argmax_c ( W @ onehot(x) )_c,   W ~ N(0, sigma^2)^{C x (D*V)}

with D = 5 word features of V = 10 values (DEFAULT_FEATURE_POOL), C = 6
classes; the rendering is synthetic_linear_task.render_example, so a prompt
reads exactly as the per-prompt task's. What changes is only WHEN W is
drawn: once per task instance, from `function_seed` (default 0), instead of
once per query.

Construction (deterministic in `function_seed`):

  1. draw W; classify the WHOLE universe (all V^D = 100,000 vectors);
  2. accept W iff every class holds at least n_train + n_test members
     (default 60 + 100); otherwise redraw, at most `max_resamples` times
     -- the same rejection rule the per-prompt generator applies to its
     6,000-vector pool, at the universe's scale;
  3. per class: a random permutation of its members, the first n_train go
     to the bank, the next n_test to the test pool; both splits are then
     shuffled so their native order is not class-blocked.

Sizes: bank 6 x 60 = 360 rows (a class clears MIN_TRAIN_PER_CLASS = 15 four
times over and keeps 56 after the 4-per-class validation reservation, so
K = 20 per class is still drawable); pool 6 x 100 = 600 rows, from which a
seed draws its 250 test queries.

Task name (the yaml): synthetic_linear_bank_per_class.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # the family modules import tools.synthetic_cache

from synthetic_linear_task import (  # noqa: E402
    DEFAULT_FEATURE_POOL, DEFAULT_TEMPLATE, _sample_weights, render_example,
)
from shared_bank_task import SharedBankTask, make_row  # noqa: E402

DEFAULT_N_FEATURES = 5
DEFAULT_N_VALUES = 10
DEFAULT_N_CLASSES = 6
DEFAULT_WEIGHT_DIST = "normal"
DEFAULT_WEIGHT_SIGMA = 1.0
DEFAULT_FUNCTION_SEED = 0
DEFAULT_N_TRAIN_PER_CLASS = 60
DEFAULT_N_TEST_PER_CLASS = 100
DEFAULT_MAX_RESAMPLES = 200


def universe(n_features: int, n_values: int) -> np.ndarray:
    """Every feature vector, [V^D, D] ints, in lexicographic order."""
    return np.array(list(itertools.product(range(n_values), repeat=n_features)),
                    dtype=np.int64)


def one_hot_matrix(X: np.ndarray, n_values: int) -> np.ndarray:
    """[N, D*V] one-hot rows, the same encoding as synthetic_linear_task
    ._one_hot_encode applied row by row (the test checks that)."""
    n, d = X.shape
    out = np.zeros((n, d * n_values), dtype=np.float64)
    cols = np.arange(d) * n_values + X
    out[np.arange(n)[:, None], cols] = 1.0
    return out


def split_by_class(rng: np.random.Generator, X: np.ndarray, y: np.ndarray,
                   n_classes: int, n_train: int, n_test: int):
    """Per class, a permutation of its members: n_train to the bank, the
    next n_test to the pool. None if some class is too thin."""
    train, test = [], []
    for c in range(n_classes):
        members = np.flatnonzero(y == c)
        if len(members) < n_train + n_test:
            return None
        order = members[rng.permutation(len(members))]
        train += [make_row(X[i], c) for i in order[:n_train]]
        test += [make_row(X[i], c) for i in order[n_train:n_train + n_test]]
    train = [train[i] for i in rng.permutation(len(train))]
    test = [test[i] for i in rng.permutation(len(test))]
    return train, test


def build_linear_bank(*, n_features=DEFAULT_N_FEATURES, n_values=DEFAULT_N_VALUES,
                      n_classes=DEFAULT_N_CLASSES, weight_dist=DEFAULT_WEIGHT_DIST,
                      weight_sigma=DEFAULT_WEIGHT_SIGMA,
                      function_seed=DEFAULT_FUNCTION_SEED,
                      n_train_per_class=DEFAULT_N_TRAIN_PER_CLASS,
                      n_test_per_class=DEFAULT_N_TEST_PER_CLASS,
                      max_resamples=DEFAULT_MAX_RESAMPLES):
    rng = np.random.default_rng(function_seed)
    X = universe(n_features, n_values)
    H = one_hot_matrix(X, n_values)
    for draw in range(1, max_resamples + 1):
        W = _sample_weights(rng, n_classes, n_features * n_values,
                            weight_dist, weight_sigma)
        y = np.argmax(H @ W.T, axis=1)
        counts = np.bincount(y, minlength=n_classes)
        r = split_by_class(rng, X, y, n_classes, n_train_per_class, n_test_per_class)
        if r is not None:
            train, test = r
            meta = {"function_seed": function_seed, "weight_draws": draw,
                    "hidden_W": W.tolist(),
                    "universe_class_counts": counts.tolist()}
            return train, test, meta
    raise RuntimeError(
        f"no W in {max_resamples} draws gave {n_train_per_class + n_test_per_class} "
        f"members of every one of {n_classes} classes over {len(X)} vectors")


class SyntheticLinearBank(SharedBankTask):
    TASK_NAME = "synthetic_linear_bank_per_class"
    DATASET_NAME = "synthetic_linear_bank"
    N_CLASSES = DEFAULT_N_CLASSES

    def __init__(self, **kwargs):
        self._params = {
            "n_features": int(kwargs.pop("n_features", DEFAULT_N_FEATURES)),
            "n_values": int(kwargs.pop("n_values", DEFAULT_N_VALUES)),
            "n_classes": int(kwargs.pop("n_classes", DEFAULT_N_CLASSES)),
            "weight_dist": kwargs.pop("weight_dist", DEFAULT_WEIGHT_DIST),
            "weight_sigma": float(kwargs.pop("weight_sigma", DEFAULT_WEIGHT_SIGMA)),
            "function_seed": int(kwargs.pop("function_seed", DEFAULT_FUNCTION_SEED)),
            "n_train_per_class": int(kwargs.pop("n_train_per_class", DEFAULT_N_TRAIN_PER_CLASS)),
            "n_test_per_class": int(kwargs.pop("n_test_per_class", DEFAULT_N_TEST_PER_CLASS)),
            "max_resamples": int(kwargs.pop("max_resamples", DEFAULT_MAX_RESAMPLES)),
        }
        if self._params["n_classes"] != self.N_CLASSES:
            raise ValueError(f"n_classes must be {self.N_CLASSES} for "
                             f"{self.TASK_NAME} (the label table is module-level)")
        if len(DEFAULT_FEATURE_POOL) != self._params["n_features"] or any(
                len(v) != self._params["n_values"] for v in DEFAULT_FEATURE_POOL.values()):
            raise ValueError("n_features / n_values must match DEFAULT_FEATURE_POOL")
        self._feature_keys = list(DEFAULT_FEATURE_POOL.keys())
        super().__init__(**kwargs)

    def build_rows(self):
        return build_linear_bank(**self._params)

    def render(self, features, label):
        return render_example(features, DEFAULT_FEATURE_POOL, label,
                              DEFAULT_TEMPLATE, self._feature_keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--function-seed", type=int, default=DEFAULT_FUNCTION_SEED)
    ap.add_argument("--print-example", action="store_true")
    args = ap.parse_args()
    t = SyntheticLinearBank(function_seed=args.function_seed)
    t.download()
    m = dict(t._bank_meta)
    m.pop("hidden_W")
    print(t.bank_summary().split("  meta ")[0] + f"  meta {m}")
    if args.print_example:
        t.set_fewshot(2, 42)
        q = t.dataset["test"][0]
        print(t.doc_to_text(q) + t.doc_to_target(q))
    return 0


if __name__ == "__main__":
    sys.exit(main())
