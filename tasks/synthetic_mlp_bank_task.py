"""Synthetic MLP classifier as a shared-bank task: ONE hidden network per
task instance, a balanced demonstration bank and a disjoint, balanced test
pool over the same {0..9}^5 universe.

The hidden function is synthetic_mlp_task's, unchanged::

    f(x) = argmax_c ( W2 @ relu(W1 @ (x - 4.5) + b1) + b2 )_c

(D = 5 digit features, H = 10 hidden units, C = 6 classes, Xavier weights,
N(0, 0.1) biases, centred input -- sample_mlp / mlp_classes), rendered by
synthetic_mlp_task.render_example ("Feature: 3 7 1 9 2\\nType: A"). The
network is drawn once per task instance from `function_seed` (default 0)
instead of once per query.

Construction and sizes follow synthetic_linear_bank_task: classify the
whole universe (100,000 vectors), accept the draw iff every class holds at
least n_train + n_test members (default 60 + 100), split each class by a
random permutation, shuffle both splits.

Task name (the yaml): synthetic_mlp_bank_per_class.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # the family modules import tools.synthetic_cache

from synthetic_mlp_task import (  # noqa: E402
    DEFAULT_ACTIVATION, DEFAULT_BIAS_SIGMA, DEFAULT_FEATURE_MAX,
    DEFAULT_FEATURE_MIN, DEFAULT_HIDDEN_DIM, DEFAULT_INPUT_CENTER,
    DEFAULT_N_CLASSES, DEFAULT_N_FEATURES, DEFAULT_WEIGHT_INIT, mlp_classes,
    render_example, sample_mlp,
)
from synthetic_linear_bank_task import split_by_class  # noqa: E402
from shared_bank_task import SharedBankTask  # noqa: E402

DEFAULT_FUNCTION_SEED = 0
DEFAULT_N_TRAIN_PER_CLASS = 60
DEFAULT_N_TEST_PER_CLASS = 100
DEFAULT_MAX_RESAMPLES = 200


def digit_universe(n_features: int, feature_min: int, feature_max: int) -> np.ndarray:
    """Every digit vector, [(max-min+1)^D, D] ints, lexicographic."""
    grids = np.meshgrid(*[np.arange(feature_min, feature_max + 1)] * n_features,
                        indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1).astype(np.int64)


def build_mlp_bank(*, n_features=DEFAULT_N_FEATURES, feature_min=DEFAULT_FEATURE_MIN,
                   feature_max=DEFAULT_FEATURE_MAX, hidden_dim=DEFAULT_HIDDEN_DIM,
                   n_classes=DEFAULT_N_CLASSES, weight_init=DEFAULT_WEIGHT_INIT,
                   bias_sigma=DEFAULT_BIAS_SIGMA, activation=DEFAULT_ACTIVATION,
                   input_center=DEFAULT_INPUT_CENTER,
                   function_seed=DEFAULT_FUNCTION_SEED,
                   n_train_per_class=DEFAULT_N_TRAIN_PER_CLASS,
                   n_test_per_class=DEFAULT_N_TEST_PER_CLASS,
                   max_resamples=DEFAULT_MAX_RESAMPLES):
    rng = np.random.default_rng(function_seed)
    X = digit_universe(n_features, feature_min, feature_max)
    offset = (feature_min + feature_max) / 2.0 if input_center else 0.0
    for draw in range(1, max_resamples + 1):
        mlp = sample_mlp(rng, n_features, hidden_dim, n_classes, weight_init, bias_sigma)
        mlp["offset"] = offset
        y = mlp_classes(mlp, X, activation)
        counts = np.bincount(y, minlength=n_classes)
        r = split_by_class(rng, X, y, n_classes, n_train_per_class, n_test_per_class)
        if r is not None:
            train, test = r
            meta = {"function_seed": function_seed, "weight_draws": draw,
                    "hidden_mlp": {k: (np.asarray(v).tolist() if k != "offset" else float(v))
                                   for k, v in mlp.items()},
                    "universe_class_counts": counts.tolist()}
            return train, test, meta
    raise RuntimeError(
        f"no MLP in {max_resamples} draws gave {n_train_per_class + n_test_per_class} "
        f"members of every one of {n_classes} classes over {len(X)} vectors")


class SyntheticMLPBank(SharedBankTask):
    TASK_NAME = "synthetic_mlp_bank_per_class"
    DATASET_NAME = "synthetic_mlp_bank"
    N_CLASSES = DEFAULT_N_CLASSES

    def __init__(self, **kwargs):
        self._params = {
            "n_features": int(kwargs.pop("n_features", DEFAULT_N_FEATURES)),
            "feature_min": int(kwargs.pop("feature_min", DEFAULT_FEATURE_MIN)),
            "feature_max": int(kwargs.pop("feature_max", DEFAULT_FEATURE_MAX)),
            "hidden_dim": int(kwargs.pop("hidden_dim", DEFAULT_HIDDEN_DIM)),
            "n_classes": int(kwargs.pop("n_classes", DEFAULT_N_CLASSES)),
            "weight_init": kwargs.pop("weight_init", DEFAULT_WEIGHT_INIT),
            "bias_sigma": float(kwargs.pop("bias_sigma", DEFAULT_BIAS_SIGMA)),
            "activation": kwargs.pop("activation", DEFAULT_ACTIVATION),
            "input_center": bool(kwargs.pop("input_center", DEFAULT_INPUT_CENTER)),
            "function_seed": int(kwargs.pop("function_seed", DEFAULT_FUNCTION_SEED)),
            "n_train_per_class": int(kwargs.pop("n_train_per_class", DEFAULT_N_TRAIN_PER_CLASS)),
            "n_test_per_class": int(kwargs.pop("n_test_per_class", DEFAULT_N_TEST_PER_CLASS)),
            "max_resamples": int(kwargs.pop("max_resamples", DEFAULT_MAX_RESAMPLES)),
        }
        if self._params["n_classes"] != self.N_CLASSES:
            raise ValueError(f"n_classes must be {self.N_CLASSES} for "
                             f"{self.TASK_NAME} (the label table is module-level)")
        super().__init__(**kwargs)

    def build_rows(self):
        return build_mlp_bank(**self._params)

    def render(self, features, label):
        return render_example(features, label)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--function-seed", type=int, default=DEFAULT_FUNCTION_SEED)
    ap.add_argument("--print-example", action="store_true")
    args = ap.parse_args()
    t = SyntheticMLPBank(function_seed=args.function_seed)
    t.download()
    m = dict(t._bank_meta)
    m.pop("hidden_mlp")
    print(t.bank_summary().split("  meta ")[0] + f"  meta {m}")
    if args.print_example:
        t.set_fewshot(2, 42)
        q = t.dataset["test"][0]
        print(t.doc_to_text(q) + t.doc_to_target(q))
    return 0


if __name__ == "__main__":
    sys.exit(main())
