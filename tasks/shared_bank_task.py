"""The shared-bank form of the generated tasks: ONE hidden concept per task
instance, a fixed train bank and a disjoint test pool, k-per-class demos
drawn from the bank -- the shape the method line reads (tools/method_status
--task-check) and the shape trec_fine / banking77 / ... already have.

The per-prompt tasks (synthetic_linear_task, synthetic_mlp_task, monk_task)
resample the hidden function for EVERY query and bake that query's own
balanced demonstrations into its test doc. That is the right object for the
rediscovery chain (a fresh function per prompt is what makes the in-context
learning claim generic) and the wrong one for the section-13.5
constructions, which read a memory built offline from a shared
demonstration bank and score queries the bank never showed: with a function
per prompt there is no bank and no offline memory. So the shared-bank tasks
fix the function once (Monk's three published rules are already fixed;
the two synthetic families take a `function_seed`) and expose an ordinary
train / test split. The rendering is byte-identical to the per-prompt
task of the same family (the same `render_example`), so a model reads the
same prompt grammar in both forms.

A document is ``{"features": [ints], "text": "<the ints, space-joined>",
"label": int}``. `features` is what the family's renderer consumes; `text`
is the row's identity, the string tools/prereg_task hashes and the
validation reservation compares (a list would not hash, and the joined
form is what tools.prereg_task.text_key would produce for it anyway).

The demonstration draw is trec_fine_per_class's, verbatim in its semantics:
distinct (class, text) rows, the optional `_allowed_classes` narrowing
(tools/prereg_task.load_task), the validation reservation withheld BEFORE
the draw (`_excluded_docs`, tasks/per_class_draw.withhold_reserved), the
classes walked in sorted order, K per class, then the demos shuffled so the
classes interleave. A class that cannot supply K is an error here rather
than a warning: the banks are sized (tasks/*_bank_task.py) so this cannot
happen below K = 20 with the registered reservation, and a thinner class
would make the balanced prefix silently unbalanced.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from collections import defaultdict
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets  # type: ignore
from lm_eval.api.task import ConfigurableTask  # type: ignore

from per_class_draw import withhold_reserved

_logger = logging.getLogger(__name__)

FEWSHOT_DELIMITER = "\n\n"
DEFAULT_LABEL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def canonical_text(features) -> str:
    """The row identity: the integer features, space-joined ("3 7 1 9 2")."""
    return " ".join(str(int(v)) for v in features)


def make_row(features, label: int) -> dict[str, Any]:
    feats = [int(v) for v in features]
    return {"features": feats, "text": canonical_text(feats), "label": int(label)}


class SharedBankTask(ConfigurableTask):
    """Base: subclasses define TASK_NAME, N_CLASSES, `build_rows()` and
    `render(features, label)`."""

    VERSION = 1.0
    TASK_NAME = "shared_bank"          # the lm_eval task name (== the yaml's)
    DATASET_NAME = "shared_bank"
    N_CLASSES = 2
    DEFAULT_SEED = 42

    def __init__(self, **kwargs):
        self._labels_str = kwargs.pop("labels", DEFAULT_LABEL_CHARS)
        self._labels: list[str] = list(self._labels_str[: self.N_CLASSES])
        self._num_fewshot = 0
        self._fewshot_seed = self.DEFAULT_SEED
        self._fewshot_examples = None
        self._bank_meta: dict[str, Any] = {}
        kwargs.pop("task", None)
        super().__init__(
            config={
                "task": self.TASK_NAME,
                "dataset_path": self.DATASET_NAME,
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

    # ---- what a family supplies --------------------------------------------

    def build_rows(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """(train_rows, test_rows, bank_meta); rows from `make_row`."""
        raise NotImplementedError

    def render(self, features, label: str | None) -> str:
        """The family's `render_example`; `label=None` is the query stub."""
        raise NotImplementedError

    # ---- dataset -------------------------------------------------------------

    def download(self, dataset_kwargs=None):
        train_rows, test_rows, meta = self.build_rows()
        self._check_rows(train_rows, test_rows)
        self._bank_meta = dict(meta)
        self.dataset = datasets.DatasetDict({
            "train": datasets.Dataset.from_list(train_rows),
            "test": datasets.Dataset.from_list(test_rows),
        })

    def _check_rows(self, train_rows, test_rows):
        """The bank's invariants, checked where the rows are made rather than
        trusted downstream: distinct rows within a split, no row in both
        splits, every class in both splits, labels inside the label space."""
        name = self.TASK_NAME
        for split, rows in (("train", train_rows), ("test", test_rows)):
            if not rows:
                raise ValueError(f"{name}: the {split} split is empty")
            texts = [r["text"] for r in rows]
            if len(set(texts)) != len(texts):
                raise ValueError(f"{name}: the {split} split repeats a row")
            bad = sorted({r["label"] for r in rows} - set(range(self.N_CLASSES)))
            if bad:
                raise ValueError(f"{name}: {split} labels {bad} outside "
                                 f"0..{self.N_CLASSES - 1}")
        both = {r["text"] for r in train_rows} & {r["text"] for r in test_rows}
        if both:
            raise ValueError(f"{name}: {len(both)} rows are in BOTH splits, "
                             f"e.g. {sorted(both)[:3]}; the test pool must be "
                             "disjoint from the demonstration bank")
        for split, rows in (("train", train_rows), ("test", test_rows)):
            missing = sorted(set(range(self.N_CLASSES)) - {r["label"] for r in rows})
            if missing:
                raise ValueError(f"{name}: classes {missing} have no {split} row")

    # ---- fewshot: the per-class draw ----------------------------------------

    def set_fewshot(self, num_fewshot: int, seed: int = 42):
        self._num_fewshot = int(num_fewshot)
        self._fewshot_seed = int(seed)
        self._fewshot_examples = None

    def _get_fewshot_examples(self):
        if self._fewshot_examples is None and self._num_fewshot > 0:
            k = self._num_fewshot
            rng = random.Random(self._fewshot_seed)
            by_class = defaultdict(list)
            seen = set()
            for doc in self.dataset["train"]:
                key = (int(doc["label"]), doc["text"])
                if key in seen:
                    continue
                seen.add(key)
                by_class[int(doc["label"])].append(doc)

            allowed = getattr(self, "_allowed_classes", None)
            if allowed is not None:
                allowed = {int(c) for c in allowed}
                missing = allowed - set(by_class)
                if missing:
                    raise ValueError(
                        f"_allowed_classes names {sorted(missing)}, which have "
                        "no training example. A class with no demo cannot be "
                        "in the decision space.")
                by_class = {c: v for c, v in by_class.items() if c in allowed}

            by_class = withhold_reserved(
                by_class, getattr(self, "_excluded_docs", None),
                "label", "text", _logger)

            demos = []
            thin = {}
            for label in sorted(by_class.keys()):
                candidates = list(by_class[label])
                rng.shuffle(candidates)
                if len(candidates) < k:
                    thin[label] = len(candidates)
                demos.extend(candidates[:k])
            if thin:
                raise ValueError(
                    f"{self.TASK_NAME}: classes below k={k} after the "
                    f"reservation: {thin}. The bank is sized so this cannot "
                    "happen at the registered K; a thinner class would make "
                    "the balanced prefix unbalanced without a trace.")
            rng.shuffle(demos)
            self._fewshot_examples = demos
            _logger.info(f"[per_class] k={k}, classes={len(by_class)}, "
                         f"total_shots={len(demos)}")
        return self._fewshot_examples or []

    def _build_fewshot_prefix(self) -> str:
        examples = self._get_fewshot_examples()
        if not examples:
            return ""
        demos = [self.render(ex["features"], self._labels[int(ex["label"])])
                 for ex in examples]
        return FEWSHOT_DELIMITER.join(demos) + FEWSHOT_DELIMITER

    # ---- prompt --------------------------------------------------------------

    def doc_to_text(self, doc, *args, **kwargs) -> str:
        return self._build_fewshot_prefix() + self.render(doc["features"], None)

    def doc_to_target(self, doc, *args, **kwargs) -> str:
        return f" {self._labels[int(doc['label'])]}"

    def doc_to_choice(self, doc, *args, **kwargs):
        return [f" {label}" for label in self._labels]

    # ---- inspection ----------------------------------------------------------

    def bank_summary(self) -> str:
        counts = {}
        for split in ("train", "test"):
            c = defaultdict(int)
            for d in self.dataset[split]:
                c[int(d["label"])] += 1
            counts[split] = dict(sorted(c.items()))
        return (f"{self.TASK_NAME}: train {len(self.dataset['train'])} "
                f"{counts['train']}  test {len(self.dataset['test'])} "
                f"{counts['test']}  meta {self._bank_meta}")
