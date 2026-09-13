"""Monk-1/2/3 as shared-bank tasks: the UCI ``.train`` file is the
demonstration bank, ``.test − .train`` is the test pool (tasks/monk/,
UCI dataset 70; rows ``<class> a1..a6 <id>``).

Each rule is one fixed Boolean concept over the same 432-robot universe:

  rule 1  (a1 = a2) or (a5 = 1)
  rule 2  exactly two of the six attributes take their first value
  rule 3  (a5 = 3 and a4 = 1) or (a5 != 4 and a2 != 3)   [5 % label noise in .train]

so the per-prompt task (monk_task, a fresh balanced draw per query) and
this one share the concept and the rendering (monk_task.render_example);
only the demonstration structure differs. Sizes as released:

  rule   .train (c0 / c1)   .test − .train
    1     124 (62 / 62)          308
    2     169 (105 / 64)         263
    3     122 (62 / 60)          310

Both classes clear tools.prereg_config.MIN_TRAIN_PER_CLASS in every rule,
and every pool exceeds the 250 test queries a seed draws.

Task names (the yaml files): monk_bank_r1_per_class, monk_bank_r2_per_class,
monk_bank_r3_per_class.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monk_task import (  # noqa: E402
    MONK_DATA_DIR, _features_key, _load_monk_file, render_example,
)
from shared_bank_task import SharedBankTask, make_row  # noqa: E402

N_UNIVERSE = 432


def load_monk_bank(rule_id: int):
    """(train_rows, test_rows) for one rule: the .train file, and the .test
    rows whose features are not in .train (the per-prompt task's 'heldout')."""
    train = _load_monk_file(MONK_DATA_DIR / f"monks-{rule_id}.train")
    test = _load_monk_file(MONK_DATA_DIR / f"monks-{rule_id}.test")
    if len(test) != N_UNIVERSE:
        raise RuntimeError(f"monks-{rule_id}.test: {len(test)} rows, expected "
                           f"{N_UNIVERSE} (the whole attribute universe)")
    train_keys = {_features_key(r["features"]) for r in train}
    if len(train_keys) != len(train):
        raise RuntimeError(f"monks-{rule_id}.train repeats a robot")
    heldout = [r for r in test if _features_key(r["features"]) not in train_keys]
    return ([make_row(r["features"], r["label"]) for r in train],
            [make_row(r["features"], r["label"]) for r in heldout])


class MonkBank(SharedBankTask):
    N_CLASSES = 2
    DATASET_NAME = "monk_bank"
    RULE_ID = 1

    def __init__(self, **kwargs):
        self._rule_id = int(kwargs.pop("rule_id", self.RULE_ID))
        if self._rule_id not in (1, 2, 3):
            raise ValueError(f"rule_id must be 1, 2 or 3, got {self._rule_id}")
        super().__init__(**kwargs)

    def build_rows(self):
        train, test = load_monk_bank(self._rule_id)
        return train, test, {"rule_id": self._rule_id, "source": "UCI dataset 70"}

    def render(self, features, label):
        return render_example(features, label)


class MonkBankR1(MonkBank):
    TASK_NAME = "monk_bank_r1_per_class"
    RULE_ID = 1


class MonkBankR2(MonkBank):
    TASK_NAME = "monk_bank_r2_per_class"
    RULE_ID = 2


class MonkBankR3(MonkBank):
    TASK_NAME = "monk_bank_r3_per_class"
    RULE_ID = 3


if __name__ == "__main__":
    for cls in (MonkBankR1, MonkBankR2, MonkBankR3):
        t = cls()
        t.download()
        print(t.bank_summary())
        t.set_fewshot(2, 42)
        demos = t._get_fewshot_examples()
        print(f"  K=2 seed 42 prefix: {[(d['label'], d['text']) for d in demos]}")
    print()
    print(MonkBankR1().render([1, 1, 1, 1, 3, 1], "A"))
