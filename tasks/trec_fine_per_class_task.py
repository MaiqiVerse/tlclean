"""TREC 50-class with k-per-class balanced fewshot. num_fewshot = k per class."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random
import logging
from collections import defaultdict
from trec_fine_task import TRECFine

_logger = logging.getLogger(__name__)


class TRECFinePerClass(TRECFine):

    def __init__(self, **kwargs):
        kwargs.pop("task", None)
        super().__init__(**kwargs)
        self.config.task = "trec_fine_per_class"

    def _get_fewshot_examples(self):
        if self._fewshot_examples is None and self._num_fewshot > 0:
            k = self._num_fewshot
            rng = random.Random(self._fewshot_seed)
            train_data = list(self.dataset["train"])

            # TREC's train split repeats some questions verbatim. Two rows with
            # the same (class, text) are the SAME example, and letting both in
            # would put a demonstration's identical twin in the prompt -- which
            # for a leave-one-out class prototype means mu_{-i} contains a copy
            # of v_i itself, pulling the "prototype" toward the very example it
            # is supposed to exclude.
            by_class = defaultdict(list)
            seen, n_dup = set(), 0
            for doc in train_data:
                key = (int(doc["fine_label"]), doc["text"])
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                by_class[doc["fine_label"]].append(doc)
            if n_dup:
                _logger.info(f"[per_class] {n_dup} duplicate (class, text) "
                             "train rows dropped before the draw")

            # Optional class restriction, set by the caller (see
            # tools/prereg_task.load_task). The default is every train class;
            # the preregistered setting narrows it to the classes that also
            # occur in the test split, because a demonstration of a class no
            # query can ever have costs prompt length and dilutes the candidate
            # space while being unanswerable by construction.
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

            # Documents reserved for validation, as (fine_label, text) pairs.
            # Validation is drawn from the train split FIRST and demos come out
            # of what is left, which is the only ordering that makes the
            # arithmetic safe: with n=15 and three independent K=5 draws the
            # union covers about 10.6 examples on average, leaving ~4.4 -- too
            # close to the 4 validation queries a class must supply. Reserving
            # first turns "usually enough" into "enough by construction", and
            # removes a circularity: a filter defined on what the draw happened
            # to leave over would change the draw, which would change the
            # filter.
            excluded = getattr(self, "_excluded_docs", None)
            if excluded:
                excluded = {(int(c), t) for c, t in excluded}
                before = sum(len(v) for v in by_class.values())
                by_class = {c: [d for d in v
                                if (int(d["fine_label"]), d["text"])
                                not in excluded]
                            for c, v in by_class.items()}
                after = sum(len(v) for v in by_class.values())
                _logger.info(f"[per_class] reserved for validation: "
                             f"{before - after} train docs withheld from demos")
                empty = sorted(c for c, v in by_class.items() if not v)
                if empty:
                    raise ValueError(
                        f"class(es) {empty} have no training example left after "
                        "the validation reservation. Raise the minimum-train "
                        "threshold rather than letting a class lose its demos.")

            demos = []
            thin = {}
            for label in sorted(by_class.keys()):
                candidates = list(by_class[label])
                rng.shuffle(candidates)
                if len(candidates) < k:
                    thin[label] = len(candidates)
                demos.extend(candidates[:k])

            rng.shuffle(demos)
            self._fewshot_examples = demos
            if thin:
                # Not a warning to be scrolled past: N_c is load-bearing for a
                # leave-one-out prototype, and N_c = 1 makes it undefined.
                _logger.info(f"[per_class] classes below k={k}: {thin}")
                if min(thin.values()) < 2:
                    raise ValueError(
                        f"class(es) {[c for c, n in thin.items() if n < 2]} "
                        "have a single training example, so a leave-one-out "
                        "class mean is undefined for them.")
            _logger.info(f"[per_class] k={k}, classes={len(by_class)}, "
                         f"total_shots={len(demos)}")
        return self._fewshot_examples or []