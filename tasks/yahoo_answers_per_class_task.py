"""Yahoo Answers with k-per-class balanced fewshot. num_fewshot = k per class."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random
import logging
from collections import defaultdict
from yahoo_answers_task import YahooAnswersTask

_logger = logging.getLogger(__name__)


class YahooAnswersPerClass(YahooAnswersTask):

    def __init__(self, **kwargs):
        kwargs.pop("task", None)
        super().__init__(**kwargs)
        self.config.task = "yahoo_answers_per_class"

    def _get_fewshot_examples(self):
        if self._fewshot_examples is None and self._num_fewshot > 0:
            k = self._num_fewshot
            rng = random.Random(self._fewshot_seed)
            train_data = list(self.dataset["train"])

            by_class = defaultdict(list)
            for doc in train_data:
                by_class[doc["topic"]].append(doc)

            # Documents reserved for validation (tools/prereg_task.load_task,
            # experiments/data_calibration.py --exclude-validation) are
            # withheld before the draw, as trec_fine_per_class does; with no
            # reservation set this changes nothing (tasks/per_class_draw).
            from per_class_draw import withhold_reserved
            by_class = withhold_reserved(
                by_class, getattr(self, "_excluded_docs", None),
                'topic', ('question_title', 'question_content', 'best_answer'), _logger)

            demos = []
            for label in sorted(by_class.keys()):
                candidates = list(by_class[label])
                rng.shuffle(candidates)
                if len(candidates) < k:
                    _logger.warning(f"Class {label}: only {len(candidates)} available, requested {k}")
                demos.extend(candidates[:k])

            rng.shuffle(demos)
            self._fewshot_examples = demos
            _logger.info(f"[per_class] k={k}, classes={len(by_class)}, total_shots={len(demos)}")
        return self._fewshot_examples or []
