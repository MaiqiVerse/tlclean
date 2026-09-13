"""
Yahoo Answers: 10-class topic classification.
Dataset: yahoo_answers_topics on HuggingFace.
Long examples (~130 tokens avg), 10 topic classes.

Format: "Text: {question_title} {question_content} {best_answer}\nTopic: {label}"
"""
import os
import random
import csv
import datasets
from lm_eval.api.task import ConfigurableTask

_LABEL_NAMES = [
    "society_culture",
    "science_mathematics",
    "health",
    "education_reference",
    "computers_internet",
    "sports",
    "business_finance",
    "entertainment_music",
    "family_relationships",
    "politics_government",
]

# Direct parquet URLs (bypass loading script issues)
_DATA_URL = "https://huggingface.co/datasets/yahoo_answers_topics/resolve/main"
_TRAIN_PARQUET = f"{_DATA_URL}/data/train-00000-of-00001.parquet"
_TEST_PARQUET = f"{_DATA_URL}/data/test-00000-of-00001.parquet"

FEWSHOT_DELIMITER = "\n\n"


def _build_dataset():
    """Load Yahoo Answers from parquet files."""
    try:
        ds = datasets.load_dataset("parquet", data_files={
            "train": _TRAIN_PARQUET,
            "test": _TEST_PARQUET,
        })
    except Exception:
        # Fallback: try direct load
        ds = datasets.load_dataset("yahoo_answers_topics", trust_remote_code=True)
    return ds


def _format_text(doc):
    """Combine question title, content, and best answer into one text."""
    parts = []
    for field in ["question_title", "question_content", "best_answer"]:
        val = doc.get(field, "")
        if val and val.strip():
            parts.append(val.strip())
    return " ".join(parts)


def _format_example(doc):
    text = _format_text(doc)
    # Truncate very long examples to ~300 tokens worth of chars (~1200 chars)
    if len(text) > 1200:
        text = text[:1200] + "..."
    label = _LABEL_NAMES[doc["topic"]]
    return f"Text: {text}\nTopic: {label}"


class YahooAnswersTask(ConfigurableTask):
    VERSION = 1.0
    DATASET_NAME = "yahoo_answers"

    def __init__(self, **kwargs):
        self._fewshot_examples = None
        self._num_fewshot = 0
        self._fewshot_seed = 42
        super().__init__(
            config={
                "task": "yahoo_answers",
                "dataset_path": "yahoo_answers",
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

    def download(self, dataset_kwargs=None):
        self.dataset = _build_dataset()

    def set_fewshot(self, num_fewshot: int, seed: int = 42):
        self._num_fewshot = num_fewshot
        self._fewshot_seed = seed
        self._fewshot_examples = None

    def _get_fewshot_examples(self):
        if self._fewshot_examples is None and self._num_fewshot > 0:
            rng = random.Random(self._fewshot_seed)
            train_data = list(self.dataset["train"])
            self._fewshot_examples = rng.sample(train_data, min(self._num_fewshot, len(train_data)))
        return self._fewshot_examples or []

    def _build_fewshot_prefix(self):
        examples = self._get_fewshot_examples()
        if not examples:
            return ""
        demos = [_format_example(ex) for ex in examples]
        return FEWSHOT_DELIMITER.join(demos) + FEWSHOT_DELIMITER

    def doc_to_text(self, doc, *args, **kwargs) -> str:
        prefix = self._build_fewshot_prefix()
        text = _format_text(doc)
        if len(text) > 1200:
            text = text[:1200] + "..."
        return f"{prefix}Text: {text}\nTopic:"

    def doc_to_target(self, doc, *args, **kwargs):
        return f" {_LABEL_NAMES[doc['topic']]}"

    def doc_to_choice(self, doc, *args, **kwargs):
        return [f" {label}" for label in _LABEL_NAMES]
