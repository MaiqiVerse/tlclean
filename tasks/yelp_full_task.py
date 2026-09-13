"""
Yelp Full: 5-class review sentiment classification (1-5 stars).
Dataset: yelp_review_full on HuggingFace.
Long examples (~180 tokens avg), 5 star-rating classes.

Format: "Review: {text}\nRating: {label}"
"""
import random
import datasets
from lm_eval.api.task import ConfigurableTask

_LABEL_NAMES = [
    "1_star",
    "2_stars",
    "3_stars",
    "4_stars",
    "5_stars",
]

_DATA_URL = "https://huggingface.co/datasets/yelp_review_full/resolve/main"
_TRAIN_PARQUET = f"{_DATA_URL}/yelp_review_full/train-00000-of-00001.parquet"
_TEST_PARQUET = f"{_DATA_URL}/yelp_review_full/test-00000-of-00001.parquet"

FEWSHOT_DELIMITER = "\n\n"


def _build_dataset():
    """Load Yelp Full from parquet files."""
    try:
        ds = datasets.load_dataset("parquet", data_files={
            "train": _TRAIN_PARQUET,
            "test": _TEST_PARQUET,
        })
    except Exception:
        ds = datasets.load_dataset("yelp_review_full", trust_remote_code=True)
    return ds


def _format_example(doc):
    text = doc["text"].strip()
    # Truncate very long reviews to ~300 tokens worth (~1200 chars)
    if len(text) > 1200:
        text = text[:1200] + "..."
    label = _LABEL_NAMES[doc["label"]]
    return f"Review: {text}\nRating: {label}"


class YelpFullTask(ConfigurableTask):
    VERSION = 1.0
    DATASET_NAME = "yelp_full"

    def __init__(self, **kwargs):
        self._fewshot_examples = None
        self._num_fewshot = 0
        self._fewshot_seed = 42
        super().__init__(
            config={
                "task": "yelp_full",
                "dataset_path": "yelp_full",
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
        text = doc["text"].strip()
        if len(text) > 1200:
            text = text[:1200] + "..."
        return f"{prefix}Review: {text}\nRating:"

    def doc_to_target(self, doc, *args, **kwargs):
        return f" {_LABEL_NAMES[doc['label']]}"

    def doc_to_choice(self, doc, *args, **kwargs):
        return [f" {label}" for label in _LABEL_NAMES]
