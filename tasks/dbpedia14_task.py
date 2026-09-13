"""
DBPedia: 14-class Wikipedia entity classification.
Dataset: fancyzhx/dbpedia_14 on HuggingFace.
Medium-long examples (~70 tokens avg), 14 entity type classes.

Format: "Title: {title}\nDescription: {content}\nType: {label}"
"""
import random
import datasets
from lm_eval.api.task import ConfigurableTask

_LABEL_NAMES = [
    "company",
    "educational_institution",
    "artist",
    "athlete",
    "office_holder",
    "mean_of_transportation",
    "building",
    "natural_place",
    "village",
    "animal",
    "plant",
    "album",
    "film",
    "written_work",
]

_DATA_URL = "https://huggingface.co/datasets/fancyzhx/dbpedia_14/resolve/main"
_TRAIN_PARQUET = f"{_DATA_URL}/dbpedia_14/train-00000-of-00001.parquet"
_TEST_PARQUET = f"{_DATA_URL}/dbpedia_14/test-00000-of-00001.parquet"

FEWSHOT_DELIMITER = "\n\n"


def _build_dataset():
    """Load DBPedia from parquet files."""
    try:
        ds = datasets.load_dataset("parquet", data_files={
            "train": _TRAIN_PARQUET,
            "test": _TEST_PARQUET,
        })
    except Exception:
        ds = datasets.load_dataset("fancyzhx/dbpedia_14", trust_remote_code=True)
    return ds


def _format_example(doc):
    title = doc["title"].strip()
    content = doc["content"].strip()
    # Truncate long descriptions to ~200 tokens worth (~800 chars)
    if len(content) > 800:
        content = content[:800] + "..."
    label = _LABEL_NAMES[doc["label"]]
    return f"Title: {title}\nDescription: {content}\nType: {label}"


class DBPedia14Task(ConfigurableTask):
    VERSION = 1.0
    DATASET_NAME = "dbpedia14"

    def __init__(self, **kwargs):
        self._fewshot_examples = None
        self._num_fewshot = 0
        self._fewshot_seed = 42
        super().__init__(
            config={
                "task": "dbpedia14",
                "dataset_path": "dbpedia14",
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
        title = doc["title"].strip()
        content = doc["content"].strip()
        if len(content) > 800:
            content = content[:800] + "..."
        return f"{prefix}Title: {title}\nDescription: {content}\nType:"

    def doc_to_target(self, doc, *args, **kwargs):
        return f" {_LABEL_NAMES[doc['label']]}"

    def doc_to_choice(self, doc, *args, **kwargs):
        return [f" {label}" for label in _LABEL_NAMES]
