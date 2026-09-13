"""
BANKING77: 77 fine-grained banking intent classification.
Dataset: PolyAI/banking77 on HuggingFace.
13,083 customer service queries, 77 intent classes.

Format: "Query: {text}\nIntent: {label_name}"

Label names are loaded dynamically from the dataset features to ensure correct ordering.
"""
import random
import datasets
from lm_eval.api.task import ConfigurableTask

FEWSHOT_DELIMITER = "\n\n"

# Will be populated from dataset features at download time
_LABEL_NAMES = None


def _build_dataset():
    """Load BANKING77 from HuggingFace and extract label names."""
    global _LABEL_NAMES
    ds = datasets.load_dataset("legacy-datasets/banking77")
    # Extract label names from features (guaranteed correct ordering)
    _LABEL_NAMES = ds["train"].features["label"].names
    return ds


def _format_example(doc):
    return f"Query: {doc['text']}\nIntent: {_LABEL_NAMES[doc['label']]}"


class Banking77Task(ConfigurableTask):
    VERSION = 1.0
    DATASET_NAME = "banking77"

    def __init__(self, **kwargs):
        self._fewshot_examples = None
        self._num_fewshot = 0
        self._fewshot_seed = 42
        super().__init__(
            config={
                "task": "banking77",
                "dataset_path": "banking77",
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
        return f"{prefix}Query: {doc['text']}\nIntent:"

    def doc_to_target(self, doc, *args, **kwargs):
        return f" {_LABEL_NAMES[doc['label']]}"

    def doc_to_choice(self, doc, *args, **kwargs):
        return [f" {label}" for label in _LABEL_NAMES]
