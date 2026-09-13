"""
CLINIC150 (CLINC-OOS): 150 fine-grained intent classification across 10 domains.
Dataset: clinc/clinc_oos on HuggingFace (config="plus" for full in-scope data).
15,000 training examples (100 per intent), 4,500 test.

We use only the 150 in-scope intents (exclude out-of-scope "oos" class).

Format: "Query: {text}\nIntent: {label_name}"

Label names are loaded dynamically from the dataset features.
"""
import random
import datasets
from lm_eval.api.task import ConfigurableTask

FEWSHOT_DELIMITER = "\n\n"

# Will be populated from dataset features at download time
_LABEL_NAMES = None
_OOS_LABEL_IDX = None  # index of "oos" class to exclude


def _build_dataset():
    """Load CLINC150 from HuggingFace, filtering out OOS examples."""
    global _LABEL_NAMES, _OOS_LABEL_IDX
    # "plus" config has 150 in-scope + 1 OOS class, with more training data
    ds = datasets.load_dataset("clinc/clinc_oos", "plus")
    
    # Get all label names from features
    all_label_names = ds["train"].features["intent"].names
    
    # Find OOS label index
    _OOS_LABEL_IDX = None
    for i, name in enumerate(all_label_names):
        if name.lower() == "oos":
            _OOS_LABEL_IDX = i
            break
    
    # Build in-scope label names (exclude OOS)
    _LABEL_NAMES = [name for i, name in enumerate(all_label_names) if i != _OOS_LABEL_IDX]
    
    # Build old_idx → new_idx mapping (re-index after removing OOS)
    old_to_new = {}
    new_idx = 0
    for i in range(len(all_label_names)):
        if i == _OOS_LABEL_IDX:
            continue
        old_to_new[i] = new_idx
        new_idx += 1
    
    # Filter out OOS and re-map labels
    def filter_and_remap(example):
        return example["intent"] != _OOS_LABEL_IDX
    
    def remap_label(example):
        example["label"] = old_to_new[example["intent"]]
        return example
    
    filtered = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        split_data = ds[split].filter(filter_and_remap)
        split_data = split_data.map(remap_label)
        filtered[split] = split_data
    
    return filtered


def _format_example(doc):
    return f"Query: {doc['text']}\nIntent: {_LABEL_NAMES[doc['label']]}"


class Clinc150Task(ConfigurableTask):
    VERSION = 1.0
    DATASET_NAME = "clinc150"

    def __init__(self, **kwargs):
        self._fewshot_examples = None
        self._num_fewshot = 0
        self._fewshot_seed = 42
        super().__init__(
            config={
                "task": "clinc150",
                "dataset_path": "clinc150",
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
