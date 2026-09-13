"""
TREC-fine: 50 fine-grained question classification.
Same data as TREC coarse (6 classes), but uses the fine-grained label (50 classes).

Format: "Question: {text}\nType: {fine_label}"
"""
import os
import random
import datasets
from lm_eval.api.task import ConfigurableTask

_TRAIN_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
_TEST_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"

# All 50 fine-grained labels (COARSE:fine format in raw data)
_FINE_LABELS = [
    "abbreviation_abbreviation", "abbreviation_expansion",
    "entity_animal", "entity_body", "entity_color", "entity_creation",
    "entity_currency", "entity_disease", "entity_event", "entity_food",
    "entity_instrument", "entity_language", "entity_letter", "entity_other",
    "entity_plant", "entity_product", "entity_religion", "entity_sport",
    "entity_substance", "entity_symbol", "entity_technique", "entity_term",
    "entity_vehicle", "entity_word",
    "description_definition", "description_description", "description_manner",
    "description_reason",
    "human_description", "human_group", "human_individual", "human_title",
    "location_city", "location_country", "location_mountain", "location_other",
    "location_state",
    "numeric_code", "numeric_count", "numeric_date", "numeric_distance",
    "numeric_money", "numeric_order", "numeric_other", "numeric_percent",
    "numeric_period", "numeric_speed", "numeric_temperature", "numeric_size",
    "numeric_weight",
]

# The raw "COARSE:fine" strings as they actually appear in TREC's .label files,
# in the SAME ORDER as _FINE_LABELS above.
#
# WHY THIS EXISTS. _parse_line used to build a lookup key by spelling the fine
# name out -- "entity_" + fine -- and check it against _FINE_LABELS. But TREC
# writes fine labels ABBREVIATED ("ENTY:cremat", "HUM:ind", "NUM:dist") while
# _FINE_LABELS spells them ("entity_creation", "human_individual",
# "numeric_distance"). Only 30 of the 50 happened to coincide, and every row of
# the other 20 was silently dropped by the `return None` below: 2582 of 5452
# train rows and 247 of 500 test rows. The project's "30 train classes" was that
# accident, not a property of TREC.
#
# Keeping this list in _FINE_LABELS' order means the 30 classes that already
# worked KEEP THEIR INDEX; the other 20 simply start arriving at the index they
# always should have had. Nothing that reads _FINE_LABELS by name or position
# needs to change.
_FINE_KEYS = [
    "ABBR:abb", "ABBR:exp",
    "ENTY:animal", "ENTY:body", "ENTY:color", "ENTY:cremat",
    "ENTY:currency", "ENTY:dismed", "ENTY:event", "ENTY:food",
    "ENTY:instru", "ENTY:lang", "ENTY:letter", "ENTY:other",
    "ENTY:plant", "ENTY:product", "ENTY:religion", "ENTY:sport",
    "ENTY:substance", "ENTY:symbol", "ENTY:techmeth", "ENTY:termeq",
    "ENTY:veh", "ENTY:word",
    "DESC:def", "DESC:desc", "DESC:manner",
    "DESC:reason",
    "HUM:desc", "HUM:gr", "HUM:ind", "HUM:title",
    "LOC:city", "LOC:country", "LOC:mount", "LOC:other",
    "LOC:state",
    "NUM:code", "NUM:count", "NUM:date", "NUM:dist",
    "NUM:money", "NUM:ord", "NUM:other", "NUM:perc",
    "NUM:period", "NUM:speed", "NUM:temp", "NUM:volsize",
    "NUM:weight",
]
assert len(_FINE_KEYS) == len(_FINE_LABELS) == 50, (
    f"{len(_FINE_KEYS)} keys vs {len(_FINE_LABELS)} readable names; they are "
    "positional partners and must stay aligned")
assert len(set(_FINE_KEYS)) == 50, "duplicate TREC key"

_FINE_TO_IDX = {key: i for i, key in enumerate(_FINE_KEYS)}
_IDX_TO_NAME = {i: name for i, name in enumerate(_FINE_LABELS)}

FEWSHOT_DELIMITER = "\n\n"


def _parse_line(line: str):
    """Parse 'DESC:manner How did serfdom develop ...' → text + fine label index."""
    line = line.strip()
    if not line:
        return None
    coarse_fine, text = line.split(" ", 1)
    # Look the RAW key up directly. Every line of TREC carries one of the 50
    # keys in _FINE_KEYS, so an unknown key means the file is not the TREC
    # release this task expects -- worth failing on rather than dropping half
    # the corpus in silence, which is what the previous spelled-out key did.
    if coarse_fine not in _FINE_TO_IDX:
        raise ValueError(
            f"unknown TREC label {coarse_fine!r}. The 50 expected keys are in "
            "_FINE_KEYS; a missing one means the data file is not the expected "
            "release. Dropping the row instead would silently shrink the task.")
    return {"text": text, "fine_label": _FINE_TO_IDX[coarse_fine]}


def _load_split(url: str):
    import urllib.request
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "trec_raw")
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(fname):
        urllib.request.urlretrieve(url, fname)
    rows = []
    with open(fname, "r", encoding="latin-1") as f:
        for line in f:
            parsed = _parse_line(line)
            if parsed:
                rows.append(parsed)
    return rows


def _build_dataset():
    train_rows = _load_split(_TRAIN_URL)
    test_rows = _load_split(_TEST_URL)
    return datasets.DatasetDict({
        "train": datasets.Dataset.from_list(train_rows),
        "test": datasets.Dataset.from_list(test_rows),
    })


def _format_example(doc):
    return f"Question: {doc['text']}\nType: {_FINE_LABELS[doc['fine_label']]}"


class TRECFine(ConfigurableTask):
    VERSION = 1.0
    DATASET_NAME = "trec_fine"

    def __init__(self, **kwargs):
        self._fewshot_examples = None
        self._num_fewshot = 0
        self._fewshot_seed = 42
        super().__init__(
            config={
                "task": "trec_fine",
                "dataset_path": "trec_fine",
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
        return f"{prefix}Question: {doc['text']}\nType:"

    def doc_to_target(self, doc, *args, **kwargs):
        return f" {_FINE_LABELS[doc['fine_label']]}"

    def doc_to_choice(self, doc, *args, **kwargs):
        return [f" {label}" for label in _FINE_LABELS]
