"""The method line's prompt renderer, driven by the task instead of TREC.

`tools/probe_prototype_shrinkage.render_demo` spells "Question: ...\\nType: ..."
and every receiver, probe and baseline runner rendered through it, so the
section-13.5 constructions could only ever run on TREC. This module renders
through the same per-task formatters `experiments/data_calibration.py` uses
(`test_unified_task_learning._get_task_info`), which is what makes the rebuilt
prefix reproduce the calibration prompt on every task -- and `verify_prefix`
still checks that it does, byte for byte, before any model is loaded.

Documents, not strings. The formatters take the task's own document (dbpedia
reads title and content, yahoo joins three fields), so the lookups here map
a query id to the DOCUMENT and the prefix is built from demonstration
documents. On TREC the result is byte-identical to the old renderer
(tools/test_prompt_render.py pins it).

    r = TaskRenderer("banking77_per_class")
    blocks = r.build_prefix(demo_docs, abstract_labels)      # [(class, doc)]
    prompt = r.render_prompt(blocks, query_doc)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prereg_ids import content_hash  # noqa: E402
from tools.prereg_task import doc_fields, row_text, test_label_field  # noqa: E402


class TaskRenderer:
    """One task's demo / query formatters, from the calibration's own table."""

    def __init__(self, task_name):
        from test_unified_task_learning import _get_task_info
        self.task_name = task_name
        (self.label_field, self.label_names, make_fmt,
         self.query_fn) = _get_task_info(task_name)
        # The label is passed explicitly, so the mapping the maker takes is
        # irrelevant; every maker in _get_task_info ignores it.
        self.fmt = make_fmt(None)
        self.text_field, _ = doc_fields(task_name)

    def render_demo(self, doc, label):
        return self.fmt(doc, label)

    def render_query(self, doc):
        return self.query_fn(doc)

    def build_prefix(self, demo_docs, abstract_labels):
        """The fixed prefix's blocks, in the task's demo order.

        `demo_docs` is [(class_idx, doc)] as tools.prereg_task.prefix_demo_docs
        returns it; `abstract_labels[class_idx]` is the surface written.
        """
        return [self.render_demo(d, abstract_labels[int(c)]) for c, d in demo_docs]

    def render_prompt(self, prefix_blocks, query_doc):
        return "\n\n".join(list(prefix_blocks) + [self.render_query(query_doc)])


def build_doc_lookup(task, task_name):
    """content_hash -> document, over BOTH native splits.

    The manifest keys queries by content_hash(class, text) where `text` is the
    row text tools.prereg_task.train_rows / test_rows report (the raw field,
    or the joined fields of a multi-field task). Validation queries come from
    TRAIN and test queries from TEST, so both splits are indexed.
    """
    from tools.prereg_task import test_docs_of, train_docs_of
    out = {}
    for c, t, d in train_docs_of(task, task_name):
        out[content_hash(int(c), t)] = d
    for c, t, d in test_docs_of(task, task_name):
        out[content_hash(int(c), t)] = d
    return out


def query_doc_of(lookup, query_id, class_idx=None):
    """The document behind a manifest query, refusing to guess."""
    d = lookup.get(query_id)
    if d is None:
        raise SystemExit(
            f"query {str(query_id)[:12]}"
            + (f" (class {class_idx})" if class_idx is not None else "")
            + " is not in the task's train or test split. The manifest and "
            "the task have drifted apart; do not guess a text.")
    return d


__all__ = ["TaskRenderer", "build_doc_lookup", "query_doc_of", "row_text",
           "test_label_field"]
