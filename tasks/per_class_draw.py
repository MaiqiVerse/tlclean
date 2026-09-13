"""The validation reservation for every k-per-class task.

`trec_fine_per_class_task` withholds the documents reserved for validation
(`self._excluded_docs`, set by tools/prereg_task.load_task and
experiments/data_calibration.py --exclude-validation) before it draws its
demonstrations. The other per-class tasks did not, so on them a validation
query could sit in its own prompt as a demonstration and nothing would raise
-- the manifest's disjointness check is the only thing that would notice, and
only for the seeds it was asked about. This is TREC's block, factored so every
per-class task withholds the same way; with no reservation set it changes
nothing, so every existing draw is reproduced exactly.

The reservation is a set of (class_idx, text) pairs where `text` is what
tools.prereg_task.row_text reports for the document: the raw field for a
single-field task, the fields joined with a newline for a multi-field one.
"""

from __future__ import annotations


def row_text(doc, text_fields):
    fields = text_fields if isinstance(text_fields, tuple) else (text_fields,)
    if len(fields) == 1:
        return doc[fields[0]]
    return "\n".join(str(doc.get(f, "")) for f in fields)


def withhold_reserved(by_class, excluded, label_field, text_fields, logger=None):
    """Drop every reserved document from the per-class pools.

    Raises if a class is left with nothing: a class with no demonstration
    cannot be in the decision space, and silently thinning it is how a
    balanced draw stops being balanced.
    """
    if not excluded:
        return by_class
    excluded = {(int(c), t) for c, t in excluded}
    before = sum(len(v) for v in by_class.values())
    kept = {c: [d for d in v
                if (int(d[label_field]), row_text(d, text_fields)) not in excluded]
            for c, v in by_class.items()}
    after = sum(len(v) for v in kept.values())
    if logger is not None:
        logger.info(f"[per_class] reserved for validation: {before - after} "
                    "train docs withheld from demos")
    empty = sorted(c for c, v in kept.items() if not v)
    if empty:
        raise ValueError(
            f"class(es) {empty} have no training example left after the "
            "validation reservation. Raise the minimum-train threshold rather "
            "than letting a class lose its demos.")
    return kept
