"""The two hashes the preregistration identifies queries by. No dependencies.

Sections 2.1(2) and 2.4(1) both hash `(class_idx, text)`, for different jobs, and
the whole disjointness argument rests on the two hashes agreeing about what a
query IS. If `build_query_manifest.py` and `method_discovery_split.py` each
carried their own copy, a change to one -- a strip(), a lower(), a different
separator -- would make the intersection test compare two different things and
still come back empty. That is the same shape as the class-space bug that had
two scripts writing different answers for one convention (working rules 2.6.2), so
there is exactly one definition and both import it.

RAW BYTES, DELIBERATELY. Section 2.1(5) pins the hash input to the text field's
raw UTF-8 bytes with no Unicode normalization, case folding or whitespace
handling. It even notes that the name `<normalized_text>` is kept only for
compatibility and in fact means raw text. Determinism comes from the data
file's own registered SHA256, not from a normalization step -- which is the
stronger arrangement, because a normalizer is one more thing that can differ
between two machines.
"""

from __future__ import annotations

import hashlib

QUERY_TAG = "trec_fine"          # section 2.1(2): the query_id
VAL_TAG = "trec_fine_mval"       # section 2.1: validation ordering


def _h(tag: str, class_idx: int, raw_text: str) -> str:
    """SHA256 over tag \\0 class \\0 raw UTF-8 bytes."""
    return hashlib.sha256((f"{tag}\0{class_idx}\0".encode("utf-8")
                           + raw_text.encode("utf-8"))).hexdigest()


def content_hash(class_idx: int, raw_text: str) -> str:
    """query_id (section 2.1(2)), and the identity used for cross-split
    exclusion. One value, two jobs, on purpose: the exclusion is only
    meaningful if it compares the same thing the manifest keys on.

    The exclusion it serves is now validation-vs-test-draw only. It used to
    also keep the discovery pool apart from both; that pool was abolished
    (section 14.0a) when head selection moved onto validation."""
    return _h(QUERY_TAG, class_idx, raw_text)


def validation_hash(prefix_seed: int, class_idx: int, raw_text: str) -> str:
    """Ordering key for the validation draw, PER PREFIX SEED.

    A different tag from `content_hash`, so the ordering carries no
    information about the identity the manifest keys on.

    `prefix_seed` leads. Validation is drawn independently for each of
    42/43/44 (section 14.0a), and the seed enters HERE, in the ordering, and
    nowhere else.

    ⚠ It must NOT enter `content_hash`. query_id = H(class, text) has to stay
    seed-free: a text that lands in two seeds' validation sets is the SAME
    query, and section 13.6.4 clusters on unique query_id precisely to keep
    its cells together. Put the seed in the identity and every cluster becomes
    size one -- the clustering vanishes while every check still passes.
    """
    return _h(f"{VAL_TAG}\0{int(prefix_seed)}", class_idx, raw_text)
