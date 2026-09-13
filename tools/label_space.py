"""The frozen, per-model label space. Builder + loader. ZERO GPU.

WHY THIS IS A SEPARATE ARTIFACT FROM THE QUERY MANIFEST

A class is model-independent: class 21 is class 21 whatever tokenizer is
loaded. A label TOKEN is not. `discover_single_token_labels` picks the label
strings by what tokenizes to a single token under the tokenizer it is handed,
and `label_token_ids` are those strings' ids there. Llama-2 and Llama-3.1
therefore have different surfaces and different ids for the same class.

Keeping both in one manifest is how the two got confused: the manifest's
candidate tokens were Llama-2 ids while the method main table runs Llama-3.1,
and nothing in the file recorded which tokenizer had produced them. So the
split is now structural --

    query_manifest       model-INDEPENDENT: eligible classes, query ids, the
                         per-seed draws, the validation reservation
    label_space_<tag>    model-SPECIFIC: surfaces, token ids, the candidate
                         tokens for the eligible classes, and the provenance of
                         the tokenizer that produced them

-- and a confirmatory run must be handed BOTH, with the model named, so the
mismatch is refused instead of computed.

PROVENANCE IS MORE THAN A MODEL STRING. Two checkouts of "the same" model can
tokenize differently: a different Transformers version, a different revision, a
patched special-token map. The provenance block therefore records the tokenizer
class, the Transformers version, the revision if the hub exposed one, and
hashes of the canonicalised vocabulary and the special-token map. A mismatch on
any of those means the ids in this file were produced by something other than
the tokenizer now loaded.

EVERYTHING DOWNSTREAM SPEAKS CLASSES. Token ids never cross models: predictions,
gold, the fixed opponent and every statistic are class ids, and the label space
is the only place the mapping lives.

    python tools/build_label_space.py \\
        --uuid-jsonl data/calibration_trec_fine_per_class_K5_seed42_uuid.jsonl \\
        --query-manifest results/prereg_method_A_query_manifest.json \\
        --model meta-llama/Llama-3.1-8B \\
        --output results/label_space_llama31.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCHEMA_VERSION = 1

# Fields that must agree exactly between a label space and the tokenizer now
# loaded. `revision` is advisory -- the hub does not always expose one -- but
# the two hashes are not: they are computed from the tokenizer itself.
PROVENANCE_KEYS = ("model", "tokenizer_class", "transformers_version",
                   "vocab_size", "vocab_sha256", "special_tokens_sha256",
                   "backend_tokenizer_sha256")
PROVENANCE_STRICT = ("model", "tokenizer_class", "vocab_size", "vocab_sha256",
                     "special_tokens_sha256", "backend_tokenizer_sha256")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokenizer_provenance(tok, model_name: str) -> dict:
    """Everything that could make two 'same model' tokenizers disagree.

    The vocabulary is canonicalised by id then surface before hashing, so the
    hash does not depend on dict ordering. The special-token map is hashed too:
    a patched BOS or an added pad token changes how a prompt tokenises without
    changing the vocabulary at all.

    THE VOCABULARY IS NOT ENOUGH. Two tokenizers can share every entry and
    still split text differently -- different merge rules, a different
    normalizer, a different pre-tokenizer. `backend_tokenizer.to_str()` is the
    fast tokenizer's own complete serialisation: merges, normalizer,
    pre-tokenizer, post-processor and decoder. Hashing it is the closest thing
    to an identity for the tokenising BEHAVIOUR rather than for its dictionary.
    A slow tokenizer has no backend; the field is then None and says so, which
    is itself a difference worth catching.
    """
    import transformers

    vocab = tok.get_vocab()
    blob = "\n".join(f"{i}\t{t}" for t, i in
                     sorted(vocab.items(), key=lambda kv: (kv[1], kv[0])))
    special = getattr(tok, "special_tokens_map_extended", None) \
        or getattr(tok, "special_tokens_map", {})
    revision = (getattr(tok, "_commit_hash", None)
                or (getattr(tok, "init_kwargs", {}) or {}).get("revision"))
    backend = getattr(tok, "backend_tokenizer", None)
    backend_hash = None
    if backend is not None:
        try:
            backend_hash = _sha256_text(backend.to_str())
        except Exception:                       # pragma: no cover
            backend_hash = None
    return {
        "model": model_name,
        "tokenizer_class": type(tok).__name__,
        "transformers_version": transformers.__version__,
        "revision": revision,
        "is_fast": bool(getattr(tok, "is_fast", False)),
        "vocab_size": len(vocab),
        "vocab_sha256": _sha256_text(blob),
        "special_tokens_sha256": _sha256_text(
            json.dumps(special, sort_keys=True, default=str)),
        # merges + normalizer + pre-tokenizer + post-processor + decoder
        "backend_tokenizer_sha256": backend_hash,
    }


def first_token_in_context(tok, prompt: str, surface: str):
    """The token a label surface contributes when appended to a real prompt.

    NOT `tok(" " + surface)`. Tokenizers are context-sensitive at the boundary:
    the id a surface produces after "...\\nType:" is not necessarily the id it
    produces on its own, which is exactly why
    `discover_single_token_labels` works by tokenize-and-diff against a real
    prompt. Verifying with the standalone form would check a different
    quantity from the one the experiment reads.

    Returns None when appending the surface changes nothing.
    """
    p_ids = tok(prompt, add_special_tokens=True).input_ids
    f_ids = tok(prompt + " " + surface, add_special_tokens=True).input_ids
    for i in range(min(len(p_ids), len(f_ids))):
        if p_ids[i] != f_ids[i]:
            return int(f_ids[i])
    return int(f_ids[len(p_ids)]) if len(f_ids) > len(p_ids) else None


def provenance_mismatch(recorded: dict, current: dict) -> list:
    """Fields on which a label space and the loaded tokenizer disagree."""
    bad = []
    for k in PROVENANCE_STRICT:
        if recorded.get(k) != current.get(k):
            bad.append(f"{k}: recorded {recorded.get(k)!r} vs current "
                       f"{current.get(k)!r}")
    for k in ("transformers_version", "revision"):
        if recorded.get(k) != current.get(k):
            bad.append(f"[advisory] {k}: recorded {recorded.get(k)!r} vs "
                       f"current {current.get(k)!r}")
    return bad


def output_dir_conflict(existing_headers, model):
    """Which already-present calibration files are not this run's.

    `existing_headers` is {filename: header-or-None}. Three kinds of conflict,
    and the rule is the same for all of them: a file that cannot be shown to
    belong to THIS model must not be overwritten silently.

      * a different model -- obvious;
      * a header with no 'model' field -- predates provenance, so nothing can
        attribute its token ids to a tokenizer;
      * an unreadable or header-less file -- likewise unattributable. Skipping
        it would be fail-open in a design whose whole premise is that
        unverifiable means refused; the file may well be a calibration file
        whose header simply failed to parse.

    Moving such a file aside is a deliberate act, not something a run does for
    you.
    """
    bad = []
    for name, hdr in sorted(existing_headers.items()):
        if not hdr:
            bad.append(f"{name}: no readable header, so it cannot be shown to "
                       "belong to this model")
            continue
        got = hdr.get("model")
        if got is None:
            bad.append(f"{name}: no 'model' field (predates provenance; its "
                       "token ids cannot be attributed to any tokenizer)")
        elif got != model:
            bad.append(f"{name}: belongs to {got!r}")
    return bad


def assert_header_matches(label_space, header, *, path="", model=None,
                          query_manifest_sha256=None):
    """The calibration header a run reads must be the one the space was built
    from. Raises on any disagreement.

    Passing the wrong --uuid-jsonl is easy and quiet: the file parses, the
    prompts render, and the only thing wrong is that its surfaces map to
    different ids than the frozen space says. Every field the space copied out
    of a header is therefore re-compared against the header actually loaded.
    """
    ls = (label_space.data if isinstance(label_space, FrozenLabelSpace)
          else label_space)
    bad = []
    if model is not None and header.get("model") != model:
        bad.append(f"header model {header.get('model')!r} != run model "
                   f"{model!r}")
    if header.get("model") != ls["provenance"]["model"]:
        bad.append(f"header model {header.get('model')!r} != label-space model "
                   f"{ls['provenance']['model']!r}")
    if list(header.get("abstract_labels") or []) != list(ls["abstract_labels"]):
        n = sum(1 for a, b in zip(header.get("abstract_labels") or [],
                                  ls["abstract_labels"]) if a != b)
        bad.append(f"surfaces differ from the label space ({n} positions "
                   "compared pairwise, lengths "
                   f"{len(header.get('abstract_labels') or [])} vs "
                   f"{len(ls['abstract_labels'])})")
    if [int(t) for t in (header.get("label_token_ids") or [])] != \
            [int(t) for t in ls["label_token_ids"]]:
        bad.append("label token ids differ from the label space")
    # A MISSING field is a failure, not a skipped check. Treating absence as
    # "nothing to compare" is fail-open, and every one of these fields is
    # absent precisely in the files this refactor exists to keep out: those
    # written before the provenance and manifest-hash fields existed.
    if header.get("eligible_class_indices") is None:
        bad.append("header has no 'eligible_class_indices'; it cannot be shown "
                   "to describe this class space")
    elif [int(c) for c in header["eligible_class_indices"]] != \
            [int(c) for c in ls["eligible_classes"]]:
        bad.append("eligible classes differ from the label space")

    hp = header.get("tokenizer_provenance")
    if not hp:
        bad.append("header has no 'tokenizer_provenance'; its token ids cannot "
                   "be attributed to any tokenizer")
    else:
        hard = [b for b in provenance_mismatch(hp, ls["provenance"])
                if not b.startswith("[advisory]")]
        bad += [f"tokenizer provenance: {b}" for b in hard]

    if query_manifest_sha256 is not None:
        for src, val in (("header", header.get("query_manifest_sha256")),
                         ("label space", ls.get("query_manifest_sha256"))):
            if val is None:
                bad.append(f"{src} records no query-manifest hash, so it "
                           "cannot be tied to the class space this run uses")
            elif val != query_manifest_sha256:
                bad.append(f"{src} was built against query manifest "
                           f"{str(val)[:12]}, this run uses "
                           f"{query_manifest_sha256[:12]}")
    if bad:
        raise ValueError(
            f"the calibration header{' ' + path if path else ''} is not the "
            "one this label space was built from: " + "; ".join(bad)
            + ". Check --uuid-jsonl.")


class FrozenLabelSpace:
    """A loaded label space, with its consistency already checked."""

    def __init__(self, data: dict, path=None):
        self.data = data
        self.path = str(path) if path else None
        for k in ("schema_version", "provenance", "eligible_classes",
                  "abstract_labels", "label_token_ids", "candidate_token_ids"):
            if k not in data:
                raise ValueError(
                    f"label space missing {k!r}; this is not a schema "
                    f"v{SCHEMA_VERSION} artifact (keys: {sorted(data)})")
        if int(data["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(
                f"label space schema v{data['schema_version']}, this code "
                f"expects v{SCHEMA_VERSION}. Rebuild it rather than reading an "
                "older shape whose fields may mean something else.")
        self.model = data["provenance"]["model"]
        self.eligible_classes = [int(c) for c in data["eligible_classes"]]
        self.candidate_token_ids = [int(t) for t in data["candidate_token_ids"]]
        self.abstract_labels = list(data["abstract_labels"])
        self.label_token_ids = [int(t) for t in data["label_token_ids"]]

        # The file must be internally consistent before anything reads it. A
        # label space that merely LOOKS well formed is the failure mode this
        # whole artifact exists to remove, so every relation it asserts is
        # re-derived here rather than trusted.
        n = data.get("n_classes")
        if n is not None and not (int(n) == len(self.abstract_labels)
                                 == len(self.label_token_ids)):
            raise ValueError(
                f"n_classes={n} but {len(self.abstract_labels)} surfaces and "
                f"{len(self.label_token_ids)} token ids")
        if len(self.abstract_labels) != len(self.label_token_ids):
            raise ValueError(
                f"{len(self.abstract_labels)} surfaces vs "
                f"{len(self.label_token_ids)} token ids")
        if len(set(self.eligible_classes)) != len(self.eligible_classes):
            raise ValueError(f"eligible classes repeat: "
                             f"{sorted(self.eligible_classes)}")
        bad = [c for c in self.eligible_classes
               if not 0 <= c < len(self.label_token_ids)]
        if bad:
            raise ValueError(
                f"eligible classes {bad} fall outside the "
                f"{len(self.label_token_ids)}-entry label mapping")
        if len(self.eligible_classes) != len(self.candidate_token_ids):
            raise ValueError(
                f"{len(self.eligible_classes)} eligible classes but "
                f"{len(self.candidate_token_ids)} candidate tokens")
        want = [self.label_token_ids[c] for c in self.eligible_classes]
        if self.candidate_token_ids != want:
            where = [(c, w, g) for c, w, g in
                     zip(self.eligible_classes, want,
                         self.candidate_token_ids) if w != g]
            raise ValueError(
                f"candidate_token_ids are not the eligible classes' entries of "
                f"label_token_ids; {len(where)} disagree, first three "
                f"{where[:3]}")
        if len(set(self.candidate_token_ids)) != len(self.candidate_token_ids):
            raise ValueError("candidate token ids are not distinct; two "
                             "classes would be indistinguishable at readout")

    @classmethod
    def load(cls, path, *, model=None, query_manifest=None, tokenizer=None):
        """Load and verify. Every optional argument given is CHECKED.

        `model` guards the commonest error -- reading one model's ids while
        running another. `query_manifest` guards the second: a label space
        built against a different class space. `tokenizer` is the strongest
        check, comparing the recorded provenance against the live object.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ls = cls(data, path)
        if model is not None and ls.model != model:
            raise ValueError(
                f"{path} was built for {ls.model!r} but this run uses "
                f"{model!r}. Label token ids do not transfer between "
                "tokenizers; they would index valid but wrong columns.")
        if query_manifest is not None:
            want = file_sha256(query_manifest)
            got = data.get("query_manifest_sha256")
            if got != want:
                raise ValueError(
                    f"{path} was built against query manifest {str(got)[:12]} "
                    f"but {query_manifest} hashes to {want[:12]}. The class "
                    "space may have changed under it.")
        if tokenizer is not None:
            bad = provenance_mismatch(data["provenance"],
                                      tokenizer_provenance(tokenizer,
                                                           ls.model))
            hard = [b for b in bad if not b.startswith("[advisory]")]
            if hard:
                raise ValueError(
                    f"{path}: the loaded tokenizer is not the one that "
                    "produced these ids: " + "; ".join(hard))
            for b in bad:
                print(f"  [warn] {b}")
        return ls

    def candidate_space(self):
        """The metric-side object, in class terms."""
        from tools.method_a_stats import CandidateSpace
        return CandidateSpace(self.eligible_classes, self.candidate_token_ids)

    def summary(self):
        p = self.data["provenance"]
        return (f"{len(self.eligible_classes)} candidates for {p['model']} "
                f"({p['tokenizer_class']}, vocab {p['vocab_size']}, "
                f"hash {p['vocab_sha256'][:12]})")
