"""Build one model's frozen label space. ZERO GPU (tokenizer only).

The schema, the provenance computation and the loader all live in
`tools/label_space.py` and are IMPORTED here. They used to be duplicated in
both files, which is the arrangement where the next change gets made on one
side only.

    python tools/build_label_space.py \\
        --uuid-jsonl data/method_a/llama31/calibration_..._seed42_uuid.jsonl \\
        --query-manifest results/prereg_method_A_query_manifest.json \\
        --model meta-llama/Llama-3.1-8B \\
        --output results/label_space_llama31.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.label_space import (SCHEMA_VERSION, FrozenLabelSpace,  # noqa: E402
                               _sha256_text, file_sha256,
                               first_token_in_context, tokenizer_provenance)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid-jsonl", required=True,
                    help="calibration file whose header holds the frozen "
                         "surfaces and ids for THIS model")
    ap.add_argument("--query-manifest", required=True,
                    help="model-independent manifest; supplies the class space")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-verify-prompts", type=int, default=3,
                    help="how many real prompts to re-derive every id in")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from tools.icl_common import load_jsonl

    hdr, rows = load_jsonl(Path(args.uuid_jsonl))
    if hdr is None:
        raise SystemExit(f"{args.uuid_jsonl}: no header line")
    if not rows:
        raise SystemExit(f"{args.uuid_jsonl}: no prompt rows")
    if hdr.get("model") != args.model:
        raise SystemExit(
            f"{args.uuid_jsonl} records model {hdr.get('model')!r} but --model "
            f"is {args.model!r}. A calibration file built with one tokenizer "
            "cannot define another's label space; regenerate it.")

    man = json.loads(Path(args.query_manifest).read_text(encoding="utf-8"))
    if man.get("candidate_space"):
        raise SystemExit(
            f"{args.query_manifest} still carries 'candidate_space'; that field "
            "held tokenizer-specific ids in a model-independent file and was "
            "removed on 2026-09-01. Rebuild the manifest.")
    eligible = man.get("eligible_classes")
    if not eligible:
        raise SystemExit(f"{args.query_manifest}: no 'eligible_classes'")
    eligible = [int(c) for c in eligible]
    if hdr.get("eligible_class_indices") is not None \
            and [int(c) for c in hdr["eligible_class_indices"]] != eligible:
        raise SystemExit(
            "the calibration header and the query manifest disagree about the "
            f"eligible classes: {hdr['eligible_class_indices']} vs {eligible}")

    labels = list(hdr["abstract_labels"])
    ids = [int(x) for x in hdr["label_token_ids"]]
    if len(labels) != len(ids):
        raise SystemExit(f"{len(labels)} surfaces vs {len(ids)} token ids")

    tok = AutoTokenizer.from_pretrained(args.model)
    prov = tokenizer_provenance(tok, args.model)

    # Re-derive every id IN REAL PROMPT CONTEXT under the tokenizer now loaded.
    # Not `tok(" " + surface)`: tokenizers are context-sensitive at the
    # boundary, and the quantity the experiment reads is the token a surface
    # contributes after a prompt ending in "Type:". Several prompts are used
    # because one could agree by luck.
    prompts = [r["prompt"] for r in rows[:max(1, args.n_verify_prompts)]]
    bad = []
    for c, (surface, recorded) in enumerate(zip(labels, ids)):
        for pi, prompt in enumerate(prompts):
            got = first_token_in_context(tok, prompt, surface)
            if got != recorded:
                bad.append((c, surface, recorded, got, pi))
                break
    if bad:
        raise SystemExit(
            f"{len(bad)} label surfaces do not produce their recorded id under "
            f"{args.model} in real prompt context. First three "
            f"(class, surface, recorded, got, prompt#): {bad[:3]}. The "
            "calibration file was built with a different tokenizer.")

    cand = [ids[c] for c in eligible]
    if len(set(cand)) != len(cand):
        dup = sorted({t for t in cand if cand.count(t) > 1})
        raise SystemExit(
            f"candidate token ids are not distinct: {dup}. Two eligible "
            "classes share a first token and could never be told apart.")

    out = {
        "schema_version": SCHEMA_VERSION,
        "spec": "prereg_method_A.md sections 2.1, 6.1 -- model-specific label "
                "space; the class space lives in the query manifest",
        "provenance": prov,
        "task": hdr.get("task"), "K": hdr.get("K"),
        "n_classes": len(labels),
        "eligible_classes": eligible,
        "abstract_labels": labels,
        "label_token_ids": ids,
        "candidate_token_ids": cand,
        "verified_in_context": {"n_prompts": len(prompts),
                                "source": args.uuid_jsonl},
        "source_calibration": args.uuid_jsonl,
        "source_calibration_sha256": file_sha256(args.uuid_jsonl),
        "source_calibration_header_sha256": _sha256_text(
            json.dumps(hdr, sort_keys=True, ensure_ascii=False)),
        "query_manifest": args.query_manifest,
        "query_manifest_sha256": file_sha256(args.query_manifest),
    }

    # Load it back through the reader every consumer uses, so a file that would
    # fail at read time never reaches disk looking valid.
    FrozenLabelSpace(out)

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print(f"FROZEN LABEL SPACE -- {args.model}")
    print("=" * 78)
    print(f"  tokenizer      : {prov['tokenizer_class']} "
          f"(fast={prov['is_fast']}), transformers "
          f"{prov['transformers_version']}, revision {prov['revision']}")
    print(f"  vocab          : {prov['vocab_size']} entries, sha256 "
          f"{prov['vocab_sha256'][:16]}...")
    print(f"  backend        : sha256 "
          f"{str(prov['backend_tokenizer_sha256'])[:16]}... "
          "(merges + normalizer + pre-tokenizer + decoder)")
    print(f"  special tokens : sha256 {prov['special_tokens_sha256'][:16]}...")
    print(f"  classes        : {len(labels)} total, {len(eligible)} eligible")
    print(f"  candidates     : {len(cand)} distinct token ids")
    print(f"  every surface re-derived in {len(prompts)} real prompt(s) and "
          "matched")
    print(f"\n  [output] {p}  (sha256 {file_sha256(p)[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
