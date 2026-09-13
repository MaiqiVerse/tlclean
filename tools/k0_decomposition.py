"""Section 13.5.2's unique decomposition of a natural K=5 prompt.

    P_s(D_s, q) = C_s || D_s || R_s(q)

  C_s  the query-independent common prefix every method keeps (BOS, any
       instruction/header, and the fixed separator before the demonstrations)
  D_s  ALL K demonstration blocks with their internal and trailing separators
  R_s(q)  what the query contributes

EVERYTHING IN 13.5 STANDS ON THIS. The K=0 receiver's whole claim is that it
reads from an offline memory built out of D_s while its own prompt contains no
demonstrations; if the boundary between D_s and R_s(q) is off by one token,
the memory contains part of the query or the prompt contains part of a
demonstration, and every number afterwards describes something else.

THE ONE WAY TO GET IT WRONG, WHICH I HAVE DONE TWICE TODAY. Tokenising the
pieces and concatenating their ids is NOT tokenising the prompt. Tokenisers are
context sensitive at boundaries -- build_label_space says so in as many words,
and it is why first_token_in_context exists. So the boundaries here are found
by tokenising the FULL prompt once and locating cut points inside it, and each
cut is verified: the ids up to the cut must equal the ids of the text up to the
cut, exactly. A cut that fails that test is reported, never rounded to the
nearest token.

ZERO GPU: this is a tokeniser and string arithmetic.
"""

from __future__ import annotations


def prefix_cuts(tok, full_text, cut_texts):
    """Token indices where `cut_texts` end inside `full_text`'s tokenisation.

    `cut_texts` are the successive PREFIXES of full_text at which a boundary
    falls, shortest first. Returns one index per cut.

    Each is verified rather than assumed: tokenising the prefix must produce
    exactly the first n ids of the whole prompt's tokenisation. When it does
    not, the boundary does not exist at the token level and this raises --
    13.5.2 asks for a decomposition, and a cut that only nearly works is not
    one.
    """
    full = list(tok(full_text, add_special_tokens=True).input_ids)
    out, last = [], 0
    for ct in cut_texts:
        if not full_text.startswith(ct):
            raise ValueError(
                f"the cut text (len {len(ct)}) is not a prefix of the prompt; "
                "the decomposition is defined on prefixes of the SAME string, "
                "not on separately rendered pieces")
        ids = list(tok(ct, add_special_tokens=True).input_ids)
        n = len(ids)
        if full[:n] != ids:
            i = next((j for j in range(min(n, len(full)))
                      if full[j] != ids[j]), min(n, len(full)))
            raise ValueError(
                f"tokenising the prefix does not reproduce the prompt's first "
                f"{n} ids (first difference at {i}). The boundary does not "
                "exist at the token level -- tokenisers are context sensitive "
                "there, and rounding to the nearest token would move text "
                "between D_s and R_s(q).")
        if n <= last:
            raise ValueError(
                f"cut at {n} does not advance past the previous cut at {last}; "
                "the three parts must be non-empty and in order")
        out.append(n)
        last = n
    if last >= len(full):
        raise ValueError(
            f"the last cut at {last} leaves nothing for R_s(q) "
            f"({len(full)} ids total); the query must contribute at least one "
            "token or there is no receiver")
    return out


def decompose(tok, full_text, common_text, through_demos_text):
    """(C_ids, D_ids, R_ids) for one prompt, each verified against the whole.

    `common_text` is full_text up to and including the separator that precedes
    the first demonstration; `through_demos_text` is full_text up to and
    including the separator that follows the last one.
    """
    n_c, n_d = prefix_cuts(tok, full_text, [common_text, through_demos_text])
    full = list(tok(full_text, add_special_tokens=True).input_ids)
    return full[:n_c], full[n_c:n_d], full[n_d:]


def decomposition_faults(parts_by_query):
    """Why a seed's decompositions are not what 13.5.2 requires.

    `parts_by_query` maps query id -> (C_ids, D_ids, R_ids).

    Two invariants, and they are the ones that make the K=0 arm meaningful:
    C_s and D_s are QUERY-INDEPENDENT -- they are the seed's fixed prefix and
    its demonstrations -- and R_s(q) is not, or the query contributes nothing.
    A run where D_s drifted between queries would have built its memory from
    something other than the prefix it claims.
    """
    bad = []
    if len(parts_by_query) < 2:
        return ["fewer than two queries: query-independence cannot be checked, "
                "and asserting it from one example is how it would be missed"]
    items = sorted(parts_by_query.items())
    (_, (c0, d0, _)) = items[0]
    for q, (c, d, _) in items[1:]:
        if c != c0:
            bad.append(f"query {str(q)[:12]}: C_s differs from the first "
                       f"query's ({len(c)} ids vs {len(c0)}). C_s is the "
                       "query-INDEPENDENT common prefix by definition")
        if d != d0:
            bad.append(f"query {str(q)[:12]}: D_s differs ({len(d)} ids vs "
                       f"{len(d0)}). The demonstrations do not depend on the "
                       "query, so a difference means the boundary moved and "
                       "the offline memory would not be built from the "
                       "prefix it claims")
    rs = {tuple(r) for _, (_, _, r) in items}
    if len(rs) == 1:
        bad.append("every query produced the SAME R_s(q); either the queries "
                   "are identical or the boundary swallowed them, and in "
                   "either case the receiver reads nothing query-specific")
    if any(not r for _, (_, _, r) in items):
        bad.append("some query contributed no tokens at all")
    return bad


def reassembly_faults(tok, full_text, parts):
    """The three parts must BE the prompt, not merely resemble it."""
    c, d, r = parts
    full = list(tok(full_text, add_special_tokens=True).input_ids)
    joined = list(c) + list(d) + list(r)
    if joined != full:
        i = next((j for j in range(min(len(joined), len(full)))
                  if joined[j] != full[j]), min(len(joined), len(full)))
        return [f"C||D||R has {len(joined)} ids and the prompt has "
                f"{len(full)}, first difference at {i}. The decomposition must "
                "be exact -- 13.5.2 says UNIQUE, and a reassembly that differs "
                "anywhere is describing a different prompt"]
    return []
