"""Append arms to a receiver npz instead of re-reading every arm.

A receiver (run_k0_receiver, run_k10_increment) reads all its arms into one
npz: `<arm>_seed<s>` arrays of candidate logits, `query_ids`, `class_idx`
and a json `meta` whose `arms` lists what is there. When a cell gains the
four baselines (FV / TV / ICV / I2CL), a finished level's npz lacks their
arms and method_status --needs-arm sends the level back to the receiver,
which used to read all ~28 arms again -- the eleven old ones bit for bit,
for nothing. `--append` reads only the arms the file lacks and writes the
union.

What makes that legitimate, and what refuses it:

  * the file must be THIS run's file: same model, method, dtype, kernel,
    task, K, K_base, head cut, split, limit, seeds, and the same carrier
    bundle / query manifest / label space (sha256), and the arrays' query
    ids and classes must be the rows this run would read. Anything else
    is refused with both values printed (working rules 3.6): a stale file is
    deleted by hand, never patched over.
  * the kernel is part of the identity: an eager file is not appended to
    under sdpa, or the file would mix two numerics (RESULTS 63.17d).
  * the old arms are kept as they were (np.stack of the stored rows is the
    stored array); the new alpha = 0 arms are still checked elementwise
    against the STORED base arm, so a machine or library that no longer
    reproduces the old forward is caught rather than averaged in.
  * the meta keeps the old provenance under `provenance_original`, records
    the appended arms with this run's provenance and cache gate under
    `appended`, and takes the family descriptions (fv / tv / icv / i2cl /
    tsla) from this run, which built them from the same sidecars.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

IDENTITY_KEYS = ("model", "method", "dtype", "attn", "task", "K", "K_base", "top_n",
                 "split", "limit", "seeds", "carrier_bundle_sha256",
                 "query_manifest_sha256", "label_space_sha256")
FAMILY_KEYS = ("fv", "tv", "icv", "i2cl", "tsla")


def _norm(k, v):
    if k == "attn":
        return v or "eager"           # a file from before the kernel was recorded was eager
    if k == "seeds":
        return [int(x) for x in (v or [])]
    if k in ("K", "K_base", "limit"):
        return None if v is None else int(v)
    return v


def load_for_append(path, expect, ran_by_seed):
    """(old_meta, arrays) of an existing receiver npz that this run may append
    to; SystemExit naming every mismatch otherwise. `expect` holds the run's
    values for IDENTITY_KEYS (missing keys are not compared); `ran_by_seed`
    maps seed -> the rows this run reads (dicts with query_id / class_idx)."""
    p = Path(path)
    z = np.load(p, allow_pickle=False)
    if "meta" not in z.files:
        raise SystemExit(f"--append: {p} has no meta (arrays: {sorted(z.files)[:8]}); "
                         "not a receiver npz")
    meta = json.loads(str(z["meta"]))
    bad = []
    for k in IDENTITY_KEYS:
        if k not in expect:
            continue
        if k == "dtype" and "dtype" not in meta:
            # the receivers began recording dtype on 2026-09-13; a file
            # without it was written by this cell's driver in the cell's
            # dtype, which is what this run carries -- said, not assumed
            print(f"  --append: {p} does not record its dtype (written before the "
                  f"receivers did); this run is {expect[k]}, the cell's setting")
            continue
        old, now = _norm(k, meta.get(k)), _norm(k, expect[k])
        if old != now:
            bad.append(f"{k}: file {old!r}, this run {now!r}")
    seeds = [int(s) for s in ran_by_seed]
    q_old = np.asarray(z["query_ids"]).astype(str)
    c_old = np.asarray(z["class_idx"]).astype(int)
    lens = [len(ran_by_seed[s]) for s in seeds]
    if q_old.ndim != 2 or q_old.shape[0] != len(seeds) or any(n != q_old.shape[1] for n in lens):
        bad.append(f"query_ids: file {tuple(q_old.shape)}, this run {len(seeds)} seeds x {lens} rows")
    else:
        q_now = np.array([[str(r["query_id"]) for r in ran_by_seed[s]] for s in seeds])
        c_now = np.array([[int(r["class_idx"]) for r in ran_by_seed[s]] for s in seeds])
        if not np.array_equal(q_old, q_now):
            bad.append(f"query_ids: {int((q_old != q_now).sum())} of {q_old.size} entries differ")
        elif not np.array_equal(c_old, c_now):
            bad.append(f"class_idx: {int((c_old != c_now).sum())} entries differ")
    arms = list(meta.get("arms") or [])
    lack = [(a, s) for a in arms for s in seeds if f"{a}_seed{s}" not in z.files]
    if lack:
        bad.append(f"meta.arms names arrays the file lacks: {lack[:4]}")
    if bad:
        raise SystemExit(f"--append refused for {p}: it is not this run's file. "
                         + "; ".join(bad) + ". Delete it to read the level afresh.")
    arrays = {k: np.asarray(z[k]) for k in z.files if k != "meta"}
    return meta, arrays


def plan_append(old_meta, run_arms):
    """(todo, all_arms): the arms this run must read, and the union it writes
    (run order first, then any stored arm the run did not ask for)."""
    old = list(old_meta.get("arms") or [])
    todo = [a for a in run_arms if a not in old]
    all_arms = list(run_arms) + [a for a in old if a not in run_arms]
    return todo, all_arms


def prefill_out(out, arrays, old_arms, seeds):
    """Put the stored rows into `out` so the stored arms are written back
    unchanged and the gates that read them (cache equivalence, alpha = 0)
    see the stored values."""
    for a in old_arms:
        for s in seeds:
            out.setdefault(a, {})[s] = [row for row in arrays[f"{a}_seed{s}"]]
    return out


def merged_meta(old_meta, new_meta, appended):
    """The union's meta: the stored one, this run's family descriptions and
    per-seed additions, the arms of the union, and the append record."""
    m = dict(old_meta)
    m["arms"] = list(new_meta["arms"])
    for k in FAMILY_KEYS:
        if new_meta.get(k) is not None:
            m[k] = new_meta[k]
    ps_old, ps_new = old_meta.get("per_seed") or {}, new_meta.get("per_seed") or {}
    m["per_seed"] = {s: {**(ps_old.get(s) or {}), **(ps_new.get(s) or {})}
                     for s in sorted(set(ps_old) | set(ps_new))}
    m["provenance_original"] = old_meta.get("provenance_original", old_meta.get("provenance"))
    m["provenance"] = new_meta.get("provenance")
    rec = {"arms": list(appended), "provenance": new_meta.get("provenance"),
           "cache_gate_now": new_meta.get("cache_gate"), "attn": new_meta.get("attn")}
    m["appended"] = (old_meta.get("appended") or []) + [rec]
    return m


def save_receiver_npz(path, out, arms, ran_by_seed, seeds, meta):
    """The receivers' one writer: <arm>_seed<s> stacks, the rows' ids and
    classes, the meta json; written beside the target and moved into place,
    so a crash mid-write leaves the old file whole."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".partial")
    np.savez_compressed(
        tmp,
        **{f"{a}_seed{s}": np.stack(out[a][s]) for a in arms for s in out[a]},
        query_ids=np.array([[str(r["query_id"]) for r in ran_by_seed[s]] for s in seeds]),
        class_idx=np.array([[int(r["class_idx"]) for r in ran_by_seed[s]] for s in seeds]),
        meta=json.dumps(meta))
    written = tmp if tmp.exists() else tmp.with_name(tmp.name + ".npz")
    os.replace(written, p)
    return p
