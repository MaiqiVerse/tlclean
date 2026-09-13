"""The frozen query manifest, the PCW per-seed draws, and the test lock.

TWO SAMPLING REGIMES, BECAUSE THE TWO SETTINGS ASK DIFFERENT QUESTIONS.

  * L2, the exact MECHANISM setting (section 2.1) and the section 7
    reconciliation that reads from it, keep ONE common query set across the
    three demo seeds. Reconciliation compares a per-query mechanism prediction
    with that same query's behaviour, so the query has to be held fixed while
    the seed varies -- that is the comparison.

  * L3.1, the METHOD main table and the H6 baseline comparison (section 2.2),
    adopt PCW's evaluation setting -- FIXED TEST SIZE 250 plus MULTIPLE RANDOM
    SEEDS. For each demo seed, draw exactly 250 queries from the train-class
    test pool, independently per seed. Within a seed every arm and every
    baseline sees the SAME 250, so the per-query pairing the paired test needs
    is intact; across seeds the query sets differ and are not aligned.

    PCW is the SOURCE OF THE SETTING here, not a baseline, and reproducing the
    particular 250 queries its historical runs happened to draw is not required.
    So the draw is a SHA-derived seed plus NumPy sampling without replacement;
    what is actually frozen is only: exactly 250 per seed, independent across
    seeds, one shared subset for all methods within a seed, a reproducible rule
    and manifest, and results aggregated over the three seeds (section 6.2(B)).

    The consequence is statistical: there is no "same query across seeds" to
    average over, and the three draws OVERLAP because they come from one pool.
    So the confirmatory test clusters on unique query_id -- see
    tools/method_a_stats.cluster_weights -- and the 750 (query, seed) cells are
    never treated as independent observations.

THE DRAWS EXCLUDE VALIDATION, AND THAT IS NOT OPTIONAL. gamma* and every
baseline hyperparameter are chosen on the validation set (section 5). A per-seed
draw taken from the whole pool would sooner or later include validation
queries, and the tuning would then have happened on part of its own test set. So
the 250 are drawn from pool-minus-validation.

Validation is drawn PER PREFIX SEED (section 14.0a). It used to be fixed and
common so that ONE global gamma* could be frozen on it; gamma and the carrier
heads are now selected per seed, so that reason is gone, and sharing one set
was actively wrong: the three configurations came out of the same sample, so
their errors were correlated and the between-seed spread -- the quantity the
seeds exist to measure -- carried no validation-sampling noise.

The seed enters the ORDERING key only. query_id stays H(class, text), so a
text drawn under two seeds is one query with two cells, which is what
13.6.4's clustering on unique query_id requires.

THIS FILE IS MODEL-INDEPENDENT. It carries the CLASS space -- which classes are
eligible, which queries exist, which 250 each seed drew, what validation
reserved -- and nothing that depends on a tokenizer. Candidate token ids used to
live here, and that is exactly how a set of Llama-2 ids came to describe a
Llama-3.1 run: a class is the same class under any tokenizer, a label token is
not. The mapping now lives in a per-model label space built by
tools/build_label_space.py, and a confirmatory run is handed both.

    python tools/build_query_manifest.py \\
        --task trec_fine_per_class --K 5 --demo-seeds 42,43,44 \\
        --output results/prereg_method_A_query_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.prereg_ids import QUERY_TAG, content_hash  # noqa: E402
from tools.prototype_targets import derive_seed  # noqa: E402

# Section 2.1/2.2, from the one place they are written (tools/prereg_config).
# Re-exported so existing importers of this module are unaffected.
from tools.prereg_config import (N_TEST_PER_SEED,  # noqa: E402,F401
                                 VALIDATION_PER_CLASS)
SPLITS = ("validation", "test_common", "test_seed")


def build_entries(rows, eligible):
    """Every eligible-class TEST query, hash-ordered. All of them are the pool.

    rows: iterable of (class_idx, raw_text, native_index).

    Validation no longer comes out of this split -- it is drawn from TRAIN by
    prereg_task.draw_validation_from_train -- so the whole test split is
    available for the per-seed draws.
    """
    train = {int(c) for c in eligible}
    entries, seen, n_skipped = [], {}, 0
    for class_idx, raw_text, native in rows:
        c = int(class_idx)
        if c not in train:
            n_skipped += 1
            continue
        qid = content_hash(c, raw_text)
        if qid in seen:
            raise ValueError(
                f"duplicate query_id {qid[:16]} at native rows {seen[qid]} and "
                f"{native}: identical class and identical bytes. Deduplicate "
                "the data rather than letting one copy be validation and the "
                "other eligible for a test draw.")
        seen[qid] = native
        entries.append({"query_id": qid, "class_idx": c,
                        "native_index": int(native)})

    entries.sort(key=lambda e: e["query_id"])
    for e in entries:
        e["role"] = "pool"
    return entries, n_skipped


def draw_for_seed(pool_ids, demo_seed, n=N_TEST_PER_SEED):
    """The PCW draw: exactly n queries for one demo seed, without replacement.

    Derived from the same SHA256 key schedule as every other random object in
    this preregistration (section 4), so the draw is reproducible from the run's
    identity and cannot be re-rolled. The pool is sorted first: a draw taken
    from a set's iteration order would depend on insertion history.
    """
    pool = sorted(pool_ids)
    if len(pool) < n:
        raise ValueError(
            f"the draw pool holds {len(pool)} queries but the PCW protocol "
            f"needs exactly {n} per seed. Section 2.2 fixes the count; it must "
            "not be lowered to fit, and drawing with replacement would break "
            "the per-query pairing within the seed.")
    rng = np.random.default_rng(
        derive_seed("query_draw", "L3.1", int(demo_seed), "*", "*", "*"))
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(idx)]


def load_split(manifest_path, split, *, demo_seed=None, freeze_manifest=None,
               expected_roles=None):
    """Read one split's entries, enforcing the test lock (sections 2.1, 13.4(4)).

    split='validation'   the class-balanced draw from TRAIN, drawn PER PREFIX
                         SEED (section 14.0a), so `demo_seed` is required and
                         has no default; open before the freeze
    split='test_common'  every eligible-class TEST query; L2 mechanism setting
    split='test_seed'    one demo seed's 250 draws; L3.1 method setting

    Both test splits are refused unless a freeze manifest is supplied AND still
    hashes correctly on disk. Before the freeze there is nothing to pass, so the
    call cannot succeed; after a single edited artifact it stops succeeding
    again.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    by_id = {e["query_id"]: e for e in man["entries"]}

    if split != "validation":
        if freeze_manifest is None:
            raise PermissionError(
                f"the {split} split is locked. Section 2.1 forbids generating "
                "any test prediction before the three gamma*, the control "
                "carriers/positions and the analysis code are frozen, and "
                "section 13.4(4) makes the freeze manifest the evidence that "
                "they are. Pass --freeze-manifest, or use validation.")
        # NOT verify_freeze_manifest. That answers "has anything drifted",
        # which ANY self-signed JSON with one correct hash answers perfectly
        # while pinning nothing -- a file registering only the query manifest
        # itself opened this lock. Section 2.1 asks a different question, and
        # it is asked here.
        #
        # THIS manifest is bound, not merely "some manifest": a valid freeze
        # built for manifest A opened manifest B, which was not in `files` at
        # all. Callers that also load carriers or a label space pass those
        # through `expected_roles` so the same binding covers them.
        bad = freeze_manifest_blockers(
            freeze_manifest,
            expected_roles={**(expected_roles or {}),
                            "query_manifest": manifest_path})
        if bad:
            raise PermissionError(
                "the freeze manifest may not open the test lock: "
                + "; ".join(bad))

    if split == "validation":
        # drawn from the TRAIN split and stored separately, so it can never be
        # confused with a test query by role alone. Per seed since 14.0a, so
        # the caller must say WHICH -- there is no defensible default, and a
        # silent one would hand back seed 42's set to a seed 44 run.
        vbs = man["validation_by_seed"]
        if demo_seed is None:
            raise ValueError(
                "validation is drawn per prefix seed; pass demo_seed=42/43/44."
                f" available: {sorted(vbs)}")
        if str(demo_seed) not in vbs:
            raise ValueError(f"no validation for seed {demo_seed}; "
                             f"the manifest has {sorted(vbs)}")
        return list(vbs[str(demo_seed)])
    if split == "test_common":
        return [e for e in man["entries"] if e["role"] == "pool"]

    if demo_seed is None:
        raise ValueError("split='test_seed' needs demo_seed: under the PCW "
                         "protocol each seed has its OWN 250 queries, so there "
                         "is no seed-independent test set to return")
    key = str(int(demo_seed))
    if key not in man["test_by_seed"]:
        raise ValueError(f"no draw recorded for demo seed {key}; the manifest "
                         f"has {sorted(man['test_by_seed'])}")
    return [by_id[q] for q in man["test_by_seed"][key]]


FREEZE_KIND = "prereg_method_A_freeze_manifest"

# Section 2.1 names what must be frozen before any test prediction: the three
# gamma*, the control carriers/positions, and the analysis code. A freeze
# manifest that does not register these is not evidence of that freeze, so
# each is required BY ROLE -- a path is easy to omit by accident, a named role
# is not.
FREEZE_ROLES = {
    "query_manifest": "the draws being scored",
    "carriers":       "the control carriers/positions (section 2.1)",
    "label_space":    "the frozen readout the predictions are argmaxed over",
    "gammas":         "the three frozen gamma* (L2, group, head), section "
                      "13.4. A ROLE, not an optional extra: while it was only "
                      "mentioned in prose, every passing fixture opened the "
                      "lock without one, and the baseline spec freeze does "
                      "not record Method A's gammas either",
    "spec_freeze":    "the section 13.4(4) baseline spec freeze, which pins "
                      "the baseline runners AND (since METHOD_A_CODE) the "
                      "Method A analysis modules",
}


GAMMA_KEYS = ("gamma_L2", "gamma_group", "gamma_head")


def freeze_gamma_path(freeze_manifest):
    """(path, why) for the gamma artifact a freeze registers. Never raises.

    Callers on the test path need the FILE, not just the assurance that one
    exists, because the value they are about to use has to be checked against
    it. Returns (None, reason) rather than raising so it composes with the
    caller's problem list.
    """
    if not freeze_manifest:
        return None, "no freeze manifest, so no frozen gamma* to check against"
    try:
        doc = json.loads(Path(freeze_manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, (f"{freeze_manifest}: unreadable -- "
                      f"{type(e).__name__}: {e}")
    roles = doc.get("roles") if isinstance(doc.get("roles"), dict) else {}
    p = roles.get("gammas")
    if not isinstance(p, str) or not p:
        return None, (f"{freeze_manifest} registers no 'gammas' role, so the "
                      "three frozen gamma* are not pinned by anything")
    return p, None


def load_frozen_gammas(path):
    """(dict, faults) for the three frozen gamma*. Never raises.

    A HASH IS NOT A CONSTRAINT. The `gammas` role made the freeze register a
    file and nothing more: nobody parsed it, and the probe took its gammas
    from --gamma, so after the freeze one could still change gamma* or sweep
    the whole grid on test while the lock verified clean. Writing `not-json`
    into the artifact changed nothing. The file has to be READ, and the value
    a run uses has to be checked against it -- see the probe's validate().
    """
    p = Path(path)
    if not p.is_file():
        return {}, [f"{path}: no such file"]
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {}, [f"{path}: unreadable as the frozen gamma* -- "
                    f"{type(e).__name__}: {e}"]
    if not isinstance(doc, dict):
        return {}, [f"{path}: top level is {type(doc).__name__}, expected an "
                    f"object with {list(GAMMA_KEYS)}"]
    faults, out = [], {}
    for k in GAMMA_KEYS:
        v = doc.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            faults.append(f"{path}: {k} is {v!r}, expected a number")
        else:
            out[k] = float(v)
    return out, faults


def write_freeze_manifest(path, *, query_manifest, carriers, label_space,
                          gammas, spec_freeze, extra=()):
    """Write a freeze manifest that `freeze_manifest_blockers` can accept.

    THE ONLY PRODUCER. Until now nothing in the repo wrote this file -- every
    site that made one was a fixture hand-rolling the format, which is how the
    format drifted into "any JSON with a non-empty 'files' map" in the first
    place. One writer means the roles, the kind tag and the hashing are stated
    once, and a fixture cannot quietly disagree with production.

    `extra` names further files to hash-pin without giving them a role.
    Everything registered is checked on every later read, so adding a file
    here makes editing it re-lock the test splits. The three gamma* are NOT
    an extra -- they are a required role, because an optional slot is one
    nobody fills.
    """
    roles = {"query_manifest": query_manifest, "carriers": carriers,
             "label_space": label_space, "gammas": gammas,
             "spec_freeze": spec_freeze}
    missing = [r for r, p in roles.items() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"cannot write a freeze manifest: {missing} do not exist. A "
            "freeze that registers a file which is not there pins nothing.")
    doc = {
        "kind": FREEZE_KIND,
        "roles": {r: str(p) for r, p in roles.items()},
        "files": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                  for p in list(roles.values()) + list(extra)},
    }
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    return doc


def _same_file(a, b):
    """Path equality that survives ./, .., symlinks and a missing file."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


def freeze_manifest_blockers(path, *, expected_roles=None):
    """Why this file may NOT open the test lock. Empty means it may.

    `expected_roles` maps role -> the path THIS RUN is actually going to load,
    and every entry given must match the role the freeze registers. Without it
    the freeze is checked in a vacuum: a perfectly valid freeze built for
    manifest A opened manifest B, which was not in `files` at all, and the
    carriers and label space a runner actually read were never compared to the
    ones the freeze pinned. A freeze that does not name the artifacts in use
    is not evidence about the run being made.

    DELIBERATELY NOT `verify_freeze_manifest`. That one asks "has anything
    drifted since this was written" -- a question a hand-authored JSON
    registering a single unrelated file answers perfectly, while pinning
    nothing at all. It was the whole check behind section 2.1's lock, so
    before any real freeze existed a two-line self-signed file could unlock
    `test_common`. Drift and authority are different questions and only the
    second one governs test reads.

    A file may open the lock when it (1) declares itself, (2) registers every
    role section 2.1 names, with each role's path also hash-pinned in `files`,
    (3) has not drifted, (4) names the artifacts this run is loading, and
    (5) points at a spec freeze that is at the CODE stage, passes
    `freeze_baseline_spec.validate_freeze`, AND still matches the checkout per
    `compare`. Code stage is what "the analysis code is frozen" means, and
    `compare` is what makes it true of the code that is here NOW: validity is
    a property of the record, drift is a property of the working tree, and
    only checking the first left an ordinary post-freeze edit to
    `tools/baselines/*.py` invisible -- no forgery required.

    ⚠ WHAT THIS STILL DOES NOT DO. Nothing here proves the freeze was written
    before the results; that is what committing it is for. A file-based freeze
    is a discipline artifact, and claiming more for it than it does is how the
    first version of this check passed review while pinning nothing.
    """
    bad = verify_freeze_manifest(path)
    if bad:
        return bad
    try:
        man = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as e:                       # already read once; belt
        return [f"{path}: not readable as JSON -- {e}"]
    if man.get("kind") != FREEZE_KIND:
        return [f"{path}: kind is {man.get('kind')!r}, expected "
                f"{FREEZE_KIND!r}. An arbitrary JSON that happens to carry a "
                "'files' map is not a freeze"]
    files = man.get("files") or {}
    # RETURNS the fault. `roles` is attacker-or-typo controlled and a string
    # there raised AttributeError out of .get -- valid JSON, so the previous
    # round's corrupt-JSON guard did not cover it.
    role_map = man.get("roles")
    if role_map is None:
        role_map = {}
    elif not isinstance(role_map, dict):
        return [f"{path}: 'roles' is {type(role_map).__name__}, expected an "
                f"object mapping {sorted(FREEZE_ROLES)} to paths"]
    resolved = {}
    for role, why in sorted(FREEZE_ROLES.items()):
        rel = role_map.get(role)
        if rel is not None and not isinstance(rel, str):
            bad.append(f"role {role!r} is {type(rel).__name__}, expected a "
                       "path string")
            continue
        if not rel:
            bad.append(f"no {role!r} role -- {why}")
            continue
        if not any(_same_file(rel, k) for k in files):
            bad.append(f"role {role!r} names {rel}, which is not registered "
                       "in 'files', so nothing pins its contents")
            continue
        resolved[role] = rel
    if bad:
        return bad

    # THE ARTIFACTS THIS RUN IS ACTUALLY LOADING. A valid freeze says nothing
    # about a run that reads different files.
    for role, used in sorted((expected_roles or {}).items()):
        if role not in FREEZE_ROLES:
            bad.append(f"caller asked to bind unknown role {role!r}; known "
                       f"roles are {sorted(FREEZE_ROLES)}")
        elif not _same_file(resolved[role], used):
            bad.append(f"this run loads {used} as {role!r}, but the freeze "
                       f"pins {resolved[role]}. A freeze for one artifact is "
                       "not evidence about another")
    if bad:
        return bad

    # THE GAMMA ARTIFACT IS PARSED, not merely hashed. Registering a file
    # nobody reads is a hash with no contract behind it.
    _g, gfaults = load_frozen_gammas(resolved["gammas"])
    if gfaults:
        return gfaults

    spec_path = Path(resolved["spec_freeze"])
    try:
        rec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{spec_path}: unreadable as a spec freeze -- {e}"]
    from tools.baselines.freeze_baseline_spec import (build_freeze, compare,
                                                      validate_freeze)
    try:
        # NOT require_baselines_ready. Section 2.1's condition on reading test
        # is "the three gamma*, the control carriers/positions and the
        # analysis code are frozen" -- it says nothing about the six opponents
        # being implemented, and requiring that here made writing five
        # baseline runners a prerequisite for reading Method A's own test
        # split. Section 14.1's C12 already ruled on this friction once; the
        # staging exists so that a runner written later is not drift. H6 and
        # the viability gate, whose claims ARE about the opponents, keep the
        # full check.
        blockers = validate_freeze(rec, require_baselines_ready=False)
    except Exception as e:                        # noqa: BLE001
        return [f"{spec_path}: could not be validated -- "
                f"{type(e).__name__}: {e}"]
    if blockers:
        return [f"{spec_path} is not a valid spec freeze: {b}"
                for b in blockers]
    if rec.get("stage") != "code":
        return [f"{spec_path} is a {rec.get('stage')!r}-stage freeze. Section "
                "2.1 requires the analysis code frozen before any test "
                "prediction, and only the code stage records that"]
    # VALIDITY IS NOT UNDRIFTEDNESS. validate_freeze reads only the record, so
    # editing a file the record hashes -- any tools/baselines/*.py, any result
    # NPZ -- left it valid and left the lock open. compare re-derives the
    # hashes from this checkout.
    try:
        # include_baselines=False: this run does not read the opponent
        # configurations, so the registry having changed since the freeze does
        # not make it less reproducible. `compare` no longer looks at source
        # files either -- the commit does that, and it is REPORTED rather than
        # gated (see below). What is left is the documents and the artifacts,
        # both of which this run does read.
        # include_repo=False as well: 14.0b-2 (v) allows a run at a different
        # commit from the freeze, because the result carries its own commit
        # through run_provenance and refusing would rebuild the friction that
        # dropping per-file hashes removed. `--verify` still reports it.
        drift = compare(rec, build_freeze("code"), include_baselines=False,
                        include_repo=False)
    except Exception as e:                        # noqa: BLE001
        return [f"{spec_path}: the checkout could not be re-hashed for "
                f"comparison -- {type(e).__name__}: {e}"]
    if drift:
        return [f"{spec_path} no longer matches this checkout ({len(drift)} "
                f"item(s), first: {drift[0]}). Section 2.1's lock closes again "
                "the moment a frozen file is edited"]
    # NO SOURCE FILES ARE HASHED, here or anywhere. A per-file list missed a
    # module twice, and the import closure that replaced it pinned 19 files of
    # which 12 are shared with other lines of work, so the test splits
    # re-locked whenever icl_common or model_loader was edited for an
    # unrelated reason. The freeze records a COMMIT instead -- forty
    # characters covering all of that plus tasks/ and script/, which no list
    # named -- and every result carries its own commit through
    # icl_common.run_provenance. Whether the run's tree is committable is a
    # property of the RUN, not of this file: see attribution_blockers.
    return []


def attribution_blockers():
    """Why a run about to touch test could not be attributed to a commit.

    Separate from `freeze_manifest_blockers` because it is a property of the
    working tree at run time, not of the freeze file: folding it in there
    would have meant the fixtures only pass on a clean checkout, which is the
    environment-dependent gate problem in reverse.

    A DIRTY TREE is refused rather than reported -- it is the one state that
    makes a commit id a lie, and a test prediction produced from uncommitted
    code cannot be reproduced from anything. Running at a DIFFERENT commit
    from the freeze is NOT refused: the result carries its own commit, so it
    stays attributable, and refusing would re-create the friction that
    dropping per-file hashes exists to remove.
    """
    from tools.icl_common import run_provenance
    prov = run_provenance()
    if str(prov.get("repo_commit", "")).startswith("UNAVAILABLE"):
        return [f"the repository commit could not be read "
                f"({prov['repo_commit']}), so this run could not be "
                "attributed to any code state"]
    if prov.get("repo_dirty") == "true":
        return ["the working tree has uncommitted changes, so no commit id "
                "identifies the code about to generate test predictions. "
                "Commit or stash first -- a result that cannot be traced to a "
                "commit cannot be reproduced from one."]
    return []


def verify_freeze_manifest(path):
    """Has anything DRIFTED since this was written? Empty means nothing has.

    ⚠ This is not the test lock. It answers a narrow question and answers it
    for any well-formed file; `freeze_manifest_blockers` is what section 2.1
    requires, and it is what `load_split` calls.
    """
    p = Path(path)
    if not p.exists():
        return [f"{path}: no such file"]
    # RETURNS the fault, never raises. A truncated or hand-edited freeze is
    # exactly the state this is meant to report on, and a JSONDecodeError
    # escaping here reaches callers that are catching PermissionError.
    try:
        man = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{path}: unreadable -- {type(e).__name__}: {e}"]
    if not isinstance(man, dict):
        return [f"{path}: top level is {type(man).__name__}, expected an "
                "object with a 'files' map"]
    files = man.get("files")
    if not isinstance(files, dict) or not files:
        return [f"{path}: no 'files' section, so it locks nothing"]
    bad = []
    for rel, want in sorted(files.items()):
        f = Path(rel)
        if not f.exists():
            bad.append(f"{rel}: missing")
            continue
        # RETURNS the fault. A registered path that is a directory, or that
        # the process cannot read, raised out of read_bytes -- again valid
        # JSON, so the corrupt-JSON guard above did not cover it.
        if not f.is_file():
            bad.append(f"{rel}: not a regular file, so it hashes to nothing")
            continue
        try:
            got = hashlib.sha256(f.read_bytes()).hexdigest()
        except OSError as e:
            bad.append(f"{rel}: unreadable -- {type(e).__name__}: {e}")
            continue
        if got != want:
            bad.append(f"{rel}: {got[:12]} != {str(want)[:12]}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="trec_fine_per_class")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--demo-seeds", default="42,43,44")
    ap.add_argument("--n-queries", type=int, default=250)
    ap.add_argument("--validation-per-class", type=int,
                    default=VALIDATION_PER_CLASS,
                    help="validation queries per eligible class, drawn from "
                         "the TRAIN split (section 2.1)")
    ap.add_argument("--n-test-per-seed", type=int, default=N_TEST_PER_SEED)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from tools.prereg_task import ALL_ELIGIBLE
    if args.task in ALL_ELIGIBLE:
        raise SystemExit(
            f"{args.task}: this task carries its demonstrations INSIDE each "
            "query (a per-prompt hidden function), so there is no shared "
            "demonstration bank for the section-13.5 constructions to read "
            "and no train split to reserve validation from. The rediscovery "
            "chain (script/rediscover_cell.sh) covers it; the method line "
            "(script/method_cell.sh) does not.")

    from tools.prereg_task import (draw_validation_from_train,
                                   eligible_classes, eligibility_report,
                                   load_task, prefix_demo_rows, test_rows,
                                   train_classes, train_rows)

    seeds = [int(s) for s in args.demo_seeds.split(",")]
    task = load_task(args.task, args.K, seeds[0], args.n_queries)
    # The decision space is the classes present in BOTH splits, not merely the
    # train classes: a class with no test example can never be a gold answer,
    # so it is not a candidate and gets no demonstration either.
    train = eligible_classes(task, args.task)
    all_train = train_classes(task, args.task)
    rows = test_rows(task, args.task)
    if not rows:
        raise SystemExit(f"{args.task}: the native test split is empty")

    # THIS FILE IS MODEL-INDEPENDENT. Candidate token ids used to live here,
    # which is how a set of Llama-2 ids came to describe a Llama-3.1 run: a
    # class is the same class under any tokenizer, a label token is not. The
    # mapping now lives in a per-model label space
    # (tools/build_label_space.py), and this manifest carries only the class
    # space. Nothing here needs a calibration file, so the two-phase dance the
    # old cycle required is gone as well.

    entries, n_skipped = build_entries(rows, train)
    # `pool` is derived AFTER the validation collisions are removed, below --
    # deriving it here and filtering `entries` afterwards would leave the two
    # describing different sets, and the draw reads `pool`.

    # Validation FIRST, then demonstrations from what is left. The reverse
    # order is not safe: with 15 training examples and three independent K=5
    # draws the union covers ~10.6 of them, leaving ~4.4 against a quota of 4.
    # Reserving first makes the margin structural, and it removes a circularity
    # -- a filter defined on the draw's leftovers would change the draw.
    # PER SEED (section 14.0a). Each prefix seed draws its own 144, and its
    # own demos come out of what its own draw left. The exclusion is per seed
    # rather than over the union: seed 42's prefix must not contain seed 42's
    # validation, but nothing is wrong with it containing seed 43's, and
    # excluding the union would take three times as much out of the demo pool
    # for no gain.
    validation_by_seed, vdiag_by_seed, demo_hashes_by_seed = {}, {}, {}
    trows = train_rows(task, args.task)
    for sd in seeds:
        v_s, vd_s = draw_validation_from_train(
            trows, train, set(), sd, per_class=args.validation_per_class)
        reserved_s = {(v["class_idx"], v["source_text"]) for v in v_s}
        validation_by_seed[sd], vdiag_by_seed[sd] = v_s, vd_s
        task._excluded_docs = reserved_s
        task.set_fewshot(num_fewshot=args.K, seed=sd)
        demo_hashes_by_seed[sd] = {
            content_hash(c, t)
            for pairs in prefix_demo_rows(
                args.task, args.K, [sd], args.n_queries,
                excluded_docs=reserved_s).values()
            for c, t in pairs}
    # No seeds[0] shorthand. It was introduced here "for the census printout"
    # and then reached the disjointness assertions, where it checked seed 42's
    # validation against all three draws -- an overlap under 43 or 44 passed
    # unseen. A convenience alias for one seed's data has no safe scope in a
    # function that handles three.

    # Print the census BEFORE drawing. A shortfall is the interesting outcome,
    # not an inconvenience, and diagnostics that only appear on success tell you
    # nothing on the run where you need them.
    print("=" * 78)
    print("FROZEN QUERY MANIFEST (prereg 2.1 / 2.2)")
    print("=" * 78)
    elig_rep = eligibility_report(task, args.task)
    print(f"  train classes (have a demo)  : {len(all_train)}")
    print(f"    train-only (no test query) : {len(elig_rep['train_only'])} "
          f"{elig_rep['train_only']}")
    print(f"    below min_train="
          f"{elig_rep['min_train']:<2}         : "
          f"{len(elig_rep['below_min_train'])} {elig_rep['below_min_train']}")
    print(f"  ELIGIBLE classes             : {len(train)} "
          "(in BOTH splits, and >= 15 deduplicated train rows)")
    _drop = sorted(set(all_train) - set(train))
    if _drop:
        print(f"    excluded in total          : {len(_drop)} {_drop}")
    print(f"  native test docs             : {len(rows)}")
    print(f"  eligible-class test queries  : {len(entries)} "
          f"({n_skipped} dropped as non-eligible-class)")
    for sd in seeds:
        vd_s = vdiag_by_seed[sd]
        print(f"  validation seed {sd} (TRAIN, "
              f"{args.validation_per_class}/class) : "
              f"{len(validation_by_seed[sd])} over {vd_s['n_classes']} classes "
              f"({vd_s['n_excluded_by_hash']} train rows skipped as its own "
              "prefix demos)")
        if vd_s["classes_short_of_quota"]:
            print(f"    seed {sd} classes below quota : "
                  f"{vd_s['classes_short_of_quota']}")
    _distinct = {v["query_id"] for sd in seeds for v in validation_by_seed[sd]}
    _cells = sum(len(validation_by_seed[sd]) for sd in seeds)
    _multi = sum(1 for q in _distinct
                 if sum(any(v["query_id"] == q for v in validation_by_seed[sd])
                        for sd in seeds) > 1)
    # THE POOL MUST NOT CONTAIN A VALIDATION QUERY, and split membership does
    # not guarantee that: query_id is H(class, text), and the same (class,
    # text) exists in both TREC splits, so a train row and a test row can be
    # ONE query. The server run found exactly this -- seed 44 drew one of its
    # own validation queries -- and the assertion below is what caught it.
    #
    # UNION, not per seed. test_common is scored under all three prefixes, so
    # a query in ANY seed's validation would be tuned on by that seed and then
    # scored under its own prefix as part of the shared test set.
    _val_union = {v["query_id"] for sd in seeds for v in validation_by_seed[sd]}
    _collide = [e for e in entries if e["query_id"] in _val_union]
    if _collide:
        entries = [e for e in entries if e["query_id"] not in _val_union]
        print(f"  content-hash collisions      : {len(_collide)} test "
              "queries dropped because the SAME (class, text) is also a "
              "validation query for some seed. Section 2.1's pool is every "
              "eligible test query that is not also a validation query; "
              "split membership does not imply that, because query_id is "
              "H(class, text) and TREC ships duplicates across splits")
        for e in _collide[:5]:
            print(f"      class {e['class_idx']:>2}  {e['query_id'][:12]}")
    pool = [e["query_id"] for e in entries if e["role"] == "pool"]
    print(f"  validation, distinct queries : {len(_distinct)} of {_cells} "
          f"cells => {_cells - len(_distinct)} REPEATED CELLS, over {_multi} "
          "queries drawn by more than one seed (a query in all three "
          "contributes two repeats, so the two numbers differ; the sign-flip "
          "clusters on the query, not the cell)")
    print(f"  draw pool (whole test split) : {len(pool)}")
    print(f"  L2 test_common               : {len(pool)} (all remaining)")
    print(f"  L3.1 requested per seed      : {args.n_test_per_seed}")
    if len(pool) < args.n_test_per_seed:
        print()
        print(f"  ** the pool cannot supply {args.n_test_per_seed} per seed: "
              f"only {len(pool)} eligible test queries exist. **")
        print("  This is a specification conflict, not a bug: the per-seed size "
              "and the eligible test split are incompatible. Decide the design "
              "before lowering any number -- shrinking the per-seed draw, "
              "loosening the class filter and using a common test set are "
              "different experiments, and picking whichever makes the "
              "arithmetic work is a preregistration change (section 13.4(5)).")
    per_class_pool = {}
    for e in entries:
        if e["role"] == "pool":
            per_class_pool[e["class_idx"]] = per_class_pool.get(
                e["class_idx"], 0) + 1
    if per_class_pool:
        print(f"  pool covers {len(per_class_pool)} classes, "
              f"{min(per_class_pool.values())}-{max(per_class_pool.values())} "
              "queries each")

    test_by_seed = {str(s): draw_for_seed(pool, s, args.n_test_per_seed)
                    for s in seeds}
    for s in seeds:
        drawn = test_by_seed[str(s)]
        print(f"  L3.1 seed {s} draw            : {len(drawn)}")
    if len(seeds) > 1:
        pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
        for a, b in pairs:
            ov = len(set(test_by_seed[str(a)]) & set(test_by_seed[str(b)]))
            print(f"    overlap {a} n {b}            : {ov} queries "
                  f"({ov / args.n_test_per_seed:.0%}) -- expected under "
                  "independent draws from one pool, and not a defect: the "
                  "seeds are not required to be disjoint, only unaligned")

    # ---- disjointness, asserted not assumed --------------------------------
    # PER SEED. This block took a single val_ids from validation_by_seed
    # [seeds[0]], so an overlap existing only under seed 43 or 44 passed
    # unseen. Each seed's own validation is now checked against its own draw
    # and against the whole test pool.
    all_entry_ids = {e["query_id"] for e in entries}
    for sd in seeds:
        val_ids_s = {v["query_id"] for v in validation_by_seed[sd]}
        assert len(val_ids_s) == len(validation_by_seed[sd]), (
            f"seed {sd}: duplicate validation query")
        drawn = test_by_seed[str(sd)]
        assert not (set(drawn) & val_ids_s), (
            f"seed {sd}: drew {len(set(drawn) & val_ids_s)} of its OWN "
            "validation queries -- gamma_s would have been tuned on part of "
            "its own test set")
        assert len(set(drawn)) == len(drawn), f"seed {sd}: duplicate draw"
        assert not (val_ids_s & all_entry_ids), (
            f"seed {sd}: a validation query (from train) collides by content "
            "hash with a test query -- the splits are not disjoint at the "
            "content level")
    for _sd in seeds:
        _v = {v["query_id"] for v in validation_by_seed[_sd]}
        assert not (_v & demo_hashes_by_seed[_sd]), (
            f"seed {_sd}: a validation query is one of its own prefix demos")
    print("  validation n test / n demos  : PASS (empty)")
    print("  validation n every seed draw : PASS (empty)")

    out = {
        "spec": "prereg_method_A.md sections 2.1, 2.2 (PCW protocol)",
        "hash_convention":
            f'query_id = SHA256("{QUERY_TAG}\\0<class_idx>\\0<raw utf-8 text>")'
            "; raw bytes, no normalization (2.1(5))",
        "sampling": {
            "validation_by_seed": "drawn independently per demo seed (14.0a); "
                                  "ordered by validation_hash(prefix_seed, ...). "
                                  "query_id carries NO seed, so a text drawn "
                                  "under two seeds is one query with two cells",
            "test_common": "all remaining train-class queries; L2 mechanism "
                           "setting and the section 7 reconciliation",
            "test_by_seed": f"{args.n_test_per_seed} drawn per demo seed "
                            "without replacement from the pool, independently "
                            "per seed; L3.1 method main table and H6"},
        "candidate_space_moved": {
            "why_absent": "REMOVED from the manifest on 2026-09-01. Candidate token "
                    "ids are tokenizer-specific and now live in a per-model "
                    "label space built by tools/build_label_space.py; this "
                    "file carries the model-independent class space only. A "
                    "confirmatory run must be given both artifacts.",
            "n_candidates": len(train),
            "classes": train,
            "note": "the ELIGIBLE classes: those present in both the train "
                    "and test splits. The task is closed-set over the "
                    "classes with training demonstrations and gold support, so "
                    "the remaining classes are outside the decision space by "
                    "definition of the task -- not because anything makes them "
                    "unreachable. Token ids are SELECTED from the calibration "
                    "header's existing mapping by class id; nothing is "
                    "re-discovered or renumbered, so the frozen carriers "
                    "remain valid."},
        "validation_per_class": args.validation_per_class,
        # Per seed. Recording only seeds[0]'s counts would have described one
        # of three draws while reading as though it described the manifest.
        "n_validation_by_seed": {str(sd): len(validation_by_seed[sd])
                                 for sd in seeds},
        "n_validation_distinct_queries": len(
            {v["query_id"] for sd in seeds for v in validation_by_seed[sd]}),
        "validation_source": "train split, class-balanced, drawn per prefix "
                             "seed; that seed's own prefix demos excluded by "
                             "content hash",
        "validation_diagnostics_by_seed": {str(sd): vdiag_by_seed[sd]
                                           for sd in seeds},
        "eligibility": elig_rep,
        "n_test_per_seed": args.n_test_per_seed,
        "demo_seeds": seeds,
        "eligible_classes": train,
        "train_classes_all": all_train,
        "inputs": {"task": args.task, "K": args.K,
                   "note": "no calibration file: this artifact is "
                           "model-independent"},
        "diagnostics": {"n_total": len(entries),
                        "n_skipped_non_train_class": n_skipped,
                        "n_pool": len(pool)},
        "validation_by_seed": {str(sd): validation_by_seed[sd]
                               for sd in seeds},
        "entries": entries,
        "test_by_seed": test_by_seed,
    }
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  [output] {p}  (sha256 "
          f"{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}...)")
    print("  No text is stored. Both test splits stay unreadable through "
          "load_split() until a verified freeze manifest exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
