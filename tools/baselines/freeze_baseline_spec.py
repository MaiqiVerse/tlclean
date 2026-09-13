"""Freeze the six H6 baselines BEFORE the viability gate runs. ZERO GPU.

Section 13.6.1. Once a viability result is visible there is room to adjust the
opponent -- widen a grid, move an injection layer, read a head output on the
other side of W_O -- and every one of those moves has a good story available.
The defence is not discipline, it is a hash written down first.

WHAT IS HASHED, AND WHY EACH ONE.

  * `tools/baselines/registry.py` -- the candidate configs and the HOOK
    BOUNDARY. The boundary is the most adjustable thing about an adapted
    baseline, which is exactly why it is frozen text rather than an
    implementation detail.
  * `baseline_under_review.md` -- the prose spec the adapters must follow.
  * The section 13.2 block of `prereg_method_A.md` -- the frozen algorithms.
    Hashed as a SLICE, not as the whole file, because the preregistration is
    still open for the schema revision and will legitimately change elsewhere;
    hashing the whole file would make this freeze fire on every unrelated edit
    and be switched off within a week.
  * Every `tools/baselines/*.py` -- but ONLY at stage `code`. See below.
  * Each upstream repository's git HEAD, when it is a checkout.

TWO STAGES, BECAUSE ONE STAGE CONTRADICTS ITSELF. Hashing every runner file
before the viability gate is incoherent: five of the six runners do not exist
yet, and the whole point of the gate is to decide whether to write them.
Freeze everything now and the first new runner registers as drift; wait until
all six exist and the early stop has already been paid for.

    stage `spec`  (before carrier discovery and the viability gate)
                  registry.py -- algorithms, configs, hook boundaries;
                  prereg 13.2 and 13.6; baseline_under_review.md;
                  upstream commits. These are the things a visible result
                  could tempt someone to adjust, and none of them depend
                  on a runner existing.

    stage `code`  (at the final READY freeze, after all six runners exist)
                  every tools/baselines/*.py, so a later "small refactor"
                  is visible.

`--verify` checks whichever stages are present, so it is meaningful from the
moment the spec stage is written.

WHAT THIS DOES NOT DO. It does not prevent a change; it makes one visible and
dateable. Section 13.6.1 requires any drift after the gate to be recorded in
section 14 as having happened AFTER the viability result.

    python tools/baselines/freeze_baseline_spec.py --write
    python tools/baselines/freeze_baseline_spec.py --verify
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import numbers
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.baselines import registry
from tools.method_a_viability import strict_int  # noqa: E402
from tools.baselines.common import REGISTERED_DEMO_SEEDS  # noqa: E402

OUTPUT = REPO / "results" / "baseline_spec_freeze.json"
PREREG = REPO / "prereg_method_A.md"
SPEC_DOC = REPO / "baseline_under_review.md"
SECTION_START = "### 13.2"
SECTION_END = "### 13.3"
UPSTREAM = ("ZeroTuning", "StaICC", "function_vectors", "Localizing_TR_TL",
            "UniBias", "DeepThinking", "in-context-mechanism")

# Where the upstream checkouts live. The sibling directory is only a
# DEFAULT -- it happened to be true on the machine this was written on,
# and hard-coding it made the check report "directory not present" on a
# server where the clones sit elsewhere. The resolved root is recorded in
# the freeze and compared, so re-pointing it later is drift rather than a
# silent substitution.
UPSTREAM_ROOT = REPO.parent


def upstream_root():
    return Path(UPSTREAM_ROOT)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section_slice(path: Path, start: str, end: str) -> str:
    """The text between two headings, newline-normalised.

    Newlines are normalised because this repository is edited on Windows and
    run on Linux; a hash that flipped with the checkout's line endings would
    fire on every machine change and teach everyone to ignore it.
    """
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    i = text.find(start)
    if i < 0:
        raise ValueError(f"{path.name}: heading {start!r} not found, so the "
                         "frozen-algorithm block cannot be located")
    j = text.find(end, i + len(start))
    if j < 0:
        raise ValueError(f"{path.name}: heading {end!r} not found after "
                         f"{start!r}; the slice would run to end of file")
    if j <= i:
        raise ValueError(f"{path.name}: {end!r} precedes {start!r}")
    span = text[i:j]
    # working rules 6b(v): a slice guarded only from above passes trivially when it
    # is empty. Give it a floor too.
    if len(span) < 500:
        raise ValueError(
            f"{path.name}: the {start}..{end} slice is only {len(span)} chars, "
            "which is too short to be the frozen-algorithm block. Refusing to "
            "freeze a hash of almost nothing.")
    return span


def git_head(path: Path):
    """The checkout's HEAD, its cleanliness, and the hash of any local diff.

    ⚠ An unreadable repository is a FAILURE, not a recorded absence. The first
    version returned `available: False` with a reason and let `--write` succeed,
    and a review found all seven repositories unreadable (git ownership) while
    the freeze still printed PASS -- i.e. on the machine where it matters the
    check silently pinned nothing. `available` and `clean` are now compared like
    any other field, and `--write` refuses unless every repository is readable.

    A DIRTY checkout is not refused outright, because section 0.5 expects
    adaptation diffs to exist; but the diff is hashed so that editing upstream
    code under an unchanged HEAD shows up as drift. Without that, "same HEAD"
    was compatible with arbitrarily different source.
    """
    if not path.exists():
        return {"available": False, "why": "directory not present"}
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return {"available": False, "why": f"git failed: {e}"}
    if out.returncode != 0:
        return {"available": False,
                "why": f"not a git checkout ({out.stderr.strip()[:120]})"}
    dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=30)
    clean = not dirty.stdout.strip()
    rec = {"available": True, "head": out.stdout.strip(), "clean": clean}
    if not clean:
        diff = subprocess.run(["git", "-C", str(path), "diff", "HEAD"],
                              capture_output=True, text=True, timeout=60)
        rec["dirty_files"] = sorted(
            l[3:] for l in dirty.stdout.strip().split("\n") if l[3:])
        rec["diff_sha256"] = sha256_text(diff.stdout)
    return rec


def unreadable_repos(repos):
    """Which upstream repositories could not be pinned. Empty means all were."""
    return [f"{k}: {v.get('why', 'unreadable')}"
            for k, v in sorted(repos.items()) if not v.get("available")]


def pinned_repos():
    """{repo_dir: (baseline name, commit)} over BOTH the H6 six and the
    audited-but-excluded seventh.

    `in-context-mechanism` is in UPSTREAM and section 13.1 pins it, so leaving
    it out of the commit check would mean claiming to freeze seven repositories
    while freezing six.
    """
    out = {}
    for name, spec in list(registry.BASELINES.items()) \
            + list(registry.EXCLUDED.items()):
        if spec.get("registered_commit") and spec.get("repo_dir"):
            out[spec["repo_dir"]] = (name, spec["registered_commit"])
    return out


def wrong_commit(repos):
    """Repositories whose HEAD is not the commit the preregistration names."""
    bad = []
    for key, (name, want) in sorted(pinned_repos().items()):
        got = (repos.get(key) or {}).get("head")
        if got != want:
            bad.append(f"{key}: HEAD {str(got)[:12]} but section 13.1 fixes "
                       f"{want[:12]} for {name}")
    return bad


def dirty_repos(repos):
    """Repositories with local modifications.

    A dirty checkout was previously allowed with its diff hashed and a [note]
    printed. That is not enough: a hashed diff makes a LATER change visible,
    but it still lets the freeze be taken over source that is not the upstream
    source, and the note scrolls past. Section 0.5 puts adaptation diffs in
    results/baseline_adapt_diffs/<name>.patch precisely so the checkouts stay
    pristine, so dirty is a blocker. The diff hash is still recorded for
    `compare`, which is a different job -- catching an edit made after a clean
    freeze.
    """
    return [f"{k}: {len(v.get('dirty_files') or [])} locally modified file(s), "
            f"diff {str(v.get('diff_sha256'))[:12]}"
            for k, v in sorted(repos.items())
            if v.get("available") and not v.get("clean")]


# Section 6.1's status vocabulary for a finished adapter. NOT-IMPLEMENTED and
# the BLOCKED-* states are, by construction, not among them.
READY_STATUSES = ("READY", "ADAPTED", "ADAPTED-REIMPL", "CACHE-ADAPTED-REIMPL",
                  "NAMING-FIXED", "IMPLEMENTED")


def missing_registered_commits():
    """Repositories in UPSTREAM that nothing pins.

    Recording whatever HEAD happened to be checked out is not pinning a
    version, it is transcribing the machine. Checked over UPSTREAM rather than
    over BASELINES, so a repository that is listed as frozen but belongs to no
    H6 arm cannot slip through unpinned.
    """
    pinned = pinned_repos()
    return [f"{d}: in UPSTREAM but no registered_commit anywhere in the "
            "registry, so the freeze would record whatever HEAD it happens to "
            "be at"
            for d in UPSTREAM if d not in pinned]


def missing_adapt_diffs(only=None):
    """Baselines that declare they modify upstream but have no patch on disk.

    "At least one *.patch exists" was too weak: five of the six modify
    upstream, and one patch does not vouch for the other four. Which baselines
    need one is declared in the registry with a reason, so BC -- a formula
    reimplemented from the paper's code, touching no upstream file -- is not
    forced to invent a diff.
    """
    d = Path(REPO, "results", "baseline_adapt_diffs")
    names = set(only) if only is not None else set(registry.BASELINES)
    return [f"{name}: needs_adapt_diff is True ({spec['upstream'].split(' ')[0]})"
            f" but results/baseline_adapt_diffs/{name}.patch is missing"
            for name, spec in sorted(registry.BASELINES.items())
            if name in names and spec.get("needs_adapt_diff")
            and not (d / f"{name}.patch").is_file()]


SELECTION_SCHEMA_VERSION = 1
SELECTION_SPEC = "prereg_method_A.md section 5(4)"
SELECTION_METRIC = "three-seed-mean validation candidate NLL"
SELECTION_TIE_RULE = ("ties within 1e-8 take the earlier configuration in the "
                      "frozen list")
SELECTION_TIE_TOL = 1e-8
SELECTION_REQUIRED = ("schema_version", "spec", "demo_seeds", "metric",
                      "tie_rule", "query_manifest_sha256", "label_space_sha256",
                      "selected", "evidence")
# Per-arm evidence. `result_npz_path` is what turns the scores from an
# assertion into something recomputable; without it `result_npz_sha256` names
# no file and the recorded NLLs answer to nothing.
EVIDENCE_REQUIRED = ("result_npz_path", "result_npz_sha256", "per_config_nll")
NLL_RECOMPUTE_TOL = 1e-9

# The registered validation experiment, so an NPZ can be shown to BE it rather
# than merely to exist. Recomputing from a file proves the arithmetic; these
# prove the file is the right file.
RESULT_DIR = "results/baselines"
# Section 2.1, from tools/prereg_config -- N_VALIDATION is derived there.
from tools.prereg_config import N_ELIGIBLE, N_VALIDATION  # noqa: E402,F401
REGISTERED_TASK = "trec_fine_per_class"
REGISTERED_K = 5
REGISTERED_MODEL = "meta-llama/Llama-3.1-8B"
REGISTERED_DTYPE = ("torch.bfloat16", "bfloat16", "bf16")
# Baselines that publish a companion artifact alongside the result, and the
# file each one publishes. For these, `bias_sha256` is REQUIRED, not optional.
BIAS_BEARING = {"inductive_bc": "baseline_inductive_bc_bias.json"}
SIDECAR_REQUIRED = ("baseline", "mode", "model", "task", "K", "dtype",
                    "attn_implementation", "demo_seeds", "npz_sha256",
                    "query_manifest_sha256", "label_space_sha256")


def expected_result_npz(name):
    """The one path a baseline's validation result may live at.

    An arbitrary path was accepted before: absolute, `../`, or two baselines
    pointing at the same file. Pinning the name means the selection cannot
    quietly source its evidence from somewhere else, and it makes the
    code-stage hash of that file unambiguous.
    """
    return f"{RESULT_DIR}/baseline_{name}_validation.npz"


def npz_identity_faults(name, npz_path, data, sidecar, *, manifest, label_space,
                        configs):
    """Is this the REGISTERED validation run for `name`? Returns faults.

    Everything here is something that would otherwise decide a configuration
    from the wrong numbers while the arithmetic checked out: a test NPZ, a
    subset of the queries, another model's candidate space. The fixture that
    accompanied the first version was 3 seeds x 12 queries x 4 candidates and
    passed -- against a registered setting of 3 x 144 x 36.
    """
    import numpy as np

    faults = []
    # READ and PARSE inside the contract too. A missing file raised
    # FileNotFoundError and malformed JSON raised JSONDecodeError, both of
    # them past every caller that expects a fault list -- and this path is
    # reachable from the code-stage NLL recomputation.
    try:
        man = json.loads(Path(manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"the query manifest could not be read: "
                f"{type(e).__name__}: {e}"]
    # THE REAL LOADER, not a bare json.load. FrozenLabelSpace re-derives every
    # relation the file asserts and binds it to the manifest; reading the JSON
    # directly accepted a two-field stub that the runners themselves would
    # refuse -- so the selection validator was more permissive than the code
    # that produced the data.
    from tools.label_space import FrozenLabelSpace
    try:
        lsx = FrozenLabelSpace.load(label_space, model=str(sidecar.get("model")),
                                    query_manifest=manifest)
        ls = {"eligible_classes": lsx.eligible_classes,
              "candidate_token_ids": lsx.candidate_token_ids}
    except Exception as e:                                   # noqa: BLE001
        return [f"the label space did not load and verify: {e}"]
    # PER SEED. One id list described one of three draws and vouched for
    # all of them; a selection made on seed 43's queries passed by matching
    # seed 42's.
    if not isinstance(man, dict):
        return [f"the query manifest is {type(man).__name__}, expected a JSON "
                "object; .get would raise before any fault could be reported"]
    vbs = man.get("validation_by_seed")
    # INNER SHAPE, layer by layer. Validating only the root left three legal
    # JSON documents raising instead of reporting: a numeric
    # validation_by_seed (TypeError on `str(sd) in vbs`), a numeric value
    # under a seed (TypeError on iteration), and a row that is a mapping
    # without query_id (KeyError). Each is checked before the value it
    # guards is used.
    if vbs is not None and not isinstance(vbs, dict):
        return faults + [f"validation_by_seed is {type(vbs).__name__}, "
                         "expected an object keyed by seed"]
    if isinstance(vbs, dict):
        for _sd in REGISTERED_DEMO_SEEDS:
            _rows = vbs.get(str(_sd))
            if _rows is None:
                continue
            if not isinstance(_rows, list):
                faults.append(f"validation_by_seed[{_sd}] is "
                              f"{type(_rows).__name__}, expected a list")
                continue
            for _i, _r in enumerate(_rows):
                if not isinstance(_r, dict):
                    faults.append(f"validation_by_seed[{_sd}][{_i}] is "
                                  f"{type(_r).__name__}, expected an object")
                    break
                _miss = [k for k in ("query_id", "class_idx") if k not in _r]
                if _miss:
                    faults.append(f"validation_by_seed[{_sd}][{_i}] has no "
                                  f"{_miss}")
                    break
        if faults:
            return faults
    if vbs is None:
        return ["the query manifest has no 'validation_by_seed'; a single "
                "'validation' list predates section 14.0a and describes a "
                "draw that no longer exists"]
    want_ids = {str(sd): [str(v["query_id"]) for v in vbs[str(sd)]]
                for sd in REGISTERED_DEMO_SEEDS if str(sd) in vbs}
    try:
        want_cls = {str(sd): [strict_int(v["class_idx"],
                                         f"manifest seed {sd} class_idx")
                              for v in vbs[str(sd)]]
                    for sd in REGISTERED_DEMO_SEEDS if str(sd) in vbs}
    except (ValueError, TypeError) as e:
        return faults + [str(e)]
    missing_seeds = [sd for sd in REGISTERED_DEMO_SEEDS
                     if str(sd) not in vbs]
    if missing_seeds:
        faults.append(f"the manifest has no validation for seed(s) "
                      f"{missing_seeds}")

    # strict_int, not int(): bare int() truncates, so seed 42.9 became
    # seed 42 and matched the registered value. The same contract A5
    # established for the viability loader, imported rather than copied so
    # the two cannot drift.
    try:
        seeds = [strict_int(x, f"seeds[{i}]")
                 for i, x in enumerate(data.get("seeds", []))]
    except (ValueError, TypeError) as e:
        return faults + [f"seeds: {e}"]
    if seeds != list(REGISTERED_DEMO_SEEDS):
        faults.append(f"seeds {seeds} are not the registered "
                      f"{list(REGISTERED_DEMO_SEEDS)}")
    # query_ids and gold_class are [seed, query] now: each seed scored its
    # configurations on its OWN 144, so a single vector could only describe
    # one of them while reading as though it described the file.
    qarr = np.asarray(data.get("query_ids", []), dtype=object)
    garr = np.asarray(data.get("gold_class", []), dtype=object)
    if qarr.ndim != 2 or garr.shape != qarr.shape:
        faults.append(
            f"query_ids {qarr.shape} and gold_class {garr.shape} must both "
            "be [seed, query]; a 1-D vector is the pre-14.0a shared axis")
    elif qarr.shape[0] != len(REGISTERED_DEMO_SEEDS):
        faults.append(f"query_ids has {qarr.shape[0]} seed rows, expected "
                      f"{len(REGISTERED_DEMO_SEEDS)}")
    else:
        for si, sd in enumerate(REGISTERED_DEMO_SEEDS):
            w_ids, w_cls = want_ids.get(str(sd)), want_cls.get(str(sd))
            if w_ids is None:
                continue
            got = [str(q) for q in qarr[si]]
            if got != w_ids:
                faults.append(
                    f"seed {sd}: query_ids are not the manifest's "
                    f"{len(w_ids)} validation queries in order "
                    f"({len(set(got) & set(w_ids))} in common). A different "
                    "subset would score the configurations on different "
                    "data.")
            if len(got) != N_VALIDATION:
                faults.append(f"seed {sd}: {len(got)} queries; section 2.1 "
                              f"fixes validation at {N_VALIDATION} PER SEED")
            try:
                _g = [strict_int(g, f"gold_class[{si},{j}]")
                      for j, g in enumerate(garr[si])]
            except (ValueError, TypeError) as e:
                faults.append(f"seed {sd}: {e}")
                continue
            if _g != w_cls:
                faults.append(f"seed {sd}: gold_class disagrees with the "
                              "manifest's validation classes")
    try:
        cc = [strict_int(c, f"candidate_classes[{i}]")
              for i, c in enumerate(data.get("candidate_classes", []))]
    except (ValueError, TypeError) as e:
        return faults + [str(e)]
    if cc != [int(c) for c in ls["eligible_classes"]]:
        faults.append("candidate_classes are not the label space's eligible "
                      "classes")
    try:
        ct = [strict_int(t, f"candidate_token_ids[{i}]")
              for i, t in enumerate(data.get("candidate_token_ids", []))]
    except (ValueError, TypeError) as e:
        return faults + [str(e)]
    if ct != [int(t) for t in ls["candidate_token_ids"]]:
        faults.append("candidate_token_ids are not the label space's; these "
                      "logits index a different readout")

    try:
        logits = np.asarray(data.get("candidate_logits"), dtype=np.float64)
    except (ValueError, TypeError) as e:
        return faults + [f"candidate_logits does not convert to float64: {e}"]
    # 1-D, checked here too. Only the identity-free copy was fixed last
    # round; this is the path the real selection takes, and a 0-D array
    # raised on iteration.
    _aa = np.asarray(data.get("arms", []))
    if _aa.ndim != 1:
        return faults + [f"arms is {_aa.ndim}-D, expected a 1-D list of arm "
                         "names"]
    arms = [str(a) for a in _aa]
    want_shape = (len(arms), len(REGISTERED_DEMO_SEEDS), N_VALIDATION,
                  N_ELIGIBLE)
    if logits.shape != want_shape:
        faults.append(f"candidate_logits {logits.shape}, expected "
                      f"{want_shape} = (arm, seed, query, candidate)")
    elif not np.all(np.isfinite(logits)):
        faults.append(f"{int((~np.isfinite(logits)).sum())} non-finite logits")
    if len(set(arms)) != len(arms):
        faults.append(f"arms repeat: {arms}")
    need_arms = {config_arm_name(c) for c in configs}
    if not need_arms <= set(arms):
        faults.append(f"arms {sorted(need_arms - set(arms))} are missing")

    miss = [k for k in SIDECAR_REQUIRED if sidecar.get(k) is None]
    if miss:
        faults.append(f"the sidecar records no {miss}")
        return faults
    if sidecar["baseline"] != name:
        faults.append(f"the sidecar says baseline {sidecar['baseline']!r}, "
                      f"this is {name!r}")
    if sidecar["mode"] != "validation":
        faults.append(f"the sidecar says mode {sidecar['mode']!r}; a "
                      "configuration may only be selected on validation")
    if str(sidecar["model"]) != REGISTERED_MODEL:
        faults.append(f"model {sidecar['model']!r} != {REGISTERED_MODEL!r}")
    if str(sidecar["task"]) != REGISTERED_TASK:
        faults.append(f"task {sidecar['task']!r} != {REGISTERED_TASK!r}")
    try:
        if strict_int(sidecar["K"], "sidecar K") != REGISTERED_K:
            faults.append(f"K={sidecar['K']}, registered is {REGISTERED_K}")
    except (ValueError, TypeError) as e:
        faults.append(str(e))
    if str(sidecar["dtype"]) not in REGISTERED_DTYPE:
        faults.append(f"dtype {sidecar['dtype']!r} is not bfloat16")
    if str(sidecar["attn_implementation"]) != "eager":
        faults.append(f"attn_implementation "
                      f"{sidecar['attn_implementation']!r} is not 'eager'")
    try:
        _sd_ok = [strict_int(x, f"sidecar demo_seeds[{i}]")
                  for i, x in enumerate(sidecar["demo_seeds"])] == list(
                      REGISTERED_DEMO_SEEDS)
    except (ValueError, TypeError) as e:
        faults.append(str(e))
        _sd_ok = True                     # already reported
    if not _sd_ok:
        faults.append(f"the sidecar's demo_seeds {sidecar['demo_seeds']} are "
                      "not the registered three")
    # BC's result and its bias must come from ONE run, bound by hash.
    #
    # ⚠ This used to be `if sidecar.get("bias_sha256")`, i.e. fail-OPEN:
    # deleting the field from the sidecar skipped the entire binding check.
    # For a baseline that HAS a companion artifact the field is required, and
    # its absence is the failure.
    if name in BIAS_BEARING:
        bp = Path(npz_path).parent / BIAS_BEARING[name]
        if not sidecar.get("bias_sha256"):
            faults.append(
                f"the sidecar records no bias_sha256, so {bp.name} cannot be "
                f"shown to come from the same run as this result. {name} "
                "writes both together and binds them; a missing field is not "
                "an absent requirement.")
        elif not bp.is_file():
            faults.append(f"the sidecar records a bias_sha256 but "
                          f"{bp.name} is not beside the result")
        elif sha256_file(bp) != sidecar["bias_sha256"]:
            faults.append(
                f"{bp.name} hashes to {sha256_file(bp)[:12]} but the result's "
                f"sidecar records {str(sidecar['bias_sha256'])[:12]}: the bias "
                "and the result come from different runs")
    on_disk = sha256_file(Path(npz_path))
    if sidecar["npz_sha256"] != on_disk:
        faults.append(f"the sidecar records npz_sha256 "
                      f"{str(sidecar['npz_sha256'])[:12]} but the file hashes "
                      f"to {on_disk[:12]}")
    for key, path_ in (("query_manifest_sha256", manifest),
                       ("label_space_sha256", label_space)):
        want = sha256_file(Path(path_))
        if sidecar[key] != want:
            faults.append(f"the sidecar's {key} {str(sidecar[key])[:12]} != "
                          f"{Path(path_).name} on disk {want[:12]}")
    return faults


# One definition, in common.py, imported by both the runners and this
# validator. It used to live only here, and BC named its arm "bc" while this
# looked for "default".
from tools.baselines.common import config_arm_name  # noqa: E402,F401


def recompute_per_config_nll(npz_path, configs, *, name=None, manifest=None,
                             label_space=None):
    """Three-seed-mean validation candidate NLL per configuration, FROM the run.

    Returns (scores, faults). The candidate-conditional NLL is
    logsumexp over the candidate slice minus the gold candidate's logit
    (section 6.1), averaged over every (seed, query) cell -- which is what
    "three-seed-mean" means on a validation set shared by the three seeds.
    """
    import numpy as np

    p = Path(npz_path)
    if not p.is_file():
        return None, [f"{npz_path}: no such file, so nothing can be recomputed"]
    try:
        with np.load(p, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
    except Exception as e:                                   # noqa: BLE001
        return None, [f"{npz_path}: unreadable ({e})"]
    need = {"candidate_logits", "gold_class", "candidate_classes", "arms",
            "seeds", "query_ids", "candidate_token_ids"}
    missing = need - set(data)
    if missing:
        return None, [f"{npz_path}: missing {sorted(missing)}"]

    # IDENTITY before arithmetic. Recomputing from a file establishes that the
    # scores came from that file; it does not establish that the file is the
    # registered validation run.
    if name is not None and manifest is not None and label_space is not None:
        side = Path(str(npz_path).replace(".npz", ".json"))
        if not side.is_file():
            return None, [f"{npz_path}: no sidecar {side.name}, so the run "
                          "cannot be attributed to a model, mode or setting"]
        try:
            sidecar = json.loads(side.read_text(encoding="utf-8"))
        except Exception as e:                               # noqa: BLE001
            return None, [f"{side.name}: unreadable ({e})"]
        faults = npz_identity_faults(name, npz_path, data, sidecar,
                                     manifest=manifest, label_space=label_space,
                                     configs=configs)
        if faults:
            return None, [f"{Path(npz_path).name}: {f}" for f in faults]

    # THE ARM AXIS. A 0-D arms array raises on iteration, and a logits
    # first axis shorter than arms raised IndexError at arms.index() -- both
    # past the fault contract. Duplicate arms are refused rather than
    # silently resolved to the first: arms.index() picks one and the score
    # then describes a slab the caller did not name.
    _arms_arr = np.asarray(data["arms"])
    if _arms_arr.ndim != 1:
        return None, [f"{npz_path}: arms is {_arms_arr.ndim}-D, expected a "
                      "1-D list of arm names"]
    arms = [str(a) for a in _arms_arr]
    _dup_a = sorted({a for a in arms if arms.count(a) > 1})
    if _dup_a:
        return None, [f"{npz_path}: arms repeat {_dup_a}; index() would take "
                      "the first and the score would describe a slab the "
                      "caller did not name"]
    try:
        cand = [strict_int(c, f"candidate_classes[{i}]")
                for i, c in enumerate(data["candidate_classes"])]
    except (ValueError, TypeError) as e:
        return None, [f"{npz_path}: {e}"]
    # [S, Q]: the gold class differs per seed since section 14.0a, so a
    # single vector would score every seed against seed 42's golds.
    gold_arr = np.asarray(data["gold_class"])
    if gold_arr.ndim != 2:
        return None, [f"{npz_path}: gold_class is {gold_arr.ndim}-D, expected "
                      "[seed, query]; a 1-D vector is the pre-14.0a shared "
                      "axis"]
    try:
        gold = [[strict_int(g, f"gold_class[{i},{j}]")
                 for j, g in enumerate(row)] for i, row in enumerate(gold_arr)]
    except (ValueError, TypeError) as e:
        return None, [f"{npz_path}: {e}"]
    try:
        logits = np.asarray(data["candidate_logits"], dtype=np.float64)
    except (ValueError, TypeError) as e:
        return None, [f"{npz_path}: candidate_logits does not convert to "
                      f"float64: {e}"]
    # FINITE, checked here and not only by the identity validator. This
    # function explicitly supports being called without the identity
    # arguments, and with all-inf logits it returned ([nan], []) -- NaN
    # scores while reporting no fault. A function that vouches for its own
    # arithmetic has to check its own inputs.
    if not np.all(np.isfinite(logits)):
        # ⚠ The refusal stands; the REASON given did not. "every score would
        # be NaN" is false: logits [0, -inf] with gold at index 0 give
        # lse = log(e^0 + 0) = 0 and NLL = 0.0, perfectly finite. A -inf
        # entry is a legitimate "impossible class" in some codebases. What
        # is true is narrower and enough: the recorded run violated the
        # finite-logits contract, so its arithmetic cannot be audited
        # against a recomputation that assumes finite inputs.
        return None, [f"{npz_path}: candidate_logits holds "
                      f"{int((~np.isfinite(logits)).sum())} non-finite "
                      "values, which violates the finite-logits contract "
                      "this recomputation audits against. Some of the "
                      "resulting scores may still be finite -- that is not "
                      "the point; a slab this function cannot reproduce "
                      "cannot vouch for the one on disk."]
    if logits.ndim != 4:
        return None, [f"{npz_path}: candidate_logits is {logits.ndim}-D, "
                      "expected (arm, seed, query, candidate)"]
    # AFTER ndim. Reading shape[0] first raised IndexError on a scalar
    # array -- the check meant to protect the index was itself unguarded.
    if logits.shape[0] != len(arms):
        return None, [f"{npz_path}: candidate_logits has {logits.shape[0]} "
                      f"arm slabs but arms names {len(arms)}; the index used "
                      "below comes from the name list"]
    # A ZERO-SIZE axis passes every shape comparison and then max(axis=-1)
    # raises on an empty reduction. Both axes are required non-empty: a run
    # with no candidates or no observations has nothing to score.
    if logits.shape[3] == 0 or logits.shape[2] == 0:
        return None, [f"{npz_path}: candidate_logits has shape "
                      f"{logits.shape}; a zero-length candidate or query "
                      "axis reduces to nothing and raises rather than "
                      "scoring"]
    if (logits.shape[1] != gold_arr.shape[0]
            or logits.shape[2] != gold_arr.shape[1]
            or logits.shape[3] != len(cand)):
        return None, [f"{npz_path}: candidate_logits {logits.shape} disagrees "
                      f"with gold {gold_arr.shape} and {len(cand)} "
                      "candidates"]
    _dup_c = sorted({c for c in cand if cand.count(c) > 1})
    if _dup_c:
        return None, [f"{npz_path}: candidate_classes repeat {_dup_c}; the "
                      "position map would keep the LAST occurrence and the "
                      "gold would be read from the wrong column"]
    pos = {c: i for i, c in enumerate(cand)}
    if any(g not in pos for row in gold for g in row):
        return None, [f"{npz_path}: a gold class is outside the candidate set, "
                      "so the candidate-conditional NLL is undefined"]
    gi = np.array([[pos[g] for g in row] for row in gold])   # [S, Q]

    scores, faults = [], []
    for cfg in configs:
        arm = config_arm_name(cfg)
        if arm not in arms:
            faults.append(f"{Path(npz_path).name}: no arm {arm!r} for "
                          f"configuration {cfg!r}")
            scores.append(float("nan"))
            continue
        z4 = logits[arms.index(arm)]                       # (seed, query, cand)
        m = z4.max(axis=-1, keepdims=True)
        lse = np.squeeze(m, -1) + np.log(np.exp(z4 - m).sum(axis=-1))
        nll = lse - np.take_along_axis(
            z4, gi[:, :, None], axis=-1).squeeze(-1)
        # FINITE OUTPUT, not merely finite input. lse - logit overflows
        # on [-1e308, 1e308]: both inputs are finite and the difference is
        # not, and the function returned ([inf], []) -- a score it could not
        # have computed, with no fault.
        if not np.all(np.isfinite(nll)):
            faults.append(
                f"{Path(npz_path).name}: arm {arm!r} produced "
                f"{int((~np.isfinite(nll)).sum())} non-finite NLL cells from "
                "finite logits; the subtraction overflowed")
            scores.append(float("nan"))
            continue
        _m = float(nll.mean())
        if not math.isfinite(_m):
            faults.append(f"{Path(npz_path).name}: arm {arm!r} has a "
                          "non-finite mean NLL")
            scores.append(float("nan"))
            continue
        scores.append(_m)
    return scores, faults


def rederive_selection(per_config, configs, tie_tol=SELECTION_TIE_TOL):
    """Which configuration section 5(4)'s rule picks from recorded scores.

    `per_config` is a list of scores aligned with `configs`. Ties within the
    tolerance take the EARLIER entry -- the same shape as the gamma rule and
    the carrier ranking: the selection is re-derived, so the artifact's claim
    about what it chose is checkable rather than asserted.
    """
    best = min(per_config)
    for i, v in enumerate(per_config):
        if v <= best + tie_tol:
            return i
    return int(per_config.index(best))          # unreachable; kept explicit


def validation_selection_blockers(only=None):
    """The selection artifact must PROVE its selection, not merely be on grid.

    `only` scopes the PER-BASELINE loop at the end, so freezing one opponent
    does not require the other five to have selected anything. Everything
    above that loop is a property of the DOCUMENT -- absent, unparseable,
    wrong schema, wrong seeds, wrong metric, wrong tie rule, hashes that do
    not match the manifest and label space on disk -- and is owed by every
    baseline that reads it, so scoping does not touch it. That asymmetry is
    the whole point: the previous caller dropped exactly these, because a
    fault about a shared document has no baseline name to match on.

    Hashing the file proves it did not change. Checking membership of the grid
    proves someone typed a legal value. Neither shows the configuration is the
    one the registered rule picks -- a hand-filled on-grid choice satisfies
    both. So the artifact records the per-configuration validation scores, the
    metric, the tie rule, the seeds and the hashes of everything the scores
    were computed on, and the selection is RE-DERIVED here from those scores.
    """
    p = Path(REPO, "results", "baseline_validation_selection.json")
    if not p.is_file():
        return [f"results/{p.name} is missing (section 5(4): the configuration "
                "each baseline selected on validation)"]
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                   # noqa: BLE001
        return [f"results/{p.name} did not parse: {e}"]
    # THE ROOT. `42`, `null` and `true` are valid JSON and satisfied
    # nothing below -- `k not in doc` raises TypeError on an int before any
    # blocker can be produced.
    if not isinstance(doc, dict):
        return [f"{p.name}: the document is {type(doc).__name__}, expected a "
                "JSON object"]
    bad = [f"{p.name}: no {k!r}" for k in SELECTION_REQUIRED if k not in doc]
    if bad:
        return bad
    # SHAPE, not just presence. `evidence: []` satisfied every key check and
    # then raised AttributeError on .get -- a structured-blocker function
    # that throws is not producing blockers. Each entry too: a list where a
    # mapping belongs fails the same way one level down.
    for k in ("selected", "evidence"):
        if not isinstance(doc[k], dict):
            bad.append(f"{p.name}: {k!r} is {type(doc[k]).__name__}, expected "
                       "an object keyed by baseline name")
    if bad:
        return bad
    for k, v in doc["evidence"].items():
        if not isinstance(v, dict):
            bad.append(f"{p.name}: evidence[{k!r}] is "
                       f"{type(v).__name__}, expected an object")
    if bad:
        return bad

    try:
        _sv = strict_int(doc["schema_version"], "schema_version")
    except (ValueError, TypeError) as e:
        return bad + [f"{p.name}: {e}"]
    if _sv != SELECTION_SCHEMA_VERSION:
        bad.append(f"{p.name}: schema v{doc['schema_version']}, this code "
                   f"expects v{SELECTION_SCHEMA_VERSION}; rebuild rather than "
                   "let an older shape be reinterpreted")
    if doc["spec"] != SELECTION_SPEC:
        bad.append(f"{p.name}: spec {doc['spec']!r} != {SELECTION_SPEC!r}")
    try:
        _ds_ok = [strict_int(s, f"demo_seeds[{i}]")
                  for i, s in enumerate(doc["demo_seeds"])] == list(
                      REGISTERED_DEMO_SEEDS)
    except (ValueError, TypeError) as e:
        bad.append(f"{p.name}: {e}")
        _ds_ok = True
    if not _ds_ok:
        bad.append(f"{p.name}: seeds {doc['demo_seeds']} are not the registered "
                   f"{list(REGISTERED_DEMO_SEEDS)}")
    if doc["metric"] != SELECTION_METRIC:
        bad.append(f"{p.name}: metric {doc['metric']!r} != {SELECTION_METRIC!r}")
    if doc["tie_rule"] != SELECTION_TIE_RULE:
        bad.append(f"{p.name}: tie rule {doc['tie_rule']!r} is not the "
                   "registered one")
    for key, rel in (("query_manifest_sha256",
                      "results/prereg_method_A_query_manifest.json"),
                     ("label_space_sha256", "results/label_space_llama31.json")):
        f = Path(REPO, rel)
        if not f.is_file():
            bad.append(f"{rel} is missing, so {key} cannot be checked")
        elif doc[key] != sha256_file(f):
            bad.append(f"{p.name}: {key} {str(doc[key])[:12]} != {rel} on disk "
                       f"{sha256_file(f)[:12]}")

    sel, ev = doc["selected"], doc["evidence"]
    scope = set(only) if only is not None else set(registry.BASELINES)
    for name, spec in sorted(registry.BASELINES.items()):
        if name not in scope:
            continue
        if name not in sel:
            bad.append(f"{p.name}: no selection recorded for {name}")
            continue
        if sel[name] not in spec["configs"]:
            bad.append(f"{p.name}: {name} selected {sel[name]!r}, which is not "
                       f"in its frozen grid {spec['configs']}")
            continue
        e = ev.get(name) or {}
        miss = [k for k in EVIDENCE_REQUIRED if not e.get(k)]
        if miss:
            bad.append(f"{p.name}: {name} records no {miss}; without a path "
                       "the hash names no file and the scores answer to "
                       "nothing")
            continue
        # The PATH first: an absolute path or a traversal does not exist, so
        # the existence check below would short-circuit with a message about
        # a missing file and the registered-path rule would never be reached.
        want_path = expected_result_npz(name)
        if e["result_npz_path"] != want_path:
            bad.append(f"{p.name}: {name}'s result_npz_path "
                       f"{e['result_npz_path']!r} is not the one registered "
                       f"path {want_path!r}; an arbitrary path lets the "
                       "evidence come from anywhere, including another arm's "
                       "file")
            continue
        npz = Path(REPO, e["result_npz_path"])
        if not npz.is_file():
            bad.append(f"{p.name}: {name}'s result_npz_path "
                       f"{e['result_npz_path']!r} does not exist")
            continue
        on_disk = sha256_file(npz)
        if e["result_npz_sha256"] != on_disk:
            bad.append(f"{p.name}: {name}'s NPZ hashes to {on_disk[:12]} but "
                       f"the selection records {str(e['result_npz_sha256'])[:12]}")
            continue

        # RECOMPUTE. Checking that the recorded scores are self-consistent with
        # the recorded choice proves only that whoever filled the file was
        # consistent with themselves: hand-written NLLs, a fabricated hash, and
        # a matching choice passed every earlier check.
        scores, faults = recompute_per_config_nll(
            npz, spec["configs"], name=name,
            manifest=Path(REPO, "results",
                          "prereg_method_A_query_manifest.json"),
            label_space=Path(REPO, "results", "label_space_llama31.json"))
        if faults:
            bad += [f"{p.name}: {name}: {f}" for f in faults]
            continue
        recorded = e["per_config_nll"]
        # SPLIT: "not a list" and "wrong length" are different faults, and
        # the combined form called len() on whatever it was handed -- 1 has
        # no len(), so the blocker function raised.
        if not isinstance(recorded, list):
            bad.append(f"{p.name}: {name}'s per_config_nll is "
                       f"{type(recorded).__name__}, expected a list")
            continue
        if len(recorded) != len(spec["configs"]):
            bad.append(f"{p.name}: {name} records {len(recorded)} scores "
                       f"for {len(spec['configs'])} configurations")
            continue
        # CONVERT under control. numbers.Real admits an unbounded int, and
        # float(10**1000) raises OverflowError -- the same escape the
        # identity fields had, moved here. Non-finite is refused too: a
        # recorded inf compares to nothing meaningful.
        _rec = []
        _bad_r = []
        for i, r in enumerate(recorded):
            if isinstance(r, bool) or not isinstance(r, numbers.Real):
                _bad_r.append((i, f"{type(r).__name__} {r!r}"))
                continue
            try:
                f = float(r)
            except (OverflowError, ValueError) as e:
                _bad_r.append((i, f"does not convert to a float: {e}"))
                continue
            if not math.isfinite(f):
                _bad_r.append((i, f"is {f}"))
                continue
            _rec.append(f)
        if _bad_r:
            bad.append(f"{p.name}: {name}'s per_config_nll holds "
                       f"{len(_bad_r)} unusable entries (first at "
                       f"{_bad_r[0][0]}: {_bad_r[0][1]})")
            continue
        recorded = _rec
        off = [(i, r, c) for i, (r, c) in enumerate(zip(recorded, scores))
               if not abs(r - c) <= NLL_RECOMPUTE_TOL]
        if off:
            bad.append(
                f"{p.name}: {name}'s recorded NLLs do not match the ones "
                f"recomputed from its own NPZ ({len(off)} of "
                f"{len(scores)} differ; first config {off[0][0]}: recorded "
                f"{off[0][1]}, recomputed {off[0][2]:.9f})")
            continue
        # the selection follows the RECOMPUTED values, not the recorded ones
        want = spec["configs"][rederive_selection(scores, spec["configs"])]
        if sel[name] != want:
            bad.append(
                f"{p.name}: {name} selected {sel[name]!r} but the registered "
                f"rule applied to the NLLs RECOMPUTED from its NPZ picks "
                f"{want!r}. On grid is not the same as chosen by the rule.")
    extra = sorted(set(sel) - set(registry.BASELINES))
    if extra:
        bad.append(f"{p.name}: selections for unknown arms {extra}")
    return bad


# Artifacts the CODE stage must hash, beyond the runner sources. Existence is
# not enough: an empty baseline_adapt_diffs/ satisfied the previous check, and
# a discovery product that is never hashed can be regenerated between the
# freeze and the run without anything noticing. The discovery POOL is no
# longer among them: it was abolished in section 14.0a, and listing a file
# that cannot exist would make the freeze unsatisfiable rather than strict.
#
# THE FOURTH FIELD SAYS WHOSE OBLIGATION IT IS, because a per-baseline freeze
# has to answer that and the tuple used not to carry it:
#
#   method_a   not a baseline's at all. Requiring Method A's carriers before
#              an OPPONENT can be code-frozen is exactly the coupling the
#              user's ruling removed ("only the dataset is shared").
#   selection  one shared file, per-baseline CONTENT.
#   adapt      one shared directory, per-baseline FILES.
#   shared     genuinely every baseline's, and owed by each on its own.
#
# The last three keep their blocker here only for an unscoped freeze; when a
# subset is being frozen, the per-baseline owners below emit the message with
# the name in it, so the same fault is reported once and attributably.
CODE_STAGE_ARTIFACTS = (
    ("results/method_a_carriers_L31_top8.json", "file", "method_a",
     "the frozen Method A carriers (section 2.4(2))"),
    ("results/baseline_validation_selection.json", "file", "selection",
     "the configuration each baseline selected on validation (section 5(4))"),
    ("results/baseline_adapt_diffs", "dir:*.patch", "adapt",
     "the adaptation diffs (section 0.5)"),
    ("results/baseline_verification_report.md", "file", "shared",
     "the verification report (section 6.1)"),
)


def selection_result_npzs(only=None):
    """The result NPZs the selection points at, so they are hashed too.

    Without this a freeze could be taken and the NPZ edited afterwards: the
    selection file would still hash the same and its recomputation would then
    agree with the NEW numbers.
    """
    p = Path(REPO, "results", "baseline_validation_selection.json")
    if not p.is_file():
        return []
    try:
        ev = (json.loads(p.read_text(encoding="utf-8")).get("evidence") or {})
    except Exception:                                        # noqa: BLE001
        return []
    if only is not None:
        ev = {k: v for k, v in ev.items() if k in set(only)}
    out = set()
    for v in ev.values():
        if isinstance(v, dict) and v.get("result_npz_path"):
            out.add(v["result_npz_path"])
            out.add(v["result_npz_path"].replace(".npz", ".json"))
    return sorted(out)


def artifact_hashes(only=None, include_method_a=True):
    """{path: sha256} for the code-stage artifacts, or a list of what is
    missing. Directories are expanded, and an EMPTY one is missing.

    `only` restricts to one baseline's obligations. It changes WHICH FAULTS
    ARE REPORTED HERE, never which files are hashed: everything present is
    still hashed, because a freeze that pins fewer files than it read is the
    original problem. What a scoped call drops is `method_a` (not an
    opponent's obligation) and the messages the per-baseline owners below
    state better -- with the baseline's name in them.
    """
    out, bad = {}, []
    for rel in selection_result_npzs(only=only):
        f = Path(REPO, rel)
        if not f.is_file():
            bad.append(f"{rel} is missing (a result NPZ the validation "
                       "selection points at)")
        else:
            out[rel] = sha256_file(f)
    for rel, kind, scope, what in CODE_STAGE_ARTIFACTS:
        if scope == "method_a" and not include_method_a:
            continue
        # `selection` and `adapt` are reported by their per-baseline owners
        # whenever a scope was given, so that the message carries the name.
        mine = only is None or scope in ("shared", "method_a")
        p = Path(REPO, rel)
        if kind == "file":
            if not p.is_file():
                if mine:
                    bad.append(f"{rel} is missing ({what})")
            else:
                out[rel] = sha256_file(p)
            continue
        pattern = kind.split(":", 1)[1]
        files = sorted(p.glob(pattern)) if p.is_dir() else []
        if not files and mine:
            bad.append(f"{rel}/ holds no {pattern} ({what}); an empty "
                       "directory satisfied the old existence check")
        for f in files:
            out[f"{rel}/{f.name}"] = sha256_file(f)
    return out, bad


def validate_freeze(rec, *, require_baselines_ready=True):
    """Is a freeze file VALID -- not merely undrifted? Returns blockers.

    THE FREEZE IS FOR REPRODUCIBILITY, NOT FAIRNESS. It exists so that a
    result file can be traced to the code, the gamma*, the carriers and the
    queries that produced it. It cannot prove nobody tuned an opponent after
    seeing a result -- anyone can edit and re-freeze -- so trying to enforce
    that here bought friction and no evidence. What proves it is a DATED
    RECORD: the commit, and section 14.

    `require_baselines_ready=False` therefore drops everything that is about
    the OPPONENTS -- their statuses, their runner files, their continued
    presence in the registry -- and keeps everything about the analysis code
    and the record's integrity. Section 2.1's condition on reading test is
    "the three gamma*, the control carriers/positions and the analysis code
    are frozen"; it says nothing about the opponents, and welding the two
    together made writing five baseline runners a prerequisite for reading
    Method A's own test split (section 14.1's C12 friction, heavier). H6 and
    the viability gate keep the default, because their runs DO read the
    opponent configurations and so their reproducibility depends on them.

    ⚠ A consumer that only calls `compare` asks "has anything changed since
    this was written", which a freeze written by the old fail-open tool, or a
    hand-authored one, answers perfectly well while pinning nothing. The
    validity conditions have to be re-checked against the RECORDED content, by
    whoever is about to rely on it.
    """
    bad = []
    if not isinstance(rec, dict):
        return ["the freeze is not a JSON object"]
    for key in ("baselines", "documents", "upstream_repos", "stage",
                "decision_sha256"):
        if key not in rec:
            bad.append(f"no {key!r} section; this is not a freeze this code "
                       "wrote")
    if bad:
        return bad
    if rec["stage"] not in STAGES:
        bad.append(f"unknown stage {rec['stage']!r}")
    # TYPE FIRST. `set(rec["baselines"])` below raises on an int and silently
    # iterates characters on a string; both are valid JSON.
    if not isinstance(rec.get("baselines"), dict):
        return bad + [f"'baselines' is "
                      f"{type(rec.get('baselines')).__name__}, expected an "
                      "object keyed by baseline name"]
    repos = rec.get("upstream_repos") or {}
    if set(repos) != set(UPSTREAM):
        bad.append(f"records {len(repos)} upstream repos, expected "
                   f"{len(UPSTREAM)}")
    bad += [f"recorded as unreadable -- {b}" for b in unreadable_repos(repos)]
    bad += [f"recorded as dirty -- {b}" for b in dirty_repos(repos)]
    bad += [f"recorded at the wrong commit -- {b}" for b in wrong_commit(repos)]
    # SUBSET, not equality. The opponent set is still provisional; requiring
    # it to match the registry exactly meant adding a seventh baseline
    # invalidated a freeze that described the six it recorded perfectly, and a
    # freeze that self-invalidates on ordinary work is one people rewrite
    # casually. 13.6.1 guards against DROPPING or ALTERING an opponent after
    # seeing a result -- an addition cannot launder anything, since a new
    # opponent only makes the comparison harder and Holm over a larger family
    # is strictly more conservative.
    vanished = sorted(set(rec["baselines"]) - set(registry.BASELINES))
    if vanished and require_baselines_ready:
        bad.append(f"records baselines {vanished} that the registry no longer "
                   "has, so an H6 run cannot reproduce what this freeze "
                   "describes")
    # THE STORED DECISION HASH, AGAINST THE RECORD'S OWN CONTENT. `compare`
    # stopped reading this field when it began recomputing the hash over the
    # recorded opponents only (14.0b-2 (q)), which left the stored value
    # decorative: editing a config and the hash together drifted nothing.
    # Recomputing it here from the record alone catches that, and does not
    # involve the registry, so adding a baseline still costs nothing.
    want_dec = _decision_hash(rec, sorted(rec["baselines"]))
    if rec.get("decision_sha256") != want_dec:
        bad.append(
            f"decision_sha256 says {str(rec.get('decision_sha256'))[:12]} but "
            f"this record's own baselines hash to {want_dec[:12]}; the file "
            "has been edited since it was written")

    # THE CHAIN, IF THERE IS ONE. 14.0a requires the superseded freeze kept
    # and still verifiable; the first attempt recorded `supersedes` as a plain
    # string that validate_freeze ignored, so deleting it passed --verify and
    # the field was a deletable log rather than evidence.
    if "supersedes" in rec:
        sup = rec["supersedes"]
        if not isinstance(sup, dict):
            bad.append(f"'supersedes' is {type(sup).__name__}, expected an "
                       "object with path, sha256, stage and reason")
        else:
            miss = [k for k in SUPERSEDES_REQUIRED if not sup.get(k)]
            if miss:
                bad.append(f"'supersedes' is missing {miss}; a predecessor "
                           "recorded without all of these cannot be found, "
                           "checked, or explained")
            elif sup["stage"] not in STAGES:
                bad.append(f"'supersedes' names unknown stage "
                           f"{sup['stage']!r}")
            elif STAGES.index(rec["stage"]) < STAGES.index(sup["stage"]):
                bad.append(
                    f"a {rec['stage']!r}-stage freeze may not supersede a "
                    f"{sup['stage']!r}-stage one. Superseding DOWNWARD is a "
                    "route around the final code freeze")
            else:
                prev = Path(sup["path"])
                if not prev.is_file():
                    bad.append(
                        f"the superseded freeze {sup['path']} is not on disk. "
                        "14.0a keeps the old file and requires it to stay "
                        "verifiable -- a chain whose predecessor is gone "
                        "records nothing")
                elif sha256_file(prev) != sup["sha256"]:
                    bad.append(
                        f"{sup['path']} hashes to "
                        f"{sha256_file(prev)[:12]} but the chain records "
                        f"{str(sup['sha256'])[:12]}; the predecessor has been "
                        "edited since it was superseded")
    if rec["stage"] == "code":
        if not rec.get("artifacts"):
            bad.append("a code-stage freeze records no artifact hashes")
        # THE `code` SECTION ITSELF. Requiring only `artifacts` meant a
        # spec-stage record with the stage label flipped to "code" and any
        # non-empty artifacts map validated -- and `compare` then skipped the
        # code group entirely, because it only compares a group the record
        # carries. The two gaps composed into "relabel and you are frozen".
        # THE COMMIT IS WHAT PINS THE CODE. A record that does not name one,
        # or names one taken from a dirty tree, cannot say what produced a
        # number -- and a spec-stage freeze relabelled "code" has no `repo`
        # block at all, which is what makes that relabelling detectable.
        repo = rec.get("repo") if isinstance(rec.get("repo"), dict) else {}
        if not repo.get("available") or not repo.get("head"):
            bad.append(
                "a code-stage freeze records no repository commit, so nothing "
                "says which code produced its numbers (a spec-stage freeze "
                "relabelled 'code' looks exactly like this)")
        elif not repo.get("clean"):
            bad.append(
                f"the recorded commit {str(repo.get('head'))[:12]} was taken "
                "from a DIRTY tree, so it does not identify the code that ran")
        # THE RECORDED STATUSES. code_stage_blockers refuses to WRITE one
        # while five registry entries say NOT-IMPLEMENTED; a reader has to
        # re-check that from the record, or a hand-made file walks past it.
        # RETURNS the fault on a malformed entry: `"baselines": {"x": "READY"}`
        # is valid JSON and used to raise AttributeError out of .get, past
        # every caller catching PermissionError.
        recorded = rec.get("baselines")
        illtyped = sorted(n for n, s in recorded.items()
                          if not isinstance(s, dict))
        if illtyped:
            bad.append(f"baseline entries {illtyped} are not objects; each "
                       "must be a record with a 'status'")
        notready = sorted(n for n, s in recorded.items()
                          if isinstance(s, dict)
                          and s.get("status") not in READY_STATUSES)
        if notready and require_baselines_ready:
            bad.append(f"a code-stage freeze records {len(notready)} baseline"
                       f"(s) that are not ready: {notready}. The code stage IS "
                       "the claim that the six runners exist and are done")
    return bad


def runner_exists_at(name, commit):
    """Does tools/baselines/run_<name>.py exist at `commit`?

    THE COMMIT IS THE STATUS. A baseline whose runner is absent there is
    NOT-IMPLEMENTED by that fact, so nothing has to maintain a status field
    by hand and nothing can disagree with the repository. Returns None when
    the question cannot be answered -- no commit recorded, or git could not
    be reached -- which is different from False and is reported as such.
    """
    if not commit:
        return None
    path = f"tools/baselines/run_{name}.py"
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "-e", f"{commit}:{path}"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.returncode == 0


def implemented_baselines():
    """{name: bool|None} -- built, not built, or unanswerable."""
    return {n: runner_exists_at(n, sp.get("runner_commit"))
            for n, sp in sorted(registry.BASELINES.items())}


def code_stage_blockers(only=None):
    """Why this checkout cannot be frozen at code stage, PER BASELINE.

    NOT ALL-OR-NOTHING any more. The old version required all six runners to
    exist before anybody could freeze, which welded together two different
    claims -- "the opponent set is decided", which 13.6.1 needs before the
    viability gate, and "the opponents are built", which only matters when H6
    runs. It made writing five baseline runners a precondition for a code
    freeze that is otherwise about Method A's own analysis code.

    Each baseline stands alone; the only thing they share is the dataset.
    `only` restricts the check to a subset, so one baseline can be frozen
    without waiting for the others.

    Everything downstream is scoped to the baselines that CLAIM TO BE BUILT:
    an adaptation diff, a validation selection and a result NPZ are evidence
    ABOUT A RUN, and demanding them of a runner that does not exist is
    demanding evidence about something that has not happened.
    """
    names = sorted(only if only is not None else registry.BASELINES)
    bad = []
    built = []
    for name in names:
        spec = registry.BASELINES[name]
        commit = spec.get("runner_commit")
        if not commit:
            bad.append(f"{name}: no runner_commit, so nothing records whether "
                       "it is built. Freeze it on its own once its runner "
                       "exists; it does not block the others")
            continue
        ex = runner_exists_at(name, commit)
        if ex is None:
            bad.append(f"{name}: runner_commit {str(commit)[:12]} could not be "
                       "read (unknown commit, or git unavailable)")
        elif not ex:
            bad.append(f"{name}: run_{name}.py does not exist at "
                       f"{str(commit)[:12]}, which IS what NOT-IMPLEMENTED "
                       "means. Recording that is fine; claiming a code stage "
                       "for it is not")
        else:
            built.append(name)
    if not built:
        return bad
    # SCOPED BY THE CALLEE, NOT BY A SUBSTRING MATCH ON ITS PROSE.
    #
    # This was `[b for b in f() if any(n in b for n in built)]` on all three,
    # and a fault about a SHARED FILE has no baseline name in it to match --
    # so the missing validation selection, the missing verification report,
    # the empty diff directory and the missing carriers were all silently
    # discarded. code_stage_blockers(only=["inductive_bc"]) returned [] while
    # inductive_bc had met none of them, and the integration arm asserting
    # `== []` was green FOR THAT REASON. The arm below now asserts that a
    # shared obligation survives scoping, which is what it should have said.
    # TWO DIFFERENT SCOPES, and conflating them is what broke this.
    #
    #   per-baseline EVIDENCE (its adaptation diff, its selection entry, its
    #   result NPZ) is scoped to `built` even on an unscoped call: asking a
    #   runner that does not exist for a diff of what was changed to run it
    #   is asking for evidence about something that has not happened;
    #
    #   the SHARED DOCUMENT (the verification report, the selection file's
    #   existence and schema, and -- only when freezing everything -- Method
    #   A's carriers) is owed regardless of who is built, and has no baseline
    #   name in it. That is precisely why the old substring filter dropped it.
    bad += artifact_hashes(only=list(built),
                           include_method_a=(only is None))[1]
    bad += missing_adapt_diffs(only=list(built))
    bad += validation_selection_blockers(only=list(built))
    return bad


STAGES = ("spec", "code")          # ordered: index is the strength order

# Every field a recorded predecessor needs. `reason` is required because
# 14.0b-2 (s) makes this chain the DATED RECORD that a gate could never be --
# "why was this re-frozen" is exactly what section 14 wants, and it is not
# something anyone reconstructs later.
SUPERSEDES_REQUIRED = ("path", "sha256", "stage", "reason")

def build_freeze(stage="spec") -> dict:
    """The freeze content for one stage. See the module docstring for why two.

    `code` is a SUPERSET: it carries the spec block too, so a code-stage freeze
    still pins the configs and hook boundaries.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    caps = registry.check_caps()
    if caps:
        raise SystemExit("the registry violates section 2.4(5):\n  "
                         + "\n  ".join(caps))
    out = {
        "spec": "prereg_method_A.md section 13.6.1 -- freeze the opponents "
                "before the viability gate",
        "why": "a visible viability result creates room to adjust the "
               "baselines; this file makes any such adjustment visible and "
               "dateable, it does not prevent one",
        "baselines": {
            name: {
                "arm_name": s["arm_name"],
                "status": s["status"],
                "n_configs": len(s["configs"]),
                "configs": s["configs"],
                "config_axis": s["config_axis"],
                "placebo": s["placebo"],
                "placebo_kind": s["placebo_kind"],
                "hook_boundary": s["hook_boundary"],
                "hook_boundary_sha256": sha256_text(s["hook_boundary"]),
                "selection": s["selection"],
                "upstream": s["upstream"],
            } for name, s in sorted(registry.BASELINES.items())},
        "excluded": registry.EXCLUDED,
        "config_cap": {"max_total_configs": registry.MAX_CONFIGS,
                       "exceptions": ["unibias_code", "deepthinking_dev"],
                       "cap_check": "PASS"},
        "documents": {
            "baseline_under_review.md": sha256_file(SPEC_DOC),
            "prereg_method_A.md::13.2": sha256_text(
                section_slice(PREREG, SECTION_START, SECTION_END)),
            "prereg_method_A.md::13.6": sha256_text(
                section_slice(PREREG, "### 13.6", "## 14.")),
        },
        "upstream_root": str(upstream_root()),
        "upstream_repos": {
            name: git_head(upstream_root() / name) for name in UPSTREAM},
    }
    out["stage"] = stage
    # One hash over the entire frozen decision -- every baseline field, the
    # exclusions and the config cap -- so nothing depends on a maintained list
    # of which fields matter. sort_keys makes it independent of dict order.
    out["decision_sha256"] = sha256_text(json.dumps(
        {"baselines": out["baselines"], "excluded": out["excluded"],
         "config_cap": out["config_cap"]},
        sort_keys=True, ensure_ascii=False))
    if stage == "code":
        # THE COMMIT, NOT THE FILES. Hashing sources meant a hand-written
        # list that missed a module twice, an import-closure that pinned 19
        # files of which 12 are shared with other work, and a test split that
        # re-locked whenever one of them was edited for an unrelated reason.
        # One commit id covers all of it and everything the list would have
        # missed -- tasks/, script/, the fixtures -- and it is what a reader
        # needs to reproduce a number.
        out["repo"] = git_head(REPO)
        # The discovery products, adaptation diffs and selected validation
        # configurations, hashed rather than merely present.
        out["artifacts"] = artifact_hashes()[0]
    return out


def _decision_hash(rec, names):
    """The canonical decision hash, restricted to `names`.

    Used so that ADDING an opponent does not invalidate a freeze that still
    describes the recorded ones exactly. The full-registry hash is still
    written into every freeze as `decision_sha256`; this is what `compare`
    asks, because the question there is "did anything I froze change", not
    "is the registry identical to what it was".
    """
    b = rec.get("baselines") or {}
    return sha256_text(json.dumps(
        {"baselines": {k: b[k] for k in names if k in b},
         "excluded": rec.get("excluded"),
         "config_cap": rec.get("config_cap")},
        sort_keys=True, ensure_ascii=False))


def compare(recorded: dict, current: dict, *, include_baselines=True,
            include_repo=True) -> list:
    """Every drifted item, named. Missing on either side is drift too.

    Source files are NOT compared. A code-stage freeze records the repository
    commit, and `report_repo_drift` says whether this checkout is that commit;
    per-file hashing was tried twice, missed a module both times, and re-locked
    the test splits whenever a shared utility was edited for another purpose.
    """
    bad = []
    # NO `code` group: the commit in `repo` is compared below instead, and a
    # per-file hash list was both incomplete and too noisy to live with.
    groups = ["documents"] + (["artifacts"] if "artifacts" in recorded else [])
    for group in groups:
        r, c = recorded.get(group, {}), current.get(group, {})
        for k in sorted(set(r) | set(c)):
            if k not in r:
                # A file the freeze never pinned is not a file that moved.
                # Adding a baseline brings tools/baselines/run_<name>.py with
                # it, and calling that drift would make the provisional
                # opponent set impossible to extend. REMOVED and CHANGED below
                # are what 13.6.1 and section 2.1 actually guard.
                continue
            elif k not in c:
                bad.append(f"{group}/{k}: REMOVED since the freeze")
            elif r[k] != c[k]:
                bad.append(f"{group}/{k}: {str(r[k])[:12]} -> {str(c[k])[:12]}")
    # THE WHOLE DECISION, canonically hashed. A field whitelist -- configs,
    # hook, placebo, selection, config_axis -- left `status`, `placebo_kind`,
    # `arm_name`, `excluded` and `config_cap` free to change without drifting,
    # and would have needed maintenance every time a field was added. Hashing
    # the canonicalised block means a field nobody thought to list is covered
    # the moment it exists.
    # RESTRICTED TO THE RECORDED OPPONENTS. The full hash covers every
    # baseline in the registry, so ADDING a seventh opponent flipped it and
    # invalidated a freeze that still described the six it recorded perfectly.
    # 13.6.1 forbids adjusting an opponent after seeing the result -- loosening
    # a grid, moving an injection layer, dropping one that beat us. Adding one
    # is not that move: a new opponent only makes the comparison harder, and
    # Holm over a larger family is strictly more conservative, so it cannot
    # launder a bad result. It is dated by the commit; it is not drift.
    # THE COMMIT. `include_repo=False` for the test lock: 14.0b-2 (v) allows a
    # run at a different commit from the freeze, because the result carries
    # its own commit and refusing would rebuild the friction that dropping
    # per-file hashes removed. `--verify` wants to hear about it.
    # ONLY WHEN THE RECORD HAS ONE, like `artifacts`. A spec-stage freeze
    # deliberately carries no `repo`, so comparing it against a code stage's
    # made the sanctioned spec -> code upgrade look like drift.
    if include_repo and recorded.get("repo"):
        r, c = recorded["repo"], current.get("repo") or {}
        if r.get("head") != c.get("head"):
            bad.append(f"repo/head: {str(r.get('head'))[:12]} -> "
                       f"{str(c.get('head'))[:12]} -- this checkout is not "
                       "the commit the freeze was taken at")
        elif r.get("clean") != c.get("clean"):
            bad.append(f"repo/clean: {r.get('clean')} -> {c.get('clean')}")
    if not include_baselines:
        # THE FREEZE IS FOR REPRODUCIBILITY, NOT FAIRNESS. A consumer whose
        # run does not read the opponent configurations -- Method A's own test
        # forward, for instance -- is not made less reproducible by the
        # registry having changed since: the record still says exactly what it
        # said, and this run does not use it. Blocking there imported a
        # fairness argument into a provenance check.
        return bad
    rec_names = sorted(recorded.get("baselines", {}))
    if (_decision_hash(recorded, rec_names)
            != _decision_hash(current, rec_names)):
        bad.append("decision_sha256 (restricted to the recorded opponents): "
                   "the frozen baseline decision changed. The per-field lines "
                   "below say where.")
    r, c = recorded.get("baselines", {}), current.get("baselines", {})
    for k in sorted(set(r) | set(c)):
        if k not in r:
            # NOT drift. The baseline set is still provisional, and a freeze
            # that self-invalidates every time one is added is a freeze that
            # gets rewritten casually -- which pins nothing. What 13.6.1
            # guards is below: a recorded opponent removed or altered.
            continue
        elif k not in c:
            bad.append(f"baselines/{k}: REMOVED since the freeze")
            continue
        else:
            for field in sorted(set(r[k]) | set(c[k])):
                if r[k].get(field) != c[k].get(field):
                    bad.append(f"baselines/{k}/{field} CHANGED -- this is the "
                               "kind of change section 13.6.1 exists to catch")
    for block in ("excluded", "config_cap"):
        if recorded.get(block) != current.get(block):
            bad.append(f"{block} CHANGED -- dropping an excluded method or "
                       "moving the config cap is a change to the opponent set")
    if recorded.get("upstream_root") != current.get("upstream_root"):
        bad.append(f"upstream_root: {recorded.get('upstream_root')} -> "
                   f"{current.get('upstream_root')}; the checkouts being "
                   "pinned are not the ones the freeze recorded")
    for k in sorted(set(recorded.get("upstream_repos", {}))
                    | set(current.get("upstream_repos", {}))):
        a = (recorded.get("upstream_repos") or {}).get(k) or {}
        b = (current.get("upstream_repos") or {}).get(k) or {}
        # `available`, `clean` and `diff_sha256` are compared alongside `head`.
        # Comparing HEAD alone made "same commit, edited working tree" invisible
        # and let an unreadable repository pass as unchanged.
        for field in ("available", "head", "clean", "diff_sha256"):
            if a.get(field) != b.get(field):
                bad.append(f"upstream/{k}/{field}: {str(a.get(field))[:12]} -> "
                           f"{str(b.get(field))[:12]}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--stage", choices=STAGES, default="spec",
                    help="'spec' before the viability gate (configs, hook "
                         "boundaries, documents, upstream commits); 'code' at "
                         "the final READY freeze, which adds every runner file")
    ap.add_argument("--upstream-root", default=None,
                    help="directory holding the seven upstream checkouts "
                         "(default: the parent of this repository)")
    ap.add_argument("--output", default=str(OUTPUT))
    ap.add_argument("--supersedes",
                    help="path of the freeze this one replaces. The old file "
                         "is KEPT and stays verifiable (14.0a); --output must "
                         "therefore name a different file.")
    ap.add_argument("--supersede-reason",
                    help="why the re-freeze happened, one line. Required with "
                         "--supersedes: this chain is the dated record that a "
                         "gate cannot be (14.0b-2 (s)).")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--integration-test", action="store_true",
                    help="check the REAL upstream checkouts (needs git access "
                         "to them); kept out of --self-test so unit fixtures "
                         "do not depend on local git ownership")
    ap.add_argument("--propose-commits", action="store_true",
                    help="print the current upstream HEADs as a registry block "
                         "to paste in after confirming them on the machine "
                         "that will run the experiments")
    args = ap.parse_args(argv)
    if args.upstream_root:
        globals()["UPSTREAM_ROOT"] = Path(args.upstream_root)
    if args.self_test:
        return self_test()
    if args.integration_test:
        return integration_test()
    if args.propose_commits:
        # ⚠ NOT the answer to "the server is at a different commit". Section
        # 13.1 already fixes the versions, so a mismatch means the CHECKOUT is
        # wrong, not the registry: check the server out to the registered
        # commit. Editing the registry to match whatever the server has is
        # transcribing the machine -- exactly the failure the registered
        # commits exist to prevent -- and it would rewrite a preregistered
        # value to match an observation.
        #
        # This mode exists for one case: the registered commit is genuinely
        # unavailable (upstream force-pushed, repository gone). That is a
        # DEVIATION, must be recorded in section 14 BEFORE any result is seen,
        # and needs prereg 13.1 updated in the same change.
        print("# !! A MISMATCH IS NORMALLY THE CHECKOUT'S FAULT, NOT THE "
              "REGISTRY'S.")
        print("#   Section 13.1 fixes these versions. If a repository is at a "
              "different")
        print("#   commit, check it out to the registered one:")
        # pinned_repos(), not BASELINES: the seventh repository is pinned
        # too, so iterating the six printed six commands under a claim of
        # "all seven".
        for d, (_n, c) in sorted(pinned_repos().items()):
            print(f"#     git -C {upstream_root() / d} checkout {c}")
        print("#")
        print("#   Use the block below ONLY if a registered commit is truly "
              "unavailable.")
        print("#   That is a deviation: record it in section 14 BEFORE seeing "
              "any result,")
        print("#   and update prereg 13.1 in the same change.")
        # pinned_repos(), like the loop above. Fixing only the checkout
        # loop and then claiming "all seven" was wrong twice over: the
        # paste-in block still walked BASELINES, and my check counted
        # `git -C` lines -- exactly the half I had fixed.
        for d, (name, _c) in sorted(pinned_repos().items()):
            h = git_head(upstream_root() / d)
            if h.get("available"):
                print(f'    "{name}": "registered_commit": "{h["head"]}",'
                      + ("" if h.get("clean") else "   # DIRTY"))
            else:
                print(f'    # {name} ({d}): UNREADABLE -- {h.get("why")}')
        return 0
    if args.write == args.verify:
        raise SystemExit("pass exactly one of --write / --verify")

    cur = build_freeze(args.stage)
    p = Path(args.output)

    if args.write:
        # Fail-closed on the upstream side. A freeze that records "could not
        # read this repository" pins nothing, and printing PASS over seven of
        # those is worse than having no freeze at all -- it looks like
        # evidence.
        blockers = (missing_registered_commits()
                    + unreadable_repos(cur["upstream_repos"])
                    + wrong_commit(cur["upstream_repos"])
                    + dirty_repos(cur["upstream_repos"]))
        if blockers:
            raise SystemExit(
                "refusing to write a freeze that pins nothing:\n  "
                + "\n  ".join(blockers)
                + "\n\nEvery upstream repository must be READABLE, CLEAN, and "
                  "at the commit the preregistration names.\n"
                  "  * unreadable: fix it, do not freeze an absence -- "
                  "`git config --global --add safe.directory <path>`\n"
                  "  * dirty: section 0.5 puts adaptation diffs in "
                  "results/baseline_adapt_diffs/<name>.patch, so the checkouts "
                  "themselves are meant to be pristine. Commit, stash or "
                  "extract the change.\n"
                  "  * wrong commit: check out the registered one.")
        if args.stage == "code":
            blockers = code_stage_blockers()
            if blockers:
                raise SystemExit(
                    "refusing to write a READY freeze:\n  "
                    + "\n  ".join(blockers)
                    + "\nThe code stage asserts that the six runners exist and "
                      "are done. Writing it while five are NOT-IMPLEMENTED "
                      "would make 'READY' a label rather than a fact.")
        if args.supersedes:
            # THE PREDECESSOR IS READ BY PATH AND KEPT. The withdrawn version
            # wrote to the SAME path, so the freeze it claimed to record was
            # deleted -- a flag that announces it is preserving history and
            # then erases it. --output must name a different file, and the old
            # one has to still be a valid freeze, not merely a file that
            # parses: `{"stage": "code"}` was enough for the byte comparison
            # the first attempt did.
            old = Path(args.supersedes)
            if not args.supersede_reason:
                raise SystemExit(
                    "--supersedes requires --supersede-reason. This chain is "
                    "the dated record that a gate cannot be (14.0b-2 (s)); "
                    "'why was this re-frozen' is what section 14 wants, and "
                    "nobody reconstructs it later.")
            if not old.is_file():
                raise SystemExit(f"--supersedes {old}: no such file")
            try:
                oldrec = json.loads(old.read_text(encoding="utf-8"))
            except ValueError as e:
                raise SystemExit(f"--supersedes {old}: not JSON -- {e}")
            if old.resolve() == p.resolve():
                raise SystemExit(
                    f"--output is the file being superseded ({old}). 14.0a "
                    "keeps the old freeze and requires it to stay verifiable, "
                    "so writing over it destroys exactly what --supersedes "
                    "claims to record. Choose a new --output.")
            oldbad = validate_freeze(oldrec)
            if oldbad:
                raise SystemExit(
                    f"--supersedes {old} is not a valid freeze:\n  "
                    + "\n  ".join(oldbad[:6])
                    + "\nSuperseding something that pinned nothing records "
                      "nothing. Fix or discard it instead.")
            if STAGES.index(args.stage) < STAGES.index(oldrec["stage"]):
                raise SystemExit(
                    f"refusing to supersede a {oldrec['stage']!r}-stage "
                    f"freeze with a {args.stage!r}-stage one. Superseding "
                    "DOWNWARD is a route around the final code freeze; if the "
                    "code stage was wrong, say so in section 14 and write a "
                    "new code stage.")
            cur["supersedes"] = {
                "path": str(old), "sha256": sha256_file(old),
                "stage": oldrec["stage"], "reason": args.supersede_reason}
            print(f"  [chain] supersedes {old} "
                  f"({oldrec['stage']}, {sha256_file(old)[:12]}) -- "
                  f"{args.supersede_reason}")

        if p.exists():
            prev = json.loads(p.read_text(encoding="utf-8"))
            if not (args.stage == "code" and prev.get("stage") == "spec"):
                raise SystemExit(
                    f"{p} already exists at stage {prev.get('stage')!r}. "
                    "Overwriting a freeze is how a freeze stops meaning "
                    "anything -- move it aside deliberately, and if the "
                    "viability gate has already run, record the change in "
                    "section 14 as having happened AFTER the result. (The one "
                    "permitted upgrade is spec -> code at the final freeze.)")
            drift = compare(prev, cur)
            if drift:
                raise SystemExit(
                    "refusing to upgrade spec -> code: the spec block has "
                    f"drifted since it was frozen ({len(drift)} item(s)):\n  "
                    + "\n  ".join(drift[:8])
                    + "\nThe upgrade must preserve what was frozen before the "
                      "viability result; record the change in section 14 "
                      "instead.")
            print(f"  [upgrade] stage spec -> code, spec block unchanged")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cur, indent=2, ensure_ascii=False),
                     encoding="utf-8")
        print("=" * 78)
        print(f"BASELINE SPEC FROZEN -- stage {args.stage}")
        print("=" * 78)
        for name, s in sorted(cur["baselines"].items()):
            print(f"  {name:<20} {s['n_configs']:>2} config(s)  "
                  f"{s['status']:<22} hook {s['hook_boundary_sha256'][:12]}")
        print(f"\n  excluded: {', '.join(sorted(cur['excluded']))} "
              f"(recorded BEFORE any result, not dropped after one)")
        for k, v in sorted(cur["upstream_repos"].items()):
            print(f"  upstream {k:<22} "
                  + (f"{v['head'][:12]}" + ("" if v.get("clean") else " DIRTY")
                     if v.get("available") else f"[{v['why']}]"))
        print(f"\n  [output] {p}  (sha256 {sha256_file(p)[:16]}...)")
        if args.stage == "spec":
            print("  The viability gate may now run. Any later change to a "
                  "config, hook boundary or spec document shows up in "
                  "--verify.")
            print("  The repository COMMIT is deliberately not recorded "
                  "yet: five of the six runners do not exist, and whether to "
                  "write them is what the gate decides. Re-run with --stage "
                  "code at the final freeze.")
        else:
            print(f"  repo commit {str(cur['repo'].get('head'))[:12]} "
                  f"(clean={cur['repo'].get('clean')}) now recorded as well. "
                  "This is the READY freeze. Source files are NOT hashed -- "
                  "the commit covers them, and everything a file list would "
                  "have missed (14.0b-2 (t)).")
        return 0

    if not p.exists():
        raise SystemExit(
            f"{p}: no freeze recorded. Section 13.6.1 requires the six "
            "baselines to be frozen BEFORE carrier discovery and the viability "
            "gate. Run --write first.")
    rec = json.loads(p.read_text(encoding="utf-8"))
    # Verify against the stage that was RECORDED, not the one requested: a
    # spec-stage freeze must not be reported as failing merely because the
    # caller forgot --stage.
    cur = build_freeze(rec.get("stage", "spec"))
    print("=" * 78)
    print(f"BASELINE SPEC VERIFY -- {p.name} (stage {rec.get('stage','spec')})")
    print("=" * 78)
    invalid = validate_freeze(rec)
    if invalid:
        for b in invalid:
            print(f"  [INVALID] {b}")
        print(f"\nVERDICT: FAIL -- this freeze is not VALID, independently of "
              "whether anything drifted. A file written by the earlier "
              "fail-open tool, or one authored by hand, passes a drift check "
              "while pinning nothing.")
        return 3
    drift = compare(rec, cur)
    if not drift:
        n = len(rec["documents"]) + len(rec.get("code") or {})
        print(f"  [PASS] all {n} hashed items and {len(rec['baselines'])} "
              "baseline specs unchanged")
        if "code" not in rec:
            print("  [note] stage 'spec': runner files are not hashed yet, by "
                  "design (see --stage code)")
        print("\nVERDICT: PASS -- the opponents are as frozen.")
        return 0
    for d in drift:
        print(f"  [DRIFT] {d}")
    print(f"\nVERDICT: FAIL -- {len(drift)} item(s) drifted. If the viability "
          "gate has already run, this change happened AFTER the result and "
          "must be recorded in section 14 as such.")
    return 3


# ==========================================================================
# fixtures
# ==========================================================================
def _mock_heads(path):
    """A clean checkout at each baseline's registered commit.

    Unit fixtures must not depend on this machine's git ownership. The earlier
    version built its reference freeze from the REAL checkouts, so it passed
    here and failed on a reviewer's machine where all seven were unreadable --
    which made "all fixtures green" a claim about my laptop. The real
    checkouts are the job of `--integration-test`.
    """
    # pinned_repos(), not BASELINES: the excluded seventh repository is in
    # UPSTREAM and pinned too, and a mock that did not know that would make
    # the wrong-commit arms fail on a correct implementation.
    by_dir = {d: c for d, (_n, c) in pinned_repos().items()}
    return {"available": True, "clean": True,
            "head": by_dir.get(path.name) or ("e" * 40)}


def install_fixture_mocks(artifacts=None):
    """Make `build_freeze` deterministic for OTHER modules' fixtures.

    Anything that has to construct a lock-opening freeze needs the same two
    things this file's own self_test needs: `git_head` mocked, because a real
    checkout may be missing, dirty, or refused by git's ownership check; and
    `artifact_hashes` mocked, because the result NPZs live only on the server.

    It lives here rather than in each fixture because three of them now need
    it and the alternative is three copies that drift. It exists at all
    because the alternative in test_probe_queries was a SKIP that did not
    affect the exit code: on a machine where git refused the upstream repos,
    every lock-opening arm stood down and the suite still reported green.

    The registry statuses are forced READY for the same reason. Five entries
    still say NOT-IMPLEMENTED, so `code_stage_blockers` correctly refuses to
    write a code-stage freeze today -- and a fixture about the LOCK cannot
    wait for the runners to exist. What it must not do is hide the check: the
    fixtures keep an arm that a freeze recording a NOT-READY baseline is
    refused, so the condition is armed even while it is mocked here.

    NOT importable into production paths -- it disables the checks it mocks.
    """
    globals()["git_head"] = _mock_heads
    globals()["artifact_hashes"] = lambda: (
        dict(artifacts or {"results/fixture.npz": "ab" * 32}), [])
    for spec in registry.BASELINES.values():
        if spec["status"] not in READY_STATUSES:
            spec["status"] = "READY"
    # NO code_stage_blockers stub. Nothing on a fixture's path calls it any
    # more -- the test lock stopped re-executing it, since section 2.1 asks
    # about gamma*, carriers and the analysis code, not about the opponents
    # being implemented. A mock that hides nothing is a mock that will later
    # be mistaken for one that does.



def registry_seeds():
    """The prefix seeds, from the one place they are written."""
    from tools.prereg_config import REGISTERED_SEEDS
    return REGISTERED_SEEDS


def self_test() -> int:
    real_head = globals()["git_head"]
    globals()["git_head"] = _mock_heads
    try:
        return _self_test_body()
    finally:
        globals()["git_head"] = real_head


def _self_test_body() -> int:  # noqa: C901
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {detail}" if detail else ""))
        ok = ok and bool(cond)

    def red(name, fn, want=Exception, has=None):
        """`has` names WHICH refusal is expected.

        Without it an arm passes on any refusal, and several of these guards
        overlap -- disabling the same-path check let "already exists at
        stage" fire instead, so the mutation was invisible.
        """
        try:
            fn()
        except want as e:
            if has is not None and has not in str(e):
                check(name, False,
                      f"refused, but for the wrong reason: {str(e)[:70]}")
                return
            print(f"  [PASS] {name}  correctly refused: {str(e)[:70]}...")
            return
        except Exception as e:
            check(name, False, f"raised {type(e).__name__}, wanted {want}")
            return
        check(name, False, "SILENTLY SUCCEEDED -- the gate does not bite")

    print("the registry obeys its own cap")
    check("no baseline exceeds five configs except the two registered "
          "exceptions", registry.check_caps() == [],
          f"{len(registry.BASELINES)} baselines, "
          f"{sum(len(s['configs']) for s in registry.BASELINES.values())} "
          "configs in total")
    # EACH BASELINE FREEZES ON ITS OWN. code_stage_blockers used to demand
    # all six runners exist before anybody could freeze, welding "the
    # opponent set is decided" (13.6.1, before the viability gate) to "the
    # opponents are built" (only when H6 runs), and making five unwritten
    # runners a precondition for a code freeze about Method A's own code.
    print("\neach baseline freezes on its own; only the data is shared")
    _impl = implemented_baselines()
    _built_now = {"inductive_bc", "zerotuning_level1"}
    check("the runner's COMMIT carries its status, not a hand-written field",
          all(_impl[n] is True for n in _built_now)
          and all(_impl[n] is None for n in _impl if n not in _built_now),
          f"{_impl}; None means no runner_commit recorded, which is a "
          "different state from 'recorded, and the file is not there'")
    # INDEPENDENCE IS "NO OTHER BASELINE'S NAME APPEARS", NOT "NO BLOCKERS".
    # This arm used to assert `== []`, and passed because code_stage_blockers
    # filtered its sub-checks with `any(name in message)` -- so every fault
    # about a SHARED file, having no baseline name to match, was discarded.
    # inductive_bc looked freezable with no validation selection, no
    # verification report and no adaptation diffs on disk. An arm whose green
    # comes from a dropped check is worse than no arm.
    _others = [n for n in registry.BASELINES if n != "inductive_bc"]
    _bc = code_stage_blockers(only=["inductive_bc"])
    check("an implemented baseline's blockers name NO other baseline",
          not [b for b in _bc if any(o in b for o in _others)],
          "; ".join(_bc)[:90])
    check("...and an unfrozen one says so about ITSELF only",
          not [b for b in code_stage_blockers(only=["fv_on_icl"])
               if any(o in b for o in registry.BASELINES if o != "fv_on_icl")],
          "a blocker naming another baseline would be the coupling again")
    # THE ARM THAT WOULD HAVE CAUGHT IT. A shared obligation has no name in
    # it; scoping must narrow WHOSE obligations are asked about, never drop
    # the ones that belong to everybody.
    check("a shared obligation SURVIVES scoping to one baseline",
          any("baseline_validation_selection.json" in b for b in _bc)
          and any("baseline_verification_report.md" in b for b in _bc),
          f"got {_bc}; section 5(4) and 6.1 are owed by each baseline on its "
          "own, so a per-baseline freeze must still ask for them")
    # ...and the one that is NOT shared: Method A's carriers are Method A's.
    check("Method A's carriers are not an OPPONENT's obligation",
          not any("method_a_carriers" in b for b in _bc)
          and any("method_a_carriers" in b for b in code_stage_blockers()),
          "requiring them per baseline is the coupling the ruling removed; "
          "dropping them from the full freeze would be the opposite error")
    _saved = registry.BASELINES["inductive_bc"]["runner_commit"]
    try:
        registry.BASELINES["inductive_bc"]["runner_commit"] = "0" * 40
        _b = code_stage_blockers(only=["inductive_bc"])
        check("a commit where the runner does not exist IS NOT-IMPLEMENTED",
              any("does not exist at" in x or "could not be read" in x
                  for x in _b),
              "; ".join(_b)[:70])
    finally:
        registry.BASELINES["inductive_bc"]["runner_commit"] = _saved
    check("...and it is clean again afterwards",
          code_stage_blockers(only=["inductive_bc"]) == _bc)

    # SECTION 2.4(4) SAYS WHICH BASELINES FIT SOMETHING PER PREFIX SEED --
    # FV and TSLA their vectors, UniBias its components -- and until now it
    # said so only in prose. A run producing ONE shared FV vector for all
    # three prefixes passed every machine check, and it is a different
    # experiment: validation is drawn per seed (14.0a) precisely so each
    # prefix gets its own adaptation, and the seed-to-seed spread carries
    # that adaptation's sampling noise.
    check("every baseline DECLARES what it fits per prefix seed",
          all("per_seed_artifacts" in sp
              for sp in registry.BASELINES.values()),
          "an empty tuple is an answer; an absent field is a question nobody "
          "answered, and check_caps now refuses it")
    check("...and the ones 2.4(4) names are not empty",
          all(registry.per_seed_artifacts(n)
              for n in ("fv_on_icl", "tsla_tl_on_icl", "unibias_code")),
          f"fv={registry.per_seed_artifacts('fv_on_icl')} "
          f"tsla={registry.per_seed_artifacts('tsla_tl_on_icl')} "
          f"unibias={registry.per_seed_artifacts('unibias_code')}")
    check("...while ZeroTuning declares nothing, because it fits nothing",
          registry.per_seed_artifacts("zerotuning_level1") == (),
          "r is a global hyperparameter, chosen by the same three-seed-mean "
          "rule that freezes gamma*")
    _seeds = list(registry_seeds())
    check("ONE artifact for three prefixes is a fault",
          any("different experiment" in b for b in
              registry.artifact_seed_faults(
                  "fv_on_icl", {"fv_vector": [_seeds[0]]}, _seeds)),
          "this is the run that used to pass everything")
    check("...a missing one is a fault too",
          any("no 'fv_vector'" in b for b in
              registry.artifact_seed_faults("fv_on_icl", {}, _seeds)))
    check("...and three of them is clean",
          registry.artifact_seed_faults(
              "fv_on_icl", {"fv_vector": list(_seeds)}, _seeds) == [])
    check("...and a baseline with nothing per seed is clean either way",
          registry.artifact_seed_faults("zerotuning_level1", {}, _seeds) == [],
          "the check must not fire on a correct run, which is how the two "
          "gates this work already withdrew went wrong")

    check("the two exceptions are the ones section 2.4(5) names",
          registry.config_count("deepthinking_dev") == 15
          and registry.config_count("unibias_code") == 1,
          "DT is one nested 15-round trajectory; UniBias is one "
          "discovery-only flow")
    check("every baseline declares a hook boundary and a placebo kind",
          all(s.get("hook_boundary") and s.get("placebo_kind")
              for s in registry.BASELINES.values()))
    check("BC is the only baseline with no possible placebo",
          [n for n, s in registry.BASELINES.items()
           if s["placebo_kind"] == "none-possible"] == ["inductive_bc"],
          "b=0 gives softmax(p), not p")
    check("Yu & Ananiadou is recorded as excluded, with the SETTING as the "
          "primary reason", "yu_ananiadou" in registry.EXCLUDED
          and "setting is the primary one" in
          registry.EXCLUDED["yu_ananiadou"]["why"])

    print("\nthe section slice is bounded from BOTH sides")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "doc.md"
        f.write_text("### 13.2\n" + ("x" * 600) + "\n### 13.3\ntail\n",
                     encoding="utf-8")
        s = section_slice(f, "### 13.2", "### 13.3")
        check("a well-formed slice is returned", 600 < len(s) < 700
              and "tail" not in s)
        f.write_text("### 13.2\nshort\n### 13.3\n", encoding="utf-8")
        red("a slice too short to be the block is refused, not hashed",
            lambda: section_slice(f, "### 13.2", "### 13.3"), ValueError)
        f.write_text("### 13.3\nx\n### 13.2\n" + "y" * 600, encoding="utf-8")
        red("an end heading that PRECEDES the start is refused",
            lambda: section_slice(f, "### 13.2", "### 13.3"), ValueError)
        f.write_text("### 13.2\n" + "z" * 600, encoding="utf-8")
        red("a missing end heading is refused rather than running to EOF",
            lambda: section_slice(f, "### 13.2", "### 13.3"), ValueError)
        red("a missing start heading is refused",
            lambda: section_slice(f, "### 99.9", "### 13.3"), ValueError)

    print("\ndrift detection")
    a = build_freeze()
    check("a freeze compares equal to itself", compare(a, a) == [])
    b = json.loads(json.dumps(a))
    b["baselines"]["fv_on_icl"]["configs"].append({"alpha": 4.0})
    d = compare(a, b)
    check("widening a grid is caught", any("fv_on_icl/configs" in x for x in d),
          d[0] if d else "")
    b = json.loads(json.dumps(a))
    b["baselines"]["tsla_tl_on_icl"]["hook_boundary_sha256"] = "0" * 64
    d = compare(a, b)
    check("moving a hook boundary is caught",
          any("hook_boundary_sha256" in x for x in d), d[0] if d else "")

    # The WHOLE decision, not a maintained list of fields. `status`,
    # `placebo_kind`, `arm_name`, `excluded` and `config_cap` were all free to
    # change without drifting under the old field whitelist.
    for label, mangle in (
            ("status", lambda r: r["baselines"]["fv_on_icl"].update(
                {"status": "READY"})),
            ("placebo_kind", lambda r: r["baselines"]["fv_on_icl"].update(
                {"placebo_kind": "none-possible"})),
            ("arm_name", lambda r: r["baselines"]["fv_on_icl"].update(
                {"arm_name": "Function Vectors (improved)"})),
            ("the excluded set", lambda r: r["excluded"].pop("yu_ananiadou")),
            ("the config cap", lambda r: r["config_cap"].update(
                {"max_total_configs": 9}))):
        m = json.loads(json.dumps(a))
        mangle(m)
        m["decision_sha256"] = sha256_text(json.dumps(
            {"baselines": m["baselines"], "excluded": m["excluded"],
             "config_cap": m["config_cap"]}, sort_keys=True,
            ensure_ascii=False))
        check(f"changing {label} is caught", compare(a, m) != [],
              (compare(a, m) or [""])[0][:66])
    check("the decision hash covers every field, including ones added later",
          "decision_sha256" in a and len(a["decision_sha256"]) == 64,
          "a whitelist needs maintenance; a canonical hash does not")
    stale_hash = json.loads(json.dumps(a))
    stale_hash["decision_sha256"] = "0" * 64
    # NOT via compare: since (q) it recomputes the hash over the recorded
    # opponents rather than reading the stored field, so a tampered field is
    # invisible there. validate_freeze re-derives it from the record's own
    # content, which catches editing a config and its hash together.
    check("...and a tampered hash alone is caught by validate_freeze",
          any("decision_sha256" in x for x in validate_freeze(stale_hash)),
          (validate_freeze(stale_hash) or [""])[0][:66])
    b = json.loads(json.dumps(a))
    del b["baselines"]["unibias_code"]
    check("removing a baseline outright is caught",
          any("REMOVED" in x for x in compare(a, b)))
    print("\nthe two stages, and why one stage was incoherent")
    check("the spec stage does NOT hash runner files", "code" not in a,
          "five of the six runners do not exist yet, and whether to write them "
          "is what the viability gate decides")
    after = json.loads(json.dumps(a))
    after["baselines"]["zerotuning_level1"]["status"] = "IMPLEMENTED"
    check("...so writing a new runner later is not drift at the spec stage",
          not [x for x in compare(a, build_freeze("spec")) if "code/" in x])
    codef = build_freeze("code")
    # NO per-file hashes since 14.0b-2 (t): a hand-written list missed a
    # module twice and the closure that replaced it pinned 19 files, 12 of
    # them shared, so editing a shared utility for another line of work
    # re-locked the test splits. The commit covers all of it, plus tasks/ and
    # script/, which no list would have named.
    check("the code stage pins the code by COMMIT, and is a superset",
          "code" not in codef
          and len(str(codef.get("repo", {}).get("head", ""))) == 40
          and codef["baselines"] == a["baselines"],
          f"repo {str(codef.get('repo', {}).get('head'))[:12]}")
    b = json.loads(json.dumps(codef))
    b["repo"] = {"available": True, "clean": True, "head": "1" * 40}
    check("a code stage taken at a DIFFERENT commit is caught",
          any("repo" in x for x in compare(codef, b)),
          "the freeze says which code it froze; a different commit is a "
          "different answer to that question")
    red("an unknown stage is refused", lambda: build_freeze("later"),
        ValueError)
    b = json.loads(json.dumps(a))
    b["documents"]["baseline_under_review.md"] = "2" * 64
    check("editing the spec document is caught",
          any("baseline_under_review.md" in x for x in compare(a, b)))
    b = json.loads(json.dumps(a))
    for k in b["upstream_repos"]:
        b["upstream_repos"][k] = {"available": True, "head": "f" * 40,
                                  "clean": True}
    check("an upstream repo moving is caught",
          len([x for x in compare(a, b) if x.startswith("upstream/")]) >= 1)
    moved_root = json.loads(json.dumps(a))
    moved_root["upstream_root"] = "/somewhere/else"
    check("re-pointing --upstream-root is drift, not a silent substitution",
          any("upstream_root" in x for x in compare(a, moved_root)),
          "the root is a default, not a fact about every machine, so it is "
          "recorded rather than assumed")
    check("the freeze records the root it actually looked in",
          a.get("upstream_root") == str(upstream_root()))

    print("\nthe upstream side is fail-CLOSED (it was fail-open)")
    # ⚠ These arms use MOCKED repository state. An earlier version asserted
    # that the real checkouts on this machine were readable, which made the
    # unit suite depend on local git ownership -- it passed here and failed on
    # a reviewer's machine, so "all fixtures green" was not a reproducible
    # claim. The real checkouts are checked by --integration-test instead.
    blind = json.loads(json.dumps(a))
    for k in blind["upstream_repos"]:
        blind["upstream_repos"][k] = {"available": False,
                                      "why": "detected dubious ownership"}
    check("an unreadable repository is reported as a blocker, not recorded and "
          "waved through",
          len(unreadable_repos(blind["upstream_repos"]))
          == len(blind["upstream_repos"]))
    check("...and it registers as DRIFT against a freeze that could read it",
          any("/available:" in x for x in compare(a, blind)),
          "comparing only HEAD made an unreadable repo look unchanged")
    dirtied = json.loads(json.dumps(a))
    k0 = sorted(dirtied["upstream_repos"])[0]
    dirtied["upstream_repos"][k0] = {**dirtied["upstream_repos"][k0],
                                     "clean": False, "diff_sha256": "d" * 64}
    check("editing upstream code under an UNCHANGED head is drift",
          any(x.startswith(f"upstream/{k0}/clean") for x in compare(a, dirtied)),
          "the diff is hashed, so same-commit-different-source is visible")
    check("the registered UniBias commit is the one checked out",
          wrong_commit(a["upstream_repos"]) == [],
          registry.BASELINES["unibias_code"]["registered_commit"][:12])
    moved = json.loads(json.dumps(a))
    moved["upstream_repos"]["UniBias"]["head"] = "0" * 40
    check("a UniBias HEAD other than section 13.2.5's is a blocker",
          len(wrong_commit(moved["upstream_repos"])) == 1,
          wrong_commit(moved["upstream_repos"])[0][:70])
    check("a DIRTY checkout is a blocker, not a [note]",
          len(dirty_repos(dirtied["upstream_repos"])) == 1,
          "a hashed diff makes a LATER edit visible, but it still lets the "
          "freeze be taken over source that is not the upstream source")
    check("...and a clean tree has none", dirty_repos(a["upstream_repos"]) == [])

    # END-TO-END: the reviewer's exact scenario. Seven readable repositories at
    # the right commits, all locally modified -- previously written with a
    # [note] per repo and VERDICT PASS.
    # ⚠ Patch through globals(), NOT `import tools.baselines.freeze_baseline_spec
    # as SELF`. Run as a script this file IS `__main__`, and importing it by
    # its package path creates a SECOND module object -- the patch would land
    # on the copy and both arms would report SILENTLY SUCCEEDED against gates
    # that work. (It did.)
    want = registry.BASELINES["unibias_code"]["registered_commit"]
    real_head = globals()["git_head"]
    for label, fake in (
            ("every repo is dirty at the right commit",
             lambda p: {"available": True, "clean": False,
                        "head": want if p.name == "UniBias" else "a" * 40,
                        "dirty_files": ["x.py"], "diff_sha256": "d" * 64}),
            ("every repo is unreadable",
             lambda p: {"available": False, "why": "dubious ownership"})):
        globals()["git_head"] = fake
        try:
            with tempfile.TemporaryDirectory() as td2:
                red(f"--write is refused when {label}",
                    lambda o=Path(td2) / "f.json": main(
                        ["--write", "--output", str(o)]), SystemExit)
        finally:
            globals()["git_head"] = real_head
    check("the patch was actually removed again",
          globals()["git_head"] is real_head)

    print("\nthe code stage names WHICH baselines are not ready")
    blockers = code_stage_blockers()
    check("today's checkout is NOT a whole-registry READY freeze",
          blockers != [],
          f"{len(blockers)} blocker(s), first: {blockers[0][:60]}")
    # PER BASELINE, WHERE THE FAULT IS PER BASELINE. The old arm demanded a
    # name in EVERY blocker, which is the same wrong assumption that made
    # code_stage_blockers filter its sub-checks by substring: a missing
    # verification report belongs to no single baseline and has no name to
    # carry. What must hold is the converse -- a fault ABOUT a baseline names
    # it, and a ready baseline contributes none.
    check("...and no blocker is about an unnamed baseline",
          all(any(n in b for n in registry.BASELINES)
              or b.startswith("results/") for b in blockers),
          f"{[b for b in blockers if not any(n in b for n in registry.BASELINES)]}"
          " -- a per-baseline fault must name it; a shared-artifact fault "
          "names the artifact")
    check("...while the ready one contributes none",
          not any("inductive_bc" in b for b in blockers),
          "it is frozen at its own runner_commit and does not wait for the "
          "other five")
    # The adaptation diffs and the verification report are still required --
    # OF A BASELINE THAT CLAIMS TO BE BUILT. Asking them of a runner that
    # does not exist is asking for evidence about something that has not
    # happened.
    _fake = dict(registry.BASELINES["fv_on_icl"])
    check("an unbuilt baseline is not asked for its adaptation diff",
          not any("fv_on_icl" in b and "adapt_diffs" in b for b in blockers),
          "the diff records what was changed to run it; there is nothing to "
          "record yet")

    print("\nthe freeze refuses to overwrite itself")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "freeze.json"
        check("the first --write succeeds",
              main(["--write", "--output", str(out)]) == 0)
        red("a second --write is refused",
            lambda: main(["--write", "--output", str(out)]), SystemExit)
        check("--verify passes immediately after --write",
              main(["--verify", "--output", str(out)]) == 0)
        # spec -> code is the one permitted overwrite, but TODAY it is refused
        # for a second reason: the runners do not exist. Both gates matter, so
        # both get an arm.
        red("the spec -> code upgrade is refused while five runners are "
            "missing", lambda: main(
                ["--write", "--stage", "code", "--output", str(out)]),
            SystemExit)
        rec = json.loads(out.read_text(encoding="utf-8"))
        check("...and the spec freeze on disk was left untouched by the "
              "refusal", "repo" not in rec,
              "a spec stage records no commit; that is what makes a relabelled "
              "one detectable")
        rec["baselines"]["zerotuning_level1"]["configs"] = [{"r": 1.0}]
        out.write_text(json.dumps(rec), encoding="utf-8")
        check("--verify returns 3 (not 0, not 1) when a config drifted",
              main(["--verify", "--output", str(out)]) == 3)

    # ---- the supersession chain (A6) -------------------------------------
    # Every arm below is a way the withdrawn first attempt failed.
    real_blockers = globals()["code_stage_blockers"]
    globals()["code_stage_blockers"] = lambda: []
    # AND the recorded statuses. A code freeze written with the blockers
    # stubbed still RECORDS five NOT-IMPLEMENTED baselines, so validate_freeze
    # refuses it -- correctly -- and the downgrade arm below was passing on
    # that refusal instead of on the one it names.
    _saved = {n: sp["status"] for n, sp in registry.BASELINES.items()}
    for sp in registry.BASELINES.values():
        sp["status"] = "READY"
    # artifact_hashes reads results/, which lives only on the server, so a
    # code freeze written here records none and validate_freeze refuses it --
    # the second guard the downgrade arm was passing on.
    _real_art = globals()["artifact_hashes"]
    globals()["artifact_hashes"] = lambda: ({"results/fixture.npz": "ab" * 32},
                                            [])
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            first = td / "freeze_a.json"
            second = td / "freeze_b.json"
            main(["--write", "--output", str(first)])

            red("--supersedes without a reason is refused",
                lambda: main(["--write", "--output", str(second),
                              "--supersedes", str(first)]), SystemExit,
                has="--supersede-reason")
            # THE DEFECT THAT CAUSED THE WITHDRAWAL: it wrote to the same
            # path, so the freeze it claimed to record was deleted.
            red("superseding INTO the same path is refused",
                lambda: main(["--write", "--output", str(first),
                              "--supersedes", str(first),
                              "--supersede-reason", "x"]), SystemExit,
                has="is the file being superseded")
            check("...and the predecessor is still on disk and still valid",
                  first.is_file() and validate_freeze(json.loads(
                      first.read_text(encoding="utf-8"))) == [],
                  "14.0a keeps the old file and requires it to stay "
                  "verifiable; the first attempt erased it")
            junk = td / "junk.json"
            junk.write_text(json.dumps({"stage": "code"}), encoding="utf-8")
            red("superseding something that is not a valid freeze is refused",
                lambda: main(["--write", "--output", str(second),
                              "--supersedes", str(junk),
                              "--supersede-reason", "x"]), SystemExit,
                has="is not a valid freeze")

            check("spec -> spec supersession succeeds",
                  main(["--write", "--output", str(second),
                        "--supersedes", str(first),
                        "--supersede-reason", "the prereg 13.6 text moved"])
                  == 0)
            chain = json.loads(second.read_text(encoding="utf-8"))
            check("...and records the predecessor by PATH and HASH, with the "
                  "reason",
                  chain["supersedes"]["path"] == str(first)
                  and chain["supersedes"]["sha256"] == sha256_file(first)
                  and chain["supersedes"]["stage"] == "spec"
                  and "13.6" in chain["supersedes"]["reason"],
                  "a hash alone cannot find the file; a path alone cannot "
                  "show it is unchanged")
            check("...and both freezes verify",
                  main(["--verify", "--output", str(first)]) == 0
                  and main(["--verify", "--output", str(second)]) == 0,
                  "the chain does not cost the predecessor its validity")

            # DELETABLE LOG vs EVIDENCE. validate_freeze ignored these fields
            # in the first attempt, so dropping them passed --verify.
            for field in ("path", "sha256", "stage", "reason"):
                broken = json.loads(json.dumps(chain))
                del broken["supersedes"][field]
                check(f"dropping supersedes/{field} is a blocker",
                      any("supersedes" in b for b in validate_freeze(broken)))
            moved = json.loads(json.dumps(chain))
            moved["supersedes"]["path"] = str(td / "nowhere.json")
            check("a predecessor that is no longer on disk is a blocker",
                  any("not on disk" in b for b in validate_freeze(moved)),
                  "a chain whose predecessor is gone records nothing")
            edited = json.loads(json.dumps(chain))
            edited["supersedes"]["sha256"] = "0" * 64
            check("a predecessor edited since it was superseded is a blocker",
                  any("hashes to" in b for b in validate_freeze(edited)))

            # THE DOWNGRADE ROUTE. The first attempt compared bytes only, so
            # {"stage": "code"} could be superseded down to spec, returning 0.
            code_f = td / "freeze_code.json"
            check("code -> code supersession succeeds (14.0b-2 (s): "
                  "re-freezing is a normal act)",
                  main(["--write", "--stage", "code", "--output",
                        str(code_f), "--supersedes", str(second),
                        "--supersede-reason", "icl_common gained a helper"])
                  == 0)
            down = td / "freeze_down.json"
            red("superseding a code stage DOWN to spec is refused",
                lambda: main(["--write", "--output", str(down),
                              "--supersedes", str(code_f),
                              "--supersede-reason", "x"]), SystemExit,
                has="DOWNWARD")
            # A record whose OWN stage is spec while its predecessor's was
            # code -- the first version of this arm set only the outer stage
            # and so was not a downgrade at all, and passed for that reason.
            downrec = json.loads(code_f.read_text(encoding="utf-8"))
            downrec["stage"] = "spec"
            downrec["supersedes"] = {**downrec["supersedes"], "stage": "code"}
            check("...and a hand-authored downgrade is a blocker too",
                  any("DOWNWARD" in b for b in validate_freeze(downrec)),
                  "the CLI is not the only way to write one of these")
    finally:
        globals()["code_stage_blockers"] = real_blockers
        globals()["artifact_hashes"] = _real_art
        for n, st in _saved.items():
            registry.BASELINES[n]["status"] = st

    # The upgrade path itself, exercised with the code-stage blockers stubbed
    # out -- otherwise it cannot be tested until all six runners exist, and an
    # untested upgrade path is how the final freeze goes wrong.
    real_blockers = globals()["code_stage_blockers"]
    globals()["code_stage_blockers"] = lambda: []
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "freeze.json"
            main(["--write", "--output", str(out)])
            check("with the runners in place, spec -> code succeeds",
                  main(["--write", "--stage", "code", "--output",
                        str(out)]) == 0)
            _up = json.loads(out.read_text(encoding="utf-8"))
            # The COMMIT, not the artifacts: artifact_hashes reads results/,
            # which lives only on the server, so asserting it here would make
            # the arm pass or fail on where it ran.
            check("...and it really added the repository commit",
                  len(str(_up.get("repo", {}).get("head", ""))) == 40,
                  f"repo {str(_up.get('repo', {}).get('head'))[:12]}; source "
                  "files are not hashed at all since 14.0b-2 (t)")
            red("a code -> code overwrite is still refused",
                lambda: main(["--write", "--stage", "code", "--output",
                              str(out)]), SystemExit)
        with tempfile.TemporaryDirectory() as td:
            out2 = Path(td) / "freeze.json"
            main(["--write", "--output", str(out2)])
            rec = json.loads(out2.read_text(encoding="utf-8"))
            rec["baselines"]["fv_on_icl"]["configs"].append({"alpha": 4.0})
            out2.write_text(json.dumps(rec), encoding="utf-8")
            red("the upgrade is refused when the SPEC block drifted first",
                lambda: main(["--write", "--stage", "code", "--output",
                              str(out2)]), SystemExit)
    finally:
        globals()["code_stage_blockers"] = real_blockers
    red("--verify with no freeze on disk is refused",
        lambda: main(["--verify", "--output",
                      str(Path(tempfile.gettempdir()) / "nope_freeze.json")]),
        SystemExit)
    red("passing neither --write nor --verify is refused",
        lambda: main([]), SystemExit)

    print("\na freeze must be VALID, not merely undrifted")
    check("a freeze this tool wrote validates", validate_freeze(a) == [])
    for label, mangle in (
            ("an upstream repo recorded as unreadable",
             lambda r: r["upstream_repos"].update(
                 {"UniBias": {"available": False, "why": "old tool"}})),
            ("an upstream repo recorded as dirty",
             lambda r: r["upstream_repos"]["UniBias"].update(
                 {"clean": False, "dirty_files": ["x"], "diff_sha256": "d"})),
            ("an upstream repo at the wrong commit",
             lambda r: r["upstream_repos"]["UniBias"].update(
                 {"head": "0" * 40})),
            ("a missing upstream repo",
             lambda r: r["upstream_repos"].pop("StaICC")),
            ("a baseline the registry does not have",
             lambda r: r["baselines"].update({"ghost": {}})),
            ("no stage field", lambda r: r.pop("stage"))):
        m = json.loads(json.dumps(a))
        mangle(m)
        check(f"...but one with {label} does NOT", validate_freeze(m) != [],
              (validate_freeze(m) or [""])[0][:64])
    check("a hand-authored object is refused outright",
          validate_freeze({"looks": "plausible"}) != [])
    check("a code-stage freeze with no artifact hashes is refused",
          validate_freeze({**a, "stage": "code"}) != [])

    print("\nthe code stage hashes ARTIFACTS, not just their existence")
    names = [r for r, _k, _s, _w in CODE_STAGE_ARTIFACTS]
    # FOUR, not five: the discovery split went with the artifact (14.0a).
    # Listing a file that cannot exist would make the code stage
    # unsatisfiable rather than strict.
    check("the carriers, validation selection, adapt diffs and report are "
          "all covered", len(names) == 4,
          ", ".join(Path(n).name for n in names))
    _, missing = artifact_hashes()
    check("today they are missing, so the code stage is blocked",
          len(missing) == 4)
    check("an EMPTY adapt-diffs directory counts as missing, not as present",
          any("holds no *.patch" in m for m in missing),
          "existence alone satisfied the old check")

    print("\nevery UPSTREAM repository pins a version, including the seventh")
    check("nothing in UPSTREAM is unpinned",
          missing_registered_commits() == [], f"{len(UPSTREAM)} repositories")
    check("the audited-but-excluded seventh is pinned too",
          "in-context-mechanism" in pinned_repos(),
          "it is in UPSTREAM and section 13.1 fixes it, so leaving it out "
          "would mean claiming to freeze seven while freezing six")
    # The 13.1 SLICE, not the whole document. Reading the whole file meant a
    # commit deleted from the audit table still passed as long as it survived
    # anywhere else -- section 14's correction log quotes several of them.
    audit = section_slice(PREREG, "### 13.1", "### 13.2")
    unlisted = [f"{n} ({c[:12]})" for n, c in pinned_repos().values()
                if c not in audit]
    check("every registered commit appears VERBATIM in the 13.1 AUDIT TABLE",
          not unlisted, "; ".join(unlisted) if unlisted
          else f"{len(pinned_repos())} commits cross-checked against the "
               f"{len(audit)}-char section 13.1 slice, not the whole document")
    whole = PREREG.read_text(encoding="utf-8")
    check("...and the slice is genuinely narrower than the document",
          len(audit) < len(whole) / 4,
          f"{len(audit)} of {len(whole)} chars -- a whole-file search would "
          "accept a commit that only survives in the section 14 log")

    print("\nthe code stage checks CONTENT, not only presence")
    need = sorted(k for k, v in registry.BASELINES.items()
                  if v.get("needs_adapt_diff"))
    check("five baselines declare that they modify upstream; BC does not",
          need == ["deepthinking_dev", "fv_on_icl", "tsla_tl_on_icl",
                   "unibias_code", "zerotuning_level1"],
          "BC is a formula reimplemented from the paper's code and touches no "
          "upstream file, so it is not forced to invent a diff")
    check("each of them needs its OWN patch, not just one somewhere in the "
          "directory", len(missing_adapt_diffs()) == len(need))
    vb = validation_selection_blockers()
    check("a missing validation selection blocks the code stage", vb != [],
          vb[0][:70])
    real_repo = globals()["REPO"]
    with tempfile.TemporaryDirectory() as td:
        import numpy as np
        res = Path(td) / "results"
        (res / "baselines").mkdir(parents=True)

        # ⚠ THE REGISTERED SHAPE. An earlier fixture used 3 seeds x 12 queries
        # x 4 candidates and passed, against a setting of 3 x 144 x 36 -- so
        # every identity check it was supposed to exercise was vacuous. A
        # fixture at the wrong scale cannot test a scale check.
        NQ, NCAND = N_VALIDATION, N_ELIGIBLE
        # [S, Q]: each seed draws its own 144 (section 14.0a). The rows must
        # DIFFER, or a validator that ignored the seed would pass every arm,
        # and must OVERLAP, so the shared-query case is exercised too. A pool
        # of NQ+16 with an 8-wide stride gives both.
        _pool = [f"{i:064x}" for i in range(NQ + 16)]
        qids = [[_pool[8 * si + j] for j in range(NQ)]
                for si in range(len(REGISTERED_DEMO_SEEDS))]
        cls = [[j % NCAND for j in range(NQ)]
               for _ in REGISTERED_DEMO_SEEDS]
        assert len({tuple(r) for r in qids}) == len(qids)
        assert set(qids[0]) & set(qids[1])
        qm = res / "prereg_method_A_query_manifest.json"
        qm.write_text(json.dumps({
            "eligible_classes": list(range(NCAND)),
            "validation_by_seed": {
                str(sd): [{"query_id": q, "class_idx": c}
                          for q, c in zip(qids[si], cls[si])]
                for si, sd in enumerate(REGISTERED_DEMO_SEEDS)}}),
            encoding="utf-8")
        # ⚠ A REAL label space. The two-field stub here used to pass because
        # the validator read the JSON directly; the runners themselves would
        # have refused it, so the selection check was more permissive than the
        # code that produces the data it checks.
        lsp = res / "label_space_llama31.json"
        lsp.write_text(json.dumps({
            "schema_version": 1,
            "provenance": {"model": REGISTERED_MODEL,
                           "tokenizer_class": "PreTrainedTokenizerFast",
                           "transformers_version": "4.52.3", "revision": None,
                           "is_fast": True, "vocab_size": 128256,
                           "vocab_sha256": "v", "special_tokens_sha256": "s",
                           "backend_tokenizer_sha256": "b"},
            "task": REGISTERED_TASK, "K": REGISTERED_K, "n_classes": NCAND,
            "eligible_classes": list(range(NCAND)),
            "abstract_labels": [f"L{i}" for i in range(NCAND)],
            "label_token_ids": [1000 + i for i in range(NCAND)],
            "candidate_token_ids": [1000 + i for i in range(NCAND)],
            "query_manifest_sha256": sha256_file(qm)}), encoding="utf-8")
        sel = res / "baseline_validation_selection.json"
        globals()["REPO"] = Path(td)

        def write_npz(name, configs, *, winner=0, skip_arm=None,
                      qids_over=None, gold_over=None,
                      seeds_over=None, cand_over=None,
                      tok_over=None, **over):
            """A REAL result NPZ at the registered shape, with its sidecar."""
            arms, rows = ["natural"], [np.zeros((3, NQ, NCAND))]
            for i, c in enumerate(configs):
                arm = config_arm_name(c)
                if arm == skip_arm:
                    continue
                z = np.zeros((3, NQ, NCAND))
                for si in range(3):
                    for qi, gc in enumerate(cls[si]):
                        z[si, qi, gc] = 2.0 if i == winner else 0.5
                arms.append(arm)
                rows.append(z)
            f = Path(td) / expected_result_npz(name)
            np.savez(f, candidate_logits=np.stack(rows),
                     arms=np.array(arms),
                     seeds=np.array([42, 43, 44] if seeds_over is None
                                    else seeds_over),
                     query_ids=np.array(qids if qids_over is None
                                       else qids_over),
                     # dtype PRESERVED when overridden: forcing int64
                     # made a fractional gold unconstructible, so the
                     # strict-type path had no reachable arm.
                     gold_class=(np.array(cls, dtype=np.int64)
                                 if gold_over is None
                                 else np.array(gold_over)),
                     # both [S, Q] now; np.array of the nested lists gives
                     # that shape directly
                     candidate_classes=(
                         np.arange(NCAND, dtype=np.int64)
                         if cand_over is None else np.array(cand_over)),
                     candidate_token_ids=(
                         np.array([1000 + i for i in range(NCAND)],
                                  dtype=np.int64)
                         if tok_over is None else np.array(tok_over)))
            # A bias-bearing baseline publishes its companion beside the
            # result and binds it by hash; the default fixture must therefore
            # produce a correctly bound pair, or every later arm would be
            # testing against an already-invalid baseline.
            extra = {}
            if name in BIAS_BEARING:
                bfp = f.parent / BIAS_BEARING[name]
                if not bfp.is_file():
                    bfp.write_text('{"bias_by_seed": {}}', encoding="utf-8")
                extra["bias_sha256"] = sha256_file(bfp)
            side = {**extra, "baseline": name, "mode": "validation",
                    "model": REGISTERED_MODEL, "task": REGISTERED_TASK,
                    "K": REGISTERED_K, "dtype": "torch.bfloat16",
                    "attn_implementation": "eager",
                    "demo_seeds": [42, 43, 44],
                    "npz_sha256": sha256_file(f),
                    "query_manifest_sha256": sha256_file(qm),
                    "label_space_sha256": sha256_file(lsp)}
            side.update(over)
            Path(str(f).replace(".npz", ".json")).write_text(
                json.dumps(side), encoding="utf-8")
            return f

        def doc(winner=0, skip_arm=None, side_over=None, **over):
            selected, evidence = {}, {}
            for k, v in registry.BASELINES.items():
                f = write_npz(k, v["configs"], winner=winner,
                              skip_arm=skip_arm, **(side_over or {}))
                scores, faults = recompute_per_config_nll(
                    f, v["configs"], name=k, manifest=qm, label_space=lsp)
                assert scores is not None or skip_arm or side_over, faults
                selected[k] = v["configs"][winner]
                evidence[k] = {
                    "result_npz_path": expected_result_npz(k),
                    "result_npz_sha256": sha256_file(f),
                    "per_config_nll": scores}
            d = {"schema_version": SELECTION_SCHEMA_VERSION,
                 "spec": SELECTION_SPEC, "demo_seeds": [42, 43, 44],
                 "metric": SELECTION_METRIC, "tie_rule": SELECTION_TIE_RULE,
                 "query_manifest_sha256": sha256_file(qm),
                 "label_space_sha256": sha256_file(lsp),
                 "selected": selected, "evidence": evidence}
            d.update(over)
            return d

        def put(d):
            sel.write_text(json.dumps(d), encoding="utf-8")

        try:
            put(doc())
            # TYPES, not values. The existing arms use schema_version=0
            # and seeds [42,43,45]; those are wrong VALUES and the value
            # comparison rejects them on its own, so reverting strict_int
            # left them green.
            for _lbl, _mut in (
                    ('schema_version "1", a string',
                     lambda d: d.__setitem__("schema_version", "1")),
                    ("schema_version 1.9, a fraction",
                     lambda d: d.__setitem__("schema_version", 1.9)),
                    ("demo_seeds [42.9, 43, 44]",
                     lambda d: d.__setitem__("demo_seeds", [42.9, 43, 44]))):
                _dd = doc()
                _mut(_dd)
                put(_dd)
                _bl = validation_selection_blockers()
                check(f"{_lbl} is reported as a blocker",
                      bool(_bl) and any("integer" in x or "real number" in x
                                        for x in _bl),
                      "; ".join(_bl)[:70] or "SILENTLY CLEAN")
            # SHAPE: every key present, but evidence is a list. This used to
            # raise AttributeError rather than return a blocker.
            # EVERY layer, not one. The previous version tested only
            # evidence=[], so deleting the `selected` check or the per-entry
            # check changed nothing while the report claimed all three.
            for _lbl, _mut, _want in (
                    ("evidence as a LIST",
                     lambda d: d.__setitem__("evidence", []),
                     "expected an object"),
                    ("selected as a LIST",
                     lambda d: d.__setitem__("selected", []),
                     "expected an object"),
                    ("one evidence ENTRY as a list",
                     lambda d: d["evidence"].__setitem__(
                         next(iter(d["evidence"])), []),
                     "expected an object"),
                    ("per_config_nll as an int, which has no len()",
                     lambda d: d["evidence"][next(iter(d["evidence"]))]
                     .__setitem__("per_config_nll", 1),
                     "expected a list"),
                    ("per_config_nll holding 10**1000, which float() "
                     "cannot take",
                     lambda d: d["evidence"][next(iter(d["evidence"]))]
                     .__setitem__("per_config_nll",
                                  [10 ** 1000] * len(registry.BASELINES[
                                      next(iter(d["evidence"]))]["configs"])),
                     "unusable entries"),
                    ("per_config_nll holding inf",
                     lambda d: d["evidence"][next(iter(d["evidence"]))]
                     .__setitem__("per_config_nll",
                                  [float("inf")] * len(registry.BASELINES[
                                      next(iter(d["evidence"]))]["configs"])),
                     "unusable entries"),
                    ("per_config_nll holding a string",
                     lambda d: d["evidence"][next(iter(d["evidence"]))]
                     .__setitem__("per_config_nll",
                                  ["x"] * len(registry.BASELINES[
                                      next(iter(d["evidence"]))]["configs"])),
                     "unusable entries")):
                _dd = doc()
                _mut(_dd)
                put(_dd)
                _bl = validation_selection_blockers()
                check(f"{_lbl} is a blocker, not an exception",
                      bool(_bl) and any(_want in x for x in _bl),
                      "; ".join(_bl)[:70] or "SILENTLY CLEAN")
            # the ROOT: valid JSON that is not an object at all
            for _root in (42, None, True):
                _sp = Path(td) / "results" / "baseline_validation_selection.json"
                _sp.write_text(json.dumps(_root), encoding="utf-8")
                _bl = validation_selection_blockers()
                check(f"a document that is JSON {_root!r} is a blocker",
                      bool(_bl) and any("expected a JSON object" in x
                                        for x in _bl),
                      "; ".join(_bl)[:70] or "SILENTLY CLEAN")
            put(doc())              # restore, the arms below share this file

            check("a selection recomputed from REGISTERED-shape NPZs passes",
                  validation_selection_blockers() == [],
                  f"3 seeds x {NQ} queries x {NCAND} candidates")

            # EVERY seed, not just the first. Sabotage showed that checking
            # seed 42 alone left every existing arm green, so the loop over
            # the three seeds was guarding nothing: one id list described one
            # of three draws and vouched for all of them.
            _bad_q = [list(qids[0]),
                      [f"{i + 10 ** 7:064x}" for i in range(NQ)],
                      list(qids[2])]
            _n1 = next(iter(registry.BASELINES))
            _cfgs = registry.BASELINES[_n1]["configs"]
            _fb = write_npz(_n1, _cfgs, qids_over=_bad_q)
            _db = dict(np.load(_fb, allow_pickle=False))
            _sb = json.loads(Path(str(_fb).replace(".npz", ".json"))
                             .read_text(encoding="utf-8"))
            _fa = npz_identity_faults(_n1, _fb, _db, _sb, manifest=qm,
                                      label_space=lsp, configs=_cfgs)
            check("a result whose SEED 43 queries are wrong is caught",
                  any("seed 43" in x for x in _fa),
                  "; ".join(_fa)[:70]
                  or "SILENTLY CLEAN -- only seed 42 was being checked")
            _fo = write_npz(_n1, _cfgs)
            _do = dict(np.load(_fo, allow_pickle=False))
            _so = json.loads(Path(str(_fo).replace(".npz", ".json"))
                             .read_text(encoding="utf-8"))
            # GOLD, not only query_ids: the arm above varies the ids, so
            # deleting the per-seed gold comparison left it green. Seed 43's
            # gold is wrong here and its ids are right.
            _bad_g = [list(cls[0]),
                      [(c + 1) % NCAND for c in cls[1]],
                      list(cls[2])]
            _fg = write_npz(_n1, _cfgs, gold_over=_bad_g)
            _dg = dict(np.load(_fg, allow_pickle=False))
            _sg2 = json.loads(Path(str(_fg).replace(".npz", ".json"))
                              .read_text(encoding="utf-8"))
            _fga = npz_identity_faults(_n1, _fg, _dg, _sg2, manifest=qm,
                                       label_space=lsp, configs=_cfgs)
            check("a result whose SEED 43 gold classes are wrong is caught",
                  any("seed 43" in x and "gold" in x for x in _fga),
                  "; ".join(_fga)[:70]
                  or "SILENTLY CLEAN -- the gold path had no arm")
            _frac = write_npz(_n1, _cfgs,
                              seeds_over=np.array([42.9, 43.0, 44.0]))
            _dfr = dict(np.load(_frac, allow_pickle=False))
            _sfr = json.loads(Path(str(_frac).replace(".npz", ".json"))
                              .read_text(encoding="utf-8"))
            check("a FRACTIONAL seed is refused, not truncated to 42",
                  any("42.9" in x or "integer" in x for x in
                      npz_identity_faults(_n1, _frac, _dfr, _sfr,
                                          manifest=qm, label_space=lsp,
                                          configs=_cfgs)),
                  "bare int() accepted it and matched the registered value")

            # STRICT IDENTITY TYPES, each on a reachable artifact. The
            # earlier gold arm varied only the VALUE, and the fixture forced
            # int64, so a fractional gold could not be constructed and
            # reverting strict_int left it green.
            for _lbl, _kw in (
                    ("gold_class", {"gold_over": [
                        [float(c) for c in cls[0]],
                        [0.75] + [float(c) for c in cls[1][1:]],
                        [float(c) for c in cls[2]]]}),
                    ("candidate_classes", {"cand_over":
                                           [0.75] + list(range(1, NCAND))}),
                    ("candidate_token_ids", {"tok_over":
                                             [1000.75] + [1001 + i for i
                                                          in range(NCAND - 1)]}),
                    ("sidecar K", {"K": 5.9}),
                    ("sidecar demo_seeds", {"demo_seeds": [42.9, 43, 44]})):
                _fx = write_npz(_n1, _cfgs, **_kw)
                _dx = dict(np.load(_fx, allow_pickle=False))
                _sx = json.loads(Path(str(_fx).replace(".npz", ".json"))
                                 .read_text(encoding="utf-8"))
                # RETURNED, not raised. _raise_if used to wrap this, and
                # the exception fired while evaluating its argument -- so the
                # arm passed whether the function raised or returned, which
                # is exactly the contract under test.
                _ff = npz_identity_faults(_n1, _fx, _dx, _sx, manifest=qm,
                                          label_space=lsp, configs=_cfgs)
                check(f"a fractional {_lbl} is REPORTED as a fault, not "
                      "raised",
                      bool(_ff) and any("integer" in x for x in _ff),
                      "; ".join(_ff)[:70] or "SILENTLY CLEAN")

            # THE MANIFEST's ROOT, and an unreadable manifest. The root
            # arms above all mutate the SELECTION document; none of them
            # reaches npz_identity_faults with a manifest that is not an
            # object, so the check there was unguarded.
            _fr = write_npz(_n1, _cfgs)
            _dr = dict(np.load(_fr, allow_pickle=False))
            _sr = json.loads(Path(str(_fr).replace(".npz", ".json"))
                             .read_text(encoding="utf-8"))
            for _lbl, _txt, _want in (
                    ("a manifest whose ROOT is a number", "42",
                     "expected a JSON object"),
                    ("a manifest that is not JSON at all", "{not json",
                     "could not be read")):
                _qb = Path(td) / f"manifest_bad_{len(_txt)}.json"
                _qb.write_text(_txt, encoding="utf-8")
                # the label space binds to the manifest's hash, so it must be
                # re-pointed or IT fails first and the arm passes for the
                # wrong reason -- which is what the first version did.
                _lb = Path(td) / f"label_space_bad_{len(_txt)}.json"
                _lj = json.loads(lsp.read_text(encoding="utf-8"))
                _lj["query_manifest_sha256"] = sha256_file(_qb)
                _lb.write_text(json.dumps(_lj), encoding="utf-8")
                _srb = dict(_sr)
                _srb["query_manifest_sha256"] = sha256_file(_qb)
                _srb["label_space_sha256"] = sha256_file(_lb)
                _rf = npz_identity_faults(_n1, _fr, _dr, _srb, manifest=_qb,
                                          label_space=_lb, configs=_cfgs)
                check(f"{_lbl} is reported, not raised",
                      bool(_rf) and any(_want in x for x in _rf),
                      "; ".join(_rf)[:70] or "SILENTLY CLEAN")
            # THE INNER LAYERS. Validating the root alone left three
            # legal JSON documents raising rather than reporting, each one
            # level deeper than the last.
            def _bad_manifest(tag, mutate):
                _m = json.loads(qm.read_text(encoding="utf-8"))
                mutate(_m)
                _q = Path(td) / f"manifest_{tag}.json"
                _q.write_text(json.dumps(_m), encoding="utf-8")
                _l = Path(td) / f"label_space_{tag}.json"
                _lj = json.loads(lsp.read_text(encoding="utf-8"))
                _lj["query_manifest_sha256"] = sha256_file(_q)
                _l.write_text(json.dumps(_lj), encoding="utf-8")
                _sd2 = dict(_sr)
                _sd2["query_manifest_sha256"] = sha256_file(_q)
                _sd2["label_space_sha256"] = sha256_file(_l)
                return npz_identity_faults(_n1, _fr, _dr, _sd2, manifest=_q,
                                           label_space=_l, configs=_cfgs)

            for _lbl, _mut, _want in (
                    ("validation_by_seed as a number",
                     lambda m: m.__setitem__("validation_by_seed", 42),
                     "expected an object keyed by seed"),
                    ("one seed's value as a number",
                     lambda m: m["validation_by_seed"].__setitem__("42", 17),
                     "expected a list"),
                    ("a row that is a number",
                     lambda m: m["validation_by_seed"]["42"].__setitem__(0, 5),
                     "expected an object"),
                    ("a row with no query_id",
                     lambda m: m["validation_by_seed"]["42"].__setitem__(
                         0, {"class_idx": 0}),
                     "has no ['query_id']"),
                    # BOTH required fields. Only query_id was armed, so
                    # dropping class_idx from the check left every arm green
                    # while a real absence raised KeyError later, in want_cls.
                    ("a row with no class_idx",
                     lambda m: m["validation_by_seed"]["42"].__setitem__(
                         0, {"query_id": "0" * 64}),
                     "has no ['class_idx']")):
                _ff2 = _bad_manifest(_lbl.replace(" ", "_")[:20], _mut)
                check(f"{_lbl} is reported, not raised",
                      bool(_ff2) and any(_want in x for x in _ff2),
                      "; ".join(_ff2)[:70] or "SILENTLY CLEAN")

            # NON-NUMERIC LOGITS, on both fault APIs. np.asarray(..., float)
            # of a string array raises, and the conversion sat outside the
            # try in each.
            _fstr = Path(td) / "results" / "baselines" / "strlogits.npz"
            _fstr.parent.mkdir(parents=True, exist_ok=True)
            np.savez(_fstr,
                     candidate_logits=np.array([["a"]], dtype="<U1"),
                     arms=np.array(["natural"]),
                     seeds=np.array([42, 43, 44]),
                     query_ids=np.array(qids),
                     gold_class=np.array(cls, dtype=np.int64),
                     candidate_classes=np.arange(NCAND, dtype=np.int64),
                     candidate_token_ids=np.array(
                         [1000 + i for i in range(NCAND)], dtype=np.int64))
            _dstr = dict(np.load(_fstr, allow_pickle=False))
            # the PRODUCTION identity path, with 0-D arms. Last round
            # fixed only the identity-free copy.
            _f0d = Path(td) / "results" / "baselines" / "arms0d.npz"
            _f0d.parent.mkdir(parents=True, exist_ok=True)
            _base0 = dict(_dr)
            _base0["arms"] = np.array("default")
            np.savez(_f0d, **_base0)
            _d0d = dict(np.load(_f0d, allow_pickle=False))
            _f0 = npz_identity_faults(_n1, _f0d, _d0d, _sr, manifest=qm,
                                      label_space=lsp, configs=_cfgs)
            check("0-D arms are a fault on the IDENTITY path too",
                  bool(_f0) and any("1-D" in x for x in _f0),
                  "; ".join(_f0)[:70] or "SILENTLY CLEAN")

            _f_id = npz_identity_faults(_n1, _fstr, _dstr, _sr, manifest=qm,
                                        label_space=lsp, configs=_cfgs)
            check("string candidate_logits are reported by the identity API",
                  bool(_f_id) and any("float64" in x for x in _f_id),
                  "; ".join(_f_id)[:70] or "SILENTLY CLEAN")
            # ALL-INF logits, on the identity-free path. It returned
            # ([nan], []) -- NaN scores while reporting no fault.
            _finf = Path(td) / "results" / "baselines" / "inflogits.npz"
            np.savez(_finf,
                     candidate_logits=np.full(
                         (len(_cfgs) + 1, 3, NQ, NCAND), np.inf),
                     arms=np.array(["natural"] + [config_arm_name(c)
                                                  for c in _cfgs]),
                     seeds=np.array([42, 43, 44]),
                     query_ids=np.array(qids),
                     gold_class=np.array(cls, dtype=np.int64),
                     candidate_classes=np.arange(NCAND, dtype=np.int64),
                     candidate_token_ids=np.array(
                         [1000 + i for i in range(NCAND)], dtype=np.int64))
            # THE IDENTITY-FREE PATH'S OWN PREMISES. Everything here is
            # driven WITHOUT the identity arguments, because that is the mode
            # in which this function vouches for its own arithmetic.
            _armnames = ["natural"] + [config_arm_name(c) for c in _cfgs]

            def _mk(tag, **kw):
                _f = Path(td) / "results" / "baselines" / f"{tag}.npz"
                _f.parent.mkdir(parents=True, exist_ok=True)
                base = dict(
                    candidate_logits=np.zeros(
                        (len(_armnames), 3, NQ, NCAND)),
                    arms=np.array(_armnames),
                    seeds=np.array([42, 43, 44]),
                    query_ids=np.array(qids),
                    gold_class=np.array(cls, dtype=np.int64),
                    candidate_classes=np.arange(NCAND, dtype=np.int64),
                    candidate_token_ids=np.array(
                        [1000 + i for i in range(NCAND)], dtype=np.int64))
                base.update(kw)
                np.savez(_f, **base)
                return _f

            for _lbl, _kw, _want in (
                    ("a 0-D arms array",
                     {"arms": np.array("default")}, "1-D"),
                    ("fewer logit slabs than arms",
                     {"candidate_logits": np.zeros((1, 3, NQ, NCAND))},
                     "arm slabs"),
                    ("a repeated arm name",
                     {"arms": np.array([_armnames[0]] * len(_armnames))},
                     "arms repeat"),
                    ("a repeated candidate class",
                     {"candidate_classes": np.array(
                         [0] + list(range(NCAND - 1)), dtype=np.int64)},
                     "candidate_classes repeat"),
                    # ndim must be read BEFORE any axis: shape[0] on a
                    # scalar raised IndexError, so the guard protecting the
                    # index was itself unguarded.
                    ("a SCALAR candidate_logits",
                     {"candidate_logits": np.float64(1.0)}, "0-D"),
                    # zero-size axes pass every shape comparison and then
                    # reduce to nothing
                    ("an empty candidate axis",
                     {"candidate_logits": np.zeros(
                         (len(_armnames), 3, NQ, 0)),
                      "candidate_classes": np.array([], dtype=np.int64),
                      "candidate_token_ids": np.array([], dtype=np.int64)},
                     "zero-length"),
                    ("an empty query axis",
                     {"candidate_logits": np.zeros(
                         (len(_armnames), 3, 0, NCAND)),
                      "query_ids": np.zeros((3, 0), dtype="<U64"),
                      "gold_class": np.zeros((3, 0), dtype=np.int64)},
                     "zero-length")):
                _sx2, _fx2 = recompute_per_config_nll(
                    _mk(_lbl.replace(" ", "_")[:24], **_kw), _cfgs)
                check(f"{_lbl} is a fault, not an exception",
                      _sx2 is None and _fx2 and any(_want in x for x in _fx2),
                      f"{_sx2}, {_fx2}"[:70] or "SILENTLY CLEAN")

            # THE REASON, pinned. A -inf in a non-gold column leaves the
            # NLL finite -- lse = log(e^0 + 0) = 0, so NLL = 0.0 -- so the
            # refusal cannot be justified by "every score would be NaN",
            # which is what the message used to claim. This arm holds the
            # wording to what is actually true: the contract is violated.
            _mix = np.zeros((len(_armnames), 3, NQ, NCAND))
            _mix[:, :, :, 1] = -np.inf          # a NON-gold column
            _fmix = _mk("mixed_neginf", candidate_logits=_mix)
            _smix, _fmixf = recompute_per_config_nll(_fmix, _cfgs)
            check("a -inf in a non-gold column is still refused, and the "
                  "reason is the CONTRACT rather than NaN",
                  _smix is None and _fmixf
                  and any("finite-logits contract" in x for x in _fmixf)
                  and not any("would be NaN" in x for x in _fmixf),
                  "; ".join(_fmixf or [])[:70])
            # ...and the arithmetic that makes the old wording false
            _z = np.array([0.0, -np.inf])
            _mx = _z.max()
            check("...because that case's NLL is 0.0, not NaN",
                  float(_mx + np.log(np.exp(_z - _mx).sum()) - _z[0]) == 0.0,
                  "log(e^0 + e^-inf) = 0, so lse - logit_gold = 0")

            # FINITE IN, non-finite OUT: both logits are finite and their
            # difference is not. It returned ([inf], []).
            _ov = np.zeros((len(_armnames), 3, NQ, NCAND))
            _ov[:, :, :, 0] = -1e308
            _ov[:, :, :, 1] = 1e308
            _sov, _fov = recompute_per_config_nll(
                _mk("overflow", candidate_logits=_ov), _cfgs)
            check("finite logits whose difference overflows are a fault",
                  bool(_fov) and any("overflow" in x or "non-finite" in x
                                     for x in _fov),
                  f"{_sov}, {_fov}"[:70] or "SILENTLY CLEAN")

            _sci, _fli = recompute_per_config_nll(_finf, _cfgs)
            check("all-inf logits are a fault, not NaN scores with none",
                  _sci is None and _fli and any("non-finite" in x
                                                for x in _fli),
                  f"{_sci}, {_fli}"[:70] or "SILENTLY CLEAN")

            _sc3, _fl3 = recompute_per_config_nll(_fstr, _cfgs)
            check("...and by the recomputation, with identity args omitted",
                  _sc3 is None and _fl3 and any("float64" in x for x in _fl3),
                  "; ".join(_fl3 or [])[:70] or "SILENTLY CLEAN")

            _qmiss = Path(td) / "manifest_absent.json"
            _rf = npz_identity_faults(_n1, _fr, _dr, _sr, manifest=_qmiss,
                                      label_space=lsp, configs=_cfgs)
            check("...and a MISSING manifest too",
                  bool(_rf) and any("could not be read" in x for x in _rf),
                  "; ".join(_rf)[:70] or "SILENTLY CLEAN")

            # THE MANIFEST's own class ids. Reachable after all: the
            # label space records the manifest's sha256, so updating THAT
            # alongside lets FrozenLabelSpace.load pass and the class check
            # is then what fires. An earlier note here called the site
            # unreachable -- that was a conclusion drawn from a failed
            # attempt rather than from the structure.
            _qmf = Path(td) / "manifest_fracclass.json"
            _mm = json.loads(qm.read_text(encoding="utf-8"))
            _mm["validation_by_seed"]["43"][0]["class_idx"] = 0.75
            _qmf.write_text(json.dumps(_mm), encoding="utf-8")
            _lsf = Path(td) / "label_space_fracclass.json"
            _ll = json.loads(lsp.read_text(encoding="utf-8"))
            _ll["query_manifest_sha256"] = sha256_file(_qmf)
            _lsf.write_text(json.dumps(_ll), encoding="utf-8")
            _fm = write_npz(_n1, _cfgs)
            _dm = dict(np.load(_fm, allow_pickle=False))
            _sm2 = json.loads(Path(str(_fm).replace(".npz", ".json"))
                              .read_text(encoding="utf-8"))
            _sm2["query_manifest_sha256"] = sha256_file(_qmf)
            _sm2["label_space_sha256"] = sha256_file(_lsf)
            Path(str(_fm).replace(".npz", ".json")).write_text(
                json.dumps(_sm2), encoding="utf-8")
            _mf = npz_identity_faults(_n1, _fm, _dm, _sm2, manifest=_qmf,
                                      label_space=_lsf, configs=_cfgs)
            check("a fractional class_idx in the MANIFEST is reported",
                  any("class_idx" in x and "integer" in x for x in _mf),
                  "; ".join(_mf)[:70] or "SILENTLY CLEAN")

            # recompute_per_config_nll's OWN candidate check. Its identity
            # arguments are OPTIONAL, so calling without them skips the
            # validator that was shadowing it.
            _fc2 = write_npz(_n1, _cfgs,
                             cand_over=[0.75] + list(range(1, NCAND)))
            _sc2, _fl2 = recompute_per_config_nll(_fc2, _cfgs)
            check("...and the recomputation refuses a fractional candidate "
                  "class on its OWN path, with the validator skipped",
                  _sc2 is None and _fl2 and any("integer" in x for x in _fl2),
                  "; ".join(_fl2 or [])[:70] or "SILENTLY CLEAN")

            # the RECOMPUTATION reads gold separately from the identity
            # check, so it needs its own arm: reverting one and not the other
            # must still show.
            _fg2 = write_npz(_n1, _cfgs, gold_over=[
                [float(c) for c in cls[0]],
                [0.75] + [float(c) for c in cls[1][1:]],
                [float(c) for c in cls[2]]])
            _sc, _fl = recompute_per_config_nll(
                _fg2, _cfgs, name=_n1, manifest=qm, label_space=lsp)
            check("...and the NLL recomputation refuses it too, separately",
                  _sc is None and _fl,
                  "; ".join(_fl or [])[:70]
                  or "SILENTLY CLEAN -- it reads gold on its own path")

            # REBUILT here: write_npz writes the same path for a given
            # baseline, so the arms above overwrote the file the green
            # companion was built from -- it failed for that reason and not
            # because the validator was wrong.
            _fo = write_npz(_n1, _cfgs)
            _do = dict(np.load(_fo, allow_pickle=False))
            _so = json.loads(Path(str(_fo).replace(".npz", ".json"))
                             .read_text(encoding="utf-8"))
            check("...while the correct three seeds pass",
                  not npz_identity_faults(_n1, _fo, _do, _so, manifest=qm,
                                          label_space=lsp, configs=_cfgs),
                  "otherwise the arm above would fire on anything")

            off = doc()
            off["selected"]["zerotuning_level1"] = \
                registry.BASELINES["zerotuning_level1"]["configs"][3]
            put(off)
            check("an ON-GRID configuration the rule does NOT pick is caught",
                  any("RECOMPUTED from its NPZ" in x
                      for x in validation_selection_blockers()))

            faked = doc()
            faked["evidence"]["fv_on_icl"]["per_config_nll"] = \
                [0.1, 9.0, 9.0, 9.0, 9.0]
            put(faked)
            check("recorded NLLs that disagree with the NPZ are caught",
                  any("do not match the ones recomputed" in x
                      for x in validation_selection_blockers()))

            print("\n  ...and the NPZ must BE the registered validation run")
            base = doc()
            for label, mangle in (
                    ("holds 12 queries instead of 144",
                     lambda z: {**z, "query_ids": z["query_ids"][:12],
                                "gold_class": z["gold_class"][:12],
                                "candidate_logits":
                                    z["candidate_logits"][:, :, :12, :]}),
                    ("was run on other seeds",
                     lambda z: {**z, "seeds": np.array([42, 43, 45])}),
                    ("uses another candidate space",
                     lambda z: {**z, "candidate_token_ids":
                                np.arange(NCAND, dtype=np.int64)}),
                    ("scored a different query set",
                     lambda z: {**z, "query_ids": np.array(
                         [f"{i + 10 ** 6:064x}" for i in range(NQ)])}),
                    ("has 4 candidates rather than 36",
                     lambda z: {**z, "candidate_classes": np.arange(4),
                                "candidate_token_ids": np.arange(4),
                                "candidate_logits":
                                    z["candidate_logits"][:, :, :, :4]})):
                f = Path(td) / expected_result_npz("fv_on_icl")
                with np.load(f, allow_pickle=False) as zf:
                    arrs = {k: zf[k] for k in zf.files}
                np.savez(f, **mangle(arrs))
                side = Path(str(f).replace(".npz", ".json"))
                sd = json.loads(side.read_text(encoding="utf-8"))
                sd["npz_sha256"] = sha256_file(f)
                side.write_text(json.dumps(sd), encoding="utf-8")
                d = json.loads(json.dumps(base))
                d["evidence"]["fv_on_icl"]["result_npz_sha256"] = sha256_file(f)
                put(d)
                check(f"an NPZ that {label} is caught",
                      validation_selection_blockers() != [],
                      (validation_selection_blockers() or [""])[0][:66])
                write_npz("fv_on_icl",
                          registry.BASELINES["fv_on_icl"]["configs"])

            print("\n  ...and a bias-bearing arm must BIND its bias")
            bpath = res / "baselines" / BIAS_BEARING["inductive_bc"]
            base_ok = doc()
            put(base_ok)
            check("a correctly bound bias passes",
                  validation_selection_blockers() == [],
                  (validation_selection_blockers() or [""])[0][:58])
            # ⚠ THE FAIL-OPEN. `if sidecar.get("bias_sha256")` meant that
            # DELETING the field skipped the entire binding check.
            sp = Path(td) / expected_result_npz("inductive_bc")
            sidep = Path(str(sp).replace(".npz", ".json"))
            sd = json.loads(sidep.read_text(encoding="utf-8"))
            del sd["bias_sha256"]
            sidep.write_text(json.dumps(sd), encoding="utf-8")
            check("a MISSING bias_sha256 is a FAILURE, not a skipped check",
                  any("records no bias_sha256" in x
                      for x in validation_selection_blockers()),
                  (validation_selection_blockers() or [""])[0][-58:])
            sd["bias_sha256"] = "0" * 64
            sidep.write_text(json.dumps(sd), encoding="utf-8")
            check("a bias_sha256 that does not match the file is caught",
                  any("different runs" in x
                      for x in validation_selection_blockers()))
            bpath.unlink()
            check("a bias named but absent beside the result is caught",
                  any("is not beside the result" in x
                      for x in validation_selection_blockers()))
            write_npz("inductive_bc",
                      registry.BASELINES["inductive_bc"]["configs"])
            put(doc())
            check("...and rebuilding the pair restores validity",
                  validation_selection_blockers() == [])

            print("\n  ...and the LABEL SPACE goes through the real loader")
            good_ls = lsp.read_text(encoding="utf-8")
            # Build the selection while the label space is still good --
            # `doc()` recomputes, and recomputation is what these arms break.
            keep = doc()
            for label, payload in (
                    ("a two-field stub -- what this fixture used to use",
                     {"eligible_classes": list(range(NCAND)),
                      "candidate_token_ids": [1000 + i for i in range(NCAND)]}),
                    ("a space built against another manifest",
                     {**json.loads(good_ls), "query_manifest_sha256": "0" * 64}),
                    ("a space whose candidates are not its eligible entries",
                     {**json.loads(good_ls),
                      "candidate_token_ids": [9000 + i for i in range(NCAND)]})):
                lsp.write_text(json.dumps(payload), encoding="utf-8")
                # Update the recorded hash too, so the ONLY thing left to
                # object to is the loader. Otherwise the header's hash check
                # fires first and the arm proves nothing about the loader.
                put({**keep, "label_space_sha256": sha256_file(lsp)})
                blk = validation_selection_blockers()
                hit = [x for x in blk if "label space did not load" in x]
                check(f"{label} is refused BY THE LOADER", bool(hit),
                      (hit or blk or [""])[0][-64:])
            lsp.write_text(good_ls, encoding="utf-8")

            print("\n  ...and its SIDECAR must say the registered setting")
            for label, over in (
                    ("mode 'frozen' -- a test run choosing a configuration",
                     {"mode": "frozen"}),
                    ("another model", {"model": "meta-llama/Llama-2-7b-hf"}),
                    ("another task", {"task": "sst2"}),
                    ("K=10", {"K": 10}),
                    ("float16", {"dtype": "torch.float16"}),
                    ("sdpa attention", {"attn_implementation": "sdpa"}),
                    ("another baseline's name", {"baseline": "unibias_code"}),
                    ("a stale manifest hash",
                     {"query_manifest_sha256": "0" * 64}),
                    ("a stale label-space hash",
                     {"label_space_sha256": "0" * 64}),
                    ("an npz hash that does not match the file",
                     {"npz_sha256": "0" * 64})):
                write_npz("fv_on_icl",
                          registry.BASELINES["fv_on_icl"]["configs"], **over)
                d = json.loads(json.dumps(base))
                f = Path(td) / expected_result_npz("fv_on_icl")
                d["evidence"]["fv_on_icl"]["result_npz_sha256"] = sha256_file(f)
                put(d)
                check(f"a sidecar declaring {label} is caught",
                      validation_selection_blockers() != [],
                      (validation_selection_blockers() or [""])[0][:66])
            write_npz("fv_on_icl", registry.BASELINES["fv_on_icl"]["configs"])

            print("\n  ...and the path is the ONE registered path")
            put(doc())
            for label, path in (
                    ("an absolute path", str(Path(td) / "elsewhere.npz")),
                    ("a traversal", "../outside.npz"),
                    ("another arm's file",
                     expected_result_npz("unibias_code"))):
                d = doc()
                d["evidence"]["fv_on_icl"]["result_npz_path"] = path
                put(d)
                check(f"{label} is refused",
                      any("registered path" in x
                          for x in validation_selection_blockers()))

            print("\n  ...and the rest of the header")
            put(doc())
            for label, over in (
                    ("the wrong metric", {"metric": "accuracy"}),
                    ("a rewritten tie rule", {"tie_rule": "highest wins"}),
                    ("an older schema version", {"schema_version": 0}),
                    ("a different spec string", {"spec": "my notes"}),
                    ("seeds that are not the registered three",
                     {"demo_seeds": [42, 43, 45]})):
                put(doc(**over))
                check(f"{label} is caught",
                      validation_selection_blockers() != [])

            put(doc())
            hashed = artifact_hashes()[0]
            check("the result NPZs AND their sidecars join the code-stage "
                  "hashes",
                  sum(k.endswith("_validation.npz") for k in hashed) == 6
                  and sum(k.endswith("_validation.json") for k in hashed) == 6,
                  f"{len(hashed)} artifact hashes")
            sc = doc()["evidence"]["fv_on_icl"]["per_config_nll"]
            check("the recomputed scores make config 0 the argmin",
                  sc[0] < min(sc[1:]), f"{[round(x, 4) for x in sc]}")
        finally:
            globals()["REPO"] = real_repo

    print()
    if not ok:
        print("FIXTURES FAILED -- do not freeze.")
        return 2
    print("the opponent freeze holds: configs, hook boundaries, spec documents, "
          "artifacts and upstream commits are hashed, a freeze must be VALID "
          "and not merely undrifted, and these fixtures do not depend on this "
          "machine's git state (see --integration-test).")
    return 0


def integration_test() -> int:
    """The REAL upstream checkouts. Needs git access to them; deliberately not
    part of --self-test."""
    print("=" * 78)
    print("UPSTREAM INTEGRATION CHECK -- the real checkouts")
    print("=" * 78)
    print(f"  root: {upstream_root()}")
    repos = {name: git_head(upstream_root() / name) for name in UPSTREAM}
    bad = (missing_registered_commits() + unreadable_repos(repos)
           + wrong_commit(repos) + dirty_repos(repos))
    for name in UPSTREAM:
        v = repos[name]
        state = (f"{v['head'][:12]} {'clean' if v.get('clean') else 'DIRTY'}"
                 if v.get("available") else f"UNREADABLE ({v.get('why')})")
        print(f"  {name:<24} {state}")
    if bad:
        print("\n  blockers:")
        for b in bad:
            print(f"    - {b}")
        print("\nVERDICT: FAIL -- --write --stage spec would be refused. Fix "
              "these on the machine that will run the experiments; "
              "`--propose-commits` prints the current HEADs.")
        return 3
    print("\nVERDICT: PASS -- every repository is readable, clean, and at its "
          "registered commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
