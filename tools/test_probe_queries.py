"""queries_for_run and main()'s wiring to it, armed without a model.

WHY THIS FILE EXISTS. Three defects lived in this logic in three consecutive
rounds and not one of them was catchable, because the only way in was main()
and main() needs a GPU:

  * one load reused for every seed, so three result blocks all came from seed
    42's validation while being labelled 42, 43 and 44-- a file, not an error;
  * a single-seed alias kept "for the census" that also fed the missing-query
    check, so seed 43's and 44's own absences surfaced as a KeyError deep in
    the forward loop instead of up front;
  * --limit slicing that alias while the loop read the mapping, so the limit
    had no effect at all.

WHAT IS AND IS NOT COVERED. Sections 1-6 drive `queries_for_run` and the
section 2.1 test lock directly: neither touches a model or a tokenizer, so
these are the real code. Section 7 is a WIRING gate over main()'s SYNTAX
TREE, not an end-to-end run -- reaching main()'s forward loop needs a
tokenizer, a label space, a calibration file, a model and carriers, so
behavioural coverage there is not available off a GPU. It is narrow on
purpose, pinning only the joints a helper arm cannot see:

  * the limit/text_of call must be assigned to `queries_by_seed`, not to some
    ignored name while the loop keeps reading the census result;
  * its faults must actually be read;
  * every call must pass `expected_roles`, or the freeze is not bound to the
    carriers and label space the run loads;
  * the forward loop must index `queries_by_seed` by ITS OWN loop variable.

Naming that limit matters -- an earlier version of this file was described as
a main-path fixture when it never called main() at all.

NO SKIPS. `install_fixture_mocks` is called unconditionally so that a machine
where git refuses the upstream checkouts still runs every arm. The earlier
version skipped the lock-opening, test_seed and test_common arms there and
still exited 0: fifteen PASS lines and a green light while the arms that
matter never ran.

THE THREE SPLITS DIFFER:

  validation   drawn per seed (14.0a); a run spans all three
  test_seed    drawn per seed (2.2), but probe `validate()` restricts a run to
               ONE seed, so the three-seed shape is not its production shape
  test_common  ONE shared set that load_split returns whatever demo_seed is
               passed, because holding the queries fixed under three prefixes
               is what the L2 comparison needs

The per-seed rule under test is PAIRING, not distinctness: each seed's rows
must equal the manifest's own rows FOR THAT SEED, in order. "the results must
differ" passes a permuted loop (42's rows handed to 43, 43's to 44, 44's to
42) in which every output carries the wrong seed label, and it also forbids a
coincidence that independent draws do not forbid.

Run: python tools/test_probe_queries.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.build_query_manifest import write_freeze_manifest  # noqa: E402
from tools.probe_prototype_shrinkage import queries_for_run  # noqa: E402

SEEDS = [42, 43, 44]
NC, PER = 4, 4
PROBE_SRC = REPO / "tools" / "probe_prototype_shrinkage.py"


def write_manifest(d):
    """A manifest with per-seed validation and a shared test pool."""
    def rows(off):
        return [{"query_id": f"{(off + i):064x}", "class_idx": i % NC,
                 "source_text": f"t{off + i}"}
                for i in range(NC * PER)]

    pool = [{"query_id": f"{(900 + i):064x}", "class_idx": i % NC,
             "role": "pool"} for i in range(8)]
    Path(d).mkdir(parents=True, exist_ok=True)
    p = Path(d) / "prereg_method_A_query_manifest.json"
    p.write_text(json.dumps({
        "eligible_classes": list(range(NC)),
        "validation_by_seed": {str(sd): rows(100 * i)
                               for i, sd in enumerate(SEEDS)},
        "entries": pool,
        # each seed a DIFFERENT slice, so a permuted loop is detectable
        "test_by_seed": {str(sd): [e["query_id"] for e in pool[i:i + 4]]
                         for i, sd in enumerate(SEEDS)},
    }), encoding="utf-8")
    return p


def write_freeze(td, man, *, stage="code", kind=True, drop_role=None,
                 roles=True, carriers=None, label_space=None, spec_rec=None,
                 tag=""):
    """A freeze manifest that may open the test lock, and its knobs."""
    import tools.baselines.freeze_baseline_spec as _fbs
    rec = spec_rec if spec_rec is not None else _fbs.build_freeze(stage)
    spec = Path(td) / f"spec_freeze_{stage}{tag}.json"
    spec.write_text(json.dumps(rec), encoding="utf-8")

    car = Path(carriers or Path(td) / "carriers.json")
    if not car.exists():
        car.write_text(json.dumps({"by_layer": {}}), encoding="utf-8")
    lsp = Path(label_space or Path(td) / "label_space.json")
    if not lsp.exists():
        lsp.write_text(json.dumps({"eligible_classes": []}), encoding="utf-8")
    gam = Path(td) / "gamma_star.json"
    if not gam.exists():
        gam.write_text(json.dumps({"gamma_L2": 0.5, "gamma_group": 0.5,
                                   "gamma_head": 0.25}), encoding="utf-8")

    # THE PRODUCTION WRITER, then knobs on top. Hand-rolling the format here
    # would let the fixture and production drift apart, which is how "any JSON
    # with a non-empty 'files' map" became the de facto schema.
    slug = (f"{stage}_{kind}_{drop_role}_{roles}_{car.stem}"
            f"_{Path(man).parent.name}{tag}")
    out = Path(td) / f"freeze_{slug}.json"
    doc = write_freeze_manifest(out, query_manifest=man, carriers=car,
                                label_space=lsp, gammas=gam, spec_freeze=spec)
    if not kind:
        doc.pop("kind")
    if not roles:
        doc.pop("roles")
    elif drop_role:
        doc["roles"].pop(drop_role)
    out.write_text(json.dumps(doc), encoding="utf-8")
    return out


def main_wiring_faults():
    """Structural facts about main() that sections 1-4 cannot see.

    Categorical, not numeric (working rules 11b): a wrong subscript is a
    different name, which no tolerance is needed to detect and which a
    mutation cannot imitate.
    """
    tree = ast.parse(PROBE_SRC.read_text(encoding="utf-8"))
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    if fn is None:
        return ["probe_prototype_shrinkage has no main()"]
    out = []

    # THE CALL, AND WHAT IS DONE WITH ITS RESULT. Checking only that a call
    # with limit/text_of exists passes a main() that assigns it to
    # `_ignored_by_seed, _ignored_faults` and keeps using the FIRST result --
    # the limit applied, the missing check run, and both thrown away. So the
    # assignment targets are pinned, and the faults name must be consumed.
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name)
               and n.value.func.id == "queries_for_run"]
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "queries_for_run"]
    if len(calls) != 2:
        out.append(f"main() calls queries_for_run {len(calls)} times, "
                   "expected 2 (census, then again once text_of exists)")
    if len(assigns) != len(calls):
        out.append(f"{len(calls) - len(assigns)} queries_for_run call(s) in "
                   "main() are not assigned at all, so their result is "
                   "discarded where it is computed")
    # THE FREEZE BINDING. queries_for_run defaults expected_roles to None, so
    # a call that simply omits it type-checks, runs, and leaves the lock
    # willing to accept a freeze built against different carriers and a
    # different label space than the ones this run loads.
    # `expected_roles=None` is the same as omitting it, so the keyword's
    # PRESENCE proves nothing -- and neither does a NAME, since the name can
    # be bound to None. The value must resolve to a non-empty dict literal,
    # following one level of local binding.
    def dict_literal(v):
        if isinstance(v, ast.Dict):
            return bool(v.keys)
        if isinstance(v, ast.Name):
            binds = [a for a in ast.walk(fn) if isinstance(a, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == v.id
                             for t in a.targets)]
            return bool(binds) and all(dict_literal(a.value) for a in binds)
        return False

    def binds_roles(c):
        kw = {k.arg: k.value for k in c.keywords}
        return "expected_roles" in kw and dict_literal(kw["expected_roles"])

    unbound = [c for c in calls if not binds_roles(c)]
    if unbound:
        out.append(f"{len(unbound)} queries_for_run call(s) in main() bind no "
                   "expected_roles (absent, None, or empty), so the freeze is "
                   "not tied to the carriers and label space this run loads")
    full = [a for a in assigns
            if {k.arg for k in a.value.keywords} >= {"limit", "text_of"}]
    if len(full) != 1:
        out.append(f"{len(full)} assigned queries_for_run calls pass BOTH "
                   "limit and text_of, expected exactly 1; the missing check "
                   "and --limit would be computed nowhere, or twice")
    else:
        tgt = full[0].targets[0]
        names = ([e.id for e in tgt.elts if isinstance(e, ast.Name)]
                 if isinstance(tgt, ast.Tuple) else [])
        if not names or names[0] != "queries_by_seed":
            out.append(
                f"the limit/text_of call assigns to {names or ast.dump(tgt)[:40]}, "
                "not queries_by_seed; the forward loop would keep reading the "
                "census result and both the limit and the missing check would "
                "have been computed and thrown away")
        elif len(names) < 2:
            out.append("the limit/text_of call does not bind its faults at all")

    # THE FAULTS, CHECKED LOCALLY. "the name is Loaded somewhere in main()"
    # is not a dataflow check: both calls bind `_qfaults`, so deleting the
    # SECOND refusal block left the FIRST call's `if _qfaults:` satisfying it
    # and the gate green while a manifest/task disagreement went unread. Each
    # assignment must be followed IMMEDIATELY by a branch on its own faults.
    def follower(target):
        for node in ast.walk(fn):
            for field in ("body", "orelse", "finalbody"):
                blk = getattr(node, field, None)
                if not isinstance(blk, list):
                    continue
                for i, st in enumerate(blk):
                    if st is target:
                        return blk[i + 1] if i + 1 < len(blk) else None
        return None

    for a in assigns:
        tgt = a.targets[0]
        nm = ([e.id for e in tgt.elts if isinstance(e, ast.Name)]
              if isinstance(tgt, ast.Tuple) else [])
        if len(nm) < 2:
            continue                      # already reported above
        nxt = follower(a)
        # The test must be the BARE NAME. "the name appears in the test" is
        # satisfied by `if False and _qfaults:`, which never fires -- a
        # membership check on a syntax tree is not a check on the condition.
        tested = (isinstance(nxt, ast.If)
                  and isinstance(nxt.test, ast.Name)
                  and nxt.test.id == nm[1]
                  and nxt.body
                  and isinstance(nxt.body[0], (ast.Raise, ast.Return)))
        if not tested:
            out.append(
                f"the queries_for_run call binding {nm[1]!r} is not followed "
                f"immediately by `if {nm[1]}:` raising or returning; faults "
                "would be computed and left unread, and a sibling call's "
                "identical check would hide it")

    seen = False
    for node in ast.walk(fn):
        if not (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Name)
                and node.iter.id == "demo_seeds"):
            continue
        var = node.target.id
        for sub in ast.walk(node):
            if not (isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "queries_by_seed"):
                continue
            seen = True
            if not (isinstance(sub.slice, ast.Name) and sub.slice.id == var):
                out.append(
                    "the forward loop over demo_seeds indexes queries_by_seed "
                    f"with {ast.dump(sub.slice)[:40]} instead of its own loop "
                    f"variable {var!r}; every seed would be scored on one "
                    "seed's queries while being labelled with its own")
    if not seen:
        out.append("no `queries_by_seed[<loop var>]` inside a loop over "
                   "demo_seeds; the forward loop no longer reads per-seed "
                   "queries")
    return out


def _write(p, s):
    Path(p).write_text(s, encoding="utf-8")
    return p


def reseal(fbs, rec, *, status):
    """A copy with one baseline's status changed AND the decision hash redone.

    validate_freeze re-derives decision_sha256 from the record's own baselines
    block, so editing a status and leaving the stored hash alone is -- quite
    correctly -- reported as a file edited after it was written. A fixture
    that wants to say something about an unfinished OPPONENT has to hand over
    a self-consistent record, or it is testing tamper detection by accident.
    """
    out = json.loads(json.dumps(rec))
    out["baselines"][sorted(out["baselines"])[0]]["status"] = status
    out["decision_sha256"] = fbs._decision_hash(out, sorted(out["baselines"]))
    return out


def main() -> int:
    ok = True
    # NOT a skip. An earlier version let the real upstream checkouts decide:
    # where git refused them for ownership, every lock-opening, test_seed and
    # test_common arm stood down and the suite still exited 0 -- fifteen PASS
    # lines and a green light while the arms that matter never ran. A gate
    # that stands down when the environment is inconvenient is worse than no
    # gate, because it still reports.
    from tools.baselines.freeze_baseline_spec import install_fixture_mocks
    install_fixture_mocks()

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {detail}" if detail else ""))
        ok = ok and bool(cond)

    def refused(fn_, name, want):
        try:
            fn_()
            check(name, False, "SILENTLY SUCCEEDED -- the lock opened")
        except PermissionError as e:
            check(name, want in str(e), f"{str(e)[:64]}")

    with tempfile.TemporaryDirectory() as td:
        man = write_manifest(td)

        print("1. validation pairs each seed with ITS OWN manifest rows")
        by, faults = queries_for_run(man, "validation", SEEDS)
        check("no faults on a well-formed manifest", faults == [],
              "; ".join(faults)[:70])
        raw = json.loads(man.read_text(encoding="utf-8"))["validation_by_seed"]
        check("each seed's rows equal the manifest's rows for THAT seed",
              all([q["query_id"] for q in by[sd]]
                  == [q["query_id"] for q in raw[str(sd)]] for sd in SEEDS),
              "positional, not set-equality: a permuted loop keeps the three "
              "sets distinct while labelling every output with the wrong seed")
        check("...and every seed is present", sorted(by) == SEEDS,
              f"{sorted(by)}")

        # THE FAULT ITSELF, driven. The two arms above compare the result to
        # the manifest a second time in the fixture, so they stay green when
        # the production check is deleted -- they prove the loader is right,
        # not that queries_for_run would notice a wrong one. A WRONG LOADER is
        # installed here instead: the seam is legitimate because a defective
        # load_split is exactly the failure this check exists for, and only
        # the loader is replaced, never the logic under test.
        import tools.build_query_manifest as _bqm
        real = _bqm.load_split

        def permuting(mp, sp, *, demo_seed=None, **kw):
            nxt = SEEDS[(SEEDS.index(demo_seed) + 1) % len(SEEDS)]
            return real(mp, sp, demo_seed=nxt, **kw)

        _bqm.load_split = permuting
        try:
            _bp, fp = queries_for_run(man, "validation", SEEDS)
        finally:
            _bqm.load_split = real
        check("a permuted loader IS a fault, and the fault names the owner",
              bool(fp) and all("first difference at" in x for x in fp)
              and any("permuted" in x for x in fp),
              "; ".join(fp)[:80] or "SILENTLY CLEAN -- three distinct sets, "
              "every one labelled with the wrong seed")

        def shifted(mp, sp, *, demo_seed=None, **kw):
            return list(reversed(real(mp, sp, demo_seed=demo_seed, **kw)))

        _bqm.load_split = shifted
        try:
            _bs, fs = queries_for_run(man, "validation", SEEDS)
        finally:
            _bqm.load_split = real
        check("...and so is the right set in the wrong ORDER",
              bool(fs) and all("first difference at" in x for x in fs),
              "; ".join(fs)[:70] or "SILENTLY CLEAN -- set-equality would "
              "pass this, and the rows are zipped with logits by position")

        print("\n2. test_seed is per seed, and runs ONE seed at a time")
        # probe validate() refuses --split test_seed with more than one seed,
        # so this, not the three-seed shape, is what production calls.
        frz = write_freeze(td, man)
        one, f1 = queries_for_run(man, "test_seed", [43],
                                  freeze_manifest=str(frz))
        want = json.loads(man.read_text(encoding="utf-8"))["test_by_seed"]
        check("a one-seed run gets that seed's own 250-equivalent",
              f1 == [] and [q["query_id"] for q in one[43]] == want["43"],
              "; ".join(f1)[:70])
        # load_split reads test_by_seed first and refuses, which is why
        # queries_for_run carries no "no such seed" branch of its own --
        # it would be dead code.
        try:
            queries_for_run(man, "test_seed", [99], freeze_manifest=str(frz))
            check("a seed the manifest lacks is refused", False,
                  "SILENTLY SUCCEEDED")
        except ValueError as e:
            check("a seed the manifest lacks is refused by load_split",
                  "no draw recorded" in str(e), str(e)[:56])

        print("\n3. test_common is the WHOLE pool, identical under each seed")
        byc, fc = queries_for_run(man, "test_common", SEEDS,
                                  freeze_manifest=str(frz))
        pool = [e["query_id"] for e in
                json.loads(man.read_text(encoding="utf-8"))["entries"]
                if e.get("role") == "pool"]
        check("the three seeds get the SAME queries", fc == []
              and len({tuple(q["query_id"] for q in byc[sd])
                       for sd in SEEDS}) == 1,
              "load_split ignores demo_seed here; the L2 cross-seed "
              "comparison depends on the queries being fixed")
        check("...and that shared set IS the manifest's pool, in order",
              all([q["query_id"] for q in byc[sd]] == pool for sd in SEEDS),
              f"{len(pool)} pool rows; 'the three agree' is satisfied by three "
              "EMPTY lists and by three copies of the wrong set")
        check("...so the per-seed pairing rule is NOT applied to it",
              not any("seed 43:" in x for x in fc),
              "an arm that treated all splits alike would enforce the wrong "
              "contract on this one")

        def empty(mp, sp, *, demo_seed=None, **kw):
            return []

        _bqm.load_split = empty
        try:
            _be, fe = queries_for_run(man, "test_common", SEEDS,
                                      freeze_manifest=str(frz))
        finally:
            _bqm.load_split = real
        check("...and three EMPTY lists are a fault, not agreement",
              bool(fe) and any("pool rows" in x for x in fe),
              "; ".join(fe)[:70] or "SILENTLY CLEAN")

        print("\n4. the section 2.1 test lock")
        refused(lambda: queries_for_run(man, "test_common", SEEDS),
                "no freeze manifest at all", "is locked")
        # THE HOLE THIS ROUND FOUND. A self-signed JSON registering only the
        # query manifest -- correct hash, non-empty 'files' -- satisfied the
        # old drift-only check and opened the lock before any freeze existed.
        selfsigned = Path(td) / "selfsigned.json"
        selfsigned.write_text(json.dumps({"files": {
            str(man): hashlib.sha256(man.read_bytes()).hexdigest()}}),
            encoding="utf-8")
        refused(lambda: queries_for_run(man, "test_common", SEEDS,
                                        freeze_manifest=str(selfsigned)),
                "a self-signed one-file JSON with a CORRECT hash",
                "not a freeze")
        refused(lambda: queries_for_run(
            man, "test_common", SEEDS,
            freeze_manifest=str(write_freeze(td, man, kind=False))),
            "a role-complete manifest that does not declare its kind",
            "expected")
        refused(lambda: queries_for_run(
            man, "test_common", SEEDS,
            freeze_manifest=str(write_freeze(td, man, drop_role="carriers"))),
            "a manifest that registers no carriers role", "carriers")
        refused(lambda: queries_for_run(
            man, "test_common", SEEDS,
            freeze_manifest=str(write_freeze(td, man, stage="spec"))),
            "a SPEC-stage freeze (analysis code not yet frozen)", "code")

        # BINDING. A perfectly valid freeze built for manifest A opened
        # manifest B, which was not in `files` at all -- the lock checked the
        # freeze in a vacuum and never asked what this run loads.
        other = write_manifest(Path(td) / "otherdir")
        refused(lambda: queries_for_run(other, "test_common", SEEDS,
                                        freeze_manifest=str(frz)),
                "a valid freeze for a DIFFERENT query manifest", "not evidence")
        wrong_car = _write(Path(td) / "other_carriers.json", '{"by_layer": {}}')
        refused(lambda: queries_for_run(
            man, "test_common", SEEDS, freeze_manifest=str(frz),
            expected_roles={"carriers": str(wrong_car)}),
            "carriers the run loads that the freeze does not pin",
            "not evidence")
        ok_roles = queries_for_run(
            man, "test_common", SEEDS, freeze_manifest=str(frz),
            expected_roles={"carriers": str(Path(td) / "carriers.json")})[1]
        check("...and binding the carriers the freeze DOES pin still opens",
              ok_roles == [], "; ".join(ok_roles)[:70])

        # DRIFT, on the groups still compared. Source files are no longer
        # hashed -- the commit covers them -- but the DOCUMENTS group is, and
        # a prereg section changing after the freeze is exactly the kind of
        # thing that must re-lock: the run would be executing a design the
        # frozen text no longer describes.
        edited = json.loads(Path(frz).read_text(encoding="utf-8"))
        spec_p = Path(edited["roles"]["spec_freeze"])
        rec = json.loads(spec_p.read_text(encoding="utf-8"))
        key = sorted(rec["documents"])[0]
        rec["documents"][key] = "0" * 64
        spec_p.write_text(json.dumps(rec), encoding="utf-8")
        edited["files"][str(spec_p)] = hashlib.sha256(
            spec_p.read_bytes()).hexdigest()
        drifted = _write(Path(td) / "drifted.json", json.dumps(edited))
        refused(lambda: queries_for_run(man, "test_common", SEEDS,
                                        freeze_manifest=str(drifted)),
                f"a freeze whose recorded hash for {key} no longer matches "
                "the checkout", "no longer matches")

        refused(lambda: queries_for_run(
            man, "test_common", SEEDS,
            freeze_manifest=str(write_freeze(td, man, drop_role="gammas"))),
            "a manifest that registers no gammas role", "gammas")

        print("\n4b. the fault contract: none of these may RAISE")
        broken = _write(Path(td) / "broken.json", "{not json")
        refused(lambda: queries_for_run(man, "test_common", SEEDS,
                                        freeze_manifest=str(broken)),
                "a truncated freeze JSON (not a JSONDecodeError)", "unreadable")
        # VALID JSON, still fatal before: `roles` as a string raised
        # AttributeError out of .get, and a registered directory raised out of
        # read_bytes. The previous round's guard only covered unparseable JSON.
        d = json.loads(Path(frz).read_text(encoding="utf-8"))
        d["roles"] = "not-an-object"
        refused(lambda: queries_for_run(
            man, "test_common", SEEDS,
            freeze_manifest=str(_write(Path(td) / "strroles.json",
                                       json.dumps(d)))),
            "roles as a string (valid JSON, was an AttributeError)",
            "expected an object")
        d2 = json.loads(Path(frz).read_text(encoding="utf-8"))
        adir = Path(td) / "adir"
        adir.mkdir(exist_ok=True)
        d2["files"][str(adir)] = "0" * 64
        refused(lambda: queries_for_run(
            man, "test_common", SEEDS,
            freeze_manifest=str(_write(Path(td) / "dirfile.json",
                                       json.dumps(d2)))),
            "a registered path that is a DIRECTORY (was an OSError)",
            "not a regular file")

        print("\n4bb. the gamma artifact is READ, not merely hashed")
        # A hash is not a constraint: while nobody parsed this file, writing
        # `not-json` into it left freeze_manifest_blockers returning [].
        badgam = Path(td) / "badgamma"
        badgam.mkdir(exist_ok=True)
        (badgam / "gamma_star.json").write_text("not-json", encoding="utf-8")
        bg = write_manifest(badgam)
        refused(lambda: queries_for_run(bg, "test_common", SEEDS,
                                        freeze_manifest=str(
                                            write_freeze(badgam, bg))),
                "a gamma artifact that is not JSON at all", "unreadable as")
        halfgam = Path(td) / "halfgamma"
        halfgam.mkdir(exist_ok=True)
        (halfgam / "gamma_star.json").write_text(
            json.dumps({"gamma_L2": 0.5}), encoding="utf-8")
        hg = write_manifest(halfgam)
        refused(lambda: queries_for_run(hg, "test_common", SEEDS,
                                        freeze_manifest=str(
                                            write_freeze(halfgam, hg))),
                "a gamma artifact missing gamma_group and gamma_head",
                "expected a number")

        print("\n4c. what a code-stage spec freeze must actually record")
        import tools.baselines.freeze_baseline_spec as fbs
        relabelled = {**fbs.build_freeze("spec"), "stage": "code",
                      "artifacts": {"results/x.npz": "ab" * 32}}
        # THE SPECIFIC BLOCKER, not merely "some blocker". A relabelled spec
        # has no `repo` block, so the DIRTY-tree branch also fires on it --
        # and while this arm only asked for bool(), deleting the
        # no-commit check left it green on the other branch's message.
        check("a spec freeze RELABELLED 'code' is caught for having no commit",
              any("no repository commit" in b
                  for b in fbs.validate_freeze(relabelled)),
              "; ".join(fbs.validate_freeze(relabelled))[:70])
        code = fbs.build_freeze("code")
        check("...and a real one pins the code by COMMIT, not by file hashes",
              code.get("repo", {}).get("head") and "code" not in code,
              "a hand-written list missed the confirmatory test the first "
              "time and prototype_targets the second; the import closure that "
              "replaced it pinned 19 files, 12 of them shared with other work")
        dirty = json.loads(json.dumps(code))
        dirty["repo"]["clean"] = False
        check("a commit recorded from a DIRTY tree is refused",
              any("DIRTY" in b for b in fbs.validate_freeze(dirty)),
              "it does not identify the code that ran, which is the only "
              "thing the commit is there to do")
        # THE OTHER HALF, and the one that actually reaches the paper: every
        # GPU artifact must carry the commit that produced it. The freeze says
        # what was frozen; run_provenance says what ran, and only the second
        # survives someone re-freezing later.
        from tools.icl_common import run_provenance
        prov = run_provenance()
        check("every result carries the commit that produced it",
              len(str(prov.get("repo_commit", ""))) == 40
              and "repo_dirty" in prov,
              f"{str(prov.get('repo_commit'))[:12]}... dirty="
              f"{prov.get('repo_dirty')}; a job id says WHICH run, a commit "
              "says WHICH CODE, and only the second reproduces a number")
        from tools.build_query_manifest import attribution_blockers
        check("...and a run that cannot be attributed to one is refused "
              "before test",
              (attribution_blockers() == []) == (prov.get("repo_dirty")
                                                 == "false"),
              "checked against this tree's actual state, so the arm is not "
              "vacuous in either direction")
        # TWO DIFFERENT CLAIMS, TWO DIFFERENT CHECKS. Section 2.1's condition
        # on reading test is "the three gamma*, the control carriers/positions
        # and the analysis code are frozen" -- it says nothing about the six
        # opponents being implemented. An earlier version re-executed the
        # writer's full readiness contract here, which made writing five
        # baseline runners a prerequisite for reading Method A's own test
        # split (section 14.1's C12 friction, in a heavier form).
        halfdone = reseal(fbs, code, status="NOT-IMPLEMENTED")
        check("an unimplemented opponent does NOT block Method A's test read",
              fbs.validate_freeze(halfdone,
                                  require_baselines_ready=False) == [],
              "section 2.1 asks about gamma*, carriers and the analysis code")
        check("...but it DOES block the claims that are about the opponents",
              any("not ready" in b for b in fbs.validate_freeze(halfdone)),
              "H6 and the viability gate keep the full check")
        print("\n4d. the freeze is for REPRODUCIBILITY, not fairness")
        # The registry changing does not make an existing result less
        # reproducible: the record still says exactly what it said, and a run
        # that does not read the opponent configurations is not using it. An
        # earlier version blocked here, which imported a fairness argument
        # into a provenance check -- and one it could not deliver anyway,
        # since anyone can edit and re-freeze. What proves nobody tuned an
        # opponent after seeing a result is a dated record, not a gate.
        import copy as _copy

        import tools.baselines.registry as _reg
        before = _copy.deepcopy(_reg.BASELINES)
        try:
            _reg.BASELINES["new_opponent"] = _copy.deepcopy(
                _reg.BASELINES["inductive_bc"])
            _reg.BASELINES.pop("inductive_bc")
            _reg.BASELINES["zerotuning_level1"]["configs"] = [{"k": 1}]
            cur = fbs.build_freeze("code")
            check("one added, one dropped, one altered: Method A's test read "
                  "is unaffected",
                  fbs.compare(code, cur, include_baselines=False) == []
                  and fbs.validate_freeze(
                      code, require_baselines_ready=False) == [],
                  "this run reads no opponent configuration, so none of that "
                  "changes what produced its numbers")
            check("...while the H6 path, whose runs DO read them, still sees "
                  "every one of the three",
                  len([b for b in fbs.compare(code, cur)
                       if "baselines/" in b or "decision" in b]) >= 2,
                  "there the opponent config IS part of the provenance")
            # AND THE LOCK ITSELF, with the registry in that state. The two
            # arms above call validate_freeze and compare directly, so
            # re-tightening the LOCK's call sites leaves them green -- the
            # same "the fixture re-does the comparison instead of driving the
            # check" mistake as sections 1-2. This drives queries_for_run.
            lockfaults = queries_for_run(man, "test_common", SEEDS,
                                         freeze_manifest=str(frz))[1]
            check("...and the LOCK opens with all three changes standing",
                  lockfaults == [], "; ".join(lockfaults)[:70])
        finally:
            _reg.BASELINES.clear()
            _reg.BASELINES.update(before)

        # A freeze whose RECORD says an opponent is unfinished. Section 2.1's
        # condition is about gamma*, carriers and the analysis code, so this
        # must not close Method A's test. Driven through the lock, because
        # that is where require_baselines_ready is passed.
        half = reseal(fbs, fbs.build_freeze("code"), status="NOT-IMPLEMENTED")
        hz = write_freeze(td, man, spec_rec=half, tag="_half")
        hf = queries_for_run(man, "test_common", SEEDS,
                             freeze_manifest=str(hz))[1]
        check("a freeze RECORDING an unfinished opponent still opens the lock",
              hf == [], "; ".join(hf)[:70])

        # ...and one taken at a DIFFERENT commit from this checkout. Not
        # refused: the result carries its own commit through run_provenance,
        # so it stays attributable, and refusing would re-create the friction
        # that dropping per-file hashes exists to remove.
        elsewhere = json.loads(json.dumps(fbs.build_freeze("code")))
        elsewhere["repo"]["head"] = "0" * 40
        ez = write_freeze(td, man, spec_rec=elsewhere, tag="_elsewhere")
        ef = queries_for_run(man, "test_common", SEEDS,
                             freeze_manifest=str(ez))[1]
        check("a freeze taken at ANOTHER commit still opens the lock",
              ef == [], "; ".join(ef)[:70]
              or "the result records the commit that produced it, which is "
              "what reproducing it needs")

        illtyped = json.loads(json.dumps(code))
        illtyped["baselines"][sorted(illtyped["baselines"])[0]] = "READY"
        check("a baseline entry that is a STRING is a blocker, not an "
              "AttributeError",
              any("not objects" in b for b in fbs.validate_freeze(illtyped)),
              "valid JSON, and it raised straight past every caller that "
              "catches PermissionError")
        notready = json.loads(json.dumps(code))
        notready["baselines"][sorted(notready["baselines"])[0]]["status"] = \
            "NOT-IMPLEMENTED"
        check("...and one recording a NOT-READY baseline is refused",
              any("not ready" in b for b in fbs.validate_freeze(notready)),
              "the fixture mocks the registry READY, so without this arm the "
              "condition would be mocked away rather than tested")

        print("\n5. --limit reaches the mapping the loop reads")
        byl, _fl = queries_for_run(man, "validation", SEEDS, limit=3)
        check("every seed is truncated, not just one",
              all(len(byl[sd]) == 3 for sd in SEEDS),
              f"{ {sd: len(byl[sd]) for sd in SEEDS} }; slicing an alias left "
              "the limit with no effect at all")
        check("...and limit=None leaves the sets whole",
              all(len(by[sd]) == NC * PER for sd in SEEDS))

        print("\n6. the missing-query check covers EVERY seed")
        text42 = {q["query_id"]: "x" for q in by[42]}
        _by3, f3 = queries_for_run(man, "validation", SEEDS, text_of=text42)
        check("a seed whose queries are absent from the task is caught",
              bool(f3) and any("not found in either split" in x for x in f3),
              "; ".join(f3)[:70] or "SILENTLY CLEAN")
        alltext = {q["query_id"]: "x" for sd in SEEDS for q in by[sd]}
        _by4, f4 = queries_for_run(man, "validation", SEEDS, text_of=alltext)
        check("...and a complete lookup passes", f4 == [], "; ".join(f4)[:70])

    print("\n7. main()'s wiring to it (AST, not an end-to-end run)")
    w = main_wiring_faults()
    check("main() forwards limit and text_of, and the forward loop indexes "
          "by its own seed", w == [], "; ".join(w)[:90])

    # ------------------------------------------------------------------
    print("\n8. which frozen gamma* a test run must use (5(4))")
    from tools.probe_prototype_shrinkage import (CARRIER_IMPLS,
                                                 gamma_key_for)
    # 5(4): "L2 raw LOO、L3.1 Method A-group 与 Method A-head 各自的完整
    # gamma 网格 ... 每个 setting/implementation 独立" -- all three run RAW
    # LOO and the setting is the write implementation. This was keyed on ARM
    # NAME, with the keys guessed from the words: proto_global got
    # gamma_group and matched_heads got gamma_head, while the MAIN method
    # proto_loo got gamma_L2. On the test path that would have checked the
    # L3.1 group run against another setting's number, and nothing tested it.
    check("the L3.1 group run takes gamma_group, not gamma_L2",
          gamma_key_for("proto_loo", "gqa_group_v")[0] == "gamma_group",
          "proto_loo IS the main method (4's table: 主方法); the group in "
          "gamma*_group is the KV group, not an arm whose name says 'global'")
    check("the L3.1 head run takes gamma_head",
          gamma_key_for("proto_loo", "gqa_head_realized")[0] == "gamma_head")
    check("the L2 run takes gamma_L2",
          gamma_key_for("proto_loo", "mha_v")[0] == "gamma_L2")
    check("a CONTROL arm takes its SETTING's gamma, not one named after it",
          gamma_key_for("proto_global", "gqa_group_v")[0] == "gamma_group"
          and gamma_key_for("matched_heads", "mha_v")[0] == "gamma_L2",
          "matched_heads run in L2 takes gamma_L2; the 'head' in gamma*_head "
          "is the per-head realised write, not this arm's name")
    check("every registered implementation has exactly one gamma*",
          {gamma_key_for("proto_loo", c)[0] for c in CARRIER_IMPLS}
          == {"gamma_L2", "gamma_group", "gamma_head"},
          "three implementations, three frozen numbers, one each")
    _k, _w = gamma_key_for("natural", "gqa_group_v")
    check("an arm that writes nothing has no key and no complaint",
          _k is None and _w is None,
          "its only defensible gamma is 0.0, which the caller supplies")
    check("a missing carrier implementation is REFUSED, not defaulted",
          gamma_key_for("proto_loo", None)[1] is not None,
          "without it the run cannot say which frozen number it claims")
    check("an unregistered arm is refused",
          gamma_key_for("not_an_arm", "mha_v")[1] is not None)

    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
