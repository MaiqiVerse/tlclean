"""SPEC_FREEZE=none: the freeze manifest a checkout without the baseline
spec documents can write (the public tree), and what the lock does with it.

  * write_freeze_manifest accepts the literal "none" for the spec_freeze
    role, hashes nothing for it, and still requires every other role to be
    a real file;
  * freeze_manifest_blockers returns exactly ONE blocker for it, carrying
    the sentence the UNSAFE bypass keys on -- so without the bypass the lock
    stays shut, and with it that one blocker is discarded and recorded while
    every other check (roles, hashes, the gammas) is kept.

No model, no GPU.

    python tools/test_spec_freeze_none.py
"""
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tools.build_query_manifest as bqm  # noqa: E402
from tools.build_query_manifest import (NO_SPEC_FREEZE, freeze_manifest_blockers,  # noqa: E402
                                        write_freeze_manifest)
from tools.test_probe_queries import SEEDS, write_manifest  # noqa: E402
from tools.UNSAFE_run_test_forward import SPEC_STAGE_MARKER, install_bypass  # noqa: E402


def fixture(td):
    man = write_manifest(td)
    car = pathlib.Path(td) / "carriers.json"
    car.write_text(json.dumps({"by_layer": {}}), encoding="utf-8")
    lsp = pathlib.Path(td) / "label_space.json"
    lsp.write_text(json.dumps({"eligible_classes": []}), encoding="utf-8")
    gam = pathlib.Path(td) / "gamma_star.json"
    gam.write_text(json.dumps({"gamma_L2": 0.5, "gamma_group": 0.5, "gamma_head": 0.25}),
                   encoding="utf-8")
    return man, car, lsp, gam


def test_manifest_records_the_literal_and_hashes_nothing_for_it():
    with tempfile.TemporaryDirectory() as td:
        man, car, lsp, gam = fixture(td)
        out = pathlib.Path(td) / "freeze.json"
        doc = write_freeze_manifest(out, query_manifest=man, carriers=car, label_space=lsp,
                                    gammas=gam, spec_freeze=NO_SPEC_FREEZE)
        assert doc["roles"]["spec_freeze"] == NO_SPEC_FREEZE
        assert NO_SPEC_FREEZE not in doc["files"]
        assert set(doc["files"]) == {str(man), str(car), str(lsp), str(gam)}


def test_every_other_role_is_still_a_real_file():
    with tempfile.TemporaryDirectory() as td:
        man, car, lsp, gam = fixture(td)
        gam.unlink()
        out = pathlib.Path(td) / "freeze.json"
        try:
            write_freeze_manifest(out, query_manifest=man, carriers=car, label_space=lsp,
                                  gammas=gam, spec_freeze=NO_SPEC_FREEZE)
        except FileNotFoundError as e:
            assert "gammas" in str(e)
        else:
            raise AssertionError("a missing gammas file was accepted beside SPEC_FREEZE=none")


def test_one_blocker_with_the_bypass_sentence_and_nothing_else():
    with tempfile.TemporaryDirectory() as td:
        man, car, lsp, gam = fixture(td)
        out = pathlib.Path(td) / "freeze.json"
        write_freeze_manifest(out, query_manifest=man, carriers=car, label_space=lsp,
                              gammas=gam, spec_freeze=NO_SPEC_FREEZE)
        bad = freeze_manifest_blockers(str(out))
        assert len(bad) == 1, bad
        assert SPEC_STAGE_MARKER in bad[0] and "SPEC_FREEZE=none" in bad[0], bad[0]
        # the lock is shut without the bypass
        seed = SEEDS[1]
        try:
            bqm.load_split(str(man), "test_seed", demo_seed=seed, freeze_manifest=str(out))
        except PermissionError as e:
            assert "stage freeze" in str(e), str(e)[:160]
        else:
            raise AssertionError("SPEC_FREEZE=none opened the test lock without the bypass")
        # with the bypass that one blocker is discarded and recorded; the rows come back
        record = {"calls": []}
        real = bqm.freeze_manifest_blockers          # install_bypass returns nothing
        install_bypass(record)
        try:
            rows = bqm.load_split(str(man), "test_seed", demo_seed=seed, freeze_manifest=str(out))
        finally:
            bqm.freeze_manifest_blockers = real
        assert rows, "no test rows came back through the bypass"
        assert record["calls"] and len(record["calls"][0]["blockers_discarded"]) == 1
        assert "SPEC_FREEZE=none" in record["calls"][0]["blockers_discarded"][0]
        assert not record["calls"][0]["blockers_kept"]


def test_a_broken_gamma_file_is_still_refused_beside_none():
    with tempfile.TemporaryDirectory() as td:
        man, car, lsp, gam = fixture(td)
        out = pathlib.Path(td) / "freeze.json"
        write_freeze_manifest(out, query_manifest=man, carriers=car, label_space=lsp,
                              gammas=gam, spec_freeze=NO_SPEC_FREEZE)
        gam.write_text("not-json", encoding="utf-8")       # drift after the freeze
        bad = freeze_manifest_blockers(str(out))
        assert bad and not any(SPEC_STAGE_MARKER in b for b in bad), bad


if __name__ == "__main__":
    test_manifest_records_the_literal_and_hashes_nothing_for_it()
    test_every_other_role_is_still_a_real_file()
    test_one_blocker_with_the_bypass_sentence_and_nothing_else()
    test_a_broken_gamma_file_is_still_refused_beside_none()
    print("SPEC_FREEZE=none: 4 tests passed")
