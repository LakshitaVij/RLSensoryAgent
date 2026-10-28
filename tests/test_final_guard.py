"""The held-out guard: nothing in development may open the held-out subjects, and final_eval refuses without confirmation.
These tests never open a held-out file."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import build_features
import export_subject
from env import WesadStressEnv

HELD = ["S2", "S8", "S10", "S11", "S17"]


def test_export_and_build_refuse_held_out_by_default():
    for sid in HELD:
        with pytest.raises(SystemExit):
            export_subject.export(sid)
        with pytest.raises(SystemExit):
            build_features.build(sid)


def test_environment_refuses_held_out_without_the_flag():
    for sid in HELD:
        with pytest.raises(ValueError):
            WesadStressEnv(subjects=[sid])


def _held_out_state():
    """What the held-out run has written so far (the real run happened once on 2026-10-23)."""
    folder = ROOT / "data" / "heldout"
    files = sorted(str(p) for p in folder.rglob("*")) if folder.exists() else []
    results = sorted(str(p) for p in (ROOT / "results").glob("final_eval_*.json"))
    return files, results


def test_final_eval_refuses_without_confirmation_and_creates_nothing():
    before = _held_out_state()
    r = subprocess.run([sys.executable, str(ROOT / "src" / "final_eval.py")], capture_output=True, text=True)
    assert r.returncode != 0 and "refusing" in r.stderr + r.stdout
    assert _held_out_state() == before


def test_prepare_script_refuses_without_confirmation():
    before = _held_out_state()
    r = subprocess.run([sys.executable, str(ROOT / "src" / "_prepare_heldout.py")], capture_output=True, text=True)
    assert r.returncode != 0
    assert _held_out_state() == before


def test_final_eval_refuses_a_second_run_even_with_confirmation():
    """Once a held-out result exists, the confirmed command must still refuse (it must not retrain or rescore)."""
    if not list((ROOT / "results").glob("final_eval_*.json")):
        pytest.skip("no held-out result exists yet; not running a confirmed held-out command from a test")
    before = _held_out_state()
    r = subprocess.run([sys.executable, str(ROOT / "src" / "final_eval.py"), "--confirm", "OPEN-HELD-OUT-ONCE"], capture_output=True, text=True)
    assert r.returncode != 0 and "already exists" in r.stderr + r.stdout
    assert _held_out_state() == before


def test_dev_code_never_references_the_held_out_data_folder():
    for name in ("env.py", "oracle.py", "loso.py", "eval_policy.py", "score_checkpoints.py", "check_baseline.py", "check_sam.py", "check_sweat.py"):
        assert "data/heldout" not in (ROOT / "src" / name).read_text() and "heldout" not in (ROOT / "src" / name).read_text().replace("held_out_ok", "")


def test_frozen_files_unchanged_since_the_freeze():
    """The code and split files frozen on 2026-10-23 must still match their fingerprints (frozen_files.json)."""
    import hashlib
    import json
    frozen = json.loads((ROOT / "frozen_files.json").read_text())["sha256"]
    changed = [f for f, h in frozen.items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    assert not changed, f"frozen files changed after the freeze: {changed}"
