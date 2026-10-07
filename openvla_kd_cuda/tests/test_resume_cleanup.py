import importlib.util
import json
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("resume_check", scripts / "check_resume.py")
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def test_cleanup_requires_pass_and_preserves_other_files(tmp_path):
    for run in ("continuous", "first", "resumed"):
        d = tmp_path / run
        d.mkdir()
        for name in ("checkpoint.pt", "policy.pt", "metrics.jsonl", "other.pt"):
            (d / name).write_text("keep until verified")
    verdict = tmp_path / "verification.json"
    verdict.write_text(json.dumps({"passed": False}))
    with pytest.raises(ValueError, match="unverified"):
        resume.cleanup_verified_weights(tmp_path)
    assert len(list(tmp_path.glob("*/*.pt"))) == 9
    verdict.write_text(json.dumps({"passed": True}))
    assert len(resume.cleanup_verified_weights(tmp_path)) == 6
    assert len(list(tmp_path.glob("*/*.pt"))) == 3
    assert len(list(tmp_path.glob("*/metrics.jsonl"))) == 3
    assert json.loads(verdict.read_text())["passed"] is True


def test_cleanup_rejects_directory_symlink(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (other / "checkpoint.pt").write_text("must survive")
    (out / "continuous").symlink_to(other, target_is_directory=True)
    (out / "verification.json").write_text(json.dumps({"passed": True}))
    with pytest.raises(ValueError, match="Unsafe"):
        resume.cleanup_verified_weights(out)
    assert (other / "checkpoint.pt").exists()
