"""Run the matched BF16 capacity experiment after a real 2B resume test passes."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_pilot import run_logged
from vla_kd.runtime import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/capacity2b_20261001")
    args = p.parse_args()
    os.chdir(ROOT)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    probe = json.loads((root / "resume_check/verification.json").read_text())
    if not (probe["passed"] and probe["student_size"] == "2b" and probe["optimizer"] == "bf16"
            and probe["exact_model_optimizer_rng_and_losses"]):
        raise ValueError("A successful actual 2B BF16 resume test is required")
    # Refuse accidental duplicate launches or overwrites, including partial runs.
    if (root / "pipeline.json").exists():
        raise FileExistsError("Inspect the existing pipeline status before resuming individual stages")
    files = [*Path("vla_kd").glob("*.py"), *Path("scripts").glob("*.py"), *Path("tests").glob("*.py")]
    state = {"status": "running", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
             "completed_stages": [], "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    def run(name, command):
        state.update(current_stage=name, command=command, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(root / "pipeline.json", state)
        run_logged([sys.executable, *command], root / f"{name}.log")
        state["completed_stages"].append(name)
        atomic_json(root / "pipeline.json", state)
    try:
        for size in ("2b", "500m", "1b"):
            name = size + "_bf16_500"
            folder = root / name
            run(name, ["-m", "vla_kd.train", "--strategy", "both", "--student-size", size,
                       "--optimizer", "bf16", "--data", "data/expanded", "--cache", "data/expanded_teacher_cache",
                       "--steps", "500", "--seed", "17", "--out", str(folder)])
            run(name + "-evaluation", ["-m", "vla_kd.evaluate", "--policy", str(folder / "policy.pt"),
                "--data", "data/expanded", "--episodes", "8", "--max-steps", "220", "--seed", "17",
                "--out", str(root / (name + "-evaluation"))])
        run("audit", ["scripts/verify_capacity2b.py", "--out", str(root)])
        state.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), finished_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        atomic_json(root / "pipeline.json", state)


if __name__ == "__main__":
    main()
