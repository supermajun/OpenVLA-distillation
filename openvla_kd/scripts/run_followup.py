"""Sequential teacher/precision/data experiments; never overlap GPU processes."""
import os
from pathlib import Path
import sys

from run_pilot import run_logged


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    root = Path("runs/followup")
    if root.exists():
        raise FileExistsError("Follow-up already exists; inspect logs before resuming individual commands")
    root.mkdir(parents=True)
    py = sys.executable
    def run(name, args):
        run_logged([py, *args], root / f"{name}.log")
    run("teacher_distinct8", ["-m", "vla_kd.teacher_evaluate", "--out", str(root / "teacher_distinct8"),
                              "--initial-indices", *map(str, range(8))])
    for size, pilot in (("500m", "pilot100"), ("1b", "pilot1b100")):
        run(size + "_baseline8", ["-m", "vla_kd.evaluate", "--policy", f"runs/{pilot}/both/policy.pt",
                                  "--episodes", "8", "--out", str(root / (size + "_baseline8"))])
    run("expanded_cache", ["-m", "vla_kd.teacher", "--data", "data/expanded", "--out", "data/expanded_teacher_cache"])
    run("resume_master", ["scripts/check_resume.py", "--student-size", "1b", "--optimizer", "master_fp32",
                          "--out", str(root / "resume_master")])
    for name, size, data, cache, steps in (
        ("1b_precision100", "1b", "data/pilot", "data/teacher_cache", 100),
        ("500m_expanded500", "500m", "data/expanded", "data/expanded_teacher_cache", 500),
        ("1b_expanded500", "1b", "data/expanded", "data/expanded_teacher_cache", 500),
    ):
        run(name, ["-m", "vla_kd.train", "--strategy", "both", "--student-size", size,
                   "--optimizer", "master_fp32", "--data", data, "--cache", cache, "--steps", str(steps),
                   "--out", str(root / name)])
        run(name + "-evaluation", ["-m", "vla_kd.evaluate", "--policy", str(root / name / "policy.pt"),
                                    "--data", data, "--episodes", "8", "--out", str(root / (name + "-evaluation"))])
    print("Follow-up experiment sequence complete", flush=True)


if __name__ == "__main__":
    main()
