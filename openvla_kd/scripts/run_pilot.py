"""Run equal-budget conditions sequentially, preserving logs and real rollout results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run_logged(command, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
    with path.open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=env)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", default="runs/pilot100")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--student-size", choices=["500m", "1b", "2b"], default="500m")
    args = p.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Use a new output directory; existing experiments are never overwritten")
    out.mkdir(parents=True)
    run_logged([sys.executable, "-m", "pytest", "-q", f"--junitxml={out / 'tests.xml'}"], out / "tests.log")
    results = []
    for strategy in ("demo", "action", "feature", "both"):
        run_dir = out / strategy
        cmd = [sys.executable, "-m", "vla_kd.train", "--strategy", strategy, "--steps", str(args.steps),
               "--seed", str(args.seed), "--out", str(run_dir), "--student-size", args.student_size]
        if strategy != "demo":
            cmd += ["--cache", "data/teacher_cache"]
        run_logged(cmd, out / f"{strategy}-train.log")
        eval_dir = out / f"{strategy}-evaluation"
        run_logged([sys.executable, "-m", "vla_kd.evaluate", "--policy", str(run_dir / "policy.pt"),
                    "--out", str(eval_dir), "--episodes", str(args.episodes), "--max-steps", str(args.max_steps),
                    "--seed", str(args.seed)], out / f"{strategy}-evaluation.log")
        results.append({"training": json.loads((run_dir / "summary.json").read_text()),
                        "evaluation": json.loads((eval_dir / "evaluation.json").read_text())})
        (out / "comparison.json").write_text(json.dumps(results, indent=2))
    print(f"Completed four-condition pilot: {out / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
