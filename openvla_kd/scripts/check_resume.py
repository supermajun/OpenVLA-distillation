"""Compare actual pretrained MPS BF16 training: uninterrupted vs checkpoint resume."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

from run_pilot import run_logged


def compare(a, b, location="checkpoint"):
    if isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise AssertionError(f"Resume differs at {location}")
        return a.numel()
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"Keys differ at {location}")
        return sum(compare(a[k], b[k], f"{location}.{k}") for k in a)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"Length differs at {location}")
        return sum(compare(x, y, f"{location}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
    if a != b:
        raise AssertionError(f"Value differs at {location}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/resume_check")
    p.add_argument("--student-size", choices=["500m", "1b", "2b"], default="500m")
    p.add_argument("--optimizer", choices=["bf16", "master_fp32"], default="bf16")
    args = p.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Use a new directory for resume verification")
    out.mkdir(parents=True)
    common = [sys.executable, "-m", "vla_kd.train", "--strategy", "both", "--cache", "data/teacher_cache",
              "--student-size", args.student_size, "--optimizer", args.optimizer]
    for name, steps, extra in (("continuous", 5, []), ("first", 3, []),
                               ("resumed", 2, ["--resume", str(out / "first/checkpoint.pt")])):
        run_logged(common + ["--steps", str(steps), "--out", str(out / name)] + extra, out / f"{name}.log")
    a = torch.load(out / "continuous/checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    b = torch.load(out / "resumed/checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    counts = {key: compare(a[key], b[key], key) for key in ("model", "optimizer", "step", "rng_cpu", "rng_mps")}
    uninterrupted = [json.loads(x) for x in (out / "continuous/metrics.jsonl").read_text().splitlines()]
    resumed = [json.loads(x) for x in (out / "resumed/metrics.jsonl").read_text().splitlines()]
    for expected, actual in zip(uninterrupted[3:], resumed):
        compare({k: v for k, v in expected.items() if k != "seconds"},
                {k: v for k, v in actual.items() if k != "seconds"}, "metrics")
    result = {"passed": True, "device": "mps", "dtype": "bf16", "strategy": "both",
              "student_size": args.student_size, "optimizer": args.optimizer,
              "continuous_steps": 5, "resumed_steps": [3, 2], "exact_model_optimizer_rng_and_losses": True,
              "compared_tensor_elements": counts}
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
