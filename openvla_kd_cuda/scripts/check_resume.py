"""Compare real pretrained training: uninterrupted vs checkpoint resume on one backend."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

from run_pilot import run_logged


def cleanup_verified_weights(out):
    """Remove only this successful short test's six large weight files."""
    out = Path(out).resolve()
    verification = out / "verification.json"
    result = json.loads(verification.read_text())
    if result.get("passed") is not True:
        raise ValueError("Do not clean up a failed or unverified resume check")
    targets = [out / run / name for run in ("continuous", "first", "resumed")
               for name in ("checkpoint.pt", "policy.pt")]
    for p in targets:
        if p.resolve().parent.parent != out or p.is_symlink():
            raise ValueError(f"Unsafe cleanup path: {p}")
    removed = []
    for p in targets:
        if p.is_file():
            p.unlink()
            removed.append(str(p.relative_to(out)))
    result["removed_short_test_weights"] = removed
    verification.write_text(json.dumps(result, indent=2))
    return removed


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
    p.add_argument("--student-size", choices=["500m", "1b", "2b"], default="2b")
    p.add_argument("--optimizer", choices=["bf16", "master_fp32"], default="master_fp32")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    p.add_argument("--data", default="data/expanded")
    p.add_argument("--cache", default="data/expanded_teacher_cache")
    p.add_argument("--cleanup-on-success", action="store_true",
                   help="After verification, remove only this short test's checkpoint/policy files; retain logs and verdict")
    args = p.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Use a new directory for resume verification")
    out.mkdir(parents=True)
    common = [sys.executable, "-m", "vla_kd.train", "--strategy", "both", "--cache", args.cache, "--data", args.data,
              "--student-size", args.student_size, "--optimizer", args.optimizer, "--device", args.device,
              "--deterministic", "--validate-every", "2", "--checkpoint-every", "0"]
    for name, steps, extra in (("continuous", 5, []), ("first", 3, []),
                               ("resumed", 2, ["--resume", str(out / "first/checkpoint.pt")])):
        run_logged(common + ["--steps", str(steps), "--out", str(out / name)] + extra, out / f"{name}.log")
    a = torch.load(out / "continuous/checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    b = torch.load(out / "resumed/checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    keys = ["model", "optimizer", "step", "rng_cpu"]
    keys += {"cuda": ["rng_cuda"], "mps": ["rng_mps"], "cpu": []}[args.device]
    counts = {key: compare(a[key], b[key], key) for key in keys}
    uninterrupted = [json.loads(x) for x in (out / "continuous/metrics.jsonl").read_text().splitlines()]
    resumed = [json.loads(x) for x in (out / "resumed/metrics.jsonl").read_text().splitlines()]
    assert len(uninterrupted) == 5 and len(resumed) == 2
    for expected, actual in zip(uninterrupted[3:], resumed):
        compare({k: v for k, v in expected.items() if k != "seconds"},
                {k: v for k, v in actual.items() if k != "seconds"}, "metrics")
    result = {"passed": True, "device": args.device, "dtype": "bf16", "strategy": "both",
              "student_size": args.student_size, "optimizer": args.optimizer,
              "continuous_steps": 5, "resumed_steps": [3, 2], "exact_model_optimizer_rng_and_losses": True,
              "compared_tensor_elements": counts}
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    if args.cleanup_on_success:
        del a, b
        result["removed_short_test_weights"] = cleanup_verified_weights(out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
