"""Evaluate the frozen teacher using exactly the student's simulation interface."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import Dataset
from .evaluate import run_rollouts
from .runtime import MemoryMonitor, atomic_json, select_device
from .teacher import Teacher, load_teacher_stats


@torch.inference_mode()
def run(args):
    if args.episodes < 1 or args.max_steps < 1 or not 1 <= args.execute_chunk <= 8:
        raise ValueError("Invalid rollout limits")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Use a new output directory")
    out.mkdir(parents=True)
    device = select_device(args.device)
    ds = Dataset(args.data, "val")
    stats, key = ds.manifest["stats"], ds.manifest["stats_key"]
    assert stats == load_teacher_stats(args.model)[key]
    with MemoryMonitor(device) as monitor:
        teacher = Teacher(args.model, args.upstream, device)
        total, count = 0., 0
        times = []
        def predict(sample):
            begin = time.perf_counter()
            actions = teacher.predict(sample, key)["actions"]
            times.append(time.perf_counter() - begin)
            return actions
        for i in range(len(ds)):
            sample = ds[i]
            pred = predict(sample)
            total += np.abs(pred - sample["actions"])[sample["valid"]].sum()
            count += sample["valid"].sum() * 7
        rollouts = run_rollouts(args, predict, stats)
        result = {"policy": "frozen_openvla_oft", "config": vars(args),
                  "dataset_fingerprint": ds.manifest["fingerprint"],
                  "validation_l1": float(total / count), "rollouts": rollouts,
                  "successes": sum(r["success"] for r in rollouts),
                  "distinct_initial_states": len({r["initial_state_index"] for r in rollouts}),
                  "mean_predict_seconds_excluding_first": float(np.mean(times[1:])),
                  "timing_scope": "includes teacher preprocessing and feature extraction, excludes simulation",
                  "frozen": all(not p.requires_grad for m in (teacher.model, teacher.head, teacher.proprio) for p in m.parameters())}
    result["sampled_peak_memory"] = monitor.peaks
    atomic_json(out / "evaluation.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/teacher")
    p.add_argument("--upstream", default="vendor/openvla-oft")
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--device", default="mps")
    p.add_argument("--libero", default="vendor/LIBERO")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task", default="pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--initial-indices", type=int, nargs="+", default=[0, 1, 0, 1, 0, 1, 0, 1])
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--execute-chunk", type=int, default=8)
    p.add_argument("--camera-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
