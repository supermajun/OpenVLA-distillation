"""Reconcile saved pilot artifacts with source data, checkpoints, and rollout files."""
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vla_kd.data import Dataset, normalize, to_policy_actions


def read(path):
    return json.loads(Path(path).read_text())


def main():
    os.chdir(ROOT)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    dataset = Dataset("data/pilot", "all", "data/teacher_cache", require_features=True)
    train_ids = {r["id"] for r in dataset.records if r["split"] == "train"}
    split_episodes = {split: {r["episode"] for r in dataset.records if r["split"] == split}
                      for split in ("train", "val")}
    assert split_episodes["train"].isdisjoint(split_episodes["val"])
    assert len(dataset) == 40 and len(train_ids) == 32
    # Reconcile every valid chunk against contiguous original HDF5 timesteps.
    with h5py.File(dataset.manifest["source"], "r") as original:
        for i, record in enumerate(dataset.records):
            sample = dataset[i]  # Also validates content hash and all teacher tensors.
            raw = original["data"][record["episode"]]["actions"]
            step = record["step"]
            expected = normalize(to_policy_actions(raw[step:step + 8]), dataset.manifest["stats"]["action"])
            np.testing.assert_array_equal(sample["actions"][sample["valid"]], expected)
    rows, all_sample_orders = [], []
    for strategy in ("demo", "action", "feature", "both"):
        run = Path("runs/pilot100") / strategy
        ev = run.parent / f"{strategy}-evaluation"
        summary, config = read(run / "summary.json"), read(run / "config.json")
        evaluation = read(ev / "evaluation.json")
        history = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert summary["steps"] == len(history) == 100
        assert [h["step"] for h in history] == list(range(1, 101))
        assert summary["all_parameters_trainable"] and all(summary["updated_components"].values())
        assert summary["checkpoint_roundtrip_exact"]
        assert summary["optimizer_state_dtypes"] == ["torch.bfloat16"]
        assert config["device"] == "mps" and config["dtype"] == "bf16"
        assert config["dataset_fingerprint"] == dataset.manifest["fingerprint"]
        assert config["seed"] == 17 and config["accumulate"] == 1
        assert evaluation["validation_l1"] == summary["final_val_l1"]
        assert evaluation["val_samples"] == summary["validation_samples"] == 8
        assert evaluation["adapters_removed"] and len(evaluation["rollouts"]) == 2
        all_sample_orders.append([h["sample_ids"] for h in history])
        for h in history:
            assert set(h["sample_ids"]).issubset(train_ids)
            assert all(np.isfinite(v) for k, v in h.items() if k != "sample_ids")
        policy = torch.load(run / "policy.pt", map_location="cpu", weights_only=True, mmap=True)
        assert not any(k.startswith("adapters.") for k in policy["model"])
        assert sum(v.numel() for v in policy["model"].values()) == evaluation["deployment_parameters"] == 462083000
        assert all(v.dtype == torch.bfloat16 and torch.isfinite(v).all() for v in policy["model"].values())
        del policy
        for record in evaluation["rollouts"]:
            episode = record["episode"]
            actions = np.load(ev / f"episode-{episode}-actions.npy", allow_pickle=False)
            assert actions.shape == (record["steps"], 7) and np.isfinite(actions).all()
            assert (np.abs(actions) <= 1).all() and 0 < record["steps"] <= 220
            with imageio.get_reader(ev / f"episode-{episode}.mp4") as video:
                count = sum(1 for frame in video if frame.shape == (128, 128, 3))
            assert count == record["steps"] + 1
        rows.append({"strategy": strategy, "initial_val_l1": summary["initial_val_l1"],
                     "final_val_l1": summary["final_val_l1"],
                     "mean_step_seconds": summary["mean_step_seconds"],
                     "sampled_peak_mps_driver_bytes": summary["sampled_peak_memory"]["mps_driver_bytes"],
                     "forward_latency_ms": evaluation["model_forward_latency_ms"],
                     "successes": sum(r["success"] for r in evaluation["rollouts"]), "episodes": 2})
    assert all(order == all_sample_orders[0] for order in all_sample_orders)
    assert len({r["initial_val_l1"] for r in rows}) == 1
    resume = read("runs/resume_check/verification.json")
    assert resume["passed"] and resume["exact_model_optimizer_rng_and_losses"]
    junit = ET.parse(reports / "tests.xml").getroot()
    suites = list(junit.iter("testsuite"))
    assert suites and all(int(s.attrib[k]) == 0 for s in suites for k in ("failures", "errors", "skipped"))
    tests = sum(int(s.attrib["tests"]) for s in suites)
    assert tests >= 27
    source_files = [*Path("vla_kd").glob("*.py"), *Path("scripts").glob("*.py"),
                    *Path("tests").glob("*.py"), Path("requirements.lock.txt"), Path("pyproject.toml")]
    result = {"passed": True, "scope": "500M single-task engineering pilot; no convergence claim",
              "tests_passed": tests, "dataset_samples": 40, "training_samples": 32, "validation_samples": 8,
              "original_hdf5_action_chunks_reconciled": True, "all_teacher_cache_records_validated": True,
              "same_initial_validation_and_training_sample_order": True,
              "deployment_weights_and_eight_rollout_videos_checked": True,
              "resume": resume, "conditions": rows,
              "teacher_sampled_peak_memory": dataset.cache_manifest["sampled_peak_memory"],
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    (reports / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
