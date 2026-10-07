"""Audit teacher, capacity, precision and expanded-data experiments from actual files."""
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vla_kd.data import Dataset


def read(path):
    return json.loads(Path(path).read_text())


def video_check(folder, evaluation):
    for r in evaluation["rollouts"]:
        actions = np.load(folder / f'episode-{r["episode"]}-actions.npy')
        assert actions.shape == (r["steps"], 7) and np.isfinite(actions).all()
        assert (np.abs(actions) <= 1).all()
        with imageio.get_reader(folder / f'episode-{r["episode"]}.mp4') as v:
            count = sum(1 for frame in v if frame.shape == (128, 128, 3))
        assert count == r["steps"] + 1


def training_check(folder, steps, size, optimizer):
    s, c = read(folder / "summary.json"), read(folder / "config.json")
    h = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert s["steps"] == len(h) == steps
    assert [r["step"] for r in h] == list(range(1, steps + 1))
    assert c.get("student_size", "500m") == size and c.get("optimizer", "bf16") == optimizer
    assert s["checkpoint_roundtrip_exact"] and s["all_parameters_trainable"]
    assert all(s["updated_components"].values())
    expected = {"500m": 462083000, "1b": 1002860600, "2b": 1995924920}[size]
    assert s["deployment_parameters"] == expected
    expected_dtype = "torch.float32" if optimizer == "master_fp32" else "torch.bfloat16"
    assert s["optimizer_state_dtypes"] == [expected_dtype]
    ds = Dataset(c["data"], "train", c["cache"] if c["strategy"] != "demo" else None,
                 require_features=c["strategy"] in ("feature", "both"))
    ids = {r["id"] for r in ds.records}
    assert all(set(r["sample_ids"]).issubset(ids) for r in h)
    assert all(np.isfinite(v) for r in h for k, v in r.items() if k != "sample_ids")
    payload = torch.load(folder / "policy.pt", map_location="cpu", weights_only=True, mmap=True)
    assert sum(v.numel() for v in payload["model"].values()) == expected
    assert not any(k.startswith("adapters.") for k in payload["model"])
    assert all(torch.isfinite(v).all() for v in payload["model"].values())
    return s, c, h


def main():
    os.chdir(ROOT)
    out = Path("reports/followup")
    teacher = {}
    for key, folder, indices in (
        ("matched", Path("runs/teacher_matched8"), [0, 1] * 4),
        ("distinct", Path("runs/followup/teacher_distinct8"), list(range(8))),
    ):
        e = read(folder / "evaluation.json")
        assert e["frozen"] and [r["initial_state_index"] for r in e["rollouts"]] == indices
        assert e["successes"] == sum(r["success"] for r in e["rollouts"])
        video_check(folder, e)
        teacher[key] = e
    rows = []
    for strategy in ("demo", "action", "feature", "both"):
        folder = Path("runs/pilot1b100") / strategy
        s, c, h = training_check(folder, 100, "1b", "bf16")
        old = Path("runs/pilot100") / strategy
        old_h = [json.loads(line) for line in (old / "metrics.jsonl").read_text().splitlines()]
        assert [r["sample_ids"] for r in h] == [r["sample_ids"] for r in old_h]
        assert s["initial_val_l1"] == read(old / "summary.json")["initial_val_l1"]
        ev_folder = folder.parent / (strategy + "-evaluation")
        e = read(ev_folder / "evaluation.json")
        assert e["validation_l1"] == s["final_val_l1"]
        assert [r["initial_state_index"] for r in e["rollouts"]] == [0, 1]
        video_check(ev_folder, e)
        rows.append({"name": strategy, "training": s, "evaluation": e})
    improved, improved_configs, improved_orders = [], [], []
    baseline_distinct = {}
    for size in ("500m", "1b"):
        folder = Path("runs/followup") / (size + "_baseline8")
        e = read(folder / "evaluation.json")
        assert [r["initial_state_index"] for r in e["rollouts"]] == list(range(8))
        video_check(folder, e)
        baseline_distinct[size] = e
    for name, size, steps in (("1b_precision100", "1b", 100),
                              ("500m_expanded500", "500m", 500), ("1b_expanded500", "1b", 500)):
        folder = Path("runs/followup") / name
        s, c, h = training_check(folder, steps, size, "master_fp32")
        improved_configs.append(c)
        improved_orders.append([r["sample_ids"] for r in h])
        e_folder = folder.parent / (name + "-evaluation")
        e = read(e_folder / "evaluation.json")
        assert e["validation_l1"] == s["final_val_l1"]
        assert [r["initial_state_index"] for r in e["rollouts"]] == list(range(8))
        video_check(e_folder, e)
        improved.append({"name": name, "training": s, "evaluation": e})
    original_config = read("runs/pilot1b100/both/config.json")
    for k in ("model", "data", "cache", "strategy", "student_size", "dtype", "steps", "accumulate", "lr", "seed",
              "action_weight", "feature_weight", "no_checkpointing", "dataset_fingerprint", "teacher_cache_fingerprint"):
        assert improved_configs[0][k] == original_config[k], f"Precision ablation changed {k}"
    original_history = [json.loads(line) for line in Path("runs/pilot1b100/both/metrics.jsonl").read_text().splitlines()]
    assert improved_orders[0] == [r["sample_ids"] for r in original_history]
    assert improved_orders[1] == improved_orders[2]
    assert improved_configs[1]["dataset_fingerprint"] == improved_configs[2]["dataset_fingerprint"]
    expanded = Dataset("data/expanded", "all", "data/expanded_teacher_cache", require_features=True)
    assert len(expanded) == 400
    for i in range(len(expanded)):
        expanded[i]
    episodes = {split: {r["episode"] for r in expanded.records if r["split"] == split} for split in ("train", "val")}
    assert len(episodes["train"]) == 20 and len(episodes["val"]) == 5
    assert episodes["train"].isdisjoint(episodes["val"])
    resume = read("runs/followup/resume_master/verification.json")
    assert resume["passed"] and resume["student_size"] == "1b" and resume["optimizer"] == "master_fp32"
    suites = list(ET.parse(out / "tests.xml").getroot().iter("testsuite"))
    assert suites and all(int(s.attrib[k]) == 0 for s in suites for k in ("failures", "errors", "skipped"))
    source = [*Path("vla_kd").glob("*.py"), *Path("scripts").glob("*.py"), *Path("tests").glob("*.py")]
    result = {"passed": True, "tests_passed": sum(int(s.attrib["tests"]) for s in suites),
              "teacher": teacher, "capacity_baseline": rows, "baseline_distinct": baseline_distinct,
              "optimization": improved, "resume": resume,
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source}}
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"passed": True, "tests_passed": result["tests_passed"],
                      "teacher_successes": {k: v["successes"] for k, v in teacher.items()},
                      "student_successes": {r["name"]: sum(e["success"] for e in r["evaluation"]["rollouts"])
                                             for r in rows + improved}}, indent=2))


if __name__ == "__main__":
    main()
