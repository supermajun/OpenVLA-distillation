"""Audit all three completed, matched BF16 capacity runs and their real videos."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_followup import training_check, video_check, read
from vla_kd.runtime import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/capacity2b_20261001")
    args = p.parse_args()
    os.chdir(ROOT)
    root = Path(args.out)
    resume = read(root / "resume_check/verification.json")
    assert resume["passed"] and resume["student_size"] == "2b" and resume["optimizer"] == "bf16"
    tests = list(ET.parse("reports/2b-tests.xml").getroot().iter("testsuite"))
    assert tests and all(int(s.attrib[k]) == 0 for s in tests for k in ("failures", "errors", "skipped"))
    baseline_config = read("runs/followup/1b_expanded500/config.json")
    baseline_order = [json.loads(line)["sample_ids"] for line in Path("runs/followup/1b_expanded500/metrics.jsonl").read_text().splitlines()]
    rows = []
    for size in ("500m", "1b", "2b"):
        folder = root / (size + "_bf16_500")
        s, c, history = training_check(folder, 500, size, "bf16")
        for key in ("model", "data", "cache", "strategy", "dtype", "steps", "accumulate", "lr", "seed",
                    "action_weight", "feature_weight", "no_checkpointing", "dataset_fingerprint", "teacher_cache_fingerprint"):
            assert c[key] == baseline_config[key], (size, key)
        assert [r["sample_ids"] for r in history] == baseline_order
        assert s["initial_val_l1"] == read("runs/followup/1b_expanded500/summary.json")["initial_val_l1"]
        assert s["validation_samples"] == 80
        ev_folder = root / (size + "_bf16_500-evaluation")
        e = read(ev_folder / "evaluation.json")
        assert e["validation_l1"] == s["final_val_l1"]
        assert [r["initial_state_index"] for r in e["rollouts"]] == list(range(8))
        assert all(r["seed"] == 17 and r["steps"] <= 220 for r in e["rollouts"])
        video_check(ev_folder, e)
        rows.append({"size": size, "successes": sum(r["success"] for r in e["rollouts"]),
                     "training": s, "evaluation": e})
    snapshot = read(root / "pipeline.json")["source_sha256"]
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in snapshot.items()), "Source changed during experiment"
    result = {"passed": True, "scope": "single-task, single-seed, full-BF16 depth-expanded capacity comparison",
              "tests_passed": sum(int(s.attrib["tests"]) for s in tests), "resume": resume,
              "rows": rows, "source_sha256": snapshot}
    atomic_json(root / "verification.json", result)
    lines = ["# 2B 配套容量实验：自动核验结果", "", "全 BF16；同一任务、seed=17、8 个不同初始状态；每组 500 步。",
             "1B/2B 为同一预训练骨干扩深，不代表原生预训练大模型。FP32 主权重的旧结果不混作本表容量对照。", "",
             "| 规模 | 成功数 | 验证 L1 | MPS driver 采样峰值 GB |", "|---|---:|---:|---:|"]
    for row in rows:
        s = row["training"]
        lines.append(f'| {row["size"]} | {row["successes"]}/8 | {s["final_val_l1"]:.6f} | {s["sampled_peak_memory"]["mps_driver_bytes"]/1e9:.3f} |')
    lines += ["", "自动检查已通过：参数/优化器精度/配置与样本顺序/初始化/保存加载/轨迹视频帧数及动作数组。",
              "后续仍需人工检查视频、解释容量和舍入风险，并补充多种子及完整任务集；不能仅凭八个回合断言规模优劣。"]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"passed": True, "successes": {r["size"]: r["successes"] for r in rows}}, indent=2))


if __name__ == "__main__":
    main()
