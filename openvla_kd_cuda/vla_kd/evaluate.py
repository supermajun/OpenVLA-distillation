"""Offline policy checks and genuine LIBERO closed-loop rollouts on Apple Silicon."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

from .data import Dataset, normalize, preprocess_image, to_environment_actions
from .model import Student, Collator
from .runtime import MemoryMonitor, atomic_json, select_device, sync
from .train import validate


def load_policy(path, model_path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema") != 1:
        raise ValueError("Unsupported policy schema")
    if any(k.startswith("adapters.") for k in payload["model"]):
        raise ValueError("Deployment policy must not contain distillation adapters")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[payload["config"]["dtype"]]
    model = Student.pretrained(model_path, dtype=dtype, checkpointing=False,
                               student_size=payload["config"].get("student_size", "500m"))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, Collator(model_path, device, dtype), payload


def prepare_libero(root, cache):
    root = Path(root).resolve()
    cache = Path(cache).resolve()
    config_dir = cache / "libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    package = root / "libero/libero"
    paths = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
             "init_states": str(package / "init_files"), "assets": str(package / "assets"),
             "datasets": str(Path("data/raw").resolve())}
    (config_dir / "config.yaml").write_text(yaml.safe_dump(paths))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["NUMBA_CACHE_DIR"] = str(cache / "numba")
    sys.path.insert(0, str(root))


def quat_to_axisangle(quat):
    quat = np.asarray(quat, np.float64)
    w = np.clip(quat[3], -1, 1)
    denominator = np.sqrt(1 - w * w)
    if denominator < 1e-8:
        return np.zeros(3, np.float32)
    return (quat[:3] * 2 * np.arccos(w) / denominator).astype(np.float32)


def load_initial_states(path):
    # Official LIBERO files contain NumPy arrays. Allow only those constructors,
    # retaining weights_only instead of enabling general pickle execution.
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype,
               type(np.dtype(np.float64)), type(np.dtype(np.float32))]
    with torch.serialization.safe_globals(allowed):
        states = torch.load(path, map_location="cpu", weights_only=True)
    states = np.asarray(states)
    if states.ndim != 2 or not len(states) or not np.isfinite(states).all():
        raise ValueError("Invalid LIBERO initial states")
    return states


def observation_sample(obs, instruction, stats):
    proprio = np.concatenate([obs["robot0_eef_pos"], quat_to_axisangle(obs["robot0_eef_quat"]),
                              obs["robot0_gripper_qpos"]])
    return {"images": np.stack([preprocess_image(obs[k]) for k in ("agentview_image", "robot0_eye_in_hand_image")]),
            "proprio": normalize(proprio, stats["proprio"]), "instruction": instruction,
            "actions": np.zeros((8, 7), np.float32), "valid": np.ones(8, bool)}


def run_rollouts(args, predict, stats):
    """Shared environment, preprocessing, action execution and success rules for all policies."""
    records = []
    prepare_libero(args.libero, ".cache")
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    import imageio.v2 as imageio
    suite = get_benchmark(args.suite)()
    task_index = suite.get_task_names().index(args.task)
    task = suite.get_task(task_index)
    # Upstream .pruned_init stores a list of NumPy arrays.
    init_path = Path(args.libero) / "libero/libero/init_files" / task.problem_folder / task.init_states_file
    initial_states = load_initial_states(init_path)
    indices = getattr(args, "initial_indices", None) or list(range(args.episodes))
    if len(indices) != args.episodes or min(indices) < 0 or max(indices) >= len(initial_states):
        raise ValueError("Requested more episodes than distinct benchmark initial states")
    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task_index),
                             camera_heights=getattr(args, "camera_size", 128), camera_widths=getattr(args, "camera_size", 128))
    try:
        env.seed(args.seed)
        for episode in range(args.episodes):
            env.reset()
            obs = env.set_init_state(initial_states[indices[episode]])
            for _ in range(10):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            frames, actions, queue = [], [], deque()
            success, replan_count = False, 0
            start = time.perf_counter()
            for step in range(args.max_steps):
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                if not queue:
                    sample = observation_sample(obs, task.language, stats)
                    prediction = predict(sample)
                    if np.asarray(prediction).shape != (8, 7) or not np.isfinite(prediction).all():
                        raise ValueError("Rollout policy must produce finite 8x7 normalized actions")
                    queue.extend(to_environment_actions(prediction, stats["action"])[:args.execute_chunk])
                    replan_count += 1
                action = queue.popleft()
                obs, _, done, _ = env.step(action.tolist())
                actions.append(action)
                success = bool(env.check_success())
                if success or done:
                    break
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            imageio.mimsave(Path(args.out) / f"episode-{episode}.mp4", frames, fps=20, macro_block_size=1)
            np.save(Path(args.out) / f"episode-{episode}-actions.npy", np.asarray(actions))
            record = {"episode": episode, "seed": args.seed, "initial_state_index": indices[episode],
                      "task": task.name, "success": success, "steps": len(actions),
                      "replans": replan_count, "seconds": time.perf_counter() - start}
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        env.close()
    return records


@torch.inference_mode()
def run(args):
    if args.episodes < 1 or args.max_steps < 1 or not 1 <= args.execute_chunk <= 8:
        raise ValueError("Positive episodes/steps and execute_chunk in [1,8] required")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Evaluation output must be a new directory")
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    with MemoryMonitor(device) as monitor:
        model, collator, payload = load_policy(args.policy, args.model, device)
        ds = Dataset(args.data, "val")
        if payload["config"]["dataset_fingerprint"] != ds.manifest["fingerprint"]:
            raise ValueError("Policy and evaluation dataset fingerprints differ")
        val_l1 = validate(model, ds, collator, max_samples=len(ds))
        probe = collator([ds[0]])
        for _ in range(2):
            model(probe["inputs"], probe["proprio"], features=False)
        timings = []
        for _ in range(10):
            sync(device)
            start = time.perf_counter()
            predicted = model(probe["inputs"], probe["proprio"], features=False)["actions"]
            sync(device)
            timings.append(time.perf_counter() - start)
        if not torch.isfinite(predicted).all():
            raise FloatingPointError("Invalid deployed policy output")
        result = {"policy": str(Path(args.policy).resolve()), "strategy": payload["config"]["strategy"],
                  "device": str(device), "dtype": payload["config"]["dtype"], "val_samples": len(ds),
                  "validation_l1": val_l1, "model_forward_latency_ms": float(np.mean(timings) * 1000),
                  "action_generation_per_second": float(8 / np.mean(timings)),
                  "timing_scope": "batch1,2 views; warm forward only; excludes preprocessing and simulation",
                  "deployment_parameters": sum(p.numel() for p in model.parameters()),
                  "policy_file_bytes": Path(args.policy).stat().st_size, "adapters_removed": len(model.adapters) == 0,
                  "rollouts": []}
        if not args.offline_only:
            def predict(sample):
                batch = collator([sample])
                return model(batch["inputs"], batch["proprio"], features=False)["actions"][0].float().cpu().numpy()
            result["rollouts"] = run_rollouts(args, predict, payload["stats"])
        result["scope"] = "single-task pilot; success is reported honestly, not a full LIBERO benchmark"
    result["sampled_peak_memory"] = monitor.peaks
    atomic_json(out / "evaluation.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="models/student")
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--device", default="mps")
    p.add_argument("--libero", default="vendor/LIBERO")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task", default="pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--execute-chunk", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--initial-indices", type=int, nargs="+")
    p.add_argument("--camera-size", type=int, default=128)
    p.add_argument("--offline-only", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
