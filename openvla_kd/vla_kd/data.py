"""Trajectory-safe LIBERO data and content-addressed teacher supervision."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

PREPROCESS = {"version": 1, "rotation": 180, "crop_area": 0.9,
              "rgb_size": 224, "augmentation": "none", "chunk": 8,
              "action_order": "xyz-axisangle-gripper_open01"}
FEATURE_KEYS = ("visual", "language", "pre_action")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalize(values, stats):
    x = np.asarray(values, dtype=np.float32)
    low, high = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), bool)
    if x.shape[-1] != len(low) or not np.isfinite(x).all():
        raise ValueError("Invalid values for normalization")
    return np.where(mask, np.clip(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1), x).astype(np.float32)


def denormalize(values, stats):
    x = np.asarray(values, dtype=np.float32)
    low, high = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), bool)
    return np.where(mask, (np.clip(x, -1, 1) + 1) * 0.5 * (high - low) + low, x).astype(np.float32)


def to_policy_actions(raw):
    x = np.array(raw, dtype=np.float32, copy=True)
    x[..., -1] = 1 - np.clip(x[..., -1], 0, 1)
    return x


def to_environment_actions(normalized, stats):
    x = denormalize(normalized, stats)
    x[..., -1] = np.where(x[..., -1] > 0.5, -1.0, 1.0)
    return np.clip(x, -1, 1)


def preprocess_image(raw):
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError("Expected uint8 HWC RGB image")
    im = Image.fromarray(np.ascontiguousarray(raw[::-1, ::-1]))
    w, h = im.size
    cw, ch = round(w * np.sqrt(0.9)), round(h * np.sqrt(0.9))
    left, top = (w - cw) // 2, (h - ch) // 2
    return np.asarray(im.crop((left, top, left + cw, top + ch)).resize((224, 224), Image.Resampling.BILINEAR))


def action_chunk(actions, step, horizon=8):
    if not 0 <= step < len(actions) or horizon < 1:
        raise ValueError("Invalid action chunk index/horizon")
    count = min(horizon, len(actions) - step)
    chunk = np.repeat(actions[step + count - 1:step + count], horizon, axis=0)
    chunk[:count] = actions[step:step + count]
    valid = np.arange(horizon) < count
    return chunk, valid


def sample_hash(sample):
    h = hashlib.sha256()
    for key in ("images", "proprio", "actions", "valid"):
        h.update(np.ascontiguousarray(sample[key]).tobytes())
    h.update(sample["instruction"].encode())
    h.update(sample["id"].encode())
    return h.hexdigest()


def prepare(hdf5_path, stats_path, out_dir, train_episodes=4, val_episodes=1, samples_per_episode=8):
    out = Path(out_dir)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Dataset already exists: {out}")
    if min(train_episodes, val_episodes, samples_per_episode) < 1:
        raise ValueError("Positive train/validation episodes and sample count required")
    stats_all = json.loads(Path(stats_path).read_text())
    if len(stats_all) != 1:
        raise ValueError("Pilot requires one explicit suite's normalization statistics")
    stats_key = next(iter(stats_all))
    stats = stats_all[stats_key]
    instruction = Path(hdf5_path).stem.removesuffix("_demo").replace("_", " ")
    manifest = {"schema": 1, "source": str(Path(hdf5_path).resolve()),
                "source_kind": "original_libero_hdf5_subset", "preprocess": PREPROCESS,
                "stats": stats, "stats_key": stats_key, "records": []}
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        if len(names) < train_episodes + val_episodes:
            raise ValueError("Not enough distinct trajectories for requested split")
        splits = {"train": names[:train_episodes], "val": names[-val_episodes:]}
        for split, episodes in splits.items():
            for episode in episodes:
                d = f["data"][episode]
                actions = normalize(to_policy_actions(d["actions"][:]), stats["action"])
                for step in np.unique(np.linspace(0, len(actions) - 1, samples_per_episode, dtype=int)):
                    obs = d["obs"]
                    images = np.stack([preprocess_image(obs[k][step]) for k in ("agentview_rgb", "eye_in_hand_rgb")])
                    proprio = np.concatenate([obs["ee_states"][step], obs["gripper_states"][step]])
                    proprio = normalize(proprio, stats["proprio"])
                    chunk, valid = action_chunk(actions, int(step))
                    sid = f"{episode}-{step:05d}"
                    sample = dict(images=images, proprio=proprio, actions=chunk, valid=valid,
                                  instruction=instruction, id=sid)
                    np.savez_compressed(out / f"{sid}.npz", **sample)
                    manifest["records"].append({"id": sid, "episode": episode, "step": int(step),
                                                "split": split, "sha256": sample_hash(sample)})
    manifest["fingerprint"] = digest({k: v for k, v in manifest.items() if k != "source"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class Dataset:
    def __init__(self, root, split, cache=None, require_features=False):
        if split not in ("train", "val", "all"):
            raise ValueError("Unknown split")
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        expected = digest({k: v for k, v in self.manifest.items() if k not in ("source", "fingerprint")})
        if expected != self.manifest["fingerprint"]:
            raise ValueError("Dataset manifest fingerprint mismatch")
        self.records = [r for r in self.manifest["records"] if split == "all" or r["split"] == split]
        if not self.records:
            raise ValueError("Empty dataset split")
        self.cache = Path(cache) if cache else None
        self.require_features = require_features
        if self.cache:
            cm = json.loads((self.cache / "manifest.json").read_text())
            if cm["dataset_fingerprint"] != self.manifest["fingerprint"]:
                raise ValueError("Teacher cache belongs to a different dataset/preprocessing")
            if not cm.get("complete") or cm.get("teacher_kind") != "openvla_oft":
                raise ValueError("Incomplete or non-OFT teacher cache")
            cached_ids = cm.get("records", [])
            if len(cached_ids) != len(set(cached_ids)) or not {r["id"] for r in self.records}.issubset(cached_ids):
                raise ValueError("Teacher cache does not cover requested samples uniquely")
            self.cache_manifest = cm

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        with np.load(self.root / f'{r["id"]}.npz', allow_pickle=False) as f:
            sample = {k: f[k] for k in f.files}
        for k in ("instruction", "id"):
            sample[k] = str(sample[k])
        if sample_hash(sample) != r["sha256"]:
            raise ValueError("Sample contents do not match manifest")
        if self.cache:
            with np.load(self.cache / f'{r["id"]}.npz', allow_pickle=False) as f:
                if str(f["sample_sha256"]) != r["sha256"]:
                    raise ValueError("Teacher cache sample mismatch")
                sample["teacher_actions"] = f["actions"].astype(np.float32)
                if sample["teacher_actions"].shape != sample["actions"].shape:
                    raise ValueError("Teacher action shape mismatch")
                if not np.isfinite(sample["teacher_actions"]).all():
                    raise ValueError("Non-finite teacher actions")
                if self.require_features:
                    sample["teacher_features"] = {k: f[k].astype(np.float32) for k in FEATURE_KEYS}
                    for key, value in sample["teacher_features"].items():
                        expected = ((8,) if key == "visual" else ()) + (self.cache_manifest["feature_dims"][key],)
                        if value.shape != expected or not np.isfinite(value).all():
                            raise ValueError(f"Invalid teacher feature: {key}")
        return sample


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--stats", default="models/teacher/dataset_statistics.json")
    p.add_argument("--out", default="data/pilot")
    p.add_argument("--train-episodes", type=int, default=4)
    p.add_argument("--val-episodes", type=int, default=1)
    p.add_argument("--samples-per-episode", type=int, default=8)
    args = p.parse_args()
    m = prepare(args.hdf5, args.stats, args.out, args.train_episodes, args.val_episodes, args.samples_per_episode)
    print(json.dumps({"samples": len(m["records"]), "fingerprint": m["fingerprint"]}, indent=2))


if __name__ == "__main__":
    main()
