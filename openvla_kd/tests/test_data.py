import json
import h5py
import numpy as np
import pytest

from vla_kd.data import (Dataset, action_chunk, denormalize, normalize, prepare, preprocess_image,
                         to_environment_actions, to_policy_actions)


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "pick_up_bowl_demo.hdf5"
    with h5py.File(source, "w") as f:
        data = f.create_group("data")
        for i, n in enumerate((2, 5, 3)):
            d = data.create_group(f"demo_{i}")
            d["actions"] = np.full((n, 7), i / 3, np.float32)
            obs = d.create_group("obs")
            obs["agentview_rgb"] = np.full((n, 32, 32, 3), 60 + i, np.uint8)
            obs["eye_in_hand_rgb"] = np.full((n, 32, 32, 3), 160 + i, np.uint8)
            obs["ee_states"] = np.zeros((n, 6), np.float32)
            obs["gripper_states"] = np.zeros((n, 2), np.float32)
    stats = {"suite": {"action": {"q01": [-1] * 6 + [0], "q99": [1] * 7,
                                   "mask": [True] * 6 + [False]},
                        "proprio": {"q01": [-1] * 8, "q99": [1] * 8}}}
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats))
    out = tmp_path / "prepared"
    prepare(source, stats_path, out, train_episodes=2, val_episodes=1, samples_per_episode=3)
    return out


def test_split_and_chunks(prepared):
    train, val = Dataset(prepared, "train"), Dataset(prepared, "val")
    assert {r["episode"] for r in train.records}.isdisjoint({r["episode"] for r in val.records})
    assert len(train) == 5 and len(val) == 3
    assert train[0]["images"].shape == (2, 224, 224, 3)
    assert train[1]["valid"].sum() == 1
    assert np.allclose(train[0]["actions"][:2], train[0]["actions"][0])


def test_chunk_boundary():
    actions = np.arange(21).reshape(3, 7)
    chunk, valid = action_chunk(actions, 2)
    assert valid.tolist() == [True] + [False] * 7
    assert (chunk == actions[-1]).all()
    with pytest.raises(ValueError):
        action_chunk(actions, 3)


def test_gripper_direction_and_roundtrip():
    raw = np.zeros((2, 7), np.float32)
    raw[:, -1] = [-1, 1]
    stats = {"q01": [-1] * 6 + [0], "q99": [1] * 7, "mask": [True] * 6 + [False]}
    policy = normalize(to_policy_actions(raw), stats)
    assert policy[:, -1].tolist() == [1, 0]
    np.testing.assert_allclose(to_environment_actions(policy, stats), raw)
    np.testing.assert_allclose(denormalize(policy, stats), to_policy_actions(raw))
    with pytest.raises(ValueError):
        normalize(np.full((7,), np.nan), stats)


def test_image_orientation():
    raw = np.zeros((224, 224, 3), np.uint8)
    raw[:112, :112] = 255
    image = preprocess_image(raw)
    assert image[-30:, -30:].mean() > 250
    assert image[:30, :30].mean() == 0
    with pytest.raises(ValueError):
        preprocess_image(raw.astype(float))


def test_tamper_detection(prepared):
    ds = Dataset(prepared, "train")
    path = prepared / (ds.records[0]["id"] + ".npz")
    with np.load(path) as f:
        sample = {k: f[k] for k in f.files}
    sample["images"][0, 0, 0, 0] += 1
    np.savez(path, **sample)
    with pytest.raises(ValueError, match="contents"):
        ds[0]


def test_cache_mismatch_and_incomplete(prepared, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    cm = {"dataset_fingerprint": "wrong", "complete": True, "teacher_kind": "openvla_oft"}
    (cache / "manifest.json").write_text(json.dumps(cm))
    with pytest.raises(ValueError, match="different dataset"):
        Dataset(prepared, "train", cache)
    cm["dataset_fingerprint"] = Dataset(prepared, "train").manifest["fingerprint"]
    cm["complete"] = False
    (cache / "manifest.json").write_text(json.dumps(cm))
    with pytest.raises(ValueError, match="Incomplete"):
        Dataset(prepared, "train", cache)


def test_no_overwrite(prepared):
    with pytest.raises(FileExistsError):
        prepare("unused", "unused", prepared)


@pytest.mark.parametrize("problem", ["missing", "sample_hash", "action_shape", "action_nan", "feature_shape", "feature_nan"])
def test_reject_invalid_cache(prepared, tmp_path, problem):
    ds = Dataset(prepared, "train")
    cache = tmp_path / "cache"
    cache.mkdir()
    ids = [r["id"] for r in ds.records]
    cm = {"dataset_fingerprint": ds.manifest["fingerprint"], "complete": True, "teacher_kind": "openvla_oft",
          "records": ids if problem != "missing" else ids[1:],
          "feature_dims": {"visual": 4, "language": 6, "pre_action": 6}}
    (cache / "manifest.json").write_text(json.dumps(cm))
    values = {"sample_sha256": ds.records[0]["sha256"], "actions": np.zeros((8, 7)),
              "visual": np.zeros((8, 4)), "language": np.zeros(6), "pre_action": np.zeros(6)}
    if problem == "sample_hash":
        values["sample_sha256"] = "wrong"
    if problem == "action_shape":
        values["actions"] = np.zeros((7, 7))
    if problem == "action_nan":
        values["actions"][0, 0] = np.nan
    if problem == "feature_shape":
        values["visual"] = np.zeros((7, 4))
    if problem == "feature_nan":
        values["language"][0] = np.nan
    np.savez(cache / (ids[0] + ".npz"), **values)
    with pytest.raises(ValueError):
        Dataset(prepared, "train", cache, require_features=True)[0]
