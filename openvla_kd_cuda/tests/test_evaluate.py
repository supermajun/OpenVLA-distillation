import numpy as np
import pytest
import torch

from vla_kd.evaluate import load_initial_states, observation_sample, quat_to_axisangle


def test_safe_numpy_initial_states(tmp_path):
    path = tmp_path / "states.pt"
    torch.save([np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64) + 1], path)
    states = load_initial_states(path)
    assert states.shape == (2, 10)
    np.testing.assert_array_equal(states[1], np.arange(10) + 1)
    torch.save([np.array([np.nan])], path)
    with pytest.raises(ValueError):
        load_initial_states(path)


def test_observation_conversion_matches_training():
    obs = {"agentview_image": np.zeros((128, 128, 3), np.uint8),
           "robot0_eye_in_hand_image": np.ones((128, 128, 3), np.uint8),
           "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
           "robot0_eef_quat": np.array([0, 0, 0, 1.]),
           "robot0_gripper_qpos": np.array([0.01, -0.01])}
    stats = {"proprio": {"q01": [-1] * 8, "q99": [1] * 8}}
    s = observation_sample(obs, "pick up bowl", stats)
    assert s["images"].shape == (2, 224, 224, 3)
    np.testing.assert_allclose(s["proprio"], [0.1, 0.2, 0.3, 0, 0, 0, 0.01, -0.01], atol=1e-7)
    np.testing.assert_allclose(quat_to_axisangle([1, 0, 0, 0]), [np.pi, 0, 0])


@pytest.mark.parametrize("bad_prediction", [False, True])
def test_shared_rollout_executes_policy_and_honors_initial_states(tmp_path, monkeypatch, bad_prediction):
    import sys
    from types import SimpleNamespace
    from vla_kd import evaluate
    obs = {"agentview_image": np.zeros((128, 128, 3), np.uint8),
           "robot0_eye_in_hand_image": np.zeros((128, 128, 3), np.uint8),
           "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1]),
           "robot0_gripper_qpos": np.zeros(2)}
    class Env:
        def __init__(self, **kwargs):
            self.starts, self.actions, self.closed = [], [], False
        def seed(self, seed):
            self.seed_value = seed
        def reset(self):
            self.steps = 0
        def set_init_state(self, state):
            self.starts.append(state[0])
            return obs
        def step(self, action):
            self.steps += 1
            if self.steps > 10:
                self.actions.append(action)
            return obs, 0, False, {}
        def check_success(self):
            return self.steps >= 12
        def close(self):
            self.closed = True
    env = Env()
    task = SimpleNamespace(name="test", language="pick", problem_folder="test", init_states_file="init")
    suite = SimpleNamespace(get_task_names=lambda: ["test"], get_task=lambda i: task,
                            get_task_bddl_file_path=lambda i: "unused")
    monkeypatch.setattr(evaluate, "prepare_libero", lambda *a: None)
    monkeypatch.setattr(evaluate, "load_initial_states", lambda p: np.array([[0], [1]]))
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", SimpleNamespace(get_benchmark=lambda name: lambda: suite))
    monkeypatch.setitem(sys.modules, "libero.libero.envs", SimpleNamespace(OffScreenRenderEnv=lambda **kw: env))
    import imageio.v2
    frames = []
    monkeypatch.setattr(imageio.v2, "mimsave", lambda p, images, **kw: frames.append(len(images)))
    args = SimpleNamespace(libero="unused", suite="unused", task="test", episodes=2, initial_indices=[1, 0],
                           seed=17, max_steps=220, execute_chunk=8, out=tmp_path)
    stats = {"proprio": {"q01": [-1] * 8, "q99": [1] * 8},
             "action": {"q01": [-1] * 6 + [0], "q99": [1] * 7, "mask": [True] * 6 + [False]}}
    def predict(sample):
        assert sample["instruction"] == "pick"
        result = np.zeros((8, 7), np.float32)
        result[:, -1] = 1  # Open must become -1 for the environment.
        return result[:1] if bad_prediction else result
    if bad_prediction:
        with pytest.raises(ValueError, match="8x7"):
            evaluate.run_rollouts(args, predict, stats)
    else:
        result = evaluate.run_rollouts(args, predict, stats)
        assert env.starts == [1, 0] and env.seed_value == 17
        assert all(r["success"] and r["steps"] == 2 and r["replans"] == 1 for r in result)
        assert [r["initial_state_index"] for r in result] == [1, 0]
        assert all(a[-1] == -1 for a in env.actions)
        assert frames == [3, 3]
    assert env.closed
