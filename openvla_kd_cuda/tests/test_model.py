import pytest
import torch
from transformers import Idefics3Config, Idefics3Model

from vla_kd.model import Student, loss_function, masked_l1
from vla_kd.runtime import restore_checkpoint, save_checkpoint
from vla_kd.teacher import bidirectional_mask


def tiny():
    config = Idefics3Config(image_token_id=1, scale_factor=2,
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                       "num_attention_heads": 2, "image_size": 16, "patch_size": 4},
        text_config={"model_type": "llama", "vocab_size": 64, "hidden_size": 32,
                     "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "pad_token_id": 0})
    return Student(Idefics3Model(config), teacher_dims={"visual": 12, "language": 24, "pre_action": 24})


def batch():
    ids = torch.tensor([[2] + [1] * 8 + [3, 4, 5]])
    return {"inputs": {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                       "pixel_values": torch.randn(1, 2, 3, 16, 16)},
            "proprio": torch.randn(1, 8), "actions": torch.randn(1, 8, 7),
            "valid": torch.tensor([[True] * 5 + [False] * 3]),
            "teacher_actions": torch.randn(1, 8, 7, requires_grad=True),
            "teacher_features": {"visual": torch.randn(1, 8, 12, requires_grad=True),
                                 "language": torch.randn(1, 24, requires_grad=True),
                                 "pre_action": torch.randn(1, 24, requires_grad=True)}}


@pytest.mark.parametrize("strategy", ["demo", "action", "feature", "both"])
def test_strategy_gradients(strategy):
    model, b = tiny(), batch()
    output = model(b["inputs"], b["proprio"])
    assert output["actions"].shape == (1, 8, 7)
    loss, terms = loss_function(model, output, b, strategy)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.action_head[-1].weight.grad.abs().sum() > 0
    assert model.backbone.vision_model.embeddings.patch_embedding.weight.grad.abs().sum() > 0
    assert (model.adapters["visual"].weight.grad is not None) == (strategy in ("feature", "both"))
    assert all(x.grad is None for x in b["teacher_features"].values())
    assert b["teacher_actions"].grad is None
    assert all(not key.startswith("adapters.") for key in model.export_state())


def test_mask_excludes_padding():
    pred = torch.zeros(1, 8, 7)
    target = torch.zeros_like(pred)
    target[:, 1:] = 1e6
    mask = torch.tensor([[True] + [False] * 7])
    assert masked_l1(pred, target, mask) == 0
    with pytest.raises(ValueError):
        masked_l1(pred, target, torch.zeros_like(mask))


def test_checkpoint_restore_next_update(tmp_path):
    model, b = tiny(), batch()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    def step(m, o):
        o.zero_grad(set_to_none=True)
        loss_function(m, m(b["inputs"], b["proprio"]), b, "both")[0].backward()
        o.step()
    step(model, opt)
    path = tmp_path / "check.pt"
    save_checkpoint(path, model, opt, {"strategy": "both"}, 1)
    step(model, opt)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    restored = tiny()
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3, foreach=False)
    config, index = restore_checkpoint(path, restored, restored_opt)
    assert index == 1 and config["strategy"] == "both"
    step(restored, restored_opt)
    for k, v in restored.state_dict().items():
        torch.testing.assert_close(v, expected[k], rtol=0, atol=0)


def test_teacher_attention_bidirectional_and_padding():
    mask = bidirectional_mask(torch.tensor([[1, 1, 1, 0]]), torch.float32)
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 2] == 0  # Future action slots remain visible.
    assert (mask[0, 0, :, 3] == torch.finfo(torch.float32).min).all()
    with pytest.raises(ValueError):
        bidirectional_mask(torch.ones(3), torch.float32)


def test_feature_shape_rejected():
    model, b = tiny(), batch()
    b["teacher_features"]["visual"] = torch.randn(1, 7, 12)
    with pytest.raises(ValueError, match="Feature shape"):
        loss_function(model, model(b["inputs"], b["proprio"]), b, "feature")


def test_teacher_uses_downstream_statistics(tmp_path):
    import json
    from vla_kd.teacher import load_teacher_stats
    (tmp_path / "config.json").write_text(json.dumps({"norm_stats": {"pretraining": {}}}))
    downstream = {"libero_spatial_no_noops": {"action": {"q01": [-1] * 7}, "proprio": {"q01": [-1] * 8}}}
    (tmp_path / "dataset_statistics.json").write_text(json.dumps(downstream))
    assert load_teacher_stats(tmp_path) == downstream
    (tmp_path / "dataset_statistics.json").write_text('{}')
    with pytest.raises(ValueError, match="statistics"):
        load_teacher_stats(tmp_path)


def test_bidirectional_mask_reaches_llama_attention():
    from transformers import LlamaConfig, LlamaModel
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2)
    config._attn_implementation = "sdpa"
    model = LlamaModel(config).eval()
    a, b = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 2, 3, 9]])
    padding = torch.ones_like(a)
    mask = bidirectional_mask(padding, torch.float32)
    with torch.no_grad():
        causal_a = model(a, attention_mask=padding, use_cache=False).last_hidden_state[:, 0]
        causal_b = model(b, attention_mask=padding, use_cache=False).last_hidden_state[:, 0]
        bidir_a = model(a, attention_mask=mask, use_cache=False).last_hidden_state[:, 0]
        bidir_b = model(b, attention_mask=mask, use_cache=False).last_hidden_state[:, 0]
    torch.testing.assert_close(causal_a, causal_b, rtol=0, atol=0)
    assert not torch.allclose(bidir_a, bidir_b)


def test_export_matches_training_policy_without_adapters():
    model, b = tiny().eval(), batch()
    export = model.export_state()
    deployed = tiny().eval()
    deployed.adapters = torch.nn.ModuleDict()
    deployed.load_state_dict(export, strict=True)
    with torch.no_grad():
        expected = model(b["inputs"], b["proprio"], features=False)["actions"]
        actual = deployed(b["inputs"], b["proprio"], features=False)["actions"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_observation_and_proprioception_affect_actions():
    model, b = tiny().eval(), batch()
    with torch.no_grad():
        base = model(b["inputs"], b["proprio"], features=False)["actions"]
        moved = model(b["inputs"], b["proprio"] + 1, features=False)["actions"]
        b["inputs"]["pixel_values"] *= -1
        vision_changed = model(b["inputs"], b["proprio"], features=False)["actions"]
        b["inputs"]["input_ids"][0, -1] = 8
        instruction_changed = model(b["inputs"], b["proprio"], features=False)["actions"]
    assert not torch.allclose(base, moved)
    assert not torch.allclose(base, vision_changed)
    assert not torch.allclose(vision_changed, instruction_changed)
