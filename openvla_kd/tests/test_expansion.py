import copy
import pytest
import torch

from test_model import tiny, batch
from vla_kd.model import Student, expand_depth, loss_function


@pytest.mark.parametrize("target_layers", [5, 12])
def test_depth_expansion_preserves_policy_and_learns_new_capacity(target_layers):
    torch.manual_seed(21)
    base = tiny().eval()
    b = batch()
    backbone = copy.deepcopy(base.backbone)
    # The tiny fixture already has a feature hook; expansion occurs before that
    # hook in the pretrained production path, so remove it here before copying.
    for layer in backbone.text_model.layers:
        layer._forward_hooks.clear()
    expand_depth(backbone, target_layers)
    grown = Student(backbone, teacher_dims={"visual": 12, "language": 24, "pre_action": 24}).eval()
    for name in ("proprio", "action_head", "adapters"):
        getattr(grown, name).load_state_dict(getattr(base, name).state_dict())
    with torch.no_grad():
        a, z = base(b["inputs"], b["proprio"]), grown(b["inputs"], b["proprio"])
        torch.testing.assert_close(a["actions"], z["actions"], rtol=0, atol=0)
        for k in a["features"]:
            torch.testing.assert_close(a["features"][k], z["features"][k], rtol=0, atol=0)
    assert len(grown.backbone.text_model.layers) == target_layers
    assert len(grown.backbone._kd_added_layers) == target_layers - 2
    expected_extra = (target_layers - 2) * sum(p.numel() for p in base.backbone.text_model.layers[0].parameters())
    assert sum(p.numel() for p in grown.parameters()) == sum(p.numel() for p in base.parameters()) + expected_extra
    opt = torch.optim.AdamW(grown.parameters(), lr=1e-3)
    loss_function(grown, grown(b["inputs"], b["proprio"]), b, "both")[0].backward()
    assert all(p.grad is not None for p in grown.parameters())
    opt.step()
    for index in grown.backbone._kd_added_layers:
        assert grown.backbone.text_model.layers[index].self_attn.o_proj.weight.abs().sum() > 0
    with torch.no_grad():
        assert not torch.equal(grown(b["inputs"], b["proprio"])["actions"], a["actions"])
    with pytest.raises(ValueError):
        expand_depth(backbone, 1)
