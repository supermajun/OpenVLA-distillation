import copy
import pytest
import torch

from vla_kd.optim import MasterAdamW


def test_master_accumulates_updates_that_bf16_loses():
    a = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    b = torch.nn.Parameter(a.detach().clone())
    full = MasterAdamW([a], lr=1e-4, weight_decay=0)
    rounded = torch.optim.AdamW([b], lr=1e-4, weight_decay=0, foreach=False)
    for _ in range(100):
        a.grad = torch.ones_like(a)
        b.grad = torch.ones_like(b)
        full.step()
        rounded.step()
    assert torch.equal(b, torch.ones_like(b))
    assert (a < 0.995).all()
    assert full.state[a]["master"].dtype == torch.float32


def test_master_matches_float32_adamw_and_restores_without_rounding():
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.randn(3, 5).bfloat16())
    reference = torch.nn.Parameter(p.detach().float())
    opt = MasterAdamW([p], lr=3e-4)
    ref_opt = torch.optim.AdamW([reference], lr=3e-4, foreach=False)
    for _ in range(8):
        p.grad = torch.randn_like(p)
        reference.grad = p.grad.float()
        opt.step()
        ref_opt.step()
    torch.testing.assert_close(opt.state[p]["master"], reference, rtol=1e-6, atol=1e-7)
    restored = torch.nn.Parameter(p.detach().clone())
    restored_opt = MasterAdamW([restored])
    restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    for key in ("master", "exp_avg", "exp_avg_sq"):
        assert restored_opt.state[restored][key].dtype == torch.float32
        torch.testing.assert_close(opt.state[p][key], restored_opt.state[restored][key], rtol=0, atol=0)
    p.grad = torch.randn_like(p)
    restored.grad = p.grad.clone()
    opt.step()
    restored_opt.step()
    torch.testing.assert_close(p, restored, rtol=0, atol=0)
    torch.testing.assert_close(opt.state[p]["master"], restored_opt.state[restored]["master"], rtol=0, atol=0)


def test_invalid_optimizer_settings():
    with pytest.raises(ValueError):
        MasterAdamW([torch.nn.Parameter(torch.ones(1))], lr=-1)
