from contextlib import contextmanager

import pytest
import torch
from torch.nn import functional as F

from test_model import tiny, batch
from vla_kd.model import pool_visual, loss_function, VISUAL_POOLING
from vla_kd.optim import MasterAdamW
from vla_kd.runtime import save_checkpoint, restore_checkpoint


@contextmanager
def strict_determinism():
    previous = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    backends = torch.backends.cuda
    settings = [(backends.enable_flash_sdp, backends.flash_sdp_enabled()),
                (backends.enable_mem_efficient_sdp, backends.mem_efficient_sdp_enabled()),
                (backends.enable_cudnn_sdp, backends.cudnn_sdp_enabled()),
                (backends.enable_math_sdp, backends.math_sdp_enabled())]
    try:
        torch.use_deterministic_algorithms(True)
        for enable, _ in settings[:-1]:
            enable(False)
        backends.enable_math_sdp(True)
        yield
    finally:
        for enable, original in settings:
            enable(original)
        torch.use_deterministic_algorithms(previous, warn_only=warn)


@pytest.mark.parametrize("tokens", [1, 3, 4, 7, 64, 81, 144])
def test_visual_pool_forward_and_gradient_match_adaptive_reference(tokens):
    # Non-contiguous input, two samples with two cameras each; includes
    # overlapping bins, a single token and realistic visual token counts.
    torch.manual_seed(43)
    hidden = torch.randn(4, 13, tokens, dtype=torch.float64).transpose(1, 2).requires_grad_()
    reference_input = hidden.detach().clone().requires_grad_()
    actual = pool_visual(hidden, 2)
    expected = F.adaptive_avg_pool1d(reference_input.transpose(1, 2), 4).transpose(1, 2).reshape(2, 8, 13)
    weights = torch.randn_like(expected)
    actual.backward(weights)
    expected.backward(weights)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(hidden.grad, reference_input.grad, rtol=1e-12, atol=1e-12)


def test_visual_pool_keeps_camera_order():
    hidden = torch.arange(4.).reshape(4, 1, 1).expand(4, 7, 3)
    result = pool_visual(hidden, 2)
    torch.testing.assert_close(result[:, :, 0], torch.tensor([[0.] * 4 + [1.] * 4, [2.] * 4 + [3.] * 4]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires allocated CUDA GPU")
@pytest.mark.parametrize("tokens", [64, 81])
def test_cuda_bf16_visual_pool_strict_backward_is_repeatable(tokens):
    torch.manual_seed(8)
    source = torch.randn(2, tokens, 960).bfloat16()
    ref = source.float().requires_grad_()
    expected = F.adaptive_avg_pool1d(ref.transpose(1, 2), 4).transpose(1, 2).reshape(1, 8, 960)
    weights = torch.randn_like(expected).bfloat16()
    expected.backward(weights.float())
    results = []
    with strict_determinism():
        for _ in range(2):
            x = source.cuda().requires_grad_()
            y = pool_visual(x, 1)
            y.backward(weights.cuda())
            results.append((y.detach().cpu(), x.grad.cpu()))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    torch.testing.assert_close(results[0][0].float(), expected.detach(), rtol=.01, atol=.002)
    torch.testing.assert_close(results[0][1].float(), ref.grad, rtol=.01, atol=.002)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_joint_distillation_strict_checkpointed_backward_and_resume(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires allocated CUDA GPU")
    torch.manual_seed(41)
    source = batch()
    def move(value):
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value.detach().to(device)
    source = move(source)
    source['inputs']['pixel_values'] = source['inputs']['pixel_values'].bfloat16()
    source['proprio'] = source['proprio'].bfloat16()
    def model():
        m = tiny().to(device=device, dtype=torch.bfloat16)
        m.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        return m
    def step(m, opt):
        m.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_function(m, m(source['inputs'], source['proprio']), source, 'both')[0]
        loss.backward()
        assert m.backbone.vision_model.embeddings.patch_embedding.weight.grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True, foreach=False)
        opt.step()
        return loss.detach().cpu()
    with strict_determinism():
        m = model()
        opt = MasterAdamW(m.parameters(), lr=1e-3)
        step(m, opt)
        config = {'visual_pooling': VISUAL_POOLING}
        save_checkpoint(tmp_path/'first.pt', m, opt, config, 1)
        expected_loss = step(m, opt)
        restored = model()
        other = MasterAdamW(restored.parameters(), lr=1e-3)
        restore_checkpoint(tmp_path/'first.pt', restored, other, expected_config=config)
        actual_loss = step(restored, other)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(restored.state_dict(), m.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(other.state_dict(), opt.state_dict(), rtol=0, atol=0)


def test_resume_rejects_old_visual_pooling_before_mutation(tmp_path):
    m = torch.nn.Linear(2, 2)
    save_checkpoint(tmp_path/'old.pt', m, torch.optim.AdamW(m.parameters()), {}, 0)
    with torch.no_grad():
        m.weight.fill_(9)
    expected = m.weight.detach().clone()
    with pytest.raises(ValueError, match='visual_pooling'):
        restore_checkpoint(tmp_path/'old.pt', m, expected_config={'visual_pooling': VISUAL_POOLING})
    torch.testing.assert_close(m.weight, expected, rtol=0, atol=0)
