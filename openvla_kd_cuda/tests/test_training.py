import copy
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_model import tiny, batch
from vla_kd import train
from vla_kd.control import StopRequest
from vla_kd.optim import MasterAdamW
from vla_kd.runtime import restore_checkpoint, save_checkpoint, select_device


def test_cuda_unavailable_fails_clearly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="allocated GPU"):
        select_device("cuda")


def test_signal_request_is_deferred_and_handlers_restored():
    original = signal.getsignal(signal.SIGTERM)
    with StopRequest() as stop:
        stop.receive(signal.SIGTERM, None)
        assert stop.requested and stop.signal == signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == original


def test_resume_rejects_wrong_precision_before_mutation(tmp_path):
    model = torch.nn.Linear(2, 2).bfloat16()
    opt = torch.optim.AdamW(model.parameters(), foreach=False)
    save_checkpoint(tmp_path/'old.pt', model, opt, {'optimizer':'bf16'}, 1)
    with torch.no_grad(): model.weight.fill_(7)
    expected = model.weight.detach().clone()
    with pytest.raises(ValueError, match='optimizer'):
        restore_checkpoint(tmp_path/'old.pt', model, MasterAdamW(model.parameters()),
                           expected_config={'optimizer':'master_fp32'})
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)


def test_periodic_logs_interruption_and_resume_match_continuous(tmp_path, monkeypatch):
    torch.manual_seed(3)
    source = batch()
    for k in ("pixel_values",): source["inputs"][k] = source["inputs"][k].bfloat16()
    source["proprio"] = source["proprio"].bfloat16()
    class Data:
        manifest = {"fingerprint": "fixture", "stats": {}}
        cache_manifest = {"feature_dims": {"visual": 12, "language": 24, "pre_action": 24}}
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 2
        def __getitem__(self, i): return {"id": str(i)}
    monkeypatch.setattr(train, "Dataset", Data)
    monkeypatch.setattr(train, "Collator", lambda *a: lambda samples: copy.deepcopy(source))
    monkeypatch.setattr(train.Student, "pretrained", lambda *a, **k: tiny().to(torch.bfloat16))
    def args(name, resume=None):
        return SimpleNamespace(out=str(tmp_path/name), resume=resume, model="fixture", student_size="500m",
            optimizer="master_fp32", data="fixture", cache="fixture", strategy="both", device="cpu", dtype="bf16",
            steps=3, total_steps=3, accumulate=1, lr=1e-3, seed=17, action_weight=1., feature_weight=.1,
            no_checkpointing=False, validate_every=1, checkpoint_every=1, deterministic=False)
    complete = train.run(args("continuous"))
    assert complete["steps"] == 3 and complete["checkpoint_roundtrip_exact"]
    stop = StopRequest()
    stop.receive(signal.SIGTERM, None)
    interrupted = train._run(args("interrupted"), stop)
    assert interrupted["status"] == "interrupted" and interrupted["step"] == 1
    assert not (tmp_path/'interrupted/summary.json').exists()
    restored = train.run(args("resumed", interrupted["checkpoint"]))
    assert restored["updates_this_run"] == 2 and restored["start_step"] == 1
    a = torch.load(tmp_path/'continuous/checkpoint.pt', weights_only=True)
    b = torch.load(tmp_path/'resumed/checkpoint.pt', weights_only=True)
    for key in a['model']: torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    for pid, state in a['optimizer']['state'].items():
        for key, value in state.items():
            if isinstance(value,torch.Tensor): torch.testing.assert_close(value,b['optimizer']['state'][pid][key],rtol=0,atol=0)
            else: assert value == b['optimizer']['state'][pid][key]
    validation = [json.loads(x) for x in (tmp_path/'continuous/validation.jsonl').read_text().splitlines()]
    assert [r['step'] for r in validation] == [0,1,2,3]
    assert json.loads((tmp_path/'resumed/status.json').read_text())['phase'] == 'completed'
    with pytest.raises(FileExistsError): train.run(args("resumed", interrupted["checkpoint"]))

    # Same rolling checkpoint through a budget boundary, then recovery after the
    # final update was saved but the policy export had not been committed.
    rolling = tmp_path / 'rolling.pt'
    first = args('rolling_first'); first.total_steps = 1; first.rolling_checkpoint = str(rolling)
    train.run(first)
    second = args('rolling_second', str(rolling)); second.rolling_checkpoint = str(rolling)
    train.run(second)
    recovered = torch.load(rolling, weights_only=True)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], recovered['model'][key], rtol=0, atol=0)
    for pid, state in a['optimizer']['state'].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, recovered['optimizer']['state'][pid][key], rtol=0, atol=0)
    finalize = args('finalized', str(rolling)); finalize.rolling_checkpoint = str(rolling)
    finalize.finalize_checkpoint = True
    result = train.run(finalize)
    assert result['finalization_only'] and result['updates_this_run'] == 0
    assert result['updated_components'] is None and result['mean_step_seconds'] is None
    final_payload = torch.load(rolling, weights_only=True)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], final_payload['model'][key], rtol=0, atol=0)
    assert (tmp_path/'finalized/policy.pt').exists()
    assert not (tmp_path/'rolling_first/checkpoint.pt').exists()
    assert not (tmp_path/'rolling_second/latest.pt').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an allocated CUDA GPU; local Mac cannot execute this gate")
def test_cuda_master_checkpoint_restores_rng_and_next_update(tmp_path):
    torch.manual_seed(99)
    p = torch.nn.Sequential(torch.nn.Linear(4, 3),torch.nn.Dropout(.4)).cuda().bfloat16()
    opt = MasterAdamW(p.parameters(),lr=1e-3)
    x = torch.ones(2,4,device='cuda',dtype=torch.bfloat16)
    def step(m,o):
        o.zero_grad(set_to_none=True); loss=m(x).float().square().mean();loss.backward();o.step()
    step(p,opt);save_checkpoint(tmp_path/'cuda.pt',p,opt,{},1)
    step(p,opt);expected={k:v.clone() for k,v in p.state_dict().items()}
    restored = copy.deepcopy(p);other=MasterAdamW(restored.parameters())
    restore_checkpoint(tmp_path/'cuda.pt',restored,other);step(restored,other)
    for k,v in restored.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
