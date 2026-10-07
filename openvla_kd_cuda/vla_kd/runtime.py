import json
import platform
import threading
import time
from pathlib import Path

import psutil
import torch


def select_device(name):
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable to this process. Do not silently substitute CPU; check sandbox/GPU access.")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run inside an allocated GPU job with a CUDA-enabled PyTorch build.")
    return torch.device(name)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


class MemoryMonitor:
    """Sampled high-water marks. RSS and driver allocation overlap: never add them."""
    def __init__(self, device):
        self.device = device
        self.peaks = {"rss_bytes": 0, "mps_allocated_bytes": 0, "mps_driver_bytes": 0}
        if device.type == "cuda":
            self.peaks.update(cuda_allocated_bytes=0, cuda_reserved_bytes=0)
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        values = {"rss_bytes": psutil.Process().memory_info().rss}
        if self.device.type == "mps":
            values.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                          mps_driver_bytes=torch.mps.driver_allocated_memory())
        elif self.device.type == "cuda":
            values.update(cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                          cuda_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        for k, v in values.items():
            self.peaks[k] = max(self.peaks[k], v)

    def _loop(self):
        while not self.stop.wait(0.1):
            self.sample()

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()


def atomic_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(p)


def doctor(out="runs/doctor.json", device_name="mps"):
    device = select_device(device_name)
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 7)).to(device, torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    x = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
    before = next(model.parameters()).detach().clone()
    losses = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = model(x).float().square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
        opt.step()
        losses.append(loss.item())
    sync(device)
    changed = not torch.equal(before, next(model.parameters()))
    if not changed:
        raise AssertionError("BF16 optimizer did not update parameters")
    result = {"torch": torch.__version__, "python": platform.python_version(), "macos": platform.mac_ver()[0],
              "device": str(device), "dtype": "bfloat16", "optimizer": "AdamW foreach=False",
              "state_dtypes": sorted({str(v.dtype) for s in opt.state.values() for k, v in s.items() if k != "step"}),
              "losses": losses, "parameters_changed": changed, "passed": True,
              "scope": "small operator probe, not the 500M acceptance run"}
    atomic_json(out, result)
    print(json.dumps(result, indent=2))


def save_checkpoint(path, model, optimizer, config, step):
    payload = {"schema": 1, "config": config, "step": step,
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "rng_cpu": torch.get_rng_state()}
    if next(model.parameters()).device.type == "mps":
        payload["rng_mps"] = torch.mps.get_rng_state()
    elif next(model.parameters()).device.type == "cuda":
        payload["rng_cuda"] = torch.cuda.get_rng_state_all()
    p = Path(path)
    tmp = p.with_suffix(".partial")
    torch.save(payload, tmp)
    tmp.replace(p)


def restore_checkpoint(path, model, optimizer=None, expected_config=None):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema") != 1:
        raise ValueError("Unsupported checkpoint schema")
    if expected_config is not None:
        defaults = {"student_size": "500m", "optimizer": "bf16", "deterministic": False}
        keys = ("model", "student_size", "optimizer", "dtype", "device", "strategy", "seed", "lr", "accumulate",
                "dataset_fingerprint", "teacher_cache_fingerprint", "teacher_dims", "action_weight", "feature_weight", "deterministic", "visual_pooling")
        for key in keys:
            if key in expected_config and payload["config"].get(key, defaults.get(key)) != expected_config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        # Release existing accelerator moments before restoring replacements.
        optimizer.state.clear()
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["rng_cpu"])
        if "rng_mps" in payload and next(model.parameters()).device.type == "mps":
            torch.mps.set_rng_state(payload["rng_mps"])
        if next(model.parameters()).device.type == "cuda":
            if "rng_cuda" not in payload:
                raise ValueError("CUDA continuation requires a CUDA checkpoint with RNG state; use a fresh run for cross-device comparisons")
            if len(payload["rng_cuda"]) != torch.cuda.device_count():
                raise ValueError("Visible CUDA device count changed since checkpoint")
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
    return payload["config"], payload["step"]
