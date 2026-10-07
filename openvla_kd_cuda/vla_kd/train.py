"""Matched-budget full-parameter training, validation, and checkpoint round-trip."""
import argparse
import gc
import json
import os
import platform
from pathlib import Path
import time

import numpy as np
import torch

from .data import Dataset, digest
from .model import Student, Collator, STRATEGIES, STUDENT_LAYERS, VISUAL_POOLING, loss_function, masked_l1
from .runtime import MemoryMonitor, atomic_json, restore_checkpoint, save_checkpoint, select_device, sync
from .optim import MasterAdamW
from .control import StopRequest


@torch.inference_mode()
def validate(model, dataset, collator, max_samples=None):
    model.eval()
    numerator, count = 0.0, 0
    for i in range(min(max_samples if max_samples is not None else len(dataset), len(dataset))):
        b = collator([dataset[i]])
        pred = model(b["inputs"], b["proprio"], features=False)["actions"]
        n = int(b["valid"].sum()) * pred.shape[-1]
        numerator += masked_l1(pred, b["actions"], b["valid"]).item() * n
        count += n
    return numerator / count


def _run(args, stop):
    if args.steps < 1 or args.accumulate < 1 or args.lr <= 0:
        raise ValueError("Positive steps, accumulation and learning rate required")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Always use a new output directory, including continuation runs")
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This implementation is single-process, single-GPU; do not launch with multi-rank torchrun")
    if device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Allocated GPU does not support BF16")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(args.seed)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    needs_features = args.strategy in ("feature", "both")
    if args.strategy != "demo" and not args.cache:
        raise ValueError("Distillation requires real teacher cache")
    train = Dataset(args.data, "train", cache=args.cache if args.strategy != "demo" else None,
                    require_features=needs_features)
    val = Dataset(args.data, "val")
    dims = train.cache_manifest["feature_dims"] if needs_features else None
    config = {**vars(args), "visual_pooling": VISUAL_POOLING,
              "dataset_fingerprint": train.manifest["fingerprint"], "teacher_dims": dims,
              "teacher_cache_fingerprint": digest(train.cache_manifest) if args.strategy != "demo" else None}
    environment = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                   "cuda_runtime": torch.version.cuda, "device": str(device)}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        environment.update(gpu=torch.cuda.get_device_name(device), gpu_total_bytes=total, gpu_free_bytes=free,
                           compute_capability=torch.cuda.get_device_capability(device))
        if args.student_size == "2b" and args.optimizer == "master_fp32" and free < 2005878200 * 16 + 2 * 1024**3:
            raise RuntimeError("Insufficient free GPU memory for 2B FP32-master training; request a larger/full GPU, not a small MIG slice")
    atomic_json(out / "environment.json", environment)
    def status(phase, **extra):
        atomic_json(out / "status.json", {"phase": phase, "time_unix": time.time(), **extra})
    def validation_record(step, value):
        row = {"step": step, "validation_l1": value, "samples": len(val)}
        with (out / "validation.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps({"event": "validation", **row}), flush=True)
    status("initializing")
    collator = Collator(args.model, device, dtype)
    with MemoryMonitor(device) as monitor:
        model = Student.pretrained(args.model, dtype, teacher_dims=dims, checkpointing=not args.no_checkpointing,
                                   student_size=args.student_size).to(device)
        optimizer = (MasterAdamW(model.parameters(), lr=args.lr) if args.optimizer == "master_fp32" else
                     torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, foreach=False))
        start = 0
        if args.resume:
            old, start = restore_checkpoint(args.resume, model, optimizer, expected_config=config)
            for key in ("strategy", "seed", "dtype", "dataset_fingerprint", "accumulate", "lr", "teacher_dims",
                        "action_weight", "feature_weight", "teacher_cache_fingerprint", "model"):
                if old[key] != config[key]:
                    raise ValueError(f"Resume configuration mismatch: {key}")
            if old.get("student_size", "500m") != args.student_size:
                raise ValueError("Resume configuration mismatch: student_size")
            if old.get("optimizer", "bf16") != args.optimizer:
                raise ValueError("Resume configuration mismatch: optimizer")
            if old.get("deterministic", False) != args.deterministic:
                raise ValueError("Resume configuration mismatch: deterministic")
        target = args.total_steps if args.total_steps is not None else start + args.steps
        if target < start or (target == start and not getattr(args, "finalize_checkpoint", False)):
            raise ValueError("Target must exceed the checkpoint step (or use --finalize-checkpoint at the same step)")
        rolling = getattr(args, "rolling_checkpoint", None)
        checkpoint_path = Path(rolling) if rolling else out / "checkpoint.pt"
        latest_path = Path(rolling) if rolling else out / "latest.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        params = sum(p.numel() for p in model.parameters())
        policy_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("adapters."))
        print(json.dumps({"parameters": params, "deployment_parameters": policy_params,
                          "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                          "dtype": str(dtype), "device": str(device)}, indent=2), flush=True)
        initial_val = validate(model, val, collator)
        validation_record(start, initial_val)
        tracked = {"vision": model.backbone.vision_model.embeddings.patch_embedding.weight,
                   "language": model.backbone.text_model.layers[0].self_attn.q_proj.weight,
                   "action_head": model.action_head[-1].weight}
        if args.student_size != "500m":
            tracked["added_layer"] = model.backbone.text_model.layers[model.backbone._kd_added_layers[0]].self_attn.o_proj.weight
        before = {k: p.detach().cpu().clone() for k, p in tracked.items()}
        timings, history = [], []
        atomic_json(out / "config.json", config)
        for step in range(start, target):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            sync(device)
            begin = time.perf_counter()
            terms_sum, ids = {}, []
            for micro in range(args.accumulate):
                index = int(np.random.default_rng(args.seed + step * args.accumulate + micro).integers(len(train)))
                sample = train[index]
                ids.append(sample["id"])
                batch = collator([sample])
                output = model(batch["inputs"], batch["proprio"], features=needs_features)
                loss, terms = loss_function(model, output, batch, args.strategy,
                                             args.action_weight, args.feature_weight)
                (loss / args.accumulate).backward()
                for k, v in {**terms, "total": loss.item()}.items():
                    terms_sum[k] = terms_sum.get(k, 0) + v / args.accumulate
            missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            if missing:
                raise AssertionError(f"Trainable parameters without gradients: {missing[:10]}")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
            optimizer.step()
            model._language_hidden = None
            sync(device)
            elapsed = time.perf_counter() - begin
            timings.append(elapsed)
            record = {"step": step + 1, **terms_sum, "grad_norm": grad_norm.item(),
                      "seconds": elapsed, "sample_ids": ids}
            history.append(record)
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
            del batch, output, loss
            status("training", step=step + 1, target=target, loss=record["total"])
            if args.validate_every and (step + 1) % args.validate_every == 0 and step + 1 < target:
                optimizer.zero_grad(set_to_none=True)
                validation_record(step + 1, validate(model, val, collator))
            if stop.requested or (args.checkpoint_every and (step + 1) % args.checkpoint_every == 0 and step + 1 < target):
                optimizer.zero_grad(set_to_none=True)
                status("saving_checkpoint", step=step + 1, target=target)
                save_checkpoint(latest_path, model, optimizer, config, step + 1)
                if stop.requested:
                    result = {"status": "interrupted", "step": step + 1, "target": target,
                              "signal": stop.signal, "checkpoint": str(latest_path)}
                    status("interrupted", **{k: v for k, v in result.items() if k != "status"})
                    print(json.dumps(result), flush=True)
                    return result
        optimizer.zero_grad(set_to_none=True)
        changed = {k: not torch.equal(before[k], p.detach().cpu()) for k, p in tracked.items()}
        if target > start and not all(changed.values()):
            raise AssertionError(f"Full-parameter update check failed: {changed}")
        final_val = validate(model, val, collator)
        validation_record(target, final_val)
        state_dtypes = sorted({str(v.dtype) for s in optimizer.state.values() for k, v in s.items() if k != "step"})
        status("saving_final_checkpoint", step=target)
        save_checkpoint(checkpoint_path, model, optimizer, config, target)
        probe = collator([val[0]])
        model.eval()
        with torch.inference_mode():
            prediction_before = model(probe["inputs"], probe["proprio"], features=False)["actions"].float().cpu()
        # Deliberately alter weights to ensure restoration actually happens.
        with torch.no_grad():
            model.action_head[-1].weight.add_(1)
        restore_checkpoint(checkpoint_path, model, optimizer)
        model.eval()
        with torch.inference_mode():
            prediction_after = model(probe["inputs"], probe["proprio"], features=False)["actions"].float().cpu()
        torch.testing.assert_close(prediction_before, prediction_after, rtol=0, atol=0)
        torch.save({"schema": 1, "config": config, "model": model.export_state(),
                    "stats": train.manifest["stats"]}, out / "policy.partial")
        (out / "policy.partial").replace(out / "policy.pt")
        summary = {"strategy": args.strategy, "steps": target,
                   "start_step": start, "updates_this_run": target - start,
                   "student_size": args.student_size, "language_layers": len(model.backbone.text_model.layers),
                   "optimizer": args.optimizer,
                   "parameters": params, "deployment_parameters": policy_params,
                   "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
                   "updated_components": changed if target > start else None,
                   "finalization_only": target == start, "optimizer_state_dtypes": state_dtypes,
                   "initial_val_l1": initial_val, "final_val_l1": final_val,
                   "validation_samples": len(val),
                   "mean_step_seconds": float(np.mean(timings)) if timings else None, "checkpoint_roundtrip_exact": True,
                   "dataset_fingerprint": train.manifest["fingerprint"],
                   "scope": "pilot training validation, not a converged policy or benchmark score"}
    summary["sampled_peak_memory"] = monitor.peaks
    atomic_json(out / "summary.json", summary)
    status("completed", step=target)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run(args):
    for key in ("validate_every", "checkpoint_every"):
        if getattr(args, key) < 0:
            raise ValueError(f"{key} must be nonnegative")
    out = Path(args.out)
    own_output = not out.exists() or not any(out.iterdir())
    try:
        with StopRequest() as stop:
            return _run(args, stop)
    except BaseException as exc:
        if own_output and out.exists():
            atomic_json(out / "status.json", {"phase": "failed", "error": repr(exc), "time_unix": time.time()})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/student")
    p.add_argument("--student-size", choices=list(STUDENT_LAYERS), default="2b")
    p.add_argument("--optimizer", choices=["bf16", "master_fp32"], default="master_fp32")
    p.add_argument("--data", default="data/expanded")
    p.add_argument("--cache", default="data/expanded_teacher_cache")
    p.add_argument("--strategy", choices=STRATEGIES, default="both")
    p.add_argument("--device", default="cuda", choices=["mps", "cpu", "cuda"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--steps", type=int, default=500, help="Updates to add, unless --total-steps is supplied")
    p.add_argument("--total-steps", type=int, help="Stop at this absolute update count, including restored updates")
    p.add_argument("--validate-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--deterministic", action="store_true", help="Deterministic CUDA algorithms and math SDPA; slower, useful for resume validation")
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--action-weight", type=float, default=1.0)
    p.add_argument("--feature-weight", type=float, default=0.1)
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--rolling-checkpoint", help="Pipeline-owned full checkpoint, atomically replaced; avoids duplicate full weights")
    p.add_argument("--finalize-checkpoint", action="store_true", help="Permit export/validation when restored step already equals target")
    p.add_argument("--out", required=True)
    result = run(p.parse_args())
    if result.get("status") == "interrupted":
        raise SystemExit(75)


if __name__ == "__main__":
    main()
