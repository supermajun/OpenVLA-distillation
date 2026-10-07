"""Matched-budget full-parameter training, validation, and checkpoint round-trip."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import Dataset, digest
from .model import Student, Collator, STRATEGIES, STUDENT_LAYERS, loss_function, masked_l1
from .runtime import MemoryMonitor, atomic_json, restore_checkpoint, save_checkpoint, select_device, sync
from .optim import MasterAdamW


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


def run(args):
    if args.steps < 1 or args.accumulate < 1 or args.lr <= 0:
        raise ValueError("Positive steps, accumulation and learning rate required")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError("Run directory is not empty; use a new directory or --resume")
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    needs_features = args.strategy in ("feature", "both")
    if args.strategy != "demo" and not args.cache:
        raise ValueError("Distillation requires real teacher cache")
    train = Dataset(args.data, "train", cache=args.cache if args.strategy != "demo" else None,
                    require_features=needs_features)
    val = Dataset(args.data, "val")
    dims = train.cache_manifest["feature_dims"] if needs_features else None
    config = {**vars(args), "dataset_fingerprint": train.manifest["fingerprint"], "teacher_dims": dims,
              "teacher_cache_fingerprint": digest(train.cache_manifest) if args.strategy != "demo" else None}
    collator = Collator(args.model, device, dtype)
    with MemoryMonitor(device) as monitor:
        model = Student.pretrained(args.model, dtype, teacher_dims=dims, checkpointing=not args.no_checkpointing,
                                   student_size=args.student_size).to(device)
        optimizer = (MasterAdamW(model.parameters(), lr=args.lr) if args.optimizer == "master_fp32" else
                     torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, foreach=False))
        start = 0
        if args.resume:
            old, start = restore_checkpoint(args.resume, model, optimizer)
            for key in ("strategy", "seed", "dtype", "dataset_fingerprint", "accumulate", "lr", "teacher_dims",
                        "action_weight", "feature_weight", "teacher_cache_fingerprint", "model"):
                if old[key] != config[key]:
                    raise ValueError(f"Resume configuration mismatch: {key}")
            if old.get("student_size", "500m") != args.student_size:
                raise ValueError("Resume configuration mismatch: student_size")
            if old.get("optimizer", "bf16") != args.optimizer:
                raise ValueError("Resume configuration mismatch: optimizer")
        params = sum(p.numel() for p in model.parameters())
        policy_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("adapters."))
        print(json.dumps({"parameters": params, "deployment_parameters": policy_params,
                          "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                          "dtype": str(dtype), "device": str(device)}, indent=2), flush=True)
        initial_val = validate(model, val, collator)
        tracked = {"vision": model.backbone.vision_model.embeddings.patch_embedding.weight,
                   "language": model.backbone.text_model.layers[0].self_attn.q_proj.weight,
                   "action_head": model.action_head[-1].weight}
        if args.student_size != "500m":
            tracked["added_layer"] = model.backbone.text_model.layers[model.backbone._kd_added_layers[0]].self_attn.o_proj.weight
        before = {k: p.detach().cpu().clone() for k, p in tracked.items()}
        timings, history = [], []
        atomic_json(out / "config.json", config)
        for step in range(start, start + args.steps):
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
        optimizer.zero_grad(set_to_none=True)
        changed = {k: not torch.equal(before[k], p.detach().cpu()) for k, p in tracked.items()}
        if not all(changed.values()):
            raise AssertionError(f"Full-parameter update check failed: {changed}")
        final_val = validate(model, val, collator)
        state_dtypes = sorted({str(v.dtype) for s in optimizer.state.values() for k, v in s.items() if k != "step"})
        save_checkpoint(out / "checkpoint.pt", model, optimizer, config, start + args.steps)
        probe = collator([val[0]])
        model.eval()
        with torch.inference_mode():
            prediction_before = model(probe["inputs"], probe["proprio"], features=False)["actions"].float().cpu()
        # Deliberately alter weights to ensure restoration actually happens.
        with torch.no_grad():
            model.action_head[-1].weight.add_(1)
        restore_checkpoint(out / "checkpoint.pt", model, optimizer)
        model.eval()
        with torch.inference_mode():
            prediction_after = model(probe["inputs"], probe["proprio"], features=False)["actions"].float().cpu()
        torch.testing.assert_close(prediction_before, prediction_after, rtol=0, atol=0)
        torch.save({"schema": 1, "config": config, "model": model.export_state(),
                    "stats": train.manifest["stats"]}, out / "policy.pt")
        summary = {"strategy": args.strategy, "steps": start + args.steps,
                   "student_size": args.student_size, "language_layers": len(model.backbone.text_model.layers),
                   "optimizer": args.optimizer,
                   "parameters": params, "deployment_parameters": policy_params,
                   "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
                   "updated_components": changed, "optimizer_state_dtypes": state_dtypes,
                   "initial_val_l1": initial_val, "final_val_l1": final_val,
                   "validation_samples": len(val),
                   "mean_step_seconds": float(np.mean(timings)), "checkpoint_roundtrip_exact": True,
                   "dataset_fingerprint": train.manifest["fingerprint"],
                   "scope": "pilot training validation, not a converged policy or benchmark score"}
    summary["sampled_peak_memory"] = monitor.peaks
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/student")
    p.add_argument("--student-size", choices=list(STUDENT_LAYERS), default="500m")
    p.add_argument("--optimizer", choices=["bf16", "master_fp32"], default="bf16")
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--cache", default=None)
    p.add_argument("--strategy", choices=STRATEGIES, default="demo")
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--action-weight", type=float, default=1.0)
    p.add_argument("--feature-weight", type=float, default=0.1)
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--out", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
