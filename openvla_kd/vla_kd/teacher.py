"""Frozen OpenVLA-OFT cache builder; sequential process, never co-resident with student.

Loads the upstream inference modules without importing its CUDA training/RLDS stack.
Uses an explicit bidirectional padding mask equivalent to OFT's modified Llama SDPA.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from .data import Dataset, FEATURE_KEYS
from .runtime import atomic_json, MemoryMonitor, select_device, sync


def bidirectional_mask(mask, dtype):
    if mask.ndim != 2:
        raise ValueError("Expected 2D padding mask")
    b, length = mask.shape
    result = torch.zeros((b, 1, length, length), dtype=dtype, device=mask.device)
    return result.masked_fill(~mask[:, None, None, :].bool(), torch.finfo(dtype).min)


def import_upstream(root):
    root = Path(root).resolve()
    # Namespace packages avoid unrelated imports from upstream __init__ files.
    for name in ("prismatic", "prismatic.extern", "prismatic.extern.hf", "prismatic.training",
                 "prismatic.vla", "prismatic.models"):
        if name in sys.modules:
            raise RuntimeError("Teacher must run in a fresh process without another prismatic import")
        package = types.ModuleType(name)
        package.__path__ = [str(root.joinpath(*name.split(".")))]
        sys.modules[name] = package
    cfg = importlib.import_module("prismatic.extern.hf.configuration_prismatic")
    modeling = importlib.import_module("prismatic.extern.hf.modeling_prismatic")
    processing = importlib.import_module("prismatic.extern.hf.processing_prismatic")
    heads = importlib.import_module("prismatic.models.action_heads")
    projectors = importlib.import_module("prismatic.models.projectors")
    return cfg, modeling, processing, heads, projectors


def load_component(module, path):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    module.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)


def load_teacher_stats(checkpoint):
    # Fine-tuned checkpoints keep base pretraining statistics in config.json.
    # The separate dataset_statistics.json is authoritative for downstream LIBERO.
    stats = json.loads((Path(checkpoint) / "dataset_statistics.json").read_text())
    if not stats or any("action" not in value or "proprio" not in value for value in stats.values()):
        raise ValueError("Teacher downstream normalization statistics are missing or malformed")
    return stats


class Teacher:
    def __init__(self, checkpoint, upstream, device):
        cfg, modeling, processing, heads, projectors = import_upstream(upstream)
        self.device, self.dtype = device, torch.bfloat16
        config = cfg.OpenVLAConfig.from_pretrained(checkpoint)
        config.text_config.use_cache = False
        self.model = modeling.OpenVLAForActionPrediction.from_pretrained(
            checkpoint, config=config, torch_dtype=self.dtype, attn_implementation="sdpa",
            low_cpu_mem_usage=True, local_files_only=True).to(device).eval()
        self.model.vision_backbone.set_num_images_in_input(2)
        self.model.norm_stats = load_teacher_stats(checkpoint)
        self.model.language_model.config.use_cache = False
        self.head = heads.L1RegressionActionHead().to(dtype=self.dtype)
        self.proprio = projectors.ProprioProjector(4096, 8).to(dtype=self.dtype)
        load_component(self.head, Path(checkpoint) / "action_head--150000_checkpoint.pt")
        load_component(self.proprio, Path(checkpoint) / "proprio_projector--150000_checkpoint.pt")
        self.head.to(device).eval()
        self.proprio.to(device).eval()
        for module in (self.model, self.head, self.proprio):
            module.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.image_processor = processing.PrismaticImageProcessor.from_pretrained(checkpoint)
        self.captured = {}
        self.model.language_model.register_forward_pre_hook(self._attention, with_kwargs=True)
        self.model.vision_backbone.register_forward_hook(self._vision)
        layers = self.model.language_model.model.layers
        layers[len(layers) // 2].register_forward_hook(self._language)
        self.head.register_forward_hook(self._actions)  # predict_action calls model directly, so also hook its MLP.
        self.head.model.register_forward_hook(self._actions)

    def _attention(self, module, args, kwargs):
        embeds = kwargs["inputs_embeds"]
        kwargs["attention_mask"] = bidirectional_mask(kwargs["attention_mask"], embeds.dtype)
        kwargs["use_cache"] = False
        return args, kwargs

    def _vision(self, module, args, output):
        self.captured["vision"] = output

    def _language(self, module, args, output):
        self.captured["language"] = output[0] if isinstance(output, tuple) else output

    def _actions(self, module, args, output):
        self.captured["actions"] = output

    @torch.inference_mode()
    def predict(self, sample, stats_key):
        self.captured = {}
        prompt = f'In: What action should the robot take to {sample["instruction"].lower()}?\nOut:'
        tokens = self.tokenizer(prompt, return_tensors="pt")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        prompt_length = tokens["input_ids"].shape[-1] - 1
        if tokens["input_ids"][0, -1].item() != 29871:
            prompt_length += 1
        pixels = [self.image_processor(Image.fromarray(im), return_tensors="pt")["pixel_values"] for im in sample["images"]]
        pixels = torch.cat(pixels, dim=1).to(self.device, self.dtype)
        _, action_hidden = self.model.predict_action(**tokens, pixel_values=pixels,
                                                     unnorm_key=stats_key, proprio=sample["proprio"],
                                                     proprio_projector=self.proprio, action_head=self.head)
        vision = self.captured["vision"]
        visual = F.adaptive_avg_pool1d(vision.reshape(2, -1, vision.shape[-1]).transpose(1, 2), 4).transpose(1, 2)
        patches = vision.shape[1] + 1  # Includes proprioception token.
        language = self.captured["language"][:, patches + 1: patches + prompt_length].mean(1)
        result = {"actions": self.captured["actions"].reshape(8, 7),
                  "visual": visual.reshape(8, -1), "language": language[0],
                  "pre_action": action_hidden.mean(1)[0]}
        result = {k: v.float().cpu().numpy() for k, v in result.items()}
        if not all(np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError("Non-finite teacher output")
        self.captured = {}
        return result


def build_cache(args):
    ds = Dataset(args.data, "all")
    teacher_stats = load_teacher_stats(args.model)
    if teacher_stats.get(ds.manifest["stats_key"]) != ds.manifest["stats"]:
        raise ValueError("Dataset normalization does not match the teacher checkpoint")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Cache destination must be empty; incomplete caches are never silently reused")
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    config_bytes = (Path(args.model) / "config.json").read_bytes()
    source_bytes = (Path(args.upstream) / "prismatic/extern/hf/modeling_prismatic.py").read_bytes()
    manifest = {"schema": 1, "teacher_kind": "openvla_oft", "complete": False,
                "dataset_fingerprint": ds.manifest["fingerprint"],
                "teacher_revision": "6d0231af0e48c5985f1ff86908f4674b84bc049b",
                "teacher_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "upstream_model_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "dtype": "bfloat16", "device": str(device), "attention": "explicit_bidirectional_padding_mask",
                "feature_spec": "vision:4 bins/camera; midpoint text tokens mean; pre-action:56 action tokens mean",
                "records": []}
    atomic_json(out / "manifest.json", manifest)
    with MemoryMonitor(device) as monitor:
        teacher = Teacher(args.model, args.upstream, device)
        for i in range(len(ds)):
            sample = ds[i]
            start = time.perf_counter()
            result = teacher.predict(sample, ds.manifest["stats_key"])
            np.savez_compressed(out / f'{sample["id"]}.npz', **result, sample_sha256=ds.records[i]["sha256"])
            manifest["records"].append(sample["id"])
            manifest["feature_dims"] = {k: int(result[k].shape[-1]) for k in FEATURE_KEYS}
            print(json.dumps({"sample": i + 1, "total": len(ds), "id": sample["id"],
                              "seconds": time.perf_counter() - start}), flush=True)
    manifest["complete"] = True
    manifest["sampled_peak_memory"] = monitor.peaks
    atomic_json(out / "manifest.json", manifest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/teacher")
    p.add_argument("--upstream", default="vendor/openvla-oft")
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--out", default="data/teacher_cache")
    p.add_argument("--device", default="mps")
    build_cache(p.parse_args())


if __name__ == "__main__":
    main()
