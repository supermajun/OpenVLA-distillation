"""Pretrained SmolVLM with parallel continuous action output, without LoRA."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from transformers import AutoProcessor, Idefics3Model

from .data import FEATURE_KEYS

STUDENT_LAYERS = {"500m": 32, "1b": 87, "2b": 188}
VISUAL_POOLING = "adaptive_bins_mean_v1"


def masked_mean(hidden, mask):
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


def pool_visual(hidden, batch_size, bins_per_camera=4):
    """Adaptive average pooling via fixed slices, including overlapping bins.

    PyTorch's adaptive_avg_pool1d delegates to adaptive_avg_pool2d, whose
    CUDA backward rejects strict determinism. Keep its floor/ceil boundaries
    and camera order using mean + slice backward instead. Mathematically
    equivalent; floating-point reduction order can differ from the old kernel.
    """
    if hidden.ndim != 3 or batch_size < 1 or bins_per_camera < 1:
        raise ValueError("Expected camera/token/feature tensor and positive batch/bins")
    if hidden.shape[0] == 0 or hidden.shape[0] % batch_size or hidden.shape[1] == 0:
        raise ValueError("Nonempty cameras and tokens, with cameras divisible by batch, required")
    tokens = hidden.shape[1]
    pooled = torch.stack([
        hidden[:, i * tokens // bins_per_camera:
               ((i + 1) * tokens + bins_per_camera - 1) // bins_per_camera].mean(dim=1)
        for i in range(bins_per_camera)
    ], dim=1)
    return pooled.reshape(batch_size, -1, hidden.shape[-1])


def expand_depth(backbone, target_layers):
    """Function-preserving residual expansion; every added block is trainable.

    Copies each source block's attention/MLP, zeroing only their residual output
    projections. Original layers retain their order. No new pretraining claimed.
    """
    original = list(backbone.text_model.layers)
    if target_layers < len(original):
        raise ValueError("Depth expansion cannot remove pretrained layers")
    layers, added = [], []
    feature_layer = None
    for i, layer in enumerate(original):
        if i == len(original) // 2:
            feature_layer = len(layers)
        layers.append(layer)
        copies = (i + 1) * target_layers // len(original) - i * target_layers // len(original) - 1
        for _ in range(copies):
            extra = copy.deepcopy(layer)
            with torch.no_grad():
                extra.self_attn.o_proj.weight.zero_()
                extra.mlp.down_proj.weight.zero_()
            added.append(len(layers))
            layers.append(extra)
    for i, layer in enumerate(layers):
        layer.self_attn.layer_idx = i
    backbone.text_model.layers = nn.ModuleList(layers)
    backbone.config.text_config.num_hidden_layers = len(layers)
    backbone.text_model.config.num_hidden_layers = len(layers)
    backbone._kd_feature_layer = feature_layer
    backbone._kd_added_layers = added


class Student(nn.Module):
    def __init__(self, backbone, chunk=8, action_dim=7, proprio_dim=8, teacher_dims=None):
        super().__init__()
        self.backbone = backbone
        h = backbone.config.text_config.hidden_size
        self.chunk, self.action_dim = chunk, action_dim
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, h), nn.GELU(), nn.Linear(h, h))
        self.action_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(),
                                         nn.Linear(h, chunk * action_dim))
        self.adapters = nn.ModuleDict({k: nn.Linear(h, dim, bias=False)
                                       for k, dim in (teacher_dims or {}).items()})
        self._language_hidden = None
        layers = self.backbone.text_model.layers
        self._hook = layers[getattr(backbone, "_kd_feature_layer", len(layers) // 2)].register_forward_hook(self._capture_language)

    def _capture_language(self, module, args, output):
        self._language_hidden = output[0] if isinstance(output, tuple) else output

    @classmethod
    def pretrained(cls, path, dtype=torch.bfloat16, teacher_dims=None, checkpointing=True, student_size="500m"):
        if student_size not in STUDENT_LAYERS:
            raise ValueError("Unsupported student size")
        backbone = Idefics3Model.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa",
                                                local_files_only=True)
        backbone.config.use_cache = False
        backbone.config.text_config.use_cache = False
        if student_size != "500m":
            expand_depth(backbone, STUDENT_LAYERS[student_size])
        if checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return cls(backbone, teacher_dims=teacher_dims).to(dtype=dtype)

    def forward(self, inputs, proprio, features=True):
        self._language_hidden = None
        out = self.backbone(**inputs, use_cache=False, return_dict=True)
        hidden = out.last_hidden_state
        mask = inputs["attention_mask"].bool()
        last_indices = torch.where(mask, torch.arange(mask.shape[1], device=mask.device), -1).max(1).values
        pre_action = hidden[torch.arange(len(hidden), device=hidden.device), last_indices] + self.proprio(proprio)
        actions = self.action_head(pre_action).reshape(-1, self.chunk, self.action_dim)
        result = {"actions": actions}
        if features:
            text_mask = mask & (inputs["input_ids"] != self.backbone.config.image_token_id)
            result["features"] = {
                "visual": pool_visual(out.image_hidden_states, len(hidden)),
                "language": masked_mean(self._language_hidden, text_mask),
                "pre_action": pre_action,
            }
        self._language_hidden = None
        return result

    def export_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("adapters.")}


class Collator:
    def __init__(self, model_path, device, dtype=torch.bfloat16):
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.processor.image_processor.do_image_splitting = False
        self.device, self.dtype = device, dtype

    def __call__(self, samples):
        prompts, images = [], []
        for s in samples:
            views = [Image.fromarray(x) for x in s["images"]]
            messages = [{"role": "user", "content": [*[{"type": "image"} for _ in views],
                {"type": "text", "text": f'Predict the robot actions to {s["instruction"]}.'}]}]
            prompts.append(self.processor.apply_chat_template(messages, add_generation_prompt=True))
            images.append(views)
        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
                  for k, v in inputs.items()}
        tensor = lambda values: torch.as_tensor(np.stack(values), device=self.device)
        batch = {"inputs": inputs, "proprio": tensor([s["proprio"] for s in samples]).to(self.dtype),
                 "actions": tensor([s["actions"] for s in samples]),
                 "valid": tensor([s["valid"] for s in samples])}
        if "teacher_actions" in samples[0]:
            batch["teacher_actions"] = tensor([s["teacher_actions"] for s in samples])
        if "teacher_features" in samples[0]:
            batch["teacher_features"] = {k: tensor([s["teacher_features"][k] for s in samples]) for k in FEATURE_KEYS}
        return batch


STRATEGIES = ("demo", "action", "feature", "both")


def masked_l1(pred, target, valid):
    if pred.shape != target.shape or valid.shape != pred.shape[:2] or not valid.any():
        raise ValueError("Invalid action targets/mask")
    error = (pred.float() - target.float()).abs()
    return (error * valid.unsqueeze(-1)).sum() / (valid.sum() * pred.shape[-1])


def loss_function(model, output, batch, strategy, action_weight=1.0, feature_weight=0.1):
    if strategy not in STRATEGIES or min(action_weight, feature_weight) < 0:
        raise ValueError("Invalid distillation configuration")
    terms = {"demo": masked_l1(output["actions"], batch["actions"], batch["valid"])}
    total = terms["demo"]
    if strategy in ("action", "both"):
        terms["action"] = masked_l1(output["actions"], batch["teacher_actions"].detach(), batch["valid"])
        total = total + action_weight * terms["action"]
    if strategy in ("feature", "both"):
        losses = []
        for key in FEATURE_KEYS:
            aligned = model.adapters[key](output["features"][key]).float()
            teacher = batch["teacher_features"][key].float().detach()
            if aligned.shape != teacher.shape:
                raise ValueError(f"Feature shape mismatch: {key}: {aligned.shape} != {teacher.shape}")
            losses.append(F.mse_loss(F.layer_norm(aligned, (aligned.shape[-1],)),
                                     F.layer_norm(teacher, (teacher.shape[-1],))))
        terms["feature"] = torch.stack(losses).mean()
        total = total + feature_weight * terms["feature"]
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite training loss")
    return total, {k: v.detach().float().item() for k, v in terms.items()}
