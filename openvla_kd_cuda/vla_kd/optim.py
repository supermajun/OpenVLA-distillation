"""BF16 compute parameters with FP32 master weights and AdamW state.

Per-parameter updates avoid retaining a second full set of FP32 gradients.
This costs 16 bytes/parameter including BF16 gradients, before activations.
"""
import math
import torch


class MasterAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr <= 0 or eps <= 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("Invalid AdamW hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are unsupported")
                state = self.state[p]
                if not state:
                    state.update(step=0, master=p.detach().float().clone(),
                                 exp_avg=torch.zeros_like(p, dtype=torch.float32),
                                 exp_avg_sq=torch.zeros_like(p, dtype=torch.float32))
                state["step"] += 1
                g = p.grad.float()
                m, v, w = state["exp_avg"], state["exp_avg_sq"], state["master"]
                m.lerp_(g, 1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = v.sqrt().div_(math.sqrt(1 - b2 ** state["step"])).add_(group["eps"])
                w.mul_(1 - group["lr"] * group["weight_decay"])
                w.addcdiv_(m, denom, value=-group["lr"] / (1 - b1 ** state["step"]))
                p.copy_(w)
        return loss

    def load_state_dict(self, state_dict):
        # Base Optimizer casts floating state to parameter dtype, which would
        # irreversibly round the master to BF16. Restore FP32 from the source.
        super().load_state_dict(state_dict)
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups):
            for sid, p in zip(saved_group["params"], group["params"]):
                source = state_dict["state"].get(sid)
                if source:
                    for key in ("master", "exp_avg", "exp_avg_sq"):
                        self.state[p][key] = source[key].to(device=p.device, dtype=torch.float32, copy=True)
