"""Rank-r adapters for the 7B tier (single-H100 / memory-bound)."""
import math, torch, torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale


TARGETS = ("qkv", "proj", "w1", "w2", "w3")


def inject_lora(model: nn.Module, rank: int = 64, alpha: int = 16):
    for name, mod in model.named_children():
        if isinstance(mod, nn.Linear) and name in TARGETS:
            setattr(model, name, LoRALinear(mod, rank, alpha))
        else:
            inject_lora(mod, rank, alpha)
    # heads stay trainable: confidence calibration (Sec 12.3) must adapt
    for n, p in model.named_parameters():
        if "confidence_head" in n or "class_head" in n or "norm" in n:
            p.requires_grad = True
    return model
