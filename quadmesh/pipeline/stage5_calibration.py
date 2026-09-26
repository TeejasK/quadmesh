"""
Stage 5 — Calibration (Sec 12.3, Sec 15).

Tunes the model's confidence_head against real-world outcome density, not
against training loss. Needs accumulated production/verification-stack
outcomes, not a download — so this reads a log of (input, verification
result) pairs written during Stage 3 grading (and, once live, real usage).
"""
from __future__ import annotations
from quadmesh.role_shapes import role_shape
import json, os, torch
import torch.nn.functional as F

from quadmesh.config import TierConfig
from quadmesh.model.transformer import build_role_model


def load_outcomes(path: str):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out.append((r["text"], 1.0 if r["passed"] else 0.0))
    return out


def run_stage5(role: str, tier: TierConfig, tokenizer, ckpt_dir: str,
              device: str = "cuda", epochs: int = 2):
    shape = role_shape(tier, role)
    shape.vocab_size = tokenizer.vocab_size
    model = build_role_model(role, shape).to(device)

    # start from stage4 if it exists (preference-tuned roles), else stage2, else stage1
    src = None
    for s in ("stage4", "stage2", "stage1"):
        cand = f"{ckpt_dir}/{s}/{role}.{s}.pt"
        if os.path.exists(cand):
            src = cand
            break
    if not src:
        raise FileNotFoundError(f"No previous checkpoint found for {role} under {ckpt_dir}")

    sd = torch.load(src, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])

    # freeze everything except the confidence head — calibration only
    for n, p in model.named_parameters():
        p.requires_grad = "confidence_head" in n

    opt = torch.optim.Adam(model.confidence_head.parameters(), lr=1e-3)
    pairs_file = f"{ckpt_dir}/stage3/{role}.pairs.jsonl"
    outcomes = load_outcomes(pairs_file) if os.path.exists(pairs_file) else []

    for ep in range(epochs):
        for i, (text, label) in enumerate(outcomes):
            ids = torch.tensor([tokenizer.encode(text)], device=device)
            if ids.numel() < 2:
                continue
            out = model(ids)
            target = torch.tensor([label], device=device)
            loss = F.binary_cross_entropy(out["confidence"], target)
            opt.zero_grad(); loss.backward(); opt.step()
            if i % 50 == 0 or i == len(outcomes) - 1:
                print(f"[stage5][{role}] epoch {ep} {i+1}/{len(outcomes)} calib_loss {loss.item():.4f}", flush=True)

    out_dir = f"{ckpt_dir}/stage5"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{role}.stage5.pt"
    torch.save({"model": model.state_dict(), "role": role, "tier": tier.name}, path)
    return path
