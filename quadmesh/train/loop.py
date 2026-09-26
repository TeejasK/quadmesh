"""
The training loop. IDENTICAL for every tier and every role.

Everything that varies lives in quadmesh/config.py (hyperparameters) and
quadmesh/data/streaming.py (which records arrive). Do not add tier branches
here — if you find yourself wanting to, it belongs in one of those two files.
"""

from __future__ import annotations
from quadmesh.role_shapes import role_shape

import math
import os
import time
from dataclasses import asdict
from typing import Optional

import torch

from quadmesh.config import TierConfig
from quadmesh.model.transformer import build_role_model


def cosine_lr(step: int, tier: TierConfig, total_steps: int = None, warmup: int = None) -> float:
    """Warmup-then-cosine schedule. Step 0 → 0, step warmup → peak, last step → min_lr."""
    total_steps = total_steps or tier.max_steps
    warmup = warmup if warmup is not None else tier.warmup_steps
    warmup = min(warmup, max(total_steps // 10, 5))
    if step < warmup:
        return tier.lr * step / max(warmup, 1)
    total = max(total_steps - warmup, 1)
    prog = min((step - warmup) / total, 1.0)
    return tier.min_lr + 0.5 * (tier.lr - tier.min_lr) * (1 + math.cos(math.pi * prog))


def make_optimizer(model, tier: TierConfig):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": tier.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tier.lr, betas=(tier.beta1, tier.beta2), fused=torch.cuda.is_available(),
    )


def maybe_lora(model, tier: TierConfig):
    """7B only. Freezes the base and injects rank-r adapters."""
    if not tier.use_lora:
        return model
    from quadmesh.model.lora import inject_lora
    return inject_lora(model, rank=tier.lora_rank)


def train_role(role: str, stage: int, tier: TierConfig, tokenizer,
               ckpt_dir: str, resume: bool = True, log_every: int = 20,
               save_every: int = 500, device: str = "cuda"):
    from quadmesh.data.streaming import build_loader

    shape = role_shape(tier, role)
    shape.vocab_size = tokenizer.vocab_size
    shape.max_seq_len = max(shape.max_seq_len, tier.seq_len)

    model = build_role_model(role, shape).to(device)
    model.gradient_checkpointing = tier.gradient_checkpointing
    model = maybe_lora(model, tier)

    opt = make_optimizer(model, tier)
    scaler_dtype = torch.bfloat16 if tier.precision == "bf16" else torch.float16

    target_steps = max(tier.max_steps // 10, 168) if stage == 2 else tier.max_steps
    save_interval = min(save_every, max(target_steps // 2, 50))

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{role}.stage{stage}.pt")
    start_step = 0
    if resume and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        start_step = sd.get("step", 0)
        print(f"[{role}] Resumed checkpoint at step {start_step}/{target_steps}", flush=True)
        if start_step >= target_steps:
            print(f"[{role}] ✓ Stage {stage} is already fully trained ({start_step}/{target_steps} steps). Proceeding!", flush=True)
            return ckpt_path

    print(f"[{role}] tier={tier.name} params={model.n_params()/1e6:.1f}M "
          f"stage={stage} steps={target_steps} tokens/step={tier.tokens_per_step:,}", flush=True)
    print(f"[{role}] Connecting to data stream with multi-threaded prefetching...", flush=True)

    loader = build_loader(role, stage, tier, tokenizer, device=device)
    model.train()
    step, t0 = start_step, time.time()
    log_every = 5  # log frequently so user sees real-time progress

    print(f"[{role}] Training started at step {step}...", flush=True)

    while step < target_steps:
        lr = cosine_lr(step, tier, total_steps=target_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(tier.grad_accum_steps):
            try:
                x, y = next(loader)
            except StopIteration:
                print(f"[{role}] stream exhausted at step {step}", flush=True)
                step = target_steps
                break
            with torch.autocast("cuda", dtype=scaler_dtype):
                loss = model(x, targets=y)["loss"] / tier.grad_accum_steps
            loss.backward()
            accumulated += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), tier.grad_clip)
        opt.step()
        step += 1

        if step == 1 or step % log_every == 0 or step == target_steps:
            elapsed = time.time() - t0
            tok_s = (log_every if step > log_every else step) * tier.tokens_per_step / max(elapsed, 1e-4)
            pct = (step / target_steps) * 100
            print(f"[{role}] step {step}/{target_steps} ({pct:.1f}%) | "
                  f"loss={accumulated:.4f} | lr={lr:.2e} | {tok_s/1e3:.1f}k tok/s", flush=True)
            t0 = time.time()

        if step % save_interval == 0 or step >= target_steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "role": role, "stage": stage,
                        "tier": tier.name, "shape": asdict(shape)}, ckpt_path)
            print(f"[{role}] 💾 Saved checkpoint at step {step}/{target_steps} -> {ckpt_path}", flush=True)

    print(f"[{role}] ✓ Finished stage {stage} ({target_steps} steps)", flush=True)
    return ckpt_path
