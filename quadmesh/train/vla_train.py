"""Training stage for the VLA: real GroundCUA screenshots, VideoCAD long-horizon CAD sessions, and your
recorded Blender demonstrations (quadmesh.data.vla_data.vla_stream).

Two things changed from the GroundCUA-only version now that VideoCAD is in the mix:
  1. VideoCAD examples carry up to 12 actions of history text (history_text() in vla_data.py) where GroundCUA
     examples are single clicks with no history - so a batch built from both without any length control pads
     every short GroundCUA row out to the longest VideoCAD row in the batch, wasting most of the compute on
     padding. `_batch` now pools several candidate examples and buckets by length before committing to a
     batch, so same-length examples land together instead of a random mix each step.
  2. Nothing previously capped a sequence to the model's own shp.max_seq_len - fine when every example was
     one click, not fine once VideoCAD histories can run long. `_batch` now truncates the *prefix* (task +
     older history) from the LEFT when a row is too long, and always keeps the action tokens intact at the
     end - losing old history is a much smaller problem than truncating the label the model is trained to
     predict.
Per-source running loss is tracked and logged, since videocad_stream() can legitimately yield nothing (no
mirror/DOI configured yet - see data/vla_data.py) and you want that visible in training logs, not silently
inferred from the loss curve alone.
"""
from __future__ import annotations
import math
import os
from collections import defaultdict

import numpy as np
import torch


def _sample_pool(stream, pool_size: int):
    """Pull `pool_size` examples so _batch can bucket by length instead of taking whatever order they arrive in."""
    return [next(stream) for _ in range(pool_size)]


def _encoded_len(e, tok) -> int:
    return len(tok.encode(e["instruction"] + "\n")) + len(tok.encode(e["action"])) + 1


def _batch(pool: list, tok, mb: int, device, max_seq_len: int):
    """Takes the mb shortest-still-unused examples from `pool` (bucketing), truncates each row's PREFIX from
    the left to fit max_seq_len (the action + eos always survive intact), and returns one padded batch."""
    pool.sort(key=lambda e: _encoded_len(e, tok))
    ex, remaining = pool[:mb], pool[mb:]

    patches = torch.from_numpy(np.stack([to_patches_cached(e["image"]) for e in ex]))
    rows, tgts = [], []
    for e in ex:
        act = tok.encode(e["action"]) + [tok.eos_id]
        budget = max(1, max_seq_len - len(act))
        prefix = tok.encode(e["instruction"] + "\n")
        if len(prefix) > budget:
            prefix = prefix[-budget:]           # drop OLDEST history first, keep the action's own context
        rows.append(prefix + act)
        tgts.append([-100] * len(prefix) + act)
    T = min(max(len(r) for r in rows), max_seq_len)
    ids = torch.full((mb, T), tok.pad_id, dtype=torch.long)
    tg = torch.full((mb, T), -100, dtype=torch.long)
    for i, (r, t) in enumerate(zip(rows, tgts)):
        r, t = r[:T], t[:T]
        ids[i, :len(r)] = torch.tensor(r)
        tg[i, :len(t)] = torch.tensor(t)
    sources = [e.get("source", "?") for e in ex]
    return patches.to(device), ids.to(device), tg.to(device), sources, remaining


def to_patches_cached(img):
    from quadmesh.data.vla_data import to_patches
    return to_patches(img)


def train_vla(tier, ckpt_dir: str, tokenizer_path: str, steps: int = 20000, demo_dir=None,
              mb: int | None = None, accum: int | None = None, device: str = "cuda", force: bool = False,
              log_every: int = 50, pool_size: int | None = None):
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.data.vla_data import vla_stream
    from quadmesh.role_shapes import role_shape
    from quadmesh.model.vla import QuadmeshVLA

    path = os.path.join(ckpt_dir, "vla.pt")
    if os.path.exists(path) and not force:
        print(f"[vla] already trained -> {path}", flush=True)
        return path
    os.makedirs(ckpt_dir, exist_ok=True)
    tok = get_or_train(tokenizer_path)
    shp = role_shape(tier, "vla"); shp.vocab_size = tok.vocab_size

    # Batch/accum default from the tier's own schedule rather than a flat mb=4 - a role sized like vla
    # (0.20 of the budget, see role_shapes.py) on a larger tier can afford a bigger micro-batch than 4, and
    # a small tier shouldn't silently over-allocate. Images are heavier than the pure-text roles' tokens, so
    # this halves the tier's usual micro_batch_size as a memory-safety margin rather than using it directly.
    if mb is None:
        mb = max(2, tier.micro_batch_size // 2)
    if accum is None:
        accum = tier.grad_accum_steps
    if pool_size is None:
        pool_size = mb * 6   # enough spare examples per draw for length-bucketing to actually help

    model = QuadmeshVLA(shp).to(device)
    peak = min(tier.lr, 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=peak, weight_decay=0.05, betas=(0.9, 0.95))
    stream = vla_stream(demo_dir)
    warm = min(200, steps // 10)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    running = 0.0
    src_loss = defaultdict(float)
    src_count = defaultdict(int)
    pool: list = []
    for step in range(steps):
        lr = peak * (step + 1) / warm if step < warm else peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(accum):
            if len(pool) < mb:
                pool.extend(_sample_pool(stream, pool_size))
            patches, ids, tg, sources, pool = _batch(pool, tok, mb, device, shp.max_seq_len)
            with amp:
                loss = model(patches, ids, targets=tg)["loss"]
            (loss / accum).backward()
            lv = loss.item()
            running += lv / accum
            for s in sources:
                src_loss[s] += lv / len(sources)
                src_count[s] += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % log_every == 0:
            mix = ", ".join(f"{s}={src_count[s]}" for s in sorted(src_count))
            print(f"[vla] step {step+1}/{steps} loss {running/log_every:.3f} lr {lr:.2e} mix[{mix}]", flush=True)
            if "videocad" not in src_count:
                print("[vla] WARNING: zero videocad examples seen so far - check "
                      "VIDEOCAD_DATAVERSE_DOI / VIDEOCAD_HF_CANDIDATES in data/vla_data.py", flush=True)
            running = 0.0
            src_loss.clear(); src_count.clear()
    torch.save({"model": model.state_dict()}, path)
    return path
