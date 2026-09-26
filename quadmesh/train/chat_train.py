"""Training stage for the chat model: synthetic tool-use / knowledge data (chat_data.py, exact-checkable) mixed with a
little general text so it can hold an ordinary conversation, plus your own recorded conversations if you have any."""
from __future__ import annotations
import math
import os

import numpy as np
import torch


def _batch(records, tok, mb, device, image_frac=0.15):
    from quadmesh.data.vla_data import to_patches
    import random
    ex = [next(records) for _ in range(mb)]
    rows, tgts, imgs, has_img = [], [], [], []
    for e in ex:
        prompt_ids = tok.encode(e["prompt"])
        ans_ids = tok.encode(e["answer"]) + [tok.eos_id]
        rows.append(prompt_ids + ans_ids)
        tgts.append([-100] * len(prompt_ids) + ans_ids)
        img = e.get("image")
        imgs.append(to_patches(img) if img is not None else None)
        has_img.append(img is not None)
    T = max(len(r) for r in rows)
    ids = torch.full((mb, T), tok.pad_id, dtype=torch.long)
    tg = torch.full((mb, T), -100, dtype=torch.long)
    for i, (r, t) in enumerate(zip(rows, tgts)):
        ids[i, :len(r)] = torch.tensor(r); tg[i, :len(t)] = torch.tensor(t)
    patches = None
    if any(has_img):
        Np = next(p.shape[0] for p in imgs if p is not None)
        arr = np.zeros((mb, Np, imgs[0].shape[1] if imgs[0] is not None else next(p for p in imgs if p is not None).shape[1]), dtype=np.float32)
        for i, p in enumerate(imgs):
            if p is not None:
                arr[i] = p
        patches = torch.from_numpy(arr)
    return ids.to(device), (patches.to(device) if patches is not None else None), tg.to(device)


def mixed_stream(seed=0, text_frac=0.15):
    """chat_data examples, with occasional plain SlimPajama continuation so the model keeps general fluency."""
    import random
    from quadmesh.chat_data import gen_example
    from quadmesh.data.streaming import slimpajama_stream
    rng = random.Random(seed)
    web = None
    while True:
        if rng.random() < text_frac:
            if web is None:
                web = iter(slimpajama_stream(seed=seed))
            try:
                doc = next(web)
                text = (doc.get("text") or "")[:600]
                if len(text) > 80:
                    cut = rng.randint(40, len(text) - 20)
                    yield {"prompt": f"USER: continue this text: {text[:cut]}\nASSISTANT:", "answer": " " + text[cut:cut + 200], "image": None}
                    continue
            except StopIteration:
                web = iter(slimpajama_stream(seed=seed + 1))
        ex = gen_example(); ex.setdefault("image", None)
        yield ex


def train_chat(tier, ckpt_dir: str, tokenizer_path: str, steps: int = 30000, mb: int = 8, accum: int = 4,
              device: str = "cuda", force: bool = False, log_every: int = 50, demo_dir: str = None):
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.model.chat import QuadmeshChat
    from quadmesh.role_shapes import role_shape

    path = os.path.join(ckpt_dir, "chat.pt")
    if os.path.exists(path) and not force:
        print(f"[chat] already trained -> {path}", flush=True)
        return path
    os.makedirs(ckpt_dir, exist_ok=True)
    tok = get_or_train(tokenizer_path)
    shape = role_shape(tier, "chat"); shape.vocab_size = tok.vocab_size
    model = QuadmeshChat(shape).to(device)
    peak = min(tier.lr, 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=peak, weight_decay=0.1, betas=(0.9, 0.95))
    stream = mixed_stream()
    warm = min(300, steps // 10)
    running = 0.0
    for step in range(steps):
        lr = peak * (step + 1) / warm if step < warm else peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(accum):
            ids, patches, tg = _batch(stream, tok, mb, device)
            loss = model(ids, patches, targets=tg)["loss"]
            (loss / accum).backward()
            running += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % log_every == 0:
            print(f"[chat] step {step+1}/{steps} loss {running/log_every:.3f} lr {lr:.2e}", flush=True)
            running = 0.0
    torch.save({"model": model.state_dict()}, path)
    return path
