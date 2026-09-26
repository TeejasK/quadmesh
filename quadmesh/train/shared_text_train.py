"""
Train the shared generative text model (model/shared_backbone.py: SharedTextModel).

Two phases, same loop:

  pretrain  packed windows of SlimPajama (+ Text2CAD in full) -> general language + CAD vocabulary.
            Expensive. Run once per tier. Budget in TOKENS (--tokens, or --budget chinchilla|4x|full).
  sft       multi-role instruction tuning from role_gen_data. Cheap, endless, verifier-grounded, re-runnable
            whenever the generators improve. Rows of ALL roles share every batch; loss is on answer tokens only.

What this fixes compared with the first shared trainer:
  * next-token targets (the model shifts internally; the first trainer trained a copy objective)
  * bf16 autocast, gradient clipping, fused AdamW, optional gradient checkpointing
  * atomic checkpoints + exact resume (Modal jobs get preempted and time out)
  * SFT examples are packed (~95% of tokens are real) instead of padded per role
  * held-out per-role validation loss every --eval-every steps, written to metrics.jsonl
  * planner REPAIR examples are split correctly; the roles actually get their own data

Cross-example attention inside a packed row is not masked (the shared attention has no varlen mask). Examples are
short and separated by <eos>; this is the standard trade-off and the per-role validation loss would show a problem.
"""
from __future__ import annotations
import json
import math
import os
import random
import time
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from quadmesh.config import TIERS
from quadmesh.model.shared_backbone import SharedTextModel
from quadmesh.pipeline.datasets import role_gen_data as RG
from quadmesh.roles_io import encode_example


# ------------------------------------------------------------------------------------------------ data
class SFTRows(IterableDataset):
    """Endless rows of `seq_len` tokens: several role examples packed back to back. Yields (ids, targets)."""

    def __init__(self, tok, seq_len: int, seed: int, split: str = "train", reviewed_path: Optional[str] = None,
                 weights: Optional[dict] = None, roles: Optional[tuple] = None):
        self.tok, self.L, self.seed, self.split = tok, seq_len, seed, split
        self.reviewed_path, self.weights, self.roles = reviewed_path, weights, roles

    def _examples(self, seed):
        if self.roles:                                   # a fixed role, for validation
            random.seed(seed)
            while True:
                r = RG.make_example(self.roles[0], self.split)
                if r:
                    yield (self.roles[0],) + r
        else:
            yield from RG.stream(self.split, seed=seed, weights=self.weights, reviewed_path=self.reviewed_path)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        pad = self.tok.pad_id
        row_i, row_t = [], []
        for role, p, a in self._examples(self.seed * 1009 + wid):
            enc = encode_example(self.tok, role, p, a, max_len=self.L)
            if enc is None:
                continue
            ids, pl = enc
            if len(row_i) + len(ids) > self.L:
                n = self.L - len(row_i)
                yield (torch.tensor(row_i + [pad] * n), torch.tensor(row_t + [-100] * n))
                row_i, row_t = [], []
            row_i += ids
            row_t += [-100] * pl + ids[pl:]


class PretrainRows(IterableDataset):
    """Packed windows of `seq_len` tokens from SlimPajama + Text2CAD. Yields (ids, ids)."""

    def __init__(self, tok, seq_len: int, seed: int, budget_tokens: int, p_cad: float = 0.15):
        self.tok, self.L, self.seed, self.budget, self.p_cad = tok, seq_len, seed, budget_tokens, p_cad

    def __iter__(self):
        from quadmesh.data import streaming as S
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        web = S.slimpajama_stream(seed=self.seed * 31 + wid)
        cad = S.text2cad_stream()
        recs = S._manual_interleave(web, cad, self.p_cad, seed=self.seed * 31 + wid)
        it = S.PackedTokenIterator(recs, self.tok, self.L - 1, max(self.budget // nw, 1))
        for w in it:
            yield w, w


def _loader(ds, mb: int, workers: int):
    kw = dict(prefetch_factor=4, persistent_workers=True) if workers else {}
    return DataLoader(ds, batch_size=mb, num_workers=workers, **kw)


# ------------------------------------------------------------------------------------------------ helpers
def _lr(step, total, warmup, peak, floor):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    prog = min((step - warmup) / max(total - warmup, 1), 1.0)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * prog))


def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def validate(model, tok, seq_len: int, mb: int, device: str, n_batches: int = 4, roles=None,
             amp_dtype=torch.bfloat16, amp_enabled: bool = True):
    """Held-out loss per role (answer tokens only). Uses split='eval', so it never overlaps training prompts."""
    model.eval()
    out = {}
    for role in roles or RG.PROCEDURAL_ROLES:
        it, tot, n = iter(SFTRows(tok, seq_len, seed=12345, split="eval", roles=(role,))), 0.0, 0
        for _ in range(n_batches):
            rows = [next(it) for _ in range(mb)]
            ids = torch.stack([r[0] for r in rows]).to(device)
            tg = torch.stack([r[1] for r in rows]).to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                o = model(ids, targets=tg)
            if "n_tokens" in o:
                tot += float(o["loss"]) * o["n_tokens"]; n += o["n_tokens"]
        out[role] = round(tot / max(n, 1), 4)
    model.train()
    return out


def resolve_tokens(budget: str, tokens: Optional[int], n_params: int, confirm_full: bool) -> int:
    if tokens:
        return int(tokens)
    if budget == "chinchilla":
        return 20 * n_params
    if budget == "4x":
        return 80 * n_params
    if budget == "full":
        if not confirm_full:
            raise SystemExit("budget=full means every one of SlimPajama's ~627B tokens (~1,500 H100-hours even for the 500M "
                             "tier) and it will not make the CAD roles better than 80 tokens/param does. "
                             "Pass --confirm-full to do it anyway.")
        return 627_000_000_000
    raise ValueError(budget)


# ------------------------------------------------------------------------------------------------ the loop
def train_shared(phase: str, tier_name: str, tok, out_dir: str, tokens: Optional[int] = None, budget: str = "4x",
                 init: Optional[str] = None, resume: bool = True, device: str = "cuda", workers: int = 4,
                 lr_scale: Optional[float] = None, eval_every: int = 250, save_every: int = 500, log_every: int = 20,
                 reviewed_path: Optional[str] = None, grad_ckpt: Optional[bool] = None, confirm_full: bool = False,
                 on_save: Optional[Callable[[], None]] = None, shape=None, mb: Optional[int] = None,
                 accum: Optional[int] = None, seq_len: Optional[int] = None, max_seconds: Optional[float] = None,
                 seed: int = 0, precision: Optional[str] = None) -> dict:
    """Returns {'step', 'total_steps', 'done', 'path'}. If max_seconds is hit the run saves and returns done=False:
    call it again (resume=True) - that is how a job longer than Modal's per-function timeout is chained.

    precision: "bf16" (default on Ampere+ GPUs, e.g. Modal's H100), "fp16" (for Turing GPUs with no bf16 tensor
    cores, e.g. Colab's free T4 - uses GradScaler since fp16's narrow exponent range can silently underflow
    without it), or "fp32" (works everywhere, no tensor-core speedup, only use if fp16 is unavailable too).
    None (default) auto-detects: bf16 if the GPU supports it, else fp16 on CUDA, else fp32 on CPU."""
    assert phase in ("pretrain", "sft")
    tier = TIERS[tier_name]
    mb, accum, L = mb or tier.micro_batch_size, accum or tier.grad_accum_steps, seq_len or tier.seq_len
    model = SharedTextModel(tier=tier_name if shape is None else None, vocab_size=tok.vocab_size, shape=shape)
    model.gradient_checkpointing = tier.gradient_checkpointing if grad_ckpt is None else grad_ckpt
    model.to(device)
    if precision is None:
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            precision = "bf16"
        elif device.startswith("cuda"):
            precision = "fp16"          # e.g. Colab T4 (Turing) - no bf16 tensor cores, fp16 tensor cores work
        else:
            precision = "fp32"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    amp_enabled = device.startswith("cuda") and precision != "fp32"
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and device.startswith("cuda")))
    print(f"[shared/{phase}] precision={precision} (GradScaler {'on' if scaler.is_enabled() else 'off'})", flush=True)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True             # autotunes conv/kernel algo picks for this shape - free win
        torch.backends.cuda.matmul.allow_tf32 = True       # only affects any fp32 matmuls that slip through autocast
        torch.backends.cudnn.allow_tf32 = True
    if device.startswith("cuda") and os.environ.get("QUADMESH_COMPILE", "1") == "1":
        try:
            model = torch.compile(model)
            print("[speed] torch.compile enabled (set QUADMESH_COMPILE=0 to disable, e.g. if it errors on your torch/CUDA version)")
        except Exception as e:
            print(f"[speed] torch.compile unavailable, running uncompiled: {e}")
    n_params = model.n_params(non_embedding=False)
    total_tokens = resolve_tokens(budget, tokens, n_params, confirm_full)
    tokens_per_step = mb * accum * L
    total_steps = max(total_tokens // tokens_per_step, 1)
    peak = tier.lr * (lr_scale if lr_scale is not None else (1.0 if phase == "pretrain" else 0.3))
    floor, warmup = peak * 0.1, min(tier.warmup_steps, max(total_steps // 10, 5))

    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tier.weight_decay}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=peak, betas=(tier.beta1, tier.beta2), fused=device.startswith("cuda"))

    # which reviewed-data roles actually have enough human-approved rows to be trained (the runtime refuses the rest)
    safety_trained = ([r for r in RG.REVIEWED_ROLES if len(RG.load_reviewed(reviewed_path, r, "train")) >= 500]
                      if phase == "sft" and reviewed_path else [])
    path = os.path.join(out_dir, f"shared_{phase}.pt")
    step = 0
    if resume and os.path.exists(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        if ck.get("done"):
            print(f"[shared/{phase}] already finished ({step} steps)", flush=True)
            return {"step": step, "total_steps": total_steps, "done": True, "path": path}
        print(f"[shared/{phase}] resumed at step {step}/{total_steps}", flush=True)
    elif init:
        model.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["model"])
        print(f"[shared/{phase}] initialised from {init}", flush=True)

    def save(final=False):
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "shape": model.shape,
                      "tier": tier_name, "phase": phase, "total_steps": total_steps, "vocab_size": tok.vocab_size,
                      "safety_roles_trained": safety_trained, "done": final}, path)
        if on_save:
            on_save()

    seed = seed + 7919 * step                                # a resumed run must not replay the same examples
    ds = (PretrainRows(tok, L, seed, (total_steps - step) * tokens_per_step)
          if phase == "pretrain" else SFTRows(tok, L, seed, reviewed_path=reviewed_path))
    loader = iter(_loader(ds, mb, workers))
    os.makedirs(out_dir, exist_ok=True)
    mlog = open(os.path.join(out_dir, f"metrics_{phase}.jsonl"), "a")
    print(f"[shared/{phase}] tier={tier_name} params={n_params/1e6:.1f}M tokens={total_tokens/1e9:.3f}B "
          f"steps={total_steps} tokens/step={tokens_per_step:,} lr={peak:g}", flush=True)

    model.train()
    t0 = t_start = time.time()
    run_loss, run_n = torch.zeros((), device=device), 0
    while step < total_steps:
        for g in opt.param_groups:
            g["lr"] = _lr(step, total_steps, warmup, peak, floor)
        try:
            batches = [next(loader) for _ in range(accum)]
        except StopIteration:                                # the data stream ended a step early (worker rounding)
            total_steps = step
            break
        for ids, tg in batches:
            ids, tg = ids.to(device, non_blocking=True), tg.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                loss = model(ids, targets=tg)["loss"]
            scaler.scale(loss / accum).backward()
            run_loss += loss.detach(); run_n += 1
        if scaler.is_enabled():
            scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % log_every == 0:
            dt = time.time() - t0
            rec = {"step": step, "loss": round(float(run_loss) / max(run_n, 1), 4), "grad_norm": round(float(gn), 3),
                   "lr": round(opt.param_groups[0]["lr"], 8), "tok_per_s": int(log_every * tokens_per_step / max(dt, 1e-9))}
            print(f"[shared/{phase}] {json.dumps(rec)}", flush=True); mlog.write(json.dumps(rec) + "\n"); mlog.flush()
            t0, run_loss, run_n = time.time(), torch.zeros((), device=device), 0
        if phase == "sft" and step % eval_every == 0:
            v = validate(model, tok, L, min(mb, 8), device, amp_dtype=amp_dtype, amp_enabled=amp_enabled)
            print(f"[shared/sft] held-out loss per role @ {step}: {v}", flush=True)
            mlog.write(json.dumps({"step": step, "val": v}) + "\n"); mlog.flush()
        if step % save_every == 0:
            save()
        if max_seconds and time.time() - t_start > max_seconds and step < total_steps:
            save()
            return {"step": step, "total_steps": total_steps, "done": False, "path": path}
    save(final=True)
    return {"step": step, "total_steps": total_steps, "done": True, "path": path}
