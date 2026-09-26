"""
From-scratch decoder-only transformer. Shared by all nine roles; the only
difference between roles is the ModelShape it's constructed with and a small
role head. No pretrained weights are loaded anywhere in this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from quadmesh.config import ModelShape


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def build_rope(seq_len: int, head_dim: int, theta: float, device, dtype, start: int = 0):
    """RoPE tables for positions start .. start+seq_len-1 (start > 0 when using the KV cache)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(start, start + seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x, cos, sin):
    # x: [b, h, t, d]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class KVCache:
    """Keys/values of tokens already processed, so generation only computes the NEW token.

    Pre-allocated for `max_len` tokens. Usage: a full prompt first, then one token at a time.
    Batch size 1 is the intended use (plan generation)."""

    def __init__(self, n_layers, n_heads, head_dim, max_len, device, dtype, batch: int = 1):
        shape = (batch, n_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.len, self.max_len = 0, max_len

    def update(self, layer, k, v):
        t = k.size(2)
        assert self.len + t <= self.max_len, "KV cache is full"
        assert t == 1 or self.len == 0, "feed the whole prompt first, then one token at a time"
        self.k[layer][:, :, self.len:self.len + t] = k
        self.v[layer][:, :, self.len:self.len + t] = v
        return self.k[layer][:, :, :self.len + t], self.v[layer][:, :, :self.len + t]

    def advance(self, t):
        self.len += t


class CustomAttention(nn.Module):
    def __init__(self, s: ModelShape):
        super().__init__()
        assert s.d_model % s.n_heads == 0
        self.n_heads = s.n_heads
        self.head_dim = s.d_model // s.n_heads
        self.qkv = nn.Linear(s.d_model, 3 * s.d_model, bias=False)
        self.proj = nn.Linear(s.d_model, s.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None, layer: int = 0):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            k, v = cache.update(layer, k, v)          # keys/values of ALL tokens so far
            # prompt (t > 1, empty cache): normal causal attention; one new token: it sees everything
            o = F.scaled_dot_product_attention(q, k, v, is_causal=(t > 1))
        return self.proj(o.transpose(1, 2).contiguous().view(b, t, c))


class QuadmeshSwiGLU(nn.Module):
    def __init__(self, s: ModelShape):
        super().__init__()
        self.w1 = nn.Linear(s.d_model, s.d_ff, bias=False)
        self.w3 = nn.Linear(s.d_model, s.d_ff, bias=False)
        self.w2 = nn.Linear(s.d_ff, s.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class QuadmeshBlock(nn.Module):
    def __init__(self, s: ModelShape):
        super().__init__()
        self.n1, self.attn = RMSNorm(s.d_model), CustomAttention(s)
        self.n2, self.ff = RMSNorm(s.d_model), QuadmeshSwiGLU(s)

    def forward(self, x, cos, sin, cache=None, layer: int = 0):
        x = x + self.attn(self.n1(x), cos, sin, cache, layer)
        return x + self.ff(self.n2(x))


class QuadmeshModel(nn.Module):
    """One role model. Constructed fresh — never `from_pretrained`."""

    def __init__(self, shape: ModelShape, role: str = "generic",
                 n_classes: Optional[int] = None):
        super().__init__()
        self.shape, self.role = shape, role
        self.embed = nn.Embedding(shape.vocab_size, shape.d_model)
        self.blocks = nn.ModuleList([QuadmeshBlock(shape) for _ in range(shape.n_layers)])
        self.norm = RMSNorm(shape.d_model)
        self.lm_head = nn.Linear(shape.d_model, shape.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight            # tied

        # Sec 12.3: every role reports a calibration confidence alongside output.
        self.confidence_head = nn.Linear(shape.d_model, 1)
        # classifier roles (risk tier, abuse pattern) get a discrete head too
        self.class_head = nn.Linear(shape.d_model, n_classes) if n_classes else None

        self.gradient_checkpointing = False
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * shape.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, cache=None, last_only: bool = False):
        b, t = idx.shape
        x = self.embed(idx)
        start = cache.len if cache is not None else 0
        cos, sin = build_rope(t, self.shape.d_model // self.shape.n_heads,
                              self.shape.rope_theta, idx.device, x.dtype, start=start)
        for i, blk in enumerate(self.blocks):
            if cache is None and self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin, cache, i)
        if cache is not None:
            cache.advance(t)
        x = self.norm(x)
        if last_only:                       # generation only needs the last position's logits
            assert targets is None
            x = x[:, -1:]
        logits = self.lm_head(x)

        out = {"logits": logits,
               "confidence": torch.sigmoid(self.confidence_head(x[:, -1])).squeeze(-1)}
        if self.class_head is not None:
            out["class_logits"] = self.class_head(x[:, -1])
        if targets is not None:
            out["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.reshape(-1))
        return out

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int = 512, eos_id=None, temperature: float = 0.0,
                 top_k: int = 0):
        """Fast generation with a KV cache. idx: [1, T] prompt ids. Returns the NEW token ids.
        temperature 0 = greedy (deterministic). Same output as the slow full-recompute loop."""
        assert idx.size(0) == 1, "generate() is written for batch size 1"
        was_training = self.training
        self.eval()
        hd = self.shape.d_model // self.shape.n_heads
        cache = KVCache(len(self.blocks), self.shape.n_heads, hd, idx.size(1) + max_new_tokens,
                        idx.device, self.embed.weight.dtype)
        logits = self(idx, cache=cache, last_only=True)["logits"][:, -1]
        out = []
        for _ in range(max_new_tokens):
            if temperature and temperature > 0:
                l = logits.float() / temperature
                if top_k:
                    v, _ = torch.topk(l, min(top_k, l.size(-1)))
                    l = l.masked_fill(l < v[:, [-1]], float("-inf"))
                nxt = torch.multinomial(torch.softmax(l, dim=-1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            tok = int(nxt.item())
            out.append(tok)
            if eos_id is not None and tok == eos_id:
                break
            logits = self(nxt, cache=cache, last_only=True)["logits"][:, -1]
        if was_training:
            self.train()
        return out

    def n_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n


class QuadmeshVisionModel(QuadmeshModel):
    """Vision-role model (ui_grounding, screenshot_diff).

    Input is NOT token ids: it is patchified pixels [B, N_patches, C*P*P] from
    data/vision.py. Targets are 4 quantized box coords [B, 4] in 0..999.
    Same blocks as the text model; only the input/output ends differ.
    """
    PATCH_DIM = 3 * 16 * 16   # must match data/vision.py (PATCH=16, RGB)
    N_BINS = 1000             # vision.py quantizes with int(v * 999)
    N_QUERY = 4               # x1, y1, x2, y2

    def __init__(self, shape: ModelShape, role: str = "generic"):
        super().__init__(shape, role=role)
        self.patch_embed = nn.Linear(self.PATCH_DIM, shape.d_model)
        self.query = nn.Parameter(torch.zeros(self.N_QUERY, shape.d_model))
        nn.init.normal_(self.query, std=0.02)
        self.box_head = nn.Linear(shape.d_model, self.N_BINS)
        nn.init.normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.box_head.weight, std=0.02)
        nn.init.zeros_(self.box_head.bias)

    def forward(self, patches, targets=None):
        b, n, _ = patches.shape
        x = self.patch_embed(patches)
        # Attention is causal, so the query tokens go LAST to see every patch.
        x = torch.cat([x, self.query.to(x.dtype).unsqueeze(0).expand(b, -1, -1)], dim=1)
        t = x.size(1)
        cos, sin = build_rope(t, self.shape.d_model // self.shape.n_heads,
                              self.shape.rope_theta, x.device, x.dtype)
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        x = self.norm(x)
        q = x[:, -self.N_QUERY:]                     # [B, 4, d]
        logits = self.box_head(q)                    # [B, 4, 1000]
        out = {"logits": logits,
               "confidence": torch.sigmoid(self.confidence_head(x[:, -1])).squeeze(-1)}
        if targets is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, self.N_BINS).float(), targets.reshape(-1))
        return out


VISION_ROLES = ("ui_grounding", "screenshot_diff")


def build_role_model(role: str, shape: ModelShape) -> QuadmeshModel:
    if role in VISION_ROLES:
        return QuadmeshVisionModel(shape, role=role)
    n_classes = {"risk_tier_classifier": 4, "abuse_pattern": 2}.get(role)
    m = QuadmeshModel(shape, role=role, n_classes=n_classes)
    return m
