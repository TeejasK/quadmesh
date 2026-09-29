"""
[v2: PURELY GENERATIVE - the per-role class/MLP heads are gone; a role is a tag token, see roles_io.py]
One shared transformer backbone + lightweight per-role heads, for the 8 TEXT-CENTRIC roles
(role_shapes.TEXT_BACKBONE_ROLES). Replaces 8 separate full QuadmeshModel instances with one trunk that all
8 roles read from and write gradients into, plus a small residual MLP head per role on top.

Deliberately NOT used for vla/ui_grounding/screenshot_diff (image patches) or asr/tts (audio) - see
role_shapes.py's note above TEXT_BACKBONE_ROLES for why those stay separate models. This module only
concerns the 8 roles that share one real input space: tokenized text.

Advantages this is meant to capture (see the design discussion this followed):
  - the shared trunk learns grammar/reasoning/CAD vocabulary ONCE, pooling all 8 roles' training signal
    into the same representation instead of splitting data 8 ways
  - nearly all the parameter budget goes into that shared representation (BACKBONE_SHARE=0.9 in
    role_shapes.py), not redundant re-derivation of basic language modeling 8 times over
  - one backbone forward/backward per training step (see train/shared_text_train.py) instead of 8 separate
    training loops, each with their own optimizer state and memory footprint
  - a consistent representation space across roles, which is what arbitration actually needs to compare
    other roles' outputs meaningfully (tokenizer.py already flagged shared TOKEN ids for this; this extends
    the same principle to the learned representations, not just the vocabulary)

The stated tradeoff also still applies here, not glossed over: roles CAN interfere with each other through
the shared trunk if their training objectives pull in conflicting directions. Nothing in this file prevents
that - it's a property of multi-task learning in general, not a bug to fix in this module. What DOES help:
each role's own head absorbs role-specific behavior (see RoleHead's residual formulation below), so the
shared trunk only needs to carry what's actually common across roles, not force every role's idiosyncratic
output format through the same final layer.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from quadmesh.model.transformer import QuadmeshBlock, RMSNorm, build_rope, KVCache


class QuadmeshSharedBackbone(nn.Module):
    """The trunk: embedding -> N transformer blocks -> final norm. No head - see RoleHead for that.
    Structurally identical to QuadmeshModel's own trunk (same QuadmeshBlock, same RoPE, same KV cache
    support) so a role's behavior under this backbone is comparable to its old standalone-model behavior -
    the change is WHERE the trunk's weights come from (shared vs. per-role), not how the trunk works."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.embed = nn.Embedding(shape.vocab_size, shape.d_model)
        self.blocks = nn.ModuleList([QuadmeshBlock(shape) for _ in range(shape.n_layers)])
        self.norm = RMSNorm(shape.d_model)
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

    def forward(self, idx: torch.Tensor, cache: Optional[KVCache] = None, last_only: bool = False) -> torch.Tensor:
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
        if last_only:
            x = x[:, -1:]
        return x


class SharedTextModel(nn.Module):
    """One trunk, one tied LM head. Every one of the 8 text roles is text-in / text-out, selected by the role tag
    token at the start of the prompt (roles_io.role_prompt). There is no classifier head and no per-role branch, so
    rows of different roles can share a batch. A single confidence head is kept for calibration (stage 5).

    Targets are ALIGNED with idx (targets[:, t] is the label for token idx[:, t]); the shift by one happens INSIDE
    forward(). -100 marks positions without loss (the prompt, padding)."""

    def __init__(self, tier=None, vocab_size: Optional[int] = None, shape=None):
        super().__init__()
        if shape is None:
            from quadmesh.role_shapes import backbone_shape
            if isinstance(tier, str):
                from quadmesh.config import TIERS
                tier = TIERS[tier]
            shape = backbone_shape(tier)
        if vocab_size is not None:
            shape.vocab_size = vocab_size
        self.backbone_shape = self.shape = shape
        self.backbone = QuadmeshSharedBackbone(shape)
        self.lm_head = nn.Linear(shape.d_model, shape.vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embed.weight            # tied
        self.confidence_head = nn.Linear(shape.d_model, 1)

    @property
    def gradient_checkpointing(self):
        return self.backbone.gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, v):
        self.backbone.gradient_checkpointing = v

    def forward(self, idx, targets=None, cache=None, last_only: bool = False, return_logits: bool = False):
        x = self.backbone(idx, cache=cache, last_only=last_only)
        out = {}
        if targets is None:
            out["logits"] = self.lm_head(x)
            out["confidence"] = torch.sigmoid(self.confidence_head(x[:, -1])).squeeze(-1)
            return out
        h, y = x[:, :-1], targets[:, 1:]
        keep = y != -100
        # Fast static path when all tokens are targets (standard pretrain):
        if bool(keep.all()):
            logits = self.lm_head(h)
            out["loss"] = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            out["n_tokens"] = keep.sum()
        else:
            logits = self.lm_head(h[keep])
            out["loss"] = F.cross_entropy(logits, y[keep])
            out["n_tokens"] = keep.sum()
        if return_logits:
            out["logits"] = logits
        return out

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int = 512, eos_id=None, temperature: float = 0.0, top_k: int = 0):
        """KV-cached generation, batch 1. Returns the NEW token ids."""
        assert idx.size(0) == 1
        was_training = self.training
        self.eval()
        hd = self.shape.d_model // self.shape.n_heads
        cache = KVCache(len(self.backbone.blocks), self.shape.n_heads, hd, idx.size(1) + max_new_tokens,
                        idx.device, self.backbone.embed.weight.dtype)
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
        n = sum(p.numel() for p in self.parameters())          # tied weights are counted once by parameters()
        return n - self.backbone.embed.weight.numel() if non_embedding else n


def load_shared(path: str, device: str = "cpu", dtype=None):
    """Load a checkpoint written by train/shared_text_train.py -> (model, ckpt_dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SharedTextModel(shape=ck["shape"])
    m.load_state_dict(ck["model"])
    m = m.to(device=device, dtype=dtype) if dtype else m.to(device)
    return m.eval(), ck
