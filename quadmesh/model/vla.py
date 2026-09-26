"""
QuadmeshVLA - the Vision-Language-Action model.

One decoder-only transformer (the same blocks as every other Quadmesh role) reads
    [ screenshot patch tokens ]  [ instruction + history tokens ]
and writes the next action as text ("click 412 380", "hotkey shift+a", 'type "cube"', "done").
Loss is only on the action tokens. Generation uses the KV cache, so one action costs one pass over the
screenshot, then a few single-token steps.

Limits worth knowing: attention is causal, so a patch only sees earlier patches (the instruction and the
action tokens see the whole screenshot); 448x448 input means tiny text is unreadable - crop to the Blender
window and keep the UI scale large.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from quadmesh.model.transformer import QuadmeshModel, KVCache, build_rope

PATCH_DIM = 3 * 16 * 16


class QuadmeshVLA(QuadmeshModel):
    def __init__(self, shape):
        super().__init__(shape, role="vla")
        self.patch_embed = nn.Linear(PATCH_DIM, shape.d_model)
        nn.init.normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)

    def _run(self, x, cache=None):
        start = cache.len if cache is not None else 0
        cos, sin = build_rope(x.size(1), self.shape.d_model // self.shape.n_heads, self.shape.rope_theta,
                              x.device, x.dtype, start=start)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, cache, i)
        if cache is not None:
            cache.advance(x.size(1))
        return self.norm(x)

    def forward(self, patches, ids, targets=None):
        """patches [B, Np, 768] float or None (text only); ids [B, T]; targets [B, T] with -100 on positions not learned."""
        xt = self.embed(ids)
        if patches is None or patches.size(1) == 0:
            xi = xt[:, :0]
        else:
            xi = self.patch_embed(patches.to(self.patch_embed.weight.dtype))
        h = self._run(torch.cat([xi, xt], dim=1))[:, xi.size(1):]
        logits = self.lm_head(h)
        out = {"logits": logits}
        if targets is not None:
            out["loss"] = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                          targets[:, 1:].reshape(-1), ignore_index=-100)
        return out

    @torch.no_grad()
    def generate_action(self, patches, ids, max_new: int = 48, eos_id=None, temperature: float = 0.0, top_k: int = 0):
        """Returns the new token ids (one action, or a chat reply when patches is None). ids [1, T]."""
        if patches is None:
            patches = self.embed.weight.new_zeros((1, 0, PATCH_DIM))
        assert patches.size(0) == 1
        was_training = self.training
        self.eval()
        hd = self.shape.d_model // self.shape.n_heads
        dev, dt = patches.device, self.embed.weight.dtype
        cache = KVCache(len(self.blocks), self.shape.n_heads, hd, patches.size(1) + ids.size(1) + max_new, dev, dt)
        x = torch.cat([self.patch_embed(patches.to(dt)), self.embed(ids)], dim=1)
        logits = self.lm_head(self._run(x, cache)[:, -1:])[:, -1]
        out = []
        for _ in range(max_new):
            if temperature and temperature > 0:
                l = logits.float() / temperature
                if top_k:
                    kth = torch.topk(l, min(top_k, l.size(-1)))[0][:, [-1]]
                    l = l.masked_fill(l < kth, float("-inf"))
                nxt = torch.multinomial(torch.softmax(l, dim=-1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            tok = int(nxt.item())
            out.append(tok)
            if eos_id is not None and tok == eos_id:
                break
            logits = self.lm_head(self._run(self.embed(nxt), cache)[:, -1:])[:, -1]
        if was_training:
            self.train()
        return out


    generate = generate_action          # same thing: a chat reply is "an action" made of words
