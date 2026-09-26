"""
QuadmeshChat: the conversational model. One decoder (same LLaMA-style blocks as every Quadmesh role, KV-cached) reads
[ image patch tokens (optional) ][ system + conversation text ] and writes the reply as text - either words for the
person, or "CALL {json}" for a tool (see chat_data.py for the exact format the model is trained on).

There is no separate vision encoder: images are cut into 16x16 patches and linearly embedded (like the VLA / UI-grounding
models), so the SAME text vocabulary and the SAME weights place matrix that reasons about text also reasons about pixels -
consistent with "how Quadmesh reads text without a separate encoder" elsewhere in this project.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from quadmesh.model.transformer import QuadmeshModel, KVCache, build_rope

PATCH_DIM = 3 * 16 * 16


class QuadmeshChat(QuadmeshModel):
    def __init__(self, shape):
        super().__init__(shape, role="chat")
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

    def _img(self, patches):
        if patches is None or patches.size(1) == 0:
            return self.embed.weight.new_zeros((patches.size(0) if patches is not None else 1, 0, self.shape.d_model))
        return self.patch_embed(patches.to(self.patch_embed.weight.dtype))

    def forward(self, ids, patches=None, targets=None):
        """ids [B,T]; patches [B,Np,768] or None; targets [B,T] with -100 on prompt/image positions."""
        x = torch.cat([self._img(patches), self.embed(ids)], dim=1)
        n_img = x.size(1) - ids.size(1)
        h = self._run(x)[:, n_img:]
        logits = self.lm_head(h)
        out = {"logits": logits}
        if targets is not None:
            out["loss"] = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                          targets[:, 1:].reshape(-1), ignore_index=-100)
        return out

    @torch.no_grad()
    def generate_reply(self, ids, patches=None, max_new: int = 300, eos_id=None, temperature: float = 0.4, top_k: int = 40):
        was_training = self.training
        self.eval()
        b = ids.size(0)
        hd = self.shape.d_model // self.shape.n_heads
        img = self._img(patches)
        cache = KVCache(len(self.blocks), self.shape.n_heads, hd, img.size(1) + ids.size(1) + max_new, ids.device, self.embed.weight.dtype)
        x = torch.cat([img, self.embed(ids)], dim=1)
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
