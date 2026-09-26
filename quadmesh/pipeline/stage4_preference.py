"""
Stage 4 — Preference / Reward Tuning (Sec 15).

DPO on Stage 3's pass/fail pairs. Only task_planner, spec_generator, and
arbitration get this stage (PREFERENCE_TUNED_ROLES in config.py) — the
classifier/screening roles don't have a "preferred continuation" to tune on.
"""
from __future__ import annotations
from quadmesh.role_shapes import role_shape
import json, os, torch
import torch.nn.functional as F

from quadmesh.config import TierConfig, PREFERENCE_TUNED_ROLES
from quadmesh.model.transformer import build_role_model
from quadmesh.train.loop import make_optimizer


def load_pairs(path: str, tokenizer, device):
    good, bad = [], []
    by_prompt = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_prompt.setdefault(r["prompt"], {"pass": [], "fail": []})
            (by_prompt[r["prompt"]]["pass"] if r["passed"] else by_prompt[r["prompt"]]["fail"]).append(r["text"])
    for p, d in by_prompt.items():
        for g in d["pass"]:
            for b in d["fail"][:1]:
                good.append(tokenizer.encode(g))
                bad.append(tokenizer.encode(b))
    return good, bad


def dpo_loss(policy_logp_good, policy_logp_bad, ref_logp_good, ref_logp_bad, beta=0.1):
    pi = policy_logp_good - policy_logp_bad
    ref = ref_logp_good - ref_logp_bad
    return -F.logsigmoid(beta * (pi - ref)).mean()


def seq_logprob(model, ids, device):
    x = torch.tensor([ids], device=device)
    out = model(x[:, :-1], targets=x[:, 1:])
    logp = -out["loss"] * (x.size(1) - 1)
    return logp


def run_stage4(role: str, tier: TierConfig, tokenizer, ckpt_dir: str,
              device: str = "cuda", steps: int = 500):
    if role not in PREFERENCE_TUNED_ROLES:
        print(f"[stage4][{role}] not a preference-tuned role, skipping")
        return None

    shape = role_shape(tier, role)
    shape.vocab_size = tokenizer.vocab_size
    policy = build_role_model(role, shape).to(device)
    ref = build_role_model(role, shape).to(device)

    src = None
    for s in ("stage2", "stage1"):
        cand = f"{ckpt_dir}/{s}/{role}.{s}.pt"
        if os.path.exists(cand):
            src = cand
            break
    if not src:
        raise FileNotFoundError(f"No stage1 or stage2 checkpoint found for {role} under {ckpt_dir}")

    sd = torch.load(src, map_location=device, weights_only=False)
    policy.load_state_dict(sd["model"])
    ref.load_state_dict(sd["model"])
    for p in ref.parameters():
        p.requires_grad = False

    pairs_path = f"{ckpt_dir}/stage3/{role}.pairs.jsonl"
    good, bad = ([], [])
    if os.path.exists(pairs_path):
        good, bad = load_pairs(pairs_path, tokenizer, device)

    out_dir = f"{ckpt_dir}/stage4"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{role}.stage4.pt"

    n = min(len(good), len(bad), steps)
    if n == 0:
        print(f"[stage4][{role}] no contrastive pairs available; preserving policy weights")
        torch.save({"model": policy.state_dict(), "role": role, "tier": tier.name}, path)
        return path

    opt = make_optimizer(policy, tier)
    for i in range(n):
        pol_g = seq_logprob(policy, good[i], device)
        pol_b = seq_logprob(policy, bad[i], device)
        with torch.no_grad():
            ref_g = seq_logprob(ref, good[i], device)
            ref_b = seq_logprob(ref, bad[i], device)
        loss = dpo_loss(pol_g, pol_b, ref_g, ref_b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 25 == 0 or i == n - 1:
            print(f"[stage4][{role}] {i+1}/{n} dpo_loss {loss.item():.4f}", flush=True)

    torch.save({"model": policy.state_dict(), "role": role, "tier": tier.name}, path)
    return path
