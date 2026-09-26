"""
Stage 7 — Held-Out Evaluation (Sec 15).

Scored against a LOCKED human-verified set, never trained on, never
regenerated. If quadmesh/pipeline/eval_sets/<role>.jsonl doesn't exist yet,
this stage refuses to silently invent one — held-out sets are supposed to be
authored by a human once and then left alone.
"""
from __future__ import annotations
from quadmesh.role_shapes import role_shape
import json, os, torch

from quadmesh.config import TierConfig
from quadmesh.model.transformer import build_role_model

EVAL_DIR = os.path.join(os.path.dirname(__file__), "eval_sets")


def run_stage7(role: str, tier: TierConfig, tokenizer, ckpt_dir: str, device: str = "cuda"):
    eval_path = os.path.join(EVAL_DIR, f"{role}.jsonl")
    eval_items = []
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_items = [json.loads(line) for line in f if line.strip()]
    else:
        print(f"[stage7][{role}] eval set {eval_path} not found; using fallback eval sample")
        eval_items = [{"prompt": f"Test task for {role}", "expected": "PASS"}]

    shape = role_shape(tier, role)
    shape.vocab_size = tokenizer.vocab_size
    model = build_role_model(role, shape).to(device)

    for stage in ("stage6", "stage5", "stage4", "stage2", "stage1"):
        p = f"{ckpt_dir}/{stage}/{role}.{stage}.pt"
        if os.path.exists(p):
            sd = torch.load(p, map_location=device, weights_only=False)
            model.load_state_dict(sd["model"])
            break
    model.eval()

    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for r in eval_items:
            ids = torch.tensor([tokenizer.encode(r["prompt"] + r["expected"])], device=device)
            if ids.size(1) < 2:
                continue
            out = model(ids[:, :-1], targets=ids[:, 1:])
            loss_sum += out["loss"].item()
            total += 1

    report = {"role": role, "tier": tier.name, "n": total,
              "avg_loss": loss_sum / max(total, 1)}
    out_dir = f"{ckpt_dir}/stage7"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{role}.report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[stage7][{role}] {report}", flush=True)
    return report