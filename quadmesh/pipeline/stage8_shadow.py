"""
Stage 8 — Shadow Deployment (Sec 15, Sec 9-10, Quality Gate 2).

Runs the new checkpoint PARALLEL to whatever's currently marked "live" for
this role, on the same inputs, logging disagreements. No weight updates here
- this is the last gate before promote_to_live() flips the pointer.
"""
from __future__ import annotations
from quadmesh.role_shapes import role_shape
import json, os, shutil, torch

from quadmesh.config import TierConfig
from quadmesh.model.transformer import build_role_model


def _latest_checkpoint(ckpt_dir: str, role: str) -> str:
    for cand_stage in ("stage6", "stage5", "stage4", "stage2", "stage1"):
        p = f"{ckpt_dir}/{cand_stage}/{role}.{cand_stage}.pt"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no checkpoint found for {role} under {ckpt_dir}")


def run_stage8(role: str, tier: TierConfig, tokenizer, ckpt_dir: str,
              shadow_inputs: list[str], device: str = "cuda",
              disagreement_threshold: float = 0.15):
    shape = role_shape(tier, role)
    shape.vocab_size = tokenizer.vocab_size

    candidate = build_role_model(role, shape).to(device)
    cand_path = _latest_checkpoint(ckpt_dir, role)
    candidate.load_state_dict(torch.load(cand_path, map_location=device, weights_only=False)["model"])
    candidate.eval()

    live_path = f"{ckpt_dir}/live/{role}.pt"
    disagreements = []
    if os.path.exists(live_path):
        live = build_role_model(role, shape).to(device)
        live.load_state_dict(torch.load(live_path, map_location=device, weights_only=False)["model"])
        live.eval()

        with torch.no_grad():
            for prompt in shadow_inputs:
                ids = torch.tensor([tokenizer.encode(prompt)], device=device)
                if ids.numel() == 0:
                    continue
                c_out = candidate(ids)["confidence"].item()
                l_out = live(ids)["confidence"].item()
                if abs(c_out - l_out) > disagreement_threshold:
                    disagreements.append({"prompt": prompt, "live_conf": l_out, "candidate_conf": c_out})
    else:
        print(f"[stage8][{role}] no existing live model — this becomes live unconditionally")

    log_dir = f"{ckpt_dir}/stage8"
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/{role}.disagreements.jsonl", "w") as f:
        for d in disagreements:
            f.write(json.dumps(d) + "\n")

    rate = len(disagreements) / max(len(shadow_inputs), 1)
    print(f"[stage8][{role}] disagreement rate {rate:.2%} ({len(disagreements)}/{len(shadow_inputs)})", flush=True)
    return {"cand_path": cand_path, "disagreement_rate": rate, "n_disagreements": len(disagreements)}


def promote_to_live(role: str, ckpt_dir: str, cand_path: str):
    """Manual, explicit call only — never automatic. Quality Gate 2 (Sec 2)
    requires human sign-off for anything that isn't clearly low-risk."""
    os.makedirs(f"{ckpt_dir}/live", exist_ok=True)
    shutil.copy(cand_path, f"{ckpt_dir}/live/{role}.pt")
    print(f"[promote][{role}] {cand_path} -> {ckpt_dir}/live/{role}.pt", flush=True)
