"""
Deployment — serves the promoted (`live/`) checkpoints behind an HTTP API.

This is what "modal deploy" actually stands up: a long-running endpoint that
loads every role's live checkpoint once, runs the Task Planner -> CAD Skill
Dispatch -> Verification Stack -> Risk-Tier Gate chain per request, and
returns either a verified STL or a "needs human sign-off" response (Sec 2's
Quality Gate 2). It never trains — only inference against `live/*.pt`.
"""
from __future__ import annotations
from quadmesh.role_shapes import role_shape
import os
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from quadmesh.config import get_tier, NINE_ROLES
from quadmesh.model.transformer import build_role_model
from quadmesh.data.tokenizer import get_or_train
from quadmesh.verification.stack import run_stack, escalate_tier, RISK_TIERS

CKPT_ROOT = os.environ.get("QUADMESH_CKPT_ROOT", "/vol")
TIER = os.environ.get("QUADMESH_SERVE_TIER", "100M")


class GenerateRequest(BaseModel):
    prompt: str
    material: str = "pla"
    min_wall_mm: float = 1.2


class GenerateResponse(BaseModel):
    passed: bool
    risk_tier: str
    requires_human_signoff: bool
    tier_escalated: bool
    note: str


def load_live_models(device: str):
    tier = get_tier(TIER)
    tok = get_or_train(f"{CKPT_ROOT}/tokenizer/quadmesh-bpe.json")
    models = {}
    for role in NINE_ROLES:
        path = f"{CKPT_ROOT}/{tier.name}/live/{role}.pt"
        if not os.path.exists(path):
            continue
        shape = role_shape(tier, role)
        shape.vocab_size = tok.vocab_size
        m = build_role_model(role, shape).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
        m.eval()
        models[role] = m
    return tier, tok, models


app = FastAPI(title="Quadmesh CAD Agent")
_state = {}


@app.on_event("startup")
def _startup():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tier, tok, models = load_live_models(device)
    missing = [r for r in NINE_ROLES if r not in models]
    if missing:
        print(f"WARNING: no live checkpoint yet for roles: {missing}. "
              f"Run promote_to_live() for these before relying on this endpoint.")
    _state.update(tier=tier, tok=tok, models=models, device=device)


@app.get("/")
def root():
    return {"app": "Quadmesh CAD Agent", "status": "running", "tier": TIER}


@app.get("/health")
def health():
    return {"status": "ok", "tier": TIER, "roles_live": list(_state.get("models", {}))}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate a plan with the shared model, screen the request, and check the plan with the hard checker.
    (The first version of this endpoint ran ONE forward pass, threw the planner output away and read a classifier
    head. It generated nothing.)"""
    from quadmesh.agent.shared_runtime import get_runtime
    from quadmesh.pipeline.plan_checker import check_plan
    rt = get_runtime(device=_state.get("device"))
    if rt is None:
        raise HTTPException(503, "no shared_text.pt under QUADMESH_CKPT_ROOT - train it first (commands.txt)")
    gate = rt.screen(req.prompt)
    if gate["blocked"]:
        raise HTTPException(422, f"request blocked by the abuse screen ({gate['category']})")
    plan = rt.plan(req.prompt)
    errors = check_plan(plan, req.prompt) if plan else ["model produced no parsable plan"]
    return GenerateResponse(
        passed=not errors,
        risk_tier=gate["tier"],
        requires_human_signoff=gate["requires_human_signoff"],
        tier_escalated=gate["escalated_by_keywords"],
        note=("plan checked by plan_checker: " + ("ok" if not errors else "; ".join(errors[:3]))
              + f" | model risk/abuse screen: {gate['model_screen']}. Mesh-level verification (verification.stack.run_stack) "
                "still has to run on the built part before anything is treated as print-ready."),
    )
