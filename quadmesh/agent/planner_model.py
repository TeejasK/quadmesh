"""
Loads the trained task_planner for ANY tier on CPU or GPU and generates plans with the KV cache.

    from quadmesh.agent.planner_model import propose
    plan = propose("Make a 60x40x5 mm plate with 4 corner holes ...", tier="500M", device="auto")

Environment defaults:  QUADMESH_CKPT_ROOT (folder holding Quadmesh-<tier>/live/task_planner.pt and
tokenizer/), QUADMESH_TIER (default 100M), QUADMESH_DEVICE (auto | cpu | cuda).
"""
from __future__ import annotations
import json
import os
from typing import Optional

from quadmesh.pipeline.plan_checker import extract_plan

_LOADED: dict = {}


def resolve_device(device: Optional[str] = None) -> str:
    import torch
    device = device or os.environ.get("QUADMESH_DEVICE", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def pick_dtype(tier: str, device: str):
    import torch
    big = tier in ("1B", "3B", "7B")
    if device == "cuda" and big and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cpu" and tier in ("3B", "7B"):
        return torch.bfloat16            # halves RAM; slow on CPU - a GPU is strongly advised for 3B/7B
    return torch.float32


def load_role(role: str, tier: Optional[str] = None, device: Optional[str] = None, ckpt_root: Optional[str] = None):
    import torch
    from quadmesh.config import get_tier
    from quadmesh.role_shapes import role_shape
    from quadmesh.model.transformer import build_role_model
    from quadmesh.data.tokenizer import get_or_train

    tier = (tier or os.environ.get("QUADMESH_TIER", "100M")).replace("Quadmesh-", "")
    root = ckpt_root or os.environ.get("QUADMESH_CKPT_ROOT")
    if not root:
        raise RuntimeError("set QUADMESH_CKPT_ROOT (or pass ckpt_root=) to the folder with the downloaded checkpoints")
    device = resolve_device(device)
    key = (role, tier, device, root)
    if key in _LOADED:
        return _LOADED[key]
    path = f"{root}/Quadmesh-{tier}/live/{role}.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"no trained planner at {path} - train it first "
                                f"(modal run modal_app.py::train_full_role --tier {tier} --role {role})")
    cfg = get_tier(tier)
    tok = get_or_train(f"{root}/tokenizer/quadmesh-bpe.json")
    shp = role_shape(cfg, role); shp.vocab_size = tok.vocab_size
    model = build_role_model(role, shp)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    model = model.to(device=device, dtype=pick_dtype(tier, device)).eval()
    _LOADED[key] = (model, tok, device)
    return _LOADED[key]


def propose(prompt: str, tier: Optional[str] = None, device: Optional[str] = None,
            prev_plan: Optional[list] = None, feedback: Optional[str] = None,
            temperature: float = 0.0, max_new: int = 700, ckpt_root: Optional[str] = None) -> Optional[list]:
    """One plan for `prompt` (or a repaired plan if prev_plan + feedback are given). None if unparseable."""
    import torch
    from quadmesh.agent.shared_runtime import get_runtime
    rt = get_runtime(ckpt_root, tier, device)                 # the shared generative model, when trained
    if rt is not None:
        return rt.plan(prompt, prev_plan=prev_plan, feedback=feedback, temperature=temperature, max_new=max_new)
    model, tok, dev = load_role("task_planner", tier, device, ckpt_root)
    if prev_plan is not None and feedback:
        text = (f"REPAIR: {prompt} PLAN {json.dumps(prev_plan, separators=(',', ':'))} "
                f"ERRORS {feedback} FIX")
    else:
        text = prompt
    ids = torch.tensor([tok.encode(text)], device=dev)
    new = model.generate(ids, max_new_tokens=max_new, eos_id=tok.eos_id, temperature=temperature, top_k=40)
    return extract_plan(tok.decode(new))


def load_planner(tier=None, device=None, ckpt_root=None):
    return load_role("task_planner", tier, device, ckpt_root)


def propose_spec(request: str, tier: Optional[str] = None, device: Optional[str] = None,
                 ckpt_root: Optional[str] = None) -> Optional[dict]:
    """Spec Generator: plain-language request -> design parameters ({"kind": "unsupported"} if it cannot be built)."""
    import torch
    from quadmesh.spec_gen_data import parse_spec
    from quadmesh.agent.shared_runtime import get_runtime
    rt = get_runtime(ckpt_root, tier, device)
    if rt is not None:
        return rt.spec(request)
    model, tok, dev = load_role("spec_generator", tier, device, ckpt_root)
    ids = torch.tensor([tok.encode(request + " SPEC")], device=dev)
    return parse_spec("SPEC " + tok.decode(model.generate(ids, max_new_tokens=120, eos_id=tok.eos_id)))
