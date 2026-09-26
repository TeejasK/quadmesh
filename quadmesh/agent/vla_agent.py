"""
Run the Vision-Language-Action model on your real desktop: it LOOKS at a screenshot of the Blender window,
reads your instruction, and outputs one keyboard/mouse action at a time until it says `done`.

    python -m quadmesh.agent.vla_agent "add a cube" --tier 100M --device auto
    python -m quadmesh.agent.vla_agent "add a cube" --dry-run          # prints the actions, touches nothing

Safety: every action is parsed and sanitised (actions.py) - no Alt+F4 / Ctrl+W / Win key / unknown keys - and
clicks are clamped to the Blender window. Slam the mouse into a screen corner to stop it (pyautogui fail-safe).
It stops by itself after --max-steps, after 3 unusable outputs in a row, or if the screen stops changing.
"""
from __future__ import annotations
import argparse
import os
from typing import Callable, Optional

from quadmesh.agent import actions as A

_LOADED: dict = {}


def load_vla(tier: Optional[str] = None, device: Optional[str] = None, ckpt_root: Optional[str] = None):
    import torch
    from quadmesh.config import get_tier
    from quadmesh.role_shapes import role_shape
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.model.vla import QuadmeshVLA
    from quadmesh.agent.planner_model import resolve_device

    tier = (tier or os.environ.get("QUADMESH_TIER", "100M")).replace("Quadmesh-", "")
    root = ckpt_root or os.environ.get("QUADMESH_CKPT_ROOT")
    if not root:
        raise RuntimeError("set QUADMESH_CKPT_ROOT to the folder with the downloaded checkpoints")
    dev = resolve_device(device)
    key = (tier, dev, root)
    if key not in _LOADED:
        path = f"{root}/Quadmesh-{tier}/vla/vla.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"no trained VLA at {path} (modal run modal_app.py::train_vla --tier {tier})")
        cfg = get_tier(tier)
        tok = get_or_train(f"{root}/tokenizer/quadmesh-bpe.json")
        cfg.shape.vocab_size = tok.vocab_size
        shp = role_shape(cfg, "vla"); shp.vocab_size = tok.vocab_size
        m = QuadmeshVLA(shp)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        _LOADED[key] = (m.to(dev).eval(), tok, dev)
    return _LOADED[key]


def propose_action(task: str, image, history: list, tier=None, device=None, ckpt_root=None, temperature=0.0,
                   state: str = "") -> str:
    import numpy as np
    import torch
    from quadmesh.data.vla_data import to_patches
    model, tok, dev = load_vla(tier, device, ckpt_root)
    from quadmesh.data.vla_data import history_text
    instr = task + (f"\nstate: {state}" if state else "") + history_text(history, 12) + "\n"
    patches = torch.from_numpy(to_patches(image))[None].to(dev)
    ids = torch.tensor([tok.encode(instr)], device=dev)
    return tok.decode(model.generate_action(patches, ids, max_new=48, eos_id=tok.eos_id, temperature=temperature))


def blender_region(io) -> tuple:
    """(left, top, width, height) of the Blender window, or the whole screen if it cannot be found."""
    try:
        import pygetwindow as gw
        w = next(x for x in gw.getWindowsWithTitle("Blender") if x.width > 300)
        L, T = max(w.left, 0), max(w.top, 0)
        sw, sh = io.size()
        return (L, T, min(w.width, sw - L), min(w.height, sh - T))
    except Exception:
        w, h = io.size()
        return (0, 0, w, h)


def _fingerprint(img) -> bytes:
    return img.convert("L").resize((16, 16)).tobytes()


def run(task: str, tier=None, device=None, ckpt_root=None, max_steps: int = 60, io=None, log=print,
        propose_fn: Optional[Callable] = None, dry_run: bool = False, step_wait: float = 0.8,
        stuck_limit: int = 4) -> dict:
    if io is None:
        from quadmesh.agent.desk import PyAutoGuiIO
        io = PyAutoGuiIO()
    if propose_fn is None:
        propose_fn = lambda t, img, hist: propose_action(t, img, hist, tier, device, ckpt_root)
    region = blender_region(io)
    L, T, W, H = region
    history, bad, same, last = [], 0, 0, None
    for step in range(1, max_steps + 1):
        img = io.screenshot().crop((L, T, L + W, T + H))
        fp = _fingerprint(img)
        same = same + 1 if fp == last else 0
        last = fp
        if same >= stuck_limit:
            return {"ok": False, "reason": "the screen stopped changing", "history": history}
        text = propose_fn(task, img, history)
        act = A.parse(text)
        if act is None:
            bad += 1
            log(f"[vla] step {step}: refused output {text!r}")
            if bad >= 3:
                return {"ok": False, "reason": "3 unusable outputs in a row", "history": history}
            continue
        bad = 0
        log(f"[vla] step {step}: {act}")
        if act.kind == "done":
            return {"ok": True, "reason": "model said done", "history": history}
        if not dry_run:
            A.execute(io, act, region)
        history.append(str(act))
        io.wait(step_wait)
    return {"ok": False, "reason": "max steps reached", "history": history}


def main():
    ap = argparse.ArgumentParser(description="Quadmesh VLA agent")
    ap.add_argument("task")
    ap.add_argument("--tier", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    r = run(a.task, tier=a.tier, device=a.device, ckpt_root=a.ckpt_root, max_steps=a.max_steps, dry_run=a.dry_run)
    print(r["reason"])
    raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
