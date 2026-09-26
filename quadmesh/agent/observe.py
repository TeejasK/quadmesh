"""
The agent's eyes (Sec 2 "perception"): after a plan runs, read the REAL Blender
scene and compare it with what the plan should have produced.

A from-scratch text model cannot read pixels, so "seeing" means structured facts:
object sizes, volume, open (non-manifold) edges. That is also what an engineer
checks first.

Save as quadmesh/agent/observe.py.
"""
from __future__ import annotations
from quadmesh.agent import blender_client as bridge
from quadmesh.pipeline.plan_checker import predicted_bbox


def steps_to_plan(steps) -> list[dict]:
    return [{"op": s.native_op, "args": dict(s.native_args or {})} for s in steps if s.native_op]


def look():
    """Ask Blender what is in the scene. Returns a list of dicts, or None."""
    try:
        resp = bridge.call("scene_info")
    except Exception:
        return None
    return resp.get("result") if resp.get("ok") else None


def report(steps, tol: float = 0.5) -> list[str]:
    """Lines to print after run_plan(): what Blender built vs what the plan predicted."""
    scene = look()
    if scene is None:
        return ["[eyes] could not read the Blender scene (restart the bridge after updating it?)"]
    lines = [f"[eyes] Blender scene: {len(scene)} mesh object(s)"]
    for o in scene:
        flag = "" if o["open_edges"] == 0 else f"  <-- {o['open_edges']} open edges"
        lines.append(f"[eyes]   {o['name']}: {o['dims'][0]} x {o['dims'][1]} x {o['dims'][2]} mm, "
                     f"volume {o['volume']}{flag}")

    plan = [s for s in steps_to_plan(steps) if not s["op"].startswith("export_")]
    want = predicted_bbox(plan)
    if want is None:
        lines.append("[eyes] no size prediction for this plan (it uses ops the checker does not model)")
        return lines
    wd = sorted([round(want[1] - want[0], 3), round(want[3] - want[2], 3), round(want[5] - want[4], 3)])
    ok = any(all(abs(a - b) <= tol for a, b in zip(sorted(o["dims"]), wd)) for o in scene)
    if ok:
        lines.append(f"[eyes] OK: an object matches the predicted size {wd} mm")
    else:
        lines.append(f"[eyes] PROBLEM: expected an object of about {wd} mm but none matches - "
                     f"a step probably failed inside Blender")
    return lines
