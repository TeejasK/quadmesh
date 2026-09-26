"""
Autopilot: give it a prompt, it keeps working until the design passes every check (or runs out of attempts).

  plan (trained task_planner, KV-cached)  ->  rule check (plan_checker)  ->  build  ->  printability check
  ->  export STL  ->  printability check of the STL FILE  ->  PASS.
  Any failure is fed back to the planner (REPAIR) and it tries again, sampling differently.

Two ways to build:
  * bridge mode (default): ops go to Blender through the local socket - fast and exact.
  * desk mode  (desk=True / --desk): the agent uses only the REAL keyboard and mouse - it hovers Blender,
    switches the area to the Python Console, types each step, exports the STL by typing, and looks at the part
    with Home / Numpad views.  The exported file is then checked.  See desk.py for how and for the limits.

    python -m quadmesh.agent.autopilot "Make a 60x40x5 mm plate with 4 corner holes of 4.2 mm diameter, 8 mm from the edges" --tier 100M
    python -m quadmesh.agent.autopilot "..." --desk

It never touches objects that were in your Blender scene before it started.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from quadmesh.pipeline.plan_checker import check_plan, predicted_bbox
from quadmesh.agent.printability import Limits, judge_report, check_file


@dataclass
class Result:
    ok: bool
    attempts: int
    plan: Optional[list] = None
    report: Optional[dict] = None
    problems: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stl_path: Optional[str] = None
    screenshots: list = field(default_factory=list)
    log: list = field(default_factory=list)


def _slug(text: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "design")


class _BridgeBackend:
    """Bridge mode: ops over the socket; Blender itself reports printability."""
    name = "bridge"

    def __init__(self, bridge, watch: bool, out_dir: str, log, visible: bool):
        self.b, self.log, self.visible, self.dir, self.shots, self.ready = bridge, log, visible, out_dir, [], False
        if watch:
            try:
                from quadmesh.agent.desktop import find_and_focus_window, screenshot
                find_and_focus_window("Blender"); screenshot(); self.ready = True
            except Exception as e:
                log(f"[watch] keyboard/mouse view disabled ({e})")

    def start(self):
        if not self.b.is_up():
            from quadmesh.agent.launch import ensure_blender      # Quadmesh opens Blender itself
            if not ensure_blender(log=self.log) or not self.b.is_up():
                raise RuntimeError("Blender could not be started. Run: python -m quadmesh.agent.launch --blender-path <path to blender.exe>")
        self.b.call("mark_scene")

    def reset(self):
        self.b.call("reset_scene")

    def frame(self, tag):
        if not self.ready:
            return
        try:
            from quadmesh.agent.desktop import Action, execute, screenshot
            from quadmesh.agent.visible_ops import VIEWPORT_CENTER
            execute(Action("move", *VIEWPORT_CENTER)); execute(Action("key", key="home"))
            execute(Action("wait", seconds=0.35))
            p = os.path.join(self.dir, f"{tag}.png"); screenshot().save(p); self.shots.append(p)
        except Exception as e:
            self.log(f"[watch] {e}"); self.ready = False

    def build(self, steps, tag):
        for i, s in enumerate(steps):
            done = False
            if self.visible:
                try:
                    from quadmesh.agent.visible_ops import run_visibly
                    done = bool(run_visibly(s["op"], s["args"]))
                except Exception:
                    done = False
            if not done:
                r = self.b.call(s["op"], **s["args"])
                if not r.get("ok"):
                    return f"step {i+1} {s['op']} failed in Blender: {r.get('error', 'unknown error')}"
            self.frame(f"{tag}_step{i+1:02d}")
        return None

    def blender_report(self, limits):
        bx, by, bz = limits.bed
        r = self.b.call("print_check", bed_x=bx, bed_y=by, bed_z=bz)
        return r["result"] if r.get("ok") else None

    def export(self, stl):
        r = self.b.call("save_stl", filepath=stl)
        return None if r.get("ok") else f"STL export failed: {r.get('error')}"

    def extra(self, op, args):
        self.b.call(op, **args)


class _DeskBackend:
    """Desk mode: real keyboard + mouse only (types into Blender's Python Console)."""
    name = "desk"

    def __init__(self, out_dir: str, log, io=None, gui: bool = False):
        from quadmesh.agent.desk import DeskController
        self.real = io is None                                       # a real desk needs a real Blender
        self.d, self.dir, self.log, self.shots = DeskController(io=io, log=log), out_dir, log, []
        self.gui = gui

    def start(self):
        if self.real:
            from quadmesh.agent.launch import ensure_blender
            if not ensure_blender(log=self.log):
                raise RuntimeError("Blender could not be started. Run: python -m quadmesh.agent.launch --blender-path <path to blender.exe>")
        self.d.setup()
        self.d.call("mark_scene")

    def reset(self):
        self.d.call("reset_scene")

    def build(self, steps, tag):
        self.d.run_steps(steps, gui=self.gui)
        self.shots += self.d.views(self.dir, tag)
        return None

    def blender_report(self, limits):
        return None                           # no socket in this mode: the STL FILE is the check

    def export(self, stl):
        if os.path.exists(stl):
            os.remove(stl)
        self.d.call("save_stl", filepath=stl)
        return None if self.d.wait_for_file(stl) else "the STL file never appeared - a typed step probably failed in Blender"

    def frame(self, tag):
        pass

    def extra(self, op, args):
        self.d.call(op, **args)


def design(prompt: str, tier: Optional[str] = None, device: Optional[str] = None, attempts: int = 6,
           limits: Limits = Limits(), out_dir: str = "designs", watch: bool = True, visible: bool = False,
           ckpt_root: Optional[str] = None, name: Optional[str] = None, desk: bool = False, gui: bool = False,
           propose_fn: Optional[Callable] = None, bridge=None, desk_io=None) -> Result:
    res = Result(ok=False, attempts=0)

    def log(msg):
        print(msg, flush=True)
        res.log.append(msg)

    if propose_fn is None:
        from quadmesh.agent.planner_model import propose
        propose_fn = lambda p, **kw: propose(p, tier=tier, device=device, ckpt_root=ckpt_root, **kw)
    os.makedirs(out_dir, exist_ok=True)
    name = name or _slug(prompt)
    try:
        if desk:
            be = _DeskBackend(out_dir, log, desk_io, gui)
        else:
            if bridge is None:
                from quadmesh.agent import blender_client as bridge
            be = _BridgeBackend(bridge, watch, out_dir, log, visible)
        be.start()
    except Exception as e:
        log(f"[autopilot] cannot start: {e}")
        res.problems = [str(e)]
        return res
    log(f"[autopilot] mode: {be.name}")

    feedback, prev = None, None
    for k in range(1, attempts + 1):
        res.attempts = k
        temp = 0.0 if k == 1 else min(0.9, 0.25 + 0.15 * k)
        log(f"[autopilot] attempt {k}/{attempts}" + (f" (retry: {feedback})" if feedback else ""))
        plan = propose_fn(prompt, prev_plan=prev, feedback=feedback, temperature=temp)
        if not plan:
            feedback, prev = "no valid JSON plan", None
            continue
        errs = check_plan(plan, prompt)
        if errs:
            feedback, prev = "; ".join(errs[:2]), plan
            log(f"[autopilot]   plan rejected before building: {feedback}")
            continue

        build = [s for s in plan if not s["op"].startswith("export_")]
        exports = [s for s in plan if s["op"].startswith("export_")]
        be.reset()
        err = be.build(build, f"{name}_a{k}")
        if err:
            feedback, prev = err, plan
            log(f"[autopilot]   {err}")
            continue

        pred = predicted_bbox(build)
        problems, warnings = [], []
        report = be.blender_report(limits)
        if report is not None:
            problems, warnings = judge_report(report, pred, limits)
            if problems:
                feedback, prev = "; ".join(problems[:2]), plan
                log(f"[autopilot]   Blender printability check failed: {'; '.join(problems)}")
                continue

        stl = os.path.abspath(os.path.join(out_dir, f"{name}.stl"))
        err = be.export(stl)
        if err:
            feedback, prev = err, plan
            log(f"[autopilot]   {err}")
            continue
        try:                                   # independent check of the exact file you would slice
            report, problems, w2 = check_file(stl, limits, pred)
            warnings += w2
        except ImportError:
            problems = ["numpy is not installed, so the STL file could not be verified (pip install numpy)"]
        except Exception as e:
            problems = [f"STL file could not be analysed: {e}"]
        if problems:
            feedback, prev = "; ".join(problems[:2]), plan
            log(f"[autopilot]   STL file check failed: {'; '.join(problems)}")
            continue

        for e in exports:                      # honour the user's own "export as ..." request
            be.extra(e["op"], e["args"])
        be.frame(f"{name}_final")
        res.ok, res.plan, res.report, res.problems, res.warnings = True, plan, report, [], warnings
        res.stl_path, res.screenshots = stl, be.shots
        log(f"[autopilot] PASS after {k} attempt(s): {stl}")
        for w in warnings:
            log(f"[autopilot]   warning: {w}")
        return res

    res.problems = [feedback] if feedback else ["no valid plan"]
    res.screenshots = be.shots
    log(f"[autopilot] FAILED after {attempts} attempts. Last problem: {feedback}. "
        f"Nothing was marked printable - do not print.")
    return res


@dataclass
class AssemblyResult:
    ok: bool
    spec: Optional[dict] = None
    parts: dict = field(default_factory=dict)
    report: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    json_path: Optional[str] = None
    log: list = field(default_factory=list)


def design_assembly(request, tier: Optional[str] = None, device: Optional[str] = None, limits: Limits = Limits(),
                    out_dir: str = "designs", desk: bool = False, gui: bool = False, watch: bool = True,
                    ckpt_root: Optional[str] = None, bridge=None, desk_io=None, spec_fn: Optional[Callable] = None,
                    attempts: int = 2) -> AssemblyResult:
    """"make a hexacopter, 1.5 kg, industrial grade"  ->  spec -> hand-calculated sizes -> every part built,
    checked and exported. Passes only if EVERY part passes; the calculations are saved next to the STL files."""
    from quadmesh.assemblies import build_assembly
    res = AssemblyResult(ok=False)

    def log(m):
        print(m, flush=True); res.log.append(m)

    if isinstance(request, dict):
        spec = request
    else:
        if spec_fn is None:
            from quadmesh.agent.planner_model import propose_spec
            spec_fn = lambda r: propose_spec(r, tier=tier, device=device, ckpt_root=ckpt_root)
        spec = spec_fn(request)
    res.spec = spec
    if not spec or spec.get("kind") not in ("hexacopter", "robot_arm"):
        log("[assembly] this request is not something Quadmesh can build yet (supported: hexacopter, robot_arm).")
        return res
    asm = build_assembly(spec, limits.bed)
    res.report, res.warnings = list(asm.report), list(asm.warnings)
    log(f"[assembly] {asm.kind}: " + ", ".join(f"{p.name} x{p.qty}" for p in asm.parts))
    for r in asm.report:
        log("[assembly]   " + r)
    if asm.warnings:
        for w in asm.warnings:
            log(f"[assembly] BLOCKED: {w}")
        return res
    os.makedirs(out_dir, exist_ok=True)
    for p in asm.parts:
        r = design(f"part {p.name}", attempts=attempts, limits=limits, out_dir=os.path.join(out_dir, asm.kind),
                   watch=watch, desk=desk, gui=gui, name=p.name, propose_fn=lambda pr, _p=p, **kw: _p.plan,
                   bridge=bridge, desk_io=desk_io)
        res.parts[p.name] = r
        if not r.ok:
            log(f"[assembly] part {p.name} FAILED - nothing is marked printable")
            return res
    path = os.path.join(out_dir, asm.kind, "assembly.json")
    with open(path, "w") as fh:
        json.dump({"kind": asm.kind, "spec": asm.spec, "report": asm.report,
                   "parts": [{"name": p.name, "qty": p.qty, "stl": res.parts[p.name].stl_path, "note": p.note} for p in asm.parts],
                   "joints": asm.joints}, fh, indent=2)
    res.ok, res.json_path = True, path
    log(f"[assembly] PASS - every part passed the checks. Print quantities as listed in {path}")
    return res


def main():
    ap = argparse.ArgumentParser(description="Quadmesh autopilot")
    ap.add_argument("prompt")
    ap.add_argument("--tier", default=None, help="100M | 500M | 1B | 3B | 7B (default: env QUADMESH_TIER or 100M)")
    ap.add_argument("--device", default=None, help="auto | cpu | cuda")
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--bed", type=float, nargs=3, default=[220, 220, 250])
    ap.add_argument("--min-wall", type=float, default=0.8)
    ap.add_argument("--supports", action="store_true", help="allow overhangs (you will print with supports)")
    ap.add_argument("--assembly", action="store_true", help="the prompt is a whole-product request (hexacopter, robot arm)")
    ap.add_argument("--spec", default=None, help='give the design parameters directly, e.g. \'{"kind":"hexacopter","mass_kg":1.5}\'')
    ap.add_argument("--desk", action="store_true", help="build with the real keyboard and mouse only")
    ap.add_argument("--gui", action="store_true", help="desk mode: make boxes/cylinders/spheres with real shortcuts + mouse (Shift+A, S, G, F2)")
    ap.add_argument("--no-watch", action="store_true", help="bridge mode: do not use keyboard/mouse to frame the view")
    ap.add_argument("--visible", action="store_true", help="bridge mode: use Blender shortcuts where a mapping exists")
    ap.add_argument("--out", default="designs")
    ap.add_argument("--ckpt-root", default=None)
    a = ap.parse_args()
    lim = Limits(bed=tuple(a.bed), min_wall=a.min_wall, allow_supports=a.supports)
    if a.assembly or a.spec:
        r = design_assembly(json.loads(a.spec) if a.spec else a.prompt, tier=a.tier, device=a.device, limits=lim,
                            out_dir=a.out, desk=a.desk or a.gui, gui=a.gui, watch=not a.no_watch, ckpt_root=a.ckpt_root)
        raise SystemExit(0 if r.ok else 1)
    r = design(a.prompt, tier=a.tier, device=a.device, attempts=a.attempts, limits=lim, out_dir=a.out,
               watch=not a.no_watch, visible=a.visible, desk=a.desk or a.gui, gui=a.gui, ckpt_root=a.ckpt_root)
    raise SystemExit(0 if r.ok else 1)


if __name__ == "__main__":
    main()
