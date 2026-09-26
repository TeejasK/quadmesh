"""
Long-horizon episode runner: keeps going for as many steps as the job needs (30, 300, 3000 ...) and only stops when the
job is finished or a step cannot be completed.

For EVERY step of a verified plan it:
  1. shows the model/recipe the screenshot + text (task, current sub-task, scene STATE, ordered action history),
  2. acts with real keyboard/mouse (GUI recipe or VLA) - or, on a retry, by typing the exact op into the console,
  3. reads the real scene back and compares it with what the plan says the scene must look like (plan_checker.scene_after),
  4. if it does not match: rolls the scene back to the last verified state (rebuilds it) and retries, escalating to the
     exact console route, so a wrong click can never derail the whole job.
Every action is written to demo.jsonl IN ORDER with the screenshot that preceded it, the scene state and the sub-task.
Runs that finish verified are marked SUCCESS: only those are used as training data (learning from its own successes).
"""
from __future__ import annotations
import json
import os
import time
from typing import Callable, Optional

from quadmesh.pipeline.plan_checker import scene_after, check_plan
from quadmesh.agent.record_demo import RecordingIO
from quadmesh.agent import actions as ACT


def state_text(scene: dict, limit: int = 12) -> str:
    if not scene:
        return "empty scene"
    items = [f"{n} {d[0]:g}x{d[1]:g}x{d[2]:g}" for n, d in sorted(scene.items()) if not n.startswith("_")][:limit]
    return "objects: " + ", ".join(items)


def scene_matches(expected: dict, actual: dict, tol: float = 0.5) -> bool:
    exp = {n: d for n, d in expected.items() if not n.startswith("_")}
    act = {n: d for n, d in actual.items() if not n.startswith("_")}
    if set(exp) != set(act):
        return False
    return all(all(abs(a - b) <= tol for a, b in zip(exp[n], act[n])) for n in exp)


def desk_state_fn(desk, path: str) -> Callable:
    """Reads the real scene in desk mode: types dump_scene(path) into Blender and reads the JSON file back."""
    def read():
        if os.path.exists(path):
            os.remove(path)
        desk.call("dump_scene", filepath=path)
        t0 = time.time()
        while time.time() - t0 < 20:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    rows = json.load(open(path))
                    return {r["name"]: tuple(r["dims"]) for r in rows}
                except Exception:
                    pass
            time.sleep(0.2)
        return {}
    return read


class Episode:
    def __init__(self, task: str, plan: list, io, desk, state_fn: Callable, out_dir: str = "episodes",
                 act_fn: Optional[Callable] = None, retries: int = 2, tol: float = 0.5, use_gui: bool = True,
                 max_actions: Optional[int] = None, time_limit: Optional[float] = None, log=print):
        self.task, self.plan, self.desk, self.state_fn = task, plan, desk, state_fn
        self.rec = RecordingIO(io, out_dir, task)          # logs every action + screenshot, in order
        desk.io = self.rec                                 # everything the desk does is now recorded
        self.retries, self.tol, self.use_gui, self.log = retries, tol, use_gui, log
        self.max_actions, self.time_limit = max_actions, time_limit
        self.act_fn = act_fn or self._recipe_actions
        self.verified = []

    # ---- ways to perform a step -----------------------------------------------------------------
    @staticmethod
    def _recipe_actions(step, ctx):
        from quadmesh.agent.gui_recipes import gui_actions
        return gui_actions(step)

    def _perform(self, step, mode, ctx):
        if mode == "gui":
            lines = self.act_fn(step, ctx)
            if lines is not None:
                self.desk.to_viewport()
                w, h = self.rec.size()
                err = ACT.run_text(self.rec, lines, (0, 0, w, h))
                if err:
                    raise RuntimeError(err)
                return "gui"
        self.desk.call(step["op"], **step["args"])          # exact route: typed into the console
        return "console"

    def _rollback(self):
        self.desk.call("reset_scene")
        for s in self.verified:
            self.desk.call(s["op"], **s["args"])

    # ---- the loop --------------------------------------------------------------------------------
    def run(self) -> dict:
        errs = check_plan(self.plan)
        if errs:
            return {"ok": False, "reason": f"plan is invalid: {errs[0]}", "steps_done": 0, "dir": self.rec.dir}
        t0 = time.time()
        self.desk.setup()
        self.desk.call("mark_scene")
        self.desk.call("reset_scene")
        n = len(self.plan)
        for k, step in enumerate(self.plan):
            expected = scene_after(self.plan[:k + 1]) or {}
            before = scene_after(self.plan[:k]) or {}
            ok = False
            for attempt in range(1 + self.retries):
                if self.max_actions and self.rec.step >= self.max_actions:
                    return self._end(False, f"action budget of {self.max_actions} reached", k)
                if self.time_limit and time.time() - t0 > self.time_limit:
                    return self._end(False, "time limit reached", k)
                ctx = {"state": state_text(before), "k": k, "n": n}
                self.rec.task = f"{self.task}\nstate: {ctx['state']}\nstep {k + 1}/{n}: {step['op']} {json.dumps(step['args'])}"
                mode = "gui" if (attempt == 0 and self.use_gui) else "console"
                try:
                    used = self._perform(step, mode, ctx)
                except Exception as e:
                    self.log(f"[episode] step {k + 1} {step['op']} failed ({e})")
                    used = mode
                actual = self.state_fn()
                if scene_matches(expected, actual, self.tol):
                    ok = True
                    self.log(f"[episode] step {k + 1}/{n} {step['op']} verified ({used}, attempt {attempt + 1})")
                    break
                self.log(f"[episode] step {k + 1}/{n} {step['op']} MISMATCH after {used}: expected "
                         f"{state_text(expected)} but Blender has {state_text(actual)} - rolling back and retrying")
                self._rollback()
            if not ok:
                return self._end(False, f"step {k + 1} ({step['op']}) could not be completed", k)
            self.verified.append(step)
        return self._end(True, "all steps verified", n)

    def _end(self, ok: bool, reason: str, done: int) -> dict:
        self.rec._log("done" if ok else "done")
        self.rec.fh.close()
        # verified-complete runs become training data (SUCCESS); failed runs are kept for study but never trained on
        open(os.path.join(self.rec.dir, "SUCCESS" if ok else "FAILED"), "w").write(reason)
        return {"ok": ok, "reason": reason, "steps_done": done, "actions": self.rec.step, "dir": self.rec.dir}
