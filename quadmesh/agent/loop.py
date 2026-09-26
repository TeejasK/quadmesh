"""
Perception-Action Loop (Sec 2, 10), running LOCALLY so you can watch it.

Two execution modes:
  - visible=True  (default): real keyboard shortcuts via visible_ops.py.
    You SEE Shift+A open the Add menu, keys navigate it, Ctrl+Alt+E export.
    This is the mode you want when you're watching the monitor.
  - visible=False: silent native bpy calls via blender_bridge. Instant, but
    nothing moves on screen — use this once you trust the agent and just
    want the output fast.

After every step: screenshot, compare to before, print what changed.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Optional

from quadmesh.agent import blender_client as bridge
from quadmesh.agent.desktop import Action, execute, screenshot, find_and_focus_window
from quadmesh.agent.visible_ops import run_visibly


@dataclass
class Step:
    description: str
    native_op: Optional[str] = None       # e.g. "add_cube"
    native_args: dict = None
    gui_action: Optional[Action] = None    # used only if no mapping in visible_ops.py


def run_plan(steps: list[Step], pause_between: float = 1.0,
            visible: bool = True, use_native_if_available: bool = True):
    native_up = use_native_if_available and bridge.is_up()
    mode = "VISIBLE (keyboard shortcuts you can watch)" if visible else \
          ("silent native calls" if native_up else "GUI fallback (bridge down)")
    print(f"[loop] mode: {mode}")

    find_and_focus_window("Blender")

    for i, step in enumerate(steps):
        print(f"\n[loop] step {i+1}/{len(steps)}: {step.description}")
        before = screenshot()

        did_something = False
        if visible and step.native_op:
            did_something = run_visibly(step.native_op, step.native_args or {})
            if not did_something:
                print(f"[loop]   no visible mapping for {step.native_op!r} — "
                      f"falling back to native call so it still happens")

        if not did_something:
            if native_up and step.native_op:
                resp = bridge.call(step.native_op, **(step.native_args or {}))
                print(f"[loop]   native call -> {resp}")
                did_something = resp.get("ok", False)
            elif step.gui_action:
                print(f"[loop]   GUI action -> {step.gui_action}")
                execute(step.gui_action)
                did_something = True

        if not did_something:
            print(f"[loop]   could not execute this step - stopping (later steps depend on it)")
            break

        time.sleep(pause_between)
        after = screenshot()
        changed = after.tobytes() != before.tobytes()
        print(f"[loop]   screenshot-diff: {'changed' if changed else 'NO CHANGE — check Blender is focused'}")


if __name__ == "__main__":
    # quick manual check — real usage goes through repl.py + planner.py now
    from quadmesh.agent.planner import plan_from_text
    run_plan(plan_from_text("add a cube of size 3, export as test.stl"))
