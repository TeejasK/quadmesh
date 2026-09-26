"""
The thing you actually run and talk to.

Starts a text loop: type an instruction in plain English, it plans steps
from it and executes them in Blender, you watch it happen, type the next
instruction. Type 'quit' or 'exit' (or Ctrl+C) to stop.

    python -m quadmesh.agent.repl
"""
from __future__ import annotations

from quadmesh.agent.planner import plan_from_text
from quadmesh.agent.loop import run_plan
from quadmesh.agent import blender_client as bridge
from quadmesh.agent.launch import open_blender


BANNER = """
Quadmesh agent — talk to it in plain English, or type 'speak' to talk by voice.
Examples:
  open blender
  add a cube of size 3
  add a cylinder radius 0.5 length 4 at (1, 0, 0)
  add a cube and a cylinder, subtract the cylinder, export as bracket.stl

Type 'quit' or 'exit' to stop.
"""


def _get_prompt() -> str | None:
    raw = input("you> ").strip()
    if raw.lower() == "speak":
        from quadmesh.agent.speech import listen
        return listen()
    return raw


def main():
    print(BANNER)
    if not bridge.is_up():
        print("[repl] Blender bridge not detected yet. Type 'open blender' to "
              "launch it automatically, or start it manually.\n")

    while True:
        try:
            prompt = _get_prompt()
        except (EOFError, KeyboardInterrupt):
            print("\n[repl] stopped.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "stop"):
            print("[repl] stopped.")
            break
        if "open blender" in prompt.lower() or "start blender" in prompt.lower():
            open_blender()
            continue

        steps = plan_from_text(prompt)
        if not steps:
            print("[repl] nothing to do with that instruction, try again.")
            continue

        run_plan(steps)
        print("[repl] done. what next?")


if __name__ == "__main__":
    main()
