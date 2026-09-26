"""
Talk to Quadmesh: ask it to move / turn a part, show it from all sides, build something, or ask what it built.

    python -m quadmesh.agent.assistant            # type, or say --voice to use the microphone
    python -m quadmesh.agent.assistant --voice --speak

What it is and is NOT:
  * it understands a fixed set of commands and questions (below) with plain rules - it is not a chat model. Anything
    else gets an honest "I can't do that", never an invented answer.
  * answers to questions come from real numbers it just produced (printability report, strength calculations,
    parts list), not from a language model's guess.
  * --speak uses your operating system's voice (pyttsx3). Quadmesh's own TTS model is a scaffold: it has no
    vocoder and did not learn to read text, so it cannot make speech yet.
  * --voice listens through quadmesh.agent.speech.listen (the from-scratch ASR; expect mistakes, read what it heard).
Directions: right = +X, left = -X, back = +Y, forward = -Y, up = +Z, down = -Z.
"""
from __future__ import annotations
import argparse
import re
from typing import Callable, Optional

DIRS = {"right": (1, 0, 0), "left": (-1, 0, 0), "back": (0, 1, 0), "backward": (0, 1, 0), "backwards": (0, 1, 0),
        "forward": (0, -1, 0), "forwards": (0, -1, 0), "up": (0, 0, 1), "down": (0, 0, -1)}
NUM = r"(-?\d+(?:\.\d+)?)"
HELP = ("I can: build parts and assemblies (e.g. \"build a hexacopter, 1.5 kg, industrial grade\"), move a part "
        "(\"move body 20 mm right\"), turn a part (\"rotate arm 30 degrees\"), show all sides (\"show 360\"), orbit the view "
        "(\"orbit right 45 degrees\"), and answer questions about what I built (size, printable, safety factor, parts, "
        "materials). I can't chat about other topics.")


def make_speaker() -> Optional[Callable]:
    """OS text-to-speech (not Quadmesh's own TTS). Returns None if pyttsx3 is missing."""
    try:
        import pyttsx3
        eng = pyttsx3.init()

        def speak(t):
            eng.say(t); eng.runAndWait()
        return speak
    except Exception:
        return None


class Assistant:
    def __init__(self, desk=None, out_dir: str = "views", speak: Optional[Callable] = None,
                 build_fn: Optional[Callable] = None, log=print):
        self.desk, self.out_dir, self.speak, self.build_fn, self.log = desk, out_dir, speak, build_fn, log
        self.last: dict = {}
        self.name = None                            # the part "it" refers to

    # ---- memory of the last design -------------------------------------------------------------
    def remember(self, result):
        """Accepts autopilot.Result or AssemblyResult."""
        self.last = {"result": result}
        if hasattr(result, "parts") and isinstance(getattr(result, "parts"), dict):
            self.last["assembly"] = result

    # ---- helpers ---------------------------------------------------------------------------------
    def _console(self, op, **args):
        if self.desk is None:
            return False
        self.desk.call(op, **args)
        return True

    def _say(self, text: str) -> str:
        self.log(f"[quadmesh] {text}")
        if self.speak:
            self.speak(text)
        return text

    # ---- the brain: rules -> action or answer ------------------------------------------------------
    def handle(self, text: str) -> str:
        t = text.strip().lower()
        if not t:
            return ""
        if re.search(r"\b(help|what can you do)\b", t):
            return self._say(HELP)

        m = re.search(r"orbit\s+(left|right)\s*(?:by\s*)?" + NUM + r"?\s*(?:degrees|deg)?", t)
        if m and self.desk:
            deg = float(m.group(2) or 15)
            for _ in range(max(1, round(deg / 15))):
                self.desk.io.key("num6" if m.group(1) == "right" else "num4")
            return self._say(f"Orbited the view {m.group(1)} by about {round(deg / 15) * 15} degrees.")

        if re.search(r"\b(360|turntable|all sides|around it|every side|turn around|show me the (?:part|object|model))\b", t):
            if self.desk is None:
                return self._say("I need desk mode to turn the view with the mouse and keyboard.")
            frames, gif = self.desk.turntable(self.out_dir, tag="turn")
            return self._say(f"Showed the part all the way around: {len(frames)} views, 15 degrees apart"
                             + (f", saved as {gif}." if gif else "."))

        m = re.search(r"move\s+(?:the\s+)?(\w+)\s+(?:by\s+)?" + NUM + r"\s*(?:mm|millimet\w+)?\s*(left|right|back\w*|forward\w*|up|down)", t)
        if m:
            name, dist, d = m.group(1), float(m.group(2)), DIRS[m.group(3)]
            if not self._console("nudge", name=name, dx=d[0] * dist, dy=d[1] * dist, dz=d[2] * dist):
                return self._say("I need desk mode (or the Blender bridge) to move parts.")
            self.name = name
            return self._say(f"Moved {name} {dist:g} mm {m.group(3)}.")

        m = re.search(r"(?:rotate|turn)\s+(?:the\s+)?(\w+)\s+(?:by\s+)?" + NUM + r"\s*(?:degrees|deg)", t)
        if m and m.group(1) not in ("view", "camera"):
            if not self._console("rotate", name=m.group(1), axis="z", deg=float(m.group(2))):
                return self._say("I need desk mode (or the Blender bridge) to turn parts.")
            return self._say(f"Turned {m.group(1)} by {float(m.group(2)):g} degrees around the vertical axis.")

        if re.match(r"\s*(make|build|design|create|generate)\b", t):
            if self.build_fn is None:
                return self._say("Building is not connected in this session.")
            self._say("Working on it. I will check every part before I say it is printable.")
            res = self.build_fn(text)
            self.remember(res)
            return self._say(self._summary())

        if self.last:
            ans = self._answer(t)
            if ans:
                return self._say(ans)
        return self._say("I can't answer that. " + HELP if not self.last else
                         "I don't have that information. Ask about size, printable, safety factor, parts or materials.")

    # ---- answers from real numbers -----------------------------------------------------------------
    def _summary(self) -> str:
        r = self.last.get("result")
        if r is None:
            return "Nothing has been built yet."
        if not getattr(r, "ok", False):
            probs = getattr(r, "problems", []) or ["it could not be completed"]
            return "It did NOT pass, so do not print. Problem: " + "; ".join(str(p) for p in probs[:2])
        if "assembly" in self.last:
            names = ", ".join(f"{n}" for n in r.parts)
            return f"Done. Every part passed the checks: {names}. The files and calculations are saved next to the STL files."
        return f"Done. The part passed the printability checks and was saved to {r.stl_path}."

    def _answer(self, t: str) -> Optional[str]:
        r = self.last["result"]
        rep = getattr(r, "report", None)
        asm = "assembly" in self.last
        if re.search(r"\b(printable|print it|can i print|safe to print|passed)\b", t):
            return self._summary()
        if re.search(r"\b(safety factor|strong|strength|hold|break|load)\b", t):
            lines = [x for x in (rep if asm and isinstance(rep, list) else []) if "safety factor" in x or "load" in x]
            return " ".join(lines) if lines else "I only calculate strength for assemblies (hexacopter, robot arm)."
        if re.search(r"\b(how big|size|dimensions?|how large|how long)\b", t):
            if not asm and isinstance(rep, dict) and rep.get("dims"):
                d = rep["dims"]; return f"{d[0]:g} by {d[1]:g} by {d[2]:g} millimetres."
            return "Ask me about one part, or check the sizes in assembly.json."
        if re.search(r"\b(parts?|how many|quantity|bom|what did you (?:build|make))\b", t):
            if asm:
                return "; ".join(f"{n}" for n in r.parts) or "No parts."
            return self._summary()
        if re.search(r"\b(material|filament|plastic)\b", t):
            spec = getattr(r, "spec", None) or {}
            return f"Material: {spec.get('material', 'PETG (default)')}." if asm else "I only record the material for assemblies."
        if re.search(r"\b(wall|thin|thickness)\b", t) and isinstance(rep, dict) and rep.get("min_wall") is not None:
            return f"The thinnest wall is about {rep['min_wall']:g} millimetres."
        if re.search(r"\b(overhang|support)\b", t) and isinstance(rep, dict) and rep.get("overhang_pct") is not None:
            return f"{rep['overhang_pct']:g} percent of the surface overhangs more than 45 degrees."
        return None


def main():
    ap = argparse.ArgumentParser(description="Talk to Quadmesh")
    ap.add_argument("--voice", action="store_true", help="listen with the microphone (from-scratch ASR)")
    ap.add_argument("--speak", action="store_true", help="read answers aloud with the OS voice")
    ap.add_argument("--desk", action="store_true", help="control the real keyboard and mouse (needed for moving/turning/360)")
    ap.add_argument("--tier", default=None)
    ap.add_argument("--out", default="designs")
    a = ap.parse_args()
    desk = None
    if a.desk:
        from quadmesh.agent.launch import ensure_blender
        ensure_blender()                                   # opens Blender by itself if it is not running
        from quadmesh.agent.desk import DeskController
        desk = DeskController()
        desk.setup()
    from quadmesh.agent.autopilot import design, design_assembly

    def build(text):
        if re.search(r"hexa|hex |six|robot|arm", text.lower()):
            return design_assembly(text, tier=a.tier, out_dir=a.out, desk=a.desk)
        return design(text, tier=a.tier, out_dir=a.out, desk=a.desk)
    bot = Assistant(desk=desk, speak=make_speaker() if a.speak else None, build_fn=build)
    print("Quadmesh is listening. Say or type 'help'. Say 'quit' to stop.")
    while True:
        if a.voice:
            from quadmesh.agent.speech import listen
            text = listen()
            if not text:
                continue
            print(f"you> {text}")
        else:
            text = input("you> ")
        if text.strip().lower() in ("quit", "exit", "stop", "bye"):
            break
        bot.handle(text)


if __name__ == "__main__":
    main()
