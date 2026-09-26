"""
Collect REAL demonstrations for the VLA - screenshots + the mouse/keyboard actions that followed them.

Two sources, both write the same JSONL files (one folder per session):

  1. YOU:      python -m quadmesh.agent.record_demo --task "add a 60x40x5 mm box named body" --out demos
               Model something in Blender as you normally would. Press Scroll Lock when you are done.
               (needs:  pip install pynput pillow mss)
  2. RECIPES:  the desk controller run through RecordingIO - every scripted GUI step is saved as a labelled
               demonstration of real Blender pixels (see record_recipe_demos below).

Each line of demo.jsonl:  {"task", "step", "screenshot", "action", "screen": [W, H]}
The `action` is in the action language of actions.py, coordinates normalised 0..999.
Everything stays on your machine; nothing is uploaded.
"""
from __future__ import annotations
import argparse
import json
import os
import threading
import time
from typing import Optional

from quadmesh.agent.actions import COORD_MAX


def _norm(v, size):
    return max(0, min(COORD_MAX, int(round(v / max(size - 1, 1) * COORD_MAX))))


class RecordingIO:
    """Wraps a real IO object; saves a screenshot BEFORE every action and appends the action to demo.jsonl."""

    def __init__(self, inner, out_dir: str, task: str, session: Optional[str] = None):
        self.io = inner
        self.task = task
        self.dir = os.path.join(out_dir, session or time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(os.path.join(self.dir, "shots"), exist_ok=True)
        self.fh = open(os.path.join(self.dir, "demo.jsonl"), "a", encoding="utf-8")
        self.step = 0
        self.W, self.H = self.io.size()

    def _log(self, action: str):
        self.step += 1
        rel = os.path.join("shots", f"{self.step:04d}.png")
        self.io.screenshot().save(os.path.join(self.dir, rel))
        self.fh.write(json.dumps({"task": self.task, "step": self.step, "screenshot": rel, "action": action,
                                  "screen": [self.W, self.H]}) + "\n")
        self.fh.flush()

    def size(self): return self.io.size()
    def screenshot(self): return self.io.screenshot()

    def move(self, x, y, dur=0.25):
        self._log(f"move {_norm(x, self.W)} {_norm(y, self.H)}"); self.io.move(x, y, dur)

    def click(self, x, y, button="left", clicks=1):
        kind = "right_click" if button == "right" else ("double_click" if clicks == 2 else "click")
        self._log(f"{kind} {_norm(x, self.W)} {_norm(y, self.H)}"); self.io.click(x, y, button, clicks)

    def drag_from(self, x1, y1, x2, y2, button="left", dur=0.5):
        kind = "middle_drag" if button == "middle" else "drag"
        self._log(f"{kind} {_norm(x1, self.W)} {_norm(y1, self.H)} {_norm(x2, self.W)} {_norm(y2, self.H)}")
        self.io.drag_from(x1, y1, x2, y2, button, dur)

    def scroll(self, n): self._log(f"scroll {int(n)}"); self.io.scroll(n)
    def key(self, name): self._log(f"key {name}"); self.io.key(name)
    def hotkey(self, *n): self._log("hotkey " + "+".join(n)); self.io.hotkey(*n)

    def type(self, text, interval=0.008):
        self._log('type "' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'); self.io.type(text, interval)

    def wait(self, s): self.io.wait(s)          # waiting is not an action worth learning from recordings

    def finish(self):
        self._log("done"); self.fh.close()


def record_recipe_demos(task_prefix: str, steps: list, out_dir: str = "demos", io=None):
    """Run plan steps through the GUI recipes on the real screen and record every action as a demonstration."""
    from quadmesh.agent.desk import PyAutoGuiIO, DeskController
    rec = RecordingIO(io or PyAutoGuiIO(), out_dir, task_prefix)
    d = DeskController(io=rec)
    try:
        from quadmesh.agent.desktop import find_and_focus_window
        find_and_focus_window("Blender")
    except Exception:
        pass
    for s in steps:
        rec.task = f"{task_prefix}: {s['op']} {json.dumps(s['args'])}"
        d.gui_step(s)
    rec.finish()
    return rec.dir


# ---------------------------------------------------------------------------
# Recording a human (pynput). Untested here - it needs a real desktop.
# ---------------------------------------------------------------------------
def record_human(task: str, out_dir: str = "demos", fps: float = 3.0, stop_key: str = "scroll_lock"):
    from pynput import keyboard, mouse
    import mss
    from PIL import Image

    sct = mss.mss()
    mon = sct.monitors[1]
    W, H = mon["width"], mon["height"]
    ring, lock, stop = [], threading.Lock(), threading.Event()

    def grabber():
        while not stop.is_set():
            shot = sct.grab(mon)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            with lock:
                ring.append((time.time(), img)); del ring[:-6]
            time.sleep(1.0 / fps)
    threading.Thread(target=grabber, daemon=True).start()

    class Sink:                                                  # reuse RecordingIO's file writer
        def size(self): return (W, H)
        def screenshot(self): return self.img
    sink = Sink()
    rec = RecordingIO(sink, out_dir, task)

    def emit(action: str, t: float):
        with lock:
            older = [im for ts, im in ring if ts <= t - 0.05]
            sink.img = (older[-1] if older else ring[-1][1])      # the frame from just BEFORE the event
        rec._log(action)

    text_buf, mods, state = [], set(), {"press": None}
    MOD = {"shift": "shift", "shift_r": "shift", "ctrl": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl",
           "alt": "alt", "alt_l": "alt", "alt_r": "alt"}

    def flush(t):
        if text_buf:
            s = "".join(text_buf); text_buf.clear()
            emit('type "' + s.replace("\\", "\\\\").replace('"', '\\"') + '"', t)

    def on_press(k):
        t = time.time()
        name = getattr(k, "name", None)
        if name == stop_key:
            stop.set(); return False
        if name in MOD:
            mods.add(MOD[name]); return
        if hasattr(k, "char") and k.char and not (mods - {"shift"}):
            if len(k.char) == 1 and 32 <= ord(k.char) < 127 and not mods:
                text_buf.append(k.char); return
        flush(t)
        key = name or (k.char.lower() if getattr(k, "char", None) else None)
        if not key:
            return
        key = {"page_up": "pageup", "page_down": "pagedown", "esc": "esc", "cmd": "win"}.get(key, key)
        if key.startswith("num_"):
            key = "num" + key[4:]
        emit(("hotkey " + "+".join(sorted(mods) + [key])) if mods else f"key {key}", t)

    def on_release(k):
        name = getattr(k, "name", None)
        if name in MOD:
            mods.discard(MOD[name])

    def on_click(x, y, button, pressed):
        t = time.time()
        flush(t)
        if pressed:
            state["press"] = (x, y, button, t)
        elif state["press"]:
            x0, y0, b0, t0 = state["press"]; state["press"] = None
            nx = lambda v: _norm(v, W); ny = lambda v: _norm(v, H)
            moved = abs(x - x0) + abs(y - y0) > 8
            if moved:
                emit(f"{'middle_drag' if b0 == mouse.Button.middle else 'drag'} {nx(x0)} {ny(y0)} {nx(x)} {ny(y)}", t0)
            else:
                emit(f"{'right_click' if b0 == mouse.Button.right else 'click'} {nx(x0)} {ny(y0)}", t0)

    def on_scroll(x, y, dx, dy):
        t = time.time(); flush(t)
        emit(f"scroll {int(dy)}", t)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as kl, \
            mouse.Listener(on_click=on_click, on_scroll=on_scroll) as ml:
        print(f"[record] recording '{task}' - do the task in Blender, then press Scroll Lock")
        kl.join()
        ml.stop()
    flush(time.time())
    emit("done", time.time())
    rec.fh.close()
    print(f"[record] saved {rec.step} steps to {rec.dir}")
    return rec.dir


def main():
    ap = argparse.ArgumentParser(description="Record a real Blender demonstration")
    ap.add_argument("--task", required=True, help='what you are about to do, e.g. "add a 60x40x5 mm box named body"')
    ap.add_argument("--out", default="demos")
    a = ap.parse_args()
    record_human(a.task, a.out)


if __name__ == "__main__":
    main()
