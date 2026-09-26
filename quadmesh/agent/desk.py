"""
Desk control: builds and checks a design using ONLY the real keyboard and mouse, like a person at the desk.

How it works (all plain Blender shortcuts):
  * the mouse hovers the big area in the middle of the screen (Blender sends keys to the area under the mouse);
  * Shift+F4 turns that area into the Python Console, Shift+F5 turns it back into the 3D viewport;
  * every plan step is TYPED into the console as one line and Enter is pressed;
  * Home / Numpad 1,3,7 frame the part and show front / right / top views; the middle mouse button orbits;
  * the STL is exported the same way, and the exported FILE is what gets checked.

Limits (be aware):
  * numbers and shapes are typed, not clicked - exact numeric input is how engineers work, and a click
    cannot hit a millimetre-exact spot without a trained screen-grounding model (ours is not trained on real data);
  * typing assumes a US keyboard layout; install `pyperclip` to paste long lines instead of typing them;
  * Blender must be the visible window and the bridge/ops script must have been run once in it
    (it registers the ops for the console).  pyautogui's fail-safe stays ON: slam the mouse into a
    screen corner to stop everything at once.
"""
from __future__ import annotations
import json
import os
import time
from typing import Optional

HELPER = '_q = lambda op, **a: bpy.app.driver_namespace["quadmesh_ops"][op](**a)'

# how long Blender needs after each op before the next line is typed (seconds)
SLOW_OPS = {"union": 1.0, "cut": 1.0, "hole_center": 1.0, "hole_at": 1.0, "bolt_circle": 2.0, "holes_grid": 2.0,
            "holes_corners": 1.5, "holes_pair_x": 1.2, "holes_pair_y": 1.2, "holes_row_x": 2.0, "holes_row_y": 2.0,
            "fillet": 1.5, "chamfer": 1.5, "shell_open_top": 1.0, "extrude_polygons": 1.5, "save_stl": 2.0,
            "array_x": 2.0, "array_y": 2.0}


class PyAutoGuiIO:
    """The ONLY class that touches the real keyboard and mouse."""

    def __init__(self):
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.03
        self.p = pyautogui

    def size(self):
        return tuple(self.p.size())

    def move(self, x, y, dur=0.25):
        self.p.moveTo(x, y, duration=dur)

    def click(self, x, y, button="left", clicks=1):
        self.p.click(x, y, clicks=clicks, button=button, interval=0.12)

    def stroke(self, pts, button="left", dur=0.15):
        """Press at the first point, glide through the rest, release - one continuous mouse stroke."""
        self.p.moveTo(pts[0][0], pts[0][1], duration=0.2)
        self.p.mouseDown(button=button)
        try:
            for x, y in pts[1:]:
                self.p.moveTo(x, y, duration=dur)
        finally:
            self.p.mouseUp(button=button)

    def drag_from(self, x1, y1, x2, y2, button="left", dur=0.5):
        self.p.moveTo(x1, y1, duration=0.25)
        self.p.dragTo(x2, y2, duration=dur, button=button)

    def key(self, name):
        self.p.press(name)

    def hotkey(self, *names):
        self.p.hotkey(*names)

    def type(self, text, interval=0.008):
        self.p.typewrite(text, interval=interval)

    def paste(self, text):
        import pyperclip
        pyperclip.copy(text)
        self.p.hotkey("ctrl", "v")

    def drag(self, x, y, button="middle", dur=0.4):
        self.p.dragTo(x, y, duration=dur, button=button)

    def scroll(self, clicks):
        self.p.scroll(clicks)

    def screenshot(self):
        return self.p.screenshot()

    def wait(self, s):
        time.sleep(s)


class DeskController:
    def __init__(self, io=None, log=print, paste_threshold: int = 600, focus=True):
        self.io = io or PyAutoGuiIO()
        self.log = log
        self.paste_threshold = paste_threshold
        self._focus = focus
        self.mode = None                     # "console" | "viewport"
        self.file_timeout = 90.0

    # ---- window / area handling -----------------------------------------------------
    def center(self):
        w, h = self.io.size()
        return w // 2, h // 2

    def setup(self):
        if self._focus:
            try:
                from quadmesh.agent.desktop import find_and_focus_window
                find_and_focus_window("Blender")
            except Exception as e:
                self.log(f"[desk] could not auto-focus Blender ({e}) - click into Blender once")
        self.to_console()
        self.line(HELPER)
        self.log("[desk] console ready: the agent now types every step into Blender's Python Console")

    def to_console(self):
        if self.mode != "console":
            self.io.move(*self.center())
            self.io.hotkey("shift", "f4")
            self.io.wait(0.5)
            self.mode = "console"

    def to_viewport(self):
        if self.mode != "viewport":
            self.io.move(*self.center())
            self.io.hotkey("shift", "f5")
            self.io.wait(0.5)
            self.mode = "viewport"

    # ---- typing ----------------------------------------------------------------------
    def line(self, text: str):
        """Type one console line and press Enter."""
        self.to_console()
        if len(text) > self.paste_threshold and hasattr(self.io, "paste"):
            try:
                self.io.paste(text)
            except Exception:
                self.io.type(text)
        else:
            self.io.type(text)
        self.io.key("enter")

    def call(self, op: str, **args):
        """Run one Blender op by typing it: _q("add_box", **{"name": "body", ...})"""
        self.line(f"_q({json.dumps(op)}, **{json.dumps(args)})")
        self.io.wait(SLOW_OPS.get(op, 0.4))

    def gui_step(self, step: dict) -> bool:
        """Do a step with REAL shortcuts and mouse (Shift+A, S, G, F2 ...) if a recipe exists. True if done."""
        from quadmesh.agent.gui_recipes import gui_actions
        from quadmesh.agent.actions import run_text
        lines = gui_actions(step)
        if lines is None:
            return False
        self.to_viewport()
        w, h = self.io.size()
        err = run_text(self.io, lines, (0, 0, w, h))
        if err:
            raise RuntimeError(err)
        return True

    def run_steps(self, steps: list, gui: bool = False):
        for i, s in enumerate(steps):
            if gui and self.gui_step(s):
                self.log(f"[desk] step {i+1}/{len(steps)}: {s['op']} - made with keyboard shortcuts and mouse in the viewport")
                continue
            self.log(f"[desk] typing step {i+1}/{len(steps)}: {s['op']}")
            self.call(s["op"], **s["args"])

    # ---- looking ---------------------------------------------------------------------
    def views(self, out_dir: str, tag: str) -> list:
        """Frame the part and take front / right / top / perspective screenshots."""
        os.makedirs(out_dir, exist_ok=True)
        self.to_viewport()
        self.io.move(*self.center())
        shots = []
        for label, key in (("iso", None), ("front", "num1"), ("right", "num3"), ("top", "num7")):
            if key:
                self.io.key(key)
            self.io.key("home")                            # View All
            self.io.wait(0.5)
            path = os.path.join(out_dir, f"{tag}_{label}.png")
            self.io.screenshot().save(path)
            shots.append(path)
        return shots

    def turntable(self, out_dir: str, tag: str = "turn", steps: int = 24, use_mouse: bool = False, gif: bool = True):
        """Show the part from ALL sides: a full 360 degrees in `steps` equal turns, a screenshot after each.
        Keyboard mode (default) presses Numpad 6 - Blender orbits 15 degrees per press (its default Orbit Step),
        so steps=24 is exactly 360. Mouse mode drags with the middle button: smooth but only approximately 360."""
        os.makedirs(out_dir, exist_ok=True)
        self.to_viewport()
        cx, cy = self.center()
        self.io.move(cx, cy)
        self.io.key("home")                                    # frame the whole part
        self.io.wait(0.4)
        presses = max(1, round((360.0 / steps) / 15.0))
        frames = []
        for i in range(steps):
            path = os.path.join(out_dir, f"{tag}_{i:02d}.png")
            self.io.screenshot().save(path)
            frames.append(path)
            if use_mouse:
                self.io.drag_from(cx - 120, cy, cx + 120, cy, button="middle")
            else:
                for _ in range(presses):
                    self.io.key("num6")
            self.io.wait(0.35)
        gif_path = None
        if gif:
            try:
                from PIL import Image
                ims = [Image.open(p).convert("RGB").resize((640, 360)) for p in frames]
                gif_path = os.path.join(out_dir, f"{tag}.gif")
                ims[0].save(gif_path, save_all=True, append_images=ims[1:], duration=120, loop=0)
            except Exception as e:
                self.log(f"[desk] could not build the GIF ({e}); the screenshots are saved")
        return frames, gif_path

    def orbit(self, dx: int = 200, dy: int = 0):
        """Rotate the view with the middle mouse button, like turning the part in your hand."""
        self.to_viewport()
        x, y = self.center()
        self.io.move(x, y)
        self.io.drag(x + dx, y + dy, button="middle")

    def zoom(self, clicks: int = 3):
        self.to_viewport()
        self.io.move(*self.center())
        self.io.scroll(clicks)

    # ---- files -----------------------------------------------------------------------
    def wait_for_file(self, path: str, timeout: Optional[float] = None) -> bool:
        timeout = self.file_timeout if timeout is None else timeout
        t0, last = time.time(), -1
        while time.time() - t0 < timeout:
            if os.path.exists(path):
                sz = os.path.getsize(path)
                if sz > 0 and sz == last:
                    return True                             # stopped growing: the write finished
                last = sz
            time.sleep(0.5)
        return False
