"""
The action language of Quadmesh's Vision-Language-Action model.

One action is one line of text, so the SAME transformer that reads the instruction also writes the action,
and the same tokenizer/LM head is used. Coordinates are normalised 0..999 over the screenshot the model saw
(0,0 = top-left), so it does not matter what the real screen resolution is.

    move X Y | click X Y | double_click X Y | right_click X Y | drag X1 Y1 X2 Y2 | middle_drag X1 Y1 X2 Y2
    scroll N | key NAME | hotkey a+b+c | type "text" | wait S | done

Everything the model outputs is parsed and SANITISED here before it can touch the real keyboard/mouse.
"""
from __future__ import annotations
import os
import re
import shlex
import time
from dataclasses import dataclass
from typing import Optional

COORD_MAX = 999
MAX_TYPE_LEN = 300
KIND_ARGS = {"move": 2, "click": 2, "double_click": 2, "right_click": 2, "drag": 4, "middle_drag": 4,
             "scroll": 1, "wait": 1}
LETTERS = set("abcdefghijklmnopqrstuvwxyz0123456789")
SPECIAL_KEYS = {"enter", "esc", "escape", "tab", "space", "backspace", "delete", "home", "end", "up", "down", "left",
                "right", "pageup", "pagedown", "shift", "ctrl", "alt", "insert", "period", "comma",
                *(f"f{i}" for i in range(1, 13)), *(f"num{i}" for i in range(10)),
                "add", "subtract", "multiply", "divide", "decimal", "-", "=", "/", ".", ","}
ALLOWED_KEYS = LETTERS | SPECIAL_KEYS
# things that could close programs, lock or leave the session: never allowed
BLOCKED_HOTKEYS = [{"alt", "f4"}, {"ctrl", "w"}, {"ctrl", "q"}, {"ctrl", "alt", "delete"}, {"win"}, {"alt", "tab"},
                   {"ctrl", "shift", "esc"}, {"ctrl", "shift", "delete"}, {"ctrl", "alt"}]


UNRESTRICTED = os.environ.get("QUADMESH_UNRESTRICTED", "0") == "1"
AUDIT_LOG = os.environ.get("QUADMESH_ACTION_LOG", "")


def set_unrestricted(flag: bool = True, audit_log: str = "") -> None:
    """Full control: every key and hotkey is allowed (Alt+F4, Win key, ...). pyautogui's fail-safe (slam the mouse into a
    screen corner) ALWAYS stays on, and with audit_log set every executed action is appended to that file."""
    global UNRESTRICTED, AUDIT_LOG
    UNRESTRICTED = bool(flag)
    AUDIT_LOG = audit_log or AUDIT_LOG
    os.environ["QUADMESH_UNRESTRICTED"] = "1" if flag else "0"


def _key_ok(k: str) -> bool:
    return k in ALLOWED_KEYS or (UNRESTRICTED and re.fullmatch(r"[a-z0-9_\-=/.,;'\[\]\\`]+|f(1[0-9]|2[0-4])", k) is not None)


@dataclass
class Action:
    kind: str
    x: Optional[int] = None
    y: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None
    text: Optional[str] = None
    keys: Optional[list] = None
    n: Optional[float] = None
    pts: Optional[list] = None
    button: Optional[str] = None

    def __str__(self):
        return fmt(self)


def fmt(a: Action) -> str:
    k = a.kind
    if k in ("move", "click", "double_click", "right_click"):
        return f"{k} {a.x} {a.y}"
    if k in ("drag", "middle_drag"):
        return f"{k} {a.x} {a.y} {a.x2} {a.y2}"
    if k == "scroll":
        return f"scroll {int(a.n)}"
    if k == "wait":
        return f"wait {a.n:g}"
    if k == "key":
        return f"key {a.keys[0]}"
    if k == "hotkey":
        return "hotkey " + "+".join(a.keys)
    if k == "type":
        return 'type "' + (a.text or "").replace("\\", "\\\\").replace('"', '\\"') + '"'
    if k == "stroke":
        return f"stroke {a.button} " + " ".join(f"{x} {y}" for x, y in a.pts)
    if k == "done":
        return "done"
    raise ValueError(k)


def parse(text: str) -> Optional[Action]:
    """Text -> Action, or None if it is not a valid action. Never raises."""
    try:
        text = text.strip().splitlines()[0].strip()
        kind, _, rest = text.partition(" ")
        kind = kind.lower()
        if kind == "done" and not rest.strip():
            return Action("done")
        if kind in KIND_ARGS:
            nums = [float(t) for t in rest.split()]
            if len(nums) != KIND_ARGS[kind]:
                return None
            if kind == "wait":
                return Action("wait", n=min(max(nums[0], 0.0), 10.0))
            if kind == "scroll":
                return Action("scroll", n=max(-30, min(30, int(nums[0]))))
            c = [max(0, min(COORD_MAX, int(round(v)))) for v in nums]
            if kind in ("drag", "middle_drag"):
                return Action(kind, c[0], c[1], c[2], c[3])
            return Action(kind, c[0], c[1])
        if kind == "stroke":                    # press, move along a polyline, release: sculpting, curve drawing, free drags
            parts = rest.split()
            if len(parts) < 5 or parts[0] not in ("left", "middle", "right"):
                return None
            nums = [max(0, min(COORD_MAX, int(round(float(v))))) for v in parts[1:]]
            if len(nums) % 2 or not 2 <= len(nums) // 2 <= 64:
                return None
            return Action("stroke", button=parts[0], pts=list(zip(nums[0::2], nums[1::2])))
        if kind == "key":
            k = rest.strip().lower()
            return Action("key", keys=[k]) if _key_ok(k) else None
        if kind == "hotkey":
            ks = [k.strip().lower() for k in rest.split("+")]
            if not 2 <= len(ks) <= 4 or any(not _key_ok(k) for k in ks):
                return None
            if not UNRESTRICTED and any(b <= set(ks) for b in BLOCKED_HOTKEYS):
                return None
            return Action("hotkey", keys=ks)
        if kind == "type":
            parts = shlex.split(rest, posix=True)
            if len(parts) != 1:
                return None
            s = parts[0]
            if len(s) > (2000 if UNRESTRICTED else MAX_TYPE_LEN) or not all(32 <= ord(ch) < 127 for ch in s):
                return None
            return Action("type", text=s)
    except Exception:
        return None
    return None


def to_pixels(a: Action, region) -> Action:
    """Normalised 0..999 -> real screen pixels inside `region` = (left, top, width, height)."""
    L, T, W, H = region
    px = lambda v: int(L + (v / COORD_MAX) * (W - 1))
    py = lambda v: int(T + (v / COORD_MAX) * (H - 1))
    b = Action(a.kind, keys=a.keys, text=a.text, n=a.n)
    if a.x is not None:
        b.x, b.y = px(a.x), py(a.y)
    if a.x2 is not None:
        b.x2, b.y2 = px(a.x2), py(a.y2)
    if a.pts:
        b.pts, b.button = [(px(x), py(y)) for x, y in a.pts], a.button
    return b


def execute(io, a: Action, region) -> None:
    """Run one already-parsed action on the real desk (io = desk.PyAutoGuiIO or a test double)."""
    if a.kind == "done":
        return
    p = to_pixels(a, region)
    if AUDIT_LOG:
        with open(AUDIT_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {a}\n")
    if a.kind == "stroke":
        io.stroke(p.pts, button=p.button)
    elif a.kind == "move":
        io.move(p.x, p.y)
    elif a.kind in ("click", "double_click", "right_click"):
        io.click(p.x, p.y, button="right" if a.kind == "right_click" else "left", clicks=2 if a.kind == "double_click" else 1)
    elif a.kind in ("drag", "middle_drag"):
        io.drag_from(p.x, p.y, p.x2, p.y2, button="middle" if a.kind == "middle_drag" else "left")
    elif a.kind == "scroll":
        io.scroll(int(a.n))
    elif a.kind == "key":
        io.key(a.keys[0])
    elif a.kind == "hotkey":
        io.hotkey(*a.keys)
    elif a.kind == "type":
        io.type(a.text)
    elif a.kind == "wait":
        io.wait(a.n)


def run_text(io, lines, region, log=print) -> Optional[str]:
    """Parse, sanitise and execute a list of action strings. Returns an error message or None."""
    for ln in lines:
        a = parse(ln)
        if a is None:
            return f"refused unsafe or invalid action: {ln!r}"
        execute(io, a, region)
    return None
