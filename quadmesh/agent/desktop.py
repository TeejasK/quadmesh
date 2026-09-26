"""
Local desktop control — runs on YOUR machine, not Modal.

This is the "thin, heavily-verified" GUI layer from Sec 2/10: the planner
never emits raw pixel coordinates itself except through here, and this layer
only ever executes a small fixed vocabulary of actions (move, click, type,
key, screenshot). Everything else in the pipeline stays symbolic.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

try:
    import pyautogui
    pyautogui.FAILSAFE = True     # slam mouse to a screen corner to abort, always on
    pyautogui.PAUSE = 0.05
except ImportError:
    pyautogui = None



@dataclass
class Action:
    kind: str              # "move" | "click" | "double_click" | "type" | "key" | "drag" | "wait"
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    key: Optional[str] = None
    seconds: float = 0.0


def screenshot():
    if pyautogui is None:
        raise RuntimeError("pyautogui is not installed. Run `pip install pyautogui` for desktop automation.")
    return pyautogui.screenshot()


def execute(action: Action, human_speed: bool = True):
    """The ONLY function that ever touches your real mouse/keyboard."""
    if pyautogui is None:
        raise RuntimeError("pyautogui is not installed. Run `pip install pyautogui` for desktop automation.")
    dur = 0.25 if human_speed else 0.0
    if action.kind == "move":
        pyautogui.moveTo(action.x, action.y, duration=dur)
    elif action.kind == "click":
        pyautogui.click(action.x, action.y, duration=dur)
    elif action.kind == "double_click":
        pyautogui.doubleClick(action.x, action.y, duration=dur)
    elif action.kind == "type":
        pyautogui.typewrite(action.text, interval=0.02)
    elif action.kind == "key":
        pyautogui.press(action.key)
    elif action.kind == "drag":
        pyautogui.dragTo(action.x, action.y, duration=max(dur, 0.3))
    elif action.kind == "wait":
        time.sleep(action.seconds)
    else:
        raise ValueError(f"unknown action kind {action.kind!r}")


def find_and_focus_window(title_substring: str = "Blender") -> bool:
    """Best-effort window focus so clicks land on Blender, not whatever else
    is on screen. Uses pygetwindow; falls back to manual Alt-Tab if unavailable."""
    try:
        import pygetwindow as gw
        wins = [w for w in gw.getAllTitles() if title_substring.lower() in w.lower()]
        if not wins:
            return False
        win = gw.getWindowsWithTitle(wins[0])[0]
        win.activate()
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f"[desktop] could not auto-focus window ({e}); "
              f"click into Blender manually once before starting.")
        return False
