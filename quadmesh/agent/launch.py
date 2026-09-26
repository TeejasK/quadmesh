"""
Opens Blender BY ITSELF with the Quadmesh bridge loaded - you never open it by hand.

    python -m quadmesh.agent.launch                        # start Blender (if it is not already running) and wait for the bridge
    python -m quadmesh.agent.launch --blender-path "D:\\Tools\\Blender 4.5\\blender.exe"     # tell it once where Blender is
    python -m quadmesh.agent.launch --keep-scene           # do not delete Blender's default cube/camera/light

Every entry point (autopilot, assistant, app, image/gear/thread tools) calls ensure_blender() first, so a normal run is just
`python -m quadmesh.agent.autopilot "..."`.  Blender is looked up in this order: QUADMESH_BLENDER, the path saved by
--blender-path, PATH, every "Blender x.y" folder under Program Files / Program Files (x86) / LocalAppData / Steam / any drive
root (newest version wins), macOS and Linux locations.
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import subprocess
import sys
import time
from typing import List, Optional

BRIDGE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_bridge.py")
SAVE_FILE = os.path.join(os.path.expanduser("~"), ".quadmesh", "blender_path.txt")
CLEAN_EXPR = "import bpy\nfor o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)"


def _version_key(path: str):
    m = re.findall(r"(\d+(?:\.\d+)*)", os.path.dirname(path).replace("\\", "/").split("/")[-1])
    return tuple(int(x) for x in m[-1].split(".")) if m else (0,)


def candidate_paths(extra_roots: Optional[List[str]] = None, include_system_dirs: bool = True) -> List[str]:
    found: List[str] = []
    roots = list(extra_roots or [])
    if include_system_dirs:
        if sys.platform.startswith("win"):
            for env in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData", "ProgramW6432"):
                v = os.environ.get(env)
                if v:
                    roots += [v, os.path.join(v, "Programs")]
            for d in "CDEFGH":
                roots.append(f"{d}:\\")
        elif sys.platform == "darwin":
            roots += ["/Applications", os.path.expanduser("~/Applications")]
        else:
            roots += ["/usr/bin", "/usr/local/bin", "/opt", "/snap/bin", os.path.expanduser("~")]
    if sys.platform.startswith("win"):
        pats = ["Blender Foundation/Blender */blender.exe", "Blender*/blender.exe", "Steam/steamapps/common/Blender/blender.exe",
                "Programs/Blender Foundation/Blender */blender.exe", "*/Blender Foundation/Blender */blender.exe",
                "Program Files/Blender Foundation/Blender */blender.exe", "Tools/Blender*/blender.exe"]
    elif sys.platform == "darwin":
        pats = ["Blender.app/Contents/MacOS/Blender"]
    else:
        pats = ["blender", "blender*/blender", "*/blender"]
    generic = ["Blender Foundation/Blender */blender.exe", "Blender Foundation/Blender */blender",
               "Blender*/blender.exe", "Blender*/blender"]
    for r in roots:
        for p in pats + (generic if r in (extra_roots or []) else []):
            found += glob.glob(os.path.join(r, p))
    return [f for f in dict.fromkeys(found) if os.path.isfile(f)]


def find_blender_exe(extra_roots: Optional[List[str]] = None, include_system_dirs: bool = True) -> Optional[str]:
    env = os.environ.get("QUADMESH_BLENDER")
    if env and os.path.isfile(env):
        return env
    if os.path.isfile(SAVE_FILE):
        p = open(SAVE_FILE, encoding="utf-8").read().strip()
        if os.path.isfile(p):
            return p
    from shutil import which
    w = which("blender") or which("blender.exe")
    if w:
        return w
    cands = candidate_paths(extra_roots, include_system_dirs=include_system_dirs)
    return max(cands, key=_version_key) if cands else None


def set_blender_path(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
    open(SAVE_FILE, "w", encoding="utf-8").write(path)


def build_command(exe: str, clean_scene: bool = True, geometry=None) -> List[str]:
    cmd = [exe]
    if geometry:
        cmd += ["--window-geometry", *[str(int(v)) for v in geometry]]
    cmd += ["--python", BRIDGE_SCRIPT]
    if clean_scene:
        cmd += ["--python-expr", CLEAN_EXPR]
    return cmd


def ensure_blender(timeout: float = 120.0, clean_scene: bool = True, log=print, is_up=None, popen=subprocess.Popen,
                   focus: bool = True) -> bool:
    """Start Blender if the bridge is not answering, then wait until it is. True when the bridge is up."""
    if is_up is None:
        from quadmesh.agent.blender_client import is_up as _iu
        is_up = _iu
    if is_up():
        return True
    exe = find_blender_exe()
    if not exe:
        log("[launch] Blender was not found. Tell Quadmesh once where it is:\n"
            '         python -m quadmesh.agent.launch --blender-path "C:\\Path\\To\\blender.exe"')
        return False
    geometry = None
    try:
        import pyautogui
        w, h = pyautogui.size()
        geometry = (0, 0, w, h)
    except Exception:
        pass
    log(f"[launch] starting Blender: {exe}")
    popen(build_command(exe, clean_scene, geometry))
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_up():
            log("[launch] Blender is up and the Quadmesh bridge is running.")
            if focus:
                try:
                    from quadmesh.agent.desktop import find_and_focus_window
                    find_and_focus_window("Blender")
                except Exception:
                    pass
            return True
        time.sleep(0.5)
    log("[launch] Blender started but the bridge did not answer in time. Open Blender's Scripting tab and check for errors.")
    return False


def open_blender(wait_seconds: float = 6.0) -> bool:      # kept for the older app/repl callers
    return ensure_blender(timeout=max(wait_seconds, 60.0))


def main():
    ap = argparse.ArgumentParser(description="Start Blender with the Quadmesh bridge")
    ap.add_argument("--blender-path", default=None)
    ap.add_argument("--keep-scene", action="store_true")
    ap.add_argument("--find", action="store_true", help="only print where Blender was found")
    a = ap.parse_args()
    if a.blender_path:
        set_blender_path(a.blender_path)
        print("saved:", a.blender_path)
    if a.find:
        print(find_blender_exe() or "not found")
        return
    raise SystemExit(0 if ensure_blender(clean_scene=not a.keep_scene) else 1)


if __name__ == "__main__":
    main()
