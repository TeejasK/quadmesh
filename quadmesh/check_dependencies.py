"""
Loud dependency check. Run this once after `pip install -r requirements.txt`, or import
`require(feature)` at the top of any entry point that needs a specific optional feature.

Why this exists: several modules already degrade silently when an optional package is missing (image3d.py
falls back to a synthetic generator, agent/speech.py falls back through two ASR paths) - that's the right
behavior for THOSE specific cases, where a documented fallback exists and prints a warning. But several
other paths (cadquery for STEP export, pyautogui for real desktop control, opencv-python for VideoCAD frame
extraction) have no such fallback: they just raise an ImportError from wherever they're first used, often
deep into a run, with no indication beforehand of what's missing or how to fix it. This module is what
turns that into a loud, immediate, actionable failure - or an explicit "not available" - at the point you
choose to check, not wherever Python happens to hit the missing import first.

    python -m quadmesh.check_dependencies              # check everything, print a report, exit 1 if core is missing
    python -m quadmesh.check_dependencies --feature step_export   # check one feature only
"""
from __future__ import annotations
import importlib
import sys

CORE = {
    "torch": "training and inference for every role model",
    "numpy": "array math used throughout (mesh_tools, geometry3d, vision, etc.)",
    "PIL": "image loading (photo3d.py, data/vla_data.py) - pip name is 'pillow'",
    "datasets": "streaming Hugging Face training data (data/streaming.py, data/vla_data.py)",
    "tokenizers": "the BPE tokenizer (data/tokenizer.py)",
}

# feature name -> (import name, pip name if different, what breaks without it, has a documented fallback?)
OPTIONAL = {
    "step_export":     ("cadquery", "cadquery", "step_io.py: exporting to STEP instead of just STL", False),
    "desktop_control": ("pyautogui", "pyautogui", "agent/desk.py, agent/desktop.py: real mouse/keyboard control", False),
    "asr_fallback":    ("speech_recognition", "SpeechRecognition", "agent/speech.py: pretrained ASR fallback path "
                        "(the trained-model path still works without this)", True),
    "videocad_frames": ("cv2", "opencv-python", "data/vla_data.py: VideoCAD Dataverse fallback frame extraction "
                        "(the whole VideoCAD source degrades to skipped, not a crash, without this)", True),
    "http_api":        ("fastapi", "fastapi", "agent/app.py: the local HTTP surface, if you use it", False),
}


def _check(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def require(feature: str) -> None:
    """Call at the top of a module/function that needs one OPTIONAL feature. Raises a clear, actionable
    RuntimeError immediately - never a bare ImportError from three call-frames deep - if it's missing."""
    if feature not in OPTIONAL:
        raise ValueError(f"unknown feature {feature!r}; known: {sorted(OPTIONAL)}")
    import_name, pip_name, what_breaks, has_fallback = OPTIONAL[feature]
    if not _check(import_name):
        raise RuntimeError(f"Missing dependency for '{feature}': {what_breaks}. "
                            f"Install it with: pip install {pip_name}=={_pinned_version(pip_name)}")


def _pinned_version(pip_name: str) -> str:
    import os
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    try:
        with open(req_path) as f:
            for line in f:
                if line.strip().lower().startswith(pip_name.lower() + "=="):
                    return line.strip().split("==")[1]
    except FileNotFoundError:
        pass
    return "<see requirements.txt>"


def report() -> bool:
    """Prints a loud pass/fail report for everything. Returns True only if every CORE dependency is present -
    optional ones missing are reported but don't fail the overall check, since documented fallbacks exist
    for some of them (see OPTIONAL's fourth field)."""
    print("=== CORE (required) ===")
    core_ok = True
    for name, why in CORE.items():
        ok = _check(name)
        core_ok &= ok
        print(f"  [{'OK' if ok else 'MISSING'}] {name} - {why}")
    print()
    print("=== OPTIONAL (feature-specific) ===")
    for feature, (import_name, pip_name, what_breaks, has_fallback) in OPTIONAL.items():
        ok = _check(import_name)
        tag = "OK" if ok else ("MISSING (has fallback)" if has_fallback else "MISSING (no fallback)")
        print(f"  [{tag}] {feature} ({pip_name}) - {what_breaks}")
    if not core_ok:
        print()
        print("CORE dependency missing - install with: pip install -r requirements.txt")
    return core_ok


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", help="check one optional feature and exit 0/1")
    args = ap.parse_args()
    if args.feature:
        try:
            require(args.feature)
            print(f"OK: {args.feature}")
        except (RuntimeError, ValueError) as e:
            print(f"FAIL: {e}")
            sys.exit(1)
        return
    ok = report()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
