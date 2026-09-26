"""
Recipes that make shapes with the REAL Blender keyboard shortcuts and mouse - what a person does:

    hover the viewport -> Shift+A, type "cube", Enter        (Add menu, type-to-search)
    S X 30 Enter / S Y 20 Enter / S Z 2.5 Enter              (scale along one axis by typing a number)
    G Z 2.5 Enter                                            (grab, move up so the part sits on the bed)
    F2 body Enter                                            (rename)

Each recipe is a list of action strings in the VLA action language (see actions.py), so the very same
lines are (a) executed by the desk controller, (b) recorded as real demonstrations, and (c) what the
VLA model is trained to produce.  Written from Blender's documented shortcuts (3.x / 4.x, default keymap,
Add-menu search enabled): NOT run against a real Blender here - expect to tune waits on your machine.

Only boxes, cylinders and spheres have a GUI recipe. Booleans, hole patterns, arrays and bevels need many
dialog clicks in Blender; those steps are still typed into the console until the VLA is trained on real
recordings of them.
"""
from __future__ import annotations
from typing import Optional

DEFAULT_CUBE_EDGE = 2.0          # Blender's default cube is 2 units on a side
DEFAULT_RADIUS = 1.0             # default cylinder / UV-sphere radius
DEFAULT_CYL_DEPTH = 2.0


def _n(v: float) -> str:
    return f"{float(v):g}"


def _add(search: str) -> list:
    return ["move 500 500", "wait 0.3", "hotkey shift+a", "wait 0.7", f'type "{search}"', "wait 0.5",
            "key enter", "wait 0.6"]


def _scale_axis(axis: str, factor: float) -> list:
    return ["key s", "wait 0.15", f"key {axis}", f'type "{_n(factor)}"', "key enter", "wait 0.25"]


def _scale_xy(factor: float) -> list:
    return ["key s", "wait 0.15", "hotkey shift+z", f'type "{_n(factor)}"', "key enter", "wait 0.25"]


def _scale_all(factor: float) -> list:
    return ["key s", "wait 0.15", f'type "{_n(factor)}"', "key enter", "wait 0.25"]


def _lift(z: float) -> list:
    return ["key g", "wait 0.15", "key z", f'type "{_n(z)}"', "key enter", "wait 0.25"]


def _rename(name: str) -> list:
    return ["key f2", "wait 0.5", f'type "{name}"', "key enter", "wait 0.4"]


def add_box(name, w, d, h) -> list:
    return (_add("cube") + _scale_axis("x", w / DEFAULT_CUBE_EDGE) + _scale_axis("y", d / DEFAULT_CUBE_EDGE)
            + _scale_axis("z", h / DEFAULT_CUBE_EDGE) + _lift(h / 2.0) + _rename(name))


def add_cyl(name, dia, h) -> list:
    return (_add("cylinder") + _scale_xy(dia / 2.0 / DEFAULT_RADIUS)
            + _scale_axis("z", h / DEFAULT_CYL_DEPTH) + _lift(h / 2.0) + _rename(name))


def add_sphere(name, dia) -> list:
    return _add("uv sphere") + _scale_all(dia / 2.0 / DEFAULT_RADIUS) + _lift(dia / 2.0) + _rename(name)


RECIPES = {"add_box": add_box, "add_cyl": add_cyl, "add_sphere": add_sphere}


def gui_actions(step: dict) -> Optional[list]:
    """Action lines for a plan step, or None if that op has no GUI recipe (it is typed into the console)."""
    fn = RECIPES.get(step.get("op"))
    return fn(**step["args"]) if fn else None


# ---------------------------------------------------------------------------------------------
# Mouse recipes: things a person does by dragging (all in the action language, all recordable as demonstrations).
# Coordinates are normalised 0..999 over the Blender window; (500, 500) is the middle of the viewport.
# ---------------------------------------------------------------------------------------------
def grab_with_mouse(dx: int, dy: int) -> list:
    """G, glide the mouse, click to drop: moves the selected object by dragging it."""
    x, y = max(0, min(999, 500 + dx)), max(0, min(999, 500 + dy))
    return ["move 500 500", "key g", "wait 0.15", f"move {x} {y}", "wait 0.15", f"click {x} {y}"]


def rotate_with_mouse(dx: int) -> list:
    x = max(0, min(999, 500 + dx))
    return ["move 500 500", "key r", "wait 0.15", f"move {x} 500", "wait 0.15", f"click {x} 500"]


def scale_with_mouse(dx: int) -> list:
    x = max(0, min(999, 500 + dx))
    return ["move 500 500", "key s", "wait 0.15", f"move {x} 500", "wait 0.15", f"click {x} 500"]


def extrude_with_mouse(dx: int, dy: int) -> list:
    """Edit mode, extrude the selection along the mouse, back to object mode."""
    x, y = max(0, min(999, 500 + dx)), max(0, min(999, 500 + dy))
    return ["move 500 500", "key tab", "wait 0.4", "key e", "wait 0.15", f"move {x} {y}", "wait 0.15", f"click {x} {y}", "key tab", "wait 0.3"]


def orbit_with_mouse(dx: int, dy: int = 0) -> list:
    x, y = max(0, min(999, 500 + dx)), max(0, min(999, 500 + dy))
    return ["move 500 500", f"middle_drag 500 500 {x} {y}"]


def box_select(x1: int, y1: int, x2: int, y2: int) -> list:
    return [f"drag {x1} {y1} {x2} {y2}"]


def sculpt_strokes(strokes: list) -> list:
    """Enter Sculpt Mode (F3 search), paint each stroke (a list of (x, y) points) with the mouse, return to Object Mode."""
    out = ["move 500 500", "key f3", "wait 0.5", 'type "sculpt mode"', "wait 0.4", "key enter", "wait 0.6"]
    for pts in strokes:
        pts = pts[:64]
        if len(pts) >= 2:
            out.append("stroke left " + " ".join(f"{int(x)} {int(y)}" for x, y in pts))
            out.append("wait 0.2")
    return out + ["key f3", "wait 0.5", 'type "object mode"', "wait 0.4", "key enter", "wait 0.5"]


MOUSE_RECIPES = {"grab": grab_with_mouse, "rotate": rotate_with_mouse, "scale": scale_with_mouse, "extrude": extrude_with_mouse,
                 "orbit": orbit_with_mouse, "select": box_select, "sculpt": sculpt_strokes}
