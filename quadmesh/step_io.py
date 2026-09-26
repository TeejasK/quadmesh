"""
STEP / B-Rep support through CadQuery (OpenCascade) - a real CAD kernel.

  plan_to_step(plan, "part.step")   run a verified Quadmesh plan in the B-Rep kernel and write STEP (also STL): the same
                                    design, but as exact B-Rep geometry for SolidWorks / Fusion / FreeCAD / a machine shop
  step_to_mesh("part.step")         read ANY STEP file, tessellate it, return a Mesh + facts (volume, size, faces) so it
                                    can go through the printability gate and into Blender as STL
Install:  pip install cadquery      (large; not needed for anything else)
Fillets/chamfers on complex edges can fail in any B-Rep kernel: those steps are skipped and reported, never hidden.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from quadmesh.pipeline.plan_checker import _anchors, CIRCLE_OPS, BOSS_OPS


def _cq():
    try:
        import cadquery as cq
        return cq
    except ImportError as e:
        raise ImportError("STEP support needs CadQuery: pip install cadquery") from e


def _bb(obj):
    b = obj.val().BoundingBox()
    return (b.xmin, b.xmax, b.ymin, b.ymax, b.zmin, b.zmax)


def plan_to_solids(plan: List[dict]) -> Tuple[Dict[str, object], List[str]]:
    """Execute the plan in OpenCascade. Returns ({name: Workplane}, warnings)."""
    cq = _cq()
    V = cq.Vector
    parts: Dict[str, object] = {}
    warn: List[str] = []

    def cyl(x, y, z0, z1, dia):
        return cq.Workplane("XY").workplane(offset=z0).center(x, y).circle(dia / 2.0).extrude(z1 - z0)

    def box(cx, cy, z0, w, d, h):
        return cq.Workplane("XY").box(w, d, h, centered=(True, True, False)).translate((cx, cy, z0))

    def cut(p, tool):
        return p.cut(tool)

    for i, s in enumerate(plan):
        op, a = s["op"], s.get("args", {})
        if op.startswith("export_"):
            continue
        if op == "add_box":
            parts[a["name"]] = cq.Workplane("XY").box(a["w"], a["d"], a["h"], centered=(True, True, False))
        elif op == "add_cyl":
            parts[a["name"]] = cq.Workplane("XY").circle(a["dia"] / 2.0).extrude(a["h"])
        elif op == "add_cone":
            parts[a["name"]] = cq.Workplane("XY").add(cq.Solid.makeCone(a["dia1"] / 2.0, a["dia2"] / 2.0, a["h"]))
        elif op == "add_prism":
            parts[a["name"]] = cq.Workplane("XY").polygon(int(a["sides"]), a["dia"]).extrude(a["h"])
        elif op == "add_sphere":
            parts[a["name"]] = cq.Workplane("XY").add(cq.Solid.makeSphere(a["dia"] / 2.0, angleDegrees1=-90)).translate((0, 0, a["dia"] / 2.0))
        elif op == "add_torus":
            parts[a["name"]] = cq.Workplane("XY").add(cq.Solid.makeTorus(a["major"] / 2.0, a["minor"] / 2.0)).translate((0, 0, a["minor"] / 2.0))
        elif op == "move":
            p = parts[a["name"]]; b = _bb(p)
            parts[a["name"]] = p.translate((a["x"] - (b[0] + b[1]) / 2, a["y"] - (b[2] + b[3]) / 2, a["z"] - b[4]))
        elif op == "rotate":
            p = parts[a["name"]]; b = _bb(p)
            c = V((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
            axis = {"x": V(1, 0, 0), "y": V(0, 1, 0), "z": V(0, 0, 1)}[a["axis"]]
            parts[a["name"]] = p.rotate(c, c + axis, a["deg"])
        elif op in ("stack_on", "align_back", "align_side"):
            p, r = parts[a["name"]], parts[a["ref"]]; pb, rb = _bb(p), _bb(r)
            if op == "stack_on":
                d = ((rb[0] + rb[1]) / 2 - (pb[0] + pb[1]) / 2, (rb[2] + rb[3]) / 2 - (pb[2] + pb[3]) / 2, rb[5] - pb[4])
            else:
                side = "back" if op == "align_back" else a["side"]
                d = {"back": (0, rb[3] - pb[3], 0), "front": (0, rb[2] - pb[2], 0), "right": (rb[1] - pb[1], 0, 0),
                     "left": (rb[0] - pb[0], 0, 0)}[side]
            parts[a["name"]] = p.translate(d)
        elif op == "union":
            parts[a["target"]] = parts[a["target"]].union(parts.pop(a["tool"]))
        elif op == "cut":
            parts[a["target"]] = parts[a["target"]].cut(parts.pop(a["tool"]))
        elif op in CIRCLE_OPS:
            p = parts[a["target"]]; b = _bb(p); anchor, style = CIRCLE_OPS[op]
            pts, _ = _anchors(anchor, a, b)
            for (x, y) in pts:
                if style in ("plain", "cbore", "csink"):
                    p = p.cut(cyl(x, y, b[4] - 1, b[5] + 1, a["dia"]))
                if style == "cbore":
                    p = p.cut(cyl(x, y, b[5] - a["cb_depth"], b[5] + 1, a["cb_dia"]))
                elif style == "csink":
                    h = (a["cs_dia"] - a["dia"]) / 2.0
                    p = p.cut(cq.Workplane("XY").add(cq.Solid.makeCone(a["dia"] / 2.0, a["cs_dia"] / 2.0 + 0.5, h + 0.5, V(x, y, b[5] - h), V(0, 0, 1))))
                elif style == "pocket":
                    p = p.cut(cyl(x, y, b[5] - a["depth"], b[5] + 1, a["dia"]))
            parts[a["target"]] = p
        elif op in BOSS_OPS:
            p = parts[a["target"]]; b = _bb(p)
            pts, _ = _anchors(BOSS_OPS[op], a, b)
            for (x, y) in pts:
                p = p.union(cyl(x, y, b[5] - 0.01, b[5] + a["h"], a["dia"]))
            parts[a["target"]] = p
        elif op == "hole_at":
            p = parts[a["target"]]; b = _bb(p)
            parts[a["target"]] = p.cut(cyl((b[0] + b[1]) / 2 + a["x"], (b[2] + b[3]) / 2 + a["y"], b[4] - 1, b[5] + 1, a["dia"]))
        elif op == "window_at":
            p = parts[a["target"]]; b = _bb(p)
            parts[a["target"]] = p.cut(box((b[0] + b[1]) / 2 + a["x"], (b[2] + b[3]) / 2 + a["y"], b[4] - 1, a["w"], a["d"], b[5] - b[4] + 2))
        elif op == "slot_center":
            p = parts[a["target"]]; b = _bb(p)
            slot = cq.Workplane("XY").workplane(offset=b[4] - 1).center((b[0] + b[1]) / 2, (b[2] + b[3]) / 2).slot2D(a["length"], a["width"]).extrude(b[5] - b[4] + 2)
            parts[a["target"]] = p.cut(slot)
        elif op in ("pocket_rect", "window_rect"):
            p = parts[a["target"]]; b = _bb(p); cx, cy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
            z0, hh = (b[5] - a["depth"], a["depth"] + 1) if op == "pocket_rect" else (b[4] - 1, b[5] - b[4] + 2)
            parts[a["target"]] = p.cut(box(cx, cy, z0, a["w"], a["d"], hh))
        elif op == "shell_open_top":
            p = parts[a["target"]]; b = _bb(p); t = a["wall"]
            parts[a["target"]] = p.cut(box((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, b[4] + t, b[1] - b[0] - 2 * t, b[3] - b[2] - 2 * t, b[5] - b[4] - t + 1))
        elif op in ("array_x", "array_y"):
            p = parts[a["name"]]; q = p
            for k in range(1, int(a["count"])):
                q = q.union(p.translate(((k * a["pitch"]) if op == "array_x" else 0, (k * a["pitch"]) if op == "array_y" else 0, 0)))
            parts[a["name"]] = q
        elif op == "array_polar":
            p = parts[a["name"]]; q = p
            for k in range(1, int(a["count"])):
                q = q.union(p.rotate(V(0, 0, 0), V(0, 0, 1), 360.0 * k / int(a["count"])))
            parts[a["name"]] = q
        elif op in ("fillet", "chamfer"):
            p = parts[a["target"]]
            try:
                q = p.edges().fillet(a["r"]) if op == "fillet" else p.edges().chamfer(a["size"])
                if q.val().isValid() and q.val().Volume() > 0:
                    parts[a["target"]] = q
                else:
                    warn.append(f"step {i+1}: {op} gave an invalid solid, skipped")
            except Exception as e:
                warn.append(f"step {i+1}: {op} skipped by the B-Rep kernel ({type(e).__name__})")
        elif op == "extrude_polygons":
            wp = cq.Workplane("XY").polyline([tuple(p) for p in a["outer"]]).close().extrude(a["h"])
            for h in a.get("holes", []):
                wp = wp.cut(cq.Workplane("XY").polyline([tuple(p) for p in h]).close().extrude(a["h"]))
            parts[a["name"]] = wp
        else:
            warn.append(f"step {i+1}: op {op!r} has no B-Rep translation (skipped)")
    return parts, warn


def plan_to_step(plan: List[dict], path: str, also_stl: bool = True):
    """Write STEP (and optionally STL) for the final single solid of a plan. Returns (paths, warnings, solid)."""
    cq = _cq()
    parts, warn = plan_to_solids(plan)
    if len(parts) != 1:
        raise ValueError(f"expected one final part, found {sorted(parts)}")
    wp = next(iter(parts.values()))
    cq.exporters.export(wp, path)
    out = [path]
    if also_stl:
        stl = path.rsplit(".", 1)[0] + ".stl"
        cq.exporters.export(wp, stl, tolerance=0.02, angularTolerance=0.1)
        out.append(stl)
    return out, warn, wp


def step_to_mesh(path: str, tolerance: float = 0.02):
    """Read a STEP file -> (Mesh, facts). Several bodies are merged."""
    cq = _cq()
    from quadmesh.geometry3d import Mesh
    shape = cq.importers.importStep(path)
    solids = shape.solids().vals()
    Vs, Fs, off = [], [], 0
    for s in (solids or [shape.val()]):
        vs, tris = s.tessellate(tolerance, 0.1)
        Vs.append(np.array([[v.x, v.y, v.z] for v in vs], float)); Fs.append(np.array(tris, int) + off); off += len(vs)
    mesh = Mesh(np.vstack(Vs), np.vstack(Fs))
    b = shape.val().BoundingBox()
    facts = {"solids": len(solids), "faces": len(shape.faces().vals()), "volume_mm3": round(sum(s.Volume() for s in solids), 3),
             "size_mm": [round(b.xlen, 3), round(b.ylen, 3), round(b.zlen, 3)]}
    return mesh, facts
