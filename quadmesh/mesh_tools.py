"""One front door for the shape generators: validated parameters -> Mesh(es) -> printability gate -> STL (+ Blender view)."""
from __future__ import annotations
import math
import os
from typing import Dict, Optional

import numpy as np

from quadmesh import geometry3d as G, mechanisms as M
from quadmesh.agent.printability import Limits, analyze_mesh, judge_report

# kind -> {param: (type, min, max, default)}
SCHEMAS: Dict[str, Dict[str, tuple]] = {
    "threaded_rod": {"major_dia": (float, 2, 60, 8), "pitch": (float, 0.4, 6, 1.25), "length": (float, 2, 300, 20)},
    "nut": {"major_dia": (float, 2, 60, 8), "pitch": (float, 0.4, 6, 1.25), "height": (float, 1, 60, 6.5), "clearance": (float, 0, 1, 0.3)},
    "gear": {"module": (float, 0.5, 10, 2), "teeth": (int, 8, 150, 20), "thickness": (float, 1, 60, 8), "bore": (float, 0, 100, 5)},
    "gear_pair": {"module": (float, 0.5, 10, 2), "z1": (int, 8, 100, 16), "z2": (int, 8, 150, 24), "thickness": (float, 1, 60, 8),
                  "bore1": (float, 0, 60, 5), "bore2": (float, 0, 60, 5), "backlash": (float, 0, 1, 0.2)},
    "text": {"text": (str, 1, 40, "Quadmesh"), "height_mm": (float, 3, 100, 12), "depth_mm": (float, 0.6, 30, 3)},
    "spring": {"wire_dia": (float, 0.6, 10, 2), "coil_dia": (float, 4, 100, 20), "pitch": (float, 1, 30, 6), "turns": (float, 1, 40, 6)},
    "pipe": {"outer_dia": (float, 3, 200, 20), "inner_dia": (float, 1, 190, 16), "length": (float, 2, 300, 50)},
    "vase": {"height": (float, 20, 300, 100), "base_dia": (float, 20, 200, 60), "belly_dia": (float, 20, 260, 80),
             "neck_dia": (float, 10, 200, 40), "lip_dia": (float, 10, 220, 50), "wall": (float, 1.2, 6, 2.4)},
    "transition": {"w": (float, 5, 200, 40), "d": (float, 5, 200, 20), "dia": (float, 5, 200, 25), "height": (float, 5, 200, 30)},
    "four_bar": {"ground": (float, 20, 300, 100), "crank": (float, 8, 200, 30), "coupler": (float, 20, 300, 90), "rocker": (float, 20, 300, 70),
                 "width": (float, 6, 40, 14), "thickness": (float, 2, 20, 5), "pin_dia": (float, 2, 10, 4)},
}
RELAXED = {"threaded_rod", "nut", "spring", "text", "vase", "gear", "gear_pair", "four_bar", "transition"}      # supports / thin details are normal here


def validate(kind: str, params: dict) -> dict:
    """Clamp-free validation: unknown kinds / keys / out-of-range values raise ValueError (a wrong tool call must never run)."""
    if kind not in SCHEMAS:
        raise ValueError(f"unknown shape {kind!r}; known: {sorted(SCHEMAS)}")
    out = {}
    for k, (t, lo, hi, dflt) in SCHEMAS[kind].items():
        v = params.get(k, dflt)
        if t is str:
            v = str(v)
            if not lo <= len(v) <= hi or not all(32 <= ord(c) < 127 for c in v):
                raise ValueError(f"{k}: text must be {lo}-{hi} plain characters")
        else:
            v = t(v)
            if not (math.isfinite(v) and lo <= v <= hi):
                raise ValueError(f"{k}={v} is outside the allowed range {lo}..{hi}")
        out[k] = v
    extra = set(params) - set(SCHEMAS[kind])
    if extra:
        raise ValueError(f"unknown parameters for {kind}: {sorted(extra)}")
    return out


def make_shape(kind: str, **params):
    """Returns ({name: Mesh}, notes list). Every mesh is centred on x,y and rests on z=0."""
    p = validate(kind, params)
    notes = []
    if kind == "threaded_rod":
        meshes = {"threaded_rod": G.threaded_rod(p["major_dia"], p["pitch"], p["length"])}
        notes.append(f"M{p['major_dia']:g}x{p['pitch']:g} thread; print it VERTICAL or with supports. Test-fit the nut before printing many.")
    elif kind == "nut":
        meshes = {"nut": G.nut(p["major_dia"], p["pitch"], p["height"], clearance=p["clearance"])}
        notes.append(f"{p['clearance']:g} mm thread clearance so a printed bolt can turn in it.")
    elif kind == "gear":
        meshes = {"gear": G.gear(p["module"], p["teeth"], p["thickness"], p["bore"])}
        notes.append(f"module {p['module']:g}, {p['teeth']} teeth, pitch diameter {p['module']*p['teeth']:g} mm, 20 degree pressure angle.")
    elif kind == "gear_pair":
        gp = M.gear_pair(p["module"], p["z1"], p["z2"], p["thickness"], p["bore1"], p["bore2"], p["backlash"])
        meshes = {"gear1": gp["gear1"], "gear2": gp["gear2"].translated(-gp["centre_distance"], 0, 0)}
        notes.append(f"centre distance {gp['centre_distance']:g} mm, ratio {gp['ratio']:g}:1, backlash {p['backlash']:g} mm - place the shafts exactly this far apart.")
    elif kind == "text":
        meshes = {"text": G.text3d(p["text"], p["height_mm"], p["depth_mm"])}
        notes.append("each letter is a separate solid; print it on a base plate or glue it on.")
    elif kind == "spring":
        r = p["coil_dia"] / 2
        meshes = {"spring": G.sweep(G.circle_profile(p["wire_dia"] / 2, 20), G.helix_path(r, p["pitch"], p["turns"]))}
        notes.append("a printed spring is a flexure - the material's stiffness decides how it behaves; PETG or nylon work best.")
    elif kind == "pipe":
        if p["inner_dia"] >= p["outer_dia"] - 1.6:
            raise ValueError("inner_dia must be at least 1.6 mm smaller than outer_dia (a wall of 0.8 mm each side)")
        ro, ri, L = p["outer_dia"] / 2, p["inner_dia"] / 2, p["length"]
        meshes = {"pipe": G.lathe([(ri, 0), (ro, 0), (ro, L), (ri, L)], 96)}
    elif kind == "vase":
        H, w = p["height"], p["wall"]
        hs = np.array([0, 0.35, 0.75, 1.0]) * H
        rs = np.array([p["base_dia"], p["belly_dia"], p["neck_dia"], p["lip_dia"]]) / 2
        zz = np.linspace(0, H, 30)
        r = np.interp(zz, hs, rs); r = np.convolve(np.pad(r, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        floor = w * 1.2
        outer = [(0.0, 0.0)] + list(zip(r, zz))
        inner = [(max(ri - w, 0.4), zi) for ri, zi in zip(r[::-1], zz[::-1]) if zi >= floor]
        meshes = {"vase": G.lathe(outer + inner + [(0.0, floor)], 96)}
        notes.append("print in vase mode / spiral or with 2+ walls; smooth curves print best.")
    elif kind == "transition":
        meshes = {"transition": G.loft([G.rect_ring(p["w"], p["d"], 0, 64, 0.35), G.rect_ring(p["w"] * .7 + p["dia"] * .3, p["d"] * .7 + p["dia"] * .3, p["height"] / 2, 64, 0.7),
                                        G.circle_ring(p["dia"] / 2, p["height"], 64)])}
    elif kind == "four_bar":
        fb = M.four_bar(p["ground"], p["crank"], p["coupler"], p["rocker"], p["width"], p["thickness"], p["pin_dia"])
        meshes = dict(fb["parts"]); a = fb["analysis"]
        notes += [f"{a['type']}; crank turns fully: {a['crank_turns_fully']}; rocker sweep {a['rocker_sweep_deg']:.0f} deg; min transmission angle {a['min_transmission_angle_deg']:.0f} deg", fb["note"]]
        if a["warning"]:
            notes.append("WARNING: " + a["warning"])
    return {k: m.on_bed() for k, m in meshes.items()}, notes


def finish(name: str, meshes: Dict[str, "G.Mesh"], out_dir: str = "designs", limits: Optional[Limits] = None, kind: str = "") -> dict:
    """Write one STL per mesh and run the printability gate on each. ok only if every mesh passes."""
    os.makedirs(out_dir, exist_ok=True)
    lim = limits or Limits(allow_supports=kind in RELAXED, min_wall=0.0 if kind in ("threaded_rod", "nut", "spring") else 0.8,
                           expected_islands=None if kind in ("text", "gear_pair", "four_bar") else 1)
    res = {"ok": True, "files": [], "problems": [], "warnings": [], "reports": {}}
    for k, m in meshes.items():
        path = G.write_stl(os.path.join(out_dir, f"{name}_{k}.stl" if len(meshes) > 1 else f"{name}.stl"), m)
        rep = analyze_mesh(m.triangles(), samples=80)
        pr, wa = judge_report(rep, None, lim)
        res["files"].append(path); res["reports"][k] = rep
        res["problems"] += [f"{k}: {x}" for x in pr]; res["warnings"] += [f"{k}: {x}" for x in wa]
    res["ok"] = not res["problems"]
    return res
