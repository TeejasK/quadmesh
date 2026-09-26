"""
Moving mechanisms: printable parts + the motion analysis that tells you whether they actually move.

  gear_pair(module, z1, z2, thickness)          two meshing spur gears (centre distance, ratio, tooth phase, backlash)
  four_bar(ground, crank, coupler, rocker)      four-bar linkage: Grashof type, crank/rocker sweep, worst transmission angle
Links are stadium-shaped bars with pin holes (clearance for a 3-D printed pin); joints are returned as data
({name, a, b, axis, pivot, range}) so the assembly can be built in Blender / URDF.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple

import numpy as np

from quadmesh import geometry3d as G


def gear_pair(module: float = 2.0, z1: int = 16, z2: int = 24, thickness: float = 8.0, bore1: float = 5.0, bore2: float = 5.0,
              backlash: float = 0.2) -> Dict:
    a = module * (z1 + z2) / 2.0
    g1 = G.gear(module, z1, thickness, bore1, backlash=backlash)
    g2 = G.gear(module, z2, thickness, bore2, backlash=backlash)
    phi = math.pi - math.pi / z2                                   # a gap of gear 2 faces the tooth of gear 1
    c, s = math.cos(phi), math.sin(phi)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    g2 = G.Mesh(g2.V @ R.T + np.array([a, 0, 0]), g2.F)
    return {"gear1": g1, "gear2": g2, "centre_distance": a, "ratio": z2 / z1, "backlash": backlash,
            "joints": [{"name": "input", "part": "gear1", "axis": "z", "pivot": [0, 0, 0], "range_deg": [-360, 360]},
                       {"name": "output", "part": "gear2", "axis": "z", "pivot": [a, 0, 0], "range_deg": [-360, 360],
                        "ratio": -z1 / z2}]}


def stadium(length: float, width: float, n: int = 20) -> List[Tuple[float, float]]:
    r = width / 2.0
    right = [(length + r * math.cos(t), r * math.sin(t)) for t in np.linspace(-math.pi / 2, math.pi / 2, n)]
    left = [(r * math.cos(t), r * math.sin(t)) for t in np.linspace(math.pi / 2, 3 * math.pi / 2, n)]
    return right + left


def link(length: float, width: float, thickness: float, pin_dia: float, clearance: float = 0.3) -> G.Mesh:
    hole = lambda cx: [(cx + (pin_dia + clearance) / 2 * math.cos(2 * math.pi * k / 40), (pin_dia + clearance) / 2 * math.sin(2 * math.pi * k / 40)) for k in range(40)]
    return G.extrude_polygon(stadium(length, width), [hole(0.0), hole(length)], thickness)


def _circle_hit(p0, r0, p1, r1):
    d = math.dist(p0, p1)
    if d > r0 + r1 or d < abs(r0 - r1) or d == 0:
        return None
    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h = math.sqrt(max(r0 * r0 - a * a, 0.0))
    xm, ym = p0[0] + a * (p1[0] - p0[0]) / d, p0[1] + a * (p1[1] - p0[1]) / d
    return ((xm + h * (p1[1] - p0[1]) / d, ym - h * (p1[0] - p0[0]) / d), (xm - h * (p1[1] - p0[1]) / d, ym + h * (p1[0] - p0[0]) / d))


def four_bar_analysis(g: float, a: float, b: float, c: float, steps: int = 720) -> Dict:
    """Ground pivots O2=(0,0), O4=(g,0); crank a from O2, coupler b, rocker c from O4. Input angle sweeps 0..360."""
    s, l = min(g, a, b, c), max(g, a, b, c)
    p, q = sorted([g, a, b, c])[1:3]
    grashof = s + l <= p + q
    shortest = [n for n, v in (("ground", g), ("crank", a), ("coupler", b), ("rocker", c)) if v == s][0]
    kind = "non-Grashof (double rocker)"
    if grashof:
        kind = {"crank": "crank-rocker", "rocker": "crank-rocker (drive the ROCKER side: it swings, the short link turns)",
                "ground": "double-crank (drag-link)", "coupler": "double-rocker (Grashof)"}[shortest]
    ok, rocker, trans, reachable = [], [], [], 0
    prev = None
    for k in range(steps):
        th = 2 * math.pi * k / steps
        A = (a * math.cos(th), a * math.sin(th))
        hit = _circle_hit(A, b, (g, 0.0), c)
        if hit is None:
            continue
        reachable += 1
        Bp = max(hit, key=lambda P: P[1]) if prev is None else min(hit, key=lambda P: math.dist(P, prev))
        prev = Bp
        rocker.append(math.atan2(Bp[1], Bp[0] - g))
        v1, v2 = (Bp[0] - A[0], Bp[1] - A[1]), (Bp[0] - g, Bp[1])
        cosm = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
        trans.append(math.degrees(math.acos(max(-1, min(1, cosm)))))
    rocker = list(np.degrees(np.unwrap(rocker))) if rocker else []
    full = reachable == steps
    return {"grashof": grashof, "type": kind, "crank_turns_fully": full, "reachable_fraction": reachable / steps,
            "rocker_sweep_deg": (max(rocker) - min(rocker)) if rocker else 0.0,
            "min_transmission_angle_deg": min(min(t, 180 - t) for t in trans) if trans else 0.0,
            "warning": None if (trans and min(min(t, 180 - t) for t in trans) >= 30) else
            "transmission angle drops below 30 degrees: the linkage will bind or need a lot of force"}


def four_bar(ground: float, crank: float, coupler: float, rocker: float, width: float = 14.0, thickness: float = 5.0,
             pin_dia: float = 4.0) -> Dict:
    an = four_bar_analysis(ground, crank, coupler, rocker)
    parts = {"crank": link(crank, width, thickness, pin_dia), "coupler": link(coupler, width, thickness, pin_dia),
             "rocker": link(rocker, width, thickness, pin_dia), "ground_link": link(ground, width, thickness, pin_dia)}
    joints = [{"name": "O2", "a": "ground_link", "b": "crank", "axis": "z", "pivot": [0, 0, 0]},
              {"name": "A", "a": "crank", "b": "coupler", "axis": "z", "pivot": "at crank tip"},
              {"name": "B", "a": "coupler", "b": "rocker", "axis": "z", "pivot": "at rocker tip"},
              {"name": "O4", "a": "ground_link", "b": "rocker", "axis": "z", "pivot": [ground, 0, 0]}]
    return {"parts": parts, "joints": joints, "analysis": an, "pin_dia": pin_dia,
            "note": f"print {pin_dia} mm pins (or use {pin_dia} mm rod); holes are {pin_dia + 0.3:g} mm for a sliding fit"}
