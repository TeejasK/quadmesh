"""
Parametric assemblies: from a few design PARAMETERS to a list of printable parts, each with a verified plan.

Why parameters and not a text model doing the maths: the language model only has to READ the request
("1.5 kg hexacopter, industrial grade") and extract numbers/choices; the sizes come from hand-calculation
code (strength.py) that is exact and inspectable. Every part plan then goes through plan_checker and the
printability gate like any other design.

    asm = build_assembly({"kind": "hexacopter", "mass_kg": 1.5, "grade": "industrial", "material": "petg"})
    asm.report   # the calculations
    asm.parts    # [Part(name, qty, plan)]
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from quadmesh.strength import (MATERIALS, GRADES, G, allowable, bending_stress, min_height, min_pin_diameter,
                               min_plate_thickness_pullthrough, line)

BED_DEFAULT = (220.0, 220.0, 250.0)


def S(op, **a):
    return {"op": op, "args": a}


@dataclass
class Part:
    name: str
    qty: int
    plan: list
    note: str = ""


@dataclass
class Assembly:
    kind: str
    parts: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    report: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    spec: dict = field(default_factory=dict)


def _clean(x):
    if isinstance(x, float):
        x = round(x, 2)
        return int(x) if x == int(x) else x
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    return x


def _material(spec):
    m = str(spec.get("material", "petg")).lower().replace("-", "").replace(" ", "")
    m = {"nylon12": "nylon", "pa12": "nylon", "pacf": "pa12cf", "nylonCF": "pa12cf", "carbonnylon": "pa12cf"}.get(m, m)
    return m if m in MATERIALS else "petg"


# ---------------------------------------------------------------------------------------------- hexacopter
def hexacopter(spec: dict) -> Assembly:
    mass = float(spec.get("mass_kg", 1.5))
    grade = spec.get("grade", "industrial") if spec.get("grade") in GRADES else "industrial"
    mat = _material(spec)
    wb = float(spec.get("wheelbase_mm", 450 if mass <= 2.5 else 550))          # motor centre to opposite motor centre
    R = wb / 2.0
    sig = allowable(mat, grade)
    A = Assembly("hexacopter", spec=dict(spec))

    hub_d = round(0.36 * wb / 2) * 2
    hub_t = 6 if grade == "hobby" else 8
    r_hub = hub_d / 2.0
    clamp_r = r_hub - 14                                       # arm clamp bolt circle radius
    arm_w = 26.0                                               # fits the 16x16 motor pattern
    r_in = clamp_r - 14                                        # arm starts here (14 mm behind the bolt)
    r_out = R + 16                                             # 16 mm beyond the motor centre
    La = r_out - r_in

    T_max = 2.0 * mass * G / 6.0                               # 2:1 thrust-to-weight, per motor
    dyn = 1.5 if grade != "hobby" else 1.2                     # gusts / hard landings
    F = T_max * dyn
    Lever = R - clamp_r
    h = min_height(F, Lever, arm_w, sig, floor=6.0 if grade == "hobby" else 8.0)
    stress = bending_stress(F, Lever, arm_w, h)
    A.report += [f"hexacopter: {mass:g} kg, wheelbase {wb:g} mm, material {mat} ({MATERIALS[mat]['note']}), grade {grade}",
                 f"design load per arm: {F:.1f} N (max thrust {T_max:.1f} N x dynamic factor {dyn:g}) at a lever of {Lever:.0f} mm",
                 line("arm bending", F, "N", stress, sig, GRADES[grade], MATERIALS[mat]["sigma"]),
                 f"arm section {arm_w:g} x {h:g} mm ({'on edge, load along the height'}), hub plates {hub_d:g} mm x {hub_t:g} mm"]

    # ---- arm: bar with 1 clamp hole and the 16x16 motor pattern + shaft hole, holes placed by absolute position
    cx = (r_in + r_out) / 2.0                                  # bar centre in radial coordinate
    rel = lambda x_abs: x_abs - cx
    x_motor = R
    arm = [S("add_box", name="body", w=La, d=arm_w, h=h),
           S("hole_at", target="body", x=rel(clamp_r), y=0, dia=4.5),
           S("hole_at", target="body", x=rel(x_motor), y=0, dia=8)]
    for sx in (-8, 8):
        for sy in (-8, 8):
            arm.append(S("hole_at", target="body", x=rel(x_motor + sx), y=sy, dia=3.2))
    A.parts.append(Part("arm", 6, arm, "motor end has 4x M3 (16x16) + 8 mm shaft hole; one M4 clamp bolt at the hub"))

    # ---- hub plates: centre hole, 6 clamp bolts on the clamp circle (aligned with the arms), FC pattern on top
    hub_common = [S("add_cyl", name="body", dia=hub_d, h=hub_t), S("hole_center", target="body", dia=32)]
    bottom = hub_common + [S("bolt_circle", target="body", pcd=2 * clamp_r, n=6, dia=4.5)]
    top = list(bottom)
    for sx in (-15.25, 15.25):                                  # 30.5 x 30.5 flight-controller pattern
        for sy in (-15.25, 15.25):
            top.append(S("hole_at", target="body", x=sx, y=sy, dia=3.2))
    A.parts.append(Part("hub_plate_bottom", 1, bottom, "arms are clamped between the two plates"))
    A.parts.append(Part("hub_plate_top", 1, top, "flight controller mounts here"))
    A.report.append("NOT checked: stiffness, vibration/resonance, fatigue, creep, heat, impact - check these separately")
    A.report.append("BOM extras (not printed): 6x M4 bolts + nuts for the clamps, 24x M3 for motors, 6 motors, 6 ESCs, "
                    "battery, landing gear")
    return A


# ---------------------------------------------------------------------------------------------- robot arm
def robot_arm(spec: dict) -> Assembly:
    reach = float(spec.get("reach_mm", 400))
    payload = float(spec.get("payload_kg", 0.5))
    n_links = int(spec.get("links", 2)) if int(spec.get("links", 2)) in (2, 3) else 2
    grade = spec.get("grade", "industrial") if spec.get("grade") in GRADES else "industrial"
    mat = _material(spec)
    sig = allowable(mat, grade)
    A = Assembly("robot_arm", spec=dict(spec))
    dyn = 1.5 if grade != "hobby" else 1.25
    link_len = reach / n_links
    b = max(24.0, round(0.07 * reach / 2) * 2)                  # link width
    A.report.append(f"robot arm: reach {reach:g} mm, payload {payload:g} kg, {n_links} links of {link_len:g} mm, "
                    f"material {mat}, grade {grade}")

    # links: size from bending, loaded from the tip inwards (payload + own weight of the links beyond)
    links, hs, W_beyond = [], [], 0.0
    for i in range(n_links, 0, -1):                             # tip link first
        Fi = (payload * G + W_beyond) * dyn
        pin_d = min_pin_diameter(Fi, sig)
        hi = min_height(Fi, link_len, b, sig, floor=8.0)
        w_link = (link_len * b * hi) * MATERIALS[mat]["rho"] / 1e6 * G      # weight in N (mm^3 x g/cm^3 -> kg)
        A.report.append(line(f"link {i} bending", Fi, "N", bending_stress(Fi, link_len, b, hi), sig, GRADES[grade],
                             MATERIALS[mat]["sigma"]) + f"; pin dia {pin_d:g} mm (double shear)")
        links.append((i, hi, pin_d)); W_beyond += w_link
    # one shared link size = the largest needed (parts are identical, easier to print and stock)
    h_link = max(x[1] for x in links)
    pin = max(x[2] for x in links)
    pin_hole = pin + 0.4                                        # clearance for the pin
    inset = round(pin_hole * 1.2 + 3, 1)
    link = [S("add_box", name="body", w=link_len + 2 * inset, d=b, h=h_link),
            S("holes_pair_x", target="body", inset=inset, dia=pin_hole)]
    A.parts.append(Part("link", n_links, link, f"pin holes {pin_hole:g} mm, pins {pin:g} mm steel or PETG rod"))

    # stand: column + base plate, sized for the overturning moment at the base
    M = ((payload * G) * dyn + W_beyond) * reach                 # N*mm about the base
    base_w = min(max(160.0, round(0.6 * reach / 10) * 10), float(spec.get("_bed_min", 1e9)))   # never wider than the printer bed
    a = base_w / 2 - 15
    T_bolt = M / (2 * a) / 2                                     # tension per bolt on the tension side
    t_base = min_plate_thickness_pullthrough(T_bolt, washer_d=16, sigma_allow=sig)
    col_d = max(40.0, math.ceil((32 * M / (math.pi * sig)) ** (1 / 3) / 2) * 2)     # solid round section in bending
    col_h = max(80.0, round(0.3 * reach / 10) * 10)
    A.report += [f"base moment {M/1000:.1f} N*m; tension per base bolt {T_bolt:.0f} N (use M6 bolts with 16 mm washers, "
                 f"steel plate underneath if the surface is soft)",
                 f"base plate {base_w:g} x {base_w:g} x {t_base:g} mm (pull-through), column {col_d:g} mm x {col_h:g} mm"]
    A.parts.append(Part("base_plate", 1, [S("add_box", name="body", w=base_w, d=base_w, h=t_base),
                                          S("holes_corners", target="body", inset=15, dia=6.5),
                                          S("hole_center", target="body", dia=min(30.0, base_w / 4))],
                        "screw this to your bench"))
    A.parts.append(Part("column", 1, [S("add_cyl", name="body", dia=col_d, h=col_h),
                                      S("hole_center", target="body", dia=12)], "cable channel in the middle"))
    # joint metadata for later (URDF-style): pivots in assembly coordinates, z up
    z0 = t_base + col_h
    A.joints = [{"name": "shoulder", "a": "column", "b": "link1", "axis": "y", "pivot": [0, 0, z0], "range_deg": [-90, 90]}]
    x = 0.0
    for i in range(1, n_links):
        x += link_len
        A.joints.append({"name": f"joint{i+1}", "a": f"link{i}", "b": f"link{i+1}", "axis": "y",
                         "pivot": [x, 0, z0], "range_deg": [-120, 120]})
    A.report.append("NOT checked: stiffness/deflection, bearing wear, fatigue, creep, heat - check these separately")
    A.report.append("BOM extras: servos/motors sized for the shoulder torque "
                    f"{(payload*G*dyn + W_beyond) * reach / 1000:.1f} N*m at the first joint, pins, bearings, bolts")
    return A


BUILDERS = {"hexacopter": hexacopter, "robot_arm": robot_arm}


def build_assembly(spec: dict, bed=BED_DEFAULT) -> Optional[Assembly]:
    fn = BUILDERS.get(spec.get("kind"))
    if fn is None:
        return None
    A = fn(dict(spec, _bed_min=float(min(bed[0], bed[1])) - 10.0))
    A.spec = dict(spec)
    for p in A.parts:
        p.plan = _clean(p.plan)
    for p in A.parts:                                           # does each printable part fit the bed?
        from quadmesh.pipeline.plan_checker import predicted_bbox
        bb = predicted_bbox(p.plan)
        if bb is None:
            A.warnings.append(f"part {p.name}: plan is invalid")
            continue
        d = sorted([bb[1]-bb[0], bb[3]-bb[2]], reverse=True)
        if d[0] > max(bed[0], bed[1]) or d[1] > min(bed[0], bed[1]) or bb[5]-bb[4] > bed[2]:
            A.warnings.append(f"part {p.name} is {bb[1]-bb[0]:.0f} x {bb[3]-bb[2]:.0f} x {bb[5]-bb[4]:.0f} mm and does "
                              f"not fit the {bed[0]:.0f} x {bed[1]:.0f} x {bed[2]:.0f} mm bed - split it or use a bigger printer")
    return A


# ---------------------------------------------------------------------------
# Item 10: bootstrap labels for a learned DECOMPOSITION model - deliberately scoped narrower than "learn the
# whole thing", for a reason worth stating plainly rather than glossing over.
#
# Two very different things happen inside hexacopter()/robot_arm() above:
#   (a) STRUCTURE - how many parts, what they're called, how they connect (the joints list), which features
#       each part needs (holes, bolt circles, pockets). This is a genuinely learnable pattern: "a hexacopter
#       has 6 identical arms bolted to 2 hub plates" is a shape of an answer, not a safety-critical number.
#   (b) SIZING - arm cross-section height from bending_stress(), bolt/pin diameters from allowable() and
#       min_pin_diameter(), base-plate thickness from pull-through math. This is real structural engineering,
#       computed from material yield strength and load factors in strength.py. A model predicting these
#       numbers by pattern-matching training examples, instead of computing them from the actual load in the
#       actual request, is the same category of regression flagged for printability's hard invariants: a
#       confidently-wrong learned guess at a load-bearing dimension is not a worse UX, it's a part that
#       breaks under load with no warning that anything was uncertain.
#
# So this export function only lets (a) become training data. `decomposition_example()` strips every
# computed dimension out of a real build_assembly() result and keeps only the shape of the decomposition -
# part names, quantities, feature types (not their numbers), and the joint graph. A future learned
# decomposition model's job is: given a spec, propose THIS SHAPE for kinds strength.py doesn't hand-code yet -
# and its output must still be re-sized by calling strength.py-style hand-calculation for the specific
# numbers in the request, never by recalling a memorized dimension from training. That hand-off point (learned
# structure -> hand-calculated sizing) is intentional, not a gap to be closed later.
# ---------------------------------------------------------------------------

def _feature_shape(plan: list) -> list:
    """A part's plan with every numeric arg dropped, keeping only which ops were used and in what order -
    the thing that's actually a reusable PATTERN across different sizes of the same part."""
    return [step["op"] for step in plan]


def decomposition_example(spec: dict, bed=BED_DEFAULT) -> Optional[dict]:
    """One (request, decomposition-shape) pair for training a structure-proposing model. Returns None if the
    spec doesn't build (same rule as build_assembly). Every number in the output is a COUNT (qty, n_joints),
    never a computed dimension - see the module note above for why that line is intentional."""
    A = build_assembly(spec, bed)
    if A is None:
        return None
    return {
        "spec": dict(spec),
        "parts": [{"name": p.name, "qty": p.qty, "features": _feature_shape(p.plan)} for p in A.parts],
        "joints": [{"name": j["name"], "a": j["a"], "b": j["b"], "axis": j["axis"]} for j in A.joints],
    }


def write_decomposition_dataset(path: str, n: int = 4000, seed: int = 0):
    """Sweeps both existing builders across their spec space and writes (request-shape, decomposition) pairs.
    Bootstrap-only: this teaches a model hexacopter/robot_arm's existing structure, not new assembly kinds -
    extending BUILDERS with a third hand-written family is what actually grows the set this can learn from."""
    import json
    import os
    import random
    random.seed(seed)
    kinds = list(BUILDERS)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    written = 0
    with open(path, "w") as fh:
        while written < n:
            kind = random.choice(kinds)
            if kind == "hexacopter":
                spec = {"kind": kind, "mass_kg": round(random.uniform(0.4, 6), 1),
                        "grade": random.choice(list(GRADES)), "material": random.choice(list(MATERIALS))}
            else:
                spec = {"kind": kind, "reach_mm": random.choice([200, 300, 400, 500, 650, 800]),
                        "payload_kg": round(random.uniform(0.1, 3), 1), "links": random.choice([2, 3]),
                        "grade": random.choice(list(GRADES)), "material": random.choice(list(MATERIALS))}
            ex = decomposition_example(spec)
            if ex is None:
                continue
            fh.write(json.dumps(ex) + "\n")
            written += 1
    print(f"wrote {written} decomposition examples to {path}")
