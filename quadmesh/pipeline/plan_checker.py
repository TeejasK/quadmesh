"""
Plan checker + geometry simulator for Quadmesh feature-op plans (v2).

Pure Python (no Blender, no torch). It decides whether a plan is geometrically
sane and 3D-printable-by-rule BEFORE anything is sent to Blender, and it
predicts the final bounding box so the real Blender result can be compared.

Conventions (identical on the Blender side):
  * 1 Blender unit = 1 mm.  Parts are centred on x=0,y=0 with the bottom at z=0.
  * Holes / pockets / slots / bosses work on the TOP face, straight along Z.
  * Blind features (pockets, counterbores, countersinks) must be made BEFORE
    bosses or unions, while the part is still one plain solid.
"""
from __future__ import annotations
import json
import math
import re
from typing import Optional

_C4 = {"target", "dia"}
OPS = {
    "add_box": {"name", "w", "d", "h"},
    "add_cyl": {"name", "dia", "h"},
    "add_cone": {"name", "dia1", "dia2", "h"},
    "add_sphere": {"name", "dia"},
    "add_prism": {"name", "sides", "dia", "h"},
    "add_torus": {"name", "major", "minor"},
    "move": {"name", "x", "y", "z"},
    "rotate": {"name", "axis", "deg"},
    "array_x": {"name", "count", "pitch"},
    "array_y": {"name", "count", "pitch"},
    "array_polar": {"name", "count"},
    "hole_at": {"target", "x", "y", "dia"},
    "window_at": {"target", "x", "y", "w", "d"},
    "align_back": {"name", "ref"},
    "align_side": {"name", "ref", "side"},
    "stack_on": {"name", "ref"},
    "union": {"target", "tool"},
    "cut": {"target", "tool"},
    "hole_center": {"target", "dia"},
    "holes_pair_x": {"target", "inset", "dia"},
    "holes_corners": {"target", "inset", "dia"},
    "holes_grid": {"target", "nx", "ny", "pitch", "dia"},
    "holes_pair_y": {"target", "inset", "dia"},
    "holes_row_x": {"target", "n", "pitch", "dia"},
    "holes_row_y": {"target", "n", "pitch", "dia"},
    "cbore_pair_y": {"target", "inset", "dia", "cb_dia", "cb_depth"},
    "csink_pair_y": {"target", "inset", "dia", "cs_dia"},
    "bosses_pair_y": {"target", "inset", "dia", "h"},
    "bolt_circle": {"target", "pcd", "n", "dia"},
    "cbore_center": {"target", "dia", "cb_dia", "cb_depth"},
    "cbore_pair_x": {"target", "inset", "dia", "cb_dia", "cb_depth"},
    "cbore_corners": {"target", "inset", "dia", "cb_dia", "cb_depth"},
    "cbore_bolt_circle": {"target", "pcd", "n", "dia", "cb_dia", "cb_depth"},
    "csink_center": {"target", "dia", "cs_dia"},
    "csink_pair_x": {"target", "inset", "dia", "cs_dia"},
    "csink_corners": {"target", "inset", "dia", "cs_dia"},
    "csink_bolt_circle": {"target", "pcd", "n", "dia", "cs_dia"},
    "pocket_round": {"target", "dia", "depth"},
    "slot_center": {"target", "length", "width"},
    "pocket_rect": {"target", "w", "d", "depth"},
    "window_rect": {"target", "w", "d"},
    "boss_center": {"target", "dia", "h"},
    "bosses_pair_x": {"target", "inset", "dia", "h"},
    "bosses_corners": {"target", "inset", "dia", "h"},
    "shell_open_top": {"target", "wall"},
    "fillet": {"target", "r"},
    "chamfer": {"target", "size"},
    "export_stl": {"filepath"}, "export_obj": {"filepath"}, "export_ply": {"filepath"},
    "export_fbx": {"filepath"}, "export_gltf": {"filepath"},
}
EXPORT_OPS = {k for k in OPS if k.startswith("export_")}
STR_KEYS = {"name", "ref", "target", "tool", "filepath", "side", "axis"}
SIDES = {"left", "right", "front", "back"}

# op -> (anchor, style)
CIRCLE_OPS = {
    "hole_center": ("center", "plain"), "holes_pair_x": ("pair", "plain"),
    "holes_corners": ("corners", "plain"), "holes_grid": ("grid", "plain"),
    "bolt_circle": ("bolt", "plain"),
    "holes_pair_y": ("pair_y", "plain"), "holes_row_x": ("row_x", "plain"), "holes_row_y": ("row_y", "plain"),
    "cbore_pair_y": ("pair_y", "cbore"), "csink_pair_y": ("pair_y", "csink"),
    "cbore_center": ("center", "cbore"), "cbore_pair_x": ("pair", "cbore"),
    "cbore_corners": ("corners", "cbore"), "cbore_bolt_circle": ("bolt", "cbore"),
    "csink_center": ("center", "csink"), "csink_pair_x": ("pair", "csink"),
    "csink_corners": ("corners", "csink"), "csink_bolt_circle": ("bolt", "csink"),
    "pocket_round": ("center", "pocket"),
}
BOSS_OPS = {"boss_center": "center", "bosses_pair_x": "pair", "bosses_pair_y": "pair_y", "bosses_corners": "corners"}

# 3D-printing design rules (FDM, 0.4 mm nozzle)
MIN_WALL = 0.8      # thinnest wall / rim
MIN_RIM_RECT = 1.5  # material left around a pocket / window / slot
MIN_FLOOR = 1.2     # material left under a blind feature
MIN_HOLE = 2.0      # smallest hole diameter
MIN_EDGE_FRAC = 0.4
GAP = 0.8           # gap between neighbouring features
EPS = 1e-6


class _Part:
    def __init__(self):
        self.comps = []     # {"kind": box|cyl|cone, "bbox", "r", "tag"}
        self.voids = []     # {"shape": circle|rect, cx, cy, r | hw, hd}
        self.min_wall = None
        self.thin = None    # thinnest feature (limits fillet / chamfer size)
        self.free = False   # sphere / torus / tilted part: no flat top face for features

    def base_comps(self):
        return [c for c in self.comps if c.get("tag") != "boss"]

    def plain(self):
        return len(self.comps) == 1

    def bbox(self):
        cs = [c["bbox"] for c in self.comps]
        return (min(b[0] for b in cs), max(b[1] for b in cs), min(b[2] for b in cs),
                max(b[3] for b in cs), min(b[4] for b in cs), max(b[5] for b in cs))

    def shift(self, dx, dy, dz):
        for c in self.comps:
            b = c["bbox"]
            c["bbox"] = (b[0]+dx, b[1]+dx, b[2]+dy, b[3]+dy, b[4]+dz, b[5]+dz)
            if "c" in c:
                c["c"] = (c["c"][0]+dx, c["c"][1]+dy)
        for v in self.voids:
            v["cx"] += dx; v["cy"] += dy

    def note_thin(self, v):
        self.thin = v if self.thin is None else min(self.thin, v)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _rect(v):
    if v["shape"] == "circle":
        return (v["cx"]-v["r"], v["cx"]+v["r"], v["cy"]-v["r"], v["cy"]+v["r"])
    return (v["cx"]-v["hw"], v["cx"]+v["hw"], v["cy"]-v["hd"], v["cy"]+v["hd"])


def _collide(a, b, gap):
    if a["shape"] == "circle" and b["shape"] == "circle":
        return math.hypot(a["cx"]-b["cx"], a["cy"]-b["cy"]) < a["r"]+b["r"]+gap-EPS
    ra, rb = _rect(a), _rect(b)
    return not (ra[1]+gap <= rb[0]+EPS or rb[1]+gap <= ra[0]+EPS or
                ra[3]+gap <= rb[2]+EPS or rb[3]+gap <= ra[2]+EPS)


def _classify(comp, v):
    """(state, margin): is void v inside / outside / partly outside this component?"""
    x0, x1, y0, y1 = comp["bbox"][:4]
    fx0, fx1, fy0, fy1 = _rect(v)
    if comp["kind"] == "box":
        margin = min(fx0-x0, x1-fx1, fy0-y0, y1-fy1)
        if margin >= -EPS:
            return "inside", margin
        if fx1 <= x0+EPS or fx0 >= x1-EPS or fy1 <= y0+EPS or fy0 >= y1-EPS:
            return "outside", 0.0
        if v["shape"] == "circle":       # circle vs rectangle: exact distance test
            dx = max(x0-v["cx"], 0.0, v["cx"]-x1); dy = max(y0-v["cy"], 0.0, v["cy"]-y1)
            if math.hypot(dx, dy) >= v["r"]-EPS:
                return "outside", 0.0
        return "partial", 0.0
    R = comp["r"]
    ccx, ccy = comp.get("c", ((x0+x1)/2, (y0+y1)/2))
    if v["shape"] == "circle":
        dist = math.hypot(v["cx"]-ccx, v["cy"]-ccy)
        if dist + v["r"] <= R+EPS:
            return "inside", R-dist-v["r"]
        if dist >= R+v["r"]-EPS:
            return "outside", 0.0
        return "partial", 0.0
    far = max(math.hypot(px-ccx, py-ccy) for px in (fx0, fx1) for py in (fy0, fy1))
    if far <= R+EPS:
        return "inside", R-far
    dx = max(fx0-ccx, 0.0, ccx-fx1); dy = max(fy0-ccy, 0.0, ccy-fy1)
    return ("outside", 0.0) if math.hypot(dx, dy) >= R-EPS else ("partial", 0.0)


def _anchors(anchor, a, bb):
    cx, cy = (bb[0]+bb[1])/2, (bb[2]+bb[3])/2
    w, d = bb[1]-bb[0], bb[3]-bb[2]
    if anchor == "center":
        return [(cx, cy)], None
    if anchor == "pair":
        s = a["inset"]
        if s <= 0 or s >= w/2:
            return None, f"inset {s} does not fit in width {w:.1f}"
        return [(cx-(w/2-s), cy), (cx+(w/2-s), cy)], None
    if anchor == "pair_y":
        s = a["inset"]
        if s <= 0 or s >= d/2:
            return None, f"inset {s} does not fit in depth {d:.1f}"
        return [(cx, cy-(d/2-s)), (cx, cy+(d/2-s))], None
    if anchor in ("row_x", "row_y"):
        n, p = a["n"], a["pitch"]
        if n != int(n) or not 2 <= n <= 24 or p <= 0:
            return None, "row needs integer n in 2..24 and pitch > 0"
        n = int(n)
        if anchor == "row_x":
            return [(cx+(i-(n-1)/2)*p, cy) for i in range(n)], None
        return [(cx, cy+(i-(n-1)/2)*p) for i in range(n)], None
    if anchor == "corners":
        s = a["inset"]
        if s <= 0 or s >= min(w, d)/2:
            return None, f"inset {s} does not fit in {w:.1f}x{d:.1f}"
        return [(cx+sx*(w/2-s), cy+sy*(d/2-s)) for sx in (-1, 1) for sy in (-1, 1)], None
    if anchor == "bolt":
        n, pcd = a["n"], a["pcd"]
        if n != int(n) or n < 2 or n > 64 or pcd <= 0:
            return None, "bolt circle needs integer n between 2 and 64 and pcd > 0"
        n = int(n)
        return [(cx+pcd/2*math.cos(2*math.pi*k/n), cy+pcd/2*math.sin(2*math.pi*k/n))
                for k in range(n)], None
    if anchor == "grid":
        nx, ny, p = a["nx"], a["ny"], a["pitch"]
        if nx != int(nx) or ny != int(ny) or not (1 <= nx <= 12 and 1 <= ny <= 12) or nx*ny < 2 or p <= 0:
            return None, "grid needs integer nx, ny in 1..12 (at least 2 holes) and pitch > 0"
        nx, ny = int(nx), int(ny)
        return [(cx+(i-(nx-1)/2)*p, cy+(j-(ny-1)/2)*p) for i in range(nx) for j in range(ny)], None
    return None, "unknown anchor"


def _pts_ok(ring):
    return (isinstance(ring, list) and len(ring) >= 3 and
            all(isinstance(p, (list, tuple)) and len(p) == 2 and _num(p[0]) and _num(p[1]) for p in ring))


def _extrude_polygons(i, args, parts, err):
    """Validate an image-traced extrusion and register it as a free-form part (no top-face features)."""
    name, outer, holes, h = args.get("name"), args.get("outer"), args.get("holes", []), args.get("h")
    if not isinstance(name, str) or not name or name in parts:
        err(i, "extrude_polygons needs a new non-empty name"); return
    if not _pts_ok(outer) or not isinstance(holes, list) or not all(_pts_ok(x) for x in holes):
        err(i, "extrude_polygons needs outer=[[x,y],...] (3+ points) and holes=[[[x,y],...],...]"); return
    if not _num(h) or h <= 0:
        err(i, "extrude_polygons needs h > 0"); return
    xs, ys = [p[0] for p in outer], [p[1] for p in outer]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    for ring in holes:
        if any(not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1) for p in ring):
            err(i, "a hole lies outside the outline"); return
    p = _Part(); p.free = True
    p.comps.append({"kind": "free", "r": 0.001, "bbox": (x0, x1, y0, y1, 0.0, float(h))})
    p.note_thin(min(x1-x0, y1-y0, h)); parts[name] = p


def simulate(plan: list):
    """Run the plan. Returns (errors, parts, hole_events)."""
    errors: list = []
    parts: dict = {}
    events = 0
    seen_export = False

    def err(i, msg):
        errors.append(f"step {i+1}: {msg}")

    def get(i, name):
        if name not in parts:
            err(i, f"unknown object {name!r}")
            return None
        return parts[name]

    def flat(i, part):
        if part.free:
            err(i, "this part (sphere / torus / tilted) has no flat top face for features")
            return False
        return True

    def place(i, part, v, need, label):
        """Validate one void; returns True if it was accepted."""
        states = [(_classify(c, v), c) for c in part.base_comps()]
        if any(s[0] == "partial" for s, _ in states):
            err(i, f"{label} at ({v['cx']:.1f},{v['cy']:.1f}) breaks through an edge of the part")
            return False
        inside = [s for s, _ in states if s[0] == "inside"]
        if not inside:
            err(i, f"{label} at ({v['cx']:.1f},{v['cy']:.1f}) does not land on the part")
            return False
        best = max(s[1] for s in inside)
        if best < need-EPS:
            err(i, f"{label} at ({v['cx']:.1f},{v['cy']:.1f}) is too close to an edge")
            return False
        for o in part.voids:
            if _collide(v, o, GAP):
                err(i, f"{label} at ({v['cx']:.1f},{v['cy']:.1f}) overlaps another feature")
                return False
        part.voids.append(v)
        size = v["r"] if v["shape"] == "circle" else min(v["hw"], v["hd"])
        part.note_thin(min(size, best))
        return True

    for i, step in enumerate(plan):
        if not isinstance(step, dict) or "op" not in step:
            err(i, "step is not {op, args}"); continue
        op, args = step["op"], step.get("args", {})
        if op == "extrude_polygons":                       # image -> solid (see agent/image3d.py)
            _extrude_polygons(i, args, parts, err)
            continue
        if op not in OPS:
            err(i, f"unknown op {op!r}"); continue
        if not isinstance(args, dict):
            err(i, "args must be an object"); continue
        missing, extra = OPS[op]-set(args), set(args)-OPS[op]
        if missing:
            err(i, f"{op} missing args {sorted(missing)}"); continue
        if extra:
            err(i, f"{op} has unexpected args {sorted(extra)}"); continue
        bad = False
        for k, v in args.items():
            if k in STR_KEYS:
                if not isinstance(v, str) or not v:
                    err(i, f"{k} must be a non-empty string"); bad = True
            elif not _num(v):
                err(i, f"{k} must be a number"); bad = True
        if bad:
            continue
        if seen_export:
            err(i, "nothing may come after the export step")
        if op in EXPORT_OPS:
            seen_export = True
            if not parts:
                err(i, "export with nothing in the scene")
            continue

        # ---------------------------------------------------------------- primitives
        if op in ("add_box", "add_cyl", "add_cone"):
            if args["name"] in parts:
                err(i, f"name {args['name']!r} already used"); continue
            p = _Part()
            if op == "add_box":
                w, d, h = args["w"], args["d"], args["h"]
                if min(w, d, h) <= 0:
                    err(i, "box dimensions must be positive"); continue
                p.comps.append({"kind": "box", "bbox": (-w/2, w/2, -d/2, d/2, 0.0, h)})
                p.note_thin(min(w, d, h))
            elif op == "add_cyl":
                dia, h = args["dia"], args["h"]
                if min(dia, h) <= 0:
                    err(i, "cylinder dimensions must be positive"); continue
                R = dia/2
                p.comps.append({"kind": "cyl", "r": R, "bbox": (-R, R, -R, R, 0.0, h)})
                p.note_thin(min(dia, h))
            else:
                d1, d2, h = args["dia1"], args["dia2"], args["h"]
                if min(d1, d2, h) <= 0:
                    err(i, "cone dimensions must be positive"); continue
                Rb = max(d1, d2)/2
                p.comps.append({"kind": "cone", "r": min(d1, d2)/2, "bbox": (-Rb, Rb, -Rb, Rb, 0.0, h)})
                p.note_thin(min(d1, d2, h))
            parts[args["name"]] = p

        elif op == "add_sphere":
            if args["name"] in parts:
                err(i, f"name {args['name']!r} already used"); continue
            if args["dia"] <= 0:
                err(i, "sphere diameter must be positive"); continue
            R = args["dia"]/2
            p = _Part(); p.free = True
            p.comps.append({"kind": "sphere", "r": 0.001, "bbox": (-R, R, -R, R, 0.0, args["dia"])})
            p.note_thin(args["dia"]); parts[args["name"]] = p

        elif op == "add_prism":
            n, dia, h = args["sides"], args["dia"], args["h"]
            if n != int(n) or not 3 <= n <= 24 or dia <= 0 or h <= 0:
                err(i, "prism needs integer sides 3..24, dia > 0, h > 0"); continue
            if args["name"] in parts:
                err(i, f"name {args['name']!r} already used"); continue
            n = int(n); R = dia/2
            xs = [R*math.cos(2*math.pi*k/n) for k in range(n)]
            ys = [R*math.sin(2*math.pi*k/n) for k in range(n)]
            p = _Part()
            p.comps.append({"kind": "cyl", "r": R*math.cos(math.pi/n), "c": (0.0, 0.0),
                            "bbox": (min(xs), max(xs), min(ys), max(ys), 0.0, h)})
            p.note_thin(min(dia*math.cos(math.pi/n), h)); parts[args["name"]] = p

        elif op == "add_torus":
            mj, mn = args["major"], args["minor"]
            if mn <= 0 or mj <= mn:
                err(i, "torus needs minor > 0 and major > minor"); continue
            if args["name"] in parts:
                err(i, f"name {args['name']!r} already used"); continue
            R = mj/2 + mn/2
            p = _Part(); p.free = True
            p.comps.append({"kind": "torus", "r": 0.001, "bbox": (-R, R, -R, R, 0.0, mn)})
            p.note_thin(mn); parts[args["name"]] = p

        elif op == "move":
            a = get(i, args["name"])
            if a is None:
                continue
            bb = a.bbox()
            a.shift(args["x"]-(bb[0]+bb[1])/2, args["y"]-(bb[2]+bb[3])/2, args["z"]-bb[4])

        elif op == "rotate":
            a = get(i, args["name"])
            if a is None:
                continue
            ax, deg = args["axis"], float(args["deg"])
            if ax not in ("x", "y", "z"):
                err(i, "rotate needs axis x, y or z"); continue
            if a.voids:
                err(i, "rotate a part before adding holes/pockets to it"); continue
            if abs(deg) % 360 < 1e-9:
                continue
            bb = a.bbox()
            cx, cy, cz = (bb[0]+bb[1])/2, (bb[2]+bb[3])/2, (bb[4]+bb[5])/2
            th = math.radians(deg)
            ct, st = math.cos(th), math.sin(th)
            for c in a.comps:
                b = c["bbox"]
                pts = [(x-cx, y-cy, z-cz) for x in (b[0], b[1]) for y in (b[2], b[3]) for z in (b[4], b[5])]
                if ax == "z": pts = [(x*ct-y*st, x*st+y*ct, z) for x, y, z in pts]
                elif ax == "x": pts = [(x, y*ct-z*st, y*st+z*ct) for x, y, z in pts]
                else: pts = [(x*ct+z*st, y, -x*st+z*ct) for x, y, z in pts]
                c["bbox"] = (min(p[0] for p in pts)+cx, max(p[0] for p in pts)+cx,
                             min(p[1] for p in pts)+cy, max(p[1] for p in pts)+cy,
                             min(p[2] for p in pts)+cz, max(p[2] for p in pts)+cz)
                c["c"] = ((c["bbox"][0]+c["bbox"][1])/2, (c["bbox"][2]+c["bbox"][3])/2)
                if abs(deg) % 90 > 1e-9 or ax != "z":     # not an exact quarter turn about z: no flat-face features
                    c["kind"], c["r"] = "free", 0.001
            if abs(deg) % 90 > 1e-9 or ax != "z":
                a.free = True

        elif op == "array_polar":
            a = get(i, args["name"])
            n = args["count"]
            if a is None:
                continue
            if n != int(n) or not 2 <= n <= 120:
                err(i, "array_polar needs integer count 2..120"); continue
            import copy as _copy
            base = _copy.deepcopy(a.comps)
            for kk in range(1, int(n)):
                th = 2*math.pi*kk/int(n); ct, st = math.cos(th), math.sin(th)
                for c in base:
                    c2 = _copy.deepcopy(c); b = c2["bbox"]
                    pts = [(x*ct-y*st, x*st+y*ct) for x in (b[0], b[1]) for y in (b[2], b[3])]
                    c2["bbox"] = (min(p[0] for p in pts), max(p[0] for p in pts), min(p[1] for p in pts),
                                  max(p[1] for p in pts), b[4], b[5])
                    c2["kind"], c2["r"] = "free", 0.001
                    c2["c"] = ((c2["bbox"][0]+c2["bbox"][1])/2, (c2["bbox"][2]+c2["bbox"][3])/2)
                    a.comps.append(c2)
            a.free = True
            a.voids = []

        elif op in ("array_x", "array_y"):
            a = get(i, args["name"])
            if a is None:
                continue
            n, pitch = args["count"], args["pitch"]
            if n != int(n) or not 2 <= n <= 60 or pitch <= 0:
                err(i, "array needs integer count 2..60 and pitch > 0"); continue
            import copy as _copy
            base_comps, base_voids = _copy.deepcopy(a.comps), _copy.deepcopy(a.voids)
            for kk in range(1, int(n)):
                dx, dy = (kk*pitch, 0.0) if op == "array_x" else (0.0, kk*pitch)
                for c in base_comps:
                    c2 = _copy.deepcopy(c); b = c2["bbox"]
                    c2["bbox"] = (b[0]+dx, b[1]+dx, b[2]+dy, b[3]+dy, b[4], b[5])
                    if "c" in c2: c2["c"] = (c2["c"][0]+dx, c2["c"][1]+dy)
                    a.comps.append(c2)
                for v in base_voids:
                    v2 = dict(v); v2["cx"] += dx; v2["cy"] += dy; a.voids.append(v2)

        elif op == "window_at":
            t = get(i, args["target"])
            if t is None or not flat(i, t):
                continue
            bb = t.bbox()
            if min(args["w"], args["d"]) < 4.0:
                err(i, "window must be at least 4 mm in each direction"); continue
            place(i, t, {"shape": "rect", "cx": (bb[0]+bb[1])/2+args["x"], "cy": (bb[2]+bb[3])/2+args["y"],
                         "hw": args["w"]/2, "hd": args["d"]/2}, MIN_RIM_RECT, "window")

        elif op == "hole_at":
            t = get(i, args["target"])
            if t is None or not flat(i, t):
                continue
            bb = t.bbox()
            dia = args["dia"]
            if dia < MIN_HOLE:
                err(i, f"diameter {dia} is below the printable minimum"); continue
            need = max(MIN_WALL, min(MIN_EDGE_FRAC*dia, 3.0))
            if place(i, t, {"shape": "circle", "cx": (bb[0]+bb[1])/2+args["x"],
                            "cy": (bb[2]+bb[3])/2+args["y"], "r": dia/2}, need, "hole"):
                events += 1

        # ---------------------------------------------------------------- placement
        elif op in ("align_back", "align_side", "stack_on"):
            a, b = get(i, args["name"]), get(i, args["ref"])
            if a is None or b is None:
                continue
            ab, bb = a.bbox(), b.bbox()
            if op == "stack_on":
                a.shift((bb[0]+bb[1])/2-(ab[0]+ab[1])/2, (bb[2]+bb[3])/2-(ab[2]+ab[3])/2, bb[5]-ab[4])
            else:
                side = "back" if op == "align_back" else args["side"]
                if side not in SIDES:
                    err(i, f"side must be one of {sorted(SIDES)}"); continue
                if side == "back":
                    a.shift(0.0, bb[3]-ab[3], 0.0)
                elif side == "front":
                    a.shift(0.0, bb[2]-ab[2], 0.0)
                elif side == "right":
                    a.shift(bb[1]-ab[1], 0.0, 0.0)
                else:
                    a.shift(bb[0]-ab[0], 0.0, 0.0)

        # ---------------------------------------------------------------- booleans
        elif op in ("union", "cut"):
            a, b = get(i, args["target"]), get(i, args["tool"])
            if a is None or b is None:
                continue
            if args["target"] == args["tool"]:
                err(i, "target and tool are the same object"); continue
            ab, bb = a.bbox(), b.bbox()
            ox = min(ab[1], bb[1])-max(ab[0], bb[0])
            oy = min(ab[3], bb[3])-max(ab[2], bb[2])
            oz = min(ab[5], bb[5])-max(ab[4], bb[4])
            if op == "union":
                if ox < -EPS or oy < -EPS or oz < -EPS:
                    err(i, "union parts do not touch"); continue
                blocked = False
                for src, dst in ((a, b), (b, a)):
                    for v in src.voids:
                        for c in dst.base_comps():
                            zr = min(src.bbox()[5], c["bbox"][5])-max(src.bbox()[4], c["bbox"][4])
                            if zr >= -EPS and _classify(c, v)[0] != "outside":
                                blocked = True
                if blocked:
                    err(i, "union would cover or block a hole / pocket"); continue
                a.comps += b.comps; a.voids += b.voids
                if b.thin is not None:
                    a.note_thin(b.thin)
            else:
                if ox <= EPS or oy <= EPS or oz <= EPS:
                    err(i, "cut tool does not overlap the target"); continue
                if (bb[0] <= ab[0] and bb[1] >= ab[1] and bb[2] <= ab[2] and bb[3] >= ab[3]
                        and bb[4] <= ab[4] and bb[5] >= ab[5]):
                    err(i, "cut tool swallows the whole target"); continue
                c = b.comps[0]
                cx, cy = (c["bbox"][0]+c["bbox"][1])/2, (c["bbox"][2]+c["bbox"][3])/2
                if c["kind"] in ("cyl", "cone"):
                    place(i, a, {"shape": "circle", "cx": cx, "cy": cy, "r": c["r"]}, MIN_WALL, "cut")
                else:
                    place(i, a, {"shape": "rect", "cx": cx, "cy": cy,
                                 "hw": (c["bbox"][1]-c["bbox"][0])/2, "hd": (c["bbox"][3]-c["bbox"][2])/2},
                          MIN_RIM_RECT, "cut")
            del parts[args["tool"]]

        # ---------------------------------------------------------------- circular features
        elif op in CIRCLE_OPS:
            t = get(i, args["target"])
            if t is None or not flat(i, t):
                continue
            anchor, style = CIRCLE_OPS[op]
            bb = t.bbox()
            pts, msg = _anchors(anchor, args, bb)
            if msg:
                err(i, msg); continue
            dia = args.get("dia")
            h = bb[5]-bb[4]
            if dia < (4.0 if style == "pocket" else MIN_HOLE):
                err(i, f"diameter {dia} is below the printable minimum"); continue
            fd = dia
            if style in ("cbore", "csink", "pocket") and not t.plain():
                err(i, f"{op} must be done while the part is a single plain solid"); continue
            if style == "cbore":
                cb, cd = args["cb_dia"], args["cb_depth"]
                if cb < dia+1.5 or cd < 0.8 or h-cd < MIN_FLOOR:
                    err(i, "counterbore does not fit (needs cb_dia >= dia+1.5, depth >= 0.8, floor >= 1.2)"); continue
                fd = cb
            elif style == "csink":
                cs = args["cs_dia"]
                if cs < dia+1.0 or (cs-dia)/2 > h-MIN_FLOOR:
                    err(i, "countersink does not fit"); continue
                fd = cs
            elif style == "pocket":
                if args["depth"] < 0.8 or h-args["depth"] < MIN_FLOOR:
                    err(i, "pocket depth leaves less than 1.2 mm floor"); continue
            need = max(MIN_WALL, min(MIN_EDGE_FRAC*fd, 3.0))
            ok = True
            for (x, y) in pts:
                ok &= place(i, t, {"shape": "circle", "cx": x, "cy": y, "r": fd/2}, need, "feature")
            if ok and anchor != "center" and anchor != "grid" and style != "pocket":
                events += len(pts)

        # ---------------------------------------------------------------- rectangular features
        elif op in ("slot_center", "pocket_rect", "window_rect"):
            t = get(i, args["target"])
            if t is None or not flat(i, t):
                continue
            bb = t.bbox()
            h = bb[5]-bb[4]
            cx, cy = (bb[0]+bb[1])/2, (bb[2]+bb[3])/2
            if op == "slot_center":
                L, W = args["length"], args["width"]
                if W < 2.0 or L < 1.5*W:
                    err(i, "slot needs width >= 2 and length >= 1.5 x width"); continue
                hw, hd = L/2, W/2
            else:
                hw, hd = args["w"]/2, args["d"]/2
                if min(hw, hd) < 2.0:
                    err(i, "pocket/window must be at least 4 mm in each direction"); continue
                if op == "pocket_rect":
                    if not t.plain():
                        err(i, "pocket_rect must be done while the part is a single plain solid"); continue
                    if args["depth"] < 0.8 or h-args["depth"] < MIN_FLOOR:
                        err(i, "pocket depth leaves less than 1.2 mm floor"); continue
            place(i, t, {"shape": "rect", "cx": cx, "cy": cy, "hw": hw, "hd": hd}, MIN_RIM_RECT, op)

        # ---------------------------------------------------------------- bosses
        elif op in BOSS_OPS:
            t = get(i, args["target"])
            if t is None or not flat(i, t):
                continue
            if len(t.base_comps()) != 1:
                err(i, "bosses go on a single plain top face (before joining parts)"); continue
            bb = t.bbox()
            base = t.base_comps()[0]
            pts, msg = _anchors(BOSS_OPS[op], args, (base["bbox"][0], base["bbox"][1], base["bbox"][2],
                                                     base["bbox"][3], base["bbox"][4], base["bbox"][5]))
            if msg:
                err(i, msg); continue
            dia, hb = args["dia"], args["h"]
            if dia < 2.0 or hb < 1.0:
                err(i, "boss needs dia >= 2 and h >= 1"); continue
            for (x, y) in pts:
                v = {"shape": "circle", "cx": x, "cy": y, "r": dia/2}
                if not place(i, t, v, 1.0, "boss"):
                    break
                t.comps.append({"kind": "cyl", "r": dia/2, "tag": "boss",
                                "bbox": (x-dia/2, x+dia/2, y-dia/2, y+dia/2, base["bbox"][5], base["bbox"][5]+hb)})
            t.note_thin(dia/2)

        # ---------------------------------------------------------------- shell / edges
        elif op == "shell_open_top":
            t = get(i, args["target"])
            if t is None or not flat(i, t):
                continue
            wall = args["wall"]
            bb = t.bbox()
            w, d, h = bb[1]-bb[0], bb[3]-bb[2], bb[5]-bb[4]
            if not t.plain() or t.comps[0]["kind"] != "box" or t.voids:
                err(i, "shell_open_top needs a single plain box"); continue
            if wall < MIN_WALL or wall > min(w, d)/2-1 or wall > h-1:
                err(i, f"wall {wall} does not fit a {w:.1f}x{d:.1f}x{h:.1f} box"); continue
            t.min_wall = wall
            t.note_thin(wall)

        elif op in ("fillet", "chamfer"):
            t = get(i, args["target"])
            if t is None:
                continue
            r = args["r"] if op == "fillet" else args["size"]
            thinnest = min(t.thin if t.thin is not None else 1e9,
                           min(min(c["bbox"][1]-c["bbox"][0], c["bbox"][3]-c["bbox"][2],
                                   c["bbox"][5]-c["bbox"][4]) for c in t.comps))
            if t.min_wall:
                thinnest = min(thinnest, t.min_wall)
            if r <= 0 or r > 0.4*thinnest+EPS:
                err(i, f"{op} {r} too large for thinnest feature {thinnest:.1f}")

    if not errors and len(parts) != 1:
        errors.append(f"end: {len(parts)} objects left in the scene, expected exactly 1 "
                      f"(a cutter or part was never merged): {sorted(parts)}")
    return errors, parts, events


# ---------------------------------------------------------------------------
# Prompt consistency
# ---------------------------------------------------------------------------
_TRIPLE = re.compile(r"(\d+(?:\.\d+)?)\s*[x×X]\s*(\d+(?:\.\d+)?)\s*[x×X]\s*(\d+(?:\.\d+)?)")
_MM_NUM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?=\s*(?:mm\b|[x×X]\s*\d))")
_XYZ = re.compile(r"(?<![A-Za-z])[xyz]\s*=\s*(-?\d+(?:\.\d+)?)")
_COUNT = re.compile(r"(?<![\d.])(\d+)\s+(?!mm)(?:[a-zA-Z-]+\s+){0,2}?holes?\b")


def _consistency(prompt: str, plan: list, events: int) -> list:
    errs = []
    m = _TRIPLE.search(prompt)
    if m:
        want = sorted(float(g) for g in m.groups())
        boxes = [sorted((s["args"]["w"], s["args"]["d"], s["args"]["h"]))
                 for s in plan if s.get("op") == "add_box"]
        if not any(all(abs(a-b) < 1e-6 for a, b in zip(want, bx)) for bx in boxes):
            errs.append(f"prompt asks for a {m.group(0)} box but no add_box has those dimensions")
    counts = [int(c) for c in _COUNT.findall(prompt)]
    if counts and sum(counts) != events:
        errs.append(f"prompt asks for {sum(counts)} holes but plan drills {events}")
    nums = [v for s in plan for v in s.get("args", {}).values() if _num(v)]
    for tok in _MM_NUM.findall(prompt):
        if not any(abs(float(tok)-v) < 1e-6 for v in nums):
            errs.append(f"prompt mentions {tok} but no plan step uses that value")
    for tok in _XYZ.findall(prompt):
        if not any(abs(float(tok)-v) < 1e-6 for v in nums):
            errs.append(f"prompt gives coordinate {tok} but no plan step uses that value")
    if re.search(r"\bexport(?:ed)? as\b", prompt, re.I):
        if not plan or plan[-1].get("op") not in EXPORT_OPS:
            errs.append("prompt asks for an export but the plan does not end with one")
    return errs


def check_plan(plan: list, prompt: Optional[str] = None) -> list:
    errors, _, events = simulate(plan)
    if prompt and not errors:
        errors += _consistency(prompt, plan, events)
    return errors


def predicted_bbox(plan: list):
    """Final bounding box (x0,x1,y0,y1,z0,z1) the plan should produce, or None if invalid."""
    errors, parts, _ = simulate(plan)
    if errors or len(parts) != 1:
        return None
    return next(iter(parts.values())).bbox()


def extract_plan(text: str) -> Optional[list]:
    try:
        return json.loads(text[text.index("["):text.rindex("]")+1])
    except Exception:
        return None


def grade_text(prompt: str, text: str):
    """Grader used by Stage 3: returns (passed, errors)."""
    plan = extract_plan(text)
    if not isinstance(plan, list) or not plan:
        return False, ["no valid JSON plan found"]
    errs = check_plan(plan, prompt)
    return (not errs), errs


def scene_after(plan: list) -> Optional[dict]:
    """{object name: (dx, dy, dz)} that should exist after running `plan` (a prefix of a longer plan), or None if
    the prefix is already invalid. Unlike predicted_bbox it allows several objects (mid-build states)."""
    errors, parts, _ = simulate(plan)
    errors = [e for e in errors if not e.startswith("end:")]
    if errors:
        return None
    out = {}
    for n, p in parts.items():
        b = p.bbox()
        out[n] = (round(b[1]-b[0], 3), round(b[3]-b[2], 3), round(b[5]-b[4], 3))
    return out


# ---------------------------------------------------------------------------
# Item 7: learned ranking ON TOP OF check_plan, not instead of it.
#
# check_plan/simulate above stay exactly as they were - they are the hard-invariant layer (manifold-sane,
# collision-free, matches what the prompt asked for) and the offline test suite asserts against them
# directly, so nothing above this line changed. What this file didn't have before was a way to choose BETWEEN
# several structurally-valid plans - it could only say "valid" or "list of errors", not "which of these five
# valid plans is the better build". That's what task_planner's MCTS rollout (build order item 8) needs:
# a scoring function to rank candidates, not another pass/fail gate.
#
# rank_candidates() takes candidate plans that have ALREADY passed check_plan (it discards invalid ones
# outright - a learned model never gets a vote on physical validity), and orders the survivors using an
# optional learned scorer. With no scorer supplied it falls back to a transparent heuristic (fewer build
# steps, more compact bounding box) so the function is always usable even before spec_auditor is trained.
# ---------------------------------------------------------------------------

def _heuristic_score(plan: list, bbox) -> float:
    """Used only when no learned scorer is supplied. Smaller/simpler is scored higher - a legible default,
    not a claim that it's what a trained spec_auditor would learn to prefer."""
    step_penalty = -0.01 * len(plan)
    if bbox is None:
        return step_penalty
    x0, x1, y0, y1, z0, z1 = bbox
    volume_mm3 = max(0.0, x1 - x0) * max(0.0, y1 - y0) * max(0.0, z1 - z0)
    return step_penalty - 1e-6 * volume_mm3


def rank_candidates(plans: list, prompt: Optional[str] = None, scorer=None) -> list:
    """plans: list of candidate op-plans (e.g. several MCTS rollouts, or spec_generator sampled at different
    temperatures). Returns [(score, plan, errors_or_None), ...] sorted best-first. Plans that fail check_plan
    are still returned (errors is non-empty, score is None) so a caller can log why a candidate was dropped,
    but they always sort after every valid candidate regardless of score.

    scorer(plan, bbox, prompt) -> float: pass the trained spec_auditor's scoring function once it exists.
    Higher must mean better for both the learned scorer and the heuristic fallback."""
    scored, invalid = [], []
    for plan in plans:
        errors = check_plan(plan, prompt)
        if errors:
            invalid.append((None, plan, errors))
            continue
        bbox = predicted_bbox(plan)
        score = scorer(plan, bbox, prompt) if scorer is not None else _heuristic_score(plan, bbox)
        scored.append((score, plan, None))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored + invalid
