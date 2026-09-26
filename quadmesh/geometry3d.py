"""
Shape generators that make REAL triangle meshes in plain Python + numpy (no Blender, no CAD kernel):
revolved solids (lathe), lofts between profiles, sweeps along a path (pipes, springs, rails, handles), threads
(threaded rod, nut), extruded polygons with holes (gears, text, stencils), inflated silhouettes.

Every generator returns a Mesh (vertices, triangles). Meshes are closed and consistently oriented by construction and are
checked with agent/printability.analyze_mesh before anything is called printable. write_stl() lets any slicer open them, and
Blender imports them with the `import_stl` op - so none of these shapes depend on untested Blender code.
Units: millimetres.
"""
from __future__ import annotations
import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Mesh:
    V: np.ndarray            # (n, 3) float
    F: np.ndarray            # (m, 3) int

    def triangles(self) -> np.ndarray:
        return self.V[self.F]

    def volume(self) -> float:
        t = self.triangles()
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    def bbox(self):
        return self.V.min(axis=0), self.V.max(axis=0)

    def translated(self, dx=0.0, dy=0.0, dz=0.0) -> "Mesh":
        return Mesh(self.V + np.array([dx, dy, dz]), self.F.copy())

    def on_bed(self) -> "Mesh":
        """Centre on x,y and rest on z = 0 (how a printer wants it)."""
        lo, hi = self.bbox()
        return self.translated(-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])


def merge(*meshes: Mesh) -> Mesh:
    V, F, off = [], [], 0
    for m in meshes:
        V.append(m.V); F.append(m.F + off); off += len(m.V)
    return Mesh(np.vstack(V), np.vstack(F))


def orient_outward(m: Mesh) -> Mesh:
    """Flip all faces if the signed volume is negative (faces pointing inward)."""
    return Mesh(m.V, m.F[:, [0, 2, 1]]) if m.volume() < 0 else m


def write_stl(path: str, m: Mesh, name: str = "quadmesh") -> str:
    t = m.triangles()
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    with open(path, "wb") as fh:
        fh.write(name.encode()[:80].ljust(80, b"\0") + struct.pack("<I", len(t)))
        rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, t
        fh.write(rec.tobytes())
    return path


# ------------------------------------------------------------------------------------------------ 2-D polygons
def _area(p) -> float:
    return 0.5 * sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))


def _seg_hit(a, b, c, d) -> bool:
    """Proper intersection of segments ab and cd (touching at endpoints does not count)."""
    def o(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = o(a, b, c), o(a, b, d), o(c, d, a), o(c, d, b)
    return (o1 * o2 < -1e-12) and (o3 * o4 < -1e-12)


def _in_tri(p, a, b, c) -> bool:
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    return not ((d1 < -1e-12 or d2 < -1e-12 or d3 < -1e-12) and (d1 > 1e-12 or d2 > 1e-12 or d3 > 1e-12)) \
        and abs(d1) > 1e-12 and abs(d2) > 1e-12 and abs(d3) > 1e-12


def triangulate(outer: Sequence, holes: Sequence[Sequence] = ()):
    """Ear-clipping triangulation of a polygon with holes. Returns (points, triangles): triangles index the concatenated
    list outer + holes[0] + holes[1] ... and use ONLY those vertices, so extrusions stay watertight."""
    pts = [tuple(map(float, p)) for p in outer] + [tuple(map(float, p)) for h in holes for p in h]
    n0 = len(outer)
    rings = [list(range(n0))]
    o = n0
    for h in holes:
        rings.append(list(range(o, o + len(h)))); o += len(h)
    if _area([pts[i] for i in rings[0]]) < 0:
        rings[0].reverse()
    for k in range(1, len(rings)):
        if _area([pts[i] for i in rings[k]]) > 0:
            rings[k].reverse()
    edges = [(r[i], r[(i + 1) % len(r)]) for r in rings for i in range(len(r))]
    poly = list(rings[0])
    for hr in sorted(rings[1:], key=lambda r: -max(pts[i][0] for i in r)):     # bridge each hole into the outline
        m = max(hr, key=lambda i: pts[i][0])
        cands = sorted(range(len(poly)), key=lambda j: (pts[poly[j]][0] - pts[m][0]) ** 2 + (pts[poly[j]][1] - pts[m][1]) ** 2)
        for j in cands:
            v = poly[j]
            if not any(_seg_hit(pts[m], pts[v], pts[a], pts[b]) for a, b in edges if a not in (m, v) and b not in (m, v)):
                break
        k = hr.index(m)
        hole_walk = hr[k:] + hr[:k] + [m]
        poly = poly[:j + 1] + hole_walk + [poly[j]] + poly[j + 1:]
        edges.append((m, v))
    tris, idx = [], list(poly)
    guard = 0
    while len(idx) > 3 and guard < 10 * len(poly) ** 2:
        guard += 1
        clipped = False
        for i in range(len(idx)):
            a, b, c = idx[i - 1], idx[i], idx[(i + 1) % len(idx)]
            pa, pb, pc = pts[a], pts[b], pts[c]
            if (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]) <= 1e-12:
                continue                                   # reflex or flat
            if any(_in_tri(pts[q], pa, pb, pc) for q in idx if q not in (a, b, c) and pts[q] not in (pa, pb, pc)):
                continue
            tris.append((a, b, c)); idx.pop(i); clipped = True
            break
        if not clipped:                                    # degenerate sliver: drop the flattest vertex
            j = min(range(len(idx)), key=lambda i: abs((pts[idx[i]][0] - pts[idx[i - 1]][0]) * (pts[idx[(i + 1) % len(idx)]][1] - pts[idx[i - 1]][1])
                                                        - (pts[idx[i]][1] - pts[idx[i - 1]][1]) * (pts[idx[(i + 1) % len(idx)]][0] - pts[idx[i - 1]][0])))
            idx.pop(j)
    if len(idx) == 3:
        tris.append(tuple(idx))
    return pts, tris


def extrude_polygon(outer, holes=(), height: float = 5.0, z0: float = 0.0) -> Mesh:
    """Solid prism from a polygon (with holes): caps + side walls on the SAME vertices, so it is closed."""
    pts, tris = triangulate(outer, holes)
    n = len(pts)
    V = np.array([[x, y, z0] for x, y in pts] + [[x, y, z0 + height] for x, y in pts], float)
    F = [(a, c, b) for a, b, c in tris] + [(n + a, n + b, n + c) for a, b, c in tris]
    rings, o = [list(range(len(outer)))], len(outer)
    for h in holes:
        rings.append(list(range(o, o + len(h)))); o += len(h)
    for k, r in enumerate(rings):
        ccw = _area([pts[i] for i in r]) > 0
        if (k == 0) != ccw:                              # outer must run counter-clockwise, holes clockwise
            r = r[::-1]
        for i in range(len(r)):
            a, b = r[i], r[(i + 1) % len(r)]
            F.append((a, b, n + b)); F.append((a, n + b, n + a))
    return orient_outward(Mesh(V, np.array(F, int)))


# ------------------------------------------------------------------------------------------------ revolve / loft / sweep
def lathe(profile: Sequence[Tuple[float, float]], segments: int = 96) -> Mesh:
    """Solid of revolution about the Z axis. `profile` is a CLOSED polygon of (radius, z) points, radius >= 0, e.g. a vase
    wall: [(0,0),(30,0),(34,40),(20,90),(22,100),(18,100),(16,90),(28,40),(0,3)]. Vertices on the axis are shared."""
    prof = [(max(0.0, float(r)), float(z)) for r, z in profile]
    V, ring = [], []
    for r, z in prof:
        if r < 1e-9:
            ring.append([len(V)]); V.append([0.0, 0.0, z])
        else:
            ids = []
            for k in range(segments):
                a = 2 * math.pi * k / segments
                ids.append(len(V)); V.append([r * math.cos(a), r * math.sin(a), z])
            ring.append(ids)
    F = []
    for i in range(len(prof)):
        A, B = ring[i], ring[(i + 1) % len(prof)]
        if len(A) == 1 and len(B) == 1:
            continue
        for k in range(segments):
            k2 = (k + 1) % segments
            if len(A) == 1:
                F.append((A[0], B[k], B[k2]))
            elif len(B) == 1:
                F.append((A[k], B[0], A[k2]))
            else:
                F.append((A[k], B[k], B[k2])); F.append((A[k], B[k2], A[k2]))
    return orient_outward(Mesh(np.array(V, float), np.array(F, int)))


def loft(rings: Sequence[np.ndarray], cap: bool = True) -> Mesh:
    """Skin through cross-sections. rings: list of (n,3) arrays with the SAME point count n (use the *_ring helpers)."""
    R = [np.asarray(r, float) for r in rings]
    n = len(R[0])
    V = np.vstack(R)
    F = []
    for i in range(len(R) - 1):
        for k in range(n):
            k2 = (k + 1) % n
            a, b, c, d = i * n + k, i * n + k2, (i + 1) * n + k2, (i + 1) * n + k
            F.append((a, b, c)); F.append((a, c, d))
    if cap:
        for j, sign in ((0, -1), (len(R) - 1, 1)):
            cen = R[j].mean(axis=0); ci = len(V)
            V = np.vstack([V, cen])
            for k in range(n):
                a, b = j * n + k, j * n + (k + 1) % n
                F.append((ci, a, b) if sign > 0 else (ci, b, a))
    return orient_outward(Mesh(V, np.array(F, int)))


def circle_ring(r: float, z: float, n: int = 64, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    a = 2 * np.pi * np.arange(n) / n
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, z)], 1)


def ellipse_ring(rx: float, ry: float, z: float, n: int = 64, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    a = 2 * np.pi * np.arange(n) / n
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a), np.full(n, z)], 1)


def rect_ring(w: float, d: float, z: float, n: int = 64, corner: float = 0.0) -> np.ndarray:
    """Rounded rectangle resampled to n points (a super-ellipse, so lofts to circles are smooth)."""
    e = 2.0 / max(0.15, 1.0 - min(corner, 0.95)) if corner < 1 else 2.0
    a = 2 * np.pi * np.arange(n) / n
    c, s = np.cos(a), np.sin(a)
    return np.stack([(w / 2) * np.sign(c) * np.abs(c) ** (2 / e), (d / 2) * np.sign(s) * np.abs(s) ** (2 / e), np.full(n, z)], 1)


def sweep(profile: np.ndarray, path: np.ndarray, twist_deg: float = 0.0, cap: bool = True) -> Mesh:
    """Move a 2-D profile ((n,2) points, in the plane perpendicular to the path) along a 3-D polyline `path` ((m,3)),
    using parallel-transport frames so it does not spin. Pipes, springs, rails, handles, wires."""
    P = np.asarray(path, float)
    prof = np.asarray(profile, float)
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    up = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N = np.cross(up, T[0]); N /= np.linalg.norm(N)
    rings = []
    for i in range(len(P)):
        if i > 0:                                        # rotate the frame from tangent i-1 to tangent i
            ax = np.cross(T[i - 1], T[i]); s = np.linalg.norm(ax)
            if s > 1e-9:
                ax /= s; ang = math.atan2(s, float(np.dot(T[i - 1], T[i])))
                N = N * math.cos(ang) + np.cross(ax, N) * math.sin(ang) + ax * float(np.dot(ax, N)) * (1 - math.cos(ang))
        B = np.cross(T[i], N)
        t = math.radians(twist_deg) * i / max(1, len(P) - 1)
        c, s_ = math.cos(t), math.sin(t)
        Nr, Br = N * c + B * s_, -N * s_ + B * c
        rings.append(P[i] + np.outer(prof[:, 0], Nr) + np.outer(prof[:, 1], Br))
    return loft(rings, cap=cap)


def circle_profile(r: float, n: int = 32) -> np.ndarray:
    a = 2 * np.pi * np.arange(n) / n
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def helix_path(radius: float, pitch: float, turns: float, per_turn: int = 48) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi * turns, int(per_turn * turns) + 1)
    return np.stack([radius * np.cos(t), radius * np.sin(t), pitch * t / (2 * np.pi)], 1)


# ------------------------------------------------------------------------------------------------ threads
def _thread_radius(phase: np.ndarray, pitch: float, r_major: float, r_minor: float, flat: float = 0.125) -> np.ndarray:
    """Radius as a function of the position inside one pitch (0..1). ISO-style trapezoid: flat crest, 60-degree flanks, flat root."""
    p = np.mod(phase, 1.0)
    crest, root = flat, 0.25
    rise = (1.0 - crest - root) / 2.0
    out = np.empty_like(p)
    for i, x in enumerate(p):
        if x < crest / 2 or x >= 1 - crest / 2:
            out[i] = r_major
        elif x < crest / 2 + rise:
            out[i] = r_major - (r_major - r_minor) * (x - crest / 2) / rise
        elif x < crest / 2 + rise + root:
            out[i] = r_minor
        else:
            out[i] = r_minor + (r_major - r_minor) * (x - (crest / 2 + rise + root)) / rise
    return out


def threaded_rod(major_dia: float, pitch: float, length: float, minor_dia: Optional[float] = None,
                 segments: int = 72, samples_per_pitch: int = 48) -> Mesh:
    """External thread (a bolt shank / lead screw / threaded rod). Default depth follows ISO metric: minor = major - 1.2269 pitch."""
    rM = major_dia / 2
    rm = (minor_dia if minor_dia else major_dia - 1.2269 * pitch) / 2
    nz = max(8, int(round(length / pitch * samples_per_pitch)))
    z = np.linspace(0, length, nz + 1)
    ang = 2 * np.pi * np.arange(segments) / segments
    rings = []
    for zi in z:
        r = _thread_radius(zi / pitch - ang / (2 * np.pi), pitch, rM, rm)
        rings.append(np.stack([r * np.cos(ang), r * np.sin(ang), np.full(segments, zi)], 1))
    return loft(rings, cap=True)


def nut(major_dia: float, pitch: float, height: float, across_flats: Optional[float] = None, clearance: float = 0.3,
        segments: int = 72, samples_per_pitch: int = 48) -> Mesh:
    """Hex nut with an internal thread. `clearance` (mm) opens the thread so a printed bolt actually turns in it."""
    af = across_flats or 1.6 * major_dia
    rM = major_dia / 2 + clearance
    rm = (major_dia - 1.2269 * pitch) / 2 + clearance
    nz = max(8, int(round(height / pitch * samples_per_pitch)))
    z = np.linspace(0, height, nz + 1)
    ang = 2 * np.pi * np.arange(segments) / segments
    R_hex = af / math.sqrt(3)                                     # across corners / 2
    hexr = np.array([R_hex * math.cos(math.pi / 6) / max(math.cos(((a + math.pi / 6) % (math.pi / 3)) - math.pi / 6), 1e-9) for a in ang])
    hexr = np.array([(af / 2) / math.cos(((a % (math.pi / 3)) - math.pi / 6)) for a in ang])
    inner = []
    for zi in z:
        r = _thread_radius(zi / pitch - ang / (2 * np.pi), pitch, rM, rm)
        inner.append(np.stack([r * np.cos(ang), r * np.sin(ang), np.full(segments, zi)], 1))
    outer = [np.stack([hexr * np.cos(ang), hexr * np.sin(ang), np.full(segments, zi)], 1) for zi in z]
    n = segments
    V = np.vstack(outer + inner)
    F = []
    oo = lambda i, k: i * n + k % n
    io = lambda i, k: (len(z) + i) * n + k % n
    for i in range(len(z) - 1):
        for k in range(n):
            F += [(oo(i, k), oo(i, k + 1), oo(i + 1, k + 1)), (oo(i, k), oo(i + 1, k + 1), oo(i + 1, k))]     # outside wall
            F += [(io(i, k), io(i + 1, k + 1), io(i, k + 1)), (io(i, k), io(i + 1, k), io(i + 1, k + 1))]     # threaded bore
    for i, flip in ((0, True), (len(z) - 1, False)):                 # top / bottom annulus
        for k in range(n):
            a, b, c, d = oo(i, k), oo(i, k + 1), io(i, k + 1), io(i, k)
            F += [(a, c, b), (a, d, c)] if flip else [(a, b, c), (a, c, d)]
    return orient_outward(Mesh(V, np.array(F, int)))


# ------------------------------------------------------------------------------------------------ gears
def _inv(a: float) -> float:
    return math.tan(a) - a


def gear_outline(module: float, teeth: int, pressure_deg: float = 20.0, clearance: float = 0.25, backlash: float = 0.0,
                 points_per_flank: int = 9, tip_points: int = 3) -> List[Tuple[float, float]]:
    """Involute spur-gear outline (counter-clockwise polygon). Tooth 0 points along +X.
    module m: pitch diameter = m * teeth, tip = pitch + 2m, root = pitch - 2.5m. `backlash` (mm) thins each tooth."""
    m, z = float(module), int(teeth)
    a = math.radians(pressure_deg)
    rp, rb = m * z / 2, m * z / 2 * math.cos(a)
    ra, rf = rp + m, rp - (1.0 + clearance) * m
    r0 = max(rb, rf)
    hp = math.pi / (2 * z) - backlash / (2 * rp)
    half = lambda r: hp + _inv(a) - _inv(math.acos(min(1.0, rb / r)))
    rs = np.linspace(r0, ra, points_per_flank)
    pts: List[Tuple[float, float]] = []
    for k in range(z):
        c = 2 * math.pi * k / z
        cn = 2 * math.pi * ((k + 1) % z if k + 1 < z else z) / z
        right = [(r * math.cos(c - half(r)), r * math.sin(c - half(r))) for r in rs]
        tip = [(ra * math.cos(c + t), ra * math.sin(c + t)) for t in np.linspace(-half(ra), half(ra), tip_points)[1:-1]]
        left = [(r * math.cos(c + half(r)), r * math.sin(c + half(r))) for r in rs[::-1]]
        if rf < rb:                                            # radial line down to the root circle, then along the root
            left.append((rf * math.cos(c + half(rb)), rf * math.sin(c + half(rb))))
        nxt = 2 * math.pi * (k + 1) / z
        root = [(rf * math.cos(t), rf * math.sin(t)) for t in np.linspace(c + half(r0), nxt - half(r0), 4)[1:-1]] if rf < rb else []
        pts += right + tip + left + root
        if rf < rb:
            pts.append((rf * math.cos(nxt - half(rb)), rf * math.sin(nxt - half(rb))))
    return pts


def gear(module: float, teeth: int, thickness: float, bore: float = 0.0, pressure_deg: float = 20.0, backlash: float = 0.0,
         hub_dia: float = 0.0, hub_h: float = 0.0) -> Mesh:
    """Printable spur gear with an optional centre bore (and an optional raised hub)."""
    outline = gear_outline(module, teeth, pressure_deg, 0.25, backlash)
    holes = []
    if bore > 0:
        holes.append([(bore / 2 * math.cos(2 * math.pi * k / 48), bore / 2 * math.sin(2 * math.pi * k / 48)) for k in range(48)])
    g = extrude_polygon(outline, holes, thickness)
    if hub_dia > 0 and hub_h > 0:
        hub = extrude_polygon([(hub_dia / 2 * math.cos(2 * math.pi * k / 64), hub_dia / 2 * math.sin(2 * math.pi * k / 64)) for k in range(64)],
                              holes, hub_h, z0=thickness)
        g = merge(g, hub)
    return g


# ------------------------------------------------------------------------------------------------ polygons from images
def polygons_from_mask(mask: np.ndarray, simplify_tol: float = 0.7):
    """Binary mask -> [(outer, [holes...])] in pixel units (y down). Nested shapes alternate outer / hole / outer ..."""
    from quadmesh.agent.image3d import trace_contours, simplify, _inside
    rings = [r for r in trace_contours(mask) if abs(_area(r)) > 6.0]
    rings = [simplify(r, simplify_tol) for r in rings]
    rings = [r for r in rings if len(r) >= 3]
    depth = [sum(1 for o in rings if o is not r and abs(_area(o)) > abs(_area(r)) and _inside(r[0], o)) for r in rings]
    out = []
    for r, d in zip(rings, depth):
        if d % 2 == 0:
            holes = [h for h, dh in zip(rings, depth) if dh == d + 1 and _inside(h[0], r)]
            out.append((r, holes))
    return out


def text3d(text: str, height_mm: float = 12.0, depth_mm: float = 3.0, font_path: Optional[str] = None) -> Mesh:
    """Raised 3-D text, one solid per letter (letters like O, A, B keep their holes). Uses a TrueType font if one is found."""
    from PIL import Image, ImageDraw, ImageFont
    size = 240
    font = None
    for cand in ([font_path] if font_path else []) + ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "arial.ttf", "DejaVuSans.ttf",
                                                       "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            font = ImageFont.truetype(cand, size); break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default(size=size) if hasattr(ImageFont, "load_default") else ImageFont.load_default()
    probe = Image.new("L", (10, 10)); bb = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    im = Image.new("L", (bb[2] - bb[0] + 40, bb[3] - bb[1] + 40), 255)
    ImageDraw.Draw(im).text((20 - bb[0], 20 - bb[1]), text, fill=0, font=font)
    mask = np.asarray(im) < 128
    polys = polygons_from_mask(mask, 1.2)
    if not polys:
        raise ValueError("no text could be rendered")
    ys = [p[1] for o, _ in polys for p in o]
    s = height_mm / (max(ys) - min(ys))
    y0 = max(ys)
    tf = lambda ring: [(p[0] * s, (y0 - p[1]) * s) for p in ring]
    return merge(*[extrude_polygon(tf(o), [tf(h) for h in hs], depth_mm) for o, hs in polys])


# ------------------------------------------------------------------------------------------------ inflate a silhouette
def _edt(mask: np.ndarray) -> np.ndarray:
    """Distance (in cells) from each True cell to the nearest False cell."""
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(mask)
    except Exception:
        pass
    INF = 1e12

    def d1(f):
        n = len(f); v = [0] * n; z = [0.0] * (n + 1); k = 0; out = np.empty(n)
        v[0] = 0; z[0] = -INF; z[1] = INF
        for q in range(1, n):
            while True:
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
                if s <= z[k] and k > 0:
                    k -= 1
                else:
                    break
            if s <= z[k]:
                k = 0; v[0] = q; z[0] = -INF; z[1] = INF
            else:
                k += 1; v[k] = q; z[k] = s; z[k + 1] = INF
        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1
            out[q] = (q - v[k]) ** 2 + f[v[k]]
        return out
    f = np.where(mask, INF, 0.0)
    f = np.apply_along_axis(d1, 0, f)
    f = np.apply_along_axis(d1, 1, f)
    return np.sqrt(f)


def inflate_mask(mask: np.ndarray, width_mm: float, thickness_mm: float, roundness: float = 0.75,
                 min_frac: float = 0.25, max_cells: int = 150) -> Mesh:
    """Silhouette -> puffy solid: thickness grows with the distance from the edge (a pillow). The classic 'inflation' guess
    from a single outline. Watertight by construction; the edge is stair-stepped at the grid resolution (max_cells)."""
    ys, xs = np.nonzero(mask)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    step = max(1, int(math.ceil(max(m.shape) / max_cells)))
    if step > 1:
        m = m[::step, ::step]
    m = np.flipud(m)                                            # rows now run upward (y up)
    m = np.pad(m, 1)
    d = _edt(m)
    dmax = max(d.max(), 1.0)
    rel = np.clip(d / (roundness * dmax + 1e-9), 0, 1)
    hc = (thickness_mm / 2) * (min_frac + (1 - min_frac) * np.sqrt(1 - (1 - rel) ** 2))     # half-height per cell
    R, C = m.shape
    dx = width_mm / (m.shape[1] - 2)
    hv = np.zeros((R + 1, C + 1)); cnt = np.zeros((R + 1, C + 1))
    for dr in (0, 1):
        for dc in (0, 1):
            hv[dr:R + dr, dc:C + dc] += np.where(m, hc, 0); cnt[dr:R + dr, dc:C + dc] += m
    hv = np.where(cnt > 0, hv / np.maximum(cnt, 1), 0)
    ids_top, ids_bot, V = {}, {}, []
    def vid(r, c, top):
        store = ids_top if top else ids_bot
        if (r, c) not in store:
            store[(r, c)] = len(V)
            V.append([(c - 1) * dx, (r - 1) * dx, (thickness_mm / 2 + hv[r, c]) if top else (thickness_mm / 2 - hv[r, c])])
        return store[(r, c)]
    F = []
    for r in range(R):
        for c in range(C):
            if not m[r, c]:
                continue
            t = [vid(r, c, 1), vid(r, c + 1, 1), vid(r + 1, c + 1, 1), vid(r + 1, c, 1)]
            b = [vid(r, c, 0), vid(r, c + 1, 0), vid(r + 1, c + 1, 0), vid(r + 1, c, 0)]
            F += [(t[0], t[1], t[2]), (t[0], t[2], t[3]), (b[0], b[2], b[1]), (b[0], b[3], b[2])]
            for (nr, nc, ia, ib) in ((r - 1, c, 0, 1), (r, c + 1, 1, 2), (r + 1, c, 2, 3), (r, c - 1, 3, 0)):
                if not (0 <= nr < R and 0 <= nc < C and m[nr, nc]):
                    F += [(b[ia], b[ib], t[ib]), (b[ia], t[ib], t[ia])]
    return orient_outward(Mesh(np.array(V, float), np.array(F, int)))
