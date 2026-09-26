"""
Image -> 3D solid (deterministic, no training needed).

A drawing, silhouette, logo, stencil or sketch becomes an extruded solid: the dark shape is traced,
holes inside it stay holes, and it is scaled to the width you give. Then it goes through the same
printability gate as every other design.

    from quadmesh.agent.image3d import design_from_image
    design_from_image("bracket.png", width_mm=80, thickness_mm=5)

Works best on high-contrast images (black shape on white, or a transparent PNG). It does NOT
understand photos of 3D objects - a photo has no clean outline and no depth.
"""
from __future__ import annotations
import math
import os
from typing import Optional


# ---------------------------------------------------------------------------
# 1. image -> binary mask
# ---------------------------------------------------------------------------
def load_mask(path: str, threshold: Optional[float] = None, invert: Optional[bool] = None,
              max_side: int = 700):
    import numpy as np
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        alpha = np.asarray(rgba.split()[-1], dtype=np.float64)
        if alpha.min() < 250:                        # transparent background: the shape is the opaque part
            g = 255.0 - alpha
            invert_default = False
        else:
            g = np.asarray(im.convert("L"), dtype=np.float64); invert_default = None
    else:
        g = np.asarray(im.convert("L"), dtype=np.float64); invert_default = None
    h, w = g.shape
    if max(h, w) > max_side:                         # keep the tracer fast
        s = max_side / max(h, w)
        g = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((max(1, int(w*s)), max(1, int(h*s))),
                                                                   Image.BILINEAR), dtype=np.float64)
    t = threshold if threshold is not None else _otsu(g)
    mask = g < t                                     # dark = shape
    if invert is None:
        invert = bool(mask[0, :].mean()*0.25 + mask[-1, :].mean()*0.25 + mask[:, 0].mean()*0.25
                      + mask[:, -1].mean()*0.25 > 0.5) if invert_default is None else invert_default
    if invert:
        mask = ~mask
    return mask


def _otsu(g) -> float:
    import numpy as np
    hist, _ = np.histogram(g, bins=256, range=(0, 256))
    total = g.size
    sum_all = float((np.arange(256) * hist).sum())
    wb = sb = 0.0
    best, thr = -1.0, 128.0
    for i in range(256):
        wb += hist[i]
        if wb == 0:
            continue
        wf = total - wb
        if wf == 0:
            break
        sb += i * hist[i]
        mb, mf = sb / wb, (sum_all - sb) / wf
        var = wb * wf * (mb - mf) ** 2
        if var > best:
            best, thr = var, i + 0.5
    return thr


# ---------------------------------------------------------------------------
# 2. mask -> closed contours (marching squares) -> simplified polygons
# ---------------------------------------------------------------------------
def trace_contours(mask):
    """Closed polylines around every region of True cells. Points are (x, y) in pixel units."""
    import numpy as np
    m = np.pad(mask.astype(np.uint8), 1)
    H, W = m.shape
    # each cell (i, j) has corners tl, tr, br, bl; edge midpoints named by side
    segs = {}

    def add(p, q):
        segs.setdefault(p, []).append(q)
        segs.setdefault(q, []).append(p)

    tl, tr, br, bl = m[:-1, :-1], m[:-1, 1:], m[1:, 1:], m[1:, :-1]
    code = tl * 8 + tr * 4 + br * 2 + bl
    ys, xs = np.nonzero((code > 0) & (code < 15))
    for i, j in zip(ys, xs):
        c = code[i, j]
        T, R, B, L = (j + 0.5, float(i)), (j + 1.0, i + 0.5), (j + 0.5, i + 1.0), (float(j), i + 0.5)
        table = {1: [(L, B)], 2: [(B, R)], 3: [(L, R)], 4: [(T, R)], 5: [(T, L), (B, R)], 6: [(T, B)],
                 7: [(T, L)], 8: [(T, L)], 9: [(T, B)], 10: [(T, R), (L, B)], 11: [(T, R)], 12: [(L, R)],
                 13: [(B, R)], 14: [(L, B)]}
        for p, q in table[int(c)]:
            add(p, q)
    contours, used = [], set()
    for start in list(segs):
        if start in used:
            continue
        ring, prev, cur = [start], None, start
        used.add(start)
        while True:
            nxt = [n for n in segs[cur] if n != prev]
            if not nxt:
                break
            cand = [n for n in nxt if n not in used] or nxt
            n = cand[0]
            if n == start:
                break
            if n in used:
                break
            ring.append(n); used.add(n); prev, cur = cur, n
        if len(ring) >= 4:
            contours.append([(x - 1.0, y - 1.0) for x, y in ring])      # undo the padding
    return contours


def _area(poly) -> float:
    return 0.5 * sum(poly[i][0]*poly[(i+1) % len(poly)][1] - poly[(i+1) % len(poly)][0]*poly[i][1]
                     for i in range(len(poly)))


def _inside(pt, poly) -> bool:
    x, y, c = pt[0], pt[1], False
    for i in range(len(poly)):
        x1, y1, x2, y2 = poly[i][0], poly[i][1], poly[(i+1) % len(poly)][0], poly[(i+1) % len(poly)][1]
        if (y1 > y) != (y2 > y) and x < (x2-x1)*(y-y1)/(y2-y1+1e-30) + x1:
            c = not c
    return c


def simplify(poly, tol: float):
    """Douglas-Peucker on a closed polygon."""
    if len(poly) < 8:
        return poly
    i0 = 0
    i1 = max(range(len(poly)), key=lambda k: (poly[k][0]-poly[0][0])**2 + (poly[k][1]-poly[0][1])**2)

    def dp(pts):
        if len(pts) < 3:
            return pts
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        dx, dy = x2-x1, y2-y1
        L = math.hypot(dx, dy) or 1e-12
        idx, dmax = 0, -1.0
        for k in range(1, len(pts)-1):
            d = abs(dy*(pts[k][0]-x1) - dx*(pts[k][1]-y1)) / L
            if d > dmax:
                idx, dmax = k, d
        if dmax > tol:
            return dp(pts[:idx+1])[:-1] + dp(pts[idx:])
        return [pts[0], pts[-1]]
    a = dp(poly[i0:i1+1]); b = dp(poly[i1:] + [poly[0]])
    return a[:-1] + b[:-1]


def outline_from_image(path: str, width_mm: float, thickness_mm: float, depth_mm: Optional[float] = None,
                       name: str = "body", threshold: Optional[float] = None, invert: Optional[bool] = None,
                       max_points: int = 900, min_hole_frac: float = 0.0005) -> dict:
    """Plan step data for one extruded solid: {"name","outer","holes","h"} in millimetres, centred on 0,0."""
    mask = load_mask(path, threshold, invert)
    if not mask.any() or mask.all():
        raise ValueError("could not find a shape: the image needs a clear dark shape on a light background")
    rings = trace_contours(mask)
    if not rings:
        raise ValueError("no outline found")
    rings = [r for r in rings if abs(_area(r)) > 4.0]
    outers = [r for r in rings if not any(_inside(r[0], o) for o in rings if o is not r and abs(_area(o)) > abs(_area(r)))]
    outer = max(outers, key=lambda r: abs(_area(r)))
    total = abs(_area(outer))
    holes = [r for r in rings if r is not outer and _inside(r[0], outer) and abs(_area(r)) >= min_hole_frac * total]
    # simplify until the point budget is met
    tol = 0.6
    while True:
        o2 = simplify(outer, tol)
        h2 = [simplify(h, tol) for h in holes]
        if len(o2) + sum(len(h) for h in h2) <= max_points or tol > 20:
            break
        tol *= 1.4
    xs = [p[0] for p in o2]; ys = [p[1] for p in o2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = width_mm / (x1 - x0)
    sy = (depth_mm / (y1 - y0)) if depth_mm else sx           # keep the aspect ratio unless depth is given

    def tf(p):                                                # image y points down -> flip; centre bbox on 0,0
        return [float(round((p[0] - (x0+x1)/2) * sx, 2)), float(round(-(p[1] - (y0+y1)/2) * sy, 2))]
    O = [tf(p) for p in o2]
    Hs = [[tf(p) for p in h] for h in h2 if len(h) >= 3]
    if _area(O) < 0:                                          # outer ring counter-clockwise
        O.reverse()
    Hs = [h if _area(h) < 0 else h[::-1] for h in Hs]         # holes clockwise
    return {"name": name, "outer": O, "holes": Hs, "h": round(thickness_mm, 3)}


# ---------------------------------------------------------------------------
# 3. build + verify (same gate as every other design)
# ---------------------------------------------------------------------------
def design_from_image(path: str, width_mm: float, thickness_mm: float, depth_mm: Optional[float] = None,
                      limits=None, out_dir: str = "designs", desk: bool = False, watch: bool = True,
                      bridge=None, desk_io=None, name: Optional[str] = None, **trace_kw):
    """Trace the image, build the solid in Blender, check printability, export the STL."""
    from quadmesh.agent.printability import Limits
    from quadmesh.agent.autopilot import design
    limits = limits or Limits()
    step = {"op": "extrude_polygons", "args": outline_from_image(path, width_mm, thickness_mm, depth_mm, **trace_kw)}
    nm = name or os.path.splitext(os.path.basename(path))[0]
    return design(f"image {nm}", attempts=1, limits=limits, out_dir=out_dir, watch=watch, desk=desk, desk_io=desk_io,
                  propose_fn=lambda p, **kw: [step], bridge=bridge, name=nm)
