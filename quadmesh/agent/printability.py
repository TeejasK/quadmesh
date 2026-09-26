"""
Design printability gate (FDM). Two ways to get the same report, one judge:

  * Blender side:  bridge.call("print_check", ...)   (see blender_features.py)
  * File side:     analyze_stl("part.stl")           (numpy only; checks the exact file you will slice)

judge_report(report, ...) turns a report into (problems, warnings). A design is only called
"printable" when problems == [].  Warnings never block, but are shown.

Command line:  python -m quadmesh.agent.printability part.stl --bed 220 220 250
"""
from __future__ import annotations
import math
import re
import struct
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Limits:
    bed: tuple = (220.0, 220.0, 250.0)   # x, y, z build volume in mm
    min_wall: float = 0.8                # mm (two 0.4 mm perimeters)
    max_overhang_pct: float = 15.0       # % of surface steeper than 45 degrees (needs supports)
    allow_supports: bool = False         # True: overhang becomes a warning instead of a problem
    expected_islands: Optional[int] = 1  # separate solids the design should contain (None = any, e.g. text)
    bbox_tol: float = 0.6                # mm, real size vs size predicted from the plan
    sit_on_bed_tol: float = 0.05         # mm, lowest point must be at z = 0


# ---------------------------------------------------------------------------
# The judge (shared by Blender reports and STL reports)
# ---------------------------------------------------------------------------
def judge_report(report: dict, predicted_bbox: Optional[tuple] = None, limits: Limits = Limits()):
    problems, warnings = [], []
    if report.get("objects", 1) != 1:
        problems.append(f"scene has {report['objects']} objects, expected 1")
    if limits.expected_islands is not None and report.get("loose_parts", 1) != limits.expected_islands:
        problems.append(f"{report['loose_parts']} separate solids, expected {limits.expected_islands}")
    if report.get("open_edges", 0) > 0:
        problems.append(f"{report['open_edges']} open edges (mesh is not watertight)")
    if report.get("nonmanifold_edges", 0) > 0:
        problems.append(f"{report['nonmanifold_edges']} non-manifold edges")
    if report.get("flipped_edges", 0) > 0:
        problems.append(f"{report['flipped_edges']} edges with inconsistent face direction")
    if report.get("degenerate_faces", 0) > 0:
        warnings.append(f"{report['degenerate_faces']} zero-area faces")
    vol = report.get("volume", 0.0)
    if vol <= 0:
        problems.append("volume is zero or negative (faces point inward or the mesh is empty)")
    dims = report.get("dims")
    if dims:
        if dims[2] > limits.bed[2] + 1e-6 or not (
                (dims[0] <= limits.bed[0] and dims[1] <= limits.bed[1]) or
                (dims[1] <= limits.bed[0] and dims[0] <= limits.bed[1])):
            problems.append(f"size {dims[0]:.1f}x{dims[1]:.1f}x{dims[2]:.1f} mm does not fit the "
                            f"{limits.bed[0]:.0f}x{limits.bed[1]:.0f}x{limits.bed[2]:.0f} mm bed")
    if abs(report.get("zmin", 0.0)) > limits.sit_on_bed_tol:
        problems.append(f"lowest point is at z={report['zmin']:.2f} mm, the part must sit on z=0")
    mw = report.get("min_wall")
    if mw is not None and mw < limits.min_wall - 1e-6:
        problems.append(f"thinnest wall is {mw:.2f} mm, below the {limits.min_wall} mm minimum")
    oh = report.get("overhang_pct")
    if oh is not None and oh > limits.max_overhang_pct:
        msg = f"{oh:.1f}% of the surface overhangs more than 45 degrees (limit {limits.max_overhang_pct}%)"
        (warnings if limits.allow_supports else problems).append(msg + (" - supports needed" if limits.allow_supports else ""))
    if predicted_bbox is not None and dims:
        want = sorted([predicted_bbox[1]-predicted_bbox[0], predicted_bbox[3]-predicted_bbox[2],
                       predicted_bbox[5]-predicted_bbox[4]])
        got = sorted(dims)
        if any(abs(a-b) > limits.bbox_tol for a, b in zip(want, got)):
            problems.append("real size %s mm differs from the size the plan predicts %s mm" %
                            ([round(v, 2) for v in got], [round(v, 2) for v in want]))
    return problems, warnings


# ---------------------------------------------------------------------------
# STL file analysis (numpy)
# ---------------------------------------------------------------------------
def load_stl(path: str):
    import numpy as np
    raw = open(path, "rb").read()
    if len(raw) >= 84:
        n = struct.unpack("<I", raw[80:84])[0]
        if 84 + 50*n == len(raw):
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(raw, dtype=dt, count=n, offset=84)["v"].astype(np.float64)
    txt = raw.decode("utf-8", errors="ignore")
    nums = re.findall(r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", txt)
    if not nums:
        raise ValueError("not a valid STL file")
    return np.array(nums, dtype=np.float64).reshape(-1, 3, 3)


def analyze_mesh(tris, bed=None, samples: int = 200, seed: int = 0) -> dict:
    """tris: (N,3,3) float array of triangles. Returns a report like Blender's print_check."""
    import numpy as np
    N = len(tris)
    pts = tris.reshape(-1, 3)
    uniq, inv = np.unique(np.round(pts, 4), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    faces = inv.reshape(N, 3)
    # edges
    d = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    und = np.sort(d, axis=1)
    key = und[:, 0].astype(np.int64) * (len(uniq) + 1) + und[:, 1]
    _, counts = np.unique(key, return_counts=True)
    open_edges = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    dkey = d[:, 0].astype(np.int64) * (len(uniq) + 1) + d[:, 1]
    _, dcounts = np.unique(dkey, return_counts=True)
    flipped = int((dcounts > 1).sum())
    # geometry
    e1, e2 = tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0]
    cr = np.cross(e1, e2)
    area2 = np.linalg.norm(cr, axis=1)
    area = area2 / 2.0
    degenerate = int((area < 1e-9).sum())
    nrm = cr / np.maximum(area2[:, None], 1e-30)
    vol = float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    dims = (hi - lo).tolist()
    cz = tris[:, :, 2].mean(axis=1)
    over = (nrm[:, 2] < -0.7072) & (cz > lo[2] + 0.01)
    overhang_pct = float(100.0 * area[over].sum() / max(area.sum(), 1e-12))
    # islands (union-find on vertices)
    parent = np.arange(len(uniq))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, c in faces:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = find(ra)
    used = np.unique(faces.reshape(-1))
    islands = len({find(v) for v in used})
    # thinnest wall: cast rays inward from sampled faces (Moller-Trumbore, vectorised)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(samples, N), replace=False) if N else []
    min_wall = None
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    for i in idx:
        if area[i] < 1e-9:
            continue
        o = tris[i].mean(axis=0) - nrm[i]*0.01
        dvec = -nrm[i]
        p = np.cross(dvec, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv_det = np.where(ok, 1.0/np.where(ok, det, 1.0), 0.0)
        tv = o - v0
        u = np.einsum("ij,ij->i", tv, p) * inv_det
        q = np.cross(tv, e1)
        v = (q @ dvec) * inv_det
        t = np.einsum("ij,ij->i", e2, q) * inv_det
        hit = ok & (u >= 0) & (v >= 0) & (u+v <= 1) & (t > 1e-4)
        if hit.any():
            w = float(t[hit].min()) + 0.01
            min_wall = w if min_wall is None else min(min_wall, w)
    rep = {"objects": 1, "loose_parts": islands, "open_edges": open_edges,
           "nonmanifold_edges": nonmanifold, "flipped_edges": flipped, "degenerate_faces": degenerate,
           "volume": round(vol, 3), "dims": [round(x, 3) for x in dims], "zmin": round(float(lo[2]), 4),
           "min_wall": None if min_wall is None else round(min_wall, 3),
           "overhang_pct": round(overhang_pct, 2), "triangles": int(N)}
    return rep


def analyze_stl(path: str, **kw) -> dict:
    return analyze_mesh(load_stl(path), **kw)


def check_file(path: str, limits: Limits = Limits(), predicted_bbox=None):
    rep = analyze_stl(path)
    problems, warnings = judge_report(rep, predicted_bbox, limits)
    return rep, problems, warnings


# ---------------------------------------------------------------------------
# Item 9: learned risk score ON TOP OF judge_report, not instead of it.
#
# judge_report above is UNCHANGED - it still decides pass/fail on the things that are genuinely binary and
# physical: watertight (open_edges==0), manifold, one solid island, positive volume, sits on the bed, fits
# the build volume. Those stay hard asserts on purpose (same reasoning as item 7's check_plan): a learned
# model being confidently wrong about whether a part is a closed solid is not a UX problem, it's a failed
# print. The offline test suite (`printability gate flags every bad mesh, passes a good one`) exercises
# exactly these hard cases and this file makes no change to how any of them are judged.
#
# What WAS a rigid rule, and genuinely shouldn't have been: min_wall (0.8mm) and max_overhang_pct (15%) are
# fixed thresholds that don't actually depend on physics alone - they depend on material, nozzle diameter,
# layer height, and how much support/warping risk the person is willing to accept, none of which judge_report
# knows about. A part at min_wall=0.79mm on a 0.4mm-nozzle PETG print is not meaningfully riskier than one at
# 0.81mm, but the hard threshold treats them as pass/fail opposites. That's the genuinely learnable part:
# HOW risky a borderline wall/overhang value is, not WHETHER the mesh is a valid solid.
#
# risk_score() is purely additive - judge_report's problems/warnings lists are never touched by it. It gives
# a continuous score for the two threshold-based checks so a caller (e.g. rank_candidates' scorer, or a UI
# showing "borderline, printable with care" instead of a flat pass/fail) can use it. No learned model exists
# yet, so the default is a transparent sigmoid-around-the-threshold heuristic - anyone training a real risk
# model can drop it in as `scorer` with the same signature.
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _default_risk(report: dict, limits: Limits) -> dict:
    """0 = no risk, 1 = certain to fail in practice. Margin-based, not a hard cutoff - a value just past a
    threshold scores near 0.5, not 1.0, unlike judge_report's binary problems/warnings split."""
    out = {}
    mw = report.get("min_wall")
    if mw is not None:
        # margin in units of the limit itself, so it scales sensibly whether min_wall is 0.4mm or 4mm
        out["wall_risk"] = _sigmoid((limits.min_wall - mw) / max(limits.min_wall * 0.25, 1e-6))
    oh = report.get("overhang_pct")
    if oh is not None:
        out["overhang_risk"] = _sigmoid((oh - limits.max_overhang_pct) / max(limits.max_overhang_pct * 0.5, 1e-6))
    return out


def risk_score(report: dict, limits: Limits = Limits(), scorer=None) -> dict:
    """{"wall_risk": 0..1, "overhang_risk": 0..1, "overall": 0..1}. `scorer(report, limits) -> dict` swaps in
    a trained model once one exists; must return the same key shape. Does NOT replace judge_report - call
    both; judge_report still decides pass/fail, this decides how worried to be about a pass that was close."""
    parts = (scorer or _default_risk)(report, limits)
    if parts:
        parts["overall"] = max(parts.values())
    return parts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Printability check of an STL file")
    ap.add_argument("stl")
    ap.add_argument("--bed", type=float, nargs=3, default=[220, 220, 250])
    ap.add_argument("--min-wall", type=float, default=0.8)
    ap.add_argument("--supports", action="store_true", help="allow overhangs (supports will be used)")
    a = ap.parse_args()
    lim = Limits(bed=tuple(a.bed), min_wall=a.min_wall, allow_supports=a.supports)
    rep, probs, warns = check_file(a.stl, lim)
    print(rep)
    for w in warns:
        print("WARNING:", w)
    for p in probs:
        print("PROBLEM:", p)
    print("PRINTABLE" if not probs else "NOT PRINTABLE")
