"""
Engineering plan generator (v3) - a CATALOG of part families for task_planner.

A "family" = (body archetype) + (a set of features). 14 body archetypes and ~50
feature variants combine into thousands of structurally different designs; every
family accepts ANY dimensions (sampled log-uniformly, so tiny and huge parts both
appear). Run `python -m quadmesh.pipeline.datasets.engineering_gen --count` to have
the code enumerate and count the families that really produce valid plans.

Design rules that make a small model able to learn this:
  * every number in the prompt is copied unchanged into the plan (no arithmetic);
  * Blender-side feature ops (holes_corners, bolt_circle, pocket_rect, ...) do the maths;
  * every example is validated by plan_checker before it is kept.

Add a family: add a Body (new archetype) or a Feature (new op) - see the tables below.
"""
from __future__ import annotations
import itertools
import json
import math
import random
import re
from typing import Callable, Optional

from quadmesh.pipeline.plan_checker import check_plan

# --------------------------------------------------------------------------- helpers
HOLES = [2.5, 3, 3.2, 3.5, 4, 4.2, 4.5, 5, 5.5, 6, 6.5, 8, 8.5, 10, 12, 14, 16, 20, 25]
VERBS = ["Make", "Create", "Design", "Model", "Build", "Generate", "Construct"]


def f(v) -> str:
    v = round(float(v), 2)
    return str(int(v)) if v == int(v) else str(v)


def _clean(x):
    """Integral floats -> ints (20.0 -> 20) so plan numbers match the prompt text exactly."""
    if isinstance(x, float):
        x = round(x, 3)
        return int(x) if x == int(x) else x
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def _an_num(txt: str) -> bool:
    i = int(float(txt))
    return i in (8, 11, 18) or 80 <= i <= 89


def fix_articles(s: str) -> str:
    """a/an agreement, so prompts read naturally ('an 8 mm', 'a U-channel', 'an L-bracket')."""
    s = re.sub(r"\b(?:a|an) (\d+(?:\.\d+)?)", lambda m: f"{'an' if _an_num(m.group(1)) else 'a'} {m.group(1)}", s)

    def word(m):
        w = m.group(1)
        vowel = w[0].lower() in "aeiou" and not w.startswith("U-") or w.startswith(("L-", "L "))
        return f"{'an' if vowel else 'a'} {w}"
    return re.sub(r"\b(?:a|an) ([A-Za-z][\w-]*)", word, s)


def S(op, **args):
    return {"op": op, "args": args}


def U(a, b):
    return random.uniform(a, b)


def rnd(lo, hi, step=1.0):
    n = int(round((hi - lo) / step))
    return round(lo + random.randint(0, max(n, 0)) * step, 3)


def logs(lo, hi, step=1.0):
    """Log-uniform sample (small parts as likely as big ones), snapped to `step`."""
    v = math.exp(random.uniform(math.log(lo), math.log(hi)))
    return max(lo, min(hi, round(round(v / step) * step, 3)))


def pick_dia(maxd, minimum=HOLES[0]):
    ok = [c for c in HOLES if minimum <= c <= maxd]
    return random.choice(ok) if ok else None


# --------------------------------------------------------------------------- bodies
# A body is a dict:
#   id, fp ('rect'|'round'), W,D,H | DIA,H (base part), minspan, thin,
#   phrase, base (steps creating "body"), join (steps that add+union more parts),
#   flags: center_ok, perim_ok, boss_ok, edge_ok, post (bool), core (span for a through bore)
BODIES: dict[str, Callable] = {}
BODY_WEIGHT: dict[str, float] = {}


def body(name, weight=1.0):
    def deco(fn):
        BODIES[name] = fn
        BODY_WEIGHT[name] = weight
        return fn
    return deco


def _rect(id_, W, D, H, phrase, **kw):
    b = dict(id=id_, fp="rect", W=W, D=D, H=H, minspan=min(W, D), thin=min(W, D, H), phrase=phrase,
             base=[S("add_box", name="body", w=W, d=D, h=H)], join=[], center_ok=True, perim_ok=True,
             boss_ok=True, edge_ok=True, post=False, core=min(W, D))
    b.update(kw)
    return b


def _round(id_, DIA, H, phrase, **kw):
    b = dict(id=id_, fp="round", DIA=DIA, H=H, minspan=DIA, thin=min(DIA, H), phrase=phrase,
             base=[S("add_cyl", name="body", dia=DIA, h=H)], join=[], center_ok=True, perim_ok=True,
             boss_ok=True, edge_ok=True, post=False, core=DIA)
    b.update(kw)
    return b


@body("plate", 3)
def _plate():
    W, D, H = logs(30, 220, 5), logs(20, 160, 5), rnd(2, 12)
    noun = random.choice(["plate", "panel", "base plate", "mounting plate", "cover plate", "sheet"])
    return _rect("plate", W, D, H, f"a {f(W)}x{f(D)}x{f(H)} mm {noun}")


@body("block", 2)
def _block():
    W, D, H = logs(15, 140, 5), logs(15, 120, 5), logs(15, 70, 5)
    noun = random.choice(["block", "mounting block", "pillow block", "spacer block", "billet"])
    return _rect("block", W, D, H, f"a {f(W)}x{f(D)}x{f(H)} mm {noun}")


@body("bar", 1.5)
def _bar():
    W, D, H = logs(80, 260, 5), rnd(8, 30), rnd(3, 20)
    noun = random.choice(["bar", "strap", "flat bar", "rail", "tab"])
    return _rect("bar", W, D, H, f"a {f(W)}x{f(D)}x{f(H)} mm {noun}")


@body("disc", 2)
def _disc():
    DIA, H = logs(25, 220, 5), rnd(2, 14)
    noun = random.choice(["disc", "round plate", "puck", "flange blank", "washer blank"])
    return _round("disc", DIA, H, f"a {f(DIA)} mm diameter, {f(H)} mm thick {noun}")


@body("cylinder", 2)
def _cylinder():
    DIA, H = logs(10, 110, 1), logs(15, 120, 5)
    noun = random.choice(["cylinder", "post", "hub", "round bar section", "spacer"])
    return _round("cylinder", DIA, H, f"a {f(DIA)} mm diameter, {f(H)} mm tall {noun}")


@body("frustum", 1)
def _frustum():
    d1 = logs(25, 120, 1); d2 = rnd(10, max(11, d1 - 6)); H = logs(10, 90, 1)
    b = dict(id="frustum", fp="round", DIA=d2, H=H, minspan=d2, thin=min(d2, H),
             phrase=f"a cone frustum, {f(d1)} mm at the base, {f(d2)} mm at the top and {f(H)} mm tall",
             base=[S("add_cone", name="body", dia1=d1, dia2=d2, h=H)], join=[], center_ok=True,
             perim_ok=False, boss_ok=False, edge_ok=True, post=False, core=d2)
    return b


@body("l_bracket", 2.5)
def _l_bracket():
    W, D, H, t = logs(30, 120, 5), logs(30, 100, 5), logs(20, 110, 5), rnd(2, 8)
    noun = random.choice(["L-bracket", "angle bracket", "corner bracket", "L bracket"])
    return _rect("l_bracket", W, D, t,
                 f"an {noun}: base {f(W)}x{f(D)} mm, upright {f(H)} mm tall, {f(t)} mm thick",
                 join=[S("add_box", name="wall", w=W, d=t, h=H), S("align_back", name="wall", ref="body"),
                       S("union", target="body", tool="wall")], thin=t)


@body("t_bracket", 1.5)
def _t_bracket():
    W, D, t, tt, H = logs(40, 160, 5), logs(30, 100, 5), rnd(2, 8), rnd(3, 10), logs(15, 80, 5)
    return _rect("t_bracket", W, D, t,
                 f"a T-bracket: base {f(W)}x{f(D)} mm, {f(t)} mm thick, with a {f(tt)} mm thick rib "
                 f"{f(H)} mm tall along the center",
                 join=[S("add_box", name="rib", w=W, d=tt, h=H), S("stack_on", name="rib", ref="body"),
                       S("union", target="body", tool="rib")],
                 center_ok=False, boss_ok=False, thin=min(t, tt))


@body("u_channel", 1.5)
def _u_channel():
    W, D, t, tw, H = logs(40, 160, 5), logs(30, 140, 5), rnd(2, 8), rnd(2, 8), logs(15, 70, 5)
    return _rect("u_channel", W, D, t,
                 f"a U-channel: {f(W)}x{f(D)} mm base, {f(t)} mm thick, with two {f(tw)} mm thick walls "
                 f"{f(H)} mm tall",
                 join=[S("add_box", name="wall1", w=tw, d=D, h=H), S("align_side", name="wall1", ref="body", side="left"),
                       S("union", target="body", tool="wall1"),
                       S("add_box", name="wall2", w=tw, d=D, h=H), S("align_side", name="wall2", ref="body", side="right"),
                       S("union", target="body", tool="wall2")], thin=min(t, tw))


@body("stepped_block", 2)
def _stepped_block():
    W, D, H1 = logs(40, 160, 5), logs(40, 130, 5), rnd(4, 30)
    w2, d2, H2 = rnd(15, max(16, W - 20), 5), rnd(15, max(16, D - 20), 5), logs(8, 60, 2)
    return _rect("stepped_block", W, D, H1,
                 f"a stepped block: {f(W)}x{f(D)}x{f(H1)} mm base with a {f(w2)}x{f(d2)}x{f(H2)} mm block on top",
                 join=[S("add_box", name="top", w=w2, d=d2, h=H2), S("stack_on", name="top", ref="body"),
                       S("union", target="body", tool="top")],
                 center_ok=False, boss_ok=False, post=True, core=min(w2, d2))


@body("stepped_round", 2)
def _stepped_round():
    D1, H1 = logs(30, 160, 5), rnd(4, 30)
    D2, H2 = rnd(12, max(13, D1 - 14), 2), logs(8, 70, 2)
    return _round("stepped_round", D1, H1,
                  f"a stepped cylinder: {f(D1)} mm diameter and {f(H1)} mm tall, then {f(D2)} mm diameter "
                  f"and {f(H2)} mm tall",
                  join=[S("add_cyl", name="top", dia=D2, h=H2), S("stack_on", name="top", ref="body"),
                        S("union", target="body", tool="top")],
                  center_ok=False, boss_ok=False, post=True, core=D2)


@body("block_on_disc", 1)
def _block_on_disc():
    D1, H1 = logs(50, 180, 5), rnd(4, 20)
    w2, d2 = rnd(15, max(16, D1 * 0.5), 5), rnd(15, max(16, D1 * 0.5), 5); H2 = logs(8, 60, 2)
    return _round("block_on_disc", D1, H1,
                  f"a {f(D1)} mm diameter, {f(H1)} mm thick disc with a {f(w2)}x{f(d2)}x{f(H2)} mm block on top",
                  join=[S("add_box", name="top", w=w2, d=d2, h=H2), S("stack_on", name="top", ref="body"),
                        S("union", target="body", tool="top")],
                  center_ok=False, boss_ok=False, post=True, core=min(w2, d2))


@body("post_on_block", 1)
def _post_on_block():
    W, D, H1 = logs(40, 160, 5), logs(40, 130, 5), rnd(4, 25)
    D2, H2 = rnd(10, max(11, min(W, D) - 15), 2), logs(10, 80, 2)
    return _rect("post_on_block", W, D, H1,
                 f"a {f(W)}x{f(D)}x{f(H1)} mm base with a {f(D2)} mm diameter, {f(H2)} mm tall post on top",
                 join=[S("add_cyl", name="top", dia=D2, h=H2), S("stack_on", name="top", ref="body"),
                       S("union", target="body", tool="top")],
                 center_ok=False, boss_ok=False, post=True, core=D2)


@body("enclosure", 2)
def _enclosure():
    W, D, H, wall = logs(30, 220, 5), logs(30, 160, 5), logs(15, 90, 5), rnd(1.5, 5, 0.5)
    noun = random.choice(["enclosure", "box", "housing", "case", "open-top box", "tray"])
    return _rect("enclosure", W, D, H, f"an {noun}, {f(W)}x{f(D)}x{f(H)} mm outside, {f(wall)} mm walls, open at the top",
                 join=[], center_ok=False, perim_ok=False, boss_ok=False, shell=wall, thin=wall)


def _prism_body(n, label):
    DIA, H = logs(20, 160, 2), logs(4, 80, 2)
    ins = round(DIA * math.cos(math.pi / n), 2)
    return _round(f"prism{n}", ins, H, f"a {label} prism, {f(DIA)} mm across corners and {f(H)} mm tall",
                  base=[S("add_prism", name="body", sides=n, dia=DIA, h=H)])


for _n, _label in ((6, "hexagonal"), (8, "octagonal"), (10, "decagonal"), (12, "dodecagonal")):
    BODIES[f"prism{_n}"] = (lambda n=_n, label=_label: _prism_body(n, label))
    BODY_WEIGHT[f"prism{_n}"] = 0.8


@body("sphere", 0.3)
def _sphere():
    D = logs(10, 150, 1)
    return dict(id="sphere", fp="round", DIA=D, H=D, minspan=D, thin=D, phrase=f"a {f(D)} mm diameter sphere",
                base=[S("add_sphere", name="body", dia=D)], join=[], center_ok=False, perim_ok=False,
                boss_ok=False, edge_ok=False, post=False, core=D)


@body("torus", 0.3)
def _torus():
    mn = rnd(3, 25); mj = round(mn + rnd(10, 120))
    return dict(id="torus", fp="round", DIA=mj, H=mn, minspan=mj, thin=mn,
                phrase=f"a ring (torus), {f(mj)} mm across the tube centerline, {f(mn)} mm tube diameter",
                base=[S("add_torus", name="body", major=mj, minor=mn)], join=[], center_ok=False,
                perim_ok=False, boss_ok=False, edge_ok=False, post=False, core=mj)


def _layer(kind, span, k):
    """One stacked layer smaller than `span`. Returns (phrase, steps, new_span)."""
    name = f"t{k}"
    cur = rnd(12, max(13, span - 12), 2)
    h = logs(6, 60, 2)
    if kind == "block":
        w2, d2 = cur, rnd(12, max(13, span - 12), 2)
        return (f"a {f(w2)}x{f(d2)}x{f(h)} mm block on top",
                [S("add_box", name=name, w=w2, d=d2, h=h)], min(w2, d2))
    if kind == "cyl":
        return (f"a {f(cur)} mm diameter, {f(h)} mm tall cylinder on top",
                [S("add_cyl", name=name, dia=cur, h=h)], cur)
    if kind == "hex":
        ins = round(cur * math.cos(math.pi / 6), 2)
        return (f"a {f(cur)} mm across-corners, {f(h)} mm tall hexagonal prism on top",
                [S("add_prism", name=name, sides=6, dia=cur, h=h)], ins)
    d2 = rnd(8, max(9, cur - 4), 2)
    return (f"a cone {f(cur)} mm at the base narrowing to {f(d2)} mm, {f(h)} mm tall, on top",
            [S("add_cone", name=name, dia1=cur, dia2=d2, h=h)], d2)


def _stack_body(bk, tops):
    if bk == "rect":
        W, D, H = logs(60, 200, 5), logs(50, 160, 5), rnd(4, 25)
        b = _rect("", W, D, H, f"a {f(W)}x{f(D)}x{f(H)} mm base"); span = min(W, D)
    elif bk == "disc":
        DIA, H = logs(60, 200, 5), rnd(4, 25)
        b = _round("", DIA, H, f"a {f(DIA)} mm diameter, {f(H)} mm thick base"); span = DIA
    else:
        DIA, H = logs(60, 200, 5), rnd(4, 25)
        b = _prism_body(6, "hexagonal")
        ins = round(DIA * math.cos(math.pi / 6), 2)
        b.update(DIA=ins, H=H, minspan=ins, thin=min(ins, H), core=ins,
                 phrase=f"a {f(DIA)} mm across-corners, {f(H)} mm tall hexagonal base",
                 base=[S("add_prism", name="body", sides=6, dia=DIA, h=H)]); span = ins
    phrases, join, spans = [], [], []
    for k, tk in enumerate(tops, 1):
        ph, st, span = _layer(tk, span, k)
        phrases.append(ph); spans.append(span)
        join += st + [S("stack_on", name=f"t{k}", ref="body"), S("union", target="body", tool=f"t{k}")]
    b.update(id=f"stack_{bk}_" + "_".join(tops), join=join, center_ok=False, boss_ok=False, post=True,
             core=min(spans),
             phrase=f"a stack: {b['phrase']}, " + ", then ".join(phrases))
    return b


for _bk in ("rect", "disc", "hex"):
    for _t1 in ("block", "cyl", "hex", "cone"):
        BODIES[f"stack_{_bk}_{_t1}"] = (lambda bk=_bk, t1=_t1: _stack_body(bk, (t1,)))
        BODY_WEIGHT[f"stack_{_bk}_{_t1}"] = 0.5
        for _t2 in ("block", "cyl", "hex", "cone"):
            BODIES[f"stack_{_bk}_{_t1}_{_t2}"] = (lambda bk=_bk, t1=_t1, t2=_t2: _stack_body(bk, (t1, t2)))
            BODY_WEIGHT[f"stack_{_bk}_{_t1}_{_t2}"] = 0.25


# --------------------------------------------------------------------------- features
# id -> dict(group, fps, rank, fn). fn(b) -> (phrase, steps) | None
#   groups: center (<=1), perim (<=2), edge (<=1), post (<=1)
#   rank:   0 = voids (before bosses), 1 = bosses, 3 = post bore, 4 = edge treatment
FEATURES: dict[str, dict] = {}


def feature(name, group, fps, rank=0):
    def deco(fn):
        FEATURES[name] = dict(group=group, fps=set(fps), rank=rank, fn=fn)
        return fn
    return deco


def _minspan(b):
    return b["minspan"]


def _t(op, **a):
    return S(op, target="body", **a)


@feature("c_hole", "center", ["rect", "round"])
def _c_hole(b):
    d = pick_dia(0.55 * _minspan(b))
    if d is None: return None
    return random.choice([f"a {f(d)} mm center hole", f"a {f(d)} mm hole in the center"]), [_t("hole_center", dia=d)]


@feature("c_cbore", "center", ["rect", "round"])
def _c_cbore(b):
    d = pick_dia(0.28 * _minspan(b), 3)
    if d is None: return None
    cb, cd = round(d * U(1.6, 2.2), 1), round(b["H"] * U(0.3, 0.5), 1)
    return (f"a {f(d)} mm center hole with a {f(cb)} mm counterbore {f(cd)} mm deep",
            [_t("cbore_center", dia=d, cb_dia=cb, cb_depth=cd)])


@feature("c_csink", "center", ["rect", "round"])
def _c_csink(b):
    d = pick_dia(0.4 * _minspan(b), 3)
    if d is None: return None
    cs = round(d * U(1.8, 2.3), 1)
    return f"a {f(d)} mm center hole countersunk to {f(cs)} mm", [_t("csink_center", dia=d, cs_dia=cs)]


@feature("c_slot", "center", ["rect"])
def _c_slot(b):
    L, Wd = round(b["W"] * U(0.3, 0.7)), round(b["D"] * U(0.1, 0.3))
    if Wd < 3: return None
    return f"a {f(L)} mm x {f(Wd)} mm slot in the center", [_t("slot_center", length=L, width=Wd)]


@feature("c_pocket", "center", ["rect"])
def _c_pocket(b):
    pw, pd, dp = round(b["W"] * U(0.3, 0.7)), round(b["D"] * U(0.3, 0.7)), round(b["H"] * U(0.3, 0.6), 1)
    return (f"a {f(pw)}x{f(pd)} mm pocket {f(dp)} mm deep", [_t("pocket_rect", w=pw, d=pd, depth=dp)])


@feature("c_window", "center", ["rect"])
def _c_window(b):
    ww, wd = round(b["W"] * U(0.3, 0.7)), round(b["D"] * U(0.3, 0.7))
    return f"a {f(ww)}x{f(wd)} mm window cut through", [_t("window_rect", w=ww, d=wd)]


@feature("c_ppocket", "center", ["rect", "round"])
def _c_ppocket(b):
    d, dp = round(_minspan(b) * U(0.3, 0.7)), round(b["H"] * U(0.3, 0.6), 1)
    return f"a {f(d)} mm round pocket {f(dp)} mm deep", [_t("pocket_round", dia=d, depth=dp)]


@feature("c_boss", "center", ["rect", "round"], rank=1)
def _c_boss(b):
    if not b["boss_ok"]: return None
    d, hb = round(_minspan(b) * U(0.2, 0.5)), rnd(3, 25)
    return f"a {f(d)} mm diameter boss, {f(hb)} mm tall, in the center", [_t("boss_center", dia=d, h=hb)]


@feature("c_grid", "center", ["rect"])
def _c_grid(b):
    nx, ny, p = random.randint(2, 6), random.randint(1, 5), random.choice([8, 10, 12, 15, 20, 25, 30])
    d = pick_dia(0.5 * p)
    if d is None: return None
    return (f"a {nx}x{ny} grid of {f(d)} mm holes at {f(p)} mm pitch",
            [_t("holes_grid", nx=nx, ny=ny, pitch=p, dia=d)])


def _inset(d):
    return round(d * U(1.0, 2.0) + rnd(1, 4), 1)


@feature("p_pair", "perim", ["rect"])
def _p_pair(b):
    d = pick_dia(0.4 * b["thin"] + 6); d = d or 4
    i = _inset(d)
    return f"2 holes of {f(d)} mm diameter, {f(i)} mm from each end", [_t("holes_pair_x", inset=i, dia=d)]


@feature("p_corner", "perim", ["rect"])
def _p_corner(b):
    d = pick_dia(0.3 * _minspan(b)); d = d or 4
    i = _inset(d)
    return f"4 corner holes of {f(d)} mm diameter, {f(i)} mm from the edges", [_t("holes_corners", inset=i, dia=d)]


@feature("p_cbore_pair", "perim", ["rect"])
def _p_cbore_pair(b):
    d = pick_dia(0.25 * _minspan(b), 3); d = d or 4
    cb, cd, i = round(d * U(1.6, 2.1), 1), round(b["H"] * U(0.3, 0.5), 1), _inset(d * 1.8)
    return (f"2 counterbored holes of {f(d)} mm diameter ({f(cb)} mm counterbore, {f(cd)} mm deep), "
            f"{f(i)} mm from each end", [_t("cbore_pair_x", inset=i, dia=d, cb_dia=cb, cb_depth=cd)])


@feature("p_cbore_corner", "perim", ["rect"])
def _p_cbore_corner(b):
    d = pick_dia(0.2 * _minspan(b), 3); d = d or 4
    cb, cd, i = round(d * U(1.6, 2.1), 1), round(b["H"] * U(0.3, 0.5), 1), _inset(d * 1.8)
    return (f"4 counterbored corner holes of {f(d)} mm diameter ({f(cb)} mm counterbore, {f(cd)} mm deep), "
            f"{f(i)} mm from the edges", [_t("cbore_corners", inset=i, dia=d, cb_dia=cb, cb_depth=cd)])


@feature("p_csink_pair", "perim", ["rect"])
def _p_csink_pair(b):
    d = pick_dia(0.25 * _minspan(b), 3); d = d or 4
    cs, i = round(d * U(1.8, 2.2), 1), _inset(d * 1.8)
    return (f"2 countersunk holes of {f(d)} mm diameter ({f(cs)} mm countersink), {f(i)} mm from each end",
            [_t("csink_pair_x", inset=i, dia=d, cs_dia=cs)])


@feature("p_csink_corner", "perim", ["rect"])
def _p_csink_corner(b):
    d = pick_dia(0.2 * _minspan(b), 3); d = d or 4
    cs, i = round(d * U(1.8, 2.2), 1), _inset(d * 1.8)
    return (f"4 countersunk corner holes of {f(d)} mm diameter ({f(cs)} mm countersink), "
            f"{f(i)} mm from the edges", [_t("csink_corners", inset=i, dia=d, cs_dia=cs)])


@feature("p_boss_pair", "perim", ["rect"], rank=1)
def _p_boss_pair(b):
    if not b["boss_ok"]: return None
    d, hb = rnd(4, 14), rnd(3, 20)
    i = round(d * U(1.0, 1.6) + rnd(1, 4), 1)
    return (f"2 bosses of {f(d)} mm diameter and {f(hb)} mm height, {f(i)} mm from each end",
            [_t("bosses_pair_x", inset=i, dia=d, h=hb)])


@feature("p_boss_corner", "perim", ["rect"], rank=1)
def _p_boss_corner(b):
    if not b["boss_ok"]: return None
    d, hb = rnd(4, 14), rnd(3, 20)
    i = round(d * U(1.0, 1.6) + rnd(1, 4), 1)
    return (f"4 corner bosses of {f(d)} mm diameter and {f(hb)} mm height, {f(i)} mm from the edges",
            [_t("bosses_corners", inset=i, dia=d, h=hb)])


def _bolt_args(b):
    n = random.choice([3, 4, 5, 6, 8, 10, 12])
    pcd = round(b["DIA"] * U(0.55, 0.85))
    d = pick_dia(0.22 * pcd); d = d or 4
    return n, pcd, d


@feature("p_bolt", "perim", ["round"])
def _p_bolt(b):
    n, pcd, d = _bolt_args(b)
    return (f"{n} holes of {f(d)} mm diameter on a {f(pcd)} mm bolt circle",
            [_t("bolt_circle", pcd=pcd, n=n, dia=d)])


@feature("p_cbore_bolt", "perim", ["round"])
def _p_cbore_bolt(b):
    n, pcd, d = _bolt_args(b)
    d = pick_dia(0.16 * pcd, 3) or 4
    cb, cd = round(d * U(1.6, 2.1), 1), round(b["H"] * U(0.3, 0.5), 1)
    return (f"{n} counterbored holes of {f(d)} mm diameter ({f(cb)} mm counterbore, {f(cd)} mm deep) "
            f"on a {f(pcd)} mm bolt circle", [_t("cbore_bolt_circle", pcd=pcd, n=n, dia=d, cb_dia=cb, cb_depth=cd)])


@feature("p_csink_bolt", "perim", ["round"])
def _p_csink_bolt(b):
    n, pcd, d = _bolt_args(b)
    d = pick_dia(0.16 * pcd, 3) or 4
    cs = round(d * U(1.8, 2.2), 1)
    return (f"{n} countersunk holes of {f(d)} mm diameter ({f(cs)} mm countersink) on a {f(pcd)} mm bolt circle",
            [_t("csink_bolt_circle", pcd=pcd, n=n, dia=d, cs_dia=cs)])


@feature("p_pair_y", "perim", ["rect"])
def _p_pair_y(b):
    d = pick_dia(0.3 * b["D"]) or 4
    i = _inset(d)
    return f"2 holes of {f(d)} mm diameter, {f(i)} mm from each side", [_t("holes_pair_y", inset=i, dia=d)]


@feature("p_cbore_pair_y", "perim", ["rect"])
def _p_cbore_pair_y(b):
    d = pick_dia(0.22 * b["D"], 3) or 4
    cb, cd, i = round(d * U(1.6, 2.1), 1), round(b["H"] * U(0.3, 0.5), 1), _inset(d * 1.8)
    return (f"2 counterbored holes of {f(d)} mm diameter ({f(cb)} mm counterbore, {f(cd)} mm deep), "
            f"{f(i)} mm from each side", [_t("cbore_pair_y", inset=i, dia=d, cb_dia=cb, cb_depth=cd)])


@feature("p_csink_pair_y", "perim", ["rect"])
def _p_csink_pair_y(b):
    d = pick_dia(0.22 * b["D"], 3) or 4
    cs, i = round(d * U(1.8, 2.2), 1), _inset(d * 1.8)
    return (f"2 countersunk holes of {f(d)} mm diameter ({f(cs)} mm countersink), {f(i)} mm from each side",
            [_t("csink_pair_y", inset=i, dia=d, cs_dia=cs)])


@feature("p_boss_pair_y", "perim", ["rect"], rank=1)
def _p_boss_pair_y(b):
    if not b["boss_ok"]: return None
    d, hb = rnd(4, 14), rnd(3, 20)
    i = round(d * U(1.0, 1.6) + rnd(1, 4), 1)
    return (f"2 bosses of {f(d)} mm diameter and {f(hb)} mm height, {f(i)} mm from each side",
            [_t("bosses_pair_y", inset=i, dia=d, h=hb)])


def _row(b, axis):
    span = b["W"] if axis == "x" else b["D"]
    n = random.randint(2, 7)
    d = pick_dia(0.2 * span) or 4
    room = (span - 2 * (d + 4)) / (n - 1)
    ok = [p for p in (6, 8, 10, 12, 15, 20, 25, 30, 40) if d + 2 <= p <= room]
    if not ok: return None
    p = random.choice(ok)
    where = "the length" if axis == "x" else "the width"
    return (f"a row of {n} holes of {f(d)} mm diameter at {f(p)} mm pitch along {where}",
            [_t("holes_row_x" if axis == "x" else "holes_row_y", n=n, pitch=p, dia=d)])


@feature("p_row_x", "perim", ["rect"])
def _p_row_x(b):
    return _row(b, "x")


@feature("p_row_y", "perim", ["rect"])
def _p_row_y(b):
    return _row(b, "y")


@feature("q_bore", "post", ["rect", "round"], rank=3)
def _q_bore(b):
    if not b["post"]: return None
    d = pick_dia(0.55 * b["core"])
    if d is None: return None
    return f"a {f(d)} mm hole through the whole part", [_t("hole_center", dia=d)]


@feature("e_fillet", "edge", ["rect", "round"], rank=4)
def _e_fillet(b):
    ok = [r for r in (0.5, 1, 1.5, 2) if r <= 0.3 * b["thin"]]
    if not ok: return None
    r = random.choice(ok)
    return random.choice([f"{f(r)} mm fillets on the edges", f"{f(r)} mm rounded edges"]), [_t("fillet", r=r)]


@feature("e_chamfer", "edge", ["rect", "round"], rank=4)
def _e_chamfer(b):
    ok = [r for r in (0.5, 1, 1.5, 2) if r <= 0.3 * b["thin"]]
    if not ok: return None
    r = random.choice(ok)
    return random.choice([f"{f(r)} mm chamfered edges", f"{f(r)} mm chamfers on the edges"]), [_t("chamfer", size=r)]


# --------------------------------------------------------------------------- allowed feature sets
def allowed_features(b) -> list[str]:
    out = []
    for name, ft in FEATURES.items():
        g = ft["group"]
        if b["fp"] not in ft["fps"]:
            continue
        if g == "center" and not b["center_ok"]: continue
        if g == "perim" and not b["perim_ok"]: continue
        if g == "post" and not b["post"]: continue
        if g == "edge" and not b["edge_ok"]: continue
        if b["id"] == "frustum" and g == "perim": continue
        if b["id"] == "enclosure" and g != "edge": continue
        if b["id"] == "frustum" and name in ("c_slot", "c_pocket", "c_window", "c_grid"): continue
        out.append(name)
    return out


def _combos(b):
    """Every legal feature set for this body: <=1 center, <=2 perim, <=1 edge, <=1 post."""
    names = allowed_features(b)
    by = {"center": [], "perim": [], "edge": [], "post": []}
    for n in names:
        by[FEATURES[n]["group"]].append(n)
    yield ()                                   # the plain body itself is a family too
    centers = [()] + [(c,) for c in by["center"]]
    perims = [()] + [(p,) for p in by["perim"]] + list(itertools.combinations(by["perim"], 2))
    edges = [()] + [(e,) for e in by["edge"]]
    posts = [()] + [(p,) for p in by["post"]]
    for c, p, e, q in itertools.product(centers, perims, edges, posts):
        fs = c + p + e + q
        if fs:
            yield tuple(sorted(fs))


def program(bd: dict, feats: tuple):
    """Build (prompt, plan) for one body + feature set with fresh random dimensions."""
    parts, steps_by_rank = [], []
    for n in feats:
        r = FEATURES[n]["fn"](bd)
        if r is None:
            return None
        parts.append((FEATURES[n]["rank"], r[0], r[1]))
    parts.sort(key=lambda x: x[0])
    plan = list(bd["base"])
    pre = [p for p in parts if p[0] <= 1]
    post = [p for p in parts if p[0] > 1]
    for _, _, st in pre:
        plan += st
    plan += bd["join"]
    if bd.get("shell"):
        plan.append(S("shell_open_top", target="body", wall=bd["shell"]))
    for _, _, st in post:
        plan += st
    phrases = [p[1] for p in parts]
    prompt = f"{random.choice(VERBS)} {bd['phrase']}"
    if phrases:
        random.shuffle(phrases)
        joined = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
        prompt += " with " + joined
    return fix_articles(prompt), _clean(plan)


def _export(prompt, plan, name):
    if random.random() < 0.6:
        ext = random.choice([".stl", ".stl", ".stl", ".obj", ".ply"])
        op = {".stl": "export_stl", ".obj": "export_obj", ".ply": "export_ply"}[ext]
        prompt += f", export as {name}{ext}"
        plan = plan + [S(op, filepath=f"C:/Users/Public/{name}{ext}")]
    return prompt, plan


def _random_feats(bd):
    names = allowed_features(bd)
    by = {"center": [], "perim": [], "edge": [], "post": []}
    for n in names:
        by[FEATURES[n]["group"]].append(n)
    fs = []
    if by["center"] and random.random() < 0.65: fs.append(random.choice(by["center"]))
    if by["perim"] and random.random() < 0.6:
        fs += random.sample(by["perim"], 2 if len(by["perim"]) > 1 and random.random() < 0.12 else 1)
    if by["post"] and random.random() < 0.6: fs.append(random.choice(by["post"]))
    if by["edge"] and random.random() < 0.4: fs.append(random.choice(by["edge"]))
    if not fs and names and random.random() < 0.97:
        fs.append(random.choice(names))
    return tuple(sorted(fs))


def build(seed=None):
    """One validated (prompt, plan, family_id). Retries until the checker passes."""
    if seed is not None:
        random.seed(seed)
    ids = list(BODIES)
    weights = [BODY_WEIGHT[i] for i in ids]
    for _ in range(300):
        if random.random() < CSG_FRAC:
            prompt, plan = csg()
            prompt, plan = _export(prompt, plan, "part")
            if not check_plan(plan, prompt):
                return prompt, plan, "csg"
            continue
        bid = random.choices(ids, weights=weights)[0]
        bd = BODIES[bid]()
        feats = _random_feats(bd)
        r = program(bd, feats)
        if r is None:
            continue
        prompt, plan = _export(*r, bid)
        if not check_plan(plan, prompt):
            return prompt, plan, bid + "+" + "+".join(feats)
    raise RuntimeError("could not generate a valid example")


def family_count(trials: int = 6, seed: int = 0, verbose: bool = False) -> dict:
    """Enumerate every (body, feature set); a family counts if ONE random-dimension trial is valid."""
    random.seed(seed)
    total = valid = 0
    per_body = {}
    for bid, mk in BODIES.items():
        n_ok = n_all = 0
        for feats in _combos(mk()):
            n_all += 1
            for _ in range(trials):
                r = program(mk(), feats)
                if r and not check_plan(r[1], r[0]):
                    n_ok += 1
                    break
        per_body[bid] = (n_ok, n_all)
        total += n_all; valid += n_ok
    return {"valid_families": valid, "tested_combinations": total, "per_body": per_body}


# --------------------------------------------------------------------------- free-form CSG programs
# For users who describe a part step by step ("start with a box, add a cylinder at x=10, y=5 ...").
# Every position is explicit and copied into the plan, so any arrangement of primitives is expressible.
CSG_FRAC = 0.30


def csg():
    kind = random.choice(["box", "box", "cyl", "hex"])
    if kind == "box":
        W, D, H = logs(40, 180, 5), logs(30, 140, 5), rnd(4, 30)
        steps = [S("add_box", name="body", w=W, d=D, h=H)]
        intro = f"Start with a {f(W)}x{f(D)}x{f(H)} mm box"; hx, hy = W / 2, D / 2
    elif kind == "cyl":
        DIA, H = logs(30, 180, 5), rnd(4, 40)
        steps = [S("add_cyl", name="body", dia=DIA, h=H)]
        intro = f"Start with a {f(DIA)} mm diameter, {f(H)} mm tall cylinder"; hx = hy = DIA / 2 * 0.7
    else:
        DIA, H = logs(30, 180, 2), rnd(4, 40)
        steps = [S("add_prism", name="body", sides=6, dia=DIA, h=H)]
        intro = f"Start with a {f(DIA)} mm across-corners, {f(H)} mm tall hexagonal prism"; hx = hy = DIA / 2 * 0.7
    sents = [intro]
    for k in range(random.randint(1, 5)):
        x = int(round(random.uniform(-0.6 * hx, 0.6 * hx))); y = int(round(random.uniform(-0.6 * hy, 0.6 * hy)))
        t = random.choice(["boss", "block", "hole", "hole", "window"])
        nm = f"p{k}"
        if t == "boss":
            d, h = rnd(4, max(5, min(hx, hy))), rnd(3, 30)
            steps += [S("add_cyl", name=nm, dia=d, h=h), S("move", name=nm, x=x, y=y, z=H),
                      S("union", target="body", tool=nm)]
            sents.append(f"Add a {f(d)} mm diameter, {f(h)} mm tall cylinder on top at x={x}, y={y}")
        elif t == "block":
            w, d, h = rnd(6, max(7, hx), 2), rnd(6, max(7, hy), 2), rnd(3, 30)
            steps += [S("add_box", name=nm, w=w, d=d, h=h), S("move", name=nm, x=x, y=y, z=H),
                      S("union", target="body", tool=nm)]
            sents.append(f"Add a {f(w)}x{f(d)}x{f(h)} mm block on top at x={x}, y={y}")
        elif t == "hole":
            d = pick_dia(0.4 * min(hx, hy)) or 4
            steps.append(S("hole_at", target="body", x=x, y=y, dia=d))
            sents.append(f"Drill a {f(d)} mm hole at x={x}, y={y}")
        else:
            w, d = rnd(6, max(7, hx), 2), rnd(6, max(7, hy), 2)
            steps.append(S("window_at", target="body", x=x, y=y, w=w, d=d))
            sents.append(f"Cut a {f(w)}x{f(d)} mm window through it at x={x}, y={y}")
    if random.random() < 0.35:
        r = random.choice([0.5, 1, 1.5, 2])
        op = random.choice(["fillet", "chamfer"])
        steps.append(S(op, target="body", **({"r": r} if op == "fillet" else {"size": r})))
        sents.append(f"Finish with {f(r)} mm " + ("fillets" if op == "fillet" else "chamfers") + " on the edges")
    return fix_articles(". ".join(sents)), _clean(steps)


# --------------------------------------------------------------------------- repair examples + API
def dumps(plan) -> str:
    return json.dumps(plan, separators=(",", ":"))


def _corrupt(plan: list) -> Optional[list]:
    import copy
    m = copy.deepcopy(plan)
    kind = random.choice(["rename", "drop", "big_dia", "big_inset", "string_num", "tweak"])
    idx = random.randrange(len(m))
    s = m[idx]
    if kind == "rename":
        for key in ("target", "tool", "ref", "name"):
            if key in s["args"]:
                s["args"][key] = random.choice(["part", "obj", "main"]); return m
    elif kind == "drop":
        m.pop(idx); return m
    elif kind == "big_dia" and "dia" in s["args"]:
        s["args"]["dia"] = round(s["args"]["dia"] * random.choice([4, 6, 10]), 1); return m
    elif kind == "big_inset" and "inset" in s["args"]:
        s["args"]["inset"] = round(s["args"]["inset"] * random.choice([8, 12, 20]), 1); return m
    elif kind == "string_num":
        for key, v in s["args"].items():
            if isinstance(v, (int, float)):
                s["args"][key] = str(v); return m
    elif kind == "tweak":
        for key, v in s["args"].items():
            if isinstance(v, (int, float)) and key not in ("n", "nx", "ny"):
                s["args"][key] = v + random.choice([1, 2, -1]); return m
    return None


def gen_repair_example() -> Optional[dict]:
    """REPAIR: <prompt> PLAN <bad plan> ERRORS <checker messages> FIX <good plan>."""
    prompt, good, fam = build()
    for _ in range(10):
        bad = _corrupt(good)
        if bad is None or bad == good:
            continue
        errs = check_plan(bad, prompt)
        if errs:
            return {"prompt": prompt, "recipe": fam + "+repair",
                    "text": f"REPAIR: {prompt} PLAN {dumps(bad)} ERRORS {'; '.join(errs[:2])} FIX {dumps(good)}"}
    return None


def gen_example(repair_frac: float = 0.15) -> dict:
    """{"prompt","text"} - drop-in for the old generator; ~15% are repair examples."""
    if random.random() < repair_frac:
        ex = gen_repair_example()
        if ex:
            return ex
    prompt, plan, fam = build()
    return {"prompt": prompt, "text": prompt + " " + dumps(plan), "recipe": fam}


def gen_prompts(n: int = 100, seed: int = 0) -> list:
    random.seed(seed)
    return [build()[0] for _ in range(n)]


def write_dataset(path: str, n: int = 20_000, seed: int = 0):
    import os
    random.seed(seed)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for _ in range(n):
            fh.write(json.dumps(gen_example()) + "\n")
    print(f"wrote {n} examples to {path}")


if __name__ == "__main__":
    import sys
    if "--count" in sys.argv:
        r = family_count()
        print(f"valid catalog families: {r['valid_families']}  (of {r['tested_combinations']} combinations tested)")
        print(f"body archetypes: {len(r['per_body'])}   + free-form CSG programs (unbounded variety)")
        for k, (a, b) in r["per_body"].items():
            print(f"  {k:14s} {a:5d} / {b}")
    else:
        random.seed(1)
        for _ in range(8):
            ex = gen_example()
            print(ex["recipe"], "|", ex["text"][:500], "\n")
