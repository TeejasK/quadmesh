"""Training data for the chat model. Every example is {"prompt", "answer"} (loss only on the answer):
knowledge Q&A from chat_kb, answers grounded in a CONTEXT block (reports), tool calls ("CALL {json}"), clarifying questions,
and honest refusals. The model learns WHEN to call a tool; the tool call is validated (mesh_tools.validate) before it runs."""
from __future__ import annotations
import json
import random
from typing import Optional

from quadmesh.chat_kb import FACTS
from quadmesh.mesh_tools import SCHEMAS

SYSTEM = "You are Quadmesh, a CAD and 3D-printing assistant. You answer briefly and honestly, call tools to build, and never invent numbers."
GREET = ["hi", "hello", "hey", "good morning", "hello Quadmesh"]
OFFTOPIC = ["what is the weather", "tell me a joke about football", "who won the election", "write my essay", "what is the price of bitcoin", "translate this poem"]
REFUSE = ["I can't help with that - I design and check 3D-printable parts. Ask me to build something or about printing.",
          "That's outside what I can do. I make 3D parts, check them for printing and control Blender. What would you like to build?"]


def _p(u, a, ctx="", key=None):
    """`key` = the underlying request before any wrapping ("please ..."), used to keep train/eval splits disjoint."""
    return {"prompt": (f"CONTEXT: {ctx}\n" if ctx else "") + f"USER: {u}\nASSISTANT:", "answer": " " + a, "user": u, "ctx": ctx,
            "key": key or u}


def _call(d):
    return "CALL " + json.dumps(d, separators=(",", ":"))


def _n(lo, hi, step=1):
    return round(random.choice([lo + i * step for i in range(int((hi - lo) / step) + 1)]), 3)


def fact_example():
    _, qs, a = random.choice(FACTS)
    q = random.choice(qs)
    return _p(random.choice([q, q.lower(), q + "?", "quick question: " + q, "please tell me " + q.lower()]), a)


def _say(*variants):
    return random.choice(variants)


def tool_example():
    """One user request -> the tool call that carries it out. Every number the call contains is either copied from
    the request or derived from it by a fixed engineering table (an M8 thread has a 1.25 mm pitch) - never drawn
    separately. (An earlier version drew the request's numbers and the call's numbers independently for springs,
    pipes and 3D text, so the model was trained on requests whose answers contradicted them; a test now checks it.)"""
    t = random.choice(["gear", "thread", "nut", "text", "spring", "pipe", "vase", "gear_pair", "four_bar", "assembly",
                       "part", "part", "part", "move", "rotate", "show", "photo", "step"])
    if t == "gear":
        m, z, th, b = _n(1, 4, 0.5), _n(12, 60, 2), _n(4, 12), _n(3, 10)
        u = _say(f"make a gear with module {m:g} and {z} teeth, {th:g} mm thick, {b:g} mm bore",
                 f"I need a {z} tooth spur gear, module {m:g}, thickness {th:g}, bore {b:g} mm",
                 f"spur gear: {z} teeth, module {m:g}, {th:g} mm wide, {b:g} mm hole in the middle")
        return _p(u, _call({"tool": "make_shape", "kind": "gear", "module": m, "teeth": z, "thickness": th, "bore": b}))
    if t in ("thread", "nut"):
        d = random.choice([4, 5, 6, 8, 10, 12]); pitch = {4: .7, 5: .8, 6: 1, 8: 1.25, 10: 1.5, 12: 1.75}[d]; L = _n(10, 60, 5)
        if t == "thread":
            u = _say(f"make an M{d} threaded rod {L:g} mm long", f"M{d} bolt blank, {L:g} mm long", f"I need a {L:g} mm long M{d} threaded rod")
            return _p(u, _call({"tool": "make_shape", "kind": "threaded_rod", "major_dia": d, "pitch": pitch, "length": L}))
        return _p(_say(f"make an M{d} nut", f"I need a nut for an M{d} bolt", f"M{d} hex nut please"),
                  _call({"tool": "make_shape", "kind": "nut", "major_dia": d, "pitch": pitch, "height": round(d * 0.8, 1)}))
    if t == "text":
        w = random.choice(["QUAD", "HELLO", "Room 12", "Maya", "OPEN", "3D", "LAB 4", "Nikhil", "EXIT", "Workshop"])
        h, dep = _n(15, 40, 5), _n(2, 6)
        u = _say(f"make the text {w} in 3D, {h:g} mm tall, {dep:g} mm thick", f"3D letters spelling {w}, {h:g} mm high and {dep:g} mm deep",
                 f"I want the word {w} as a solid, letters {h:g} mm tall, {dep:g} mm thick")
        return _p(u, _call({"tool": "make_shape", "kind": "text", "text": w, "height_mm": h, "depth_mm": dep}))
    if t == "spring":
        turns, cd, wd = _n(2, 12), _n(10, 40, 5), _n(1, 3, 0.5)
        u = _say(f"make a spring with {turns:g} turns, {cd:g} mm coil diameter and {wd:g} mm wire",
                 f"coil spring, {turns:g} coils, {cd:g} mm across, {wd:g} mm wire", f"I need a {cd:g} mm diameter spring with {turns:g} turns of {wd:g} mm wire")
        return _p(u, _call({"tool": "make_shape", "kind": "spring", "turns": turns, "coil_dia": cd, "wire_dia": wd}))
    if t == "pipe":
        o, L = _n(20, 60, 5), _n(30, 120, 10)
        u = _say(f"make a pipe {o:g} mm outside diameter, {o-6:g} mm inside, {L:g} mm long", f"tube {L:g} mm long, {o:g} mm OD and {o-6:g} mm ID",
                 f"I need a pipe: outer diameter {o:g}, inner diameter {o-6:g}, length {L:g} mm")
        return _p(u, _call({"tool": "make_shape", "kind": "pipe", "outer_dia": o, "inner_dia": o - 6, "length": L}))
    if t == "vase":
        h = _n(80, 200, 10)
        return _p(_say(f"make a vase {h:g} mm tall", f"a {h:g} mm tall vase please", f"vase, height {h:g} mm"),
                  _call({"tool": "make_shape", "kind": "vase", "height": h}))
    if t == "gear_pair":
        m, a, b = _n(1, 3, 0.5), _n(12, 24, 2), _n(26, 48, 2)
        u = _say(f"make two meshing gears, module {m:g}, {a} and {b} teeth", f"gear pair with {a} and {b} teeth at module {m:g}")
        return _p(u, _call({"tool": "make_shape", "kind": "gear_pair", "module": m, "z1": a, "z2": b}))
    if t == "four_bar":
        g, c, cp, r = _n(60, 200, 10), _n(15, 50, 5), _n(50, 150, 10), _n(40, 120, 10)
        u = _say(f"make a four bar linkage with a {g:g} mm ground link, {c:g} mm crank, {cp:g} mm coupler and {r:g} mm rocker",
                 f"four-bar linkage: ground {g:g}, crank {c:g}, coupler {cp:g}, rocker {r:g} (mm)")
        return _p(u, _call({"tool": "make_shape", "kind": "four_bar", "ground": g, "crank": c, "coupler": cp, "rocker": r}))
    if t == "assembly":
        kind = random.choice(["hexadrone", "hexacopter", "robotic arm with stand"]); m = round(random.uniform(0.6, 5), 1)
        req = f"make a {kind}" + (f", {m:g} kg, industrial grade" if "arm" not in kind else ", 400 mm reach, 0.5 kg payload")
        return _p(req, _call({"tool": "build_assembly", "request": req}))
    if t == "part":
        # free-text part requests come from the validated catalog generator (14 body archetypes x ~50 features,
        # plus free-form CSG), so the router sees the same wide phrasing the planner will later have to read.
        from quadmesh.pipeline.datasets.engineering_gen import build
        req = build()[0]
        u = _say(req, "please " + req[0].lower() + req[1:], "can you " + req[0].lower() + req[1:] + "?")
        return _p(u, _call({"tool": "build_part", "request": req}), key=req)
    if t == "move":
        n, dist, dr = random.choice(["body", "arm", "base", "cube"]), _n(5, 80, 5), random.choice(["left", "right", "up", "down", "back", "forward"])
        return _p(_say(f"move the {n} {dist:g} mm {dr}", f"shift {n} {dr} by {dist:g} mm", f"push the {n} {dist:g} mm to the {dr}" if dr in ("left", "right") else f"move {n} {dr} {dist:g} mm"),
                  _call({"tool": "move", "name": n, "distance": dist, "direction": dr}))
    if t == "rotate":
        n, deg = random.choice(["body", "arm", "gear"]), random.choice([15, 30, 45, 60, 90])
        return _p(_say(f"rotate {n} {deg} degrees", f"turn the {n} by {deg} degrees"), _call({"tool": "rotate", "name": n, "deg": deg}))
    if t == "show":
        return _p(random.choice(["show me all sides of it", "turn it around 360", "let me see it from every angle"]), _call({"tool": "show_360"}))
    if t == "photo":
        w = _n(40, 120, 10)
        return _p(f"here is a photo of a vase, make it in 3D, {w:g} mm wide", _call({"tool": "photo_to_3d", "width_mm": w}), "IMAGE: attached photo")
    return _p(_say("save the last part as a STEP file", "export it as STEP", "give me a STEP file of that"), _call({"tool": "export_step"}))


def clarify_example():
    what = random.choice([("gear", "How many teeth, what module and how thick? For example: module 2, 20 teeth, 8 mm thick."),
                          ("bracket", "What size should it be, and what will it hold? Give me dimensions in mm, for example 60x40x5 mm with 4 holes."),
                          ("box", "How big (length x width x height in mm) and how thick should the walls be?"),
                          ("thing", "Tell me what it is and its size in mm, and I will build it.")])
    return _p(random.choice([f"make a {what[0]}", f"I want a {what[0]}", f"build me a {what[0]}"]), what[1])


def grounded_example():
    w, d, h = _n(30, 120, 5), _n(20, 80, 5), _n(3, 10); mw = round(random.uniform(0.9, 3), 1); oh = round(random.uniform(0, 12), 1)
    ok = random.random() < 0.8
    ctx = f"report: size {w:g}x{d:g}x{h:g} mm; thinnest wall {mw:g} mm; overhang {oh:g} percent; verdict {'PASS' if ok else 'FAIL - open edges'}"
    q, a = random.choice([("how big is it", f"It is {w:g} by {d:g} by {h:g} millimetres."), ("what is the thinnest wall", f"The thinnest wall is {mw:g} millimetres."),
                          ("how much overhang", f"{oh:g} percent of the surface overhangs more than 45 degrees."),
                          ("can I print it", "Yes, it passed every check. Do the first print supervised." if ok else "No - it failed the checks (open edges), so do not print it.")])
    return _p(q, a, ctx)


def misc_example():
    r = random.random()
    if r < 0.25:
        g = random.choice(GREET); return _p(g, "Hello! I'm Quadmesh. I can design 3D-printable parts, gears, threads and assemblies. What shall we build?")
    if r < 0.6:
        return _p(random.choice(OFFTOPIC), random.choice(REFUSE))
    return _p(random.choice(["thanks", "thank you", "great"]), "You're welcome! Tell me if you want to change anything.")


def gen_example() -> dict:
    r = random.random()
    return (fact_example() if r < 0.28 else tool_example() if r < 0.62 else grounded_example() if r < 0.76 else clarify_example() if r < 0.86 else misc_example())


def parse_call(text: str) -> Optional[dict]:
    """'... CALL {json}' -> dict, or None. Only the JSON object right after CALL is read."""
    if "CALL" not in text:
        return None
    s = text.split("CALL", 1)[1].strip()
    try:
        d, _ = json.JSONDecoder().raw_decode(s)
    except Exception:
        return None
    return d if isinstance(d, dict) and isinstance(d.get("tool"), str) else None
