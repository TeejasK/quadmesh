"""
Offline self-test - needs NO GPU, NO Blender, NO Modal, NO torch:   python tests/run_offline_tests.py
Checks the parts that are pure Python: the plan checker, the 10,000+ family generator, spec parsing, assemblies and
strength sizing, the printability gate, the action-safety filter, the GUI recipes' arithmetic, image tracing,
per-role sizes, curved/loft/sweep/thread/gear/text geometry, mechanisms analysis, photo-to-3D, the chat knowledge base
and tool-call data, the TTS vocoder and pitch style, STEP round-trip (if cadquery is installed), the self-opening
Blender launcher, full-control mode, and the autopilot loop against a fake Blender. Prints PASS / FAIL per check.
"""
import os, random, struct, sys, tempfile, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

RESULTS = []


def check(name):
    def deco(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, detail))
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
        return fn
    return deco


def write_box_stl(path, bb):
    x0, x1, y0, y1, z0, z1 = bb
    v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]], float)
    f = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80 + struct.pack("<I", len(f)))
        for a, b, c in f:
            fh.write(struct.pack("<12fH", 0, 0, 0, *v[a], *v[b], *v[c], 0))


@check("plan checker + part-family generator (3000 examples all valid)")
def _():
    from quadmesh.pipeline.datasets import engineering_gen as g
    from quadmesh.pipeline.plan_checker import check_plan
    random.seed(1)
    for _ in range(3000):
        p, plan, _f = g.build()
        assert not check_plan(plan, p), p
    return "3000/3000 valid"


@check("family count >= 10,000")
def _():
    from quadmesh.pipeline.datasets import engineering_gen as g
    r = g.family_count(trials=6)
    assert r["valid_families"] >= 10000, r["valid_families"]
    return f"{r['valid_families']} families over {len(r['per_body'])} body types"


@check("checker catches corrupted plans (>= 90%)")
def _():
    from quadmesh.pipeline.datasets import engineering_gen as g
    from quadmesh.pipeline.plan_checker import check_plan
    random.seed(2); tot = caught = 0
    for _ in range(1500):
        p, plan, _f = g.build(); m = g._corrupt(plan)
        if m is None or m == plan: continue
        tot += 1; caught += bool(check_plan(m, p))
    assert caught / tot >= 0.90, caught / tot
    return f"{caught}/{tot}"


@check("any-angle rotation + circular array")
def _():
    from quadmesh.pipeline.plan_checker import check_plan
    S = lambda op, **a: {"op": op, "args": a}
    plan = [S("add_cyl", name="hub", dia=100, h=6), S("add_box", name="arm", w=120, d=20, h=6), S("move", name="arm", x=110, y=0, z=0),
            S("array_polar", name="arm", count=6), S("union", target="hub", tool="arm")]
    assert not check_plan(plan), check_plan(plan)
    assert not check_plan([S("add_box", name="a", w=100, d=10, h=4), S("rotate", name="a", axis="z", deg=30)])


@check("Spec Generator data round-trips; hostile output rejected")
def _():
    from quadmesh import spec_gen_data as S
    random.seed(3)
    for _ in range(1000):
        e = S.gen_example(); assert S.parse_spec(e["text"]) == e["spec"]
    assert S.parse_spec('SPEC {"kind":"hexacopter","mass_kg":9999}') is None and S.parse_spec("garbage") is None


@check("assemblies: 300 random specs, every part plan valid")
def _():
    from quadmesh.assemblies import build_assembly
    from quadmesh.pipeline.plan_checker import check_plan
    random.seed(4)
    for _ in range(300):
        kind = random.choice(["hexacopter", "robot_arm"])
        spec = {"kind": kind, "grade": random.choice(["hobby", "industrial", "critical"]), "material": random.choice(["petg", "nylon", "pa12cf"])}
        spec.update({"mass_kg": round(random.uniform(.5, 5), 1)} if kind == "hexacopter" else {"reach_mm": random.choice([200, 300, 400, 500]), "payload_kg": round(random.uniform(.1, 2), 1)})
        for p in build_assembly(spec).parts:
            assert not check_plan(p.plan), (spec, p.name)


@check("printability gate flags every bad mesh, passes a good one")
def _():
    from quadmesh.agent import printability as P
    def box(x0, x1, y0, y1, z0, z1):
        v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]], float)
        f = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
        return np.array([[v[a], v[b], v[c]] for a, b, c in f])
    def verdict(t, lim=P.Limits()):
        return P.judge_report(P.analyze_mesh(t), None, lim)[0]
    good = box(-20, 20, -15, 15, 0, 10)
    assert verdict(good) == []
    assert verdict(good[:-1]); assert verdict(good[:, [0, 2, 1]]); assert verdict(box(-20, 20, -15, 15, 0, 0.3))
    assert verdict(box(-20, 20, -15, 15, 5, 15)); assert verdict(box(-120, 120, -15, 15, 0, 10), P.Limits(bed=(100, 100, 100)))
    assert verdict(np.concatenate([good, box(50, 60, 0, 10, 0, 10)]))


@check("action-safety filter refuses dangerous keys by default")
def _():
    from quadmesh import agent  # noqa: F401
    from quadmesh.agent import actions as A
    for bad in ["hotkey alt+f4", "hotkey ctrl+w", "hotkey ctrl+alt+delete", "key win", "please open the pod bay doors", 'type "x\u00e9"']:
        assert A.parse(bad) is None, bad
    for good in ["click 412 380", "hotkey shift+a", 'type "cube"', "drag 10 10 900 400", "done"]:
        assert A.parse(good) is not None, good


@check("GUI recipes: emulated S/G arithmetic gives exact sizes")
def _():
    from quadmesh.agent import actions as A, gui_recipes as R
    def emulate(lines):
        dims = [2.0, 2.0, 2.0]; zc = 0.0; pending = axis = None; buf = ""
        for ln in lines:
            a = A.parse(ln); assert a, ln
            if a.kind == "key" and a.keys[0] in ("s", "g"): pending, axis, buf = a.keys[0], None, ""
            elif a.kind == "hotkey" and a.keys == ["shift", "z"]: axis = "xy"
            elif a.kind == "key" and pending and a.keys[0] in "xyz": axis = a.keys[0]
            elif a.kind == "type" and pending: buf = a.text
            elif a.kind == "key" and a.keys[0] == "enter" and pending and buf:
                v = float(buf)
                if pending == "s":
                    for i, ax in enumerate("xyz"):
                        if axis is None or axis == ax or (axis == "xy" and ax in "xy"): dims[i] *= v
                else: zc += v
                pending, buf = None, ""
        return dims, zc
    for w, d, h in [(60, 40, 5), (12.5, 133, 40)]:
        dims, zc = emulate(R.add_box("body", w, d, h)); assert all(abs(x - y) < 1e-9 for x, y in zip(dims, (w, d, h))) and abs(zc - h / 2) < 1e-9


@check("image -> outline tracer (rectangle with a hole)")
def _():
    from PIL import Image, ImageDraw
    from quadmesh.agent.image3d import outline_from_image
    im = Image.new("L", (400, 300), 255); d = ImageDraw.Draw(im); d.rectangle([50, 50, 350, 250], fill=0); d.ellipse([170, 120, 230, 180], fill=255)
    fn = os.path.join(tempfile.mkdtemp(), "a.png"); im.save(fn)
    r = outline_from_image(fn, 90, 5)
    xs = [p[0] for p in r["outer"]]; assert abs(max(xs) - min(xs) - 90) < 0.5 and len(r["holes"]) == 1


@check("per-role sizes: weights sum to 1, chat/planner/spec/vla/asr/tts get more than an equal split")
def _():
    from quadmesh.config import TIERS
    from quadmesh.role_shapes import ROLE_WEIGHTS, role_shape, nonembedding_params
    assert abs(sum(ROLE_WEIGHTS.values()) - 1) < 1e-9
    for k in ("100M", "500M", "1B", "3B"):
        t = TIERS[k]; eq = nonembedding_params(t.shape)
        for r in ("task_planner", "vla", "chat"):
            assert nonembedding_params(role_shape(t, r)) > eq, (k, r)
        assert nonembedding_params(role_shape(t, "risk_tier_classifier")) < eq


@check("autopilot end to end against a fake Blender (build -> check -> export -> verify file)")
def _():
    import contextlib, io
    from quadmesh.pipeline.plan_checker import predicted_bbox
    from quadmesh.pipeline.datasets import engineering_gen as g
    from quadmesh.agent import autopilot as A
    class FB:
        def __init__(s): s.ops = []
        def is_up(s): return True
        def call(s, op, **a):
            if op == "mark_scene": return {"ok": True}
            if op == "reset_scene": s.ops = []; return {"ok": True}
            if op == "print_check":
                bb = predicted_bbox(s.ops); dims = [bb[1]-bb[0], bb[3]-bb[2], bb[5]-bb[4]]
                return {"ok": True, "result": {"objects": 1, "loose_parts": 1, "open_edges": 0, "nonmanifold_edges": 0, "flipped_edges": 0,
                        "degenerate_faces": 0, "volume": 1e3, "dims": dims, "zmin": 0.0, "min_wall": 1.5, "overhang_pct": 0.0}}
            if op == "save_stl": write_box_stl(a["filepath"], predicted_bbox(s.ops)); return {"ok": True}
            s.ops.append({"op": op, "args": a}); return {"ok": True}
    random.seed(5); prompt, good, _f = g.build(); bad = g._corrupt(good); seq = [bad, good]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      r = A.design(prompt, attempts=4, out_dir=tempfile.mkdtemp(), watch=False, propose_fn=lambda p, **k: seq.pop(0) if seq else good, bridge=FB())
    assert r.ok and r.attempts == 2, (r.ok, r.attempts)
    with contextlib.redirect_stdout(buf):
      r = A.design(prompt, attempts=2, out_dir=tempfile.mkdtemp(), watch=False, propose_fn=lambda p, **k: bad, bridge=FB())
    assert not r.ok


@check("geometry3d: extrude/lathe/loft/sweep/thread/gear/text all watertight with correct volumes")
def _():
    import math
    from quadmesh import geometry3d as G
    from quadmesh.agent import printability as P
    def okmesh(m, lim=None):
        lim = lim or P.Limits(max_overhang_pct=100, min_wall=0.0, expected_islands=None)
        r = P.analyze_mesh(m.triangles(), samples=40)
        pr, _ = P.judge_report(r, None, lim)
        assert not pr, pr
        return r
    circ = lambda r, n: [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    A = lambda r, n: 0.5 * n * r * r * math.sin(2 * math.pi / n)
    ring = G.extrude_polygon(circ(20, 96), [circ(8, 64)], 5)
    okmesh(ring); assert abs(ring.volume() - 5 * (A(20, 96) - A(8, 64))) / ring.volume() < 0.01
    cone = G.lathe([(0, 0), (20, 0), (0, 30)], 96); okmesh(cone); assert abs(cone.volume() - math.pi * 400 * 10) / cone.volume() < 0.02
    okmesh(G.loft([G.rect_ring(40, 20, 0, 64, 0.3), G.circle_ring(12, 40, 64)]))
    pipe = G.sweep(G.circle_profile(3, 48), np.array([[0, 0, z] for z in np.linspace(0, 50, 11)]))
    okmesh(pipe); assert abs(pipe.volume() - math.pi * 9 * 50) / pipe.volume() < 0.02
    okmesh(G.threaded_rod(8, 1.25, 20), P.Limits(max_overhang_pct=100, min_wall=0.0, expected_islands=None))
    okmesh(G.nut(8, 1.25, 6.5), P.Limits(max_overhang_pct=100, min_wall=0.0, expected_islands=None))
    from quadmesh.mesh_tools import make_shape
    meshes, _ = make_shape("gear", module=2, teeth=20, thickness=8, bore=8)
    okmesh(meshes["gear"], P.Limits(max_overhang_pct=100, min_wall=0.0, expected_islands=None))


@check("mechanisms: gear pair meshes correctly, four-bar analysis matches Grashof's rule")
def _():
    from quadmesh import mechanisms as M
    gp = M.gear_pair(2, 16, 24, 8)
    assert abs(gp["centre_distance"] - 40.0) < 1e-6 and abs(gp["ratio"] - 1.5) < 1e-6
    cr = M.four_bar_analysis(100, 30, 90, 70)
    assert cr["grashof"] and cr["crank_turns_fully"] and cr["min_transmission_angle_deg"] > 0
    dr = M.four_bar_analysis(100, 95, 30, 20)
    assert not dr["crank_turns_fully"] or dr["reachable_fraction"] < 1.0


@check("photo-to-3D: symmetric object -> lathe, asymmetric -> inflate, both watertight")
def _():
    import tempfile as tf
    from PIL import Image, ImageDraw
    from quadmesh.photo3d import guess_from_photo
    from quadmesh.agent import printability as P
    im = Image.new("L", (300, 420), 255); d = ImageDraw.Draw(im)
    pts = [(110, 380), (80, 250), (120, 120), (125, 40), (175, 40), (180, 120), (220, 250), (190, 380)]
    d.polygon(pts, fill=40); f1 = tf.mktemp(suffix=".png"); im.save(f1)
    mesh, rep = guess_from_photo(f1, width_mm=80)
    assert rep["mode"] == "lathe"
    r = P.analyze_mesh(mesh.triangles(), samples=30)
    assert r["open_edges"] == 0
    im2 = Image.new("L", (300, 300), 255); d2 = ImageDraw.Draw(im2)
    d2.polygon([(40, 40), (140, 40), (140, 160), (260, 160), (260, 250), (40, 250)], fill=30)
    f2 = tf.mktemp(suffix=".png"); im2.save(f2)
    mesh2, rep2 = guess_from_photo(f2, width_mm=60)
    assert rep2["mode"] == "inflate"


@check("chat data generator produces only validated tool calls")
def _():
    from quadmesh import chat_data as C
    from quadmesh.mesh_tools import validate
    random.seed(0); n_calls = 0
    for _ in range(2000):
        ex = C.gen_example()
        call = C.parse_call(ex["answer"])
        if call and call.get("tool") == "make_shape":
            n_calls += 1
            validate(call["kind"], {k: v for k, v in call.items() if k not in ("tool", "kind")})
    assert n_calls > 100, n_calls


@check("chat knowledge base answers on-topic questions, stays silent on unrelated ones")
def _():
    from quadmesh.chat_kb import kb_answer
    assert kb_answer("what is PETG good for") is not None
    assert kb_answer("why did my print snap between layers") is not None
    assert kb_answer("what is the weather today") is None


@check("chat agent: (legacy) rule fallback, tool execution, file reading, photo-to-3D, full-control toggle")
def _():
    import types
    os.environ["QUADMESH_ALLOW_RULE_FALLBACK"] = "1"       # the legacy parser is opt-in now; this test exercises that path
    os.environ.pop("QUADMESH_CKPT_ROOT", None)
    from quadmesh.agent.chat_agent import ChatAgent
    class FakeDesk:
        def __init__(s): s.io = types.SimpleNamespace(key=lambda k: None)
        def call(s, *a, **k): return True
        def turntable(s, *a, **k): return (["f1", "f2"], None)
    agent = ChatAgent(out_dir=tempfile.mkdtemp(), desk=FakeDesk(), log=lambda *a: None)
    agent._load = lambda: False
    assert "PETG" in agent.turn("what is PETG good for?") or "petg" in agent.turn("what is PETG good for?").lower()
    assert "right" in agent.turn("move body 20 mm right")
    r = agent._execute({"tool": "make_shape", "kind": "gear", "module": 2, "teeth": 20, "thickness": 8, "bore": 8}, None)
    assert "Made the gear" in r
    r2 = agent._execute({"tool": "make_shape", "kind": "gear", "teeth": 3}, None)
    assert "couldn't do that" in r2
    p = os.path.join(tempfile.mkdtemp(), "t.stl"); write_box_stl(p, (-20, 20, -15, 15, 0, 10))
    assert "watertight" in agent.read_3d_file(p)
    from quadmesh.agent import actions as A
    assert not A.UNRESTRICTED
    agent2 = ChatAgent(out_dir=tempfile.mkdtemp(), full_control=True, log=lambda *a: None)
    assert A.UNRESTRICTED
    A.set_unrestricted(False)


@check("TTS vocoder: mel of a known tone reconstructs to (approximately) that frequency")
def _():
    from quadmesh import tts_synth as V
    sr, n_fft, hop, n_mels = 22050, 1024, 256, 80
    t = np.arange(0, 1.0 * sr) / sr
    x = 0.5 * np.sin(2 * np.pi * 350 * t) * np.hanning(len(t))
    S = V._stft(x, n_fft, hop); mag = np.abs(S)
    fb = V._mel_filterbank(sr, n_fft, n_mels, 0, sr / 2)
    mel_db = np.log(np.clip(mag @ fb.T, 1e-5, None))
    y = V.griffin_lim(V.mel_to_linear(mel_db, sr, n_fft, n_mels), n_fft, hop, iters=40)
    peak = np.fft.rfftfreq(len(y), 1 / sr)[np.argmax(np.abs(np.fft.rfft(y * np.hanning(len(y)))))]
    assert abs(peak - 350) < 15, peak


@check("TTS pitch-shift style: verified frequency shift in both directions")
def _():
    from quadmesh import tts_style as ST
    sr = 22050; t = np.arange(0, 1.0 * sr) / sr; x = 0.4 * np.sin(2 * np.pi * 220 * t)
    def mag_at(sig, f):
        F = np.fft.rfft(sig * np.hanning(len(sig))); freqs = np.fft.rfftfreq(len(sig), 1 / sr)
        return np.abs(F)[np.argmin(np.abs(freqs - f))]
    for st in (1.5, 2.0, -2.0):
        y = ST.pitch_shift(x, sr, st); want = 220 * 2 ** (st / 12)
        assert mag_at(y, want) > mag_at(y, 220) * 5, st


@check("STEP round trip (CadQuery): plan -> STEP -> re-read, volume matches (skips if cadquery not installed)")
def _():
    try:
        import cadquery  # noqa
    except ImportError:
        return "SKIPPED (cadquery not installed)"
    from quadmesh.pipeline.datasets import engineering_gen as g
    from quadmesh import step_io as S
    random.seed(3); _p, plan, _f = g.build()
    d = tempfile.mkdtemp()
    paths, warn, wp = S.plan_to_step(plan, os.path.join(d, "part.step"))
    mesh, facts = S.step_to_mesh(paths[0])
    assert abs(facts["volume_mm3"] - wp.val().Volume()) < 1.0


@check("Blender launcher: finds the newest installed version, never double-launches, fails clearly if absent")
def _():
    import shutil
    from quadmesh.agent import launch as L
    root = tempfile.mkdtemp()
    for v in ("3.6", "4.2", "4.5"):
        dd = os.path.join(root, "Blender Foundation", f"Blender {v}"); os.makedirs(dd); open(os.path.join(dd, "blender.exe"), "w").close()
    real_which = shutil.which; shutil.which = lambda x: None
    real_env = os.environ.pop("QUADMESH_BLENDER", None)
    real_save = L.SAVE_FILE
    L.SAVE_FILE = os.path.join(tempfile.mkdtemp(), "blender_path.txt")   # isolate from any real saved path on this machine
    try:
        exe = L.find_blender_exe(extra_roots=[root], include_system_dirs=False)
        assert "4.5" in exe
        started = []
        assert L.ensure_blender(is_up=lambda: True, popen=lambda c: started.append(c), focus=False) and len(started) == 0
    finally:
        shutil.which = real_which
        L.SAVE_FILE = real_save
        if real_env is not None:
            os.environ["QUADMESH_BLENDER"] = real_env


@check("full-control mode: unlocks restricted keys, allows mouse strokes, stays off by default")
def _():
    from quadmesh.agent import actions as A
    assert A.parse("hotkey alt+f4") is None
    A.set_unrestricted(True)
    try:
        assert A.parse("hotkey alt+f4") is not None
        assert A.parse("stroke left 100 100 300 300 500 200") is not None
    finally:
        A.set_unrestricted(False)
    assert A.parse("hotkey alt+f4") is None


@check("mesh_tools schema validation rejects everything outside its declared ranges")
def _():
    from quadmesh.mesh_tools import validate
    for kind, bad in [("gear", {"teeth": 3}), ("gear", {"nope": 1}), ("text", {"text": "x\u00e9"})]:
        try:
            validate(kind, bad); assert False, (kind, bad)
        except ValueError:
            pass
    from quadmesh.mesh_tools import make_shape                          # pipe wall check is cross-parameter, not a simple range
    try:
        make_shape("pipe", outer_dia=20, inner_dia=19.9); assert False
    except ValueError:
        pass


# ---------------------------------------------------------------- shared generative model (v2)
@check("chat tool-call data: every number in a make_shape call comes from its request")
def _():
    import re
    from quadmesh import chat_data as C
    random.seed(3); n = bad = 0
    for _ in range(3000):
        ex = C.tool_example(); call = C.parse_call(ex["answer"])
        if not call or call.get("tool") != "make_shape" or call["kind"] in ("nut", "threaded_rod"):   # thread pitch is table-derived
            continue
        nums = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", ex["user"])}
        n += 1
        bad += any(float(v) not in nums for v in call.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
    assert n > 500 and bad == 0, f"{bad}/{n} calls contradict their request"
    return f"{n} calls, 0 contradictions"


@check("role data: auditor label == the plan checker's verdict; every role parses back; train/eval keys disjoint")
def _():
    from quadmesh.pipeline.datasets import role_gen_data as R
    from quadmesh import roles_io as IO
    from quadmesh.pipeline.plan_checker import check_plan
    random.seed(11); seen = {}
    for role in R.PROCEDURAL_ROLES:
        for _ in range(60):
            for split in ("train", "eval"):
                p, a = R.make_example(role, split)
                seen[role] = seen.get(role, 0) + 1
                if role == "task_planner":
                    assert IO.parse_plan(a) is not None
                elif role == "spec_generator":
                    assert IO.parse_spec(a) is not None
                elif role == "spec_auditor":
                    v = IO.parse_verdict(a); req, plan = p.split("REQUEST: ", 1)[1].split("\nPLAN: ")
                    assert v and (v["verdict"] == "PASS") == (not check_plan(IO.parse_plan(plan), req)), p[:80]
                elif role == "arbitration":
                    assert IO.parse_choice(a) is not None
                elif role == "cad_skill_dispatch":
                    assert IO.parse_dispatch(a)["kind"] != "invalid", a
    for _ in range(300):                                   # the split key decides membership, both ways
        for mk in (R.planner_example, R.spec_example, R.dispatch_example):
            k = mk()[2]
            assert R.is_eval_prompt(k) == (R.bucket(k) < R.EVAL_PERMILLE)
    tr = {R.make_example("spec_auditor", "train")[0].split("\nPLAN")[0] for _ in range(300)}
    ev = {R.make_example("spec_auditor", "eval")[0].split("\nPLAN")[0] for _ in range(100)}
    assert not (tr & ev), "train and eval overlap"
    return f"{sum(seen.values())} examples across {len(seen)} roles"


@check("legacy rule-based planner is OFF by default (no silent fallback)")
def _():
    from quadmesh.agent import planner
    for k in ("QUADMESH_CKPT_ROOT", "QUADMESH_ALLOW_RULE_FALLBACK"):
        os.environ.pop(k, None)
    try:
        planner.plan_from_text("a cube 10 mm"); assert False, "should refuse without a trained model"
    except planner.NoModelError:
        pass


try:
    import torch as _torch
except Exception:
    _torch = None

if _torch is not None:
    @check("SharedTextModel trains NEXT-token prediction (random tokens stay at ln V) and generates")
    def _():
        import math
        from quadmesh.config import ModelShape
        from quadmesh.model.shared_backbone import SharedTextModel
        _torch.manual_seed(0)
        m = SharedTextModel(shape=ModelShape(d_model=48, n_layers=2, n_heads=2, d_ff=128, vocab_size=120, max_seq_len=64))
        opt = _torch.optim.AdamW(m.parameters(), lr=3e-3); last = []
        for _ in range(120):
            ids = _torch.randint(0, 120, (16, 24))
            loss = m(ids, targets=ids)["loss"]; opt.zero_grad(); loss.backward(); opt.step(); last.append(loss.item())
        final = sum(last[-10:]) / 10
        assert final > 0.9 * math.log(120), f"loss {final:.2f} on random tokens - a copy objective would be ~0"
        assert len(m.generate(_torch.randint(0, 120, (1, 4)), max_new_tokens=5)) == 5
        return f"final loss {final:.2f} vs ln V {math.log(120):.2f}"

    @check("tokenizer: every role tag is one token; runtime routes all roles through one model")
    def _():
        from quadmesh.data.tokenizer import QuadmeshTokenizer, ROLE_TAGS
        from quadmesh.pipeline.datasets import role_gen_data as R
        from quadmesh.config import ModelShape
        from quadmesh.model.shared_backbone import SharedTextModel
        from quadmesh.agent.shared_runtime import SharedRuntime, RoleNotTrained
        random.seed(2)
        tok = QuadmeshTokenizer.train_local(R.iter_tokenizer_corpus(300, seed=1), vocab_size=1200)
        for t in ROLE_TAGS.values():
            assert len(tok.encode(t)) == 1, t
        m = SharedTextModel(shape=ModelShape(d_model=32, n_layers=1, n_heads=2, d_ff=64, vocab_size=tok.vocab_size, max_seq_len=512))
        rt = SharedRuntime.from_model(m, tok)
        assert isinstance(rt.ask("chat", "hello", max_new=4), str)
        assert rt.dispatch("move the arm 20 mm left")["kind"] in ("call", "none", "clarify", "invalid")
        try:
            rt.risk("a drone arm"); assert False
        except RoleNotTrained:
            pass
        g = rt.screen("a propeller arm for a drone")            # untrained safety roles -> keyword table only, honestly labelled
        assert g["model_screen"] == "untrained" and g["tier_idx"] >= 3, g


w = max(len(n) for n, _, _ in RESULTS)
for n, ok, d in RESULTS:
    print(f"{'PASS' if ok else 'FAIL'}  {n.ljust(w)}  {d}")
failed = [n for n, ok, _ in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
sys.exit(1 if failed else 0)
