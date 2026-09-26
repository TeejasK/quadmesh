"""
Training data for the SPEC GENERATOR: plain-language request -> design parameters (JSON), or {"kind":"unsupported"}.

Two closed families (hexacopter, robot_arm) are handled by hand-calculated sizing in assemblies.py +
strength.py - the model only extracts parameters there, it never invents a dimension for those two. That
stays true here: exact structural math should never come from a language model's guess.

v2 adds a THIRD kind, "freeform_ops": for anything outside those two families that still matches one of the
thousands of validated part families in pipeline/datasets/engineering_gen.py's catalog (14 body archetypes x
~50 feature variants), spec_generator can emit the op PLAN directly (reusing that catalog's own generator,
which already validates every plan through plan_checker.check_plan before it's kept) instead of falling back
to "unsupported". This is what makes the "generative, not just two fixed shapes" change land: it doesn't
invent a fourth path, it lets spec_generator draw on the wide catalog that task_planner already had, so
"unsupported" now only means "outside the whole catalog", not "outside these two families I happened to name."
Genuinely open-ended physics-unconstrained requests (a car engine, a violin) still correctly stay unsupported -
nothing here makes the model guess structural numbers where math should decide them instead.

Text format (same convention as every other role):   "<request> SPEC {json}"
"""
from __future__ import annotations
import json
import random
import re
from typing import Optional

from quadmesh.pipeline.datasets.engineering_gen import build as _catalog_build
from quadmesh.pipeline.plan_checker import check_plan

HEX = ["hexacopter", "hexa copter", "hexadrone", "hex drone", "six-rotor drone", "6 motor drone", "hex copter frame",
       "hexacopter frame", "six motor multirotor"]
ARM = ["robotic arm with stand", "robot arm with a stand", "robotic arm on a stand", "desktop robot arm with base",
       "2-joint robot arm with stand", "articulated robot arm and stand", "robotic arm with a mounting stand"]
GRADE_WORDS = {
    "industrial": ["industrial grade", "industrial-grade", "for industrial use", "heavy duty", "rugged"],
    "hobby": ["hobby grade", "light duty", "for fun", "hobby"],
    "critical": ["safety critical", "mission critical", "high reliability"],
}
MAT_WORDS = {"petg": ["PETG"], "pla": ["PLA"], "abs": ["ABS"], "asa": ["ASA"], "nylon": ["nylon", "PA12"],
             "pa12cf": ["carbon fibre nylon", "PA12-CF", "carbon-fiber nylon"], "pc": ["polycarbonate", "PC"]}
UNSUPPORTED = ["a car engine", "a full size humanoid robot", "a house", "a violin", "a bridge", "a wind turbine",
               "a jet engine", "a bicycle frame", "a smartphone case with buttons", "a chess set", "a gearbox with gears",
               "a suspension system", "a boat hull", "a chair", "a camera lens"]


def _mass_phrase(m):
    return random.choice([f"{m:g} kg", f"{m:g} kg takeoff weight", f"{int(m*1000)} g", f"weighing {m:g} kg"])


def hex_example():
    spec = {"kind": "hexacopter"}
    parts = [random.choice(["make a", "design a", "build a", "I need a", "create a", "generate a"]) + " " + random.choice(HEX)]
    if random.random() < 0.85:
        m = round(random.uniform(0.5, 6), 1); spec["mass_kg"] = m; parts.append(_mass_phrase(m))
    if random.random() < 0.35:
        wb = random.choice([300, 350, 400, 450, 500, 550, 650]); spec["wheelbase_mm"] = wb
        parts.append(random.choice([f"{wb} mm wheelbase", f"{wb} mm frame", f"{wb} mm motor to motor"]))
    if random.random() < 0.8:
        g = random.choices(list(GRADE_WORDS), weights=[6, 2, 2])[0]; spec["grade"] = g; parts.append(random.choice(GRADE_WORDS[g]))
    if random.random() < 0.5:
        mt = random.choice(list(MAT_WORDS)); spec["material"] = mt; parts.append("in " + random.choice(MAT_WORDS[mt]))
    return _join(parts), spec


def arm_example():
    spec = {"kind": "robot_arm"}
    parts = [random.choice(["make a", "design a", "build a", "I need a", "create a", "generate a"]) + " " + random.choice(ARM)]
    if random.random() < 0.8:
        r = random.choice([200, 250, 300, 400, 500, 650, 800]); spec["reach_mm"] = r
        parts.append(random.choice([f"{r} mm reach", f"reach of {r} mm", f"{r/10:g} cm reach"]) if r % 10 == 0 else f"{r} mm reach")
    if random.random() < 0.7:
        p = round(random.uniform(0.1, 3), 1); spec["payload_kg"] = p
        parts.append(random.choice([f"{p:g} kg payload", f"lifting {p:g} kg", f"carries {int(p*1000)} g"]))
    if random.random() < 0.3:
        n = random.choice([2, 3]); spec["links"] = n; parts.append(f"{n} links")
    if random.random() < 0.8:
        g = random.choices(list(GRADE_WORDS), weights=[6, 2, 2])[0]; spec["grade"] = g; parts.append(random.choice(GRADE_WORDS[g]))
    if random.random() < 0.5:
        mt = random.choice(["petg", "abs", "nylon", "pa12cf", "pla"]); spec["material"] = mt; parts.append("in " + random.choice(MAT_WORDS[mt]))
    return _join(parts), spec


def _join(parts):
    random.shuffle(parts[1:]) if len(parts) > 2 else None
    head, rest = parts[0], parts[1:]
    return head + ("" if not rest else random.choice([", ", " - ", " with "]) + ", ".join(rest))


def unsupported_example():
    return random.choice(["make ", "design ", "build ", "I need "]) + random.choice(UNSUPPORTED), {"kind": "unsupported"}


def freeform_example():
    """Draws one request+plan from engineering_gen's already-validated catalog (thousands of families) and
    reframes it as a spec_generator target: {"kind":"freeform_ops","ops":<plan>}. The plan is never invented
    here - build() already ran it through check_plan before returning it, this just re-packages it."""
    prompt, plan, _fam = _catalog_build()
    return prompt, {"kind": "freeform_ops", "ops": plan}


def gen_example() -> dict:
    r = random.random()
    if r < 0.35:
        req, spec = hex_example()
    elif r < 0.65:
        req, spec = arm_example()
    elif r < 0.9:
        req, spec = freeform_example()
    else:
        req, spec = unsupported_example()
    js = json.dumps(spec, separators=(",", ":"))
    return {"prompt": req, "text": f"{req} SPEC {js}", "spec": spec}


def parse_spec(text: str) -> Optional[dict]:
    """Model output -> spec dict (only known keys and sane values), or None."""
    m = re.search(r"\{.*\}", text.split("SPEC")[-1], re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if d.get("kind") not in ("hexacopter", "robot_arm", "unsupported", "freeform_ops"):
        return None
    if d["kind"] == "freeform_ops":
        ops = d.get("ops")
        if not isinstance(ops, list) or not ops:
            return None
        if check_plan(ops):          # re-validate: never trust a generated plan just because it parsed as JSON
            return None               # non-empty error list -> reject, same hard-invariant gate build() used
        return {"kind": "freeform_ops", "ops": ops}
    ok = {"kind", "mass_kg", "wheelbase_mm", "grade", "material", "reach_mm", "payload_kg", "links"}
    d = {k: v for k, v in d.items() if k in ok}
    lim = {"mass_kg": (0.1, 30), "wheelbase_mm": (150, 1200), "reach_mm": (100, 1500), "payload_kg": (0.01, 20), "links": (2, 3)}
    for k, (lo, hi) in lim.items():
        if k in d and not (isinstance(d[k], (int, float)) and lo <= d[k] <= hi):
            return None
    return d


if __name__ == "__main__":
    random.seed(1)
    for _ in range(8):
        e = gen_example(); print(e["text"])
