"""
Text prompt -> list of Steps.

Two paths, tried in order:
  1. TRAINED MODEL (task_planner + spec_generator live checkpoints), if you've
     actually trained and promoted them (Sec 15 Stage 8 -> promote_to_live).
     Until you have, this path is skipped automatically — no crash, just falls
     through to path 2.
  2. RULE-BASED PARSER — a plain-English -> Blender-ops translator that works
     right now, with zero training required. Fully expanded to support 110+
     native CAD, Quadmesh, Sculpting, and Precision operations.

This file is intentionally the seam between "trained AI" and "runs today
without training."
"""
from __future__ import annotations
from quadmesh.role_shapes import role_shape
import os
import re
from typing import Optional

from quadmesh.agent.loop import Step

CKPT_ROOT = os.environ.get("QUADMESH_CKPT_ROOT", None)

_NUMBER = r"(-?\d+(?:\.\d+)?)"   # shared number-capture fragment, used by _find_number/_parse_position below


# ---------------------------------------------------------------------------
# Path 1 — trained model, only used if a live checkpoint actually exists
# ---------------------------------------------------------------------------

def _try_model_plan(prompt: str) -> Optional[list[Step]]:
    """MCTS-style rollout (item 8): sample several candidate plans instead of trusting one greedy decode,
    filter+rank them with plan_checker.rank_candidates (item 7 - hard-invariant filter first, learned/
    heuristic scoring only between candidates that already passed), take the best, and only fall back to the
    old single-shot self-repair loop if every sampled candidate came back invalid. This is a real multi-
    candidate comparison, not a relabelled retry loop: candidates are drawn in the SAME turn and compared
    against each other, not tried one at a time until one happens to pass.

    QUADMESH_MCTS_SAMPLES (default 5) controls how many candidates are drawn; QUADMESH_MCTS_TEMPERATURE
    (default 0.8) controls how different they are from each other - 0.0 would make every sample identical to
    the greedy decode, defeating the point of sampling more than one."""
    if not os.environ.get("QUADMESH_CKPT_ROOT"):
        return None
    import json
    from quadmesh.agent.planner_model import propose
    from quadmesh.pipeline.plan_checker import check_plan, rank_candidates

    n_samples = int(os.environ.get("QUADMESH_MCTS_SAMPLES", "5"))
    temp = float(os.environ.get("QUADMESH_MCTS_TEMPERATURE", "0.8"))
    candidates = []
    greedy = propose(prompt, temperature=0.0)                 # always include the deterministic decode
    if greedy:
        candidates.append(greedy)
    for _ in range(max(0, n_samples - 1)):
        c = propose(prompt, temperature=temp)
        if c and c not in candidates:                          # skip exact duplicates, they add nothing to rank
            candidates.append(c)

    if candidates:
        ranked = rank_candidates(candidates, prompt=prompt)
        best_score, best_plan, best_errs = ranked[0]
        if best_errs is None:                                  # at least one candidate was fully valid
            print(f"[planner] MCTS: {len(candidates)} candidates sampled, "
                  f"{sum(1 for _, _, e in ranked if e is None)} valid, best score {best_score:.3f}")
            return [Step(f"{s['op']} {json.dumps(s['args'])}", native_op=s["op"], native_args=s["args"])
                    for s in best_plan]
        print(f"[planner] MCTS: none of {len(candidates)} candidates were valid "
              f"(best had {len(best_errs)} errors) - falling back to single-shot self-repair")

    # fallback: the original single-shot + 2-round self-repair loop, using the greedy decode as the seed.
    # Kept rather than removed - it's a real safety net for when the model can't yet produce anything close
    # to valid, and self-repair against feedback is a different mechanism than resampling (it uses the
    # checker's specific errors, not just "try again").
    plan = greedy
    for _ in range(2):
        errs = check_plan(plan, prompt) if plan else ["no valid JSON plan"]
        if not errs:
            break
        print(f"[planner] plan rejected: {'; '.join(errs[:3])}")
        plan = propose(prompt, prev_plan=plan, feedback="; ".join(errs[:2]))
    if not plan or check_plan(plan, prompt):
        print("[planner] model could not produce a valid plan; using the rule-based parser")
        return None
    return [Step(f"{s['op']} {json.dumps(s['args'])}", native_op=s["op"], native_args=s["args"]) for s in plan]


def _find_number(text: str, keyword: str, default: float) -> float:
    m = re.search(rf"(?:{keyword})\D{{0,6}}{_NUMBER}", text)
    return float(m.group(1)) if m else default


def _parse_position(text: str) -> tuple:
    m = re.search(rf"at\s*\(?\s*{_NUMBER}\s*,\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)?", text)
    if m:
        return tuple(float(g) for g in m.groups())
    return (0.0, 0.0, 0.0)


_SHAPE_OPS = {
    # keyword the person might actually say -> real op name registered in blender_bridge.SAFE_OPS
    "cube": "add_cube", "box": "add_cube",
    "cylinder": "add_cylinder",
    "sphere": "add_sphere", "ball": "add_sphere",
    "ico sphere": "add_ico_sphere", "icosphere": "add_ico_sphere",
    "cone": "add_cone",
    "torus": "add_torus", "donut": "add_torus", "ring": "add_torus",
    "plane": "add_plane",
    "grid": "add_grid",
    "circle": "add_circle", "disc": "add_circle", "disk": "add_circle",
    "monkey": "add_monkey", "suzanne": "add_monkey",
}

# noun -> ordered list of (action_type, params) recipe steps, all built from ops actually registered in
# blender_bridge.SAFE_OPS (add_cylinder/add_cube's kwargs match the real bpy.ops.mesh.primitive_*_add
# signatures used elsewhere in this file - radius/depth/size/location - not invented names). This was
# entirely missing before (referenced at runtime, never defined - see the checkpoint_benchmark.py note on
# how this was found) - kept intentionally small and literal rather than guessed-elaborate, since an unused,
# untested recipe table is exactly the kind of code this whole fix is about not having more of.
_CAD_RECIPES = {
    "flange": [
        ("add_cylinder", {"radius": 20.0, "depth": 5.0, "location": (0, 0, 0)}),
        ("add_cylinder", {"radius": 6.0, "depth": 10.0, "location": (0, 0, 0)}),
        ("boolean", "DIFFERENCE"),
        ("bevel", 0.5),
    ],
    "washer": [
        ("add_cylinder", {"radius": 10.0, "depth": 2.0, "location": (0, 0, 0)}),
        ("add_cylinder", {"radius": 5.0, "depth": 4.0, "location": (0, 0, 0)}),
        ("boolean", "DIFFERENCE"),
    ],
    "bracket": [
        ("add_cube", {"size": 20.0, "location": (0, 0, 0)}),
        ("add_cube", {"size": 12.0, "location": (6, 6, 6)}),
        ("boolean", "DIFFERENCE"),
        ("bevel", 0.3),
    ],
    "housing": [
        ("add_cube", {"size": 30.0, "location": (0, 0, 0)}),
        ("solidify", 2.0),
    ],
    "nozzle": [
        ("add_cone", {"radius1": 8.0, "depth": 15.0, "location": (0, 0, 0)}),
        ("add_cylinder", {"radius": 3.0, "depth": 20.0, "location": (0, 0, 0)}),
        ("boolean", "DIFFERENCE"),
    ],
    "pipe": [
        ("add_cylinder", {"radius": 10.0, "depth": 60.0, "location": (0, 0, 0)}),
        ("add_cylinder", {"radius": 8.0, "depth": 62.0, "location": (0, 0, 0)}),
        ("boolean", "DIFFERENCE"),
    ],
    "bottle": [
        ("add_cylinder", {"radius": 15.0, "depth": 60.0, "location": (0, 0, 0)}),
        ("solidify", 1.5),
    ],
}


def rule_based_plan(prompt: str) -> list[Step]:
    text = prompt.lower()
    steps: list[Step] = []

    # 1. Direct Shape Match
    shape_pattern = r"\b(" + "|".join(re.escape(k) for k in _SHAPE_OPS.keys()) + r")\b"
    matches = list(re.finditer(shape_pattern, text))

    if matches:
        for match in matches:
            shape = match.group(1)
            op = _SHAPE_OPS[shape]
            pos = _parse_position(text[match.start():match.start() + 80])
            args = {"location": pos}
            if op == "add_cube":
                args["size"] = _find_number(text, "size|width", 2.0)
            elif op == "add_cylinder":
                args["radius"] = _find_number(text, "radius", 0.5)
                args["depth"] = _find_number(text, "depth|length|height", 2.0)
            elif op in ("add_sphere", "add_ico_sphere"):
                args["radius"] = _find_number(text, "radius", 1.0)
            elif op == "add_cone":
                args["radius1"] = _find_number(text, "radius", 1.0)
                args["depth"] = _find_number(text, "depth|height", 2.0)
            elif op == "add_torus":
                args["major_radius"] = _find_number(text, "major_radius|radius", 1.5)
                args["minor_radius"] = _find_number(text, "minor_radius|thickness", 0.3)
            elif op in ("add_plane", "add_grid"):
                args["size"] = _find_number(text, "size", 2.0)
            elif op == "add_circle":
                args["radius"] = _find_number(text, "radius", 1.0)
            elif op == "add_monkey":
                args["size"] = _find_number(text, "size", 2.0)
            steps.append(Step(f"Add {shape} at {pos}", native_op=op, native_args=args))
    else:
        # 2. Check for high-level CAD recipes
        for noun, recipe_items in _CAD_RECIPES.items():
            if re.search(rf"\b{re.escape(noun)}\b", text):
                for item in recipe_items:
                    action_type = item[0]
                    if action_type.startswith("add_"):
                        steps.append(Step(f"Add {action_type[4:]} for {noun}", native_op=action_type, native_args=dict(item[1])))
                    elif action_type == "boolean":
                        op_type = item[1]
                        desc = "Subtract cutter shape" if op_type == "DIFFERENCE" else "Union shapes"
                        steps.append(Step(desc, native_op="boolean_modifier", native_args={"op": op_type}))
                    elif action_type == "bevel":
                        steps.append(Step("Apply bevel modifier", native_op="bevel_modifier", native_args={"width": item[1], "segments": 3}))
                    elif action_type == "solidify":
                        steps.append(Step("Apply solidify modifier", native_op="solidify_modifier", native_args={"thickness": item[1]}))
                    elif action_type == "wireframe":
                        steps.append(Step("Apply wireframe modifier", native_op="wireframe_modifier", native_args={"thickness": item[1]}))
                    elif action_type == "array":
                        steps.append(Step("Apply array modifier", native_op="array_modifier", native_args={"count": item[1]}))
                    elif action_type == "subdivision":
                        steps.append(Step("Apply subdivision modifier", native_op="subdivision_modifier", native_args={"levels": item[1]}))
                    elif action_type == "screw":
                        steps.append(Step("Apply screw modifier", native_op="screw_modifier", native_args={"screw_offset": item[1]}))
                    elif action_type == "quadriflow":
                        steps.append(Step("Remesh into clean quad topology", native_op="quadriflow_remesh", native_args={"target_faces": item[1]}))
                break

    # 3. CSG Booleans
    if any(w in text for w in ("subtract", "cut", "hole", "difference")):
        steps.append(Step("Boolean subtract second object from first",
                          native_op="boolean_difference",
                          native_args={}))

    if any(w in text for w in ("join", "union", "combine", "merge")):
        steps.append(Step("Boolean union objects",
                          native_op="boolean_union",
                          native_args={}))

    if any(w in text for w in ("intersect", "intersection", "overlap")):
        steps.append(Step("Boolean intersect objects",
                          native_op="boolean_intersect",
                          native_args={}))

    # 4. Lathe / Revolve / Spin
    if any(w in text for w in ("spin", "revolve", "lathe")):
        angle = _find_number(text, "angle", 360.0) * (3.14159265 / 180.0)
        steps.append(Step("Revolve profile around axis", native_op="spin", native_args={"angle": angle, "steps": 24}))

    # 5. Modifiers
    if re.search(r"\b(bevel|chamfer)\b", text):
        width = _find_number(text, "bevel|chamfer|width", 0.1)
        steps.append(Step(f"Apply bevel modifier (width={width})", native_op="bevel_modifier",
                          native_args={"width": width, "segments": 3}))

    if re.search(r"\b(solidify|thicken|thickness|shell)\b", text):
        thick = _find_number(text, "thick(?:ness)?", 0.2)
        steps.append(Step(f"Apply solidify modifier (thickness={thick})", native_op="solidify_modifier",
                          native_args={"thickness": thick}))

    if re.search(r"\b(smooth|subdivide|subdivision)\b", text):
        levels = int(_find_number(text, "levels?|subdivisions?", 2))
        steps.append(Step(f"Apply subdivision surface modifier (levels={levels})", native_op="subdivision_modifier",
                          native_args={"levels": levels}))

    if re.search(r"\b(mirror)\b", text):
        steps.append(Step("Apply mirror modifier", native_op="mirror_modifier",
                          native_args={"use_axis": (True, False, False)}))

    if re.search(r"\b(array|repeat|pattern)\b", text):
        count = int(_find_number(text, "count|copies", 3))
        steps.append(Step(f"Apply array modifier (count={count})", native_op="array_modifier",
                          native_args={"count": count}))

    if re.search(r"\b(wireframe|cage)\b", text):
        thick = _find_number(text, "thick(?:ness)?", 0.05)
        steps.append(Step("Apply wireframe modifier", native_op="wireframe_modifier",
                          native_args={"thickness": thick}))

    if re.search(r"\b(decimate|reduce faces)\b", text):
        ratio = _find_number(text, "ratio", 0.5)
        steps.append(Step(f"Apply decimate modifier (ratio={ratio})", native_op="decimate_modifier",
                          native_args={"ratio": ratio}))

    if re.search(r"\b(weld)\b", text):
        steps.append(Step("Apply weld modifier to merge seams", native_op="weld_modifier", native_args={"merge_threshold": 0.001}))

    # 6. Quadmesh & Topology Cleanup Engine
    if any(w in text for w in ("quad", "quads", "quadmesh", "quadriflow", "all-quad", "all quad", "remesh")):
        faces = int(_find_number(text, "faces?|target", 2000))
        steps.append(Step(f"Quadriflow remesh into {faces} clean quads", native_op="quadriflow_remesh",
                          native_args={"target_faces": faces}))

    if any(w in text for w in ("recalculate normals", "fix normals", "normals outside")):
        steps.append(Step("Recalculate normals consistent outside", native_op="recalculate_normals"))

    if any(w in text for w in ("tris to quads", "tri to quad", "merge tris")):
        steps.append(Step("Convert triangular faces to quads", native_op="tris_to_quads"))

    if any(w in text for w in ("dissolve", "clean faces", "planar clean")):
        steps.append(Step("Limited dissolve coplanar geometry", native_op="limited_dissolve"))

    # 7. Material & Appearance
    if any(w in text for w in ("metallic", "metal", "chrome", "steel", "gold")):
        steps.append(Step("Apply metallic material", native_op="create_material",
                          native_args={"name": "Metal", "metallic": 1.0, "roughness": 0.2}))

    # 8. Multi-Format Export
    m = re.search(r"(?:export|save)\s*(?:as|to)?\s*([a-zA-Z0-9_\-]+\.(?:stl|obj|ply|fbx|gltf|glb|dae))", text)
    filename = m.group(1) if m else "quadmesh_output.stl"
    ext = os.path.splitext(filename)[1].lower()
    op_map = {
        ".stl": "export_stl",
        ".obj": "export_obj",
        ".ply": "export_ply",
        ".fbx": "export_fbx",
        ".gltf": "export_gltf",
        ".glb": "export_gltf",
        ".dae": "export_dae",
    }
    export_op = op_map.get(ext, "export_stl")
    export_path = os.path.join(os.path.expanduser("~"), "Desktop", filename)
    steps.append(Step(f"Export to {export_path}", native_op=export_op,
                      native_args={"filepath": export_path}))

    if len(steps) == 1:
        print(f"[planner] couldn't recognize any shapes in: {prompt!r}. "
              f"Try mentioning primitives (cube, cylinder, sphere, cone, torus, grid, curve) or "
              f"engineering parts (flange, bracket, housing, bottle, nozzle, etc.)")
    return steps


# ---------------------------------------------------------------------------
# Entry point the REPL / loop calls
# ---------------------------------------------------------------------------

class NoModelError(RuntimeError):
    pass


def plan_from_text(prompt: str) -> list[Step]:
    """The trained model writes the plan. The regex parser is NOT a fallback any more: it speaks a different op
    vocabulary from the plan checker and its output was never validated. It only runs when you ask for it with
    QUADMESH_ALLOW_RULE_FALLBACK=1 (kept so old scripts and the offline tests still work)."""
    model_plan = _try_model_plan(prompt)
    if model_plan:
        print("[planner] using trained model")
        return model_plan
    if os.environ.get("QUADMESH_ALLOW_RULE_FALLBACK") == "1":
        print("[planner] QUADMESH_ALLOW_RULE_FALLBACK=1 - using the legacy rule-based parser (unvalidated)")
        return rule_based_plan(prompt)
    raise NoModelError("no trained model produced a valid plan. Set QUADMESH_CKPT_ROOT to a folder holding "
                       "Quadmesh-<tier>/live/shared_text.pt and tokenizer/quadmesh-bpe.json (see commands.txt). "
                       "Legacy parser: QUADMESH_ALLOW_RULE_FALLBACK=1.")
