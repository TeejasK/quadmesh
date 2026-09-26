"""
Generates the Stage 2 SFT dataset for task_planner: (instruction -> JSON plan)
pairs. This is what teaches the model to actually understand a prompt and
decide steps, instead of only doing generic language modeling from Stage 1.

Fully synthetic, fully local — no Hugging Face download needed. Expanded to
train the model on 100+ native operations, quad-remeshing, mechanical recipes,
modifiers, precision geometry, and multi-format exports.
"""
from __future__ import annotations
import json
import os
import random

def _pos():
    return (round(random.uniform(-2, 2), 1), round(random.uniform(-2, 2), 1), 0.0)


SHAPES = [
    ("cube", "add_cube", lambda: {
        "size": round(random.uniform(0.5, 4.0), 1),
        "location": _pos()
    }),
    ("cylinder", "add_cylinder", lambda: {
        "radius": round(random.uniform(0.2, 1.5), 2),
        "depth": round(random.uniform(1.0, 5.0), 1),
        "location": _pos()
    }),
    ("sphere", "add_sphere", lambda: {
        "radius": round(random.uniform(0.3, 2.0), 2),
        "location": _pos()
    }),
    ("ico_sphere", "add_ico_sphere", lambda: {
        "radius": round(random.uniform(0.3, 2.0), 2),
        "location": _pos()
    }),
    ("cone", "add_cone", lambda: {
        "radius1": round(random.uniform(0.3, 2.0), 2),
        "depth": round(random.uniform(1.0, 4.0), 1),
        "location": _pos()
    }),
    ("torus", "add_torus", lambda: {
        "major_radius": round(random.uniform(1.0, 3.0), 2),
        "minor_radius": round(random.uniform(0.1, 0.5), 2),
        "location": _pos()
    }),
    ("plane", "add_plane", lambda: {
        "size": round(random.uniform(1.0, 5.0), 1),
        "location": _pos()
    }),
    ("circle", "add_circle", lambda: {
        "radius": round(random.uniform(0.5, 3.0), 2),
        "location": _pos()
    }),
    ("grid", "add_grid", lambda: {
        "size": round(random.uniform(1.0, 4.0), 1),
        "x_subdivisions": random.choice([5, 10, 16]),
        "y_subdivisions": random.choice([5, 10, 16]),
        "location": _pos()
    }),
    ("monkey", "add_monkey", lambda: {
        "size": round(random.uniform(0.5, 2.5), 1),
        "location": _pos()
    }),
    ("bezier_curve", "add_bezier_curve", lambda: {
        "location": _pos()
    }),
    ("path", "add_path", lambda: {
        "location": _pos()
    }),
]

CAD_RECIPES = {
    "bracket": {"shapes": ["cube", "cylinder"], "boolean": "DIFFERENCE", "mod": "bevel", "quad": True},
    "mount": {"shapes": ["cube", "cylinder"], "boolean": "DIFFERENCE", "mod": None, "quad": True},
    "bearing housing": {"shapes": ["cylinder", "cylinder"], "boolean": "DIFFERENCE", "mod": "bevel", "quad": True},
    "enclosure": {"shapes": ["cube"], "boolean": None, "mod": "solidify", "quad": False},
    "drone arm": {"shapes": ["cylinder", "cube"], "boolean": "UNION", "mod": None, "quad": True},
    "hinge": {"shapes": ["cube", "cylinder"], "boolean": "UNION", "mod": None, "quad": False},
    "gear blank": {"shapes": ["cylinder", "cylinder"], "boolean": "DIFFERENCE", "mod": "bevel", "quad": True},
    "spacer": {"shapes": ["cylinder", "cylinder"], "boolean": "DIFFERENCE", "mod": None, "quad": True},
    "standoff": {"shapes": ["cylinder", "cylinder"], "boolean": "DIFFERENCE", "mod": None, "quad": False},
    "frame": {"shapes": ["cube"], "boolean": None, "mod": "wireframe", "quad": False},
    "flange": {"shapes": ["cylinder", "cylinder"], "boolean": "DIFFERENCE", "mod": "bevel", "quad": True},
    "threaded rod": {"shapes": ["cylinder"], "boolean": None, "mod": "screw", "quad": False},
    "bolt": {"shapes": ["cylinder", "cylinder"], "boolean": "UNION", "mod": "screw", "quad": False},
    "nut": {"shapes": ["cylinder", "cylinder"], "boolean": "DIFFERENCE", "mod": "bevel", "quad": True},
    "pipe": {"shapes": ["cylinder"], "boolean": None, "mod": "solidify", "quad": True},
    "tube": {"shapes": ["cylinder"], "boolean": None, "mod": "solidify", "quad": True},
    "bushing": {"shapes": ["cylinder", "cylinder"], "boolean": "DIFFERENCE", "mod": None, "quad": True},
    "pulley": {"shapes": ["cylinder", "torus"], "boolean": "DIFFERENCE", "mod": "bevel", "quad": True},
    "wheel": {"shapes": ["cylinder", "cylinder"], "boolean": "UNION", "mod": "subdivision", "quad": True},
    "axle": {"shapes": ["cylinder"], "boolean": None, "mod": None, "quad": False},
    "shaft": {"shapes": ["cylinder"], "boolean": None, "mod": "bevel", "quad": True},
    "fin": {"shapes": ["plane"], "boolean": None, "mod": "solidify", "quad": False},
    "heat sink": {"shapes": ["cube"], "boolean": None, "mod": "array", "quad": False},
    "grille": {"shapes": ["grid"], "boolean": None, "mod": "wireframe", "quad": False},
    "duct": {"shapes": ["cylinder"], "boolean": None, "mod": "solidify", "quad": True},
    "nozzle": {"shapes": ["cone", "cylinder"], "boolean": "DIFFERENCE", "mod": "solidify", "quad": True},
    "funnel": {"shapes": ["cone"], "boolean": None, "mod": "solidify", "quad": True},
    "cover plate": {"shapes": ["plane"], "boolean": None, "mod": "solidify", "quad": False},
    "chassis": {"shapes": ["cube"], "boolean": None, "mod": "bevel", "quad": True},
    "lever": {"shapes": ["cube", "cylinder"], "boolean": "UNION", "mod": None, "quad": False},
    "cam": {"shapes": ["cylinder", "cube"], "boolean": "UNION", "mod": "bevel", "quad": True},
    "coupling": {"shapes": ["cylinder", "cylinder"], "boolean": "DIFFERENCE", "mod": None, "quad": True},
    "ring": {"shapes": ["torus"], "boolean": None, "mod": None, "quad": True},
    "washer": {"shapes": ["cylinder", "cylinder"], "boolean": "DIFFERENCE", "mod": None, "quad": False},
    "adapter": {"shapes": ["cone", "cylinder"], "boolean": "DIFFERENCE", "mod": None, "quad": True},
    "bottle": {"shapes": ["cylinder", "cone"], "boolean": "UNION", "mod": "solidify", "quad": True},
    "impeller hub": {"shapes": ["cylinder", "cylinder"], "boolean": "DIFFERENCE", "mod": "array", "quad": True},
    "lathe blank": {"shapes": ["circle"], "boolean": None, "mod": "spin", "quad": True},
    "manifold": {"shapes": ["cylinder", "cylinder"], "boolean": "UNION", "mod": "solidify", "quad": True},
}

CAD_NOUNS = {k: v["shapes"] for k, v in CAD_RECIPES.items()}


def _shape_step(name: str):
    for shape, op, argfn in SHAPES:
        if shape == name:
            args = argfn()
            desc = f"Add {shape} at {args['location']}"
            return {"description": desc, "op": op, "args": args}
    raise KeyError(name)


def _modifier_step(mod_type: str):
    if mod_type == "bevel":
        return {
            "description": "Apply bevel modifier",
            "op": "bevel_modifier",
            "args": {"width": round(random.uniform(0.02, 0.2), 3), "segments": 3}
        }
    elif mod_type == "solidify":
        return {
            "description": "Apply solidify modifier",
            "op": "solidify_modifier",
            "args": {"thickness": round(random.uniform(0.05, 0.3), 2)}
        }
    elif mod_type == "array":
        return {
            "description": "Apply array modifier",
            "op": "array_modifier",
            "args": {"count": random.choice([3, 4, 6])}
        }
    elif mod_type == "wireframe":
        return {
            "description": "Apply wireframe modifier",
            "op": "wireframe_modifier",
            "args": {"thickness": round(random.uniform(0.01, 0.05), 3)}
        }
    elif mod_type == "subdivision":
        return {
            "description": "Apply subdivision surface modifier",
            "op": "subdivision_modifier",
            "args": {"levels": 2}
        }
    elif mod_type == "screw":
        return {
            "description": "Apply screw modifier",
            "op": "screw_modifier",
            "args": {"screw_offset": round(random.uniform(0.2, 1.0), 2), "iterations": random.choice([3, 4, 5])}
        }
    elif mod_type == "spin":
        return {
            "description": "Revolve profile around axis",
            "op": "spin",
            "args": {"angle": 6.28318, "steps": 24}
        }
    elif mod_type == "mirror":
        return {
            "description": "Apply mirror modifier",
            "op": "mirror_modifier",
            "args": {"use_axis": (True, False, False)}
        }
    return None


def _export_step(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    op_map = {
        ".stl": "export_stl",
        ".obj": "export_obj",
        ".ply": "export_ply",
        ".fbx": "export_fbx",
        ".gltf": "export_gltf",
        ".dae": "export_dae",
    }
    op = op_map.get(ext, "export_stl")
    return {
        "description": f"Export to {filename}",
        "op": op,
        "args": {"filepath": f"C:/Users/Public/{filename}"}
    }


def gen_example() -> dict:
    noun = random.choice(list(CAD_RECIPES))
    recipe = CAD_RECIPES[noun]
    shapes = recipe["shapes"]
    verb = random.choice(["Generate", "Create", "Design", "Model", "Make", "Construct", "Build", "Fabricate"])
    ext = random.choice([".stl", ".obj", ".ply", ".fbx", ".gltf"])
    filename = noun.replace(" ", "_") + ext
    prompt = f"{verb} a {noun}, export as {filename}"

    plan = [_shape_step(s) for s in shapes]

    # CSG Boolean
    if recipe.get("boolean") and len(shapes) > 1:
        bool_op = recipe["boolean"]
        if bool_op == "DIFFERENCE":
            plan.append({"description": "Subtract second shape from first", "op": "boolean_difference", "args": {}})
        elif bool_op == "UNION":
            plan.append({"description": "Join shapes with boolean union", "op": "boolean_union", "args": {}})
        elif bool_op == "INTERSECT":
            plan.append({"description": "Intersect shapes", "op": "boolean_intersect", "args": {}})

    # Modifier
    mod = recipe.get("mod")
    if mod:
        step = _modifier_step(mod)
        if step:
            plan.append(step)

    # Quadmesh Topology Cleanup
    if recipe.get("quad") and random.random() < 0.6:
        plan.append({
            "description": "Quadriflow remesh into clean quad topology",
            "op": "quadriflow_remesh",
            "args": {"target_faces": random.choice([1500, 2000, 3000])}
        })
        plan.append({
            "description": "Recalculate normals consistent outside",
            "op": "recalculate_normals",
            "args": {}
        })

    plan.append(_export_step(filename))

    return {"prompt": prompt, "text": prompt + " " + json.dumps(plan)}


def write_dataset(path: str, n: int = 20_000, seed: int = 0):
    random.seed(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for _ in range(n):
            f.write(json.dumps(gen_example()) + "\n")
    print(f"wrote {n} examples to {path}")


if __name__ == "__main__":
    write_dataset("/tmp/planner_sft.jsonl", n=1000)
