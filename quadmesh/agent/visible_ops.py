"""
Turns a native op into VISIBLE Blender keyboard-shortcut actions, so you can
watch the agent work instead of it happening silently through the socket bridge.
Uses Blender's real shortcuts (Shift+A, G/R/S, Alt+J, Ctrl+T, Shift+N, F3 search)
— the exact keys a human operator presses.
"""
from __future__ import annotations
import time
from typing import List

from quadmesh.agent.desktop import Action, execute

VIEWPORT_CENTER = (760, 450)


def _hover_viewport():
    execute(Action("move", *VIEWPORT_CENTER))
    time.sleep(0.15)


def _search_op(search_text: str, extra_wait: float = 0.2) -> List[Action]:
    """Universal Blender operator trigger via F3 search."""
    return [
        Action("key", key="f3"),
        Action("wait", seconds=0.3),
        Action("type", text=search_text),
        Action("wait", seconds=0.2),
        Action("key", key="enter"),
        Action("wait", seconds=extra_wait),
    ]


def _export_file_actions(export_search_term: str, filepath: str | None = None) -> List[Action]:
    acts = _search_op(export_search_term, extra_wait=0.4)
    if filepath:
        acts += [
            Action("wait", seconds=0.3),
            Action("type", text=filepath),
            Action("wait", seconds=0.2),
            Action("key", key="enter"),
        ]
    return acts


def visible_steps_for(native_op: str, args: dict) -> List[Action]:
    """Real Blender shortcuts a human would press for this op."""
    acts: List[Action] = [Action("move", *VIEWPORT_CENTER)]

    # 1. Primitives (Mesh)
    if native_op == "add_cube":
        acts += [Action("key", key="shift"), Action("key", key="a")]
        acts += [Action("wait", seconds=0.3)]
        acts += [Action("key", key="down")] * 1
        acts += [Action("key", key="enter")]
        acts += [Action("key", key="enter")]
    elif native_op == "add_cylinder":
        acts += [Action("key", key="shift"), Action("key", key="a")]
        acts += [Action("wait", seconds=0.3)]
        acts += [Action("key", key="down")] * 2
        acts += [Action("key", key="enter")]
    elif native_op == "add_sphere":
        acts += [Action("key", key="shift"), Action("key", key="a")]
        acts += [Action("wait", seconds=0.3)]
        acts += [Action("key", key="down")] * 3
        acts += [Action("key", key="enter")]
    elif native_op == "add_cone":
        acts += _search_op("Add Cone")
    elif native_op == "add_torus":
        acts += _search_op("Add Torus")
    elif native_op == "add_plane":
        acts += _search_op("Add Plane")
    elif native_op == "add_circle":
        acts += _search_op("Add Circle")
    elif native_op == "add_grid":
        acts += _search_op("Add Grid")
    elif native_op == "add_ico_sphere":
        acts += _search_op("Add Ico Sphere")
    elif native_op == "add_monkey":
        acts += _search_op("Add Monkey")
    elif native_op == "add_empty":
        acts += _search_op("Add Empty Plain Axes")
    elif native_op == "add_text":
        acts += _search_op("Add Text")

    # 2. Curve Primitives
    elif native_op in ("add_bezier_curve", "add_bezier_circle"):
        acts += _search_op("Add Bezier")
    elif native_op in ("add_path", "add_nurbs_curve"):
        acts += _search_op("Add Path")

    # 3. Transforms
    elif native_op == "move":
        acts += [Action("key", key="g")]
    elif native_op == "rotate":
        acts += [Action("key", key="r")]
    elif native_op in ("scale", "resize"):
        acts += [Action("key", key="s")]
    elif native_op == "apply_transforms":
        acts += [Action("key", key="ctrl"), Action("key", key="a")]

    # 4. Quadmesh & Topology Cleanup Engine
    elif native_op == "quadriflow_remesh":
        acts += _search_op("Quadriflow Remesh")
    elif native_op == "voxel_remesh":
        acts += _search_op("Voxel Remesh")
    elif native_op == "tris_to_quads":
        acts += [Action("key", key="alt"), Action("key", key="j")]  # Alt+J
    elif native_op == "quads_to_tris":
        acts += [Action("key", key="ctrl"), Action("key", key="t")] # Ctrl+T
    elif native_op == "recalculate_normals":
        acts += [Action("key", key="shift"), Action("key", key="n")] # Shift+N
    elif native_op == "limited_dissolve":
        acts += _search_op("Limited Dissolve")
    elif native_op in ("merge_by_distance", "remove_doubles"):
        acts += _search_op("Merge by Distance")
    elif native_op == "subdivide":
        acts += _search_op("Subdivide")

    # 5. Modifiers
    elif native_op in ("boolean_modifier", "boolean_difference", "boolean_union", "boolean_intersect"):
        acts += _search_op("Boolean")
    elif native_op == "bevel_modifier":
        acts += _search_op("Bevel")
    elif native_op == "subdivision_modifier":
        acts += _search_op("Subdivision Surface")
    elif native_op == "solidify_modifier":
        acts += _search_op("Solidify")
    elif native_op == "mirror_modifier":
        acts += _search_op("Mirror")
    elif native_op == "array_modifier":
        acts += _search_op("Array")
    elif native_op == "screw_modifier":
        acts += _search_op("Screw")
    elif native_op == "wireframe_modifier":
        acts += _search_op("Wireframe")
    elif native_op == "decimate_modifier":
        acts += _search_op("Decimate")
    elif native_op == "weld_modifier":
        acts += _search_op("Weld")

    # 6. Modeling & Lathe Ops
    elif native_op in ("spin", "revolve"):
        acts += _search_op("Spin")
    elif native_op == "extrude":
        acts += [Action("key", key="e")]
    elif native_op == "bevel":
        acts += [Action("key", key="ctrl"), Action("key", key="b")]
    elif native_op == "inset_faces":
        acts += [Action("key", key="i")]
    elif native_op == "loop_cut":
        acts += [Action("key", key="ctrl"), Action("key", key="r")]
    elif native_op == "bridge_edge_loops":
        acts += _search_op("Bridge Edge Loops")
    elif native_op == "shade_smooth":
        acts += _search_op("Shade Smooth")
    elif native_op == "shade_flat":
        acts += _search_op("Shade Flat")

    # 7. Exports
    elif native_op == "export_stl":
        filepath = args.get("filepath") if args else None
        acts += [Action("key", key="ctrl"), Action("key", key="alt"), Action("key", key="e")]
        acts += [Action("wait", seconds=0.5)]
        if filepath:
            acts += [Action("type", text=filepath), Action("key", key="enter")]
    elif native_op == "export_obj":
        acts += _export_file_actions("Export Wavefront (.obj)", args.get("filepath") if args else None)
    elif native_op == "export_ply":
        acts += _export_file_actions("Export Stanford (.ply)", args.get("filepath") if args else None)
    elif native_op == "export_fbx":
        acts += _export_file_actions("Export FBX (.fbx)", args.get("filepath") if args else None)
    elif native_op in ("export_gltf", "export_glb"):
        acts += _export_file_actions("Export glTF 2.0", args.get("filepath") if args else None)
    elif native_op == "export_dae":
        acts += _export_file_actions("Export Collada (Default)", args.get("filepath") if args else None)

    else:
        # op is executed natively via socket bridge
        return []

    return acts


def run_visibly(native_op: str, args: dict, human_speed: bool = True):
    acts = visible_steps_for(native_op, args)
    for a in acts:
        execute(a, human_speed=human_speed)
        time.sleep(0.2)
    return len(acts) > 0
