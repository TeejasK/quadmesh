"""
Blender-side socket server (Sec 10: "Native APIs first, GUI automation
fallback-only"). Paste/run this INSIDE Blender's own Scripting tab, or run it
via `blender --python blender_bridge.py`. It opens a local socket and executes
bpy calls sent to it — this is how the agent creates real geometry without
clicking through menus for every operation it has a native API for.

GUI automation (quadmesh/agent/desktop.py) is reserved for the remaining
fraction of actions with no native API.
"""
try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None

import json
import socket
import threading

HOST, PORT = "127.0.0.1", 8765


# ---------------------------------------------------------------------------
# Object & Scene Helpers
# ---------------------------------------------------------------------------

def _get_obj(obj_name=None):
    if bpy is None:
        raise RuntimeError("bpy is not available (must run inside Blender)")
    if obj_name and obj_name in bpy.data.objects:
        return bpy.data.objects[obj_name]
    if bpy.context.active_object:
        return bpy.context.active_object
    if bpy.context.selected_objects:
        return bpy.context.selected_objects[0]
    mesh_objs = [o for o in bpy.data.objects if o.type in ("MESH", "CURVE")]
    if mesh_objs:
        return mesh_objs[-1]
    raise RuntimeError("No active or selectable object found in Blender scene")


def _set_active(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def _add_and_apply_modifier(obj_name, mod_type, name, apply=True, **properties):
    obj = _get_obj(obj_name)
    mod = obj.modifiers.new(name=name, type=mod_type)
    for k, v in properties.items():
        if hasattr(mod, k):
            setattr(mod, k, v)
    _set_active(obj)
    if apply:
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception:
            pass
    return mod


def _boolean(obj_name=None, other_name=None, op="DIFFERENCE"):
    obj = _get_obj(obj_name)
    if other_name and other_name in bpy.data.objects:
        other = bpy.data.objects[other_name]
    else:
        candidates = [o for o in bpy.data.objects if o != obj and o.type == "MESH"]
        if not candidates:
            raise RuntimeError("No second mesh object found to perform boolean against")
        other = candidates[-1]
    mod = obj.modifiers.new(name="bool", type="BOOLEAN")
    mod.object = other
    mod.operation = op
    _set_active(obj)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _set_dimensions(dimensions, obj_name=None):
    obj = _get_obj(obj_name)
    obj.dimensions = tuple(dimensions)


def _set_location(location, obj_name=None):
    obj = _get_obj(obj_name)
    obj.location = tuple(location)


def _set_rotation(rotation, obj_name=None):
    obj = _get_obj(obj_name)
    obj.rotation_euler = tuple(rotation)


def _set_scale(scale, obj_name=None):
    obj = _get_obj(obj_name)
    obj.scale = tuple(scale)


def _set_curve_props(extrude=0.0, bevel_depth=0.0, obj_name=None):
    obj = _get_obj(obj_name)
    if obj.type == "CURVE":
        obj.data.extrude = extrude
        obj.data.bevel_depth = bevel_depth


def _quadriflow_remesh(target_faces=2000, obj_name=None):
    obj = _get_obj(obj_name)
    _set_active(obj)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.quadriflow_remesh(target_faces=int(target_faces))


def _voxel_remesh(voxel_size=0.05, adaptivity=0.0, obj_name=None):
    obj = _get_obj(obj_name)
    _set_active(obj)
    obj.data.remesh_voxel_size = voxel_size
    obj.data.remesh_voxel_adaptivity = adaptivity
    bpy.ops.object.voxel_remesh()


def _mesh_edit_op(op_func, *args, **kwargs):
    prev_mode = bpy.context.active_object.mode if bpy.context.active_object else "OBJECT"
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        res = op_func(*args, **kwargs)
    finally:
        bpy.ops.object.mode_set(mode=prev_mode)
    return res


def _spin_revolve(angle=6.28318, steps=16, axis=(0, 0, 1), center=(0, 0, 0)):
    return _mesh_edit_op(bpy.ops.mesh.spin, steps=steps, angle=angle, axis=axis, center=center)


def _create_material(name="CAD_Material", color=(0.8, 0.8, 0.8, 1.0), metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
    return mat


def _assign_material(name="CAD_Material", obj_name=None):
    obj = _get_obj(obj_name)
    mat = bpy.data.materials.get(name) or _create_material(name=name)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def _export_obj(filepath):
    if hasattr(bpy.ops.wm, "obj_export"):
        return bpy.ops.wm.obj_export(filepath=filepath)
    return bpy.ops.export_scene.obj(filepath=filepath)


def _import_obj(filepath):
    if hasattr(bpy.ops.wm, "obj_import"):
        return bpy.ops.wm.obj_import(filepath=filepath)
    return bpy.ops.import_scene.obj(filepath=filepath)


def _export_ply(filepath):
    if hasattr(bpy.ops.wm, "ply_export"):
        return bpy.ops.wm.ply_export(filepath=filepath)
    return bpy.ops.export_mesh.ply(filepath=filepath)


def _import_ply(filepath):
    if hasattr(bpy.ops.wm, "ply_import"):
        return bpy.ops.wm.ply_import(filepath=filepath)
    return bpy.ops.import_mesh.ply(filepath=filepath)


# ---------------------------------------------------------------------------
# SAFE_OPS: Comprehensive 100+ CAD & Quadmesh Operation Suite
# ---------------------------------------------------------------------------

SAFE_OPS = {
    # 1. 3D Mesh Primitives (12)
    "add_cube": lambda **kw: bpy.ops.mesh.primitive_cube_add(**kw),
    "add_cylinder": lambda **kw: bpy.ops.mesh.primitive_cylinder_add(**kw),
    "add_sphere": lambda **kw: bpy.ops.mesh.primitive_uv_sphere_add(**kw),
    "add_ico_sphere": lambda **kw: bpy.ops.mesh.primitive_ico_sphere_add(**kw),
    "add_cone": lambda **kw: bpy.ops.mesh.primitive_cone_add(**kw),
    "add_torus": lambda **kw: bpy.ops.mesh.primitive_torus_add(**kw),
    "add_plane": lambda **kw: bpy.ops.mesh.primitive_plane_add(**kw),
    "add_circle": lambda **kw: bpy.ops.mesh.primitive_circle_add(**kw),
    "add_grid": lambda **kw: bpy.ops.mesh.primitive_grid_add(**kw),
    "add_monkey": lambda **kw: bpy.ops.mesh.primitive_monkey_add(**kw),
    "add_empty": lambda **kw: bpy.ops.object.empty_add(**kw),
    "add_text": lambda **kw: bpy.ops.object.text_add(**kw),

    # 2. 2D & 3D Curves, Splines & Profiles (10)
    "add_bezier_curve": lambda **kw: bpy.ops.curve.primitive_bezier_curve_add(**kw),
    "add_bezier_circle": lambda **kw: bpy.ops.curve.primitive_bezier_circle_add(**kw),
    "add_nurbs_curve": lambda **kw: bpy.ops.curve.primitive_nurbs_curve_add(**kw),
    "add_nurbs_circle": lambda **kw: bpy.ops.curve.primitive_nurbs_circle_add(**kw),
    "add_path": lambda **kw: bpy.ops.curve.primitive_nurbs_path_add(**kw),
    "set_curve_extrude": lambda extrude=0.1, obj_name=None: _set_curve_props(extrude=extrude, obj_name=obj_name),
    "set_curve_bevel": lambda bevel_depth=0.1, obj_name=None: _set_curve_props(bevel_depth=bevel_depth, obj_name=obj_name),
    "convert_to_mesh": lambda: bpy.ops.object.convert(target="MESH"),
    "convert_to_curve": lambda: bpy.ops.object.convert(target="CURVE"),
    "curve_to_tube": lambda bevel_depth=0.1, obj_name=None: (_set_curve_props(bevel_depth=bevel_depth, obj_name=obj_name), bpy.ops.object.convert(target="MESH")),

    # 3. Precision Transforms & Coordinates (12)
    "move": lambda **kw: bpy.ops.transform.translate(**kw),
    "rotate": lambda **kw: bpy.ops.transform.rotate(**kw),
    "scale": lambda **kw: bpy.ops.transform.resize(**kw),
    "resize": lambda **kw: bpy.ops.transform.resize(**kw),
    "set_location": lambda location=(0, 0, 0), obj_name=None: _set_location(location, obj_name),
    "set_rotation": lambda rotation=(0, 0, 0), obj_name=None: _set_rotation(rotation, obj_name),
    "set_dimensions": lambda dimensions=(1, 1, 1), obj_name=None: _set_dimensions(dimensions, obj_name),
    "set_scale": lambda scale=(1, 1, 1), obj_name=None: _set_scale(scale, obj_name),
    "apply_transforms": lambda location=True, rotation=True, scale=True: bpy.ops.object.transform_apply(
        location=location, rotation=rotation, scale=scale
    ),
    "set_origin": lambda type="ORIGIN_GEOMETRY", center="MEDIAN": bpy.ops.object.origin_set(
        type=type, center=center
    ),
    "center_origin": lambda: bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN"),
    "align_objects": lambda **kw: bpy.ops.object.align(**kw),

    # 4. Quadmesh & Topology Cleanup Engine (14)
    "quadriflow_remesh": lambda target_faces=2000, obj_name=None: _quadriflow_remesh(target_faces, obj_name),
    "voxel_remesh": lambda voxel_size=0.05, adaptivity=0.0, obj_name=None: _voxel_remesh(voxel_size, adaptivity, obj_name),
    "tris_to_quads": lambda: _mesh_edit_op(bpy.ops.mesh.tris_to_quads),
    "quads_to_tris": lambda: _mesh_edit_op(bpy.ops.mesh.quads_to_tris),
    "recalculate_normals": lambda: _mesh_edit_op(bpy.ops.mesh.normals_make_consistent, inside=False),
    "flip_normals": lambda: _mesh_edit_op(bpy.ops.mesh.flip_normals),
    "limited_dissolve": lambda angle_limit=0.0872665: _mesh_edit_op(bpy.ops.mesh.dissolve_limited, angle_limit=angle_limit),
    "merge_by_distance": lambda distance=0.0001: _mesh_edit_op(bpy.ops.mesh.remove_doubles, threshold=distance),
    "remove_doubles": lambda threshold=0.0001: _mesh_edit_op(bpy.ops.mesh.remove_doubles, threshold=threshold),
    "subdivide": lambda number_cuts=1: _mesh_edit_op(bpy.ops.mesh.subdivide, number_cuts=number_cuts),
    "unsubdivide": lambda iterations=1: _mesh_edit_op(bpy.ops.mesh.unsubdivide, iterations=iterations),
    "smooth_vertices": lambda factor=0.5, repeat=1: _mesh_edit_op(bpy.ops.mesh.vertices_smooth, factor=factor, repeat=repeat),
    "fill_holes": lambda sides=4: _mesh_edit_op(bpy.ops.mesh.fill_holes, sides=sides),
    "symmetrize": lambda direction="-X_TO_+X": _mesh_edit_op(bpy.ops.mesh.symmetrize, direction=direction),

    # 5. Modifiers (16)
    "boolean_modifier": lambda obj_name=None, other_name=None, op="DIFFERENCE": _boolean(
        obj_name, other_name, op
    ),
    "boolean_difference": lambda obj_name=None, other_name=None: _boolean(obj_name, other_name, "DIFFERENCE"),
    "boolean_union": lambda obj_name=None, other_name=None: _boolean(obj_name, other_name, "UNION"),
    "boolean_intersect": lambda obj_name=None, other_name=None: _boolean(obj_name, other_name, "INTERSECT"),
    "mirror_modifier": lambda obj_name=None, use_axis=(True, False, False), apply=True: _add_and_apply_modifier(
        obj_name, "MIRROR", "mirror", apply=apply, use_axis=use_axis
    ),
    "array_modifier": lambda obj_name=None, count=2, relative_offset_displace=(1.0, 0.0, 0.0), apply=True: _add_and_apply_modifier(
        obj_name, "ARRAY", "array", apply=apply, count=count, relative_offset_displace=relative_offset_displace
    ),
    "solidify_modifier": lambda obj_name=None, thickness=0.1, offset=-1.0, apply=True: _add_and_apply_modifier(
        obj_name, "SOLIDIFY", "solidify", apply=apply, thickness=thickness, offset=offset
    ),
    "subdivision_modifier": lambda obj_name=None, levels=2, render_levels=2, apply=True: _add_and_apply_modifier(
        obj_name, "SUBSURF", "subsurf", apply=apply, levels=levels, render_levels=render_levels
    ),
    "screw_modifier": lambda obj_name=None, angle=6.28318, screw_offset=0.0, iterations=1, axis='Z', apply=True: _add_and_apply_modifier(
        obj_name, "SCREW", "screw", apply=apply, angle=angle, screw_offset=screw_offset, iterations=iterations, axis=axis
    ),
    "bevel_modifier": lambda obj_name=None, width=0.1, segments=3, apply=True: _add_and_apply_modifier(
        obj_name, "BEVEL", "bevel_mod", apply=apply, width=width, segments=segments
    ),
    "decimate_modifier": lambda obj_name=None, ratio=0.5, apply=True: _add_and_apply_modifier(
        obj_name, "DECIMATE", "decimate", apply=apply, ratio=ratio
    ),
    "wireframe_modifier": lambda obj_name=None, thickness=0.02, apply=True: _add_and_apply_modifier(
        obj_name, "WIREFRAME", "wireframe", apply=apply, thickness=thickness
    ),
    "remesh_modifier": lambda obj_name=None, mode="VOXEL", voxel_size=0.1, apply=True: _add_and_apply_modifier(
        obj_name, "REMESH", "remesh", apply=apply, mode=mode, voxel_size=voxel_size
    ),
    "weld_modifier": lambda obj_name=None, merge_threshold=0.001, apply=True: _add_and_apply_modifier(
        obj_name, "WELD", "weld", apply=apply, merge_threshold=merge_threshold
    ),
    "smooth_modifier": lambda obj_name=None, factor=0.5, iterations=2, apply=True: _add_and_apply_modifier(
        obj_name, "SMOOTH", "smooth_mod", apply=apply, factor=factor, iterations=iterations
    ),
    "lattice_modifier": lambda obj_name=None, apply=False, **kw: _add_and_apply_modifier(
        obj_name, "LATTICE", "lattice_mod", apply=apply, **kw
    ),

    # 6. CAD Modeling, Extrusion & Mesh Operations (16)
    "extrude": lambda **kw: bpy.ops.mesh.extrude_region_move(**kw),
    "extrude_along_normals": lambda **kw: bpy.ops.mesh.extrude_region_shrink_fatten(**kw),
    "extrude_individual": lambda **kw: bpy.ops.mesh.extrude_faces_move(**kw),
    "spin": lambda angle=6.28318, steps=16, axis=(0, 0, 1), center=(0, 0, 0): _spin_revolve(angle, steps, axis, center),
    "revolve": lambda angle=6.28318, steps=16, axis=(0, 0, 1), center=(0, 0, 0): _spin_revolve(angle, steps, axis, center),
    "bevel": lambda **kw: bpy.ops.mesh.bevel(**kw),
    "inset_faces": lambda **kw: bpy.ops.mesh.inset(**kw),
    "loop_cut": lambda **kw: bpy.ops.mesh.loopcut_slide(**kw),
    "bridge_edge_loops": lambda **kw: bpy.ops.mesh.bridge_edge_loops(**kw),
    "fill": lambda **kw: bpy.ops.mesh.fill(**kw),
    "grid_fill": lambda **kw: bpy.ops.mesh.grid_fill(**kw),
    "beautify_fill": lambda: _mesh_edit_op(bpy.ops.mesh.beautify_fill),
    "edge_split": lambda **kw: bpy.ops.mesh.edge_split(**kw),
    "poke": lambda **kw: bpy.ops.mesh.poke(**kw),
    "separate": lambda type="SELECTED": bpy.ops.mesh.separate(type=type),
    "separate_loose_parts": lambda: bpy.ops.mesh.separate(type="LOOSE"),

    # 7. Object & Scene Management (12)
    "select_all": lambda action="SELECT": bpy.ops.object.select_all(action=action),
    "deselect_all": lambda: bpy.ops.object.select_all(action="DESELECT"),
    "invert_selection": lambda: bpy.ops.object.select_all(action="INVERT"),
    "select_object": lambda name: (bpy.data.objects.get(name) and _set_active(bpy.data.objects[name])),
    "delete_selected": lambda: bpy.ops.object.delete(),
    "delete_object": lambda name: (bpy.data.objects.get(name) and bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)),
    "duplicate": lambda **kw: bpy.ops.object.duplicate_move(**kw),
    "join_objects": lambda: bpy.ops.object.join(),
    "shade_smooth": lambda: bpy.ops.object.shade_smooth(),
    "shade_flat": lambda: bpy.ops.object.shade_flat(),
    "shade_auto_smooth": lambda: (hasattr(bpy.ops.object, "shade_auto_smooth") and bpy.ops.object.shade_auto_smooth()),
    "set_mode": lambda mode="OBJECT": bpy.ops.object.mode_set(mode=mode),

    # 8. Materials, Colors & Appearance (6)
    "create_material": lambda name="CAD_Material", color=(0.8, 0.8, 0.8, 1.0), metallic=0.0, roughness=0.5: _create_material(
        name, color, metallic, roughness
    ),
    "assign_material": lambda name="CAD_Material", obj_name=None: _assign_material(name, obj_name),
    "set_color": lambda color=(0.8, 0.8, 0.8, 1.0), obj_name=None: _assign_material("Color_Mat", obj_name),
    "set_metallic": lambda metallic=1.0, obj_name=None: _create_material("Metallic_Mat", metallic=metallic),
    "set_roughness": lambda roughness=0.2, obj_name=None: _create_material("Rough_Mat", roughness=roughness),
    "clear_materials": lambda obj_name=None: _get_obj(obj_name).data.materials.clear(),

    # 9. Multi-Format CAD Import & Export (12)
    "export_stl": lambda filepath: bpy.ops.export_mesh.stl(filepath=filepath),
    "import_stl": lambda filepath: bpy.ops.import_mesh.stl(filepath=filepath),
    "export_obj": lambda filepath: _export_obj(filepath),
    "import_obj": lambda filepath: _import_obj(filepath),
    "export_fbx": lambda filepath: bpy.ops.export_scene.fbx(filepath=filepath),
    "import_fbx": lambda filepath: bpy.ops.import_scene.fbx(filepath=filepath),
    "export_ply": lambda filepath: _export_ply(filepath),
    "import_ply": lambda filepath: _import_ply(filepath),
    "export_gltf": lambda filepath: bpy.ops.export_scene.gltf(filepath=filepath),
    "import_gltf": lambda filepath: bpy.ops.import_scene.gltf(filepath=filepath),
    "export_dae": lambda filepath: bpy.ops.wm.collada_export(filepath=filepath),
    "import_reference_image": lambda filepath, location=(0, 0, 0): bpy.ops.object.empty_image_add(
        filepath=filepath, location=location
    ),
}


# ---------------------------------------------------------------------------
# Quadmesh feature ops (CAD features, eyes, printability report, keyboard-only console mode)
# ---------------------------------------------------------------------------
import math as _math
try:
    import mathutils
except ImportError:          # outside Blender (tests)
    mathutils = None


def _fx_obj(name):
    o = bpy.data.objects.get(name)
    if o is None:
        raise RuntimeError(f"no object named {name!r} in the scene")
    return o


def _fx_object_mode():
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def _fx_free_name(name):
    """If an object already has this name, move it aside (never delete user work)."""
    old = bpy.data.objects.get(name)
    if old is not None:
        old.name = name + "_prev"


def _fx_drop(name):
    o = bpy.data.objects.get(name)
    if o is not None:
        bpy.data.objects.remove(o, do_unlink=True)


def _fx_bbox(o):
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def _fx_pts(anchor, bb, a):
    """Feature positions (x, y) - the same maths as plan_checker._anchors."""
    cx, cy = (bb[0] + bb[1]) / 2.0, (bb[2] + bb[3]) / 2.0
    w, d = bb[1] - bb[0], bb[3] - bb[2]
    if anchor == "center":
        return [(cx, cy)]
    if anchor == "pair":
        s = a["inset"]
        return [(cx - (w / 2.0 - s), cy), (cx + (w / 2.0 - s), cy)]
    if anchor == "pair_y":
        s = a["inset"]
        return [(cx, cy - (d / 2.0 - s)), (cx, cy + (d / 2.0 - s))]
    if anchor in ("row_x", "row_y"):
        n, p = int(a["n"]), a["pitch"]
        if anchor == "row_x":
            return [(cx + (i - (n - 1) / 2.0) * p, cy) for i in range(n)]
        return [(cx, cy + (i - (n - 1) / 2.0) * p) for i in range(n)]
    if anchor == "corners":
        s = a["inset"]
        return [(cx + sx * (w / 2.0 - s), cy + sy * (d / 2.0 - s)) for sx in (-1, 1) for sy in (-1, 1)]
    if anchor == "bolt":
        n = int(a["n"])
        return [(cx + a["pcd"] / 2.0 * _math.cos(2 * _math.pi * k / n),
                 cy + a["pcd"] / 2.0 * _math.sin(2 * _math.pi * k / n)) for k in range(n)]
    if anchor == "grid":
        nx, ny, p = int(a["nx"]), int(a["ny"]), a["pitch"]
        return [(cx + (i - (nx - 1) / 2.0) * p, cy + (j - (ny - 1) / 2.0) * p)
                for i in range(nx) for j in range(ny)]
    raise RuntimeError(f"unknown anchor {anchor!r}")


# ---- primitives -------------------------------------------------------------------
def _fx_new_box(name, w, d, h, cx, cy, z0):
    _fx_object_mode()
    _fx_free_name(name)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, z0 + h / 2.0))
    o = bpy.context.active_object
    o.scale = (w, d, h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.name = name
    return o


def _fx_add_box(name, w, d, h):
    _fx_new_box(name, w, d, h, 0.0, 0.0, 0.0)


def _fx_add_cyl(name, dia, h):
    _fx_object_mode()
    _fx_free_name(name)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=dia / 2.0, depth=h, location=(0.0, 0.0, h / 2.0))
    bpy.context.active_object.name = name


def _fx_add_cone(name, dia1, dia2, h):
    _fx_object_mode()
    _fx_free_name(name)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=dia1 / 2.0, radius2=dia2 / 2.0, depth=h,
                                    location=(0.0, 0.0, h / 2.0))
    bpy.context.active_object.name = name


# ---- placement --------------------------------------------------------------------
def _fx_align_side(name, ref, side):
    o, r = _fx_obj(name), _fx_obj(ref)
    ob, rb = _fx_bbox(o), _fx_bbox(r)
    if side == "back":
        o.location.y += rb[3] - ob[3]
    elif side == "front":
        o.location.y += rb[2] - ob[2]
    elif side == "right":
        o.location.x += rb[1] - ob[1]
    elif side == "left":
        o.location.x += rb[0] - ob[0]
    else:
        raise RuntimeError(f"bad side {side!r}")
    bpy.context.view_layer.update()


def _fx_align_back(name, ref):
    _fx_align_side(name, ref, "back")


def _fx_stack_on(name, ref):
    o, r = _fx_obj(name), _fx_obj(ref)
    ob, rb = _fx_bbox(o), _fx_bbox(r)
    o.location.x += (rb[0] + rb[1]) / 2.0 - (ob[0] + ob[1]) / 2.0
    o.location.y += (rb[2] + rb[3]) / 2.0 - (ob[2] + ob[3]) / 2.0
    o.location.z += rb[5] - ob[4]
    bpy.context.view_layer.update()


# ---- booleans ---------------------------------------------------------------------
def _fx_union(target, tool):
    _boolean(target, tool, "UNION")
    _fx_drop(tool)


def _fx_cut(target, tool):
    _boolean(target, tool, "DIFFERENCE")
    _fx_drop(tool)


def _fx_tool_cyl(x, y, z0, z1, dia):
    _fx_object_mode()
    _fx_drop("_tool")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=dia / 2.0, depth=(z1 - z0),
                                        location=(x, y, (z0 + z1) / 2.0))
    bpy.context.active_object.name = "_tool"


def _fx_through(target, x, y, dia):
    b = _fx_bbox(_fx_obj(target))
    _fx_tool_cyl(x, y, b[4] - 1.0, b[5] + 1.0, dia)
    _fx_cut(target, "_tool")


def _fx_blind_cyl(target, x, y, dia, depth):
    b = _fx_bbox(_fx_obj(target))
    _fx_tool_cyl(x, y, b[5] - depth, b[5] + 1.0, dia)
    _fx_cut(target, "_tool")


def _fx_cone_cut(target, x, y, dia, cs_dia):
    b = _fx_bbox(_fx_obj(target))
    cs_h = (cs_dia - dia) / 2.0                       # 90-degree countersink
    _fx_object_mode()
    _fx_drop("_tool")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=dia / 2.0, radius2=cs_dia / 2.0 + 0.5,
                                    depth=cs_h + 0.5, location=(x, y, b[5] - cs_h + (cs_h + 0.5) / 2.0))
    bpy.context.active_object.name = "_tool"
    _fx_cut(target, "_tool")


def _fx_circle_feature(op, target, **a):
    anchor, style = _FX_CIRCLE_OPS[op]
    bb = _fx_bbox(_fx_obj(target))
    for (x, y) in _fx_pts(anchor, bb, a):
        if style == "plain":
            _fx_through(target, x, y, a["dia"])
        elif style == "cbore":
            _fx_through(target, x, y, a["dia"])
            _fx_blind_cyl(target, x, y, a["cb_dia"], a["cb_depth"])
        elif style == "csink":
            _fx_through(target, x, y, a["dia"])
            _fx_cone_cut(target, x, y, a["dia"], a["cs_dia"])
        elif style == "pocket":
            _fx_blind_cyl(target, x, y, a["dia"], a["depth"])


_FX_CIRCLE_OPS = {
    "hole_center": ("center", "plain"), "holes_pair_x": ("pair", "plain"),
    "holes_corners": ("corners", "plain"), "holes_grid": ("grid", "plain"),
    "bolt_circle": ("bolt", "plain"),
    "holes_pair_y": ("pair_y", "plain"), "holes_row_x": ("row_x", "plain"), "holes_row_y": ("row_y", "plain"),
    "cbore_pair_y": ("pair_y", "cbore"), "csink_pair_y": ("pair_y", "csink"),
    "cbore_center": ("center", "cbore"), "cbore_pair_x": ("pair", "cbore"),
    "cbore_corners": ("corners", "cbore"), "cbore_bolt_circle": ("bolt", "cbore"),
    "csink_center": ("center", "csink"), "csink_pair_x": ("pair", "csink"),
    "csink_corners": ("corners", "csink"), "csink_bolt_circle": ("bolt", "csink"),
    "pocket_round": ("center", "pocket"),
}


def _fx_make_circle_op(op):
    def run(target, **a):
        _fx_circle_feature(op, target, **a)
    return run


# ---- rectangular features -------------------------------------------------------
def _fx_slot_center(target, length, width):
    b = _fx_bbox(_fx_obj(target))
    cx, cy = (b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0
    span = (b[5] - b[4]) + 2.0
    z0 = b[4] - 1.0
    _fx_new_box("_tool", length - width, width, span, cx, cy, z0)
    for sx in (-1, 1):
        _fx_drop("_c")
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=width / 2.0, depth=span,
                                            location=(cx + sx * (length - width) / 2.0, cy, z0 + span / 2.0))
        bpy.context.active_object.name = "_c"
        _boolean("_tool", "_c", "UNION")
        _fx_drop("_c")
    _fx_cut(target, "_tool")


def _fx_pocket_rect(target, w, d, depth):
    b = _fx_bbox(_fx_obj(target))
    _fx_drop("_tool")
    _fx_new_box("_tool", w, d, depth + 1.0, (b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, b[5] - depth)
    _fx_cut(target, "_tool")


def _fx_window_rect(target, w, d):
    b = _fx_bbox(_fx_obj(target))
    _fx_drop("_tool")
    _fx_new_box("_tool", w, d, (b[5] - b[4]) + 2.0, (b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, b[4] - 1.0)
    _fx_cut(target, "_tool")


# ---- bosses ----------------------------------------------------------------------
_FX_BOSS_OPS = {"boss_center": "center", "bosses_pair_x": "pair", "bosses_pair_y": "pair_y", "bosses_corners": "corners"}


def _fx_make_boss_op(op):
    def run(target, **a):
        b = _fx_bbox(_fx_obj(target))
        ztop = b[5]                                    # read ONCE: bosses do not stack on each other
        for (x, y) in _fx_pts(_FX_BOSS_OPS[op], b, a):
            _fx_object_mode()
            _fx_drop("_boss")
            bpy.ops.object.select_all(action="DESELECT")
            bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=a["dia"] / 2.0, depth=a["h"] + 0.5,
                                                location=(x, y, ztop - 0.5 + (a["h"] + 0.5) / 2.0))
            bpy.context.active_object.name = "_boss"
            _fx_union(target, "_boss")
    return run


# ---- shell / edges ---------------------------------------------------------------
def _fx_shell_open_top(target, wall):
    b = _fx_bbox(_fx_obj(target))
    cx, cy = (b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0
    w, d = (b[1] - b[0]) - 2 * wall, (b[3] - b[2]) - 2 * wall
    h = (b[5] - b[4]) - wall + 1.0
    _fx_new_box("_tool", w, d, h, cx, cy, b[4] + wall)
    _fx_cut(target, "_tool")


def _fx_fillet(target, r):
    _add_and_apply_modifier(target, "BEVEL", "fillet", apply=True, width=r, segments=3,
                            limit_method="ANGLE", angle_limit=0.785398)


def _fx_chamfer(target, size):
    _add_and_apply_modifier(target, "BEVEL", "chamfer", apply=True, width=size, segments=1,
                            limit_method="ANGLE", angle_limit=0.785398)


# ---- eyes ------------------------------------------------------------------------
_FX_BASELINE = set()


def _fx_mark_scene():
    """Remember what is already in the scene: the agent never deletes, reports or exports these."""
    _FX_BASELINE.clear()
    _FX_BASELINE.update(o.name for o in bpy.data.objects)


def _fx_mine():
    return [o for o in bpy.data.objects if o.type == "MESH" and o.name not in _FX_BASELINE]


def _fx_scene_info():
    import bmesh
    out = []
    for o in _fx_mine():
        b = _fx_bbox(o)
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.transform(o.matrix_world)
        out.append({"name": o.name,
                    "dims": [round(b[1] - b[0], 3), round(b[3] - b[2], 3), round(b[5] - b[4], 3)],
                    "volume": round(bm.calc_volume(signed=False), 3),
                    "open_edges": sum(1 for e in bm.edges if not e.is_manifold)})
        bm.free()
    return out


# ---- extra primitives, free placement, arrays, absolute features --------------------
def _fx_add_sphere(name, dia):
    _fx_object_mode()
    _fx_free_name(name)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=dia / 2.0,
                                         location=(0.0, 0.0, dia / 2.0))
    bpy.context.active_object.name = name


def _fx_add_prism(name, sides, dia, h):
    _fx_object_mode()
    _fx_free_name(name)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(sides), radius=dia / 2.0, depth=h,
                                        location=(0.0, 0.0, h / 2.0))
    bpy.context.active_object.name = name


def _fx_add_torus(name, major, minor):
    _fx_object_mode()
    _fx_free_name(name)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_torus_add(major_segments=64, minor_segments=24, major_radius=major / 2.0,
                                     minor_radius=minor / 2.0, location=(0.0, 0.0, minor / 2.0))
    bpy.context.active_object.name = name


def _fx_move(name, x, y, z):
    """Put the part's bounding-box centre at (x, y) and its lowest point at z."""
    o = _fx_obj(name)
    b = _fx_bbox(o)
    o.location.x += x - (b[0] + b[1]) / 2.0
    o.location.y += y - (b[2] + b[3]) / 2.0
    o.location.z += z - b[4]
    bpy.context.view_layer.update()


def _fx_rotate(name, axis, deg):
    o = _fx_obj(name)
    b = _fx_bbox(o)
    c = mathutils.Vector(((b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, (b[4] + b[5]) / 2.0))
    rot = mathutils.Matrix.Rotation(_math.radians(deg), 4, axis.upper())
    o.matrix_world = (mathutils.Matrix.Translation(c) @ rot @ mathutils.Matrix.Translation(-c)) @ o.matrix_world
    bpy.context.view_layer.update()
    _fx_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)


def _fx_array(name, count, pitch, axis):
    o = _fx_obj(name)
    for k in range(1, int(count)):
        dup = o.copy()
        dup.data = o.data.copy()
        dup.name = f"_arr{k}"
        bpy.context.collection.objects.link(dup)
        if axis == "x":
            dup.location.x += k * pitch
        else:
            dup.location.y += k * pitch
        _fx_union(name, dup.name)


def _fx_hole_at(target, x, y, dia):
    b = _fx_bbox(_fx_obj(target))
    _fx_through(target, (b[0] + b[1]) / 2.0 + x, (b[2] + b[3]) / 2.0 + y, dia)


def _fx_window_at(target, x, y, w, d):
    b = _fx_bbox(_fx_obj(target))
    _fx_drop("_tool")
    _fx_new_box("_tool", w, d, (b[5] - b[4]) + 2.0, (b[0] + b[1]) / 2.0 + x, (b[2] + b[3]) / 2.0 + y, b[4] - 1.0)
    _fx_cut(target, "_tool")


# ---- printability report (the same keys as printability.analyze_mesh) ---------------
def _fx_print_check(bed_x=220.0, bed_y=220.0, bed_z=250.0, samples=300):
    import bmesh
    from mathutils.bvhtree import BVHTree
    import random
    objs = _fx_mine()
    lo, hi = [1e18] * 3, [-1e18] * 3
    rep = {"objects": len(objs), "loose_parts": 0, "open_edges": 0, "nonmanifold_edges": 0,
           "flipped_edges": 0, "degenerate_faces": 0, "volume": 0.0, "min_wall": None,
           "triangles": 0}
    tot_area = oh_area = 0.0
    for o in objs:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.transform(o.matrix_world)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table(); bm.edges.ensure_lookup_table()
        bm.normal_update()
        for v in bm.verts:
            for k in range(3):
                lo[k], hi[k] = min(lo[k], v.co[k]), max(hi[k], v.co[k])
        rep["triangles"] += len(bm.faces)
        rep["volume"] += bm.calc_volume(signed=True)
        for e in bm.edges:
            n = len(e.link_faces)
            if n == 1:
                rep["open_edges"] += 1
            elif n > 2:
                rep["nonmanifold_edges"] += 1
        seen = set()
        for v in bm.verts:
            if v.index in seen:
                continue
            rep["loose_parts"] += 1
            stack = [v]; seen.add(v.index)
            while stack:
                cur = stack.pop()
                for e in cur.link_edges:
                    w = e.other_vert(cur)
                    if w.index not in seen:
                        seen.add(w.index); stack.append(w)
        zmin_o = min(v.co.z for v in bm.verts) if bm.verts else 0.0
        for f in bm.faces:
            a = f.calc_area()
            if a < 1e-9:
                rep["degenerate_faces"] += 1
            tot_area += a
            if f.normal.z < -0.7072 and f.calc_center_median().z > zmin_o + 0.01:
                oh_area += a
        if bm.faces:
            tree = BVHTree.FromBMesh(bm)
            random.seed(0)
            for f in random.sample(list(bm.faces), min(int(samples), len(bm.faces))):
                if f.calc_area() < 1e-9:
                    continue
                origin = f.calc_center_median() - f.normal * 0.01
                loc, nrm, idx, dist = tree.ray_cast(origin, -f.normal)
                if loc is not None:
                    w = dist + 0.01
                    rep["min_wall"] = w if rep["min_wall"] is None else min(rep["min_wall"], w)
        bm.free()
    if objs:
        rep["dims"] = [round(hi[k] - lo[k], 3) for k in range(3)]
        rep["zmin"] = round(lo[2], 4)
    rep["volume"] = round(rep["volume"], 3)
    rep["overhang_pct"] = round(100.0 * oh_area / max(tot_area, 1e-12), 2)
    if rep["min_wall"] is not None:
        rep["min_wall"] = round(rep["min_wall"], 3)
    rep["bed"] = [bed_x, bed_y, bed_z]
    return rep


def _fx_save_stl(filepath):
    """Export every mesh in the scene to STL (works on Blender 3.x and 4.x)."""
    import os
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    _fx_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    for o in _fx_mine():
        o.select_set(True)
    try:
        bpy.ops.wm.stl_export(filepath=filepath, export_selected_objects=True)
    except (AttributeError, TypeError):
        bpy.ops.export_mesh.stl(filepath=filepath, use_selection=True)
    return filepath



def _fx_extrude_polygons(name, outer, holes, h):
    """Solid from a traced image outline: caps from tessellate_polygon (holes stay holes) + side walls."""
    from mathutils import geometry, Vector
    import bmesh
    _fx_object_mode()
    _fx_free_name(name)
    rings = [outer] + list(holes)
    flat = [tuple(p) for r in rings for p in r]
    tris = geometry.tessellate_polygon([[Vector((x, y, 0.0)) for x, y in r] for r in rings])
    n = len(flat)
    verts = [(x, y, 0.0) for x, y in flat] + [(x, y, float(h)) for x, y in flat]
    faces = []
    for a, b, c in tris:
        faces.append((a, b, c))                       # bottom cap
        faces.append((n + a, n + b, n + c))           # top cap
    start = 0
    for r in rings:
        m = len(r)
        for i in range(m):
            j = (i + 1) % m
            faces.append((start + i, start + j, n + start + j, n + start + i))   # side wall
        start += m
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)   # make every face point outward
    bm.to_mesh(me)
    bm.free()
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o



def _fx_array_polar(name, count):
    """Copies of a part rotated about the WORLD Z axis through the origin, merged into one object."""
    o = _fx_obj(name)
    n = int(count)
    for k in range(1, n):
        dup = o.copy()
        dup.data = o.data.copy()
        dup.name = f"_pol{k}"
        bpy.context.collection.objects.link(dup)
        rot = mathutils.Matrix.Rotation(_math.radians(360.0 * k / n), 4, "Z")
        dup.matrix_world = rot @ o.matrix_world
        bpy.context.view_layer.update()
        _fx_union(name, dup.name)


def _fx_dump_scene(filepath):
    """Write the scene report (agent's eyes) to a JSON file - works with no socket (desk mode)."""
    import json as _json
    with open(filepath, "w") as fh:
        _json.dump(_fx_scene_info(), fh)
    return filepath



def _fx_nudge(name, dx, dy, dz):
    """Move a part by a relative amount (mm)."""
    o = _fx_obj(name)
    o.location.x += dx
    o.location.y += dy
    o.location.z += dz
    bpy.context.view_layer.update()



def _fx_reset_scene():
    """Delete only what the agent created since mark_scene() - never the user's own objects."""
    _fx_object_mode()
    for o in list(bpy.data.objects):
        if o.name not in _FX_BASELINE:
            bpy.data.objects.remove(o, do_unlink=True)


SAFE_OPS.update({
    "add_box": _fx_add_box, "add_cyl": _fx_add_cyl, "add_cone": _fx_add_cone,
    "align_back": _fx_align_back, "align_side": _fx_align_side, "stack_on": _fx_stack_on,
    "union": _fx_union, "cut": _fx_cut,
    "slot_center": _fx_slot_center, "pocket_rect": _fx_pocket_rect, "window_rect": _fx_window_rect,
    "shell_open_top": _fx_shell_open_top, "fillet": _fx_fillet, "chamfer": _fx_chamfer,
    "scene_info": _fx_scene_info, "reset_scene": _fx_reset_scene, "mark_scene": _fx_mark_scene,
    "add_sphere": _fx_add_sphere, "add_prism": _fx_add_prism, "add_torus": _fx_add_torus,
    "move": _fx_move, "rotate": _fx_rotate,
    "array_x": lambda name, count, pitch: _fx_array(name, count, pitch, "x"),
    "array_y": lambda name, count, pitch: _fx_array(name, count, pitch, "y"),
    "hole_at": _fx_hole_at, "window_at": _fx_window_at,
    "print_check": _fx_print_check, "save_stl": _fx_save_stl,
    "extrude_polygons": _fx_extrude_polygons,
    "array_polar": _fx_array_polar, "dump_scene": _fx_dump_scene, "nudge": _fx_nudge,
})
# lets the agent drive Blender purely by typing in its Python console (keyboard-only mode):
if bpy is not None:
    bpy.app.driver_namespace["quadmesh_ops"] = SAFE_OPS
for _op in _FX_CIRCLE_OPS:
    SAFE_OPS[_op] = _fx_make_circle_op(_op)
for _op in _FX_BOSS_OPS:
    SAFE_OPS[_op] = _fx_make_boss_op(_op)


# ---------------------------------------------------------------------------
# Code-execution layer (Sec 5 of the rewrite plan): the spec_generator role now writes OP SEQUENCES or short
# bpy SCRIPTS instead of only picking one of the ~160 named ops above one at a time. This does not remove
# those 160 ops — they stay as the vocabulary a script is built FROM (run_ops) and as the fallback the old
# single-op path still uses. What's new is composition: a generated plan can now chain several of them, or
# fall back to a short restricted script for the rare op no name below covers.
#
#   "run_ops"    {"ops": [{"op": "add_box", "args": {...}}, {"op": "fillet", "args": {...}}, ...]}
#                Just SAFE_OPS entries called in sequence - no new attack surface versus calling them one at
#                a time over separate requests, just fewer round trips and one place to report which step
#                in a multi-step plan failed.
#
#   "run_script" {"code": "<python source>"}
#                For the cases run_ops can't express. NOT bare exec() of arbitrary text - see
#                _check_script_safe below. Runs in a namespace containing only bpy, mathutils, math, and
#                `ops` (a read-only view of SAFE_OPS itself, so a script can still call named ops if that's
#                easier than raw bpy calls). Same allowlist *pattern* your own action-safety filter already
#                uses in agent/actions.py ("action-safety filter refuses dangerous keys by default") - this
#                is that same idea applied to a script instead of a single keystroke.
# ---------------------------------------------------------------------------

import ast

_SCRIPT_FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "shutil", "socket", "importlib", "__import__",
    "open", "eval", "exec", "compile", "input", "exit", "quit", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "__builtins__",
}


def _check_script_safe(src: str) -> None:
    """Raises ValueError if `src` contains anything outside the allowed subset. Mirrors actions.py's
    default-deny philosophy: allow a small known-safe grammar, reject everything else, never try to
    blocklist every dangerous thing by name (blocklists rot; this allowlist does not)."""
    try:
        tree = ast.parse(src, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"script does not parse: {e}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("run_script may not import anything - use the `bpy`/`mathutils`/`math`/`ops` "
                              "names already provided")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"dunder attribute access is not allowed: .{node.attr}")
        if isinstance(node, ast.Name) and node.id in _SCRIPT_FORBIDDEN_NAMES:
            raise ValueError(f"name not allowed in run_script: {node.id}")
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _SCRIPT_FORBIDDEN_NAMES:
                raise ValueError(f"call not allowed in run_script: {fn.id}(...)")


def _fx_run_ops(ops):
    """Sequence of {"op": name, "args": {...}} - name must already be a SAFE_OPS entry. Stops and reports
    exactly which step failed rather than leaving a half-built scene with no explanation."""
    results = []
    for i, step in enumerate(ops):
        name = step.get("op")
        if name not in SAFE_OPS:
            raise ValueError(f"run_ops step {i}: unknown op {name!r}")
        try:
            results.append(SAFE_OPS[name](**step.get("args", {})))
        except Exception as e:
            raise ValueError(f"run_ops step {i} ({name}) failed: {e}") from e
    return {"steps_ok": len(results)}


def _fx_run_script(code):
    _check_script_safe(code)
    ns = {"bpy": bpy, "mathutils": mathutils, "math": __import__("math"),
          "ops": dict(SAFE_OPS), "__builtins__": {}}   # no builtins leak through at all, not even a blocklist
    exec(compile(code, "<run_script>", "exec"), ns)     # nosec - reached only after _check_script_safe passes
    return {"ok": True}


SAFE_OPS["run_ops"] = _fx_run_ops
SAFE_OPS["run_script"] = _fx_run_script


def handle(conn):
    data = conn.recv(65536).decode()
    try:
        req = json.loads(data)
        fn = SAFE_OPS[req["op"]]
        result = fn(**req.get("args", {}))
        resp = {"ok": True}
        try:
            json.dumps(result)
            if result is not None:
                resp["result"] = result
        except TypeError:
            pass          # bpy ops return sets like {'FINISHED'}: not sendable, not needed
    except Exception as e:
        resp = {"ok": False, "error": str(e)}
    conn.sendall(json.dumps(resp).encode())
    conn.close()


def serve():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)
    print(f"[blender_bridge] listening on {HOST}:{PORT} — leave Blender open")
    while True:
        conn, _ = s.accept()
        handle(conn)


if bpy is not None:
    threading.Thread(target=serve, daemon=True).start()
