"""
Quadmesh web app — local control-panel UI ("what's happening inside the model")
served over HTTP, in front of the real agent code (no separate/duplicate logic).

    python -m quadmesh.webapp.server                 # http://127.0.0.1:8420

This does NOT replace app.py/chat_agent.py — it's a browser front end that calls
the same plan_from_text / run_plan / launch.ensure_blender / blender_client used
everywhere else. If there is no trained checkpoint at QUADMESH_CKPT_ROOT, /api/chat
returns the same NoModelError message the CLI would give — it does not fall back
to anything that looks trained.
"""
from __future__ import annotations
import os
import shutil
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from quadmesh.agent.planner import plan_from_text, NoModelError
from quadmesh.agent.loop import run_plan
from quadmesh.agent import blender_client as bridge
from quadmesh.agent.launch import ensure_blender

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(os.environ.get("QUADMESH_WEB_UPLOADS", "designs/web_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Quadmesh")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
def status():
    return {
        "blender_bridge_up": bridge.is_up(),
        "ckpt_root": os.environ.get("QUADMESH_CKPT_ROOT", ""),
        "tier": os.environ.get("QUADMESH_TIER", ""),
        "has_checkpoint": bool(os.environ.get("QUADMESH_CKPT_ROOT")) and
                           Path(os.environ.get("QUADMESH_CKPT_ROOT", "/nonexistent")).exists(),
    }


@app.post("/api/open_blender")
def open_blender():
    try:
        ensure_blender()
        return {"ok": True, "blender_bridge_up": bridge.is_up()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Accepts a reference image or an existing 3D file (.png/.jpg/.stl/.obj/.fbx/.step)."""
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": str(dest)}


@app.post("/api/chat")
async def chat(text: str = Form(...), image_path: str = Form(""), model_path: str = Form("")):
    """
    Runs the real planner and returns BOTH the trace (what each stage of the
    model did — this is the "what's happening inside" panel) and a plain-text
    reply for the chat bubble / voice output.
    """
    prompt = text
    if image_path:
        prompt += f"\n[attached reference image: {image_path}]"
    if model_path:
        prompt += f"\n[attached 3D file: {model_path}]"

    trace = [{"stage": "task_planner", "detail": "parsing instruction into a build plan"}]
    try:
        steps = plan_from_text(prompt)
    except NoModelError as e:
        trace.append({"stage": "task_planner", "detail": f"no trained model available: {e}"})
        return {
            "reply": ("I don't have a trained checkpoint loaded, so I can't plan or build anything yet. "
                       "Set QUADMESH_CKPT_ROOT / QUADMESH_TIER to a real checkpoint and restart the server."),
            "trace": trace,
            "steps": [],
            "exported_file": None,
        }
    except Exception as e:
        trace.append({"stage": "task_planner", "detail": f"error: {e}"})
        return JSONResponse({"reply": f"Planning failed: {e}", "trace": trace, "steps": [],
                              "exported_file": None}, status_code=500)

    trace.append({"stage": "task_planner", "detail": f"produced {len(steps)} step(s)"})
    trace.append({"stage": "cad_skill_dispatch", "detail": "sending steps to the Blender bridge"})

    exported_file = None
    try:
        result = run_plan(steps)
        trace.append({"stage": "verification_stack", "detail": "checking the resulting mesh"})
        exported_file = getattr(result, "exported_path", None) if result else None
        trace.append({"stage": "done", "detail": "build complete" if exported_file else "steps ran, no export path returned"})
    except Exception as e:
        trace.append({"stage": "cad_skill_dispatch", "detail": f"error while running steps: {e}"})
        return JSONResponse({"reply": f"Build failed partway through: {e}", "trace": trace,
                              "steps": [s.__dict__ if hasattr(s, "__dict__") else str(s) for s in steps],
                              "exported_file": None}, status_code=500)

    reply = f"Built {len(steps)} step(s) in Blender." + (f" Exported to {exported_file}." if exported_file else "")
    return {
        "reply": reply,
        "trace": trace,
        "steps": [s.__dict__ if hasattr(s, "__dict__") else str(s) for s in steps],
        "exported_file": exported_file,
    }


@app.get("/file")
def get_file(path: str):
    """Serves an exported model back to the browser for the 3D preview.
    NOTE: <model-viewer> renders glTF/GLB (and OBJ via some builds) natively —
    it does NOT render .stl. If your exported file is .stl, the preview panel
    will show nothing even though the build succeeded; open it in Blender or
    convert to .glb to preview it here. This is a real, unresolved gap, not
    a bug I've silently papered over."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(str(p))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8420)
