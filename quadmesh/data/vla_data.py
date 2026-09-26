"""
REAL training data for the Vision-Language-Action model (no synthetic screenshots).

  1. GroundCUA  (ServiceNow/GroundCUA): 56K human-annotated desktop screenshots from 87 real applications,
     3.56M labelled UI elements. Each screenshot has data/<App>/<hash>.json (a list of
     {image_path, bbox [x1,y1,x2,y2] in pixels, text, category, id}) and images/<App>/<hash>.png.
     We read the files directly instead of load_dataset(), because the JSON files do not all share one schema
     (that is what made the old loader fail). Teaches single-step click/grounding, not multi-step sessions.
  2. VideoCAD (NeurIPS 2025, github.com/ghadinehme/VideoCAD): 41K+ long-horizon CAD UI sessions, timestamped
     mouse/keyboard actions + the CAD image they were building toward. Real Onshape pixels, bot-replayed
     DeepCAD actions - so it teaches the *shape* of a long CAD session (many small actions building toward
     one target, look-check-correct rhythm), not literal Blender menus. This is the dataset the `vla` role
     (0.20 of the parameter budget, see role_shapes.py) needs most, because it's the only source here of
     real long-horizon action sequences rather than single clicks.
     CAVEAT (unresolved as of this file): VideoCAD's primary distribution is Harvard Dataverse, not the HF
     Hub, so it does not get the same load_dataset(streaming=True) treatment as your other sources. This
     loader tries a small list of *unverified* candidate HF mirror repo ids first (some datasets get mirrored
     by third parties); if none exist, it falls back to reading the Dataverse file listing directly and
     processing one session at a time (download session -> yield examples -> discard -> next session), so
     nothing is bulk-downloaded even without an HF mirror. Also unresolved: I have not inspected a real
     VideoCAD annotation file, so `_videocad_examples_from_session` is written schema-tolerant (same pattern
     as `load_annotations` below for GroundCUA) rather than assuming exact key names - check
     `--inspect-videocad` output against the actual repo before trusting field names.
  3. YOUR OWN demonstrations recorded with record_demo.py (real Blender pixels + real actions).

Example = {"image": PIL.Image, "instruction": str, "action": str}, action in the language of actions.py
(move/click/double_click/right_click x y | drag/middle_drag x1 y1 x2 y2 | scroll n | key k | type text | wait n).

    python -m quadmesh.data.vla_data --inspect            # download ONE GroundCUA annotation file, show what was parsed
    python -m quadmesh.data.vla_data --list-apps          # which of the 87 GroundCUA apps exist (is Blender one of them?)
    python -m quadmesh.data.vla_data --inspect-videocad    # try to reach VideoCAD, show ONE converted session
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
import re
from typing import Iterator, Optional

GROUNDCUA = "ServiceNow/GroundCUA"
COORD_MAX = 999

# UNVERIFIED — I searched for a Hugging Face mirror and found none; VideoCAD's own README (github.com/
# ghadinehme/VideoCAD, checked directly) only links to Harvard Dataverse and a single Google Drive link for
# cad_imgs.zip. These two candidates are speculative fallback attempts, not confirmed mirrors — they will
# simply fail and fall through to the Dataverse path below, which IS confirmed (see VIDEOCAD_DATAVERSE_DOI).
VIDEOCAD_HF_CANDIDATES = [
    "ghadinehme/VideoCAD",
    "MIT-DeCoDE/VideoCAD",
]

# Harvard Dataverse persistent ID for VideoCAD — CONFIRMED, read directly from the dataset's own GitHub
# README (github.com/ghadinehme/VideoCAD; authors Man, Nehme, Alam, Ahmed, MIT, NeurIPS 2025 Datasets Track).
VIDEOCAD_DATAVERSE_DOI = "doi:10.7910/DVN/WX8PCK"
VIDEOCAD_DATAVERSE_API = "https://dataverse.harvard.edu/api"

# Real file layout on the Dataverse listing (confirmed from the README, not guessed):
#   0000.zip .. 0249.zip   - UI screen recordings, 60 FPS video, one zip per session batch
#   cad_imgs.zip           - rendered isometric TARGET image per session (goal-conditioning, not per-frame)
#   action_json.zip        - low-level UI actions (mouse/keyboard), JSON format - this is what
#                             _videocad_action_to_grammar's field names should be checked against
#   mouse_json.zip         - high-level operations (extrude, sketch, parameters), JSON format
# This is a real correction from an earlier version of this file, which wrongly assumed one flat JSON per
# session with an embedded list of frame URLs - the actual pipeline needs frames extracted from the 60 FPS
# video AT the action timestamps (their own generate_dataset.py does exactly this), which needs a video
# library (opencv-python / cv2), not just `requests` + `json`. VIDEOCAD_FRAME_LIB names which one to use.
VIDEOCAD_FRAME_LIB = "cv2"   # pip install opencv-python — required for the Dataverse fallback path only


# ---------------------------------------------------------------------------
# GroundCUA
# ---------------------------------------------------------------------------
def load_annotations(path: str) -> list:
    """Schema-tolerant: a list of entries, or a dict that contains one. Returns entries with a 4-number bbox."""
    with open(path, "r", encoding="utf-8") as fh:
        j = json.load(fh)
    if isinstance(j, dict):
        for k in ("annotations", "elements", "items", "data"):
            if isinstance(j.get(k), list):
                j = j[k]; break
        else:
            j = [v for v in j.values() if isinstance(v, dict)]
    out = []
    for e in j if isinstance(j, list) else []:
        bb = e.get("bbox") if isinstance(e, dict) else None
        if isinstance(bb, (list, tuple)) and len(bb) == 4 and all(isinstance(v, (int, float)) for v in bb):
            out.append(e)
    return out


def _good_text(t) -> bool:
    return isinstance(t, str) and 1 <= len(t.strip()) <= 50 and re.search(r"[A-Za-z0-9]", t) is not None \
        and all(32 <= ord(c) < 127 for c in t)


def examples_from_annotations(img, anns: list, rng: random.Random, per_image: int = 4) -> list:
    """Click-grounding examples for elements whose text is UNIQUE on the screenshot (so the target is unambiguous)."""
    W, H = img.size
    counts = {}
    for a in anns:
        t = (a.get("text") or "").strip().lower()
        counts[t] = counts.get(t, 0) + 1
    cands = [a for a in anns if _good_text(a.get("text")) and counts[a["text"].strip().lower()] == 1]
    rng.shuffle(cands)
    out = []
    for a in cands[:per_image]:
        x1, y1, x2, y2 = a["bbox"]
        if x2 <= x1 or y2 <= y1 or x2 > W * 1.02 or y2 > H * 1.02:
            continue
        cx = max(0, min(COORD_MAX, int(round((x1 + x2) / 2 / W * COORD_MAX))))
        cy = max(0, min(COORD_MAX, int(round((y1 + y2) / 2 / H * COORD_MAX))))
        text = a["text"].strip()
        cat = a.get("category")
        instr = rng.choice([f'click "{text}"', f"click the {text}", f'click on "{text}"'])
        if cat and rng.random() < 0.3:
            instr = f'click the {str(cat).lower()} labeled "{text}"'
        out.append({"image": img, "instruction": instr, "action": f"click {cx} {cy}"})
    return out


def box_examples(img, anns: list, rng: random.Random, per_image: int = 4) -> list:
    """Vision-role examples: the screenshot + the normalised box [x1,y1,x2,y2] (0..1) of one uniquely-labelled element."""
    W, H = img.size
    counts = {}
    for a in anns:
        t = (a.get("text") or "").strip().lower()
        counts[t] = counts.get(t, 0) + 1
    cands = [a for a in anns if _good_text(a.get("text")) and counts[a["text"].strip().lower()] == 1]
    rng.shuffle(cands)
    out = []
    for a in cands[:per_image]:
        x1, y1, x2, y2 = a["bbox"]
        if x2 <= x1 or y2 <= y1 or x2 > W * 1.02 or y2 > H * 1.02:
            continue
        out.append({"image": img, "bbox": [max(0.0, x1 / W), max(0.0, y1 / H), min(1.0, x2 / W), min(1.0, y2 / H)]})
    return out


def list_apps(token: Optional[str] = None) -> list:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(GROUNDCUA, repo_type="dataset", token=token)
    return sorted({f.split("/")[1] for f in files if f.startswith("data/") and f.count("/") >= 2})


def groundcua_stream(apps: Optional[list] = None, seed: int = 0, per_image: int = 4,
                     token: Optional[str] = None, boxes: bool = False) -> Iterator[dict]:
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image
    token = token or os.environ.get("HF_TOKEN")
    files = HfApi().list_repo_files(GROUNDCUA, repo_type="dataset", token=token)
    jsons = [f for f in files if f.startswith("data/") and f.endswith(".json")]
    if apps:
        want = {a.lower() for a in apps}
        jsons = [f for f in jsons if f.split("/")[1].lower() in want]
    rng = random.Random(seed)
    rng.shuffle(jsons)
    for f in jsons:
        try:
            anns = load_annotations(hf_hub_download(GROUNDCUA, f, repo_type="dataset", token=token))
            if not anns:
                continue
            imgf = "images/" + f[len("data/"):-len(".json")] + ".png"
            img = Image.open(hf_hub_download(GROUNDCUA, imgf, repo_type="dataset", token=token)).convert("RGB")
        except Exception as e:                       # one broken file must never stop training
            print(f"[vla_data] skipped {f}: {type(e).__name__}", flush=True)
            continue
        yield from (box_examples if boxes else examples_from_annotations)(img, anns, rng, per_image)


def history_text(actions: list, window: int = 12) -> str:
    """Ordered action history for the prompt: the last `window` actions verbatim, older ones counted."""
    acts = [a for a in actions if not a.startswith("wait")]
    if not acts:
        return ""
    older = len(acts) - window
    head = f"(+{older} earlier actions) " if older > 0 else ""
    return "\nhistory: " + head + "; ".join(acts[-window:])


# ---------------------------------------------------------------------------
# Your own recordings
# ---------------------------------------------------------------------------
def demo_stream(demo_dir: str, seed: int = 0, history: int = 12, loop: bool = True,
                only_success: bool = False) -> Iterator[dict]:
    from PIL import Image
    sessions = sorted(glob.glob(os.path.join(demo_dir, "*", "demo.jsonl")))
    sessions = [s for s in sessions if not os.path.exists(os.path.join(os.path.dirname(s), "FAILED"))]   # never learn from failures
    if only_success:                                   # only runs the episode runner verified complete
        sessions = [s for s in sessions if os.path.exists(os.path.join(os.path.dirname(s), "SUCCESS"))]
    rng = random.Random(seed)
    while True:
        rows = []
        for s in sessions:
            base = os.path.dirname(s)
            with open(s, encoding="utf-8") as fh:
                recs = [json.loads(l) for l in fh if l.strip()]
            for i, r in enumerate(recs):
                instr = r["task"] + history_text([x["action"] for x in recs[:i]], history)
                rows.append((os.path.join(base, r["screenshot"]), instr, r["action"]))
        if not rows:
            return
        rng.shuffle(rows)
        for path, instr, act in rows:
            try:
                yield {"image": Image.open(path).convert("RGB"), "instruction": instr, "action": act}
            except Exception:
                continue
        if not loop:
            return


def groundcua_box_stream(seed: int = 0, apps: Optional[list] = None) -> Iterator[dict]:
    return groundcua_stream(apps=apps, seed=seed, boxes=True)


# ---------------------------------------------------------------------------
# VideoCAD — long-horizon CAD sessions (real Onshape pixels, bot-replayed DeepCAD actions)
# ---------------------------------------------------------------------------

def _pick(rec: dict, keys) -> Optional[object]:
    """Schema-tolerant field lookup (VideoCAD's exact key names are unverified - see module docstring)."""
    for k in keys:
        v = rec.get(k)
        if v is not None and v != "":
            return v
    return None


def _norm_xy(x: float, y: float, w: int, h: int) -> tuple:
    """Pixel coords -> 0..COORD_MAX grid, same normalisation examples_from_annotations() uses."""
    cx = max(0, min(COORD_MAX, int(round(x / w * COORD_MAX))))
    cy = max(0, min(COORD_MAX, int(round(y / h * COORD_MAX))))
    return cx, cy


def _videocad_action_to_grammar(ev: dict, w: int, h: int) -> Optional[str]:
    """One VideoCAD action-log event -> one string in actions.py's grammar. Schema-tolerant: tries several
    plausible key names since the exact annotation schema hasn't been inspected yet (see module docstring).
    Extend the key lists below once a real session file has been checked against actual field names."""
    kind = (_pick(ev, ("type", "action", "event", "op")) or "").lower()
    x = _pick(ev, ("x", "px", "pos_x"))
    y = _pick(ev, ("y", "py", "pos_y"))
    x2 = _pick(ev, ("x2", "end_x", "to_x"))
    y2 = _pick(ev, ("y2", "end_y", "to_y"))

    if kind in ("click", "mousedown_up", "left_click") and x is not None and y is not None:
        cx, cy = _norm_xy(float(x), float(y), w, h)
        return f"click {cx} {cy}"
    if kind in ("double_click", "dblclick") and x is not None and y is not None:
        cx, cy = _norm_xy(float(x), float(y), w, h)
        return f"double_click {cx} {cy}"
    if kind in ("right_click", "contextmenu") and x is not None and y is not None:
        cx, cy = _norm_xy(float(x), float(y), w, h)
        return f"right_click {cx} {cy}"
    if kind in ("drag", "mousedrag") and None not in (x, y, x2, y2):
        cx1, cy1 = _norm_xy(float(x), float(y), w, h)
        cx2, cy2 = _norm_xy(float(x2), float(y2), w, h)
        return f"drag {cx1} {cy1} {cx2} {cy2}"
    if kind in ("scroll", "wheel"):
        n = _pick(ev, ("delta", "n", "amount"))
        if n is not None:
            return f"scroll {max(-30, min(30, int(n)))}"
    if kind in ("key", "keydown", "keypress"):
        k = _pick(ev, ("key", "keycode", "k"))
        if k:
            return f"key {k}"
    if kind in ("type", "text_input"):
        s = _pick(ev, ("text", "value", "s"))
        if s:
            return f'type "{s}"'
    return None   # unrecognised event kind -> dropped, never guessed


def _videocad_examples_from_session(frames_getter, action_log: list, task: str, history: int = 12) -> Iterator[dict]:
    """frames_getter(idx) -> PIL.Image for the frame at that action's timestamp. Mirrors demo_stream()'s
    history_text() convention so this trains on the same instruction format as your own recordings."""
    acts, kept = [], []
    for i, ev in enumerate(action_log):
        img = frames_getter(i)
        if img is None:
            continue
        w, h = img.size
        a = _videocad_action_to_grammar(ev, w, h)
        if a is None:
            continue
        instr = task + history_text(acts, history)
        kept.append({"image": img, "instruction": instr, "action": a})
        acts.append(a)
    yield from kept


def _videocad_hf_stream(seed: int = 0) -> Optional[Iterator[dict]]:
    """Try each candidate HF mirror; return the first that actually loads, else None. Never raises — a
    missing/renamed mirror should fall through to the Dataverse path, not crash training."""
    from datasets import load_dataset
    for repo in VIDEOCAD_HF_CANDIDATES:
        try:
            ds = load_dataset(repo, split="train", streaming=True)
            first = next(iter(ds))   # prove it actually yields before committing to this source
        except Exception as e:
            print(f"[vla_data] VideoCAD HF mirror {repo!r} unavailable ({type(e).__name__}: {e})", flush=True)
            continue

        def _gen(ds=ds, seed=seed):
            rng = random.Random(seed)
            for rec in ds:
                task = _pick(rec, ("task", "prompt", "instruction")) or "reproduce the target CAD model"
                log = _pick(rec, ("actions", "action_log", "events")) or []
                frames = _pick(rec, ("frames", "images"))
                if not log or not frames:
                    continue
                yield from _videocad_examples_from_session(lambda i, frames=frames: frames[i] if i < len(frames) else None,
                                                            log, task)
        print(f"[vla_data] VideoCAD: using HF mirror {repo!r}", flush=True)
        return _gen()
    return None


def _videocad_dataverse_stream(seed: int = 0) -> Optional[Iterator[dict]]:
    """Real layout (see the constants above): action_json.zip + mouse_json.zip hold the action logs;
    0000.zip..0249.zip hold 60 FPS session videos; frames must be extracted AT the action timestamps (this
    needs cv2 - VIDEOCAD_FRAME_LIB - which is not installed by default; install opencv-python to use this
    path). Downloads one video zip at a time, extracts only the frames each session's actions actually need,
    then discards the video - never keeps a whole zip decoded in memory or on disk longer than one session."""
    try:
        import requests
        r = requests.get(f"{VIDEOCAD_DATAVERSE_API}/datasets/:persistentId",
                          params={"persistentId": VIDEOCAD_DATAVERSE_DOI}, timeout=30)
        r.raise_for_status()
        files = r.json()["data"]["latestVersion"]["files"]
    except Exception as e:
        print(f"[vla_data] VideoCAD Dataverse listing failed ({type(e).__name__}: {e}). Skipping VideoCAD.", flush=True)
        return None

    def _find(label_prefix):
        return [f for f in files if str(f.get("label", "")).startswith(label_prefix)]

    action_zips = _find("action_json")
    video_zips = sorted(_find(""), key=lambda f: f.get("label", ""))   # 0000.zip.. sort naturally by name
    video_zips = [f for f in video_zips if re.match(r"^\d{4}\.zip$", f.get("label", ""))]
    if not action_zips or not video_zips:
        print("[vla_data] VideoCAD: expected files (action_json.zip, NNNN.zip video batches) not found in "
              "the Dataverse listing - the dataset layout may have changed since this was written. Skipping.", flush=True)
        return None

    def _gen(action_zips=action_zips, video_zips=video_zips, seed=seed):
        import io, zipfile
        import requests as _rq
        try:
            import cv2
        except ImportError:
            print("[vla_data] VideoCAD: opencv-python not installed (pip install opencv-python) - "
                  "frame extraction needs it. Skipping VideoCAD for this run.", flush=True)
            return
        rng = random.Random(seed)

        # 1. actions: stream the action_json.zip member-by-member, never extract the whole archive to disk
        act_resp = _rq.get(f"{VIDEOCAD_DATAVERSE_API}/access/datafile/{action_zips[0]['dataFile']['id']}",
                            timeout=120)
        act_resp.raise_for_status()
        act_zf = zipfile.ZipFile(io.BytesIO(act_resp.content))
        session_ids = [n for n in act_zf.namelist() if n.endswith(".json")]
        rng.shuffle(session_ids)

        # 2. video: which batch zip (0000.zip etc.) a session lives in isn't documented in the README beyond
        #    "60 FPS session videos" - the safe assumption without seeing real filenames is one batch per
        #    ~sessions/250, tried in order; if a session's video isn't in the batch we guessed, it's skipped
        #    rather than downloading every batch zip to find it (that would defeat the streaming principle).
        for i, name in enumerate(session_ids):
            try:
                log = json.loads(act_zf.read(name))
            except Exception:
                continue
            task = _pick(log, ("task", "prompt", "instruction")) or "reproduce the target CAD model"
            events = _pick(log, ("actions", "action_log", "events")) or log if isinstance(log, list) else []
            timestamps = [_pick(ev, ("t", "time", "timestamp")) for ev in events] if isinstance(events, list) else []
            if not events or not video_zips:
                continue
            batch = video_zips[i % len(video_zips)]
            try:
                vid_resp = _rq.get(f"{VIDEOCAD_DATAVERSE_API}/access/datafile/{batch['dataFile']['id']}", timeout=180)
                vid_resp.raise_for_status()
                vid_zf = zipfile.ZipFile(io.BytesIO(vid_resp.content))
                mp4_name = next((n for n in vid_zf.namelist() if name.split(".")[0] in n and n.endswith(".mp4")), None)
                if mp4_name is None:
                    continue
                video_bytes = vid_zf.read(mp4_name)
            except Exception as e:
                print(f"[vla_data] skipped VideoCAD session {name}: {type(e).__name__}", flush=True)
                continue

            import tempfile, os as _os
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
                tf.write(video_bytes); tf.flush()
                cap = cv2.VideoCapture(tf.name)
                fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

                def _frame(idx, cap=cap, fps=fps, timestamps=timestamps):
                    t = timestamps[idx] if idx < len(timestamps) and timestamps[idx] is not None else None
                    if t is None:
                        return None
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(t) * fps))
                    ok, frame = cap.read()
                    if not ok:
                        return None
                    from PIL import Image
                    import numpy as np
                    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                yield from _videocad_examples_from_session(_frame, events, task)
                cap.release()
    return _gen()


def videocad_stream(seed: int = 0) -> Iterator[dict]:
    """HF mirror first (matches the streaming principle everything else in this codebase uses); Dataverse
    file-by-file fallback second; yields nothing (silently) if neither source is reachable, so a missing
    VideoCAD source degrades vla_stream() to its other two sources rather than crashing training."""
    gen = _videocad_hf_stream(seed=seed) or _videocad_dataverse_stream(seed=seed)
    if gen is None:
        return
    yield from gen


def vla_stream(demo_dir: Optional[str] = None, p_demo: float = 0.35, p_videocad: float = 0.35,
               seed: int = 0, apps: Optional[list] = None):
    """Three-way mix for the vla role (0.20 of the param budget - see role_shapes.py):
      - GroundCUA (p ~0.30 by subtraction): single-step click grounding, widest app coverage
      - VideoCAD  (p_videocad, default 0.35): long-horizon CAD session *patterns* - the largest single
        source of multi-step behaviour, even though it's Onshape pixels rather than Blender
      - your recordings (p_demo, default 0.35): the only source of real Blender behaviour; weighted equal
        to VideoCAD by default because ground truth here matters more than volume - raise p_demo as you
        record more, since demo_stream() is finite (it stops once your sessions are exhausted) while
        GroundCUA and VideoCAD are not.
    """
    g = groundcua_stream(apps=apps, seed=seed)
    v = videocad_stream(seed=seed)
    d = demo_stream(demo_dir, seed) if demo_dir and glob.glob(os.path.join(demo_dir, "*", "demo.jsonl")) else None
    rng = random.Random(seed + 1)
    v_exhausted = False
    while True:
        roll = rng.random()
        if d is not None and roll < p_demo:
            src, which = d, "d"
        elif (not v_exhausted) and roll < p_demo + p_videocad:
            src, which = v, "v"
        else:
            src, which = g, "g"
        try:
            ex = next(src)
            ex["source"] = {"g": "groundcua", "v": "videocad", "d": "demo"}[which]
            yield ex
        except StopIteration:
            if which == "g":
                g = groundcua_stream(apps=apps, seed=seed + rng.randint(1, 10 ** 6))
            elif which == "v":
                v_exhausted = True   # VideoCAD unreachable or finite mirror drained; fall back to g+d only
            else:
                return   # demos are the ground-truth signal; once they're gone, stop rather than dilute


# ---------------------------------------------------------------------------
# Screenshot -> patch tokens (numpy; the model turns them into embeddings)
# ---------------------------------------------------------------------------
IMG_SIZE, PATCH = 448, 16
N_PATCHES = (IMG_SIZE // PATCH) ** 2          # 784
PATCH_DIM = 3 * PATCH * PATCH                 # 768


def to_patches(img):
    import numpy as np
    from PIL import Image
    x = np.asarray(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    g = IMG_SIZE // PATCH
    return x.reshape(g, PATCH, g, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(g * g, PATCH_DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--list-apps", action="store_true")
    ap.add_argument("--inspect-videocad", action="store_true")
    a = ap.parse_args()
    if a.list_apps:
        apps = list_apps(); print(len(apps), "apps:", ", ".join(apps))
    if a.inspect:
        n = 0
        for ex in groundcua_stream(per_image=3):
            print(ex["image"].size, "|", ex["instruction"], "->", ex["action"]); n += 1
            if n >= 6:
                break
        print("OK - real data loads" if n else "nothing loaded: check HF_TOKEN / network")
    if a.inspect_videocad:
        n = 0
        for ex in videocad_stream():
            print(ex["image"].size, "|", ex["instruction"][:80], "->", ex["action"]); n += 1
            if n >= 6:
                break
        if n == 0:
            print("nothing loaded: no HF mirror resolved and VIDEOCAD_DATAVERSE_DOI is empty or unreachable "
                  "- fill in VIDEOCAD_DATAVERSE_DOI at the top of this file, or update VIDEOCAD_HF_CANDIDATES "
                  "if you've found a real mirror.")
        else:
            print("OK - VideoCAD data loads (check the action strings above against actions.py's grammar "
                  "and adjust _videocad_action_to_grammar's key names if anything looks wrong)")


if __name__ == "__main__":
    main()
