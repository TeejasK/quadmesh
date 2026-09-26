# Datasets for Quadmesh training - only ones I verified exist (from their own papers/pages)

| Dataset | What it really is | Use in Quadmesh | Caveat |
|---|---|---|---|
| **GroundCUA** (ServiceNow) | 56K human-annotated desktop screenshots, 87 apps, 3.56M labelled UI elements (bbox + text + category) | VLA: "click <element>" grounding (`quadmesh/data/vla_data.py`) | single-step grounding only, no multi-step tasks. Run `--list-apps` to see whether Blender is among the 87 |
| **VideoCUA** (ServiceNow, same suite) | real desktop video demonstrations with action logs, linked to GroundCUA screenshots | VLA: multi-step desktop behaviour | I have NOT inspected its format - read its dataset card first; no loader written yet |
| **VideoCAD** (NeurIPS 2025; Harvard Dataverse; code: github.com/ghadinehme/VideoCAD) | 41K+ annotated videos of CAD UI operations in Onshape, timestamped mouse/keyboard actions + target CAD image. Built from human-made DeepCAD designs replayed by a rule-based bot | VLA: long-horizon CAD click/type patterns | Real Onshape pixels, but bot-generated actions, and Onshape is not Blender - it teaches the *pattern* of long CAD sessions, not Blender's menus. Needs a converter (mouse .log + video frames -> our action language) |
| **Text2CAD** (SadilKhan/Text2CAD) | text prompts -> CAD construction sequences (from DeepCAD) | already in the repo (`TEXT2CAD`); Spec/Planner language variety | sequences are sketch+extrude, a different op vocabulary than ours |
| **DeepCAD / Fusion 360 Gallery** (Onshape / Autodesk) | human-authored parametric CAD design sequences; Fusion 360 Gallery also has assemblies | reference for sequence structure and for the assembly layer; source of VideoCAD and neuralCAD-Edit | not downloaded or converted here |
| **neuralCAD-Edit** (Autodesk, 2026) | 192 multimodal edit requests + 384 edits by 10 expert designers in Fusion; inputs from Fusion 360 Gallery | **evaluation only** - its own paper describes the trend to evaluation-only sets | too small to train on; use it to test |
| **Your own recordings** | `record_demo.py` + `episode.py` (verified runs only) | VLA: real Blender behaviour, the data that matters most | you must record them |

Not used, and why:
- ABC / other geometry-only collections: geometry with no text or actions - nothing to learn from for planning or clicking.
- Web/mobile GUI sets (Mind2Web, AITW ...): wrong software; they teach browsers and phones.
- Synthetic screenshots: replaced by the above, as you asked. Synthetic data is still used where it is *exact* (part plans, spec extraction), because there the checker gives perfect labels.
