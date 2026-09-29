# Quadmesh - from-scratch CAD + chat + voice agent (planner + verifier + Blender control + VLA)

Everything here is trained from random weights. Quadmesh is a **planner + verifier** system with a conversational front
end: a chat model understands text, voice, images and 3D files and calls tools; small language models write verified CAD
operations or call the mesh-geometry engine directly; a checker verifies them; Blender (which Quadmesh opens itself) builds
them; a printability gate checks the exported STL/STEP.

**Read first - what it can and cannot do**
- Talks: text, your voice (own ASR), a photo/drawing, or an attached STL/STEP file - in one conversation. Falls back to a
  fixed-rule assistant + a verified knowledge base until `quadmesh.train.chat_train` has been run.
- Builds: 10,000+ catalog part families, free-form CSG with explicit coordinates, curved/organic shapes (lathe/loft/sweep),
  gears, threads (bolts + nuts), 3D text, vases/pipes/springs, four-bar linkages, a hexacopter frame, a robot arm with a
  stand, and STEP/B-Rep (via CadQuery) as well as STL. "Printable" means it passed geometry/printability rules - it does
  NOT mean strong or certified (`quadmesh/strength.py` is hand-calculation with placeholder material numbers). Do the first
  real print supervised.
- Photo -> 3D is a GUESS: symmetric upright objects are spun into a solid of revolution, everything else is inflated from
  its outline. The hidden sides and depth are invented - check before printing.
- Controls the real keyboard and mouse. `--full-control` unlocks every key/hotkey and free mouse strokes (drag, sculpt,
  orbit) - not just boxes/cylinders/spheres - for driving Blender's whole UI. The screen-corner fail-safe always stays on.
  Quadmesh starts Blender itself (`quadmesh/agent/launch.py`); you no longer need to open it by hand.
- Speaks with its OWN trained voice (LJSpeech is a single FEMALE speaker, so training on it gives a from-scratch female
  voice) via a Griffin-Lim vocoder - no extra network to train, playable the moment TTS v2 finishes. Falls back to your OS
  voice until then.
- Hears with its own ASR, sized larger in this build specifically to make fewer mistakes; it is still a small from-scratch
  model and will make some errors - it repeats back what it heard before acting.
- Untested by me: everything needing torch, a GPU, Modal, a real Blender, or a real microphone/speakers.
  `python tests/run_offline_tests.py` covers the pure-Python parts (20 checks; the STEP check skips itself if `cadquery`
  is not installed).

## File structure

```
quadmesh_project/
├── docs/
│   └── DATASETS.md   - verified datasets and what each is for
├── quadmesh/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── actions.py   - action language + safety filter (+ full-control mode)
│   │   ├── app.py   - desktop app (buttons: Autopilot, Desk mode, Image -> solid)
│   │   ├── assistant.py   - fixed-rule fallback: move, 360 view, questions, voice
│   │   ├── autopilot.py   - plan -> build -> check -> repair loop; assemblies; --desk/--gui; opens Blender itself
│   │   ├── blender_bridge.py   - RUN INSIDE BLENDER: 160 ops + socket server
│   │   ├── blender_client.py   - socket client
│   │   ├── chat_agent.py   - the chat agent: text/voice/image/3D-file in, tool calls + own voice out
│   │   ├── desk.py   - real keyboard + mouse controller (+ 360 turntable + mouse strokes)
│   │   ├── desktop.py   - low-level pyautogui helpers
│   │   ├── episode.py   - unlimited-length verified step loop
│   │   ├── gui_recipes.py   - shapes via Shift+A/S/G/F2 + drag/grab/rotate/sculpt mouse recipes
│   │   ├── image3d.py   - drawing / silhouette -> printable solid
│   │   ├── launch.py   - finds and starts Blender itself - you never open it by hand
│   │   ├── loop.py   - step runner with screenshots
│   │   ├── observe.py   - scene report after a run
│   │   ├── planner.py   - planner used by the app (rules + trained model)
│   │   ├── planner_model.py   - loads any role/tier on CPU/GPU, KV-cached generation
│   │   ├── printability.py   - printability gate (Blender report + STL file)
│   │   ├── record_demo.py   - records real demonstrations
│   │   ├── repl.py   - text REPL
│   │   ├── speech.py   - microphone -> text (ASR)
│   │   ├── visible_ops.py   - Blender keyboard shortcuts for a few ops
│   │   └── vla_agent.py   - runs the VLA on your desktop
│   ├── data/
│   │   ├── __init__.py
│   │   ├── streaming.py   - streaming loaders + role routing
│   │   ├── tokenizer.py   - 32k BPE tokenizer
│   │   ├── vision.py   - screenshot patching for vision roles
│   │   └── vla_data.py   - REAL data for the VLA (GroundCUA + your recordings)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── audio.py   - ASR + TTS models (TTS has a stop head)
│   │   ├── chat.py   - QuadmeshChat: multimodal (text+image) conversational model
│   │   ├── lora.py   - LoRA adapters
│   │   ├── transformer.py   - QuadmeshModel (RMSNorm, RoPE, SwiGLU) + vision model + KV cache
│   │   └── vla.py   - QuadmeshVLA: screenshot + text -> action
│   ├── pipeline/
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   ├── engineering_gen.py   - 10,000+ part families + free-form CSG generator
│   │   │   └── planner_sft_gen.py   - original simple plan generator
│   │   ├── eval_sets/   (one eval set per role, .jsonl)
│   │   ├── __init__.py
│   │   ├── plan_checker.py   - plan simulator + checker (52 ops) = the verifier
│   │   ├── stage3_selfgen.py   - stage 3 self-generation (real grader)
│   │   ├── stage4_preference.py
│   │   ├── stage5_calibration.py
│   │   ├── stage6_redteam.py
│   │   ├── stage7_eval.py
│   │   └── stage8_shadow.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── chat_train.py   - chat model training (tool-use + knowledge + a little general text)
│   │   ├── io_train.py   - ASR training
│   │   ├── loop.py   - training loop (no tier branches)
│   │   ├── tts_train2.py   - TTS v2 training: batched, stop-token loss, guided attention
│   │   └── vla_train.py   - VLA training stage
│   ├── verification/
│   │   ├── __init__.py
│   │   └── stack.py   - verification skeleton (FEA etc. not implemented)
│   ├── __init__.py
│   ├── assemblies.py   - hexacopter + robot arm (with stand) from parameters
│   ├── chat_data.py   - chat model training data + tool-call parser
│   ├── chat_kb.py   - verified facts the assistant can answer from
│   ├── config.py   - tier hyperparameters (100M...70B)
│   ├── geometry3d.py   - curves/lofts/sweeps/threads/gears/text/inflation mesh generators (numpy)
│   ├── mechanisms.py   - gear pairs + four-bar linkage sizing and motion analysis
│   ├── mesh_tools.py   - validated front door for every shape generator + printability gate
│   ├── photo3d.py   - photo -> 3D guess (lathe or inflate from the silhouette)
│   ├── role_shapes.py   - per-role model sizes (NOT an equal split)
│   ├── serve.py   - FastAPI service deployed by `modal deploy`
│   ├── spec_gen_data.py   - Spec Generator training data + safe parser
│   ├── step_io.py   - STEP / B-Rep via CadQuery: plan -> STEP, STEP -> mesh
│   ├── strength.py   - materials + hand-calculation strength checks
│   ├── tts_style.py   - pitch-shift / EQ speaking style
│   └── tts_synth.py   - Griffin-Lim vocoder: mel -> wav (no extra training)
├── tests/
│   └── run_offline_tests.py   - 20 offline checks (no GPU/Blender/torch)
├── README.md   - this file
├── check_kv.py   - proves the KV cache gives identical output
├── check_vla.py   - smoke test of the VLA model
├── commands.txt   - copy-paste command list
├── eval_planner.py   - compare tiers with real numbers
├── modal_app.py   - Modal entry points: train, stages 3-8, VLA, chat, TTS v2, deploy (`web`)
├── requirements-local-agent.txt   - desktop-agent dependencies (mouse, keyboard, voice, audio playback)
└── requirements.txt   - cloud + training dependencies
```

## 1. Environment (Windows PowerShell)

```powershell
cd D:\project\quadmesh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-local-agent.txt
python tests\run_offline_tests.py
python check_kv.py
python check_vla.py
python -m quadmesh.role_shapes
```

## 2. Accounts and secrets

```powershell
huggingface-cli login
modal setup
modal secret create huggingface-secret HF_TOKEN=<YOUR-NEW-TOKEN>
```
Never paste a token into a file that gets zipped or shared.

## 3. Train on Modal (GPU in the cloud)

```powershell
modal run modal_app.py::build_tokenizer
modal run modal_app.py::plan --tier 500M
modal run modal_app.py::roles --tier 500M
modal run modal_app.py::train_all --tier 500M
```
Compare tiers with numbers: `python eval_planner.py --tier 500M --n 200` and `--n 300 --spec`.

Real data for the VLA:
```powershell
python -m quadmesh.data.vla_data --list-apps
python -m quadmesh.data.vla_data --inspect
modal volume put quadmesh-ckpts D:\demos /demos
modal run modal_app.py::train_vla --tier 500M --steps 20000
```

Chat model and improved TTS (add train_chat / train_tts2 Modal functions per the docstrings in
quadmesh/train/chat_train.py and quadmesh/train/tts_train2.py), then:
```powershell
modal run modal_app.py::train_chat --tier 500M --steps 30000
modal run modal_app.py::train_tts2 --tier 500M --steps 30000
```

## 4. Bring the checkpoints home

```powershell
modal volume get quadmesh-ckpts /Quadmesh-500M D:\quadmesh_ckpt\Quadmesh-500M
modal volume get quadmesh-ckpts /tokenizer D:\quadmesh_ckpt\tokenizer
$env:QUADMESH_CKPT_ROOT = "D:\quadmesh_ckpt"
$env:QUADMESH_TIER      = "500M"
$env:QUADMESH_DEVICE    = "auto"
```

## 5. Talk to Quadmesh (Blender opens itself)

```powershell
python -m quadmesh.agent.chat_agent --desk --voice --speak
python -m quadmesh.agent.chat_agent --desk --full-control --voice --speak
python -m quadmesh.agent.chat_agent --image bracket.png
python -m quadmesh.agent.chat_agent --file existing_part.stl
python -m quadmesh.agent.app
python -m quadmesh.agent.autopilot "Make a 60x40x5 mm plate with 4 corner holes of 4.2 mm diameter, 8 mm from the edges" --tier 500M
python -m quadmesh.agent.autopilot --assembly "make a hexadrone, 1.5 kg, industrial grade, in PETG" --tier 500M
python -m quadmesh.agent.printability part.stl --bed 220 220 250
```
Record real demonstrations for the VLA:
```powershell
python -m quadmesh.agent.record_demo --task "add a box 60x40x5 mm and name it body" --out D:\demos
python -m quadmesh.agent.vla_agent "add a cube" --dry-run
```
Every design writes an STL (and, on request, a STEP file) to `designs\` and stops with "do not print" if any check fails.

## 6. Deploy

```powershell
modal deploy modal_app.py
curl https://<workspace>--quadmesh-web.modal.run/health
modal app stop quadmesh
```
The deployed service is the risk/safety gate (`/generate`). Blender, mouse, keyboard and voice always run locally. It
serves the tier in `modal_app.py` (`QUADMESH_SERVE_TIER`, default 500M) on an L4 GPU with `min_containers=0` (no idle bill).

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `no trained planner/chat/tts at ...` | run section 3, then section 4; check `QUADMESH_CKPT_ROOT` and the tier folder name |
| `size mismatch` when loading | checkpoint was trained with old shapes - retrain that role |
| GroundCUA warning: `TRAINING ON SYNTHETIC DATA` | real data unreachable: check `HF_TOKEN`, run `--inspect` |
| "Blender could not be started" | run `python -m quadmesh.agent.launch --blender-path "C:\Path\To\blender.exe"` once |
| Desk mode types nothing / wrong characters | Blender must be focused; US layout; install `pyperclip` for long lines |
| No sound from Quadmesh's own voice | `pip install simpleaudio`; falls back to `pyttsx3` automatically |
| STEP tools raise `ImportError` | `pip install cadquery` (large; only needed for `quadmesh/step_io.py`) |
| "FAILED ... do not print" | read the last problem line; the design was NOT marked printable |
| `torch` install is huge | use the CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
#   q u a d m e s h  
 