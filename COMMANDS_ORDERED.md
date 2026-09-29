# ==============================================================================
# QUADMESH 3B - STEP-BY-STEP TRAINING GUIDE (PowerShell - copy-paste blocks)
# ==============================================================================
#
# Every command below shows live output in the terminal AND saves a log file.
# Just paste each block in order. All logs land in logs\*.txt
#
# ------------------------------------------------------------------------------

# +------------------------------------------------------------+
# | STEP 0 - LOAD THE LOGGING HELPER                           |
# | (run this ONCE per terminal session)                       |
# +------------------------------------------------------------+
$TIER = "3B"
New-Item -ItemType Directory -Force -Path logs | Out-Null
. .\scripts\run_and_log.ps1          # <-- loads the Run-Logged function

# +------------------------------------------------------------+
# | STEP 1 - VIRTUAL ENVIRONMENT                               |
# +------------------------------------------------------------+
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# +------------------------------------------------------------+
# | STEP 2 - INSTALL DEPENDENCIES  (~5 min)                    |
# +------------------------------------------------------------+
Run-Logged "pip install -r requirements.txt"               "logs\01_setup_pip.txt"
Run-Logged "pip install -r requirements-local-agent.txt"   "logs\02_setup_pip_local_agent.txt"

# +------------------------------------------------------------+
# | STEP 3 - OFFLINE TESTS  (~1 min)                           |
# +------------------------------------------------------------+
Run-Logged "python tests\run_offline_tests.py"             "logs\03_offline_tests.txt"
# All 29 checks should PASS

# +------------------------------------------------------------+
# | STEP 4 - CLOUD AUTH  (Modal + Hugging Face)                |
# +------------------------------------------------------------+
Run-Logged "huggingface-cli login"                         "logs\04_hf_login.txt"
Run-Logged "modal setup"                                   "logs\05_modal_setup.txt"

# ---> Accept the Text2CAD license BEFORE continuing:
#      https://huggingface.co/datasets/SadilKhan/Text2CAD

# +------------------------------------------------------------+
# | STEP 5 - CREATE HF SECRET ON MODAL                         |
# +------------------------------------------------------------+
# Replace <HF_TOKEN> with your actual token:
modal secret create huggingface-secret HF_TOKEN=<HF_TOKEN>

# +------------------------------------------------------------+
# | STEP 6 - BUILD TOKENIZER  (~30-60 min)                     |
# +------------------------------------------------------------+
Run-Logged "modal run modal_app.py::build_tokenizer"       "logs\06_build_tokenizer.txt"
# Success: "[tokenizer] Training complete! Vocab size: 32,000"
#
# Check the tokenized file in the Modal cloud volume:
#   modal volume ls quadmesh-ckpts /tokenizer
# Download it locally anytime:
#   modal volume get quadmesh-ckpts /tokenizer D:\quadmesh_ckpt\tokenizer

# +------------------------------------------------------------+
# | STEP 7 - SAFETY REVIEW QUEUE  (optional - can skip to 8)   |
# +------------------------------------------------------------+
# Note: This is an optional human-in-the-loop review for safety/abuse taxonomy.
# You can skip directly to STEP 8 if you want to begin model pretraining immediately.
Run-Logged "modal run modal_app.py::shared_review_queue"   "logs\07_review_queue_build.txt"
Run-Logged "modal volume get quadmesh-ckpts /data/review_queue.jsonl review_queue.jsonl" "logs\08_review_queue_get.txt"

# Interactive review - press 'a' to approve, 'r' to reject:
Run-Logged "python -m quadmesh.pipeline.review_queue review_queue.jsonl" "logs\09_review_queue_interactive.txt"

# Push approved rows back:
Run-Logged "modal volume put quadmesh-ckpts review_queue.jsonl /data/review_queue.jsonl --force" "logs\10_review_queue_put.txt"

# +------------------------------------------------------------+
# | STEP 8 - PRE-TRAINING  (hours -> ~24h)                     |
# +------------------------------------------------------------+
# Option A: Chinchilla budget (recommended first run)
Run-Logged "modal run modal_app.py::shared_pretrain --tier $TIER --budget chinchilla --mb 8 --accum 32 --grad-ckpt 1 --workers 3 --attach" "logs\11_pretrain_chinchilla.txt"

# Option B: Full 4x budget (run directly for full 28.8B tokens)
# Run-Logged "modal run modal_app.py::shared_pretrain --tier $TIER --budget 4x --mb 8 --accum 32 --grad-ckpt 1 --workers 3 --attach" "logs\11_pretrain_4x.txt"

# Option C: Detached (survives terminal close - no live output)
# modal run --detach modal_app.py::shared_pretrain --tier $TIER --budget chinchilla --mb 8 --accum 32 --grad-ckpt 1 --workers 3

# +------------------------------------------------------------+
# | STEP 9 - INSTRUCTION TUNING / SFT (~1-4h)                  |
# +------------------------------------------------------------+
Run-Logged "modal run modal_app.py::shared_sft --tier $TIER --tokens 2000000000 --mb 8 --accum 32 --grad-ckpt 1 --workers 3 --attach" "logs\12_sft.txt"

# +------------------------------------------------------------+
# | STEP 10 - VLA TRAINING  (Blender CUA)                      |
# +------------------------------------------------------------+
Run-Logged "modal run modal_app.py::train_vla --tier $TIER --steps 20000" "logs\13_vla.txt"

# +------------------------------------------------------------+
# | STEP 11 - SPEECH: ASR + TTS                                |
# +------------------------------------------------------------+
Run-Logged "modal run modal_app.py::train_io --tier $TIER --which asr" "logs\14_asr.txt"
Run-Logged "modal run modal_app.py::train_io --tier $TIER --which tts" "logs\15_tts.txt"

# +------------------------------------------------------------+
# | STEP 12 - EVALUATION  (~5-15 min)                          |
# +------------------------------------------------------------+
Run-Logged "modal run modal_app.py::shared_eval --tier $TIER --n 100" "logs\16_eval.txt"
# Target: checker pass >= 90%

# +------------------------------------------------------------+
# | STEP 13 - DOWNLOAD CHECKPOINTS LOCALLY                     |
# +------------------------------------------------------------+
New-Item -ItemType Directory -Force -Path D:\quadmesh_ckpt | Out-Null
Run-Logged "modal volume get quadmesh-ckpts /Quadmesh-$TIER D:\quadmesh_ckpt\Quadmesh-$TIER" "logs\17_ckpt_download.txt"
Run-Logged "modal volume get quadmesh-ckpts /tokenizer D:\quadmesh_ckpt\tokenizer" "logs\18_tokenizer_download.txt"

# Set env vars for local inference:
$env:QUADMESH_CKPT_ROOT = "D:\quadmesh_ckpt"
$env:QUADMESH_TIER      = $TIER
$env:QUADMESH_DEVICE    = "auto"

# +------------------------------------------------------------+
# | STEP 14 - TRAINING REPORT  (~2-10 min)                     |
# +------------------------------------------------------------+
Run-Logged "python -m quadmesh.pipeline.training_report --tier $TIER --ckpt-dir D:\quadmesh_ckpt\Quadmesh-$TIER --out training_report.md" "logs\19_training_report.txt"
Get-Content training_report.md

# +------------------------------------------------------------+
# | STEP 15 - DEPLOY TO MODAL  (~2-5 min)                      |
# +------------------------------------------------------------+
$env:QUADMESH_SERVE_TIER = $TIER
Run-Logged "modal deploy modal_app.py" "logs\20_deploy.txt"

# +------------------------------------------------------------+
# | STEP 16 - RUN LIVE IN BLENDER                              |
# +------------------------------------------------------------+
# Autopilot (single prompt -> Blender executes it):
python -m quadmesh.agent.autopilot "Make a 60x40x5 mm plate with 4 corner holes of 4.2 mm diameter, 8 mm from the edges" --tier $TIER --visible

# Interactive chat agent (voice + vision + full Blender control):
# python -m quadmesh.agent.chat_agent --desk --full-control --voice --speak

# +------------------------------------------------------------+
# | STEP 17 - WEB APP UI                                       |
# +------------------------------------------------------------+
pip install uvicorn python-multipart
python -m quadmesh.webapp.server
# Open http://127.0.0.1:8420 in your browser

# ==============================================================================
# LOG FILES CREATED  (in order, saved as .txt)
# ==============================================================================
#   logs\01_setup_pip.txt                logs\11_pretrain_chinchilla.txt
#   logs\02_setup_pip_local_agent.txt    logs\12_sft.txt
#   logs\03_offline_tests.txt            logs\13_vla.txt
#   logs\04_hf_login.txt                 logs\14_asr.txt
#   logs\05_modal_setup.txt              logs\15_tts.txt
#   logs\06_build_tokenizer.txt          logs\16_eval.txt
#   logs\07_review_queue_build.txt       logs\17_ckpt_download.txt
#   logs\08_review_queue_get.txt         logs\18_tokenizer_download.txt
#   logs\09_review_queue_interactive.txt logs\19_training_report.txt
#   logs\10_review_queue_put.txt         logs\20_deploy.txt
# ==============================================================================