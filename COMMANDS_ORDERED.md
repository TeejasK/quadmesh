# Quadmesh — ordered PowerShell commands (tier = 3B example)

Run each command to completion before moving to the next unless noted otherwise.
Set your tier once:
```powershell
$TIER = "3B"
```

## 1. Setup — one-time (~10–20 min total)

```powershell
pip install -r requirements.txt
pip install -r requirements-local-agent.txt
python tests/run_offline_tests.py
huggingface-cli login
modal setup
```
Accept the Text2CAD license (same HF account as your token) BEFORE the next step:
https://huggingface.co/datasets/SadilKhan/Text2CAD — otherwise the tokenizer trains with 0 CAD docs.

```powershell
python -c "import getpass,subprocess; t=getpass.getpass('HF token: '); subprocess.run(['modal','secret','create','huggingface-secret',f'HF_TOKEN={t}'])"
```
~1 min.

## 2. Tokenizer — once for the whole project, not per tier

```powershell
modal run modal_app.py::build_tokenizer
```
**Estimated time: 30–60 min** (with the corpus-caching fix — a preempted/retried attempt resumes
instead of restarting, so worst case is still well under the ~4 hrs you saw before).
Confirms success when you see `[tokenizer] Training complete! Vocab size: 32,000`.

```powershell
modal run modal_app.py::shared_review_queue
modal volume get quadmesh-ckpts /data/review_queue.jsonl review_queue.jsonl
```
~2–5 min combined. Then, in a separate terminal (interactive, not scriptable):
```powershell
python -m quadmesh.pipeline.review_queue review_queue.jsonl
```
Time depends on how much you review by hand.
```powershell
modal volume put quadmesh-ckpts review_queue.jsonl /data/review_queue.jsonl --force
```
~1 min.

## 3. Pretrain — chinchilla budget

```powershell
modal run modal_app.py::shared_pretrain --tier $TIER --budget chinchilla --mb 16 --accum 16 --grad_ckpt 0 --workers 12
```
**Estimated time for 3B / chinchilla: several hours up to the tier's 24h timeout ceiling**
(config.py sets `timeout_hours=24` for 3B as the budgeted ceiling — actual completion is usually
well under that on a single H100, but depends on throughput; watch `tok_per_s` in the metrics log
below to project your own finish time: `remaining_tokens / tok_per_s`).
Run this in the foreground (no `--detach` needed) — see the loss-logging section below for how to
capture per-step numbers to a file while it runs.

Wait for `done: true` before SFT (SFT loads this checkpoint as its starting point).

## 4. SFT

```powershell
modal run modal_app.py::shared_sft --tier $TIER --tokens 2000000000 --mb 16 --accum 16 --grad_ckpt 0 --workers 12
```
**Estimated time for 3B: roughly 1–4 hours**, well under the tier's timeout ceiling — 2B SFT
tokens is much smaller than the chinchilla pretrain budget.

## 5. Eval

```powershell
modal run modal_app.py::shared_eval --tier $TIER --n 100
```
**Estimated time: 5–15 min.**

## 6. Documentation

```powershell
modal volume get quadmesh-ckpts /Quadmesh-$TIER ./Quadmesh-$TIER
python -m quadmesh.pipeline.training_report --tier $TIER --ckpt-dir ./Quadmesh-$TIER --out training_report.md
Get-Content training_report.md
```
**Estimated time: 2–10 min** (mostly the checkpoint download size).
Only documents a phase with `done: true` — a crashed/partial/still-running phase is called out
explicitly rather than shown with misleading numbers.

## 7. Optional — full 4x run, only if chinchilla results justify it

Same three commands as steps 3–5 with `--budget 4x` instead of `--budget chinchilla`.
**Estimated time: roughly 4x the chinchilla pretrain time.** This OVERWRITES the chinchilla
checkpoint on the volume (same filename) — not additive.

---

## Getting per-step loss values while training (no --detach, PowerShell)

`--detach` only controls whether the remote job survives after your local process exits — it does
not control whether you see logs, and it is NOT required to get per-step loss.

**Option A — tee the live console output to a file** (run in the same terminal as your training command):
```powershell
modal run modal_app.py::shared_pretrain --tier $TIER --budget chinchilla --mb 16 --accum 16 --grad_ckpt 0 --workers 12 | Tee-Object -FilePath pretrain_console.log
```

**Option B (recommended for documentation) — poll the real metrics file directly off the volume.**
This is the authoritative source (`{"step":..,"loss":..,"grad_norm":..,"lr":..,"tok_per_s":..}` written
every `log_every` steps by the trainer itself), independent of your terminal session. Run this in a
**second** PowerShell window while training runs in the first:
```powershell
.\scripts\poll_loss.ps1 -Tier $TIER -Phase pretrain
```
It writes/updates `loss_log_pretrain_3B.csv` every 2 minutes (configurable with `-IntervalSec`) with
only newly-appeared steps, so you get a clean, deduplicated, append-only CSV ready to drop into a
report or plot — this is the file to use for documentation rather than scraping console scrollback.
Switch `-Phase sft` for the SFT run afterwards.
