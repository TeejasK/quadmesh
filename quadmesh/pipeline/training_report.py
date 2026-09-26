"""
Build a documentation report of a completed training run, straight from the real
artifacts already on the Modal volume — no numbers are invented here.

Sources used (all already produced by shared_text_train.py / shared_eval.py, this
script only reads and formats them):
    <tier>/live/metrics_pretrain.jsonl   - one JSON line every log_every steps: step, loss, grad_norm, lr, tok_per_s
    <tier>/live/metrics_sft.jsonl        - same, plus periodic {"step":.., "val": {role: held-out loss}}
    <tier>/live/shared_pretrain.pt       - checkpoint header (step, total_steps, done, tier, phase)
    <tier>/live/shared_sft.pt            - same, for the SFT phase
    <tier>/eval_<timestamp>.json         - shared_eval report (per-role metrics + benchmark_20)

WHY "only successful runs": a checkpoint's "done" flag is only ever set True by
save(final=True), which only runs after the training loop exits normally (all
total_steps completed). A run that crashed, OOM'd, or was killed mid-way leaves
"done": False on its last saved checkpoint. This script checks that flag and
refuses to write a report section for a phase that isn't done, instead of
silently presenting a partial/crashed run's numbers as if they were a finished
result.

Usage (after pulling the volume locally, e.g.
  modal volume get quadmesh-ckpts /Quadmesh-3B ./Quadmesh-3B
):
    python -m quadmesh.pipeline.training_report --tier 3B --ckpt-dir ./Quadmesh-3B --out training_report.md
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import time
from typing import Optional


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ckpt_header(path: str) -> Optional[dict]:
    """Reads just the small scalar fields out of a checkpoint without needing torch/GPU."""
    if not os.path.exists(path):
        return None
    try:
        import torch
        d = torch.load(path, map_location="cpu", weights_only=False)
        return {k: d.get(k) for k in ("step", "total_steps", "done", "tier", "phase", "safety_roles_trained")}
    except ImportError:
        return {"error": "torch not installed in this environment - cannot read checkpoint header; "
                          "the metrics_*.jsonl section below is still accurate"}


def _phase_section(name: str, ckpt_dir: str, phase: str) -> str:
    ck = _ckpt_header(os.path.join(ckpt_dir, "live", f"shared_{phase}.pt"))
    metrics = _read_jsonl(os.path.join(ckpt_dir, "live", f"metrics_{phase}.jsonl"))
    loss_rows = [r for r in metrics if "loss" in r]
    val_rows = [r for r in metrics if "val" in r]

    out = [f"## {name}\n"]
    if ck is None:
        out.append("_No checkpoint found for this phase - it has not been run yet._\n")
        return "\n".join(out)
    if ck.get("error"):
        out.append(f"_{ck['error']}_\n")
    if not ck.get("done"):
        out.append(f"**NOT INCLUDED — this run has not finished.** Last saved at step "
                    f"{ck.get('step', '?')}/{ck.get('total_steps', '?')}, `done=False`. "
                    f"Re-run the same `shared_{phase}` command to resume; this report only documents "
                    f"completed runs, so no loss numbers are shown for an unfinished phase.\n")
        return "\n".join(out)

    out.append(f"**Status: completed.** {ck['step']:,} / {ck['total_steps']:,} steps, `done=true`.\n")
    if loss_rows:
        first, last = loss_rows[0], loss_rows[-1]
        min_loss_row = min(loss_rows, key=lambda r: r["loss"])
        avg_tok_s = sum(r["tok_per_s"] for r in loss_rows) / len(loss_rows)
        out.append(f"- Loss: {first['loss']} (step {first['step']}) → **{last['loss']}** (step {last['step']}, final)")
        out.append(f"- Lowest logged loss: {min_loss_row['loss']} at step {min_loss_row['step']}")
        out.append(f"- Average throughput: {avg_tok_s:,.0f} tokens/sec ({len(loss_rows)} logged points, "
                    f"every N steps per `log_every`)")
        out.append("")
        out.append("| step | loss | grad_norm | lr | tok/s |")
        out.append("|---|---|---|---|---|")
        # sample down to at most ~25 rows so the table stays readable on a long run
        step_n = max(1, len(loss_rows) // 25)
        for r in loss_rows[::step_n] + ([loss_rows[-1]] if loss_rows[-1] not in loss_rows[::step_n] else []):
            out.append(f"| {r['step']:,} | {r['loss']} | {r['grad_norm']} | {r['lr']} | {r['tok_per_s']:,} |")
    else:
        out.append("_No per-step loss rows found in metrics file — was `log_every` reached before completion?_")

    if val_rows:
        out.append("\n**Held-out loss per role (SFT eval checkpoints):**\n")
        out.append("| step | " + " | ".join(sorted(val_rows[-1]["val"].keys())) + " |")
        out.append("|---|" + "---|" * len(val_rows[-1]["val"]))
        for r in val_rows:
            vals = r["val"]
            out.append(f"| {r['step']:,} | " + " | ".join(str(vals.get(k, "-")) for k in sorted(vals.keys())) + " |")

    return "\n".join(out) + "\n"


def _eval_section(ckpt_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(ckpt_dir, "eval_*.json")))
    if not files:
        return "## Evaluation\n\n_No `shared_eval` report found yet — run `modal run modal_app.py::shared_eval --tier <tier>`._\n"
    latest = files[-1]
    rep = json.load(open(latest))
    out = [f"## Evaluation\n\n_Source: `{os.path.basename(latest)}`_\n"]
    out.append("| role | metrics |")
    out.append("|---|---|")
    for role, m in rep.items():
        if role == "benchmark_20":
            continue
        out.append(f"| {role} | {json.dumps(m)} |")
    if "benchmark_20" in rep:
        out.append(f"\n**Benchmark (20 hand-written prompts):** {json.dumps(rep['benchmark_20'])}\n")
    return "\n".join(out) + "\n"


def build_report(tier: str, ckpt_dir: str) -> str:
    header = (f"# Quadmesh {tier} — Training Report\n\n"
              f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from `{ckpt_dir}`.\n\n"
              f"This report only includes phases whose checkpoint has `done: true` — "
              f"a crashed, OOM'd, or preempted-and-not-yet-resumed run is called out as "
              f"not-finished rather than shown with partial numbers.\n")
    return "\n".join([
        header,
        _phase_section("Pretrain", ckpt_dir, "pretrain"),
        _phase_section("SFT", ckpt_dir, "sft"),
        _eval_section(ckpt_dir),
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--ckpt-dir", required=True, help="local folder pulled via `modal volume get`, e.g. ./Quadmesh-3B")
    ap.add_argument("--out", default="training_report.md")
    a = ap.parse_args()
    report = build_report(a.tier, a.ckpt_dir)
    open(a.out, "w").write(report)
    print(f"wrote {a.out}")
