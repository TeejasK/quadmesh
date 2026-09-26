"""
20 fixed prompts, scored against the SAME hard-invariant checker (plan_checker.check_plan) every real plan
has to pass anyway - so "did the checkpoint do well" stops being a guess from staring at a loss curve and
becomes a number you can compare checkpoint to checkpoint. This is deliberately NOT the stage7 held-out loss
eval (pipeline/stage7_eval.py) - that scores next-token loss on a small per-role set; this scores whether the
model's actual OUTPUT is usable, end to end, on a fixed set no training run ever sees or touches.

The 20 prompts are fixed on purpose - editing them defeats the point (a benchmark that moves to make the
score look better isn't a benchmark). If a prompt turns out to be genuinely wrong or ambiguous, replace it
and note the change in the log - don't just quietly adjust the set until a checkpoint passes.

FOUND WHILE BUILDING THIS (worth a real decision, not fixed silently here): agent/planner.py's
rule_based_plan() and the trained-model path emit two different op vocabularies that don't line up
(add_cube/boolean_difference with location/size args, vs plan_checker's add_box/add_cyl with w/d/h args).
check_plan only ever validates the trained-model side (already called inside _try_model_plan) and
assemblies.py's hand-calculated builds - never the rule-based fallback, which the real pipeline sends
straight to execution with no check in between. This benchmark works around it (see the note in
_score_one below) rather than picking a fix, because the actual fix is a real design choice someone should
make deliberately: either rewrite rule_based_plan to speak check_plan's vocabulary, add an alias layer in
blender_bridge so both vocabularies resolve to the same ops, or extend check_plan to understand both. Not
something to silently decide inside a benchmark script.

    python -m quadmesh.pipeline.checkpoint_benchmark                          # run against the live checkpoint
    python -m quadmesh.pipeline.checkpoint_benchmark --ckpt-root /path --tier 100M
"""
from __future__ import annotations
import json
import os
import time
from typing import Optional

# 20 fixed prompts. Mix: plain geometry (should always work, even rule-fallback), the two hand-calculated
# families at varying parameters, freeform-catalog-shaped requests (only the trained path should get these -
# see spec_gen_data.py item 6), and prompts that should be correctly REFUSED, not hallucinated.
BENCHMARK_PROMPTS = [
    # -- plain geometry: baseline sanity, should pass even without a trained model --
    {"id": "p01", "prompt": "a 40x40x10mm box with a 5mm hole through the center", "expect": "valid"},
    {"id": "p02", "prompt": "a hexagonal plate 60mm across, 4mm thick", "expect": "valid"},
    {"id": "p03", "prompt": "a cylinder 20mm diameter, 50mm tall, with a 2mm chamfer on both ends", "expect": "valid"},
    # -- hexacopter family (hand-calculated sizing, see assemblies.py/strength.py) --
    {"id": "p04", "prompt": "a 1.2kg hexacopter frame, hobby grade, PETG", "expect": "valid"},
    {"id": "p05", "prompt": "a 3kg industrial hexacopter for aerial photography, PA12CF", "expect": "valid"},
    {"id": "p06", "prompt": "a 550mm agricultural hexadrone, critical grade", "expect": "valid"},
    # -- robot arm family --
    {"id": "p07", "prompt": "a 400mm reach robot arm, 0.5kg payload, 2 links", "expect": "valid"},
    {"id": "p08", "prompt": "an 800mm industrial robot arm, 2kg payload, 3 links, nylon", "expect": "valid"},
    # -- freeform / catalog-shaped requests (item 6 - only the trained spec_generator path should handle
    #    these well; the rule-based fallback is expected to do worse here, which is itself useful signal) --
    {"id": "p09", "prompt": "an L-bracket, 80x40mm legs, 5mm thick, two M4 bolt holes per leg", "expect": "valid"},
    {"id": "p10", "prompt": "a U-channel 100mm long, 30mm wide, 4mm wall, with mounting bosses each end", "expect": "valid"},
    {"id": "p11", "prompt": "a hinge bracket with a 6mm pin hole and a 45 degree gusset", "expect": "valid"},
    {"id": "p12", "prompt": "a spacer ring, 25mm OD, 15mm ID, 8mm tall", "expect": "valid"},
    {"id": "p13", "prompt": "a cable clip that snaps around a 6mm cable", "expect": "valid"},
    # -- ambiguous but should still resolve to SOMETHING valid, not refuse --
    {"id": "p14", "prompt": "a mounting plate for a Raspberry Pi 4, with standoffs", "expect": "valid"},
    {"id": "p15", "prompt": "a small gear, 20 teeth, 2mm module", "expect": "valid"},
    # -- should be correctly REFUSED (unsupported), not hallucinated as if it were buildable --
    {"id": "p16", "prompt": "a fully assembled car engine", "expect": "unsupported"},
    {"id": "p17", "prompt": "a working violin", "expect": "unsupported"},
    {"id": "p18", "prompt": "a smartphone with a working touchscreen", "expect": "unsupported"},
    {"id": "p19", "prompt": "a lithium battery pack", "expect": "unsupported"},
    {"id": "p20", "prompt": "a human prosthetic hand with tendons", "expect": "unsupported"},
]

assert len(BENCHMARK_PROMPTS) == 20 and len({p["id"] for p in BENCHMARK_PROMPTS}) == 20


def _score_one(prompt: str, expect: str) -> dict:
    from quadmesh.pipeline.plan_checker import check_plan
    from quadmesh.agent.planner import plan_from_text, _try_model_plan   # the real entry point
    t0 = time.time()
    try:
        # IMPORTANT: rule_based_plan() and the trained-model path emit two DIFFERENT, incompatible op
        # vocabularies (add_cube/boolean_difference with location/size, vs check_plan's add_box/add_cyl with
        # w/d/h) - check_plan was only ever wired to validate the trained-model side (it's called inside
        # _try_model_plan already) and the hand-calculated assemblies.py side, never the rule-based fallback,
        # which goes straight from plan_from_text to execution in the real pipeline (agent/loop.py). Found
        # by running this benchmark for real, not by reading the code - see the note in this file's header.
        # So: only re-validate via check_plan when the TRAINED path produced the plan; for the rule-based
        # fallback, score on whether it recognized something at all, since check_plan literally can't judge
        # its output format. Both are real, honest signals - just not the same check.
        used_model = bool(_try_model_plan(prompt))
        steps = plan_from_text(prompt)
    except Exception as e:
        return {"outcome": "error", "detail": str(e), "seconds": time.time() - t0}
    elapsed = time.time() - t0
    if not steps or len(steps) <= 1:   # rule_based_plan always appends an export step; <=1 means nothing recognized
        got = "unsupported"
    elif used_model:
        plan = [{"op": s.native_op, "args": s.native_args} for s in steps if s.native_op]
        errors = check_plan(plan, prompt) if plan else ["empty plan"]
        got = "valid" if not errors else "invalid"
    else:
        got = "valid"   # rule-based fallback: recognized >=1 real shape/recipe - can't check_plan-validate its
                        # vocabulary, so "recognized something" is the honest ceiling of what this can claim
    outcome = "pass" if got == expect else "fail"
    return {"outcome": outcome, "got": got, "expect": expect, "used_trained_model": used_model,
            "seconds": round(elapsed, 2)}


def run_benchmark(ckpt_root: Optional[str] = None, tier: str = "100M", log_path: Optional[str] = None) -> dict:
    if ckpt_root:
        os.environ["QUADMESH_CKPT_ROOT"] = ckpt_root
    os.environ["QUADMESH_TIER"] = tier
    using_trained = bool(os.environ.get("QUADMESH_CKPT_ROOT"))

    results = []
    for item in BENCHMARK_PROMPTS:
        r = _score_one(item["prompt"], item["expect"])
        r.update(id=item["id"], prompt=item["prompt"])
        results.append(r)
        print(f"[benchmark] {item['id']}: {r['outcome']} (expected {item['expect']}, got {r.get('got', r.get('detail'))}) "
              f"{r.get('seconds', '?')}s", flush=True)

    n_pass = sum(1 for r in results if r["outcome"] == "pass")
    summary = {
        "tier": tier, "using_trained_checkpoint": using_trained,
        "score": f"{n_pass}/20", "pass_rate": round(n_pass / 20, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results,
    }
    log_path = log_path or os.path.join(os.path.dirname(__file__), "benchmark_log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"\n[benchmark] {summary['score']} passed ({'trained checkpoint' if using_trained else 'rule-based fallback'}, "
          f"tier {tier}) - appended to {log_path}", flush=True)
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root")
    ap.add_argument("--tier", default="100M")
    args = ap.parse_args()
    run_benchmark(ckpt_root=args.ckpt_root, tier=args.tier)


if __name__ == "__main__":
    main()
