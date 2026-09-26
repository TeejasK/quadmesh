"""
Stage 3 — Self-Generation / Rejection Sampling (Sec 15).

The model proposes candidates; the VERIFICATION STACK (never the model)
grades pass/fail. Passing candidates become the Stage 4 preference-pair pool
and the Stage 2-style SFT set for non-preference-tuned roles.
"""
from __future__ import annotations
from quadmesh.role_shapes import role_shape
import os, json, torch

from quadmesh.config import TierConfig
from quadmesh.model.transformer import build_role_model


def generate_candidates(model, tokenizer, prompts, n_per_prompt, device, max_new=400):
    model.eval()
    out = []
    total = len(prompts)
    with torch.no_grad():
        for pi, p in enumerate(prompts, 1):
            ids = torch.tensor([tokenizer.encode(p)], device=device)
            for ci in range(n_per_prompt):
                cur = ids.clone()
                for _ in range(max_new):
                    logits = model(cur)["logits"][:, -1]
                    nxt = torch.multinomial(torch.softmax(logits, -1), 1)
                    cur = torch.cat([cur, nxt], dim=1)
                    if nxt.item() == tokenizer.eos_id:
                        break
                out.append({"prompt": p, "text": tokenizer.decode(cur[0].tolist())})
            print(f"  [gen] prompt {pi}/{total} done ({n_per_prompt} candidates)", flush=True)
    return out


def grade_candidates(role: str, candidates: list[dict]) -> list[dict]:
    """Non-CAD roles (planner, auditor, etc.) get a lighter textual grader;
    spec_generator candidates go through the full geometry/manufacturability
    verification stack once meshed (Sec 3-8)."""
    if role == "task_planner":                      # REAL grader: geometry + prompt-consistency checker
        from quadmesh.pipeline.plan_checker import grade_text
        for c in candidates:
            c["passed"], c["errors"] = grade_text(c["prompt"], c["text"])
        return candidates
    if role == "spec_generator":
        from quadmesh.verification.stack import grade_for_rejection_sampling
        for c in candidates:
            if "mesh_path" in c and os.path.exists(c.get("mesh_path", "")):
                c["passed"] = grade_for_rejection_sampling(type("C", (), c)())
            else:
                txt = c.get("text", "").strip()
                from quadmesh.spec_gen_data import parse_spec
                c["passed"] = parse_spec("SPEC " + txt) is not None
    else:
        for c in candidates:
            c["passed"] = len(c["text"].strip()) > 0 and not c["text"].strip().endswith(("...", "<|eos|>"))
    return candidates


def run_stage3(role: str, tier: TierConfig, tokenizer, ckpt_dir: str,
               prompts: list[str], n_per_prompt: int = 4, device: str = "cuda"):
    shape = role_shape(tier, role)
    shape.vocab_size = tokenizer.vocab_size
    model = build_role_model(role, shape).to(device)

    # Check stage 2 first, then stage 1
    src = None
    for s in ("stage2", "stage1"):
        cand = f"{ckpt_dir}/{s}/{role}.{s}.pt"
        if os.path.exists(cand):
            src = cand
            break
    if not src:
        raise FileNotFoundError(f"No stage1 or stage2 checkpoint found for {role} under {ckpt_dir}")

    sd = torch.load(src, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])

    cands = generate_candidates(model, tokenizer, prompts, n_per_prompt, device)
    cands = grade_candidates(role, cands)

    out_dir = f"{ckpt_dir}/stage3"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{role}.pairs.jsonl"
    with open(out_path, "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")
    passed = sum(c["passed"] for c in cands)
    print(f"[stage3][{role}] {passed}/{len(cands)} passed verification", flush=True)
    return out_path
