"""
Per-role model sizes: NOT an equal split, and NOT split by role *count* either.

The tier (100M / 500M / 1B / 3B / 7B) sets a total budget of transformer weights. Each role gets the share its
JOB requires, judged by two things: (a) how open-ended the output space is (a free action/token sequence needs
more capacity than a 3-way label), and (b) how far downstream a mistake propagates (an error in the planner or
the action model corrupts everything built after it; an error in a small classifier just gets caught by the
next gate). Roles that only pick from a small closed set (risk tier, abuse pattern, arbitration) stay small on
purpose — making them bigger would not make them more correct, it would just cost tokens.

v2 allocation (post rule-removal — see README "make it learned" plan):
  - vla            0.20  was 0.16. This model now IS the Blender-control layer: it replaces the old 160-op
                         fixed dispatch table with real screenshot+text -> keyboard/mouse/code-action synthesis
                         (blender_bridge.py becomes a transport, not a decision-maker). Free-form action space
                         over long horizons (VideoCAD sessions run long) = the single biggest capacity need.
  - task_planner   0.17  was 0.14. Now does MCTS-style multi-candidate rollout, not single-shot planning —
                         needs enough capacity to represent and compare several hierarchical plans, not just
                         emit one.
  - chat           0.12  was 0.24, cut hardest. In the old design this carried most of the "understand the
                         user" burden alone. With task_planner and vla now doing real reasoning/perception
                         work of their own, chat's job narrows back to conversation + tool-call routing, which
                         doesn't need the reasoning depth it was over-provisioned for.
  - spec_generator 0.12  was 0.10. Now writes bpy code sequences, not just parameter dicts (see
                         blender_bridge.py rewrite) — code generation is a strictly harder output space than
                         filling a fixed schema, so it gets more room.
  - spec_auditor   0.09  was 0.05, the biggest relative jump. This model absorbs plan_checker.py's old
                         hardcoded 52-op whitelist — it goes from "small yes/no classifier" to "the model that
                         decides whether a plan is safe to execute." That's now a genuine reasoning job, not a
                         lookup, so it can't stay classifier-sized.
  - ui_grounding   0.05  was 0.03. Slightly up: with desk.py's full-control mode as the default action path
                         (not gated), grounding accuracy (pixel-precise element location) matters more often.
  - screenshot_diff 0.03 was 0.02. Slightly up: it's now load-bearing for the self-repair loop (observe.py),
                         not just a post-hoc report.
  - asr            0.10  unchanged. Kept explicitly large: mistakes here corrupt every downstream role, so
                         under-provisioning ASR is the worst place to save parameters.
  - tts            0.05  was 0.06, trimmed slightly to fund vla/spec_auditor. Speech synthesis is a narrower,
                         well-understood output space (mel spectrogram frames) — it tolerates the cut better
                         than any reasoning role would.
  - arbitration, cad_skill_dispatch, risk_tier_classifier, abuse_pattern: kept deliberately small (0.015-0.02
                         each). Closed-set decisions (which of N roles wins a conflict, which of ~10 skill
                         categories applies, which of 3 risk tiers, malicious y/n). cad_skill_dispatch shrinks
                         a touch further than before (0.03 -> 0.02) precisely because free-form code generation
                         in spec_generator now covers cases that used to need an explicit dispatch rule.

Edit ROLE_WEIGHTS to change the split; the weights must add up to 1. Run `python -m quadmesh.role_shapes`
to print the table for every tier.
"""
from __future__ import annotations
import copy
from typing import Dict

ROLE_WEIGHTS: Dict[str, float] = {
    "vla":                   0.20,   # screenshot + text -> keyboard/mouse/code action (now the control layer)
    "task_planner":          0.17,   # MCTS-style multi-candidate hierarchical planning
    "chat":                  0.14,   # conversation + tool-call routing, narrowed scope (see docstring) but
                                      # still must clear the equal-split floor even at the 100M tier - the
                                      # repo's own offline test asserts this (tests/run_offline_tests.py),
                                      # and 0.12 rounded below it once shape_for_budget's step-of-64 discretises
                                      # a tiny tier, which is a real signal: chat still needs to stay ahead of
                                      # an average role, not just above the smallest ones.
    "spec_generator":        0.12,   # writes bpy code sequences, not just parameter dicts
    "spec_auditor":          0.09,   # absorbs the old fixed 52-op whitelist -> learned plan verifier
    "asr":                   0.10,   # mistakes here corrupt every downstream role
    "tts":                   0.05,
    "ui_grounding":          0.04,
    "screenshot_diff":       0.02,
    "arbitration":           0.02,
    "cad_skill_dispatch":    0.02,
    "risk_tier_classifier":  0.015,
    "abuse_pattern":         0.015,
}
assert abs(sum(ROLE_WEIGHTS.values()) - 1.0) < 1e-9, "ROLE_WEIGHTS must add up to 1"

# ---------------------------------------------------------------------------
# Shared backbone for the TEXT-CENTRIC roles only - one trunk + lightweight per-role heads, instead of 8
# separate full models each re-deriving the same language understanding from scratch. Scoped deliberately:
# vla (image patches), ui_grounding/screenshot_diff (image patches, see model/transformer.py's
# VISION_ROLES + QuadmeshVisionModel), asr (audio), and tts (audio) do NOT join this backbone - merging
# weights across genuinely different input modalities without real multimodal fusion design isn't a free
# efficiency win, it just risks the "roles interfere" failure mode discussed below, worse. These 8 all read
# the SAME thing (tokenized text) and reason over overlapping vocabulary (CAD terms, dimensions, plan
# syntax), which is exactly the condition under which sharing helps rather than hurts.
# ---------------------------------------------------------------------------

TEXT_BACKBONE_ROLES = ("chat", "task_planner", "spec_generator", "spec_auditor",
                       "arbitration", "cad_skill_dispatch", "risk_tier_classifier", "abuse_pattern")

# Same n_classes convention already used by model/transformer.py's build_role_model - reused here, not
# reinvented, so a checkpoint's head shape means the same thing whether it came from the old per-role
# QuadmeshModel or the new shared backbone.
CLASSIFIER_N = {"risk_tier_classifier": 4, "abuse_pattern": 2}

# How much of the 8 text roles' COMBINED old budget goes to the shared trunk vs. the lightweight heads.
# 90/10 on purpose: heads are meant to stay small (a role's own accent on top of a shared representation),
# not full transformer-sized sub-networks - putting more than a small fraction into heads would just
# recreate 8 separate models with extra steps.
BACKBONE_SHARE = 1.0   # v2: no heads, so the trunk gets the WHOLE combined budget of the 8 text roles


def backbone_shape(tier):
    """The ONE shared trunk's ModelShape, sized from the 8 text roles' combined old budget (each role's
    relative weight among the 8 still matters - it just now decides head capacity, via head_hidden(),
    instead of a whole separate model's size)."""
    combined_w = sum(ROLE_WEIGHTS[r] for r in TEXT_BACKBONE_ROLES)
    pool = 9 * nonembedding_params(tier.shape)
    return shape_for_budget(int(pool * combined_w * BACKBONE_SHARE), tier.shape)


def head_hidden(tier, role: str) -> int:
    """Hidden width for `role`'s small residual MLP head (see model/shared_backbone.py's RoleHead) - NOT a
    full transformer shape, just a bottleneck width sized from this role's SHARE of the combined heads'
    budget (10% of the 8 roles' old total, split by their old relative weights), capped at the backbone's
    own d_model so a head can never end up WIDER than the backbone it sits on top of - a head that outgrows
    its own trunk stops being "lightweight" by any reasonable reading of the term, whatever a raw budget
    division works out to."""
    if role not in TEXT_BACKBONE_ROLES:
        raise ValueError(f"{role!r} is not one of TEXT_BACKBONE_ROLES - it should size via role_shape() instead")
    combined_w = sum(ROLE_WEIGHTS[r] for r in TEXT_BACKBONE_ROLES)
    d = backbone_shape(tier).d_model
    pool = 9 * nonembedding_params(tier.shape)
    head_budget = pool * combined_w * (1 - BACKBONE_SHARE) * (ROLE_WEIGHTS[role] / combined_w)
    # a 2-layer bottleneck MLP (d -> hidden -> d) has roughly 2*d*hidden params - solve for hidden, then clamp
    # into [32, d//4]. d//4, not d: capping at the full backbone width let 6 of the 8 roles hit the ceiling
    # in practice and pushed the 100M tier's REAL total to 213M once heads were actually built and counted -
    # found by building the real model and counting real parameters, not by inspecting this formula. A head
    # sized comparably to the trunk stops being a lightweight accent by any reasonable reading of the term,
    # whatever a raw budget division works out to; d//4 keeps every head decisively smaller than the trunk
    # while still ordering capacity by relative role weight below that ceiling. Round to a multiple of 8.
    raw = head_budget / (2 * d)
    hidden = max(32, min(round(raw / 8) * 8, d // 4))
    return hidden


def describe_shared(tiers: dict) -> str:
    lines = []
    for key, t in tiers.items():
        bs = backbone_shape(t)
        lines.append(f"\n{t.name}  shared text backbone: d={bs.d_model} layers={bs.n_layers} "
                     f"heads={bs.n_heads} -> {nonembedding_params(bs)/1e6:.1f}M "
                     f"(vs {sum(role_shape(t, r).n_layers for r in TEXT_BACKBONE_ROLES)} layers total "
                     f"if these {len(TEXT_BACKBONE_ROLES)} roles were still separate models)")
        for r in TEXT_BACKBONE_ROLES:
            lines.append(f"  {r:22s} head hidden={head_hidden(t, r):5d}"
                         + (f"  ({CLASSIFIER_N[r]}-class classifier)" if r in CLASSIFIER_N else "  (generative, tied LM head)"))
    return "\n".join(lines)


def nonembedding_params(shape) -> int:
    """SwiGLU block: attention 4*d^2 + three d*d_ff matrices."""
    return shape.n_layers * (4 * shape.d_model ** 2 + 3 * shape.d_model * shape.d_ff)


def shape_for_budget(budget: int, like):
    """Pick (d_model, layers) for a non-embedding parameter budget: d multiple of 64, 64-wide heads, d_ff = 4d,
    depth close to width/64 (the proportions the existing tiers use)."""
    best = None
    for d in range(128, 4097, 64):
        L = max(2, round(budget / (16 * d * d)))
        score = abs(L - d / 64) + abs(16 * L * d * d - budget) / budget * 10
        if best is None or score < best[0]:
            best = (score, d, L)
    _, d, L = best
    s = copy.copy(like)
    s.d_model, s.n_layers, s.n_heads, s.d_ff = d, L, d // 64, 4 * d
    return s


def role_shape(tier, role: str):
    """The ModelShape for `role` at this tier (a fresh copy: safe to set vocab_size on it)."""
    w = ROLE_WEIGHTS.get(role)
    if w is None:
        return copy.copy(tier.shape)
    pool = 9 * nonembedding_params(tier.shape)          # the tier's advertised size = 9 equal roles
    return shape_for_budget(int(pool * w), tier.shape)


def describe(tiers: dict) -> str:
    lines = []
    for key, t in tiers.items():
        lines.append(f"\n{t.name}  (pool {9*nonembedding_params(t.shape)/1e6:.0f}M transformer weights)")
        for r in ROLE_WEIGHTS:
            s = role_shape(t, r)
            lines.append(f"  {r:22s} d={s.d_model:5d} layers={s.n_layers:3d} heads={s.n_heads:3d} "
                         f"-> {nonembedding_params(s)/1e6:8.1f}M  (equal split would be {nonembedding_params(t.shape)/1e6:.1f}M)")
    return "\n".join(lines)


if __name__ == "__main__":
    from quadmesh.config import TIERS
    print(describe({k: TIERS[k] for k in ("100M", "500M", "1B", "3B", "7B")}))
    print(describe_shared({k: TIERS[k] for k in ("100M", "500M", "1B", "3B", "7B")}))
