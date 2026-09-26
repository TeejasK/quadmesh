# Shared generative text model (v6)

One trunk (`SharedTextModel`) + one tied LM head. Eight roles are selected by a role tag token and **every answer is generated text**:
chat, task_planner (plan JSON), spec_generator (`SPEC {...}`), spec_auditor (`VERDICT: PASS|FAIL`), arbitration (`CHOOSE: A|B|EITHER|NONE`),
cad_skill_dispatch (`CALL {...}` / `CLARIFY:` / `NONE`), risk_tier_classifier (`TIER: ...`), abuse_pattern (`ABUSE: yes|no`).
No classifier heads, no regex parser deciding what the user meant. `vla`, `ui_grounding`, `screenshot_diff`, `asr`, `tts` stay separate (other modalities).

## What stays outside the model on purpose (guardrails, not the brain)
* `plan_checker.check_plan` and `verification.stack` decide whether a plan/mesh is acceptable. The model proposes and audits; it cannot overrule them.
* `SharedRuntime.screen()`: keyword table always applies; the model's risk tier can only RAISE the tier; abuse "yes" blocks.
* risk/abuse are **not trained** until >= 500 human-approved rows exist per role (`python -m quadmesh.pipeline.review_queue`). The checkpoint records this and the
  runtime raises `RoleNotTrained` / reports `model_screen: untrained` instead of pretending.

## Sizes (parameters, incl. embeddings)
| tier | shared trunk | 8 separate models |
|---|---|---|
| 100M | 92.6M | 153.8M |
| 500M | 440M | 529.8M |
| 1B | 845M | 978.2M |

## Order (see commands.txt)
`build_tokenizer` -> `shared_review_queue` (+ review) -> `shared_pretrain` -> `shared_sft` -> `shared_eval`.
Pretraining = SlimPajama + Text2CAD (in full). SFT = generated, verifier-grounded role data, all roles mixed in every batch, loss on answers only.

## Compute estimate (my arithmetic, NOT measured)
6 x params x tokens on one H100 at an assumed 30% of peak; H100 $3.95/h (Modal list price, third-party pricing pages, Aug 2026; CPU/RAM billed separately).
If real utilisation is ~20%, multiply by 1.5. One GPU only (no multi-GPU code).

| trunk | 20 tokens/param | 80 tokens/param (default) | all 627B SlimPajama tokens |
|---|---|---|---|
| 92.6M (100M tier) | 1.9B tok, ~1 h, ~$4 | 7.4B tok, ~4 h, ~$15 | ~320 h, ~$1,270 |
| 440M (500M tier) | 8.8B tok, ~21 h, ~$85 | 35B tok, ~86 h, ~$340 | ~1,535 h (64 days), ~$6,000 |
| 845M (1B tier) | 16.9B tok, ~79 h, ~$315 | 67.6B tok, ~318 h, ~$1,255 | ~2,940 h, ~$11,600 |

SFT (500M tier, default 2B tokens): ~5 h, ~$20. Evaluation (L4): a few dollars.
Suggested path: 100M tier end to end first (~$20) to validate the pipeline and eval, then 500M at 80 tokens/param (~$360).

## Acceptance (my proposed thresholds - change them, but decide BEFORE looking at a run)
planner checker_pass >= 0.90 - dispatch exact_call >= 0.95 - auditor verdict_acc >= 0.95 - arbitration choice >= 0.90 - spec_generator exact >= 0.90 - benchmark_20 spec_route_pass >= 16/20.
A 2.9M-parameter model trained for 10 minutes on a CPU reached held-out loss ~0.9 on the planner yet scored 0.0 on checker_pass:
loss alone does not tell you the output works - use `shared_eval`.

## Known limits
* Requests far from the phrasing/parts the generators cover will degrade; that is a data limit, not a code one. Add real requests to the data.
* Packed rows are not attention-masked between examples.
* The autopilot uses the planner + checker; `audit`/`arbitrate` are available on the runtime but are not in its loop.
* Image chat: this trunk is text-only; a photo is announced as `IMAGE: attached photo` and routed to `photo_to_3d`. The older multimodal chat model is unchanged.
* Not run against a real Blender/GPU in the sandbox that produced this. The offline suite (29 checks) uses a fake Blender; the trainer was exercised end to end on CPU with a tiny model.
