"""
Quadmesh training on Modal (Sec 16).

  modal run modal_app.py::train_tier --tier 100M
  modal run modal_app.py::train_tier --tier 500M
  modal run modal_app.py::train_role --tier 1B --role spec_generator --stage 1
  modal run modal_app.py::train_io   --tier 100M --which asr

Nothing downloads SlimPajama. `streaming=True` pulls records over the network;
the loop stops pulling at the tier token budget. Checkpoints go to a Modal
Volume, never the ephemeral instance disk.
"""

import os
import modal

from quadmesh.config import TIERS, NINE_ROLES, get_tier

APP = "quadmesh"

# Which tier's checkpoint the deployed web() endpoint serves. Read from an environment variable at
# deploy time (decorators run locally when you invoke `modal deploy`), so you configure it without
# editing this file:
#
#   $env:QUADMESH_SERVE_TIER = "3B"
#   modal deploy modal_app.py
#
# Defaults to 500M if unset.
SERVE_TIER = os.environ.get("QUADMESH_SERVE_TIER", "500M")
if SERVE_TIER not in TIERS:
    raise ValueError(f"QUADMESH_SERVE_TIER={SERVE_TIER!r} is not a known tier ({sorted(TIERS)}). "
                      f"Set it before running modal deploy, e.g.  $env:QUADMESH_SERVE_TIER = \"3B\"")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "datasets==3.2.0",
        "huggingface_hub==0.27.0",
        "tokenizers==0.21.0",
        "hf_transfer",
        "numpy",
        "soundfile",
        "librosa",
        "pillow",
        "trimesh",
        "numpy-stl",
        "fastapi[standard]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("quadmesh")
)

app = modal.App(APP, image=image)

# Persistent across runs. Checkpoints + tokenizer only — no dataset cache.
vol = modal.Volume.from_name("quadmesh-ckpts", create_if_missing=True)
VOL = "/vol"

# Needed only for gated datasets (Common Voice). SlimPajama/Text2CAD are open.
secrets = [modal.Secret.from_name("huggingface-secret")]


def _fn(tier_key: str, **kw):
    """Builds a Modal function decorator sized for the tier."""
    t = TIERS[tier_key]
    return app.function(
        gpu=t.modal_gpu_spec,
        timeout=t.timeout_hours * 3600,
        volumes={VOL: vol},
        secrets=secrets,
        **kw,
    )


# ---------------------------------------------------------------------------
# Tokenizer (run once; cached on the volume)
# ---------------------------------------------------------------------------

@app.function(timeout=8 * 3600, volumes={VOL: vol}, secrets=secrets, cpu=16.0, memory=32768)
def build_tokenizer(vocab_size: int = 32000):
    from quadmesh.data.tokenizer import get_or_train
    # commit_fn is called periodically while streaming so a preemption never throws away already-streamed
    # corpus docs - only progress since the last commit (every ~2,000 docs) is ever lost.
    tok = get_or_train(f"{VOL}/tokenizer/quadmesh-bpe.json", vocab_size, commit_fn=vol.commit)
    print("vocab:", tok.vocab_size)


# ---------------------------------------------------------------------------
# One role, one stage
# ---------------------------------------------------------------------------

def _train_role_body(tier_key: str, role: str, stage: int):
    from quadmesh.config import get_tier
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.train.loop import train_role

    tier = get_tier(tier_key)
    tok = get_or_train(f"{VOL}/tokenizer/quadmesh-bpe.json")
    path = train_role(role, stage, tier, tok,
                      ckpt_dir=f"{VOL}/{tier.name}/stage{stage}")
    vol.commit()
    return path


@_fn("100M")
def train_100M(role: str = "spec_generator", stage: int = 1):
    return _train_role_body("100M", role, stage)

@_fn("500M")
def train_500M(role: str = "spec_generator", stage: int = 1):
    return _train_role_body("500M", role, stage)

@_fn("1B")
def train_1B(role: str = "spec_generator", stage: int = 1):
    return _train_role_body("1B", role, stage)

@_fn("3B")
def train_3B(role: str = "spec_generator", stage: int = 1):
    return _train_role_body("3B", role, stage)

@_fn("7B")
def train_7B(role: str = "spec_generator", stage: int = 1):
    return _train_role_body("7B", role, stage)

_DISPATCH = {"100M": train_100M, "500M": train_500M, "1B": train_1B,
             "3B": train_3B, "7B": train_7B}


# ---------------------------------------------------------------------------
# I/O adapters (Sec 17) — ASR and TTS, from scratch
# ---------------------------------------------------------------------------

@app.function(gpu="H100", timeout=24 * 3600, volumes={VOL: vol}, secrets=secrets)
def train_io_model(tier: str = "100M", which: str = "asr", language: str = "en"):
    from quadmesh.config import get_tier
    from quadmesh.train.io_train import train_asr, train_tts
    vol.reload()                      # see checkpoints committed by earlier runs
    t = get_tier(tier)
    if which == "asr":
        out = train_asr(t, language=language, ckpt_dir=f"{VOL}/{t.name}/io")
    else:
        out = train_tts(t, ckpt_dir=f"{VOL}/{t.name}/io")
    vol.commit()                      # persist the new checkpoint
    return out


# ---------------------------------------------------------------------------
# Stages 3-8 (Sec 15) — one role, one tier, one stage at a time
# ---------------------------------------------------------------------------

STAGE3_PROMPTS = [
    "Generate a hexacopter drone frame arm, printable in two halves.",
    "Design a bracket mounting a 12mm bearing to an aluminum extrusion.",
    "Model a snap-fit enclosure lid for a 60x40x20mm PCB.",
]


def _stage3_body(tier_key: str, role: str):
    from quadmesh.config import get_tier
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.pipeline.stage3_selfgen import run_stage3
    tier = get_tier(tier_key)
    tok = get_or_train(f"{VOL}/tokenizer/quadmesh-bpe.json")
    prompts = STAGE3_PROMPTS
    if role == "task_planner":
        from quadmesh.pipeline.datasets.engineering_gen import gen_prompts
        prompts = gen_prompts(60, seed=7)
    elif role == "spec_generator":
        from quadmesh.spec_gen_data import gen_example
        prompts = [gen_example()["prompt"] + " SPEC" for _ in range(60)]
    path = run_stage3(role, tier, tok, ckpt_dir=f"{VOL}/{tier.name}", prompts=prompts)
    vol.commit()
    return path


def _stage4_body(tier_key: str, role: str):
    from quadmesh.config import get_tier
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.pipeline.stage4_preference import run_stage4
    tier = get_tier(tier_key)
    tok = get_or_train(f"{VOL}/tokenizer/quadmesh-bpe.json")
    path = run_stage4(role, tier, tok, ckpt_dir=f"{VOL}/{tier.name}")
    vol.commit()
    return path


def _stage5_body(tier_key: str, role: str):
    from quadmesh.config import get_tier
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.pipeline.stage5_calibration import run_stage5
    tier = get_tier(tier_key)
    tok = get_or_train(f"{VOL}/tokenizer/quadmesh-bpe.json")
    path = run_stage5(role, tier, tok, ckpt_dir=f"{VOL}/{tier.name}")
    vol.commit()
    return path


def _stage6_body(tier_key: str, role: str):
    from quadmesh.config import get_tier
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.pipeline.stage6_redteam import run_stage6
    tier = get_tier(tier_key)
    tok = get_or_train(f"{VOL}/tokenizer/quadmesh-bpe.json")
    result = run_stage6(role, tier, tok, ckpt_dir=f"{VOL}/{tier.name}")
    vol.commit()
    return result


def _stage7_body(tier_key: str, role: str):
    from quadmesh.config import get_tier
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.pipeline.stage7_eval import run_stage7
    tier = get_tier(tier_key)
    tok = get_or_train(f"{VOL}/tokenizer/quadmesh-bpe.json")
    report = run_stage7(role, tier, tok, ckpt_dir=f"{VOL}/{tier.name}")
    vol.commit()
    return report


def _stage8_body(tier_key: str, role: str, promote: bool):
    from quadmesh.config import get_tier
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.pipeline.stage8_shadow import run_stage8, promote_to_live
    tier = get_tier(tier_key)
    tok = get_or_train(f"{VOL}/tokenizer/quadmesh-bpe.json")
    result = run_stage8(role, tier, tok, ckpt_dir=f"{VOL}/{tier.name}",
                        shadow_inputs=STAGE3_PROMPTS)
    if promote:
        promote_to_live(role, ckpt_dir=f"{VOL}/{tier.name}", cand_path=result["cand_path"])
    vol.commit()
    return result


@_fn("100M")
def stage3_100M(role: str = "spec_generator"): return _stage3_body("100M", role)
@_fn("500M")
def stage3_500M(role: str = "spec_generator"): return _stage3_body("500M", role)
@_fn("1B")
def stage3_1B(role: str = "spec_generator"): return _stage3_body("1B", role)
@_fn("3B")
def stage3_3B(role: str = "spec_generator"): return _stage3_body("3B", role)
@_fn("7B")
def stage3_7B(role: str = "spec_generator"): return _stage3_body("7B", role)
_STAGE3 = {"100M": stage3_100M, "500M": stage3_500M, "1B": stage3_1B,
          "3B": stage3_3B, "7B": stage3_7B}

@_fn("100M")
def stage4_100M(role: str = "spec_generator"): return _stage4_body("100M", role)
@_fn("500M")
def stage4_500M(role: str = "spec_generator"): return _stage4_body("500M", role)
@_fn("1B")
def stage4_1B(role: str = "spec_generator"): return _stage4_body("1B", role)
@_fn("3B")
def stage4_3B(role: str = "spec_generator"): return _stage4_body("3B", role)
@_fn("7B")
def stage4_7B(role: str = "spec_generator"): return _stage4_body("7B", role)
_STAGE4 = {"100M": stage4_100M, "500M": stage4_500M, "1B": stage4_1B,
          "3B": stage4_3B, "7B": stage4_7B}

@_fn("100M")
def stage5_100M(role: str = "spec_generator"): return _stage5_body("100M", role)
@_fn("500M")
def stage5_500M(role: str = "spec_generator"): return _stage5_body("500M", role)
@_fn("1B")
def stage5_1B(role: str = "spec_generator"): return _stage5_body("1B", role)
@_fn("3B")
def stage5_3B(role: str = "spec_generator"): return _stage5_body("3B", role)
@_fn("7B")
def stage5_7B(role: str = "spec_generator"): return _stage5_body("7B", role)
_STAGE5 = {"100M": stage5_100M, "500M": stage5_500M, "1B": stage5_1B,
          "3B": stage5_3B, "7B": stage5_7B}

@_fn("100M")
def stage6_100M(role: str = "spec_generator"): return _stage6_body("100M", role)
@_fn("500M")
def stage6_500M(role: str = "spec_generator"): return _stage6_body("500M", role)
@_fn("1B")
def stage6_1B(role: str = "spec_generator"): return _stage6_body("1B", role)
@_fn("3B")
def stage6_3B(role: str = "spec_generator"): return _stage6_body("3B", role)
@_fn("7B")
def stage6_7B(role: str = "spec_generator"): return _stage6_body("7B", role)
_STAGE6 = {"100M": stage6_100M, "500M": stage6_500M, "1B": stage6_1B,
          "3B": stage6_3B, "7B": stage6_7B}

@_fn("100M")
def stage7_100M(role: str = "spec_generator"): return _stage7_body("100M", role)
@_fn("500M")
def stage7_500M(role: str = "spec_generator"): return _stage7_body("500M", role)
@_fn("1B")
def stage7_1B(role: str = "spec_generator"): return _stage7_body("1B", role)
@_fn("3B")
def stage7_3B(role: str = "spec_generator"): return _stage7_body("3B", role)
@_fn("7B")
def stage7_7B(role: str = "spec_generator"): return _stage7_body("7B", role)
_STAGE7 = {"100M": stage7_100M, "500M": stage7_500M, "1B": stage7_1B,
          "3B": stage7_3B, "7B": stage7_7B}

@_fn("100M")
def stage8_100M(role: str = "spec_generator", promote: bool = False): return _stage8_body("100M", role, promote)
@_fn("500M")
def stage8_500M(role: str = "spec_generator", promote: bool = False): return _stage8_body("500M", role, promote)
@_fn("1B")
def stage8_1B(role: str = "spec_generator", promote: bool = False): return _stage8_body("1B", role, promote)
@_fn("3B")
def stage8_3B(role: str = "spec_generator", promote: bool = False): return _stage8_body("3B", role, promote)
@_fn("7B")
def stage8_7B(role: str = "spec_generator", promote: bool = False): return _stage8_body("7B", role, promote)
_STAGE8 = {"100M": stage8_100M, "500M": stage8_500M, "1B": stage8_1B,
          "3B": stage8_3B, "7B": stage8_7B}


# ---------------------------------------------------------------------------
# Deployment — long-running serving app, reads only live/*.pt
# ---------------------------------------------------------------------------

@app.function(gpu="L4", volumes={VOL: vol}, secrets=secrets,      # L4 is plenty for inference; min_containers=0 = no idle bill
              min_containers=0, timeout=3600,
              env={"QUADMESH_CKPT_ROOT": VOL, "QUADMESH_SERVE_TIER": SERVE_TIER})   # tier set via QUADMESH_SERVE_TIER at deploy time
@modal.asgi_app()
def web():
    from quadmesh.serve import app as fastapi_app
    return fastapi_app


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def train_role(tier: str = "100M", role: str = "spec_generator", stage: int = 1):
    key = tier.replace("Quadmesh-", "")
    print(_DISPATCH[key].remote(role=role, stage=stage))


@app.local_entrypoint()
def train_tier(tier: str = "100M", stage: int = 1, parallel: bool = True):
    """Trains all nine roles. They never share weights (Sec 12.1)."""
    key = tier.replace("Quadmesh-", "")
    fn = _DISPATCH[key]
    t = get_tier(key)
    print(f"{t.name}: 9 roles x {t.per_role_params} params, "
          f"{t.tokens_per_role/1e9:.2f}B tokens each, "
          f"SlimPajama slice {t.slimpajama_fraction*100:.2f}%"
          + (" + StarCoder top-up" if t.needs_starcoder else ""))
    if parallel:
        args = [((), {"role": r, "stage": stage}) for r in NINE_ROLES]
        for out in fn.starmap(args):
            print(out)
    else:
        for r in NINE_ROLES:
            print(fn.remote(role=r, stage=stage))


@app.local_entrypoint()
def train_io(tier: str = "100M", which: str = "asr"):
    print(train_io_model.remote(tier=tier, which=which))


@app.local_entrypoint()
def stage3(tier: str = "100M", role: str = "spec_generator"):
    key = tier.replace("Quadmesh-", "")
    print(_STAGE3[key].remote(role=role))


@app.local_entrypoint()
def stage4(tier: str = "100M", role: str = "spec_generator"):
    key = tier.replace("Quadmesh-", "")
    print(_STAGE4[key].remote(role=role))


@app.local_entrypoint()
def stage5(tier: str = "100M", role: str = "spec_generator"):
    key = tier.replace("Quadmesh-", "")
    print(_STAGE5[key].remote(role=role))


@app.local_entrypoint()
def stage6(tier: str = "100M", role: str = "spec_generator"):
    key = tier.replace("Quadmesh-", "")
    print(_STAGE6[key].remote(role=role))


@app.local_entrypoint()
def stage7(tier: str = "100M", role: str = "spec_generator"):
    key = tier.replace("Quadmesh-", "")
    print(_STAGE7[key].remote(role=role))


@app.local_entrypoint()
def stage8(tier: str = "100M", role: str = "spec_generator", promote: bool = False):
    key = tier.replace("Quadmesh-", "")
    print(_STAGE8[key].remote(role=role, promote=promote))


@app.local_entrypoint()
def train_all(tier: str = "100M", skip_io: bool = False):
    """One command to train ALL roles through ALL stages, then deploy.

    Handles the full pipeline per role:
      Text roles:   stage1 → stage2 → stage3 → stage4* → stage5 → stage6 → stage7 → stage8+promote
      Vision roles: stage1 → stage8+promote
      IO models:    asr + tts (unless --skip-io)

    * stage4 only for preference-tuned roles (task_planner, spec_generator, arbitration)
    """
    from quadmesh.config import (
        NINE_ROLES, VISION_ROLES, PREFERENCE_TUNED_ROLES, get_tier,
    )

    key = tier.replace("Quadmesh-", "")
    t = get_tier(key)
    train_fn = _DISPATCH[key]
    s3 = _STAGE3[key]
    s4 = _STAGE4[key]
    s5 = _STAGE5[key]
    s6 = _STAGE6[key]
    s7 = _STAGE7[key]
    s8 = _STAGE8[key]

    vision_set = set(VISION_ROLES)
    total_roles = len(NINE_ROLES)

    print(f"\n{'='*70}")
    print(f"  QUADMESH FULL PIPELINE — {t.name}")
    print(f"  {total_roles} roles, warmup={t.warmup_steps}, max_steps={t.max_steps}")
    print(f"{'='*70}\n")

    for i, role in enumerate(NINE_ROLES, 1):
        is_vision = role in vision_set
        has_stage4 = role in PREFERENCE_TUNED_ROLES
        tag = f"[{i}/{total_roles}] {role}"

        print(f"\n{'─'*70}")
        print(f"  {tag} — {'vision' if is_vision else 'text'}"
              f"{' + preference-tuned' if has_stage4 else ''}")
        print(f"{'─'*70}")

        # Skip roles that already have a live checkpoint
        live_check = _check_ckpt_remote.remote(tier=key, role=role)
        if f"live/{role}.pt" in live_check:
            print(f"  {tag} already has a live checkpoint — skipping")
            continue

        # Stage 1: pretraining
        print(f"  {tag} ► Stage 1 (pretrain)...")
        print(train_fn.remote(role=role, stage=1))
        ckpt_info = _check_ckpt_remote.remote(tier=key, role=role, stage="1")
        print(ckpt_info)

        if is_vision:
            # Vision roles: stage1 → promote
            print(f"  {tag} ► Stage 8 (promote)...")
            print(s8.remote(role=role, promote=True))
            print(f"  ✓ {tag} DONE (vision shortcut)")
            continue

        # Stage 2: SFT
        print(f"  {tag} ► Stage 2 (SFT)...")
        print(train_fn.remote(role=role, stage=2))
        ckpt_info = _check_ckpt_remote.remote(tier=key, role=role, stage="2")
        print(ckpt_info)

        # Stage 3: self-gen
        print(f"  {tag} ► Stage 3 (self-gen)...")
        print(s3.remote(role=role))

        # Stage 4: preference tuning (only for select roles)
        if has_stage4:
            print(f"  {tag} ► Stage 4 (preference)...")
            print(s4.remote(role=role))

        # Stage 5: calibration
        print(f"  {tag} ► Stage 5 (calibration)...")
        print(s5.remote(role=role))

        # Stage 6: red-team
        print(f"  {tag} ► Stage 6 (red-team)...")
        print(s6.remote(role=role))

        # Stage 7: eval
        print(f"  {tag} ► Stage 7 (eval)...")
        print(s7.remote(role=role))

        # Stage 8: shadow + promote
        print(f"  {tag} ► Stage 8 (shadow + promote)...")
        print(s8.remote(role=role, promote=True))

        # Verify final checkpoint
        ckpt_info = _check_ckpt_remote.remote(tier=key, role=role)
        print(ckpt_info)
        print(f"  ✓ {tag} DONE")

    # IO models
    if not skip_io:
        print(f"\n{'─'*70}")
        print(f"  I/O Models (ASR + TTS)")
        print(f"{'─'*70}")
        asr_check = _check_ckpt_remote.remote(tier=key, role="asr")
        if "io/asr.pt" in asr_check:
            print("  ✓ ASR already trained — skipping")
        else:
            print("  ► Training ASR...")
            print(train_io_model.remote(tier=key, which="asr"))
        tts_check = _check_ckpt_remote.remote(tier=key, role="tts")
        if "io/tts.pt" in tts_check:
            print("  ✓ TTS already trained — skipping")
        else:
            print("  ► Training TTS...")
            print(train_io_model.remote(tier=key, which="tts"))

    # Final verification
    print(f"\n{'='*70}")
    print(f"  FINAL CHECKPOINT VERIFICATION")
    print(f"{'='*70}")
    print(_check_ckpt_remote.remote(tier=key))
    print(f"\n✓ ALL TRAINING COMPLETE for {t.name}")
    print(f"  Next: modal deploy modal_app.py")


@app.local_entrypoint()
def train_full_role(tier: str = "100M", role: str = "spec_generator"):
    """Run all stages (1 through 8 + promote) for a single specific role in one command."""
    from quadmesh.config import VISION_ROLES, PREFERENCE_TUNED_ROLES, get_tier

    key = tier.replace("Quadmesh-", "")
    t = get_tier(key)
    train_fn = _DISPATCH[key]
    s3 = _STAGE3[key]
    s4 = _STAGE4[key]
    s5 = _STAGE5[key]
    s6 = _STAGE6[key]
    s7 = _STAGE7[key]
    s8 = _STAGE8[key]

    is_vision = role in set(VISION_ROLES)
    has_stage4 = role in PREFERENCE_TUNED_ROLES

    print(f"\n{'='*70}")
    print(f"  ALL STAGES FOR [{role}] — {t.name}")
    print(f"{'='*70}\n")

    print(f"  ► Stage 1 (pretrain)...")
    print(train_fn.remote(role=role, stage=1))
    print(_check_ckpt_remote.remote(tier=key, role=role, stage="1"))

    if is_vision:
        print(f"  ► Stage 8 (promote)...")
        print(s8.remote(role=role, promote=True))
        print(f"  ✓ [{role}] DONE (vision)")
        return

    print(f"  ► Stage 2 (SFT)...")
    print(train_fn.remote(role=role, stage=2))
    print(_check_ckpt_remote.remote(tier=key, role=role, stage="2"))

    print(f"  ► Stage 3 (self-gen)...")
    print(s3.remote(role=role))

    if has_stage4:
        print(f"  ► Stage 4 (preference)...")
        print(s4.remote(role=role))

    print(f"  ► Stage 5 (calibration)...")
    print(s5.remote(role=role))

    print(f"  ► Stage 6 (red-team)...")
    print(s6.remote(role=role))

    print(f"  ► Stage 7 (eval)...")
    print(s7.remote(role=role))

    print(f"  ► Stage 8 (shadow + promote)...")
    print(s8.remote(role=role, promote=True))

    print(_check_ckpt_remote.remote(tier=key, role=role))
    print(f"\n✓ ALL STAGES COMPLETE for [{role}] ({t.name})")


@app.local_entrypoint()
def plan(tier: str = "100M"):
    """Dry-run: prints the full budget for a tier without spending a GPU-second."""
    t = get_tier(tier)
    print(f"{t.name}")
    print(f"  per-role params   {t.per_role_params}  "
          f"(d_model={t.shape.d_model}, layers={t.shape.n_layers}, heads={t.shape.n_heads})")
    print(f"  tokens per role   {t.tokens_per_role/1e9:.2f}B")
    print(f"  tokens per step   {t.tokens_per_step:,}")
    print(f"  max steps         {t.max_steps:,}")
    print(f"  lr                {t.lr} -> {t.min_lr}, warmup {t.warmup_steps}")
    print(f"  batch             mb={t.micro_batch_size} x accum={t.grad_accum_steps} "
          f"x seq={t.seq_len}")
    print(f"  grad ckpt={t.gradient_checkpointing}  lora={t.use_lora} r={t.lora_rank}")
    print(f"  gpu               {t.modal_gpu_spec}  timeout {t.timeout_hours}h")
    print(f"  slimpajama slice  {t.slimpajama_fraction*100:.2f}%"
          + ("  + StarCoder (CrystalCoder mix)" if t.needs_starcoder else ""))
    print(f"  total run         9 roles x {t.max_steps:,} steps")


@app.local_entrypoint()
def check_ckpt(tier: str = "100M", role: str = "", stage: str = ""):
    """Verify checkpoints exist on the volume after a training stage."""
    result = _check_ckpt_remote.remote(tier=tier, role=role, stage=stage)
    print(result)


@app.function(timeout=120, volumes={VOL: vol})
def _check_ckpt_remote(tier: str = "100M", role: str = "", stage: str = ""):
    import os
    t = get_tier(tier)
    base = f"{VOL}/{t.name}"
    if stage:
        base = f"{base}/stage{stage}"
    lines = [f"=== Checkpoints for {t.name} ==="]
    if not os.path.exists(base):
        return f"ERROR: no checkpoint directory found at {base}"
    for root, dirs, files in os.walk(base):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            rel = os.path.relpath(fpath, f"{VOL}/{t.name}")
            if role and role not in f:
                continue
            lines.append(f"  {rel:50s}  {size_mb:8.1f} MB")
    if len(lines) == 1:
        lines.append("  (no checkpoint files found)")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Vision-Language-Action model: trained on REAL data (GroundCUA + your recorded demonstrations)
# ---------------------------------------------------------------------------

@app.function(gpu="H100", timeout=24 * 3600, volumes={VOL: vol}, secrets=secrets)
def train_vla_fn(tier: str = "100M", steps: int = 20000):
    from quadmesh.config import get_tier
    from quadmesh.train.vla_train import train_vla
    vol.reload()
    t = get_tier(tier)
    out = train_vla(t, ckpt_dir=f"{VOL}/{t.name}/vla", tokenizer_path=f"{VOL}/tokenizer/quadmesh-bpe.json",
                    steps=steps, demo_dir=f"{VOL}/demos")
    vol.commit()
    return out


@app.local_entrypoint()
def train_vla(tier: str = "100M", steps: int = 20000):
    """modal run modal_app.py::train_vla --tier 500M --steps 20000"""
    print(train_vla_fn.remote(tier=tier.replace("Quadmesh-", ""), steps=steps))


@app.local_entrypoint()
def roles(tier: str = "500M"):
    """Print the size of every role at a tier (per-role split, not equal)."""
    from quadmesh.role_shapes import describe
    key = tier.replace("Quadmesh-", "")
    print(describe({key: get_tier(key)}))


# ---------------------------------------------------------------------------
# SHARED GENERATIVE TEXT MODEL (v2): one trunk, eight roles, all by generating text.
# Order:  build_tokenizer -> shared_review_queue -> shared_pretrain -> shared_sft -> shared_eval
# Every long job saves atomically and resumes, so the same command can simply be repeated after a preemption.
# Run the two long ones with `modal run --detach` (they re-spawn their own 23h slices).
# ---------------------------------------------------------------------------
import os as _os

_SHARED_GPU = _os.environ.get("QM_GPU", "H100")           # e.g. QM_GPU=A100-80GB


def _ck(tier: str) -> str:
    return f"{VOL}/Quadmesh-{tier.replace('Quadmesh-', '')}"


@app.function(timeout=1800, volumes={VOL: vol}, cpu=4.0)
def review_queue_remote(n_each: int = 1500):
    from quadmesh.pipeline.datasets.role_gen_data import write_review_queue
    p = write_review_queue(f"{VOL}/data/review_queue.jsonl", n_each)
    vol.commit()
    return p


@app.function(gpu=_SHARED_GPU, timeout=24 * 3600, volumes={VOL: vol}, secrets=secrets, cpu=16.0, memory=65536)
def shared_train_remote(phase: str, tier: str, budget: str = "4x", tokens: int = 0, confirm_full: bool = False,
                        hours: float = 23.0, workers: int = 12, lr_scale: float = 0.0,
                        mb: int = 0, accum: int = 0, grad_ckpt: int = -1):
    """One <=24h slice of a run. Returns done=False if more slices are needed (the local entrypoint chains them)."""
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.train.shared_text_train import train_shared
    tok = get_or_train(f"{VOL}/tokenizer/quadmesh-bpe.json")
    ck = _ck(tier)
    pre = f"{ck}/shared_pretrain.pt"
    reviewed = f"{VOL}/data/review_queue.jsonl"
    res = train_shared(
        phase, tier.replace("Quadmesh-", ""), tok, ck, tokens=tokens or None, budget=budget, confirm_full=confirm_full,
        init=(pre if phase == "sft" and _os.path.exists(pre) else None), device="cuda", workers=workers,
        lr_scale=lr_scale or None, reviewed_path=(reviewed if _os.path.exists(reviewed) else None),
        mb=mb or None, accum=accum or None, grad_ckpt=(None if grad_ckpt < 0 else bool(grad_ckpt)),
        on_save=vol.commit, max_seconds=hours * 3600)
    if phase == "sft" and res["done"]:
        _os.makedirs(f"{ck}/live", exist_ok=True)
        import shutil
        shutil.copy(res["path"], f"{ck}/live/shared_text.pt")     # what the agent and the evaluator load
    vol.commit()
    if not res["done"]:            # Modal caps a function at 24h: hand the rest to a fresh container, no client needed
        shared_train_remote.spawn(phase, tier, budget=budget, tokens=tokens, confirm_full=confirm_full,
                                 hours=hours, workers=workers, lr_scale=lr_scale, mb=mb, accum=accum, grad_ckpt=grad_ckpt)
    return res


@app.function(gpu="L4", timeout=6 * 3600, volumes={VOL: vol}, secrets=secrets, cpu=4.0)
def shared_eval_remote(tier: str, n: int = 100):
    import json, time
    from quadmesh.agent.shared_runtime import get_runtime
    from quadmesh.pipeline.shared_eval import evaluate
    rt = get_runtime(VOL, tier.replace("Quadmesh-", ""), "cuda")
    assert rt is not None, f"no {_ck(tier)}/live/shared_text.pt yet - run shared_sft first"
    rep = evaluate(rt, n, reviewed_path=f"{VOL}/data/review_queue.jsonl")
    out = f"{_ck(tier)}/eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(rep, open(out, "w"), indent=1)
    vol.commit()
    return rep


def _spawn(phase, tier, attach: bool = False, **kw):
    """Starts the run and returns. Use `modal run --detach` or pass attach=False so the app - and the self-spawned continuation slices -
    keep running after your terminal closes. Progress: `modal app logs <app-id>`; metrics_<phase>.jsonl on the volume."""
    if attach:
        print(f"Running {phase} for {tier} attached to terminal with live streaming output...")
        return shared_train_remote.remote(phase, tier, **kw)
    call = shared_train_remote.spawn(phase, tier, **kw)
    print(f"started {phase} for {tier}: call id {call.object_id}. It runs detached and re-spawns itself every <=23h until done.")
    return call


@app.local_entrypoint()
def shared_review_queue(n_each: int = 1500):
    """Writes candidate risk/abuse rows to /vol/data/review_queue.jsonl. Download it, review it (README), upload it back."""
    print(f"[review_queue] Generating {n_each} safety review candidates per category on Modal...")
    res = review_queue_remote.remote(n_each)
    print(f"[review_queue] Successfully created review queue at: {res}")
    return res


@app.local_entrypoint()
def shared_pretrain(tier: str = "3B", budget: str = "4x", tokens: int = 0, confirm_full: bool = False,
                     mb: int = 0, accum: int = 0, grad_ckpt: int = -1, workers: int = 12, attach: bool = False):
    """budget: chinchilla (20 tokens/param) | 4x (80 tokens/param, default) | full (all 627B SlimPajama tokens).
    mb/accum: override the tier's default micro-batch/grad-accum split (same effective batch if mb*accum matches).
    grad_ckpt: 1 to force gradient checkpointing on, 0 to force it off, -1 (default) to use the tier's setting.
    workers: DataLoader workers pulling the streamed dataset.
    attach: True to run directly attached in console to see all live step metrics (loss, lr, tok/s)."""
    _spawn("pretrain", tier, budget=budget, tokens=tokens, confirm_full=confirm_full,
           mb=mb, accum=accum, grad_ckpt=grad_ckpt, workers=workers, attach=attach)


@app.local_entrypoint()
def shared_sft(tier: str = "3B", tokens: int = 2_000_000_000, mb: int = 0, accum: int = 0,
                grad_ckpt: int = -1, workers: int = 12, attach: bool = False):
    """Multi-role SFT on generated, verifier-grounded data. Default 2B tokens; watch held-out loss per role in
    metrics_sft.jsonl and the shared_eval numbers - if they are still rising, run again with more tokens.
    Writes /vol/Quadmesh-<tier>/live/shared_text.pt.
    attach: True to run directly attached in console to see all live step metrics (loss, lr, tok/s)."""
    _spawn("sft", tier, budget="4x", tokens=tokens, mb=mb, accum=accum, grad_ckpt=grad_ckpt, workers=workers, attach=attach)


@app.local_entrypoint()
def shared_eval(tier: str = "3B", n: int = 100):
    import json
    print(json.dumps(shared_eval_remote.remote(tier, n), indent=1))

