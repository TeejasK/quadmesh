"""
Quadmesh tier + role configuration.

This is the ONLY file you edit to move between Quadmesh-100M ... Quadmesh-7B.
Training code (quadmesh/train/*) is tier-agnostic and reads from here.

Sec 14: every tier runs all 9 roles as 9 separate models, scaled together.
Per-role budget = total_params / 9. Token budget = Chinchilla ~20x per role.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ----------------------------------------------------------------------------
# The nine roles (Sec 14). Never merged, at any tier.
# ----------------------------------------------------------------------------

TEXT_ROLES = [
    "task_planner",
    "spec_generator",
    "spec_auditor",
    "risk_tier_classifier",
    "abuse_pattern",
    "arbitration",
    "cad_skill_dispatch",
]

VISION_ROLES = [
    "ui_grounding",
    "screenshot_diff",
]

IO_ROLES = ["asr", "tts"]          # Sec 17, trained from scratch, own datasets

ALL_ROLES = TEXT_ROLES + VISION_ROLES
NINE_ROLES = ALL_ROLES             # exactly 9


# ----------------------------------------------------------------------------
# Model shape per role-parameter budget
# ----------------------------------------------------------------------------

@dataclass
class ModelShape:
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int = 32000
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0

    @property
    def approx_params(self) -> int:
        # 12 * L * d^2 (attn+ffn) + embeddings (tied)
        return 12 * self.n_layers * self.d_model ** 2 + self.vocab_size * self.d_model


@dataclass
class TierConfig:
    name: str
    total_params: str
    per_role_params: str
    tokens_per_role: int          # Chinchilla ~20x per-role params
    slimpajama_fraction: float    # fraction of the 627B-token stream to consume
    needs_starcoder: bool
    multi_gpu: bool
    shape: ModelShape
    vision_shape: ModelShape
    # --- optimizer / schedule ---
    micro_batch_size: int
    grad_accum_steps: int
    seq_len: int
    lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # --- memory / precision ---
    precision: str = "bf16"
    gradient_checkpointing: bool = False
    use_lora: bool = False
    lora_rank: int = 0
    # --- modal ---
    gpu: str = "H100"
    n_gpu: int = 1
    timeout_hours: int = 8
    # --- asr/tts (Sec 17) ---
    asr_hours: int = 200
    tts_hours: float = 24.0

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps * self.seq_len

    @property
    def max_steps(self) -> int:
        return self.tokens_per_role // self.tokens_per_step

    @property
    def modal_gpu_spec(self) -> str:
        return self.gpu if self.n_gpu == 1 else f"{self.gpu}:{self.n_gpu}"


# ----------------------------------------------------------------------------
# TIERS  —  change QUADMESH_TIER env var, nothing else.
# ----------------------------------------------------------------------------

TIERS: Dict[str, TierConfig] = {

    # ---------------------------------------------------------------- 100M --
    "100M": TierConfig(
        name="Quadmesh-100M",
        total_params="100M",
        per_role_params="~11M",
        tokens_per_role=220_000_000,
        slimpajama_fraction=0.0004,        # ~0.04%,  ~0.3 GB
        needs_starcoder=False,
        multi_gpu=False,
        shape=ModelShape(d_model=384, n_layers=6, n_heads=6, d_ff=1536, max_seq_len=1024),
        vision_shape=ModelShape(d_model=384, n_layers=6, n_heads=6, d_ff=1536, max_seq_len=1024),
        micro_batch_size=32, grad_accum_steps=4, seq_len=1024,
        lr=6e-4, min_lr=6e-5, warmup_steps=50,
        timeout_hours=3, asr_hours=100, tts_hours=24.0,
    ),

    # ---------------------------------------------------------------- 500M --
    "500M": TierConfig(
        name="Quadmesh-500M",
        total_params="500M",
        per_role_params="~56M",
        tokens_per_role=1_100_000_000,
        slimpajama_fraction=0.002,         # ~0.2%,   ~1.6 GB
        needs_starcoder=False,
        multi_gpu=False,
        shape=ModelShape(d_model=768, n_layers=8, n_heads=12, d_ff=3072, max_seq_len=2048),
        vision_shape=ModelShape(d_model=768, n_layers=8, n_heads=12, d_ff=3072, max_seq_len=2048),
        micro_batch_size=24, grad_accum_steps=8, seq_len=2048,
        lr=4e-4, min_lr=4e-5, warmup_steps=200,
        timeout_hours=8, asr_hours=200, tts_hours=24.0,
    ),

    # ------------------------------------------------------------------ 1B --
    "1B": TierConfig(
        name="Quadmesh-1B",
        total_params="1B",
        per_role_params="~111M",
        tokens_per_role=2_200_000_000,
        slimpajama_fraction=0.0035,        # ~0.35%,  ~3 GB
        needs_starcoder=False,
        multi_gpu=False,
        shape=ModelShape(d_model=768, n_layers=16, n_heads=12, d_ff=3072, max_seq_len=2048),
        vision_shape=ModelShape(d_model=768, n_layers=16, n_heads=12, d_ff=3072, max_seq_len=2048),
        micro_batch_size=16, grad_accum_steps=16, seq_len=2048,
        lr=3e-4, min_lr=3e-5, warmup_steps=500,
        timeout_hours=12, asr_hours=400, tts_hours=24.0,
    ),

    # ------------------------------------------------------------------ 3B --
    "3B": TierConfig(
        name="Quadmesh-3B",
        total_params="3B",
        per_role_params="~333M",
        tokens_per_role=6_700_000_000,
        slimpajama_fraction=0.01,          # ~1%,     ~10 GB
        needs_starcoder=False,
        multi_gpu=False,
        shape=ModelShape(d_model=1024, n_layers=26, n_heads=16, d_ff=4096, max_seq_len=2048),
        vision_shape=ModelShape(d_model=1024, n_layers=26, n_heads=16, d_ff=4096, max_seq_len=2048),
        micro_batch_size=16, grad_accum_steps=16, seq_len=2048,      # was mb=8/accum=32 — same effective
                                                                       # batch (256), fewer accum steps, better
                                                                       # H100 utilization given the memory headroom
                                                                       # at this tier (~36GB static vs 80GB)
        lr=2.5e-4, min_lr=2.5e-5, warmup_steps=1000,
        gradient_checkpointing=False,      # off by default: 3B has ~50GB headroom on an 80GB H100 for full FT
                                            # (see mb/accum note above); override with --grad_ckpt 1 on the CLI
                                            # if you push batch size further and hit OOM
        timeout_hours=24, asr_hours=800, tts_hours=24.0,
    ),

    # ------------------------------------------------------------------ 7B --
    "7B": TierConfig(
        name="Quadmesh-7B",
        total_params="7B",
        per_role_params="~778M",
        tokens_per_role=15_600_000_000,
        slimpajama_fraction=0.025,         # ~2.5%,   ~22 GB
        needs_starcoder=False,
        multi_gpu=False,
        shape=ModelShape(d_model=1536, n_layers=28, n_heads=24, d_ff=6144, max_seq_len=2048),
        vision_shape=ModelShape(d_model=1536, n_layers=28, n_heads=24, d_ff=6144, max_seq_len=2048),
        micro_batch_size=4, grad_accum_steps=64, seq_len=2048,
        lr=2e-4, min_lr=2e-5, warmup_steps=1000,
        gradient_checkpointing=True,
        use_lora=True, lora_rank=64,       # full FT is tight on one H100
        timeout_hours=24, asr_hours=1600, tts_hours=24.0,
    ),
}


# StarCoder top-up ratio (if needed).
STARCODER_MIX_RATIO = 0.55


def get_tier(name: str) -> TierConfig:
    key = name.replace("Quadmesh-", "").strip()
    if key not in TIERS:
        raise KeyError(f"unknown tier {name!r}; choose from {list(TIERS)}")
    return TIERS[key]


@dataclass
class RoleSpec:
    """Per-role data routing. Sec 14 'dataset mapping per role type'."""
    role: str
    modality: str                 # "text" | "vision"
    primary: str                  # dataset id
    stage2_overlay: Optional[str] = None
    disjoint_source: bool = False  # Sec 12.1 — auditor must not share the corpus


ROLE_SPECS: Dict[str, RoleSpec] = {
    "ui_grounding":    RoleSpec("ui_grounding", "vision", "ServiceNow/GroundCUA"),
    "screenshot_diff": RoleSpec("screenshot_diff", "vision", "ServiceNow/GroundCUA"),

    "task_planner":    RoleSpec("task_planner", "text", "gmongaras/SlimPajama-627B_Reupload",
                                stage2_overlay="local:planner_sft"),
    "spec_generator":  RoleSpec("spec_generator", "text", "gmongaras/SlimPajama-627B_Reupload",
                                stage2_overlay="local:spec_sft"),
    "spec_auditor":    RoleSpec("spec_auditor", "text", "gmongaras/SlimPajama-627B_Reupload",
                                disjoint_source=True),
    "risk_tier_classifier": RoleSpec("risk_tier_classifier", "text", "gmongaras/SlimPajama-627B_Reupload"),
    "abuse_pattern":   RoleSpec("abuse_pattern", "text", "gmongaras/SlimPajama-627B_Reupload"),
    "arbitration":     RoleSpec("arbitration", "text", "gmongaras/SlimPajama-627B_Reupload"),
    "cad_skill_dispatch": RoleSpec("cad_skill_dispatch", "text", "gmongaras/SlimPajama-627B_Reupload"),
}

# Roles that get Stage 4 preference tuning (Sec 15).
PREFERENCE_TUNED_ROLES = {"task_planner", "spec_generator", "arbitration"}

# Sec 12.1: the auditor is deliberately fed a different slice of the stream so
# it does not inherit the generator's conventions. Same code path, different
# shard offset + a standards/failure-report overlay if you have one.
AUDITOR_SHARD_OFFSET = 7919      # prime, keeps the two streams from realigning
