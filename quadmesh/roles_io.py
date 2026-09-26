"""
Prompt and answer formats for the eight text roles of the shared model. ONE place, imported by training, evaluation
and the running agent, so what the model was trained to read is byte-for-byte what the agent sends it.

    <|role_X|>{prompt}\\n            <- a role tag (one special token) conditions the shared trunk
    {answer}<|eos|>                 <- the model GENERATES this; loss is on the answer only

Every role answers in text. Nothing in the model is a softmax over a fixed label set and nothing is a parser:

    chat                 free text, or  CALL {"tool": ...}
    task_planner         a JSON list of feature operations
    spec_generator       SPEC {"kind": ...}
    spec_auditor         VERDICT: PASS   |   VERDICT: FAIL\\nREASONS: ...
    arbitration          CHOOSE: A|B|EITHER|NONE\\nWHY: ...
    cad_skill_dispatch   CALL {"tool": ...}  |  CLARIFY: ...  |  NONE
    risk_tier_classifier TIER: <tier>\\nWHY: ...
    abuse_pattern        ABUSE: yes|no\\nCATEGORY: ...\\nWHY: ...

The parse_* functions below only read the model's OUTPUT back into Python objects (like json.loads on a JSON
answer). They never interpret the user's request - that is the model's job.
"""
from __future__ import annotations
import json
import re
from typing import Optional

from quadmesh.data.tokenizer import ROLE_NAMES, ROLE_TAGS

TEXT_ROLES = ROLE_NAMES
RISK_TIERS = ("cosmetic", "structural_non_critical", "structural_critical", "flight_critical")
ABUSE_CATEGORIES = ("none", "safety_bypass", "weapon", "prompt_injection", "system_damage")
DISPATCH_TOOLS = ("make_shape", "build_part", "build_assembly", "photo_to_3d", "move", "rotate", "show_360", "export_step")


def role_prompt(role: str, prompt: str) -> str:
    """The exact text fed to the model for `role`. The answer is generated after the trailing newline."""
    if role not in ROLE_TAGS:
        raise ValueError(f"unknown role {role!r}; known: {list(ROLE_TAGS)}")
    return ROLE_TAGS[role] + prompt.strip() + "\n"


def clean_answer(text: str) -> str:
    return text.strip()


def encode_example(tok, role: str, prompt: str, answer: str, max_len: Optional[int] = None):
    """(ids, prompt_len). Loss is taken on ids[prompt_len:] only. Returns None if it does not fit max_len -
    a truncated answer would teach the model to stop mid-sentence, so over-long examples are dropped, not cut."""
    p = tok.encode(role_prompt(role, prompt))
    a = tok.encode(clean_answer(answer)) + [tok.eos_id]
    if max_len is not None and len(p) + len(a) > max_len:
        return None
    return p + a, len(p)


# ------------------------------------------------------------------------------------------------ output readers
def parse_plan(text: str) -> Optional[list]:
    from quadmesh.pipeline.plan_checker import extract_plan
    return extract_plan(text)


def parse_spec(text: str) -> Optional[dict]:
    from quadmesh.spec_gen_data import parse_spec as _p
    return _p(text if "SPEC" in text else "SPEC " + text)


def parse_call(text: str) -> Optional[dict]:
    from quadmesh.chat_data import parse_call as _p
    return _p(text)


def parse_verdict(text: str) -> Optional[dict]:
    m = re.match(r"\s*VERDICT:\s*(PASS|FAIL)\b", text)
    if not m:
        return None
    reasons = []
    r = re.search(r"REASONS:\s*(.+)", text, re.S)
    if r:
        reasons = [x.strip() for x in r.group(1).split(";") if x.strip()]
    return {"verdict": m.group(1), "reasons": reasons}


def parse_choice(text: str) -> Optional[dict]:
    m = re.match(r"\s*CHOOSE:\s*(A|B|EITHER|NONE)\b", text)
    if not m:
        return None
    w = re.search(r"WHY:\s*(.+)", text, re.S)
    return {"choice": m.group(1), "why": w.group(1).strip() if w else ""}


def parse_tier(text: str) -> Optional[dict]:
    m = re.match(r"\s*TIER:\s*([a-z_]+)", text)
    if not m or m.group(1) not in RISK_TIERS:
        return None
    w = re.search(r"WHY:\s*(.+)", text, re.S)
    return {"tier": m.group(1), "tier_idx": RISK_TIERS.index(m.group(1)), "why": w.group(1).strip() if w else ""}


def parse_abuse(text: str) -> Optional[dict]:
    m = re.match(r"\s*ABUSE:\s*(yes|no)\b", text)
    if not m:
        return None
    c = re.search(r"CATEGORY:\s*([a-z_]+)", text)
    cat = c.group(1) if c and c.group(1) in ABUSE_CATEGORIES else "none"
    w = re.search(r"WHY:\s*(.+)", text, re.S)
    return {"abuse": m.group(1) == "yes", "category": cat, "why": w.group(1).strip() if w else ""}


def parse_dispatch(text: str) -> dict:
    """{'kind': 'call', 'call': {...}} | {'kind': 'clarify', 'question': ...} | {'kind': 'none'} | {'kind': 'invalid'}"""
    t = text.strip()
    if t.startswith("CALL"):
        c = parse_call(t)
        return {"kind": "call", "call": c} if c and c.get("tool") in DISPATCH_TOOLS else {"kind": "invalid"}
    if t.startswith("CLARIFY"):
        return {"kind": "clarify", "question": t.split(":", 1)[-1].strip()}
    if t.startswith("NONE"):
        return {"kind": "none"}
    return {"kind": "invalid"}
