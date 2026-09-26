"""
The running side of the shared generative model: one checkpoint, eight roles, all by generating text.

    from quadmesh.agent.shared_runtime import get_runtime
    rt = get_runtime()                       # None if no shared_text.pt has been trained/downloaded yet
    plan = rt.plan("Make a 60x40x5 mm plate with 4 corner holes ...")
    call = rt.dispatch("move the arm 20 mm left")        # {'kind': 'call', 'call': {...}} - the model chose the tool
    verdict = rt.audit(request, plan)                    # the model's own opinion of a plan
    gate = rt.screen(request)                            # risk + abuse (see below)

Files:  <root>/Quadmesh-<tier>/live/shared_text.pt   and   <root>/tokenizer/quadmesh-bpe.json
Env:    QUADMESH_CKPT_ROOT, QUADMESH_TIER (default 500M), QUADMESH_DEVICE (auto|cpu|cuda)

SAFETY MODEL. The hard checks stay OUTSIDE the model and always win:
  * plan_checker.check_plan / verification.stack decide whether a plan or a mesh is acceptable - the model can
    propose and audit, it cannot overrule them.
  * screen(): the model's risk tier can only RAISE the tier (escalate_tier already takes the max with the keyword
    table), and an abuse "yes" blocks. If risk/abuse were not trained on human-reviewed rows (the checkpoint says so),
    screen() reports model_screen="untrained" and relies on the keyword table alone - it never pretends.
"""
from __future__ import annotations
import json
import os
from typing import Optional

from quadmesh import roles_io as IO

_RT: dict = {}


class RoleNotTrained(RuntimeError):
    pass


def _device(device):
    import torch
    device = device or os.environ.get("QUADMESH_DEVICE", "auto")
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


class SharedRuntime:
    def __init__(self, path: str, tokenizer_path: str, device: str = "cpu"):
        import torch
        from quadmesh.data.tokenizer import QuadmeshTokenizer
        from quadmesh.model.shared_backbone import load_shared
        self.tok = QuadmeshTokenizer.load(tokenizer_path)
        dt = torch.bfloat16 if device == "cuda" else None
        self.model, self.ck = load_shared(path, device=device, dtype=dt)
        self.device = device
        self.safety_trained = set(self.ck.get("safety_roles_trained", []))
        self.max_len = self.model.shape.max_seq_len

    @classmethod
    def from_model(cls, model, tok, device: str = "cpu", safety_trained=()):
        """Wrap an in-memory model (tests, training-time evaluation)."""
        self = cls.__new__(cls)
        self.tok, self.model, self.ck, self.device = tok, model.eval(), {}, device
        self.safety_trained, self.max_len = set(safety_trained), model.shape.max_seq_len
        return self

    # ---- the one primitive -----------------------------------------------------------------------
    def ask(self, role: str, prompt: str, max_new: int = 400, temperature: float = 0.0, top_k: int = 40) -> str:
        import torch
        ids = self.tok.encode(IO.role_prompt(role, prompt))
        if len(ids) >= self.max_len - 8:
            raise ValueError(f"prompt is {len(ids)} tokens; this model reads at most {self.max_len - 8}")
        ids_t = torch.tensor([ids], device=self.device)
        new = self.model.generate(ids_t, max_new_tokens=min(max_new, self.max_len - len(ids)), eos_id=self.tok.eos_id,
                                  temperature=temperature, top_k=top_k)
        if new and new[-1] == self.tok.eos_id:
            new = new[:-1]
        return self.tok.decode(new).strip()

    # ---- roles -----------------------------------------------------------------------------------
    def plan(self, request, prev_plan=None, feedback=None, temperature=0.0, max_new=700):
        if prev_plan is not None and feedback:
            from quadmesh.pipeline.datasets.engineering_gen import dumps
            request = f"REPAIR: {request} PLAN {dumps(prev_plan)} ERRORS {feedback} FIX"
        return IO.parse_plan(self.ask("task_planner", request, max_new=max_new, temperature=temperature))

    def spec(self, request):
        return IO.parse_spec(self.ask("spec_generator", request, max_new=160))

    def chat(self, prompt):
        return self.ask("chat", prompt, max_new=300)

    def dispatch(self, utterance, context: str = ""):
        return IO.parse_dispatch(self.ask("cad_skill_dispatch", utterance + (f"\n[{context}]" if context else ""), max_new=200))

    def audit(self, request, plan):
        from quadmesh.pipeline.datasets.engineering_gen import dumps
        return IO.parse_verdict(self.ask("spec_auditor", f"REQUEST: {request}\nPLAN: {dumps(plan)}", max_new=120))

    def arbitrate(self, request, plan_a, plan_b, audit_a="", audit_b=""):
        from quadmesh.pipeline.datasets.engineering_gen import dumps
        p = (f"REQUEST: {request}\nPLAN_A: {dumps(plan_a)}\nPLAN_B: {dumps(plan_b)}\n"
             f"AUDIT_A: {audit_a}\nAUDIT_B: {audit_b}")
        return IO.parse_choice(self.ask("arbitration", p, max_new=120))

    def risk(self, request):
        if "risk_tier_classifier" not in self.safety_trained:
            raise RoleNotTrained("risk_tier_classifier has no human-reviewed training data in this checkpoint")
        return IO.parse_tier(self.ask("risk_tier_classifier", request, max_new=80))

    def abuse(self, request):
        if "abuse_pattern" not in self.safety_trained:
            raise RoleNotTrained("abuse_pattern has no human-reviewed training data in this checkpoint")
        return IO.parse_abuse(self.ask("abuse_pattern", request, max_new=80))

    def screen(self, request) -> dict:
        """Risk + abuse gate. Keyword table always applies; the model can only make it stricter."""
        from quadmesh.verification.stack import escalate_tier
        out = {"model_screen": "untrained", "abuse": False, "category": "none", "blocked": False}
        model_idx = 0
        try:
            r = self.risk(request)
            a = self.abuse(request)
            out["model_screen"] = "ok"
            model_idx = r["tier_idx"] if r else 3            # unreadable model output -> assume the highest tier
            if a is None or a["abuse"]:
                out.update(abuse=True, category=(a or {}).get("category", "unparseable"), blocked=True)
        except RoleNotTrained:
            pass
        idx = escalate_tier(model_idx, request)
        out.update(tier_idx=idx, tier=IO.RISK_TIERS[idx], escalated_by_keywords=idx > model_idx,
                   requires_human_signoff=idx >= 2)
        return out


def get_runtime(ckpt_root: Optional[str] = None, tier: Optional[str] = None,
                device: Optional[str] = None) -> Optional[SharedRuntime]:
    root = ckpt_root or os.environ.get("QUADMESH_CKPT_ROOT")
    if not root:
        return None
    tier = (tier or os.environ.get("QUADMESH_TIER", "500M")).replace("Quadmesh-", "")
    path, tokp = f"{root}/Quadmesh-{tier}/live/shared_text.pt", f"{root}/tokenizer/quadmesh-bpe.json"
    if not (os.path.exists(path) and os.path.exists(tokp)):
        return None
    dev = _device(device)
    key = (path, dev)
    if key not in _RT:
        _RT[key] = SharedRuntime(path, tokp, dev)
    return _RT[key]
