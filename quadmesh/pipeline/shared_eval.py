"""
Real evaluation of the shared generative model: does what it WRITES work?

Every number here is computed from generated text, on prompts the model never trained on (role_gen_data's
held-out split, disjoint by hash), and scored against the deterministic plan checker or the reference label -
not next-token loss.

    python -m quadmesh.pipeline.shared_eval --root /vol/checkpoints --tier 500M --n 200

Per role
  task_planner        plan parses; plan passes plan_checker (the hard gate) for ITS request; exact match to reference
  spec_generator      SPEC parses; kind correct; exact match; freeform plans pass the checker
  spec_auditor        VERDICT agrees with the checker (accuracy, FAIL recall, PASS precision)
  arbitration         CHOOSE agrees with the checker
  cad_skill_dispatch  right kind (call/none/clarify); right tool; exact call
  chat                tool requests: exact call; non-tool requests: no spurious call
Then the 20 fixed benchmark prompts, scored end to end through the spec route and the direct planner route.
Risk / abuse are evaluated only if the checkpoint records human-reviewed training rows for them.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from typing import Optional

from quadmesh import roles_io as IO
from quadmesh.pipeline.datasets import role_gen_data as RG
from quadmesh.pipeline.plan_checker import check_plan


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _req_of(prompt: str) -> str:
    if prompt.startswith("REPAIR:"):
        return prompt[len("REPAIR: "):].split(" PLAN ")[0]
    return prompt


def _score(role: str, prompt: str, ref: str, out: str) -> dict:
    if role == "task_planner":
        plan, refp = IO.parse_plan(out), IO.parse_plan(ref)
        ok = plan is not None
        return {"parses": ok, "checker_pass": bool(ok and not check_plan(plan, _req_of(prompt))),
                "exact": bool(ok and refp is not None and _same(plan, refp))}
    if role == "spec_generator":
        s, r = IO.parse_spec(out), IO.parse_spec(ref)
        ok = s is not None
        res = {"parses": ok, "kind": bool(ok and r and s.get("kind") == r.get("kind")), "exact": bool(ok and r and _same(s, r))}
        if ok and s.get("kind") == "freeform_ops":
            res["checker_pass"] = not check_plan(s["ops"], prompt)
        return res
    if role == "spec_auditor":
        v, r = IO.parse_verdict(out), IO.parse_verdict(ref)
        return {"parses": v is not None, "verdict": bool(v and v["verdict"] == r["verdict"]),
                "ref_fail": r["verdict"] == "FAIL", "pred_fail": bool(v and v["verdict"] == "FAIL")}
    if role == "arbitration":
        c, r = IO.parse_choice(out), IO.parse_choice(ref)
        return {"parses": c is not None, "choice": bool(c and c["choice"] == r["choice"])}
    if role in ("cad_skill_dispatch", "chat"):
        r = IO.parse_dispatch(ref) if role == "cad_skill_dispatch" else {"kind": "call", "call": IO.parse_call(ref)} if IO.parse_call(ref) else {"kind": "none"}
        d = IO.parse_dispatch(out) if role == "cad_skill_dispatch" else ({"kind": "call", "call": IO.parse_call(out)} if IO.parse_call(out) else {"kind": "none"})
        res = {"kind": d["kind"] == r["kind"]}
        if r["kind"] == "call":
            res["tool"] = bool(d["kind"] == "call" and d["call"].get("tool") == r["call"].get("tool"))
            res["exact_call"] = bool(d["kind"] == "call" and _same(d["call"], r["call"]))
        else:
            res["no_spurious_call"] = d["kind"] != "call"
        return res
    if role == "risk_tier_classifier":
        t, r = IO.parse_tier(out), IO.parse_tier(ref)
        return {"parses": t is not None, "tier": bool(t and t["tier"] == r["tier"]),
                "under_tier": bool(t and t["tier_idx"] < r["tier_idx"])}          # the dangerous error
    if role == "abuse_pattern":
        a, r = IO.parse_abuse(out), IO.parse_abuse(ref)
        return {"parses": a is not None, "abuse": bool(a and a["abuse"] == r["abuse"]),
                "missed_abuse": bool(r["abuse"] and not (a and a["abuse"]))}
    raise ValueError(role)


def _max_new(role):
    return {"task_planner": 700, "spec_generator": 300, "spec_auditor": 120, "arbitration": 120,
            "cad_skill_dispatch": 200, "chat": 300, "risk_tier_classifier": 80, "abuse_pattern": 80}[role]


def evaluate(rt, n_per_role: int = 100, seed: int = 777, reviewed_path: Optional[str] = None, verbose: bool = True) -> dict:
    """rt: agent.shared_runtime.SharedRuntime. Returns {role: {metric: rate}} plus the benchmark."""
    roles = list(RG.PROCEDURAL_ROLES) + [r for r in RG.REVIEWED_ROLES if r in rt.safety_trained]
    random.seed(seed)
    report = {}
    for role in roles:
        rows = []
        if role in RG.REVIEWED_ROLES:
            pool = RG.load_reviewed(reviewed_path, role, "eval")
            rows = random.sample(pool, min(n_per_role, len(pool)))
        else:
            while len(rows) < n_per_role:
                r = RG.make_example(role, "eval")
                if r:
                    rows.append(r)
        acc, t0 = {}, time.time()
        for p, ref in rows:
            out = rt.ask(role, p, max_new=_max_new(role))
            for k, v in _score(role, p, ref, out).items():
                acc.setdefault(k, []).append(v)
        rep = {k: round(sum(v) / len(v), 3) for k, v in acc.items()}
        if role == "spec_auditor":                 # turn the raw flags into recall / precision
            rf, pf = acc["ref_fail"], acc["pred_fail"]
            tp = sum(1 for a, b in zip(rf, pf) if a and b)
            rep = {"parses": rep["parses"], "verdict_acc": rep["verdict"],
                   "fail_recall": round(tp / max(sum(rf), 1), 3), "fail_precision": round(tp / max(sum(pf), 1), 3)}
        rep["n"], rep["sec_per_example"] = len(rows), round((time.time() - t0) / max(len(rows), 1), 2)
        report[role] = rep
        if verbose:
            print(f"[eval] {role:22s} {rep}", flush=True)
    report["benchmark_20"] = benchmark(rt, verbose)
    return report


def benchmark(rt, verbose: bool = True) -> dict:
    from quadmesh.pipeline.checkpoint_benchmark import BENCHMARK_PROMPTS
    spec_ok = plan_ok = 0
    rows = []
    for it in BENCHMARK_PROMPTS:
        p, expect = it["prompt"], it["expect"]
        spec = rt.spec(p)
        if spec is None:
            got = "invalid"
        elif spec["kind"] == "unsupported":
            got = "unsupported"
        elif spec["kind"] in ("hexacopter", "robot_arm"):
            got = "valid"
        else:
            got = "valid" if not check_plan(spec["ops"], p) else "invalid"
        plan = rt.plan(p)
        direct = "valid" if plan and not check_plan(plan, p) else "invalid"
        spec_ok += got == expect
        plan_ok += (direct == "valid") == (expect == "valid")
        rows.append({"id": it["id"], "expect": expect, "spec_route": got, "direct_planner": direct})
        if verbose:
            print(f"[bench] {it['id']} expect={expect:11s} spec_route={got:11s} direct_planner={direct}", flush=True)
    return {"spec_route_pass": f"{spec_ok}/20", "direct_planner_agrees": f"{plan_ok}/20", "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tier", default="500M")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--reviewed", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from quadmesh.agent.shared_runtime import get_runtime
    rt = get_runtime(a.root, a.tier)
    if rt is None:
        raise SystemExit(f"no shared_text.pt under {a.root}/Quadmesh-{a.tier}/live/")
    rep = evaluate(rt, a.n, reviewed_path=a.reviewed)
    out = a.out or f"{a.root}/Quadmesh-{a.tier}/eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(rep, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
