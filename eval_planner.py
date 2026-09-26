"""
Compare model sizes on YOUR machine with real numbers instead of opinions.

    python eval_planner.py --tier 500M --n 200            # Task Planner: how often is its plan valid for a fresh prompt?
    python eval_planner.py --tier 500M --n 300 --spec     # Spec Generator: exact-match rate on fresh requests

Train the same role at two tiers, run this for both, and pick the smallest tier whose score is good enough.
"""
import argparse
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="500M")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--spec", action="store_true")
    a = ap.parse_args()
    random.seed(12345)                                       # fresh prompts the model never saw in training
    ok = 0
    if a.spec:
        from quadmesh import spec_gen_data as S
        from quadmesh.agent.planner_model import propose_spec
        for _ in range(a.n):
            ex = S.gen_example()
            got = propose_spec(ex["prompt"], a.tier, a.device, a.ckpt_root)
            ok += (got == ex["spec"])
        print(f"Spec Generator {a.tier}: exact match {ok}/{a.n} = {100*ok/a.n:.1f}%")
    else:
        from quadmesh.pipeline.datasets import engineering_gen as g
        from quadmesh.pipeline.plan_checker import check_plan
        from quadmesh.agent.planner_model import propose
        for p in g.gen_prompts(a.n, seed=12345):
            plan = propose(p, a.tier, a.device, ckpt_root=a.ckpt_root)
            ok += bool(plan) and not check_plan(plan, p)
        print(f"Task Planner {a.tier}: valid plan for {ok}/{a.n} fresh prompts = {100*ok/a.n:.1f}%")


if __name__ == "__main__":
    main()
