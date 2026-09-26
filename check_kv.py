"""
Checks that the KV cache changes SPEED only, never the answer.
Run from the project root:   python check_kv.py            (add --tier 500M etc. for other sizes)
Expected: "IDENTICAL" and a speed-up number.
"""
import argparse
import time

import torch

from quadmesh.config import get_tier
from quadmesh.model.transformer import build_role_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="100M")
    ap.add_argument("--prompt", type=int, default=60)
    ap.add_argument("--new", type=int, default=60)
    a = ap.parse_args()
    torch.manual_seed(0)
    tier = get_tier(a.tier)
    model = build_role_model("task_planner", tier.shape).eval()
    ids = torch.randint(0, tier.shape.vocab_size, (1, a.prompt))

    t0 = time.time()
    out = ids.clone()
    with torch.no_grad():
        for _ in range(a.new):                                  # the old way: rerun everything
            nxt = model(out)["logits"][:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
    slow, t_slow = out[0, a.prompt:].tolist(), time.time() - t0

    t0 = time.time()
    fast = model.generate(ids, max_new_tokens=a.new)             # the new way: KV cache
    t_fast = time.time() - t0

    same = slow == fast
    print("IDENTICAL" if same else f"DIFFERENT at first mismatch: "
          f"{next(i for i,(x,y) in enumerate(zip(slow,fast)) if x!=y)}")
    print(f"without cache: {t_slow:.2f}s   with cache: {t_fast:.2f}s   speed-up x{t_slow/max(t_fast,1e-9):.1f}")
    if not same:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
