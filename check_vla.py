"""
Smoke test for the VLA model on your machine (CPU is fine):   python check_vla.py
Checks: (1) a training step runs and gives a finite loss, (2) KV-cached action generation == plain recompute.
"""
import torch

from quadmesh.config import ModelShape
from quadmesh.model.vla import QuadmeshVLA


def main():
    torch.manual_seed(0)
    shape = ModelShape(d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=1024)
    shape.vocab_size = 200
    m = QuadmeshVLA(shape)
    patches = torch.rand(2, 16, 768)
    ids = torch.randint(0, 200, (2, 12))
    tg = ids.clone(); tg[:, :6] = -100
    loss = m(patches, ids, targets=tg)["loss"]
    loss.backward()
    assert torch.isfinite(loss), "loss is not finite"
    print(f"training step ok, loss {loss.item():.3f} (about ln(200)={torch.log(torch.tensor(200.)).item():.2f} expected)")

    m.eval()
    p1, i1 = patches[:1], ids[:1]
    fast = m.generate_action(p1, i1, max_new=10)
    seq = i1.clone()
    slow = []
    with torch.no_grad():
        for _ in range(10):
            nxt = m(p1, seq)["logits"][:, -1].argmax(-1, keepdim=True)
            slow.append(int(nxt)); seq = torch.cat([seq, nxt], dim=1)
    print("IDENTICAL" if fast == slow else f"DIFFERENT\n fast {fast}\n slow {slow}")
    raise SystemExit(0 if fast == slow else 1)


if __name__ == "__main__":
    main()
