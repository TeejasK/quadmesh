"""GroundCUA streaming for the two vision roles (ui_grounding, screenshot_diff).

Images are decoded in memory per record and patchified on the fly. Nothing is
written to local disk, same as the text path.
"""
import io, itertools
import torch
from PIL import Image

PATCH = 16


def _to_patches(img: Image.Image, size: int = 512):
    img = img.convert("RGB").resize((size, size))
    x = torch.from_numpy(__import__("numpy").asarray(img)).permute(2, 0, 1).float() / 255.0
    c, h, w = x.shape
    x = x.unfold(1, PATCH, PATCH).unfold(2, PATCH, PATCH)
    return x.permute(1, 2, 0, 3, 4).reshape(-1, c * PATCH * PATCH)


def packed_vision_batches(records, tier, device="cuda", size=512):
    it = iter(records)
    while True:
        chunk = list(itertools.islice(it, tier.micro_batch_size))
        if len(chunk) < tier.micro_batch_size:
            return
        imgs, tgts = [], []
        for r in chunk:
            im = r["image"] if isinstance(r.get("image"), Image.Image) else \
                 Image.open(io.BytesIO(r["image"]["bytes"]))
            imgs.append(_to_patches(im, size))
            # grounding target: normalized click box -> quantized token ids
            box = r.get("bbox") or r.get("box") or [0, 0, 0, 0]
            tgts.append(torch.tensor([int(v * 999) for v in box[:4]], dtype=torch.long))
        yield torch.stack(imgs).to(device), torch.stack(tgts).to(device)
