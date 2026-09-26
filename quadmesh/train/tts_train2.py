"""
TTS training v2: batched (not one utterance at a time - that stalled at loss ~1.0 last time), a STOP-token loss so the
model learns where speech ends, and a guided-attention loss so the cross-attention learns to move left-to-right through
the text instead of ignoring it (the classic Tacotron fix for "the model just copies the previous frame").

Voice: LJSpeech is a single female speaker (Linda Johnson, LibriVox), so training on it (the default here, once the
LJSpeech loader in data/streaming.py is used) gives Quadmesh a from-scratch FEMALE voice. Nothing about the voice itself
is designed in code - it comes entirely from the speaker in the training data. A gentle pitch lift in tts_style.py can
make the result read as a bit brighter/younger, but the base timbre is whatever the data teaches.
"""
from __future__ import annotations
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


def log_mel(wav, sr, n_mels=80, n_fft=1024, hop=256):
    import torchaudio
    return torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop,
                                                n_mels=n_mels).clamp(min=1e-5).log()(wav) if False else \
        torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)(wav).clamp(min=1e-5).log()


def guided_attention_loss(attn, text_len, mel_len, g: float = 0.2):
    """Penalises attention mass far from the diagonal - teaches the model to read the text in order."""
    B, T, N = attn.shape
    ti = torch.arange(T, device=attn.device).float()[None, :, None] / max(mel_len - 1, 1)
    ni = torch.arange(N, device=attn.device).float()[None, None, :] / max(text_len - 1, 1)
    W = 1.0 - torch.exp(-((ni - ti) ** 2) / (2 * g * g))
    return (attn * W).mean()


def _batch(records, tok, mb, n_mels, device):
    ex = []
    while len(ex) < mb:
        rec = next(records)
        ids = tok.encode((rec.get("text") or "").strip())
        if not ids:
            continue
        wav = torch.tensor(np.asarray(rec["audio"]["array"], dtype=np.float32))
        mel = log_mel(wav, rec["audio"]["sampling_rate"], n_mels).T                    # [t, n_mels]
        if mel.size(0) < 4 or mel.size(0) > 900:
            continue
        ex.append((ids, mel))
    Tt = max(len(i) for i, _ in ex)
    Tm = max(m.size(0) for _, m in ex)
    ids = torch.zeros(mb, Tt, dtype=torch.long)
    mel = torch.zeros(mb, Tm + 1, n_mels)                                              # +1 leading zero frame (teacher forcing)
    stop = torch.zeros(mb, Tm)
    tlen, mlen = torch.zeros(mb, dtype=torch.long), torch.zeros(mb, dtype=torch.long)
    for i, (t, m) in enumerate(ex):
        ids[i, :len(t)] = torch.tensor(t)
        mel[i, 1:m.size(0) + 1] = m
        stop[i, m.size(0) - 1:] = 1.0
        tlen[i], mlen[i] = len(t), m.size(0)
    return ids.to(device), mel.to(device), stop.to(device), tlen.to(device), mlen.to(device)


def train_tts(tier, ckpt_dir: str, tokenizer_path: str, steps: int = 30000, mb: int = 16, accum: int = 2,
             device: str = "cuda", force: bool = False, log_every: int = 50, sample_every: int = 1000):
    from quadmesh.data.streaming import ljspeech_stream
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.model.audio import TTSModel
    from quadmesh.role_shapes import role_shape

    path = os.path.join(ckpt_dir, "tts.pt")
    if os.path.exists(path) and not force:
        print(f"[tts2] already trained -> {path}", flush=True)
        return path
    os.makedirs(ckpt_dir, exist_ok=True)
    tok = get_or_train(tokenizer_path)
    shape = role_shape(tier, "tts"); shape.vocab_size = tok.vocab_size
    model = TTSModel(shape, tok.vocab_size).to(device)
    peak = min(tier.lr, 2e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=peak, weight_decay=0.01, betas=(0.9, 0.98))
    stream = ljspeech_stream()
    warm = min(500, steps // 10)
    running_mel, running_stop, running_att = 0.0, 0.0, 0.0
    for step in range(steps):
        lr = peak * (step + 1) / warm if step < warm else peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(accum):
            ids, mel, stop, tlen, mlen = _batch(stream, tok, mb, model.n_mels, device)
            out = model(ids, mel[:, :-1], need_attn=True)
            mask = (torch.arange(mel.size(1) - 1, device=device)[None] < mlen[:, None]).float()
            l_mel = (F.l1_loss(out["mel"], mel[:, 1:], reduction="none").mean(-1) * mask).sum() / mask.sum()
            l_stop = F.binary_cross_entropy_with_logits(out["stop_logits"], stop, weight=mask * 4 + 1, reduction="none")
            l_stop = (l_stop * mask).sum() / mask.sum()
            l_att = guided_attention_loss(out["attn"], int(tlen.float().mean()), int(mlen.float().mean()))
            loss = l_mel + 0.5 * l_stop + 0.05 * l_att
            (loss / accum).backward()
            running_mel += l_mel.item() / accum; running_stop += l_stop.item() / accum; running_att += l_att.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % log_every == 0:
            print(f"[tts2] step {step+1}/{steps} mel {running_mel/log_every:.3f} stop {running_stop/log_every:.3f} "
                 f"attn {running_att/log_every:.3f} lr {lr:.2e}", flush=True)
            running_mel = running_stop = running_att = 0.0
        if (step + 1) % sample_every == 0:
            _sample(model, tok, ckpt_dir, step + 1, device)
    torch.save({"model": model.state_dict(), "n_mels": model.n_mels}, path)
    _sample(model, tok, ckpt_dir, steps, device)
    return path


def _sample(model, tok, ckpt_dir, step, device, text: str = "Hello, I am Quadmesh."):
    try:
        from quadmesh.tts_synth import mel_to_wav
        ids = torch.tensor([tok.encode(text)], device=device)
        mel = model.infer(ids, max_frames=400)[0].detach().float().cpu().numpy()
        out = os.path.join(ckpt_dir, "samples"); os.makedirs(out, exist_ok=True)
        mel_to_wav(mel, os.path.join(out, f"step{step}.wav"))
        print(f"[tts2] sample saved: {out}/step{step}.wav ({len(mel)} frames)", flush=True)
    except Exception as e:
        print(f"[tts2] sample failed ({type(e).__name__}: {e})", flush=True)
