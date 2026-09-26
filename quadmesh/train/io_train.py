"""Sec 17 — ASR and TTS trained from scratch, streamed, no pretrained models."""
from quadmesh.role_shapes import role_shape
import os, torch, torch.nn as nn
from quadmesh.data.streaming import common_voice_stream, ljspeech_stream, libritts_r_stream


def log_mel(wav, sr, n_mels=80, n_fft=400, hop=160):
    import torchaudio
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)(wav).clamp(min=1e-5).log()


def train_asr(tier, language="en", ckpt_dir="/vol/io", device="cuda"):
    """Conformer-style encoder + CTC head, initialised randomly."""
    from quadmesh.model.audio import ASRModel
    from quadmesh.data.tokenizer import get_or_train
    tok = get_or_train("/vol/tokenizer/quadmesh-bpe.json")
    model = ASRModel(role_shape(tier, "asr"), vocab_size=tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tier.lr, weight_decay=0.01)
    ctc = nn.CTCLoss(blank=tok.pad_id, zero_infinity=True)

    budget_sec, seen = tier.asr_hours * 3600, 0.0
    os.makedirs(ckpt_dir, exist_ok=True)
    for i, rec in enumerate(common_voice_stream(language)):
        wav = torch.tensor(rec["audio"]["array"], dtype=torch.float32)
        seen += len(wav) / rec["audio"]["sampling_rate"]
        if seen > budget_sec:
            break
        feats = log_mel(wav, rec["audio"]["sampling_rate"]).T[None].to(device)
        ids = torch.tensor(tok.encode(rec["sentence"]), device=device)[None]
        logits = model(feats).log_softmax(-1).transpose(0, 1)
        loss = ctc(logits, ids,
                   torch.tensor([logits.size(0)], device=device),
                   torch.tensor([ids.size(1)], device=device))
        loss.backward(); opt.step(); opt.zero_grad()
        if i % 200 == 0:
            print(f"[asr] {seen/3600:.2f}/{tier.asr_hours}h loss {loss.item():.3f}")
    torch.save(model.state_dict(), f"{ckpt_dir}/asr.pt")
    return f"{ckpt_dir}/asr.pt"


def train_tts(tier, ckpt_dir="/vol/io", device="cuda"):
    """Autoregressive mel decoder. LJSpeech (24h, single speaker) by default;
    LibriTTS-R (~585h, multi-speaker) once tier.tts_hours exceeds LJSpeech."""
    from quadmesh.model.audio import TTSModel
    from quadmesh.data.tokenizer import get_or_train
    tok = get_or_train("/vol/tokenizer/quadmesh-bpe.json")
    stream = ljspeech_stream() if tier.tts_hours <= 24.0 else libritts_r_stream()
    model = TTSModel(role_shape(tier, "tts"), vocab_size=tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tier.lr, weight_decay=0.01)

    budget_sec, seen = tier.tts_hours * 3600, 0.0
    os.makedirs(ckpt_dir, exist_ok=True)
    for i, rec in enumerate(stream):
        wav = torch.tensor(rec["audio"]["array"], dtype=torch.float32)
        seen += len(wav) / rec["audio"]["sampling_rate"]
        if seen > budget_sec:
            break
        mel = log_mel(wav, rec["audio"]["sampling_rate"]).T[None].to(device)
        text = rec.get("text") or rec.get("normalized_text") or ""
        ids = torch.tensor(tok.encode(text), device=device)[None]
        pred = model(ids, mel[:, :-1])
        loss = nn.functional.l1_loss(pred, mel[:, 1:])
        loss.backward(); opt.step(); opt.zero_grad()
        if i % 200 == 0:
            print(f"[tts] {seen/3600:.2f}/{tier.tts_hours}h loss {loss.item():.3f}")
    torch.save(model.state_dict(), f"{ckpt_dir}/tts.pt")
    return f"{ckpt_dir}/tts.pt"
