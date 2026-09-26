"""From-scratch audio encoders/decoders for the Sec 17 I/O adapter layer."""
import torch, torch.nn as nn
from quadmesh.model.transformer import QuadmeshBlock, RMSNorm, build_rope


class _Stack(nn.Module):
    def __init__(self, shape, n_layers=None):
        super().__init__()
        self.shape = shape
        n = n_layers or max(shape.n_layers // 2, 2)
        self.blocks = nn.ModuleList([QuadmeshBlock(shape) for _ in range(n)])
        self.norm = RMSNorm(shape.d_model)

    def forward(self, x):
        t = x.size(1)
        cos, sin = build_rope(t, self.shape.d_model // self.shape.n_heads,
                              self.shape.rope_theta, x.device, x.dtype)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.norm(x)


class ASRModel(nn.Module):
    def __init__(self, shape, vocab_size, n_mels=80):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Conv1d(n_mels, shape.d_model, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(shape.d_model, shape.d_model, 3, stride=2, padding=1), nn.GELU())
        self.enc = _Stack(shape)
        self.head = nn.Linear(shape.d_model, vocab_size)

    def forward(self, mel):                      # mel: [b, t, n_mels]
        x = self.inp(mel.transpose(1, 2)).transpose(1, 2)
        return self.head(self.enc(x))


class TTSModel(nn.Module):
    """Tacotron-style: text encoder -> cross-attention -> autoregressive mel decoder + a STOP head (so the model
    learns where the utterance ends, instead of running for a fixed number of frames). Vocoding is Griffin-Lim
    (quadmesh/tts_synth.py) - no extra network to train."""

    def __init__(self, shape, vocab_size, n_mels=80, speaker_dim=0, n_speakers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, shape.d_model)
        self.text_enc = _Stack(shape)
        self.mel_in = nn.Sequential(nn.Linear(n_mels, shape.d_model), nn.ReLU(), nn.Linear(shape.d_model, shape.d_model))
        self.dec = _Stack(shape)
        self.xattn = nn.MultiheadAttention(shape.d_model, shape.n_heads, batch_first=True)
        self.out = nn.Linear(shape.d_model, n_mels)
        self.stop = nn.Linear(shape.d_model, 1)
        self.speaker = nn.Embedding(max(n_speakers, 1), shape.d_model) if speaker_dim else None
        self.n_mels = n_mels

    def forward(self, ids, mel_prev, speaker_id=None, need_attn: bool = False):
        mem = self.text_enc(self.embed(ids))
        if self.speaker is not None and speaker_id is not None:
            mem = mem + self.speaker(speaker_id).unsqueeze(1)
        h = self.dec(self.mel_in(mel_prev))
        h, attn = self.xattn(h, mem, mem, need_weights=need_attn, average_attn_weights=True)
        return {"mel": self.out(h), "stop_logits": self.stop(h).squeeze(-1), "attn": attn}

    @torch.no_grad()
    def infer(self, ids, max_frames: int = 800, stop_thresh: float = 0.5, speaker_id=None):
        """Greedy autoregressive synthesis: predicts one mel frame at a time until the stop head fires."""
        was_training = self.training
        self.eval()
        b = ids.size(0)
        mel = torch.zeros(b, 1, self.n_mels, device=ids.device, dtype=self.embed.weight.dtype)
        done = torch.zeros(b, dtype=torch.bool, device=ids.device)
        for _ in range(max_frames):
            out = self(ids, mel, speaker_id)
            nxt = out["mel"][:, -1:]
            mel = torch.cat([mel, nxt], dim=1)
            done |= torch.sigmoid(out["stop_logits"][:, -1]) > stop_thresh
            if bool(done.all()):
                break
        if was_training:
            self.train()
        return mel[:, 1:]                              # drop the leading zero frame
