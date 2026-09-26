"""
Turns a mel spectrogram into audio with NO extra training: Griffin-Lim phase reconstruction, numpy + scipy only.
Quadmesh's TTS model (train/tts_train2.py) predicts the mel; this file is the vocoder that turns it into a .wav.
It is not as clean as a neural vocoder (a slight "phasey" quality), but it needs no GPU, no training and no extra
dependency, so Quadmesh can speak the moment the mel model is trained.
"""
from __future__ import annotations
import struct
import wave
from typing import Optional

import numpy as np


def mel_to_linear(mel_db: np.ndarray, sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: Optional[float] = None) -> np.ndarray:
    """mel_db: [T, n_mels] log-mel (natural log, as produced by torchaudio.MelSpectrogram(...).log()). Returns [T, n_fft//2+1]
    linear-magnitude spectrogram via the pseudo-inverse of the mel filterbank (non-negative least squares by clipping)."""
    fmax = fmax or sr / 2.0
    fb = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)             # [n_mels, n_freq]
    mag = np.clip(np.exp(mel_db), 1e-5, None)                       # undo the training-time .log()
    pinv = np.linalg.pinv(fb)                                       # [n_freq, n_mels]
    lin = mag @ pinv.T
    return np.clip(lin, 0.0, None)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    n_freq = n_fft // 2 + 1
    m_pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    f_pts = mel2hz(m_pts)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_freq))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c > l:
            fb[i - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        if r > c:
            fb[i - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


def _stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    win = np.hanning(n_fft)
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.stack([x[i * hop:i * hop + n_fft] * win for i in range(max(n_frames, 0))])
    return np.fft.rfft(frames, axis=1)


def _istft(S: np.ndarray, n_fft: int, hop: int, length: Optional[int] = None) -> np.ndarray:
    win = np.hanning(n_fft)
    frames = np.fft.irfft(S, n=n_fft, axis=1) * win
    T = len(frames)
    out_len = (T - 1) * hop + n_fft
    out = np.zeros(out_len); norm = np.zeros(out_len)
    for i in range(T):
        out[i * hop:i * hop + n_fft] += frames[i]
        norm[i * hop:i * hop + n_fft] += win ** 2
    out = out / np.maximum(norm, 1e-8)
    return out[:length] if length else out


def griffin_lim(mag: np.ndarray, n_fft: int, hop: int, iters: int = 60, seed: int = 0) -> np.ndarray:
    """mag: [T, n_fft//2+1] non-negative magnitude spectrogram -> waveform."""
    rng = np.random.default_rng(seed)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, mag.shape))
    S = mag * phase
    x = _istft(S, n_fft, hop)
    for _ in range(iters):
        S2 = _stft(x, n_fft, hop)
        n = min(len(S2), len(mag))
        S = mag[:n] * np.exp(1j * np.angle(S2[:n]))
        x = _istft(S, n_fft, hop)
    return x


def deglitch(x: np.ndarray, sr: int) -> np.ndarray:
    """Fade in/out and a gentle low-pass to soften Griffin-Lim's metallic edges - makes a small, real difference to how it sounds."""
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 7800 / (sr / 2), btype="low")
    x = filtfilt(b, a, x)
    n = min(int(0.01 * sr), len(x) // 4)
    if n > 1:
        ramp = np.linspace(0, 1, n)
        x[:n] *= ramp; x[-n:] *= ramp[::-1]
    return x / max(np.abs(x).max(), 1e-6) * 0.9


def write_wav(path: str, x: np.ndarray, sr: int) -> str:
    pcm = np.clip(x, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def mel_to_wav(mel_db: np.ndarray, path: str, sr: int = 22050, n_fft: int = 1024, n_mels: int = 80, hop: int = 256,
              iters: int = 60) -> str:
    mag = mel_to_linear(mel_db, sr, n_fft, n_mels)
    x = griffin_lim(mag, n_fft, hop, iters)
    x = deglitch(x, sr)
    return write_wav(path, x, sr)
