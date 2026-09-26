"""
Speaking style, applied AFTER Griffin-Lim (tts_synth.mel_to_wav): none of this changes what the model learned, only how
the raw vocoded audio is played. Safe, reversible signal processing, numpy + scipy only.
"""
from __future__ import annotations
import numpy as np


def pitch_shift(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Simple, RELIABLE resample-based pitch shift (numpy only, no phase vocoder). It shifts pitch correctly but the
    clip's duration changes by the same ratio (a few percent for the small shifts this module uses) - for a one-off
    spoken reply that trade-off is inaudible in practice. Verified: shifting +2 semitones moves a 220 Hz tone's energy
    to 246.9 Hz."""
    if abs(semitones) < 1e-6:
        return x
    rate = 2.0 ** (semitones / 12.0)
    n_new = max(1, int(round(len(x) / rate)))
    xp = np.linspace(0, len(x) - 1, n_new)
    return np.interp(xp, np.arange(len(x)), x)


def warm_eq(x: np.ndarray, sr: int, presence_db: float = 2.0) -> np.ndarray:
    """A small lift around 2-4 kHz (speech presence) for a clearer, friendlier tone; light high-shelf roll-off above 9 kHz."""
    from scipy.signal import butter, sosfilt
    presence = sosfilt(butter(2, [2000, 4200], btype="bandpass", fs=sr, output="sos"), x) * (10 ** (presence_db / 20) - 1)
    hi = sosfilt(butter(2, 9000, btype="highpass", fs=sr, output="sos"), x) * -0.3
    y = x + presence + hi
    return y / max(np.abs(y).max(), 1e-6) * 0.9


def cute_female(x: np.ndarray, sr: int, lift_semitones: float = 1.5) -> np.ndarray:
    """A light, natural-sounding lift + presence boost. The voice is still whatever LJSpeech's speaker sounds like underneath -
    this does not turn a male-trained model female; it only brightens the trained voice. Keep `lift_semitones` small (0-2) to
    avoid a chipmunk effect."""
    return warm_eq(pitch_shift(x, sr, lift_semitones), sr, 2.0)
