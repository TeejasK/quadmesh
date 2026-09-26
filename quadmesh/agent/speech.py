"""
Speech input for the REPL.

Path 1: your trained, from-scratch ASR checkpoint (Sec 17), if promoted to
live. Path 2: a pretrained fallback (SpeechRecognition + Google's free API)
so speech input works today — this is NOT from-scratch and is clearly not
what Sec 17 specifies; it's a stopgap until your own ASR model is trained and
promoted, at which point path 1 takes over automatically and this is skipped.
"""
from __future__ import annotations
from quadmesh.role_shapes import role_shape
import os


def _try_trained_asr(seconds: float = 4.0) -> str | None:
    ckpt_root = os.environ.get("QUADMESH_CKPT_ROOT")
    tier_name = os.environ.get("QUADMESH_TIER", "100M")
    if not ckpt_root:
        return None
    live_path = f"{ckpt_root}/Quadmesh-{tier_name}/live/asr.pt"
    if not os.path.exists(live_path):
        return None

    import torch, sounddevice as sd
    from quadmesh.model.audio import ASRModel
    from quadmesh.config import get_tier
    from quadmesh.data.tokenizer import get_or_train
    from quadmesh.train.io_train import log_mel

    tier = get_tier(tier_name)
    tok = get_or_train(f"{ckpt_root}/tokenizer/quadmesh-bpe.json")
    model = ASRModel(role_shape(tier, "asr"), vocab_size=tok.vocab_size)
    model.load_state_dict(torch.load(live_path, map_location="cpu", weights_only=False))
    model.eval()

    sr = 16000
    print(f"[speech] recording {seconds}s (trained ASR)...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    sd.wait()
    wav = torch.tensor(audio.squeeze(), dtype=torch.float32)
    feats = log_mel(wav, sr).T[None]
    with torch.no_grad():
        logits = model(feats)
    ids = logits.argmax(-1)[0].unique_consecutive().tolist()
    ids = [i for i in ids if i != tok.pad_id]
    return tok.decode(ids)


def _fallback_pretrained_asr(seconds: float = 4.0) -> str | None:
    try:
        import speech_recognition as sr
    except ImportError:
        print("[speech] `pip install SpeechRecognition pyaudio` to enable "
              "the fallback speech path.")
        return None

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"[speech] listening (fallback, pretrained, {seconds}s max)...")
        audio = r.listen(source, phrase_time_limit=seconds)
    try:
        return r.recognize_google(audio)
    except Exception as e:
        print(f"[speech] could not transcribe: {e}")
        return None


def listen(seconds: float = 4.0, require_confirmation: bool = True, max_retries: int = 2) -> str | None:
    """Mandatory repeat-back confirmation (was previously only documented in chat_kb.py, never actually
    implemented anywhere listen() is called - agent/repl.py, chat_agent.py, app.py, assistant.py all called
    this with zero confirmation). Not optional by default: pass require_confirmation=False only for
    automated/test contexts where no human is present to confirm.

    No accuracy numbers exist yet for either ASR path (trained or fallback), so confirmation is the one
    honest safeguard available right now - it costs one extra turn and catches a wrong transcript before it
    reaches the planner, rather than after a plan gets built from words nobody actually said."""
    for attempt in range(max_retries + 1):
        text = _try_trained_asr(seconds)
        source = "trained ASR"
        if not text:
            print("[speech] no trained/promoted ASR found - using pretrained fallback")
            text = _fallback_pretrained_asr(seconds)
            source = "fallback ASR"
        if not text:
            print("[speech] nothing understood, try again" if attempt < max_retries else
                  "[speech] still nothing understood after retries - giving up this turn")
            continue
        print(f"[speech] ({source}) heard: {text!r}")
        if not require_confirmation:
            return text
        reply = input(f"[speech] confirm - is this right? [Y/n/retry] \"{text}\" ").strip().lower()
        if reply in ("", "y", "yes"):
            return text
        if reply in ("n", "no") and attempt == max_retries:
            print("[speech] not confirmed and out of retries - discarding")
            return None
        print("[speech] not confirmed - listening again" if reply not in ("n", "no") else
              "[speech] rejected - listening again")
    return None
