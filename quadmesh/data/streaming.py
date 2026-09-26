"""
Streaming dataset loaders (Sec 16).

Nothing here downloads a dataset. `datasets.load_dataset(..., streaming=True)`
returns an IterableDataset that pulls records over HTTP as the training loop
consumes them. The loop stops pulling once the tier's token budget is hit.

This module is THE ONLY per-role / per-tier variation point. The training
stages in quadmesh/train/ take an iterator and do not know or care where the
bytes came from.
"""

from __future__ import annotations

from __future__ import annotations
import itertools
import random
from typing import Dict, Iterable, Iterator, List, Optional

import torch
from datasets import load_dataset, interleave_datasets, Audio

from quadmesh.config import (
    ROLE_SPECS,
    STARCODER_MIX_RATIO,
    AUDITOR_SHARD_OFFSET,
    TierConfig,
)

SLIMPAJAMA = "gmongaras/SlimPajama-627B_Reupload"
STARCODER = "bigcode/starcoderdata"
TEXT2CAD = "SadilKhan/Text2CAD"
GROUNDCUA = "ServiceNow/GroundCUA"
COMMON_VOICE = "mozilla-foundation/common_voice_17_0"   # unused now (its Hub repo has no data files)
LJSPEECH = "badayvedat/LJSpeech-1.1"
LIBRITTS_R = "mythicinfinity/libritts_r"
LIBRISPEECH = "openslr/librispeech_asr"


# ----------------------------------------------------------------------------
# Raw streams
# ----------------------------------------------------------------------------

import os

def _hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

def slimpajama_stream(split: str = "train", seed: int = 0, buffer: int = 1_000):
    print(f"[slimpajama_stream] opening {SLIMPAJAMA} (token={'set' if _hf_token() else 'MISSING'})...", flush=True)
    ds = load_dataset(SLIMPAJAMA, split=split, streaming=True, token=_hf_token())
    print("[slimpajama_stream] dataset handle opened, wrapping in shuffle buffer...", flush=True)
    return ds.shuffle(seed=seed, buffer_size=buffer)

def starcoder_stream(split: str = "train", seed: int = 0, buffer: int = 10_000):
    """70B-tier top-up only (Sec 14 caveat). CrystalCoder mix ratio."""
    ds = load_dataset(STARCODER, data_dir="python", split=split, streaming=True)
    return ds.shuffle(seed=seed, buffer_size=buffer)


def text2cad_stream(split: str = "train"):
    """~660K annotations / ~390M tokens. Used IN FULL at every tier (Sec 14).
    Small enough to download, but streamed anyway so nothing special-cases."""
    return load_dataset(TEXT2CAD, split=split, streaming=True)


def synthetic_gui_stream(seed: int = 0):
    """Fallback generator that yields synthetic UI layout screenshots and bounding boxes."""
    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    while True:
        img = Image.new("RGB", (512, 512), color=(240, 242, 245))
        draw = ImageDraw.Draw(img)
        x1 = rng.uniform(0.05, 0.6)
        y1 = rng.uniform(0.05, 0.6)
        w = rng.uniform(0.1, 0.3)
        h = rng.uniform(0.05, 0.2)
        x2, y2 = min(x1 + w, 0.95), min(y1 + h, 0.95)
        draw.rectangle([x1 * 512, y1 * 512, x2 * 512, y2 * 512], fill=(59, 130, 246), outline=(37, 99, 235))
        yield {"image": img, "bbox": [x1, y1, x2, y2]}


def groundcua_stream(split: str = "train", seed: int = 0):
    """REAL UI screenshots (GroundCUA, 87 apps): image + normalised box of one labelled element.
    Reads the annotation files directly (load_dataset cannot parse their mixed schema). Falls back to the
    synthetic generator only if the real data cannot be reached - and says so loudly."""
    def _generator():
        try:
            from quadmesh.data.vla_data import groundcua_box_stream
            got = 0
            for item in groundcua_box_stream(seed=seed):
                got += 1
                yield item
            if got == 0:
                raise RuntimeError("no real examples were produced")
        except Exception as e:
            print(f"[WARNING] REAL GroundCUA data unavailable ({type(e).__name__}: {e}). TRAINING ON SYNTHETIC DATA.", flush=True)
            for item in synthetic_gui_stream(seed=seed):
                yield item
    return _generator()


def vla_action_stream(demo_dir: Optional[str] = None, seed: int = 0):
    """The vla role's actual training stream: GroundCUA (single-step grounding) + VideoCAD (long-horizon CAD
    session patterns) + your record_demo.py sessions (real Blender behaviour), mixed per vla_data.vla_stream's
    weighting. Falls back to synthetic clicks only if GroundCUA itself is unreachable (vla_stream already
    degrades gracefully if VideoCAD or demos are unavailable - see its docstring)."""
    def _generator():
        try:
            from quadmesh.data.vla_data import vla_stream
            got = 0
            for item in vla_stream(demo_dir=demo_dir, seed=seed):
                got += 1
                yield item
                if got == 1:   # cheap proof-of-life check without buffering the whole stream
                    continue
        except Exception as e:
            print(f"[WARNING] REAL vla data unavailable ({type(e).__name__}: {e}). TRAINING ON SYNTHETIC DATA.", flush=True)
            for item in synthetic_gui_stream(seed=seed):
                yield item
    return _generator()


def common_voice_stream(language: str = "en", split: str = "train"):
    """ASR source. Common Voice's Hub repo is empty, so this streams LibriSpeech (English only) and yields
    records shaped like the old ones: {"audio", "sentence"}. Name kept so train/io_train.py needs no change."""
    if language != "en":
        raise ValueError(f"LibriSpeech is English-only; pick another corpus for language={language!r}")
    last_err = None
    for cfg, spl in (("clean", "train.100"), ("all", "train.clean.100")):
        try:
            ds = load_dataset(LIBRISPEECH, cfg, split=spl, streaming=True)
            break
        except Exception as e:          # config/split names differ between repo revisions
            last_err = e
    else:
        raise last_err
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    def _gen():
        for rec in ds:
            yield {"audio": rec["audio"], "sentence": rec["text"].lower()}
    return _gen()


def _pick(rec, keys):
    for k in keys:
        v = rec.get(k)
        if v:
            return v
    return None


def _decode_audio(a):
    import io
    if isinstance(a, dict) and a.get("array") is not None:
        return {"array": a["array"], "sampling_rate": a["sampling_rate"]}
    raw = a.get("bytes") if isinstance(a, dict) else a
    if isinstance(raw, (bytes, bytearray)) and raw:
        import soundfile as sf
        arr, sr = sf.read(io.BytesIO(bytes(raw)), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return {"array": arr, "sampling_rate": sr}
    return None


def _ljspeech_records(split: str = "train", max_skipped: int = 50):
    ds = load_dataset(LJSPEECH, split=split, streaming=True)
    yielded = skipped = 0
    for n, rec in enumerate(ds):
        if n == 0:
            print("[tts] LJSpeech first record: " + str({k: type(v).__name__ for k, v in rec.items()}), flush=True)
        a = _pick(rec, ("audio", "wav", "flac", "mp3"))
        t = _pick(rec, ("normalized_transcription", "transcription", "text", "txt", "normalized_text",
                        "text_normalized", "transcript"))
        audio = None
        if a is not None and t:
            if isinstance(t, (bytes, bytearray)):
                t = t.decode("utf-8", errors="ignore")
            audio = _decode_audio(a)
        if audio is None:
            skipped += 1
            if yielded == 0 and skipped >= max_skipped:
                print(f"[tts] LJSpeech: first {skipped} records unusable; giving up.", flush=True)
                return
            continue
        yielded += 1
        yield {"audio": audio, "text": t}


def ljspeech_stream(split: str = "train"):
    """TTS. Yields {"audio": {"array","sampling_rate"}, "text": str}; falls back to LibriSpeech if LJSpeech is unusable."""
    def _gen():
        got = 0
        print("[tts] opening LJSpeech stream...", flush=True)
        try:
            for r in _ljspeech_records(split):
                got += 1
                yield r
        except Exception as e:
            print(f"[tts] LJSpeech stream failed ({type(e).__name__}: {e})", flush=True)
        if got == 0:
            print("[tts] falling back to LibriSpeech...", flush=True)
            for rec in common_voice_stream("en"):
                yield {"audio": rec["audio"], "text": rec["sentence"]}
    return _gen()


def libritts_r_stream(split: str = "train.clean.360"):
    """TTS swap-in for multi-speaker (~585h). Same from-scratch principle."""
    ds = load_dataset(LIBRITTS_R, "clean", split=split, streaming=True)
    return ds.cast_column("audio", Audio(sampling_rate=24_000))


# ----------------------------------------------------------------------------
# Role routing (Sec 12.1 — disjoint sources, not just disjoint weights)
# ----------------------------------------------------------------------------

def _text_field(rec: Dict) -> str:
    for k in ("text", "content", "description", "prompt", "code", "body"):
        if k in rec and rec[k]:
            val = rec[k]
            return val if isinstance(val, str) else str(val)
    # Fallback to any non-empty string in the record
    for v in rec.values():
        if isinstance(v, str) and v.strip():
            return v
    return ""


def role_text_stream(role: str, tier: TierConfig, seed: int = 0) -> Iterable[Dict]:
    """
    Returns the raw record stream for a text role.

    - Every text role samples SlimPajama.
    - spec_auditor is offset into a different region of the shuffle stream and
      re-seeded, so it does not co-train on the generator's exact documents.
    - 70B interleaves StarCoder to clear the Chinchilla floor.
    """
    spec = ROLE_SPECS[role]
    role_seed = seed + (AUDITOR_SHARD_OFFSET if spec.disjoint_source else 0)

    base = slimpajama_stream(seed=role_seed)
    if spec.disjoint_source:
        # skip forward so the auditor's window does not overlap the generator's
        base = base.skip(AUDITOR_SHARD_OFFSET)

    if tier.needs_starcoder:
        base = interleave_datasets(
            [base, starcoder_stream(seed=role_seed)],
            probabilities=[1.0 - STARCODER_MIX_RATIO, STARCODER_MIX_RATIO],
            seed=role_seed,
            stopping_strategy="all_exhausted",
        )
    return base


def planner_sft_stream(n: int | None = None, seed: int = 0):
    """task_planner Stage 2: ENDLESS fresh engineering-plan examples (never repeats) - see
    pipeline/datasets/engineering_gen.py (10,000+ part families + free-form CSG + repair examples)."""
    from quadmesh.pipeline.datasets.engineering_gen import gen_example
    import random
    random.seed(seed)
    i = 0
    while n is None or i < n:
        yield gen_example()
        i += 1


def spec_sft_stream(n: int | None = None, seed: int = 0):
    """spec_generator Stage 2: request -> design parameters (hexacopter / robot_arm / unsupported), endless."""
    from quadmesh.spec_gen_data import gen_example
    import random
    random.seed(seed)
    i = 0
    while n is None or i < n:
        yield gen_example()
        i += 1


def role_stage2_stream(role: str) -> Optional[Iterable[Dict]]:
    """Stage 2 SFT overlay. spec_generator gets spec_sft; task_planner gets
    the local synthetic plan dataset; everyone else has none."""
    spec = ROLE_SPECS[role]
    if spec.stage2_overlay == "local:spec_sft" or spec.stage2_overlay == TEXT2CAD:
        return spec_sft_stream()
    if spec.stage2_overlay == "local:planner_sft":
        return planner_sft_stream()
    return None


def role_vision_stream(role: str, seed: int = 0) -> Iterable[Dict]:
    return groundcua_stream(seed=seed)


def _manual_interleave(a: Iterable[Dict], b: Iterable[Dict], p_overlay: float, seed: int = 0):
    """Random-choice interleave for when one side is a plain generator
    (a local overlay), not an HF Dataset interleave_datasets can consume."""
    import random
    rng = random.Random(seed)
    ia, ib = iter(a), iter(b)
    while True:
        src = ib if rng.random() < p_overlay else ia
        try:
            yield next(src)
        except StopIteration:
            other = ia if src is ib else ib
            try:
                yield next(other)
            except StopIteration:
                return


# ----------------------------------------------------------------------------
# Token packing — turns a record stream into fixed-length training batches
# ----------------------------------------------------------------------------

class PackedTokenIterator:
    """
    Concatenates documents into a contiguous token stream and yields
    (seq_len + 1)-token windows. Counts tokens so the caller can stop exactly
    at the tier budget instead of relying on epochs over an unknown-size shard.
    """

    def __init__(self, records: Iterable[Dict], tokenizer, seq_len: int,
                 token_budget: int, text_fn=_text_field):
        self.records = iter(records)
        self.tok = tokenizer
        self.seq_len = seq_len
        self.token_budget = token_budget
        self.text_fn = text_fn
        self.consumed = 0
        self._buf: List[int] = []

    def __iter__(self) -> Iterator[torch.Tensor]:
        need = self.seq_len + 1
        while self.consumed < self.token_budget:
            while len(self._buf) < need:
                try:
                    rec = next(self.records)
                except StopIteration:
                    return
                ids = self.tok.encode(self.text_fn(rec))
                if not ids:
                    continue
                self._buf.extend(ids + [self.tok.eos_id])
            window, self._buf = self._buf[:need], self._buf[need - 1:]
            self.consumed += self.seq_len
            yield torch.tensor(window, dtype=torch.long)


import queue, threading

def prefetch_iterator(iterator, buffer_size: int = 6):
    """Prefetches batches in a background thread so the GPU never waits on HTTP/network streaming."""
    q = queue.Queue(maxsize=buffer_size)
    sentinel = object()

    def _worker():
        try:
            for item in iterator:
                q.put(item)
        except Exception as e:
            q.put(e)
        finally:
            q.put(sentinel)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def packed_batches(records, tokenizer, tier: TierConfig, token_budget: int,
                   device="cuda"):
    """Yields (inputs, targets) micro-batches of shape [mb, seq_len] with background prefetching."""
    it = iter(PackedTokenIterator(records, tokenizer, tier.seq_len, token_budget))

    def _batch_generator():
        while True:
            chunk = list(itertools.islice(it, tier.micro_batch_size))
            if len(chunk) < tier.micro_batch_size:
                return
            batch = torch.stack(chunk)
            yield batch[:, :-1].contiguous(), batch[:, 1:].contiguous()

    for x, y in prefetch_iterator(_batch_generator(), buffer_size=6):
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ----------------------------------------------------------------------------
# Convenience: one call per (role, stage)
# ----------------------------------------------------------------------------

def build_loader(role: str, stage: int, tier: TierConfig, tokenizer,
                 seed: int = 0, device: str = "cuda"):
    """
    The single entry point training stages call. Swapping tiers or roles
    changes ONLY what this returns — the training loop is untouched.
    """
    spec = ROLE_SPECS[role]

    if spec.modality == "vision":
        from quadmesh.data.vision import packed_vision_batches
        return packed_vision_batches(role_vision_stream(role, seed), tier, device)

    if stage == 1:
        stream = role_text_stream(role, tier, seed)
        budget = tier.tokens_per_role
    elif stage == 2:
        overlay = role_stage2_stream(role)
        if overlay is not None:
            spec = ROLE_SPECS[role]
            if isinstance(spec.stage2_overlay, str) and spec.stage2_overlay.startswith("local:"):
                # local:* overlays are plain Python generators, not HF Datasets
                # -> interleave_datasets doesn't apply; sample manually instead.
                stream = _manual_interleave(role_text_stream(role, tier, seed), overlay,
                                            p_overlay=(0.95 if role in ("task_planner", "spec_generator") else 0.7), seed=seed)
            else:
                stream = interleave_datasets(
                    [role_text_stream(role, tier, seed), overlay],
                    probabilities=[0.3, 0.7],      # SFT is overlay-dominant
                    seed=seed, stopping_strategy="all_exhausted",
                )
        else:
            stream = role_text_stream(role, tier, seed + 1)
        budget = max(tier.tokens_per_role // 10, 50_000_000)
        if role in ("task_planner", "spec_generator"):
            budget = max(budget, 100_000_000)
    else:
        raise ValueError(f"stage {stage} is self-generated or local; see train/")

    return packed_batches(stream, tokenizer, tier, budget, device=device)
