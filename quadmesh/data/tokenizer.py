"""
From-scratch tokenizer. No pretrained vocab pulled in.

Trained once on a short prefix of the SlimPajama stream (streaming, so still
no full download), persisted to the Modal volume, and reused by all 9 roles so
token IDs are comparable across roles during arbitration.
"""

from __future__ import annotations

import itertools
import json
import os
from typing import List

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


EOS = "<|eos|>"
PAD = "<|pad|>"
BOS = "<|bos|>"
# One role tag per text role. Appended AFTER the original specials so the original ids do not move. The shared
# text model (model/shared_backbone.py) reads the tag as a conditioning token - see quadmesh/roles_io.py.
ROLE_NAMES = ("chat", "task_planner", "spec_generator", "spec_auditor",
              "arbitration", "cad_skill_dispatch", "risk_tier_classifier", "abuse_pattern")
ROLE_TAGS = {r: f"<|role_{r}|>" for r in ROLE_NAMES}
SPECIALS = [PAD, BOS, EOS, "<|img|>", "<|action|>", "<|spec|>", "<|verdict|>"] + list(ROLE_TAGS.values())


class QuadmeshTokenizer:
    def __init__(self, tk: Tokenizer):
        self.tk = tk
        self.eos_id = tk.token_to_id(EOS)
        self.bos_id = tk.token_to_id(BOS)
        self.pad_id = tk.token_to_id(PAD)

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        return self.tk.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tk.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tk.save(path)

    @classmethod
    def load(cls, path: str) -> "QuadmeshTokenizer":
        return cls(Tokenizer.from_file(path))

    @classmethod
    def train_local(cls, texts, vocab_size: int = 3000) -> "QuadmeshTokenizer":
        """Small BPE from an iterable of strings, no network. For tests and offline development only."""
        tk = Tokenizer(models.BPE(unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        tk.train_from_iterator(texts, trainer=trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
        return cls(tk)

    @classmethod
    def build_corpus_file(cls, corpus_path: str, n_cad: int = 5_000,
                          n_docs: int = 15_000, n_role_docs: int = 12_000,
                          seed: int = 0, commit_every: int = 1000, commit_fn=None) -> str:
        """Streams the corpus ONCE and writes it to a plain-text file (one doc per line), resuming from
        wherever it left off if a previous attempt was preempted partway through. This is the slow,
        network-bound step; keeping it separate from BPE training means a preemption during BPE merges
        (or vice versa) never throws away already-streamed data."""
        from quadmesh.data.streaming import slimpajama_stream, text2cad_stream

        done_marker = corpus_path + ".done"
        if os.path.exists(done_marker):
            print(f"[tokenizer] Corpus already complete at {corpus_path}, skipping re-stream.", flush=True)
            return corpus_path

        # Resume support: count lines already written so we don't redo work that's already on disk.
        already = 0
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                already = sum(1 for _ in f)
            print(f"[tokenizer] Resuming corpus build: {already:,} docs already on disk.", flush=True)

        os.makedirs(os.path.dirname(corpus_path) or ".", exist_ok=True)

        def _escape(text: str) -> str:
            return text.replace("\n", "\\n")

        with open(corpus_path, "a", encoding="utf-8") as out:
            written = already
            # --- Text2CAD ---
            cad_target = n_cad
            if written < cad_target:
                print(f"[tokenizer] Streaming CAD docs ({cad_target - written:,} remaining of {cad_target:,})...", flush=True)
                cad_count = written
                for r in itertools.islice(text2cad_stream(), cad_target):
                    if cad_count >= cad_target:
                        break
                    # Skip docs already written on a previous (resumed) attempt.
                    if written > cad_count:
                        cad_count += 1
                        continue
                    text = r.get("prompt") or r.get("description") or r.get("text") or ""
                    if text:
                        out.write(_escape(text) + "\n")
                        out.flush()
                        cad_count += 1
                        written += 1
                        if cad_count % 1000 == 0:
                            print(f"[tokenizer] CAD {cad_count:,}/{cad_target:,}...", flush=True)
                        if commit_fn and cad_count % commit_every == 0:
                            out.flush(); commit_fn()
                print(f"[tokenizer] CAD stream done ({cad_count:,} docs, target {cad_target:,}).", flush=True)

            # --- SlimPajama ---
            text_target = cad_target + n_docs
            if written < text_target:
                print(f"[tokenizer] Streaming SlimPajama docs ({text_target - written:,} remaining)...", flush=True)
                seen = 0
                for r in itertools.islice(slimpajama_stream(seed=seed), n_docs):
                    if written >= text_target:
                        break
                    seen += 1
                    if written - cad_target > seen:
                        continue
                    text = r.get("text", "")
                    if text:
                        out.write(_escape(text) + "\n")
                        out.flush()
                        written += 1
                        if written % 2000 == 0:
                            print(f"[tokenizer] SlimPajama total {written - cad_target:,}/{n_docs:,}...", flush=True)
                        if commit_fn and written % commit_every == 0:
                            out.flush(); commit_fn()
                print(f"[tokenizer] SlimPajama stream done ({written - cad_target:,} docs).", flush=True)

            # --- Generated role docs (cheap, local, always redo — no network cost) ---
            role_count = 0
            if n_role_docs:
                from quadmesh.pipeline.datasets.role_gen_data import iter_tokenizer_corpus
                for text in iter_tokenizer_corpus(n_role_docs, seed=seed):
                    out.write(_escape(text) + "\n")
                    role_count += 1
                written += role_count
                print(f"[tokenizer] Added {role_count:,} generated role documents.", flush=True)

        with open(done_marker, "w") as f:
            f.write(str(written))
        print(f"[tokenizer] Corpus complete: {written:,} docs written to {corpus_path}.", flush=True)
        return corpus_path

    @classmethod
    def train_from_corpus_file(cls, corpus_path: str, vocab_size: int = 32000) -> "QuadmeshTokenizer":
        """Trains BPE merges from an already-materialized corpus file (fast, local, no network — safe to
        retry on preemption since it just re-reads the file)."""
        tk = Tokenizer(models.BPE(unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIALS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        def _lines():
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.rstrip("\n").replace("\\n", "\n")

        print("[tokenizer] Calculating BPE merges from cached corpus file...", flush=True)
        tk.train_from_iterator(_lines(), trainer=trainer)
        print(f"[tokenizer] Training complete! Vocab size: {tk.get_vocab_size():,}", flush=True)
        return cls(tk)

    @classmethod
    def train_from_stream(cls, vocab_size: int = 32000, n_cad: int = 10_000,
                          n_docs: int = 35_000, n_role_docs: int = 12_000,
                          seed: int = 0, corpus_path: str = "/vol/tokenizer/_corpus.txt",
                          commit_fn=None) -> "QuadmeshTokenizer":
        cls.build_corpus_file(corpus_path, n_cad=n_cad, n_docs=n_docs, n_role_docs=n_role_docs,
                              seed=seed, commit_fn=commit_fn)
        if commit_fn:
            commit_fn()  # persist the finished corpus + .done marker before the slow BPE step
        return cls.train_from_corpus_file(corpus_path, vocab_size=vocab_size)


def get_or_train(path: str = "/vol/tokenizer/quadmesh-bpe.json",
                 vocab_size: int = 32000, commit_fn=None) -> QuadmeshTokenizer:
    if os.path.exists(path):
        tok = QuadmeshTokenizer.load(path)
        missing = [t for t in ROLE_TAGS.values() if tok.tk.token_to_id(t) is None]
        if missing:
            raise RuntimeError(
                f"{path} was built before the shared text model and has no role tags ({missing[0]} ...). "
                f"Delete it and rebuild:  modal run modal_app.py::build_tokenizer   (nothing trained on the old "
                f"tokenizer can be reused - token ids change).")
        return tok
    corpus_path = os.path.join(os.path.dirname(path), "_corpus.txt")
    tok = QuadmeshTokenizer.train_from_stream(vocab_size=vocab_size, corpus_path=corpus_path, commit_fn=commit_fn)
    tok.save(path)
    if commit_fn:
        commit_fn()
    return tok
