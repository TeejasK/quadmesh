"""
QuadmeshChat agent - the assistant, now a real (small) chat model instead of only fixed rules.

Understands: typed text, spoken voice (your own ASR), a photo or drawing (traced or guessed into 3D), and an existing
3D file (STL/STEP - read, reported on, and checked). It can hold an ordinary conversation, answer from its knowledge
(quadmesh/chat_kb.py), and call tools to actually build things. Every tool call is validated before it runs - the model
proposing "CALL {...}" never bypasses plan_checker / mesh_tools validation / the printability gate.

Full system control: pass full_control=True (or --full-control) to allow ANY key, hotkey and a long-drag "sculpt" stroke,
for moving objects with the mouse, sculpting, and anything else Blender's UI needs - not just boxes/cylinders/spheres.
pyautogui's screen-corner fail-safe is ALWAYS on, restricted mode or not.

Falls back to the fixed-rule Assistant (assistant.py) and the knowledge base when no chat checkpoint is trained yet, so
this file works today and gets smarter once quadmesh.train.chat_train has run.

    python -m quadmesh.agent.chat_agent --desk --full-control --voice --speak
"""
from __future__ import annotations
import argparse
import io
import json
import os
import re
from typing import Callable, Optional

MAX_HISTORY_CHARS = 4000


class ChatAgent:
    def __init__(self, tier: Optional[str] = None, device: Optional[str] = None, ckpt_root: Optional[str] = None,
                desk=None, out_dir: str = "designs", speak: Optional[Callable] = None, full_control: bool = False, log=print):
        self.tier, self.device, self.ckpt_root = tier, device, ckpt_root
        self.desk, self.out_dir, self.speak, self.log = desk, out_dir, speak, log
        self.history = ""
        self.last_result = None
        self._model = None
        if full_control:
            from quadmesh.agent.actions import set_unrestricted
            set_unrestricted(True, audit_log=os.path.join(out_dir, "action_audit.log"))
            log("[chat] full system control ENABLED - every key/hotkey is allowed; the mouse-corner fail-safe still stops everything.")

    # ---- model ------------------------------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        from quadmesh.config import get_tier
        from quadmesh.data.tokenizer import get_or_train
        from quadmesh.model.chat import QuadmeshChat
        from quadmesh.role_shapes import role_shape
        from quadmesh.agent.planner_model import resolve_device
        root = self.ckpt_root or os.environ.get("QUADMESH_CKPT_ROOT")
        tier_name = (self.tier or os.environ.get("QUADMESH_TIER", "100M")).replace("Quadmesh-", "")
        dev = resolve_device(self.device)
        path = f"{root}/Quadmesh-{tier_name}/live/chat.pt" if root else None
        if not root or not os.path.exists(path):
            self._model = False
            return False
        cfg = get_tier(tier_name)
        tok = get_or_train(f"{root}/tokenizer/quadmesh-bpe.json")
        shape = role_shape(cfg, "chat"); shape.vocab_size = tok.vocab_size
        m = QuadmeshChat(shape)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        self._model = (m.to(dev).eval(), tok, dev)
        return self._model

    # ---- perception: text / voice / image / 3D file ------------------------------------------------
    def hear(self) -> str:
        from quadmesh.agent.speech import listen
        text = listen()
        return text or ""

    def read_image(self, path: str):
        from PIL import Image
        return Image.open(path).convert("RGB")

    def read_3d_file(self, path: str) -> str:
        """STL or STEP -> a short report (never guessed - real measurements)."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".stl":
            from quadmesh.agent.printability import analyze_stl
            r = analyze_stl(path)
            return (f"That STL has {r['triangles']} triangles, is {r['dims'][0]:.1f} x {r['dims'][1]:.1f} x {r['dims'][2]:.1f} mm, "
                   f"volume {r['volume']:.1f} mm^3, {r['loose_parts']} solid(s), {r['open_edges']} open edges"
                   + (", and it is watertight." if r["open_edges"] == 0 and r["nonmanifold_edges"] == 0 else ", so it is NOT watertight."))
        if ext in (".step", ".stp"):
            from quadmesh.step_io import step_to_mesh
            _, facts = step_to_mesh(path)
            return f"That STEP file has {facts['solids']} solid(s), {facts['faces']} faces, size {facts['size_mm']} mm, volume {facts['volume_mm3']:.1f} mm^3."
        return f"I don't read {ext} files - give me .stl or .step."

    # ---- speaking ----------------------------------------------------------------------------------
    def _say(self, text: str) -> str:
        self.log(f"[quadmesh] {text}")
        if self.speak:
            self.speak(text)
        return text

    def speak_own_voice(self, text: str, tier: Optional[str] = None) -> Optional[str]:
        """Synthesise with Quadmesh's OWN trained voice (falls back to the OS voice via self.speak if not trained)."""
        try:
            import torch
            from quadmesh.config import get_tier
            from quadmesh.data.tokenizer import get_or_train
            from quadmesh.model.audio import TTSModel
            from quadmesh.role_shapes import role_shape
            from quadmesh.tts_synth import mel_to_wav
            from quadmesh.tts_style import cute_female
            root = self.ckpt_root or os.environ.get("QUADMESH_CKPT_ROOT")
            tier_name = (tier or self.tier or os.environ.get("QUADMESH_TIER", "100M")).replace("Quadmesh-", "")
            path = f"{root}/Quadmesh-{tier_name}/live/tts.pt"
            if not root or not os.path.exists(path):
                return None
            cfg = get_tier(tier_name)
            tok = get_or_train(f"{root}/tokenizer/quadmesh-bpe.json")
            shape = role_shape(cfg, "tts"); shape.vocab_size = tok.vocab_size
            m = TTSModel(shape, tok.vocab_size)
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
            ids = torch.tensor([tok.encode(text[:400])])
            mel = m.infer(ids, max_frames=800)[0].numpy()
            wav_path = os.path.join(self.out_dir, "speech.wav")
            mel_to_wav(mel, wav_path)
            import wave
            w = wave.open(wav_path); import numpy as np
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32767
            x = cute_female(x, w.getframerate())
            from quadmesh.tts_synth import write_wav
            write_wav(wav_path, x, w.getframerate())
            self._play(wav_path)
            return wav_path
        except Exception as e:
            self.log(f"[chat] own voice unavailable ({type(e).__name__}: {e}); using the fallback voice")
            return None

    def _play(self, path: str):
        # winsound is Python's own stdlib module on Windows - zero install, no C++ build tools needed.
        # Tried FIRST now (previously simpleaudio was tried first, which needs to compile a C extension and
        # fails on Windows without Microsoft's MSVC Build Tools installed - exactly the error that prompted
        # this fix). simpleaudio is now only the fallback, for non-Windows dev environments where winsound
        # doesn't exist at all.
        try:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
            return
        except ImportError:
            pass   # not on Windows - fall through to simpleaudio below
        except Exception:
            pass   # winsound exists but playback failed for some other reason - still try the fallback
        try:
            import simpleaudio as sa
            sa.WaveObject.from_wave_file(path).play()
        except Exception:
            pass   # no working audio playback available - speech output still works, just silently, via the .wav file already written to disk

    # ---- the turn ------------------------------------------------------------------------------------
    def turn(self, text: str = "", image_path: Optional[str] = None, file_path: Optional[str] = None) -> str:
        context = ""
        if file_path:
            context = self.read_3d_file(file_path)
            text = (text + " " if text else "") + f"[attached file: {os.path.basename(file_path)}]"
        image = self.read_image(image_path) if image_path else None
        if not text.strip() and not image:
            return ""

        from quadmesh.agent.shared_runtime import get_runtime
        rt = get_runtime(self.ckpt_root, self.tier, self.device)
        if rt is not None:
            reply = self._shared_turn(rt, text, image, context)
        elif self._load():
            reply = self._model_turn(text, image, context)
        elif os.environ.get("QUADMESH_ALLOW_RULE_FALLBACK") == "1":
            reply = self._rule_turn(text, image_path, context)
        else:
            reply = ("No trained model is loaded, so I can't answer yet. Set QUADMESH_CKPT_ROOT to your checkpoints "
                     "folder (needs Quadmesh-<tier>/live/shared_text.pt).")
        return self._say(reply)

    def _shared_turn(self, rt, text: str, image, context: str) -> str:
        """Chat through the shared generative model. The model reads the request and decides whether to talk or to
        emit a CALL {...}; a photo is announced to it as context because this trunk reads text only."""
        ctx = "IMAGE: attached photo" if image is not None else ""
        ctx = (ctx + " " + context).strip()
        prompt = (f"CONTEXT: {ctx}\n" if ctx else "") + self.history[-MAX_HISTORY_CHARS:] + f"USER: {text}\nASSISTANT:"
        out = rt.chat(prompt)
        call = self._parse_call(out)
        self.history += f"USER: {text}\nASSISTANT: {out}\n"
        return self._execute(call, image) if call else out

    def _model_turn(self, text: str, image, context: str) -> str:
        import torch
        m, tok, dev = self._load()
        prompt = (f"CONTEXT: {context}\n" if context else "") + self.history[-MAX_HISTORY_CHARS:] + f"USER: {text}\nASSISTANT:"
        ids = torch.tensor([tok.encode(prompt)], device=dev)
        patches = None
        if image is not None:
            from quadmesh.data.vla_data import to_patches
            patches = torch.from_numpy(to_patches(image))[None].to(dev)
        out = tok.decode(m.generate_reply(ids, patches, eos_id=tok.eos_id))
        call = self._parse_call(out)
        if call:
            result = self._execute(call, image)
            self.history += f"USER: {text}\nASSISTANT: {out}\n"
            return result
        self.history += f"USER: {text}\nASSISTANT: {out}\n"
        return out.strip()

    def _rule_turn(self, text: str, image_path: Optional[str], context: str) -> str:
        """No chat checkpoint yet: knowledge base + the same build/move/rotate/360 rules as assistant.Assistant."""
        from quadmesh.chat_kb import kb_answer
        if context:
            return context
        if image_path and re.search(r"\b(3d|solid|print|model|make|turn (?:it|this) into)\b", text.lower()):
            return self._execute({"tool": "photo_to_3d", "width_mm": 60}, self.read_image(image_path))
        ans = kb_answer(text)
        if ans:
            return ans
        from quadmesh.agent.assistant import Assistant
        a = Assistant(desk=self.desk, out_dir=self.out_dir, log=self.log,
                     build_fn=lambda t: self._execute({"tool": "build_part", "request": t}, None))
        return a.handle(text) or ("I don't have a trained chat model yet, and I couldn't match that to something I know. "
                                  "Try: build/move/rotate/'show 360', or ask about materials, printing or gears.")

    # ---- tool execution (validated, never trusts the model blindly) --------------------------------
    def _parse_call(self, text: str) -> Optional[dict]:
        from quadmesh.chat_data import parse_call
        return parse_call(text)

    def _execute(self, call: dict, image) -> str:
        tool = call.get("tool")
        try:
            if tool == "make_shape":
                from quadmesh.mesh_tools import make_shape, finish
                kind = call.get("kind")
                params = {k: v for k, v in call.items() if k not in ("tool", "kind")}
                meshes, notes = make_shape(kind, **params)
                res = finish(kind, meshes, self.out_dir, kind=kind)
                self.last_result = res
                if not res["ok"]:
                    return "That did not pass the printability checks (" + "; ".join(res["problems"][:2]) + "), so I did not save it as printable."
                return f"Made the {kind}. " + " ".join(notes) + f" Saved: {', '.join(os.path.basename(f) for f in res['files'])}."
            if tool == "build_part":
                from quadmesh.agent.autopilot import design
                r = design(call.get("request", ""), tier=self.tier, device=self.device, out_dir=self.out_dir, watch=False)
                self.last_result = r
                return f"Built it and it passed every check: {r.stl_path}." if r.ok else "It did not pass: " + "; ".join(str(p) for p in r.problems[:2])
            if tool == "build_assembly":
                from quadmesh.agent.autopilot import design_assembly
                r = design_assembly(call.get("request", ""), tier=self.tier, device=self.device, out_dir=self.out_dir, watch=False)
                self.last_result = r
                return (f"Built {', '.join(r.parts)} - every part passed. See {r.json_path}." if r.ok else
                        "I couldn't build that: " + "; ".join(str(w) for w in (r.warnings or ["not a supported assembly"])[:2]))
            if tool == "photo_to_3d":
                if image is None:
                    return "I need an attached photo for that."
                from quadmesh.photo3d import guess_from_photo
                import tempfile
                p = os.path.join(tempfile.mkdtemp(), "photo.png"); image.save(p)
                mesh, rep = guess_from_photo(p, width_mm=float(call.get("width_mm", 60)))
                from quadmesh.geometry3d import write_stl
                path = write_stl(os.path.join(self.out_dir, "photo_guess.stl"), mesh)
                return f"My best guess ({rep['mode']}): {rep['note']} Saved to {path}. " + " ".join(rep["warnings"])
            if tool in ("move", "rotate", "show_360"):
                from quadmesh.agent.assistant import Assistant
                a = Assistant(desk=self.desk, out_dir=self.out_dir, log=self.log)
                if tool == "move":
                    return a.handle(f"move {call.get('name','body')} {call.get('distance',10)} mm {call.get('direction','right')}")
                if tool == "rotate":
                    return a.handle(f"rotate {call.get('name','body')} {call.get('deg',30)} degrees")
                return a.handle("show 360")
            if tool == "export_step":
                if not self.last_result or not getattr(self.last_result, "plan", None):
                    return "I don't have a recent plan-based part to export as STEP."
                from quadmesh.step_io import plan_to_step
                paths, warn, _ = plan_to_step(self.last_result.plan, os.path.join(self.out_dir, "part.step"))
                return f"Exported {paths[0]}." + (" Note: " + "; ".join(warn) if warn else "")
            return f"I don't have a tool called {tool!r}."
        except Exception as e:
            return f"I couldn't do that: {e}"


def make_speaker(agent: ChatAgent):
    try:
        import pyttsx3
        eng = pyttsx3.init()

        def speak(t):
            if agent.speak_own_voice(t) is None:
                eng.say(t); eng.runAndWait()
        return speak
    except Exception:
        return lambda t: agent.speak_own_voice(t)


def main():
    ap = argparse.ArgumentParser(description="Talk to Quadmesh (chat model)")
    ap.add_argument("--voice", action="store_true")
    ap.add_argument("--speak", action="store_true")
    ap.add_argument("--desk", action="store_true")
    ap.add_argument("--full-control", action="store_true", help="allow every key/hotkey and mouse-drag sculpting, not just boxes/cylinders")
    ap.add_argument("--tier", default=None)
    ap.add_argument("--out", default="designs")
    ap.add_argument("--image", default=None, help="attach an image to the first message")
    ap.add_argument("--file", default=None, help="attach an STL/STEP file to the first message")
    a = ap.parse_args()
    desk = None
    if a.desk:
        from quadmesh.agent.launch import ensure_blender
        ensure_blender()
        from quadmesh.agent.desk import DeskController
        desk = DeskController(); desk.setup()
    agent = ChatAgent(tier=a.tier, out_dir=a.out, desk=desk, full_control=a.full_control)
    if a.speak:
        agent.speak = make_speaker(agent)
    print("Quadmesh is listening. Type or say 'quit' to stop.")
    first = True
    while True:
        if a.voice:
            text = agent.hear()
            if not text:
                continue
            print(f"you> {text}")
        else:
            text = input("you> ")
        if text.strip().lower() in ("quit", "exit", "stop", "bye"):
            break
        agent.turn(text, image_path=a.image if first else None, file_path=a.file if first else None)
        first = False


if __name__ == "__main__":
    main()
