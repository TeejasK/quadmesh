"""
Quadmesh Agent — desktop app.

    python -m quadmesh.agent.app

Text box + mic for instructions, plus buttons to import a reference image or
an existing 3D model (STL/OBJ/FBX) straight into Blender. A log panel shows
what the agent is doing while you watch Blender do it on screen.
"""
from __future__ import annotations
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

from quadmesh.agent.planner import plan_from_text
from quadmesh.agent.loop import Step, run_plan
from quadmesh.agent import blender_client as bridge
from quadmesh.agent.launch import open_blender

# ---------------------------------------------------------------------------
# Palette — dark, neutral, professional. No default Tk look.
# ---------------------------------------------------------------------------
BG = "#1e1f26"
PANEL = "#262832"
FIELD = "#2d2f3a"
TEXT = "#e7e8ee"
MUTED = "#8a8d9a"
ACCENT = "#5b8cff"
ACCENT_DIM = "#3d5bb0"
OK = "#3ecf8e"
BAD = "#ef5b5b"

FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_MONO = ("Cascadia Mono", 10)
FONT_TITLE = ("Segoe UI Semibold", 13)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
MODEL_EXT = {".stl": "import_stl", ".obj": "import_obj", ".fbx": "import_fbx"}


class _LogRedirect:
    def __init__(self, out_queue: queue.Queue):
        self.out_queue = out_queue

    def write(self, text):
        if text.strip():
            self.out_queue.put(text)

    def flush(self):
        pass


class QuadmeshApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Quadmesh Agent")
        self.geometry("760x600")
        self.minsize(620, 460)
        self.configure(bg=BG)

        self.log_queue: queue.Queue = queue.Queue()
        self.attached_file: str | None = None

        self._build_style()
        self._build_ui()
        self._poll_log()
        self._refresh_status()

        self._log("Ready.", "muted")
        if not bridge.is_up():
            self._log("Blender bridge not detected — click \u201cOpen Blender\u201d to start it.", "muted")

    # -- styling ------------------------------------------------------------

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", font=FONT, background=FIELD, foreground=TEXT,
                        borderwidth=0, focusthickness=0, padding=8)
        style.map("TButton", background=[("active", ACCENT_DIM)])
        style.configure("Accent.TButton", background=ACCENT, foreground="#0d0f14",
                        font=FONT_BOLD, padding=8)
        style.map("Accent.TButton", background=[("active", "#7aa2ff")])
        style.configure("TEntry", fieldbackground=FIELD, foreground=TEXT,
                        borderwidth=0, insertcolor=TEXT, padding=8)

    def _build_ui(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=20, pady=(18, 6))
        tk.Label(header, text="Quadmesh Agent", font=FONT_TITLE, bg=BG, fg=TEXT).pack(side="left")

        self.status_dot = tk.Canvas(header, width=10, height=10, bg=BG, highlightthickness=0)
        self.status_dot.pack(side="right", padx=(0, 6))
        self._status_oval = self.status_dot.create_oval(1, 1, 9, 9, fill=BAD, outline="")
        tk.Label(header, text="Blender bridge", font=FONT, bg=BG, fg=MUTED).pack(side="right")

        # attachments row
        attach = tk.Frame(self, bg=BG)
        attach.pack(fill="x", padx=20, pady=(0, 10))
        ttk.Button(attach, text="\U0001F4C2  Import 3D model", command=self._pick_model).pack(side="left")
        ttk.Button(attach, text="\U0001F5BC  Import image", command=self._pick_image).pack(side="left", padx=8)
        ttk.Button(attach, text="Image \u2192 solid", command=self._on_image_solid).pack(side="left", padx=8)
        ttk.Button(attach, text="Open Blender", command=self._on_open_blender).pack(side="left", padx=8)
        self.attach_label = tk.Label(attach, text="No file attached", font=FONT, bg=BG, fg=MUTED)
        self.attach_label.pack(side="left", padx=10)

        # log panel
        log_frame = tk.Frame(self, bg=PANEL, highlightbackground="#333542",
                             highlightthickness=1)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        self.log = scrolledtext.ScrolledText(
            log_frame, font=FONT_MONO, bg=PANEL, fg=TEXT, insertbackground=TEXT,
            borderwidth=0, highlightthickness=0, wrap="word", state="disabled", padx=12, pady=10)
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("you", foreground=ACCENT, font=FONT_BOLD)
        self.log.tag_configure("muted", foreground=MUTED)
        self.log.tag_configure("ok", foreground=OK)
        self.log.tag_configure("bad", foreground=BAD)
        self.log.tag_configure("normal", foreground=TEXT)

        # input row
        input_row = tk.Frame(self, bg=BG)
        input_row.pack(fill="x", padx=20, pady=(0, 20))

        self.entry = ttk.Entry(input_row, font=FONT)
        self.entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.entry.bind("<Return>", lambda e: self._on_send())
        self.entry.focus_set()

        ttk.Button(input_row, text="\U0001F3A4", width=3, command=self._on_speak).pack(side="left", padx=(8, 0))
        ttk.Button(input_row, text="Send", style="Accent.TButton", command=self._on_send).pack(side="left", padx=(8, 0))
        self.desk_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(input_row, text="Desk mode", variable=self.desk_var).pack(side="left", padx=(8, 0))
        ttk.Button(input_row, text="Autopilot", command=self._on_autopilot).pack(side="left", padx=(8, 0))

    # -- status ---------------------------------------------------------

    def _refresh_status(self):
        up = bridge.is_up()
        self.status_dot.itemconfig(self._status_oval, fill=OK if up else BAD)
        self.after(3000, self._refresh_status)

    # -- log --------------------------------------------------------------

    def _log(self, text: str, tag: str = "normal"):
        self.log.configure(state="normal")
        self.log.insert("end", (text if text.endswith("\n") else text + "\n"), tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll_log(self):
        try:
            while True:
                self._log(self.log_queue.get_nowait(), "muted")
        except queue.Empty:
            pass
        self.after(150, self._poll_log)

    def _run_in_background(self, fn, *args):
        redirect = _LogRedirect(self.log_queue)

        def target():
            old_stdout = sys.stdout
            sys.stdout = redirect
            try:
                fn(*args)
            except Exception as e:
                self.log_queue.put(f"[error] {e}")
            finally:
                sys.stdout = old_stdout

        threading.Thread(target=target, daemon=True).start()

    # -- file attachments -----------------------------------------------

    def _pick_model(self):
        path = filedialog.askopenfilename(
            title="Import 3D model",
            filetypes=[("3D models", "*.stl *.obj *.fbx"), ("All files", "*.*")])
        if path:
            self.attached_file = path
            self.attach_label.config(text=os.path.basename(path), fg=TEXT)
            self._log(f"Attached model: {path}", "ok")
            ext = os.path.splitext(path)[1].lower()
            op = MODEL_EXT.get(ext)
            if op:
                self._run_in_background(self._import_model, op, path)
            else:
                self._log(f"Unsupported model type {ext!r}", "bad")

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title="Import reference image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")])
        if path:
            self.attached_file = path
            self.attach_label.config(text=os.path.basename(path), fg=TEXT)
            self._log(f"Attached image: {path}", "ok")
            self._run_in_background(self._import_image, path)

    def _import_model(self, op: str, path: str):
        print(f"[app] importing 3D model via {op} ...")
        run_plan([Step(f"Import {os.path.basename(path)}", native_op=op,
                       native_args={"filepath": path})], visible=False)

    def _import_image(self, path: str):
        print(f"[app] importing reference image ...")
        run_plan([Step(f"Import reference image {os.path.basename(path)}",
                       native_op="import_reference_image",
                       native_args={"filepath": path})], visible=False)

    # -- text / speech input ----------------------------------------------

    def _on_open_blender(self):
        self._log("you> open blender", "you")
        self._run_in_background(open_blender)

    def _on_send(self):
        prompt = self.entry.get().strip()
        if not prompt:
            return
        self.entry.delete(0, "end")
        self._log(f"you> {prompt}", "you")
        self._handle_prompt(prompt)

    def _on_speak(self):
        self._log("you> [listening...]", "you")
        self._run_in_background(self._speak_and_handle)

    def _speak_and_handle(self):
        from quadmesh.agent.speech import listen
        text = listen()
        if not text:
            print("[app] didn't catch that, try again")
            return
        print(f"you> {text}")
        self._handle_prompt(text)

    def _handle_prompt(self, prompt: str):
        if "open blender" in prompt.lower() or "start blender" in prompt.lower():
            self._run_in_background(open_blender)
            return

        def do_plan():
            steps = plan_from_text(prompt)
            if not steps:
                print("[app] nothing to do with that instruction, try again.")
                return
            run_plan(steps)
            print("[app] done. what next?")

        self._run_in_background(do_plan)

    # -- autopilot / image -> solid ---------------------------------------------------
    def _on_autopilot(self):
        prompt = self.entry.get().strip()
        if not prompt:
            return
        self.entry.delete(0, "end")
        desk = self.desk_var.get()
        self._log(f"you> [autopilot{' + desk mode' if desk else ''}] {prompt}", "you")

        def work():
            import re
            from quadmesh.agent.autopilot import design, design_assembly
            if re.search(r"hexa|robot|arm", prompt.lower()):
                r = design_assembly(prompt, desk=desk)
            else:
                r = design(prompt, desk=desk)
            print(f"PRINTABLE: {getattr(r, 'stl_path', None) or getattr(r, 'json_path', '')}" if r.ok
                  else "NOT PRINTABLE - " + "; ".join(str(p) for p in (r.problems or ["see log"])))
        self._run_in_background(work)

    def _on_image_solid(self):
        from tkinter import simpledialog
        path = filedialog.askopenfilename(title="Drawing / silhouette / logo",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if not path:
            return
        w = simpledialog.askfloat("Width", "Width of the part in mm:", parent=self, minvalue=1.0)
        t = simpledialog.askfloat("Thickness", "Thickness in mm:", parent=self, minvalue=0.5)
        if not w or not t:
            return
        desk = self.desk_var.get()
        self._log(f"you> [image -> solid] {os.path.basename(path)}  {w} mm wide, {t} mm thick", "you")

        def work():
            from quadmesh.agent.image3d import design_from_image
            r = design_from_image(path, width_mm=w, thickness_mm=t, desk=desk)
            print(f"PRINTABLE: {r.stl_path}" if r.ok else "NOT PRINTABLE - " + "; ".join(r.problems))
        self._run_in_background(work)


if __name__ == "__main__":
    QuadmeshApp().mainloop()
