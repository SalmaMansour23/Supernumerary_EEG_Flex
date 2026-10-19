#!/usr/bin/env python3
"""
control.py - control center for the supernumerary finger.   Run:  python control.py

    1. Connect the finger (USB serial).
    2. Pick a control mode:
         Flex sensor : the finger follows the flex sensor proportionally.
         EEG (ERD)   : calibrate (optional) -> baseline -> GO -> ERD analysis
                       -> ERD on ANY channel = finger flexes, holds, unflexes.

Files: control.py (this GUI + all settings), finger.py, flex.py, eeg.py
Needs: pyserial (always); numpy, scipy (EEG mode); pylsl (real EEG only, not for SIMULATE).
"""
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import messagebox, ttk

from finger import FingerIO, list_serial_ports
from flex import FlexController, FlexSource, MAX_ANGLE, angle_to_strokes

try:
    from eeg import EEGController, EEGSession, EEGSettings
    EEG_IMPORT_ERROR = None
except Exception as exc:                     # missing numpy/scipy: flex mode still works
    EEG_IMPORT_ERROR = exc

# =====================================================================================
#  SETTINGS - change every knob HERE, nothing else needs to be touched
# =====================================================================================
# --- EEG trial timing (seconds)
BASELINE_S = 3.0          # reference period before GO
TASK_S = 2.0              # movement period after GO
PAD_S = 1.0               # GET READY and STOP padding (protects the filter edges, then discarded)
SKIP_S = 0.5              # time after GO ignored when calculating ERD (reaction time)
WINDOW_S = 0.5            # sliding window length for the ERD time course
STEP_S = 0.1              # sliding window step
CALIB_PAUSE_S = 5.0       # break between the rest trial and the movement trial in calibration

# --- ERD detection
BAND_HZ = (8.0, 30.0)     # band-pass range
FILTER_ORDER = 4          # Butterworth order
ERD_THRESHOLD = -15.0     # ERD % at or below this = desynchronization (calibration can overwrite it)
MIN_FRAC = 0.5            # fraction of task windows that must be below the threshold
USE_CAR = True            # common average reference on/off

# --- finger response after a detected ERD
FLEX_STROKES = 6          # how far the finger flexes
HOLD_S = 3.0              # how long it holds before releasing
MAX_STROKES = 10          # finger travel limit (1..11)

# --- EEG device (always the same g.tec Unicorn)
LSL_TYPE = "EEG"
LSL_NAME = None           # e.g. "UnicornLSL"; None = find the stream by type
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

# --- saving
SAVE_TRIALS = True        # save every trial as .npz
SAVE_DIR = "eeg_trials"
# =====================================================================================

BG = "#f4f5f7"
CUES = {   # key -> (text, colour)
    "idle": ("Idle", "#6c757d"),
    "GET READY": ("1 · GET READY", "#e0a800"),
    "BASELINE": ("2 · BASELINE", "#17a2b8"),
    "GO": ("3 · GO", "#28a745"),
    "STOP": ("4 · STOP", "#6c757d"),
    "analyzing": ("ANALYZING", "#6f42c1"),
    "flexing": ("FLEXING", "#d9534f"),
    "holding": ("HOLDING", "#d9534f"),
    "unflexing": ("UNFLEXING", "#fd7e14"),
}
KIND_LABEL = {"rest": "CALIBRATION - REST", "move": "CALIBRATION - MOVE", "run": ""}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Supernumerary Finger - Control Center")
        root.geometry("1100x820")
        root.configure(bg=BG)
        ttk.Style().theme_use("clam")
        self.logq = queue.Queue()          # messages from background threads -> GUI
        self.fs_win = None                 # the black full-screen cue window (None = not shown)
        self.fs_label = None
        self.fs_text = None

        # ---- the objects that do the work
        self.finger = FingerIO(log=self.log, max_strokes=MAX_STROKES)
        self.flex_src = FlexSource(log=self.log)
        self.flex_ctl = FlexController(self.flex_src, self.finger, log=self.log)
        self.eeg = self.eeg_ctl = self.settings = None
        if EEG_IMPORT_ERROR is None:
            self.settings = EEGSettings(
                baseline=BASELINE_S, task=TASK_S, pad=PAD_S, skip=SKIP_S, win=WINDOW_S, step=STEP_S,
                calib_pause=CALIB_PAUSE_S, band=BAND_HZ, filter_order=FILTER_ORDER, threshold=ERD_THRESHOLD,
                min_frac=MIN_FRAC, use_car=USE_CAR, flex_strokes=FLEX_STROKES, hold=HOLD_S,
                save_trials=SAVE_TRIALS, save_dir=SAVE_DIR)
            self.eeg = EEGSession(log=self.log)
            self.eeg_ctl = EEGController(self.eeg, self.finger, self.settings, log=self.log,
                                         on_result=self.show_result,
                                         on_calibrated=lambda thr: self.logq.put(("CAL", thr)))

        # ---- the screen
        self._build_finger_panel()
        self.page_host = tk.Frame(root, bg=BG)
        self.page_host.pack(fill="both", expand=True, padx=10)
        self._build_home()
        self._build_flex_page()
        self._build_eeg_page()
        self._build_log()
        self.show_page("home")
        self.tick()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ plumbing
    def log(self, msg):
        self.logq.put(f"{time.strftime('%H:%M:%S')}  {msg}\n")

    def bg(self, fn, *a):
        """Run something in a background thread so the GUI never freezes."""
        threading.Thread(target=fn, args=a, daemon=True).start()

    def show_result(self, text):
        self.logq.put(("RESULT", text))

    # ------------------------------------------------------------------ 1. finger panel
    def _build_finger_panel(self):
        f = ttk.LabelFrame(self.root, text=" 1 · Finger (ESP32-S3, USB serial) ")
        f.pack(fill="x", padx=10, pady=(10, 4))
        r1 = tk.Frame(f)
        r1.pack(fill="x", padx=6, pady=4)
        ttk.Label(r1, text="Port:").pack(side="left")
        self.port_var = tk.StringVar()
        self.port_cb = ttk.Combobox(r1, textvariable=self.port_var, width=22)
        self.port_cb.pack(side="left", padx=4)
        ttk.Button(r1, text="Refresh", command=self.refresh_ports).pack(side="left")
        ttk.Button(r1, text="Connect", command=self.connect_finger).pack(side="left", padx=6)
        self.finger_status = ttk.Label(r1, text="not connected", foreground="#a94442")
        self.finger_status.pack(side="left", padx=8)
        ttk.Label(r1, text="Max strokes:").pack(side="right")
        self.max_var = tk.IntVar(value=MAX_STROKES)
        ttk.Spinbox(r1, from_=1, to=11, width=4, textvariable=self.max_var,
                    command=lambda: setattr(self.finger, "max_strokes", self.max_var.get())).pack(side="right", padx=4)
        r2 = tk.Frame(f)
        r2.pack(fill="x", padx=6, pady=(0, 6))
        self.fcanvas = tk.Canvas(r2, width=360, height=26, bg="white", highlightthickness=1,
                                 highlightbackground="#bbb")
        self.fcanvas.pack(side="left")
        self.fpos_lbl = ttk.Label(r2, text="open", width=24)
        self.fpos_lbl.pack(side="left", padx=8)
        ttk.Button(r2, text="Re-home (open)", command=lambda: self.bg(self.finger.rehome)).pack(side="left", padx=3)
        ttk.Button(r2, text="+1 stroke", command=lambda: self.bg(self.finger.move_to, self.finger.pos + 1)).pack(side="left", padx=3)
        ttk.Button(r2, text="-1 stroke", command=lambda: self.bg(self.finger.move_to, self.finger.pos - 1, None, False)).pack(side="left", padx=3)
        tk.Button(r2, text="STOP", bg="#d9534f", fg="white", font=("Arial", 10, "bold"),
                  command=self.emergency_stop).pack(side="right", padx=3)
        self.refresh_ports()

    def refresh_ports(self):
        ports = list_serial_ports() + ["SIMULATE"]
        self.port_cb["values"] = ports
        if not self.port_var.get():
            self.port_var.set(ports[0])
        if hasattr(self, "flex_port_cb"):               # flex page is built after the finger panel
            self.flex_port_cb["values"] = ports
            if not self.flex_port.get():
                self.flex_port.set(ports[0])

    def connect_finger(self):
        port = self.port_var.get().strip()
        self.finger.max_strokes = self.max_var.get()

        def work():
            try:
                self.finger.connect(port)
            except Exception as e:
                self.log(f"finger: connect failed: {e}")
        self.bg(work)

    def emergency_stop(self):
        self.flex_ctl.stop()
        if self.eeg_ctl:
            self.eeg_ctl.abort()
        self.finger.stop()
        self.log("EMERGENCY STOP - controllers halted. Re-home before continuing.")

    # ------------------------------------------------------------------ 2. pages
    def show_page(self, name):
        for p in (self.home, self.flex_page, self.eeg_page):
            p.pack_forget()
        {"home": self.home, "flex": self.flex_page, "eeg": self.eeg_page}[name].pack(fill="both", expand=True)

    def back_home(self):
        self.flex_ctl.stop()
        if self.eeg_ctl:
            self.eeg_ctl.abort()
        self.show_page("home")

    def _build_home(self):
        self.home = tk.Frame(self.page_host, bg=BG)
        ttk.Label(self.home, text="2 · How do you want to control the finger?",
                  font=("Arial", 14, "bold")).pack(pady=(24, 14))
        row = tk.Frame(self.home, bg=BG)
        row.pack()
        opts = [("Flex sensor", "Finger mirrors how far the flex sensor is bent,\nproportionally, in both directions.",
                 "#2d6cdf", "flex"),
                ("EEG (ERD)", "Baseline -> GO.\nERD on any channel = finger flexes, holds, then releases.",
                 "#7b3fe4", "eeg")]
        for title, desc, color, page in opts:
            card = tk.Frame(row, bg="white", highlightbackground="#ccc", highlightthickness=1)
            card.pack(side="left", padx=14, ipadx=10, ipady=10)
            tk.Button(card, text=title, bg=color, fg="white", font=("Arial", 15, "bold"), width=14, height=2,
                      command=lambda p=page: self.show_page(p)).pack(padx=14, pady=(14, 8))
            tk.Label(card, text=desc, bg="white", justify="center").pack(padx=10, pady=(0, 10))
        if EEG_IMPORT_ERROR is not None:
            ttk.Label(self.home, foreground="#a94442",
                      text=f"EEG mode unavailable: {EEG_IMPORT_ERROR}\n"
                           "(pip install numpy scipy pylsl)").pack(pady=16)

    # ------------------------------------------------------------------ flex page
    def _build_flex_page(self):
        p = self.flex_page = tk.Frame(self.page_host, bg=BG)
        top = tk.Frame(p, bg=BG)
        top.pack(fill="x", pady=(10, 4))
        ttk.Button(top, text="< Change mode", command=self.back_home).pack(side="left")
        ttk.Label(top, text="Flex sensor control", font=("Arial", 13, "bold")).pack(side="left", padx=14)

        f = ttk.LabelFrame(p, text=" Sensor connection (flex-sensor ESP32, USB serial) ")
        f.pack(fill="x", pady=6)
        r = tk.Frame(f)
        r.pack(fill="x", padx=6, pady=6)
        ttk.Label(r, text="Port:").pack(side="left")
        self.flex_port = tk.StringVar()
        self.flex_port_cb = ttk.Combobox(r, textvariable=self.flex_port, width=22)
        self.flex_port_cb.pack(side="left", padx=4)
        ttk.Button(r, text="Refresh", command=self.refresh_ports).pack(side="left")
        ttk.Label(r, text="(SIMULATE = test with the slider)").pack(side="left", padx=4)
        ttk.Button(r, text="Connect sensor", command=self.connect_flex).pack(side="left", padx=6)
        self.refresh_ports()
        self.flex_status = ttk.Label(r, text="not connected", foreground="#a94442")
        self.flex_status.pack(side="left", padx=8)

        m = ttk.LabelFrame(p, text=" Live mapping: sensor angle -> finger strokes ")
        m.pack(fill="x", pady=6)
        self.flex_canvas = tk.Canvas(m, width=600, height=34, bg="white", highlightthickness=1,
                                     highlightbackground="#bbb")
        self.flex_canvas.pack(padx=8, pady=(8, 2), anchor="w")
        self.flex_lbl = ttk.Label(m, text="sensor: --", font=("Arial", 11))
        self.flex_lbl.pack(anchor="w", padx=8, pady=(0, 4))
        sim = tk.Frame(m)
        sim.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(sim, text="Simulated sensor angle:").pack(side="left")
        ttk.Scale(sim, from_=0, to=MAX_ANGLE, length=300, command=self.on_sim_slider).pack(side="left", padx=8)

        c = tk.Frame(p, bg=BG)
        c.pack(pady=14)
        self.flex_btn = tk.Button(c, text="START following sensor", bg="#28a745", fg="white",
                                  font=("Arial", 13, "bold"), width=26, height=2, command=self.toggle_flex)
        self.flex_btn.pack()
        ttk.Label(p, foreground="#555", justify="center",
                  text="Sensor angle 0..60 deg maps to 0..max strokes. If the sensor unflexes, the finger unflexes by the\n"
                       "same proportion. Tip: press 'Re-home' once at the start so the finger position estimate is exact.").pack()

    def on_sim_slider(self, value):
        """Slider drives the flex angle, but only in SIMULATE mode (steps of 10 deg)."""
        if self.flex_src.connected and self.flex_port.get().strip().upper() == "SIMULATE":
            self.flex_src.set_simulated(round(float(value) / 10) * 10)

    def connect_flex(self):
        port = self.flex_port.get().strip()
        if port.upper() != "SIMULATE" and self.finger.connected and not self.finger.sim and \
                port == self.port_var.get().strip():
            messagebox.showwarning("Port", "That port is the finger's. Pick the flex sensor's own port.")
            return
        self.flex_src.start(port)

    def toggle_flex(self):
        if self.flex_ctl.running:
            self.flex_ctl.stop()
            return
        if not self.finger.connected:
            messagebox.showwarning("Finger", "Connect the finger first (panel 1).")
            return
        if not self.flex_src.connected:
            messagebox.showwarning("Sensor", "Connect the flex sensor first.")
            return
        self.flex_ctl.start()

    # ------------------------------------------------------------------ EEG page
    def _build_eeg_page(self):
        p = self.eeg_page = tk.Frame(self.page_host, bg=BG)
        top = tk.Frame(p, bg=BG)
        top.pack(fill="x", pady=(10, 4))
        ttk.Button(top, text="< Change mode", command=self.back_home).pack(side="left")
        ttk.Label(top, text="EEG (ERD) control", font=("Arial", 13, "bold")).pack(side="left", padx=14)
        if self.eeg is None:
            ttk.Label(p, text="EEG mode is unavailable (see home screen).", foreground="#a94442").pack(pady=20)
            return

        # ---- A. connect
        f = ttk.LabelFrame(p, text=" A · EEG (g.tec Unicorn via LSL) ")
        f.pack(fill="x", pady=4)
        r = tk.Frame(f)
        r.pack(fill="x", padx=6, pady=6)
        ttk.Button(r, text="Connect EEG", command=self.connect_eeg).pack(side="left", padx=(6, 8))
        self.sim_eeg = tk.BooleanVar(value=False)
        ttk.Checkbutton(r, text="Simulate (fixed fake signal)", variable=self.sim_eeg).pack(side="left", padx=8)
        self.eeg_status = ttk.Label(r, text="not connected", foreground="#a94442")
        self.eeg_status.pack(side="left", padx=8)

        # ---- B. trial settings (defaults come from the SETTINGS block at the top of this file)
        s = ttk.LabelFrame(p, text=" B · Trial settings ")
        s.pack(fill="x", pady=4)
        g = tk.Frame(s)
        g.pack(fill="x", padx=6, pady=6)
        st = self.settings
        self.v = dict(baseline=tk.DoubleVar(value=st.baseline), task=tk.DoubleVar(value=st.task),
                      threshold=tk.DoubleVar(value=st.threshold), flex_strokes=tk.IntVar(value=st.flex_strokes),
                      hold=tk.DoubleVar(value=st.hold), save=tk.BooleanVar(value=st.save_trials))
        spec = [("Baseline (s)", "baseline", 1, 10), ("Task (s)", "task", 1, 10), ("ERD thr (%)", "threshold", -100, 100),
                ("Finger flex (strokes)", "flex_strokes", 1, 11), ("Hold (s)", "hold", 0, 30)]
        for i, (lbl, key, lo, hi) in enumerate(spec):
            ttk.Label(g, text=lbl).grid(row=i // 3, column=(i % 3) * 2, sticky="e", padx=(8, 3), pady=2)
            ttk.Spinbox(g, from_=lo, to=hi, width=6, textvariable=self.v[key]).grid(row=i // 3, column=(i % 3) * 2 + 1, sticky="w")
        ttk.Checkbutton(g, text="Save trials (.npz)", variable=self.v["save"]).grid(row=1, column=5, sticky="w", padx=8)
        self.thr_note = ttk.Label(g, text="ERD threshold: default (no calibration yet) - editable", foreground="#555")
        self.thr_note.grid(row=2, column=0, columnspan=6, sticky="w", padx=8, pady=(2, 0))

        # ---- C. calibration / trial
        c = ttk.LabelFrame(p, text=" C · Calibration  /  Trial ")
        c.pack(fill="both", expand=True, pady=4)
        row = tk.Frame(c)
        row.pack(fill="x", padx=6, pady=6)
        self.calib_btn = tk.Button(row, text="CALIBRATE (rest + move)", bg="#e08a1e", fg="white",
                                   font=("Arial", 12, "bold"), width=24, command=self.start_calibration)
        self.calib_btn.pack(side="left")
        self.start_btn = tk.Button(row, text="START TRIAL", bg="#7b3fe4", fg="white", font=("Arial", 12, "bold"),
                                   width=16, command=self.start_trial)
        self.start_btn.pack(side="left", padx=(10, 0))
        ttk.Button(row, text="Abort", command=lambda: self.eeg_ctl.abort()).pack(side="left", padx=8)
        self.count_lbl = ttk.Label(row, text="trials: 0   triggered: 0")
        self.count_lbl.pack(side="right")
        self.cue = tk.Label(c, text="Idle", bg="#6c757d", fg="white", font=("Arial", 20, "bold"), height=2)
        self.cue.pack(fill="x", padx=8)
        self.pbar = ttk.Progressbar(c, maximum=1.0)
        self.pbar.pack(fill="x", padx=8, pady=6)

        # trace box: the full tables (raw -> final) and the flex / unflex lines
        box = tk.Frame(c)
        box.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        self.result = tk.Text(box, height=14, font=("Courier", 9), bg="#fbfbfb", wrap="none")
        ysb = ttk.Scrollbar(box, orient="vertical", command=self.result.yview)
        xsb = ttk.Scrollbar(box, orient="horizontal", command=self.result.xview)
        self.result.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")
        self.result.pack(side="left", fill="both", expand=True)

    def connect_eeg(self):
        sim = self.sim_eeg.get()           # read the checkbox here (main thread), not in the worker

        def work():
            self.log("eeg: " + ("starting simulation" if sim else "looking for LSL stream (up to 10 s)..."))
            try:
                self.eeg.connect(LSL_TYPE, LSL_NAME, CHANNELS, simulate=sim)
            except Exception as e:
                self.log(f"eeg: connect failed: {e}")
        self.bg(work)

    def _push_settings(self, trial=True):
        """Copy the GUI boxes into the settings object right before a run."""
        v, s = self.v, self.settings
        s.baseline, s.task, s.save_trials = v["baseline"].get(), v["task"].get(), v["save"].get()
        if trial:
            s.threshold, s.flex_strokes, s.hold = v["threshold"].get(), v["flex_strokes"].get(), v["hold"].get()

    def start_calibration(self):
        if not self.eeg.connected:
            messagebox.showwarning("EEG", "Connect the EEG first.")
            return
        if self.eeg_ctl.running:
            return
        self._push_settings(trial=False)
        self.result.delete("1.0", "end")
        self.eeg_ctl.start_calibration()

    def start_trial(self):
        if not self.eeg.connected:
            messagebox.showwarning("EEG", "Connect the EEG first.")
            return
        if not self.finger.connected:
            messagebox.showwarning("Finger", "Connect the finger first (panel 1).")
            return
        if self.eeg_ctl.running:
            return
        self._push_settings()
        self.result.delete("1.0", "end")
        self.eeg_ctl.start_trial()

    # ------------------------------------------------------------------ log
    def _build_log(self):
        f = ttk.LabelFrame(self.root, text=" Log ")
        f.pack(fill="x", padx=10, pady=(4, 10))
        self.logbox = tk.Text(f, height=6, font=("Courier", 9), bg="#fbfbfb")
        self.logbox.pack(fill="x", padx=4, pady=4)

    # ------------------------------------------------------------------ periodic refresh (10x per second)
    def tick(self):
        """Refresh loop. ALWAYS reschedules itself, even if something inside raises - otherwise one
        error would silently kill the loop and freeze the black cue screen on its last text ('+')."""
        try:
            self._tick()
        except Exception as e:
            traceback.print_exc()
            self.log(f"GUI refresh error: {e!r}")
        finally:
            self.root.after(100, self.tick)

    def _tick(self):
        # 1. messages from background threads
        try:
            while True:
                item = self.logq.get_nowait()
                if isinstance(item, tuple):
                    if item[0] == "CAL":                # calibration finished: fill the editable threshold box
                        self.v["threshold"].set(item[1])
                        self.thr_note.config(text=f"ERD threshold: {item[1]:.1f} % from calibration - editable",
                                             foreground="#2d7a2d")
                    elif self.eeg:                      # ("RESULT", text) -> trace box
                        self.result.insert("end", item[1] + "\n")
                        self.result.see("end")
                else:
                    self.logbox.insert("end", item)
                    self.logbox.see("end")
        except queue.Empty:
            pass

        # 2. finger bar
        fg = self.finger
        self.finger_status.config(text="connected" + (" (SIMULATED)" if fg.sim else "") if fg.connected else "not connected",
                                  foreground="#2d7a2d" if fg.connected else "#a94442")
        self._bar(self.fcanvas, 360, 26, fg.pos / max(1, fg.max_strokes), "#2d6cdf")
        self.fpos_lbl.config(text=f"{fg.pos}/{fg.max_strokes} strokes" + ("  (moving)" if fg.busy else "") +
                                  (f"  {fg.current_a:.2f} A" if not fg.sim else ""))

        # 3. flex page
        fs = self.flex_src
        self.flex_status.config(text="connected" if fs.connected else "not connected",
                                foreground="#2d7a2d" if fs.connected else "#a94442")
        a = fs.angle
        if a is not None:
            self._bar(self.flex_canvas, 600, 34, a / MAX_ANGLE, "#e08a1e")
            self.flex_lbl.config(text=f"sensor: {a} deg  ->  target {angle_to_strokes(a, fg.max_strokes)} strokes"
                                      f"   (finger at {fg.pos})")
        self.flex_btn.config(text="STOP following" if self.flex_ctl.running else "START following sensor",
                             bg="#d9534f" if self.flex_ctl.running else "#28a745")

        # 4. EEG page. Each part is isolated so a failure in one cannot stop the other.
        if self.eeg is not None:
            e = self.eeg
            self.eeg_status.config(text=(f"connected{' (SIMULATED)' if e.simulated else ''}, {e.rate:.0f} Hz")
                                   if e.connected else "not connected",
                                   foreground="#2d7a2d" if e.connected else "#a94442")
            try:
                self.update_cue()
            except Exception as ex:
                traceback.print_exc()
                self.log(f"cue error: {ex!r}")
            try:
                self.update_fullscreen()
            except Exception as ex:
                traceback.print_exc()
                self.log(f"full-screen error: {ex!r}")

    @staticmethod
    def _bar(canvas, w, h, frac, color):
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w * max(0.0, min(1.0, frac)), h, fill=color, width=0)

    def _phase(self):
        """Safely read (name, elapsed, length) from the controller's current phase."""
        try:
            name, el, ln = self.eeg_ctl.phase
            return str(name), float(el), max(1e-6, float(ln))
        except Exception:
            return None, 0.0, 1.0

    def update_cue(self):
        """Big coloured cue: just the step number and a short label."""
        ctl = self.eeg_ctl
        st = ctl.state
        frac = 0.0
        if st == "trial":
            name, el, ln = self._phase()
            text, color = CUES.get(name, (str(name), "#6c757d"))
            frac = el / ln
            kind = KIND_LABEL.get(ctl.kind, "")
            if kind:
                text = kind + "\n" + text
        elif st == "calib_wait":
            text, color = f"BREAK before movement trial\n{ctl.hold_left:.0f} s", "#e0a800"
        else:
            text, color = CUES.get(st, CUES["idle"])
            if st == "holding":
                text += f" ({ctl.hold_left:.1f} s)"
        self.cue.config(text=text, bg=color)
        self.pbar["value"] = min(1.0, max(0.0, frac))
        self.count_lbl.config(text=f"trials: {ctl.counts['trials']}   triggered: {ctl.counts['triggered']}")
        busy = "disabled" if ctl.running else "normal"
        self.start_btn.config(state=busy)
        self.calib_btn.config(state=busy)

    # ------------------------------------------------------------------ black full-screen cue screen
    def fullscreen_text(self):
        """What the black screen shows right now. None = no black screen (back to the normal window).
            +                      fixation (GET READY and STOP periods)
            Baseline / Rest / Go   the baseline and GO periods
            Break Before Movement  the pause inside calibration"""
        ctl = self.eeg_ctl
        if not ctl.running:
            return None                                   # session over -> normal screen
        if ctl.state == "calib_wait":
            return "Break Before Movement"
        if ctl.kind == "run" and ctl.state != "trial":
            return None                                   # actual trial: black screen only while recording
        if ctl.state != "trial":
            return "+"                                    # calibration, between phases (analysis takes ~0 s)
        name, _, _ = self._phase()
        name = (name or "").upper()
        if name == "BASELINE":
            return "Baseline"
        if name == "GO":
            return "Rest" if ctl.kind == "rest" else "Flex"  # calibration rest trial says Rest, all others Flex
        return "+"                                        # GET READY, STOP, or unknown

    def update_fullscreen(self):
        """Open / update / close the black full-screen window (called 10x per second)."""
        text = self.fullscreen_text()
        # 1. nothing to show -> close the black screen
        if text is None:
            if self.fs_win is not None:
                self.fs_win.destroy()
                self.fs_win = self.fs_label = self.fs_text = None
            return
        # 2. open the black screen if it is not open yet
        if self.fs_win is None:
            w = tk.Toplevel(self.root)
            w.configure(bg="black", cursor="none")
            w.attributes("-fullscreen", True)
            w.attributes("-topmost", True)
            w.bind("<Escape>", lambda e: self.eeg_ctl.abort())      # Esc = abort the session
            self.fs_label = tk.Label(w, bg="black", fg="white")
            self.fs_label.place(relx=0.5, rely=0.5, anchor="center")
            self.fs_win, self.fs_text = w, None
            w.focus_force()
        # 3. change the text only when it changed
        if text != self.fs_text:
            font = ("Arial", 100) if text == "+" else ("Arial", 60, "bold")
            self.fs_label.config(text=text, font=font)
            self.fs_text = text

    def on_close(self):
        self.flex_ctl.stop()
        if self.eeg_ctl:
            self.eeg_ctl.abort()
        self.flex_src.stop()
        if self.eeg:
            self.eeg.close()
        self.finger.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()