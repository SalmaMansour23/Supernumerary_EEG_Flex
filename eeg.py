"""
eeg.py - everything EEG: the data stream, the ERD analysis, and the trial / calibration runner.

Layout of this file
    1. EEGSettings   all numbers the analysis and the trials use (filled in by control.py)
    2. Sources       LSLSource (real g.tec Unicorn via LSL) and SimSource (fixed fake signal)
    3. EEGSession    keeps the stream flowing and records the samples of a running trial
    4. analyze_trial the ERD pipeline, step by step, keeping every intermediate number
    5. format_result prints those numbers as a table
    6. EEGController runs calibration and trials and moves the finger

ERD pipeline (per channel, no ML, no channel rejection - everything is calculated as it is):
    STEP 1  raw signal
    STEP 2  baseline correction : subtract the channel's mean over the baseline period
    STEP 3  CAR                 : subtract the average of all channels at every sample
    STEP 4  band-pass 8-30 Hz   : zero-phase Butterworth
    STEP 5  band power          : power = filtered signal squared, averaged over baseline / task
    STEP 6  ERD%                : (P_task - P_baseline) / P_baseline * 100   (negative = ERD)
    STEP 7  verdict             : ERD% <= threshold in >= min_frac of the sliding windows
    ERD on ANY channel -> finger flexes.
"""
import datetime
import os
import threading
import time
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt


# =========================================================================== 1. settings
@dataclass
class EEGSettings:
    # timing (seconds)
    baseline: float        # reference period before GO
    task: float            # movement period after GO
    pad: float             # GET READY / STOP padding (protects the filter from edge effects, then discarded)
    skip: float            # seconds after GO ignored (reaction time)
    win: float             # sliding window length for the ERD time course
    step: float            # sliding window step
    calib_pause: float     # break between the rest trial and the movement trial (calibration)
    # detection
    band: tuple            # band-pass (low, high) Hz
    filter_order: int
    threshold: float       # ERD% at or below this counts as desynchronization
    min_frac: float        # fraction of task windows that must be below the threshold
    use_car: bool
    # finger response
    flex_strokes: int
    hold: float
    # saving
    save_trials: bool
    save_dir: str


# =========================================================================== 2. sources
class SimSource:
    """Fake EEG: a FIXED signal, the same all the time (DC offset + 10 Hz sine per channel).
    It does nothing interesting - it only lets you run the whole pipeline without hardware."""

    def __init__(self, n_ch, fs=250.0):
        self.fs, self.n_ch, self.names = fs, n_ch, []
        self.n = 0                                  # samples produced so far
        self.t0 = time.time()
        k = np.arange(n_ch)
        self.dc = 40.0 + 10.0 * k                   # fixed DC offset per channel (uV)
        self.amp = 5.0 + 1.0 * k                    # fixed 10 Hz amplitude per channel (uV), different per channel

    def read(self):
        """Return the samples that 'arrived' since the last call, shape (n, n_ch), or None."""
        due = int((time.time() - self.t0) * self.fs) - self.n
        if due <= 0:
            return None
        t = (self.n + np.arange(due)) / self.fs
        self.n += due
        return self.dc + self.amp * np.sin(2 * np.pi * 10.0 * t)[:, None]

    def close(self):
        pass


class LSLSource:
    """Real EEG: pulls samples from a Lab Streaming Layer stream (g.tec Unicorn)."""

    def __init__(self, stream_type="EEG", stream_name=None, timeout=10.0):
        from pylsl import StreamInlet, resolve_byprop     # imported here so simulate mode needs no pylsl
        prop, val = ("name", stream_name) if stream_name else ("type", stream_type)
        found = resolve_byprop(prop, val, timeout=timeout)
        if not found:
            raise RuntimeError("No LSL stream found. Is the Unicorn software streaming to LSL?")
        info = found[0]
        self.inlet = StreamInlet(info, max_buflen=60)
        self.fs = float(info.nominal_srate())
        self.n_ch = info.channel_count()
        self.names = []                                    # channel labels announced by the stream (may be empty)
        try:
            ch = info.desc().child("channels").child("channel")
            for _ in range(self.n_ch):
                self.names.append(ch.child_value("label") or "")
                ch = ch.next_sibling()
        except Exception:
            pass

    def read(self):
        chunk, ts = self.inlet.pull_chunk(timeout=0.0, max_samples=1024)
        return np.asarray(chunk, dtype=float) if ts else None

    def close(self):
        self.inlet.close_stream()


# =========================================================================== 3. session
class TrialRun:
    """Collects the samples of one trial. Phase changes are driven by the SAMPLE COUNT (not the clock):
        GET READY (pad) -> BASELINE -> GO (task) -> STOP (pad)"""

    def __init__(self, s, fs):
        pre, nb, nt = int(s.pad * fs), int(s.baseline * fs), int(s.task * fs)
        self.phases = [("GET READY", pre), ("BASELINE", nb), ("GO", nt), ("STOP", pre)]
        self.total = pre + nb + nt + pre
        self.chunks, self.n = [], 0

    def progress(self):
        """(phase name, samples done in this phase, phase length)"""
        acc = 0
        for name, length in self.phases:
            if self.n < acc + length:
                return name, self.n - acc, length
            acc += length
        return "STOP", 1, 1

    @property
    def done(self):
        return self.n >= self.total

    def segment(self):
        """The recorded trial as (n_channels, total_samples)."""
        return np.vstack(self.chunks).T[:, :self.total]


class EEGSession:
    def __init__(self, log=print):
        self.log = log
        self.src = None
        self.fs = 250.0
        self.names = []
        self.connected = False
        self.simulated = False
        self.rate = 0.0                        # measured samples per second
        self.trial = None                      # the TrialRun being recorded (or None)
        self._lock = threading.Lock()
        self._quit = threading.Event()

    def connect(self, stream_type, stream_name, names, simulate=False):
        self.close()
        # 1. open the source (fake or real)
        src = SimSource(len(names)) if simulate else LSLSource(stream_type, stream_name)
        if src.n_ch < len(names):
            src.close()
            raise RuntimeError(f"Stream has {src.n_ch} columns but {len(names)} channel names were given")
        self.src, self.fs, self.names, self.simulated = src, src.fs, list(names), simulate
        # 2. start the reader thread (one quit-event per connection, so an old thread can never revive)
        self._quit = threading.Event()
        self.connected = True
        threading.Thread(target=self._reader, args=(src, self._quit), daemon=True).start()
        self.log(f"eeg: connected{' (SIMULATED fixed signal)' if simulate else ''}, "
                 f"{len(names)} channels @ {self.fs:g} Hz: {', '.join(names)}")

    def close(self):
        self._quit.set()
        self.connected = False
        if self.src:
            try:
                self.src.close()
            except Exception:
                pass
            self.src = None

    def _reader(self, src, quit_evt):
        """Background thread: move incoming samples into the running trial (if any)."""
        n_ch = len(self.names)
        count, t_mark = 0, time.time()
        while not quit_evt.is_set():
            chunk = src.read()
            if chunk is None:
                time.sleep(0.005)
                continue
            chunk = chunk[:, :n_ch]                    # the first N columns are the EEG channels
            with self._lock:
                if self.trial is not None:
                    self.trial.chunks.append(chunk)
                    self.trial.n += len(chunk)
            count += len(chunk)
            now = time.time()
            if now - t_mark >= 1.0:                    # update the measured sample rate once a second
                self.rate = count / (now - t_mark)
                count, t_mark = 0, now


# =========================================================================== 4. analysis
def _rms(a):
    """Root mean square along the time axis. Works for (n_ch, n) and (n,) arrays."""
    return np.sqrt((a ** 2).mean(axis=-1))


def analyze_trial(seg, fs, names, s):
    """seg: (n_ch, samples) including the pads. Returns a dict with EVERY intermediate value."""
    pre, nb, nt = int(s.pad * fs), int(s.baseline * fs), int(s.task * fs)
    total = pre + nb + nt + pre
    skip, w, st = int(s.skip * fs), int(s.win * fs), max(1, int(s.step * fs))
    if seg.shape[1] < total:
        return dict(valid=False, reason=f"too few samples ({seg.shape[1]} < {total})")
    if nt - skip < w:
        return dict(valid=False, reason="task time minus skip time is shorter than the window")

    base = slice(pre, pre + nb)                          # baseline samples
    task = slice(pre + nb + skip, pre + nb + nt)         # task samples (reaction-time part skipped)
    raw = seg[:, :total].astype(float)

    # STEP 1: raw signal (just measure it)
    raw_b, raw_t = _rms(raw[:, base]), _rms(raw[:, task])

    # STEP 2: baseline correction - remove each channel's mean over the baseline
    bl_removed = raw[:, base].mean(axis=1)
    x = raw - bl_removed[:, None]
    bl_b, bl_t = _rms(x[:, base]), _rms(x[:, task])

    # STEP 3: CAR - remove the average of ALL channels at every sample
    car_sig = x.mean(axis=0)
    if s.use_car:
        x = x - car_sig
    car_b, car_t = _rms(x[:, base]), _rms(x[:, task])
    car_info = dict(on=s.use_car, mean_b=car_sig[base].mean(), rms_b=_rms(car_sig[base]), rms_t=_rms(car_sig[task]))

    # STEP 4: band-pass filter (forward + backward = zero phase)
    sos = butter(s.filter_order, list(s.band), btype="band", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x, axis=1)
    band_b, band_t = _rms(xf[:, base]), _rms(xf[:, task])

    # STEP 5: band power = filtered signal squared, averaged over baseline and over the task period
    p = xf ** 2
    p_b, p_t = p[:, base].mean(axis=1), p[:, task].mean(axis=1)

    with np.errstate(all="ignore"):                      # flat channels give 0/0 -> nan; keep it, do not reject
        # STEP 6: ERD% over the whole task period (and in dB)
        erd = (p_t - p_b) / p_b * 100.0
        db = 10.0 * np.log10(p_t / p_b)
        # sliding-window ERD time course (moving average of power via cumulative sum)
        c = np.concatenate([np.zeros((len(p), 1)), np.cumsum(p[:, task], axis=1)], axis=1)
        win_pow = ((c[:, w:] - c[:, :-w]) / w)[:, ::st]
        erd_win = (win_pow - p_b[:, None]) / p_b[:, None] * 100.0
        frac = (erd_win <= s.threshold).mean(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # nanmin warns on all-nan rows (flat channel)
        erd_min = np.nanmin(erd_win, axis=1)

    # STEP 7: verdict per channel
    rows = []
    for i, name in enumerate(names):
        detected = bool(np.isfinite(erd[i]) and erd[i] <= s.threshold and frac[i] >= s.min_frac)
        rows.append(dict(name=name, bl_removed=bl_removed[i], raw_b=raw_b[i], raw_t=raw_t[i],
                         bl_b=bl_b[i], bl_t=bl_t[i], car_b=car_b[i], car_t=car_t[i],
                         band_b=band_b[i], band_t=band_t[i], p_b=p_b[i], p_t=p_t[i],
                         erd=erd[i], db=db[i], erd_min=erd_min[i], frac=frac[i], detected=detected))
    det = sorted([r for r in rows if r["detected"]], key=lambda r: r["erd"])   # strongest first
    return dict(valid=True, reason="", fs=fs, rows=rows, car=car_info, detected=bool(det),
                det_channels=[r["name"] for r in det], n_base=nb, n_task=nt - skip, n_win=erd_win.shape[1])


# =========================================================================== 5. table output
def format_result(res, s, title):
    """Turn an analysis result into a readable table (raw -> final)."""
    L = [f"==== {title} " + "=" * 70]
    if not res["valid"]:
        L.append(f"  TRIAL REJECTED: {res['reason']}")
        return "\n".join(L)
    L.append(f"  fs {res['fs']:g} Hz | baseline {s.baseline:g} s ({res['n_base']} samples) | "
             f"task window {s.skip:g}-{s.task:g} s after GO ({res['n_task']} samples) | "
             f"band {s.band[0]:g}-{s.band[1]:g} Hz")
    L.append(f"  ERD threshold {s.threshold:g}% in >= {s.min_frac * 100:.0f}% of {res['n_win']} windows "
             f"({s.win:g} s window, {s.step:g} s step) | CAR {'on' if s.use_car else 'off'}")

    # ---- table 1: steps 1-3
    L += ["", "  STEPS 1-3  raw -> baseline correction -> CAR     (uV; rmsB = RMS in baseline, rmsT = RMS in task window)",
          f"  {'ch':<5}{'BL removed':>11}{'raw rmsB':>10}{'raw rmsT':>10}{'BLcor rmsB':>11}"
          f"{'BLcor rmsT':>11}{'CAR rmsB':>10}{'CAR rmsT':>10}"]
    for r in res["rows"]:
        L.append(f"  {r['name']:<5}{r['bl_removed']:>11.2f}{r['raw_b']:>10.2f}{r['raw_t']:>10.2f}{r['bl_b']:>11.2f}"
                 f"{r['bl_t']:>11.2f}{r['car_b']:>10.2f}{r['car_t']:>10.2f}")
    c = res["car"]
    if c["on"]:
        L.append(f"  CAR signal subtracted from every channel: mean {c['mean_b']:.3f} uV, "
                 f"rms {c['rms_b']:.2f} uV (baseline) / {c['rms_t']:.2f} uV (task)")
    else:
        L.append("  CAR is OFF (nothing subtracted)")

    # ---- table 2: steps 4-7
    L += ["", "  STEPS 4-7  band-pass -> power -> ERD -> verdict",
          f"  {'ch':<5}{'bandB uV':>9}{'bandT uV':>9}{'P base uV2':>12}{'P task uV2':>12}{'ERD %':>9}"
          f"{'dB':>8}{'min win %':>11}{'win<thr':>9}  verdict"]
    for r in res["rows"]:
        L.append(f"  {r['name']:<5}{r['band_b']:>9.2f}{r['band_t']:>9.2f}{r['p_b']:>12.2f}{r['p_t']:>12.2f}"
                 f"{r['erd']:>9.1f}{r['db']:>8.2f}{r['erd_min']:>11.1f}{r['frac'] * 100:>8.0f}%  "
                 f"{'ERD' if r['detected'] else '-'}")

    # ---- final verdict
    L.append("")
    if res["detected"]:
        by_name = {r["name"]: r for r in res["rows"]}
        found = ", ".join(f"{n} ({by_name[n]['erd']:.0f}%)" for n in res["det_channels"])
        L.append(f"  VERDICT: ERD DETECTED at {found}  (strongest: {res['det_channels'][0]})")
    else:
        L.append("  VERDICT: NO ERD on any channel")
    return "\n".join(L)


# =========================================================================== 6. controller
class EEGController:
    """Runs calibration and trials in a background thread and drives the finger."""

    def __init__(self, eeg, finger, settings, log=print, on_result=None, on_calibrated=None):
        self.eeg, self.finger, self.s, self.log = eeg, finger, settings, log
        self.on_result = on_result or (lambda text: None)        # gets tables / action lines for the trace box
        self.on_calibrated = on_calibrated or (lambda thr: None)  # gets the new threshold (%)
        self.state = "idle"        # idle | trial | calib_wait | analyzing | flexing | holding | unflexing
        self.kind = "run"          # what the running trial is: run | rest | move
        self.phase = ("GET READY", 0, 1)
        self.hold_left = 0.0
        self.counts = dict(trials=0, triggered=0)
        self._abort = threading.Event()
        self._thread = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def abort(self):
        self._abort.set()

    def note(self, text):
        """Show a short action line both in the log and in the trace box."""
        self.log("eeg: " + text)
        self.on_result("  >> " + text)

    def _start(self, target):
        if self.running:
            return
        self._abort.clear()
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def start_trial(self):
        self._start(self._trial)

    def start_calibration(self):
        self._start(self._calibrate)

    # ------------------------------------------------------------------ helpers
    def _record(self):
        """Record one trial (GET READY / BASELINE / GO / STOP). Returns None if aborted."""
        eeg = self.eeg
        trial = TrialRun(self.s, eeg.fs)
        with eeg._lock:
            eeg.trial = trial
        self.state = "trial"
        while not trial.done:
            if self._abort.is_set():
                eeg.trial = None
                self.log("eeg: aborted")
                return None
            self.phase = trial.progress()
            time.sleep(0.02)
        eeg.trial = None
        return trial

    def _save(self, name, seg, **extra):
        if self.s.save_trials:
            os.makedirs(self.s.save_dir, exist_ok=True)
            fn = os.path.join(self.s.save_dir, f"{name}_{datetime.datetime.now():%H%M%S}.npz")
            np.savez(fn, seg=seg, fs=self.eeg.fs, names=self.eeg.names, **extra)

    def _reset(self):
        self.eeg.trial = None
        self.kind = "run"
        self.state = "idle"

    # ------------------------------------------------------------------ calibration
    def _calibrate(self):
        """Rest trial + movement trial -> personal ERD threshold (most negative extra ERD on any channel)."""
        s, eeg = self.s, self.eeg
        try:
            erds = {}                                            # {"rest": {ch: ERD%}, "move": {...}}
            for kind in ("rest", "move"):
                if kind == "move":                               # short break before the movement trial
                    self.state, self.kind = "calib_wait", "move"
                    t_end = time.time() + s.calib_pause
                    while time.time() < t_end:
                        if self._abort.is_set():
                            self.log("eeg: calibration aborted")
                            return
                        self.hold_left = t_end - time.time()
                        time.sleep(0.05)
                self.kind = kind
                trial = self._record()
                if trial is None:
                    return
                self.state = "analyzing"
                seg = trial.segment()
                res = analyze_trial(seg, eeg.fs, eeg.names, s)
                self.on_result(format_result(res, s, f"CALIBRATION {kind.upper()} TRIAL"))
                self._save(f"calib_{kind}", seg)
                if not res["valid"]:
                    self.log(f"eeg: calibration FAILED ({res['reason']}). Threshold unchanged.")
                    return
                erds[kind] = {r["name"]: r["erd"] for r in res["rows"] if np.isfinite(r["erd"])}

            # Extra ERD caused by the movement on top of any ERD already present at rest
            # (rest ERD > 0 is synchronization, not desynchronization -> counts as 0).
            extra = {c: erds["move"][c] - min(erds["rest"][c], 0.0) for c in erds["move"] if c in erds["rest"]}
            if not extra:
                self.log("eeg: calibration FAILED (no channel gave a usable ERD in both trials). Threshold unchanged.")
                return
            best = min(extra, key=extra.get)                     # channel with the biggest extra decrease
            lines = ["==== CALIBRATION SUMMARY " + "=" * 60,
                     f"  {'ch':<5}{'rest ERD %':>12}{'move ERD %':>12}{'extra ERD %':>13}"]
            for c in extra:
                lines.append(f"  {c:<5}{erds['rest'][c]:>12.1f}{erds['move'][c]:>12.1f}{extra[c]:>13.1f}"
                             f"{'   <- strongest' if c == best else ''}")
            thr = extra[best]
            if thr > -5.0:
                lines.append(f"  FAILED: only {thr:.1f}% extra ERD at {best}. Check electrodes / squeeze harder / retry. "
                             "Threshold unchanged.")
                self.on_result("\n".join(lines))
                self.log("eeg: calibration FAILED (movement produced almost no extra ERD)")
                return
            thr = float(max(-80.0, min(-5.0, round(thr, 1))))
            s.threshold = thr
            lines.append(f"  Personal ERD threshold = {thr:.1f}%  (from {best})")
            self.on_result("\n".join(lines))
            self.log(f"eeg: calibration done, ERD threshold = {thr:.1f}% (from {best})")
            self.on_calibrated(thr)
        except Exception as e:
            self.log(f"eeg: calibration failed: {e!r}")
        finally:
            self._reset()

    # ------------------------------------------------------------------ one experiment trial
    def _trial(self):
        s, eeg = self.s, self.eeg
        try:
            # 1. record GET READY / BASELINE / GO / STOP
            self.kind = "run"
            trial = self._record()
            if trial is None:
                return
            # 2. analyze (all intermediate values go to the trace box)
            self.state = "analyzing"
            seg = trial.segment()
            res = analyze_trial(seg, eeg.fs, eeg.names, s)
            self.counts["trials"] += 1
            self.on_result(format_result(res, s, f"TRIAL {self.counts['trials']}"))
            triggered = bool(res["valid"] and res["detected"])
            self._save(f"trial{self.counts['trials']:03d}", seg, detected=triggered)
            if not triggered:
                self.note("no ERD -> finger stays neutral")
                return
            # 3. ERD on any channel -> flex, hold, unflex
            self.counts["triggered"] += 1
            try:
                self.state = "flexing"
                self.note(f"FLEXING finger to {s.flex_strokes} strokes")
                self.finger.move_to(s.flex_strokes)
                self.state = "holding"
                self.note(f"HOLDING for {s.hold:g} s")
                t_end = time.time() + s.hold
                while time.time() < t_end:
                    self.hold_left = t_end - time.time()
                    time.sleep(0.05)
            finally:
                # always come back to neutral, even if something went wrong during the hold
                self.state = "unflexing"
                self.note("UNFLEXING finger to neutral")
                self.finger.move_to(0)
                self.note("finger back at neutral")
        except Exception as e:
            self.log(f"eeg: trial failed: {e!r}")
        finally:
            self._reset()