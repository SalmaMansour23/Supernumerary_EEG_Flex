"""
finger.py - driver for the supernumerary-finger firmware (USB-CDC serial).

The firmware moves the finger in calibrated STROKES (~11 strokes = full travel):
    f<n*10>  flex n strokes      u<n*10>  unflex n strokes
    U0       deep re-home (drives into the open hard stop)
    r        instant stop / reset
and prints "SEQ_DONE" once a command has completely finished.

The firmware has no absolute position sensor that survives a reset, so this class keeps a
software estimate `pos` (0 = fully open ... max_strokes = fully bent) and turns
"go to stroke N" requests into f/u commands of the right size.

Use port "SIMULATE" to test everything without hardware.
"""
import re
import threading
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:          # the GUI still starts; connecting to real hardware will explain why not
    serial = None
    list_ports = None

# Lines the firmware prints at 20-50 Hz that we never want in the log
_NOISE = re.compile(r"^(-?\d+,-?\d+,-?\d+,|CUR:|DBG:|=+$|Current,Target)")
# Lines that ARE worth showing in the log
_LOG_PREFIXES = ("ERROR", "PREEMPT", "RESET REASON", "Servo Controller",
                 "FLEX_TO: done", "UNFLEX: done", "UNFLEX: re-home")


def list_serial_ports():
    """Names of the serial ports currently available on this computer."""
    if list_ports is None:
        return []
    return [p.device for p in list_ports.comports()]


class FingerIO:
    def __init__(self, log=print, max_strokes=10):
        self.log = log
        self.max_strokes = max_strokes
        self.pos = 0                    # estimated stroke position, 0 = open
        self.current_a = 0.0            # latest motor current reported by the firmware
        self.connected = False
        self.sim = False                # True when running without hardware
        self.busy = False               # True while a command is running
        self._ser = None
        self._done = threading.Event()  # set when the firmware prints SEQ_DONE
        self._move_lock = threading.Lock()   # only one move at a time
        self._stop_flag = False
        self._quit = threading.Event()

    # ------------------------------------------------------------------ connection
    def connect(self, port, baud=115200):
        self.close()
        self._quit.clear()
        if port == "SIMULATE":
            self.sim = True
            self.connected = True
            self.pos = 0
            self.log("finger: SIMULATION mode (no hardware)")
            return
        if serial is None:
            raise RuntimeError("pyserial is not installed (pip install pyserial)")
        self.sim = False
        self._ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2.0)                 # the board may reboot when the port opens; let it boot
        self._ser.reset_input_buffer()
        self.connected = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        self.log(f"finger: connected on {port}")

    def close(self):
        self._quit.set()
        self.connected = False
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def _read_loop(self):
        """Background thread: read the firmware's lines (SEQ_DONE, current, messages)."""
        while not self._quit.is_set():
            try:
                line = self._ser.readline().decode(errors="ignore").strip()
            except Exception:
                self.connected = False
                self.log("finger: serial connection lost")
                return
            if not line:
                continue
            if line == "SEQ_DONE":                       # command finished
                self._done.set()
            elif line.startswith("CUR:"):                # motor current reading
                try:
                    self.current_a = float(line[4:].split(",")[0])
                except ValueError:
                    pass
            elif _NOISE.match(line):                     # high-rate debug lines: ignore
                continue
            elif line.startswith(_LOG_PREFIXES):         # useful messages: show in the log
                self.log("finger: " + line)

    # ------------------------------------------------------------------ low level
    def _send(self, cmd):
        """Send one command to the firmware (or fake it in simulation)."""
        self._done.clear()
        self.busy = True
        if self.sim:
            threading.Thread(target=self._sim_run, args=(cmd,), daemon=True).start()
        else:
            self._ser.write((cmd + "\n").encode())

    def _sim_run(self, cmd):
        """Simulation: pretend the command takes a realistic amount of time."""
        n = int(re.sub(r"\D", "", cmd) or 0)
        time.sleep(3.0 if cmd[0] == "U" else max(0.2, n / 10 * 0.6))
        self._done.set()

    def wait_done(self, timeout=90.0, abort=None):
        """Block until SEQ_DONE. Returns False on timeout / abort / lost connection."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._done.wait(0.05):
                self.busy = False
                return True
            if (abort is not None and abort.is_set()) or not self.connected:
                self.busy = False
                return False
        self.log("finger: timed out waiting for SEQ_DONE")
        self.busy = False
        return False

    # ------------------------------------------------------------------ public API
    def move_to(self, target, abort=None, overtravel=True):
        """Move to an absolute stroke position. Blocks until the firmware says SEQ_DONE.
        Returns True if the move completed normally."""
        if not self.connected:
            return False
        with self._move_lock:
            self._stop_flag = False
            target = max(0, min(self.max_strokes, int(round(target))))
            delta = target - self.pos
            if delta == 0:
                return True
            if delta > 0:
                cmd = f"f{delta * 10}"                      # bend more
            else:
                n = -delta
                if target == 0 and overtravel:
                    n = min(self.max_strokes, n + 1)        # one extra stroke presses the open stop = free re-home
                cmd = f"u{n * 10}"                          # release
            self._send(cmd)
            ok = self.wait_done(abort=abort)
            if self._stop_flag:
                self._stop_flag = False
                self.log("finger: stopped - position estimate uncertain, use Re-home")
                return False
            if ok:
                self.pos = target
            return ok

    def rehome(self, abort=None):
        """Deep re-home into the open hard stop; position becomes 0 (fully open)."""
        if not self.connected:
            return False
        with self._move_lock:
            self._stop_flag = False
            self.log("finger: re-homing (takes a few seconds)...")
            self._send("U0")
            ok = self.wait_done(abort=abort)
            if ok and not self._stop_flag:
                self.pos = 0
                self.log("finger: homed, position = 0 (open)")
            self._stop_flag = False
            return ok

    def stop(self):
        """Emergency stop: motor off immediately, any running command is abandoned."""
        if not self.connected:
            return
        self._stop_flag = True
        if self.sim:
            self._done.set()
        else:
            try:
                self._ser.write(b"r\n")
            except Exception:
                pass
        self.log("finger: STOP sent")