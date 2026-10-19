"""
flex.py - flex-sensor mode: the finger follows a flex sensor proportionally.

The flex-sensor ESP32 sends one line over USB serial whenever its angle changes:
    "f0", "f20", "f30", "f40", "f50", "f60"
FlexController maps that angle onto finger strokes:
    angle 0 -> 0 strokes (open)  ...  angle 60 -> max_strokes (fully bent)

Port "SIMULATE" skips the hardware; drive the angle with set_simulated().
"""
import re
import threading
import time

try:
    import serial
except ImportError:          # the GUI still starts; connecting to the real sensor will explain why not
    serial = None

MAX_ANGLE = 60                       # largest angle the sensor reports (degrees)
_LINE = re.compile(r"^f(\d+)$")      # matches "f30" etc.


def angle_to_strokes(angle, max_strokes, max_angle=MAX_ANGLE):
    """Proportional mapping: sensor angle (0..60) -> finger strokes (0..max_strokes)."""
    angle = max(0, min(max_angle, angle))
    return int(round(angle / max_angle * max_strokes))


class FlexSource:
    """Reads the sensor angle from the serial port in a background thread."""

    def __init__(self, log=print):
        self.log = log
        self.angle = None            # None until the first value arrives
        self.connected = False
        self._quit = threading.Event()
        self._thread = None

    def start(self, port, baud=115200):
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)       # let the old thread release the serial port
        self._quit.clear()
        if port.strip().upper() == "SIMULATE":
            self.connected = True
            self.angle = 0
            self.log("flex: SIMULATION mode (use the slider)")
            return
        if serial is None:
            self.log("flex: pyserial is not installed (pip install pyserial)")
            return
        self.angle = None
        self._thread = threading.Thread(target=self._run, args=(port.strip(), baud), daemon=True)
        self._thread.start()

    def stop(self):
        self._quit.set()
        self.connected = False

    def set_simulated(self, angle):
        self.angle = int(angle)

    def _run(self, port, baud):
        """Open the port, read angle lines, reconnect automatically if the cable is pulled."""
        while not self._quit.is_set():
            # 1. open the port
            try:
                ser = serial.Serial(port, baud, timeout=0.2)
            except Exception as e:
                self.log(f"flex: cannot open {port} ({e}); is another program using it? retrying in 3 s...")
                self._quit.wait(3.0)
                continue
            try:
                # 2. wait for the board to boot, then ask it for the current angle
                self._quit.wait(2.0)
                ser.reset_input_buffer()
                ser.write(b"?\n")
                self.connected = True
                self.log(f"flex: connected on {port}")
                # 3. read lines until stopped
                while not self._quit.is_set():
                    line = ser.readline().decode(errors="ignore").strip()
                    m = _LINE.match(line)
                    if m:
                        self.angle = int(m.group(1))
            except Exception as e:                   # cable pulled, port vanished, ...
                self.log(f"flex: serial error: {e}")
            finally:
                self.connected = False
                try:
                    ser.close()
                except Exception:
                    pass
                if not self._quit.is_set():
                    self.log("flex: connection lost, retrying...")
                    self._quit.wait(2.0)


class FlexController:
    """Background loop: keep the finger at the stroke position matching the sensor angle."""

    def __init__(self, source, finger, log=print):
        self.source, self.finger, self.log = source, finger, log
        self.target = 0
        self.running = False
        self._stop = threading.Event()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        self.log("flex mode: finger now follows the sensor")

    def stop(self):
        self._stop.set()
        self.running = False

    def _loop(self):
        while not self._stop.is_set():
            a = self.source.angle
            if a is not None and self.finger.connected:
                self.target = angle_to_strokes(a, self.finger.max_strokes)
                if self.target != self.finger.pos:
                    # Blocks until the move is done. The newest sensor value is read afterwards,
                    # so intermediate positions are skipped (newest wins).
                    self.finger.move_to(self.target, abort=self._stop)
            time.sleep(0.05)
        self.running = False