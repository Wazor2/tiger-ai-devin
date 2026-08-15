"""Motion-triggered capture loop for the live demo.

Three sources, same loop and same trigger logic:
  webcam  - cv2.VideoCapture(index); the phone via Iriun shows up as a normal
            webcam (use `python -m pench.live_capture --probe` to find its index)
  folder  - replays still images from a directory, for rehearsing without a phone
  dummy   - synthesises frames, for proving the plumbing with no camera at all

Real camera traps are trigger-based, so captures fire on inter-frame motion,
with a forced capture every FORCE_INTERVAL seconds and a MIN_GAP floor so a
moving scene cannot spam near-duplicate frames.
"""
import argparse
import threading
import time
from pathlib import Path

import numpy as np

MOTION_THRESHOLD = 6.0        # mean abs pixel delta over a downscaled frame
FORCE_INTERVAL = 5.0          # seconds: re-check a static scene anyway
MIN_GAP = 1.5                 # seconds: minimum spacing between captures


def probe_devices(max_index: int = 4) -> list:
    """Report which cv2.VideoCapture indices actually deliver a frame."""
    import cv2
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        ok, frame = (cap.read() if cap.isOpened() else (False, None))
        if ok and frame is not None:
            found.append({"index": i, "width": int(frame.shape[1]),
                          "height": int(frame.shape[0])})
        cap.release()
    return found


class FrameSource:
    """Yields BGR numpy frames. Subclasses differ only in where frames come from."""

    def read(self):
        raise NotImplementedError

    def release(self):
        pass


class WebcamSource(FrameSource):
    def __init__(self, index: int = 0):
        import cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"cannot open webcam index {index}; run "
                f"`python -m pench.live_capture --probe` to list working indices")
        self.index = index

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class FolderSource(FrameSource):
    """Replays stills from a folder, holding each for `hold` seconds.

    Each still is emitted with slight synthetic noise so the motion detector
    fires on the switch between images, as it would on a real scene change.
    """

    def __init__(self, folder, hold: float = 6.0, loop: bool = True):
        import cv2
        self.paths = sorted(p for p in Path(folder).iterdir()
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not self.paths:
            raise RuntimeError(f"no images in {folder}")
        self.frames = [cv2.imread(str(p)) for p in self.paths]
        self.hold, self.loop = hold, loop
        self.started = time.time()
        self.exhausted = False

    def read(self):
        idx = int((time.time() - self.started) / self.hold)
        if idx >= len(self.frames):
            if not self.loop:
                self.exhausted = True
                return None
            idx %= len(self.frames)
        return self.frames[idx]

    def current_name(self):
        idx = int((time.time() - self.started) / self.hold) % len(self.paths)
        return self.paths[idx].name


class DummySource(FrameSource):
    """Synthetic frames: a bright blob drifting over noise. No camera needed."""

    def __init__(self, size=(480, 640)):
        self.size = size
        self.t0 = time.time()

    def read(self):
        h, w = self.size
        frame = np.random.normal(40, 8, (h, w, 3))
        phase = (time.time() - self.t0) / 4.0
        cx = int(w * (0.5 + 0.35 * np.sin(phase)))
        cy = int(h * (0.5 + 0.2 * np.cos(phase)))
        ys, xs = np.mgrid[0:h, 0:w]
        blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 60.0 ** 2)))
        frame += (blob * 150)[..., None]
        return np.clip(frame, 0, 255).astype("uint8")


def motion_score(prev, frame) -> float:
    """Mean absolute difference on a downscaled grayscale pair."""
    import cv2
    a = cv2.cvtColor(cv2.resize(prev, (160, 120)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(a, b).mean())


class CaptureLoop(threading.Thread):
    """Runs a FrameSource and calls `on_capture(frame_bgr, meta)` on trigger."""

    daemon = True

    def __init__(self, source: FrameSource, on_capture,
                 motion_threshold: float = MOTION_THRESHOLD,
                 force_interval: float = FORCE_INTERVAL,
                 min_gap: float = MIN_GAP, poll: float = 0.2):
        super().__init__(name="capture-loop")
        self.source = source
        self.on_capture = on_capture
        self.motion_threshold = motion_threshold
        self.force_interval = force_interval
        self.min_gap = min_gap
        self.poll = poll
        # not `_stop`: threading.Thread uses that name internally and join()
        # would call this Event
        self._stop_event = threading.Event()
        self.stats = {"frames": 0, "captures": 0, "last_motion": 0.0,
                      "last_capture": None, "errors": 0}

    def stop(self):
        self._stop_event.set()

    def run(self):
        prev = None
        last_capture = 0.0
        try:
            while not self._stop_event.is_set():
                frame = self.source.read()
                if frame is None:
                    if getattr(self.source, "exhausted", False):
                        break
                    time.sleep(self.poll)
                    continue
                self.stats["frames"] += 1
                now = time.time()
                score = motion_score(prev, frame) if prev is not None else 0.0
                self.stats["last_motion"] = round(score, 2)
                prev = frame

                since = now - last_capture
                if since < self.min_gap:
                    trigger = None
                elif score >= self.motion_threshold:
                    trigger = "motion"
                elif since >= self.force_interval:
                    trigger = "interval"
                else:
                    trigger = None

                if trigger:
                    last_capture = now
                    self.stats["captures"] += 1
                    self.stats["last_capture"] = now
                    meta = {"trigger": trigger, "motion_score": round(score, 2)}
                    if hasattr(self.source, "current_name"):
                        meta["source_file"] = self.source.current_name()
                    try:
                        self.on_capture(frame, meta)
                    except Exception as exc:      # never kill the demo loop
                        self.stats["errors"] += 1
                        self.stats["last_error"] = f"{type(exc).__name__}: {exc}"
                time.sleep(self.poll)
        finally:
            self.source.release()


def build_source(kind: str, device_index: int = 0, folder=None,
                 hold: float = 6.0, loop: bool = True) -> FrameSource:
    if kind == "webcam":
        return WebcamSource(device_index)
    if kind == "folder":
        return FolderSource(folder, hold=hold, loop=loop)
    if kind == "dummy":
        return DummySource()
    raise ValueError(f"unknown source {kind!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="capture loop / webcam probe")
    ap.add_argument("--probe", action="store_true",
                    help="list working cv2.VideoCapture indices 0-4 and exit")
    args = ap.parse_args()
    if args.probe:
        devices = probe_devices()
        if not devices:
            print("no working webcam indices in 0-4 — is Iriun connected?")
        for d in devices:
            print(f"index {d['index']}: {d['width']}x{d['height']}")
