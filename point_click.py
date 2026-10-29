"""Show the webcam feed with MediaPipe hand landmarks drawn on top, and move the cursor with the index knuckle.

Pick the camera from the dropdown. Drag the input rectangle to move it, or its corners to resize it.
The knuckle's position inside the rectangle maps to the cursor's position on the primary monitor,
after One Euro smoothing tuned with the sliders. Pinch thumb and index for the left button (hold to drag),
make a fist for the right button. Hold four fingers straight and together, then move the hand up or down to scroll.
Press F8 (anywhere) to toggle cursor control, and F7 to toggle the preview. Press Q or Esc in the window to quit.
"""

import json
import os
import queue
import threading
import time
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import ttk

# Must be set before cv2 is imported. Without it, MSMF can take several seconds to open a camera.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.tasks.python import BaseOptions  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402
from pygrabber.dshow_graph import FilterGraph  # noqa: E402

import gestures  # noqa: E402
import win_input  # noqa: E402
from cursor_control import CursorController  # noqa: E402
from msmf_cameras import list_msmf_cameras  # noqa: E402
from one_euro import OneEuroFilter  # noqa: E402

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
SETTINGS_PATH = Path(__file__).parent / "settings.json"

DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Asked for explicitly, since many cameras otherwise default to 30 or less. Cameras that can't do it
# fall back to their nearest supported rate.
FRAME_RATE = 60
NUM_HANDS = 1

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
FINGERTIPS = {4, 8, 12, 16, 20}
INDEX_KNUCKLE = 5

# Input rectangle as (left, top, right, bottom) in normalized frame coordinates.
DEFAULT_INPUT_RECT = [0.25, 0.25, 0.75, 0.75]
MIN_RECT_SIZE = 0.05
HANDLE_RADIUS_PX = 8

# One Euro smoothing of the knuckle, in normalized frame coordinates (speed is in frame widths per second).
DEFAULT_MIN_CUTOFF = 0.4
DEFAULT_BETA = 15.0
D_CUTOFF = 1.0
MIN_CUTOFF_RANGE = (0.05, 5.0, 0.05)  # from, to, resolution
BETA_RANGE = (0.0, 30.0, 0.1)
# Min cutoff and beta are multiplied by this when a click gesture reaches its threshold, blending in
# as it approaches, so the cursor steadies right where the click lands.
CLICK_SMOOTHING = 0.025

CURSOR_TOGGLE_KEY = 0x77  # VK_F8
PREVIEW_TOGGLE_KEY = 0x76  # VK_F7
PREVIEW_TOGGLE_LABEL = "F7"
# Held buttons are released once the hand has been out of view this long. The delay stops a dropped
# frame from ending a drag.
HAND_LOST_GRACE_S = 0.25
# How often the window checks for a new frame from the capture thread.
DISPLAY_INTERVAL_MS = 10

# Per-frame stage timings shown on the video, in the order they run. read is cap.read(): waiting for the
# camera plus decoding its compressed frames. The rest are on the capture thread, except show, which is the Tk thread
# turning the frame into an image.
TIMED_STAGES = ("read", "prep", "detect", "logic", "draw", "out")
TIMING_SMOOTHING = 0.1
TIMING_LINE_HEIGHT = 16
# The camera mode line, then the two timing lines.
DEBUG_TEXT_LINES = 3

WINDOW_TITLE = "PointClick - hand tracking"


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def list_cameras() -> list[str]:
    # DirectShow device order, which matches OpenCV's CAP_DSHOW indices and pygrabber's.
    return FilterGraph().get_input_devices()


def open_camera(dshow_index: int, msmf_index: int | None) -> cv2.VideoCapture | None:
    """Open a camera through Media Foundation if it lists it, else through DirectShow.

    OpenCV's DirectShow backend can leave cameras stuck in their uncompressed YUY2 format, which
    can't reach 60 FPS over USB. Media Foundation picks a compressed format that can by itself.
    Cameras only DirectShow knows about, such as some virtual cameras, still open through it.
    """
    params = [cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT,
              cv2.CAP_PROP_FPS, FRAME_RATE]
    if msmf_index is not None:
        cap = cv2.VideoCapture(msmf_index, cv2.CAP_MSMF, params)
        if cap.isOpened():
            return cap
    cap = cv2.VideoCapture(dshow_index, cv2.CAP_DSHOW, params)
    return cap if cap.isOpened() else None


def describe_mode(cap: cv2.VideoCapture) -> str:
    """The backend and format the camera agreed to, which can differ from what open_camera asked for."""
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return f"Camera {cap.getBackendName()} {width}x{height} @ {cap.get(cv2.CAP_PROP_FPS):.1f} FPS"


def preview_off_image(width: int, height: int):
    """A plain RGB frame shown in place of the video while the preview is off."""
    image = np.full((height, width, 3), 40, np.uint8)
    text = f"Preview off - {PREVIEW_TOGGLE_LABEL} to show"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(image, text, ((width - text_w) // 2, (height + text_h) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (200, 200, 200), 2, cv2.LINE_AA)
    return image


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rect_to_screen(rect: list[float], x: float, y: float) -> tuple[int, int]:
    """Map a normalized frame point inside the input rectangle to a pixel on the primary monitor."""
    left, top, right, bottom = rect
    u = clamp((x - left) / (right - left), 0.0, 1.0)
    v = clamp((y - top) / (bottom - top), 0.0, 1.0)
    screen_w, screen_h = win_input.primary_screen_size()
    return round(u * (screen_w - 1)), round(v * (screen_h - 1))


def draw_hands(frame, result: vision.HandLandmarkerResult) -> None:
    h, w = frame.shape[:2]
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for conn in HAND_CONNECTIONS:
            cv2.line(frame, points[conn.start], points[conn.end], (255, 255, 255), 2, cv2.LINE_AA)

        for i, pt in enumerate(points):
            color = (0, 0, 255) if i in FINGERTIPS else (0, 200, 0)
            cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, points[INDEX_KNUCKLE], 9, (255, 0, 255), 2, cv2.LINE_AA)

        label = f"{handedness[0].category_name} {handedness[0].score:.2f}"
        x, y = points[0]
        cv2.putText(frame, label, (x - 30, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def draw_gesture_meters(frame, detector: gestures.GestureDetector, bottom: int) -> None:
    """Bars for the pinch, fist and scroll measurements, with the press (red) and release (yellow) thresholds,
    ending at y = bottom."""
    rows = (
        ("Left", detector.pinch, gestures.PINCH_PRESS, gestures.PINCH_RELEASE, detector.left.pressed),
        ("Right", detector.fist, gestures.FIST_PRESS, gestures.FIST_RELEASE, detector.right.pressed),
        ("Scroll", detector.flat, gestures.SCROLL_PRESS, gestures.SCROLL_RELEASE, detector.scroll.pressed),
    )
    x0, bar_w, bar_h, full_scale = 80, 200, 14, 1.4
    top = bottom - len(rows) * 24
    cv2.putText(frame, f"Click smoothing {detector.proximity:.0%}", (10, top - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (name, value, press, release, pressed) in enumerate(rows):
        y = top + i * 24
        cv2.putText(frame, name, (10, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        fill = int(min(value / full_scale, 1.0) * bar_w)
        cv2.rectangle(frame, (x0, y), (x0 + fill, y + bar_h), (0, 220, 0) if pressed else (160, 160, 160), -1)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (255, 255, 255), 1)
        for threshold, color in ((press, (0, 0, 255)), (release, (0, 255, 255))):
            tx = x0 + int(threshold / full_scale * bar_w)
            cv2.line(frame, (tx, y - 3), (tx, y + bar_h + 3), color, 2)


def draw_debug_text(frame, camera_mode: str, timings: dict[str, float]) -> None:
    """The camera mode and smoothed stage times in ms, as DEBUG_TEXT_LINES lines along the bottom of the frame."""
    h = frame.shape[0]
    process = timings.get("process", 0.0)
    lines = (
        camera_mode,
        "  ".join(f"{stage} {timings.get(stage, 0.0):.1f}" for stage in TIMED_STAGES) + " ms",
        f"process {process:.1f} ms (CPU max {1000 / max(process, 1e-3):.0f} FPS)  show {timings.get('show', 0.0):.1f} ms",
    )
    for i, line in enumerate(lines):
        y = h - 6 - (len(lines) - 1 - i) * TIMING_LINE_HEIGHT
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def draw_input_rect(frame, rect: list[float], active: bool) -> None:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = int(rect[0] * w), int(rect[1] * h), int(rect[2] * w), int(rect[3] * h)
    color = (0, 255, 0) if active else (0, 165, 255)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    s = HANDLE_RADIUS_PX // 2
    for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        cv2.rectangle(frame, (cx - s, cy - s), (cx + s, cy + s), color, -1)


class PointClickApp:
    """Camera reading, hand tracking, cursor control and drawing run on a capture thread, so they don't
    wait on Tk. The Tk thread only shows the latest frame and handles the controls.

    The two threads share plain attributes, each written by only one of them: input_rect, drag,
    control_cursor, preview and smoothing from Tk; frame_size and latest_frame from the capture thread.
    In timings, "show" is written by Tk and every other stage by the capture thread.
    Tk widgets and variables are only touched from the Tk thread. The capture thread owns the camera,
    and is asked to switch it through camera_requests, answering through camera_results.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.camera_names: list[str] = []  # DirectShow order, as shown in the dropdown.
        self.msmf_camera_names: list[str] = []  # Media Foundation order, used to open cameras.
        self.settings = load_settings()
        self.input_rect = list(self.settings.get("input_rect", DEFAULT_INPUT_RECT))
        self.frame_size = (FRAME_WIDTH, FRAME_HEIGHT)
        self.drag: dict | None = None
        self.toggle_keys_down: set[int] = set()
        min_cutoff = self.settings.get("min_cutoff", DEFAULT_MIN_CUTOFF)
        beta = self.settings.get("beta", DEFAULT_BETA)
        self.smoothing = (min_cutoff, beta)
        self.knuckle_filters = (OneEuroFilter(min_cutoff, beta, D_CUTOFF), OneEuroFilter(min_cutoff, beta, D_CUTOFF))
        self.gestures = gestures.GestureDetector()
        self.cursor = CursorController()
        self.control_cursor = False
        self.preview = True
        self.last_hand_time = 0.0
        self.start = time.perf_counter()
        self.last_frame_time = self.start
        self.fps = 0.0
        self.timings: dict[str, float] = {}  # Smoothed ms per stage, see TIMED_STAGES.
        self.camera_mode = ""  # Capture thread only.
        self.latest_frame = None  # RGB preview with the overlays drawn, ready to show.
        self.shown_frame = None
        self.preview_off_frame = None  # Built on the Tk thread when the preview is first turned off.
        # Requests are (DirectShow index, Media Foundation index or None, remember).
        self.camera_requests: queue.SimpleQueue[tuple[int, int | None, bool]] = queue.SimpleQueue()
        self.camera_results: queue.SimpleQueue[tuple[int, bool, bool]] = queue.SimpleQueue()
        self.stopping = threading.Event()

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=NUM_HANDS,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        root.title(WINDOW_TITLE)
        toolbar = ttk.Frame(root, padding=6)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Camera:").pack(side="left")
        self.camera_box = ttk.Combobox(toolbar, state="readonly", width=40)
        self.camera_box.pack(side="left", padx=6)
        self.camera_box.bind("<<ComboboxSelected>>", lambda _: self.select_camera(self.camera_box.current()))
        ttk.Button(toolbar, text="Refresh", command=self.refresh_cameras).pack(side="left")
        # Both are mirrored into plain attributes, since Tk variables mustn't be read from the capture thread.
        self.cursor_enabled = tk.BooleanVar(value=False)
        self.cursor_enabled.trace_add("write", lambda *_: setattr(self, "control_cursor", self.cursor_enabled.get()))
        ttk.Checkbutton(toolbar, text="Control cursor (F8)", variable=self.cursor_enabled).pack(side="left", padx=(12, 0))
        self.preview_enabled = tk.BooleanVar(value=True)
        self.preview_enabled.trace_add("write", lambda *_: self.on_preview_toggled())
        ttk.Checkbutton(toolbar, text=f"Enable Preview ({PREVIEW_TOGGLE_LABEL})",
                        variable=self.preview_enabled).pack(side="left", padx=12)
        # Polled instead of Tk key bindings so they work while another window has focus.
        self.toggle_keys = {CURSOR_TOGGLE_KEY: self.cursor_enabled, PREVIEW_TOGGLE_KEY: self.preview_enabled}
        self.status = ttk.Label(toolbar, foreground="red")
        self.status.pack(side="left", padx=6)

        smoothing = ttk.Frame(root, padding=(6, 0, 6, 6))
        smoothing.pack(fill="x")
        self.min_cutoff_scale = self.add_smoothing_scale(smoothing, "Min cutoff (Hz) - lower = less jitter", MIN_CUTOFF_RANGE, min_cutoff)
        self.beta_scale = self.add_smoothing_scale(smoothing, "Beta - higher = less lag on fast moves", BETA_RANGE, beta)

        # A plain tk.Label has no padding, so mouse coordinates on it match frame pixels.
        self.video = tk.Label(root, bd=0, highlightthickness=0)
        self.video.pack()
        self.video.bind("<ButtonPress-1>", self.on_press)
        self.video.bind("<B1-Motion>", self.on_drag)
        self.video.bind("<ButtonRelease-1>", self.on_release)
        self.video.bind("<Motion>", self.on_hover)

        root.bind("<Escape>", lambda _: self.close())
        root.bind("q", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.refresh_cameras()
        if self.camera_names:
            # Reopen the last camera by name, since indices shift when devices are plugged in or removed.
            saved = self.settings.get("camera")
            if saved in self.camera_names:
                index = self.camera_names.index(saved)
            else:
                index = min(DEFAULT_CAMERA_INDEX, len(self.camera_names) - 1)
            self.camera_box.current(index)
            self.switch_camera(index)
        self.worker = threading.Thread(target=self.capture_loop, name="capture", daemon=True)
        self.worker.start()
        self.show_frame()

    def add_smoothing_scale(self, parent: ttk.Frame, label: str, value_range: tuple, value: float) -> tk.Scale:
        low, high, resolution = value_range
        scale = tk.Scale(parent, label=label, from_=low, to=high, resolution=resolution,
                         orient="horizontal", length=280)
        scale.set(value)
        # Set after the initial value, so it can't fire before both sliders exist.
        scale.config(command=lambda _: self.read_smoothing())
        scale.pack(side="left", padx=(0, 12))
        # Save once the user lets go, not on every step of the drag.
        scale.bind("<ButtonRelease-1>", lambda _: self.save_smoothing())
        return scale

    def read_smoothing(self) -> None:
        # Copied out of the sliders on every change, since the capture thread can't read Tk widgets.
        self.smoothing = (self.min_cutoff_scale.get(), self.beta_scale.get())

    def apply_smoothing(self, click_proximity: float) -> None:
        # Applied every frame, so slider changes take effect immediately.
        min_cutoff, beta = self.smoothing
        factor = 1.0 + (CLICK_SMOOTHING - 1.0) * click_proximity
        for f in self.knuckle_filters:
            f.min_cutoff = min_cutoff * factor
            f.beta = beta * factor

    def save_smoothing(self) -> None:
        self.settings["min_cutoff"] = self.min_cutoff_scale.get()
        self.settings["beta"] = self.beta_scale.get()
        save_settings(self.settings)

    def refresh_cameras(self) -> None:
        self.camera_names = list_cameras()
        try:
            self.msmf_camera_names = list_msmf_cameras()
        except OSError:
            # Every camera still opens through DirectShow, just without Media Foundation's faster modes.
            self.msmf_camera_names = []
        self.camera_box["values"] = [f"{i}: {name}" for i, name in enumerate(self.camera_names)]
        if not self.camera_names:
            self.status.config(text="No cameras found")

    def select_camera(self, index: int) -> None:
        self.switch_camera(index, remember=True)

    def switch_camera(self, index: int, remember: bool = False) -> None:
        """Ask the capture thread to open camera index (in DirectShow order). If remember, it's saved once it opens."""
        # Media Foundation lists cameras in its own order, so the same camera is found there by name.
        # Two cameras with the same name can't be told apart this way, so the first is used.
        name = self.camera_names[index]
        msmf_index = self.msmf_camera_names.index(name) if name in self.msmf_camera_names else None
        self.camera_requests.put((index, msmf_index, remember))

    def on_preview_toggled(self) -> None:
        self.preview = self.preview_enabled.get()
        if not self.preview:
            # The rectangle can't be edited without seeing it, so end any drag where it is.
            self.on_release(None)
            self.video.config(cursor="")

    def rect_editable(self) -> bool:
        # The rectangle is locked while the hand drives the cursor, so gesture clicks over the video
        # can't move it, and while the preview is off, since it isn't shown.
        return self.preview and not self.control_cursor

    def handle_camera_results(self) -> None:
        while True:
            try:
                index, remember, opened = self.camera_results.get_nowait()
            except queue.Empty:
                return
            self.status.config(text="" if opened else f"Could not open camera {index}")
            if opened and remember:
                self.settings["camera"] = self.camera_names[index]
                save_settings(self.settings)

    def hit_test(self, px: int, py: int) -> tuple[str, int, int] | None:
        """Return ("corner", x_index, y_index) or ("move", 0, 0) for a point on the video, or None."""
        w, h = self.frame_size
        left, top, right, bottom = self.input_rect
        for xi, cx in ((0, left * w), (2, right * w)):
            for yi, cy in ((1, top * h), (3, bottom * h)):
                if abs(px - cx) <= HANDLE_RADIUS_PX and abs(py - cy) <= HANDLE_RADIUS_PX:
                    return ("corner", xi, yi)
        if left * w < px < right * w and top * h < py < bottom * h:
            return ("move", 0, 0)
        return None

    def on_hover(self, event: tk.Event) -> None:
        hit = self.hit_test(event.x, event.y) if self.rect_editable() else None
        if hit is None:
            cursor = ""
        elif hit[0] == "move":
            cursor = "fleur"
        else:
            # Top-left and bottom-right corners resize along one diagonal, the other two along the other.
            cursor = "size_nw_se" if (hit[1] == 0) == (hit[2] == 1) else "size_ne_sw"
        self.video.config(cursor=cursor)

    def on_press(self, event: tk.Event) -> None:
        if not self.rect_editable():
            return
        hit = self.hit_test(event.x, event.y)
        if hit is not None:
            self.drag = {"kind": hit[0], "xi": hit[1], "yi": hit[2],
                         "start": (event.x, event.y), "rect": list(self.input_rect)}

    def on_drag(self, event: tk.Event) -> None:
        if self.drag is None:
            return
        w, h = self.frame_size
        rect = self.drag["rect"]
        if self.drag["kind"] == "move":
            # Keep the whole rectangle inside the frame.
            dx = clamp((event.x - self.drag["start"][0]) / w, -rect[0], 1.0 - rect[2])
            dy = clamp((event.y - self.drag["start"][1]) / h, -rect[1], 1.0 - rect[3])
            self.input_rect = [rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy]
        else:
            new = list(rect)
            nx, ny = clamp(event.x / w, 0.0, 1.0), clamp(event.y / h, 0.0, 1.0)
            xi, yi = self.drag["xi"], self.drag["yi"]
            new[xi] = min(nx, rect[2] - MIN_RECT_SIZE) if xi == 0 else max(nx, rect[0] + MIN_RECT_SIZE)
            new[yi] = min(ny, rect[3] - MIN_RECT_SIZE) if yi == 1 else max(ny, rect[1] + MIN_RECT_SIZE)
            self.input_rect = new

    def on_release(self, _event: tk.Event | None) -> None:
        if self.drag is not None:
            self.drag = None
            self.settings["input_rect"] = self.input_rect
            save_settings(self.settings)

    def poll_toggle_keys(self) -> None:
        """Flip each toggle key's checkbox once per press."""
        for key, variable in self.toggle_keys.items():
            if win_input.is_key_down(key):
                if key not in self.toggle_keys_down:
                    variable.set(not variable.get())
                    self.toggle_keys_down.add(key)
            else:
                self.toggle_keys_down.discard(key)

    def capture_loop(self) -> None:
        """Capture thread: open the requested camera, then read and process frames until stopping."""
        cap: cv2.VideoCapture | None = None
        while not self.stopping.is_set():
            try:
                index, msmf_index, remember = self.camera_requests.get_nowait()
            except queue.Empty:
                pass
            else:
                if cap is not None:
                    cap.release()
                cap = open_camera(index, msmf_index)
                self.camera_mode = describe_mode(cap) if cap else ""
                for f in self.knuckle_filters:
                    f.reset()
                self.camera_results.put((index, remember, cap is not None))

            # cap.read() blocks until the next frame, which paces this loop to the camera.
            read_start = time.perf_counter()
            ok, frame = cap.read() if cap else (False, None)
            if ok:
                self.record_timing("read", read_start)
                self.process_frame(frame)
            else:
                self.stopping.wait(0.1)
        if cap is not None:
            cap.release()
        self.cursor.release_all()

    def record_timing(self, stage: str, start: float) -> float:
        """Fold the time since start into stage's smoothed ms, and return now to start the next stage."""
        now = time.perf_counter()
        ms = (now - start) * 1000
        previous = self.timings.get(stage)
        self.timings[stage] = ms if previous is None else previous + TIMING_SMOOTHING * (ms - previous)
        return now

    def process_frame(self, frame) -> None:
        """Track the hand in one camera frame, drive the cursor, and publish the preview as latest_frame."""
        process_start = stage_start = time.perf_counter()
        # Mirror so moving your hand right moves it right on screen.
        frame = cv2.flip(frame, 1)
        self.frame_size = (frame.shape[1], frame.shape[0])

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        stage_start = self.record_timing("prep", stage_start)
        # Timestamps must keep increasing across camera switches, so they use one shared start time.
        t = stage_start - self.start
        result = self.landmarker.detect_for_video(mp_image, int(t * 1000))
        stage_start = self.record_timing("detect", stage_start)

        # The filter runs even with cursor control off, so smoothing can be tuned from the debug view.
        smoothed = None
        if result.hand_landmarks:
            self.last_hand_time = t
            left, right, scrolling = self.gestures.update(result.hand_world_landmarks[0])
            self.apply_smoothing(self.gestures.proximity)
            knuckle = result.hand_landmarks[0][INDEX_KNUCKLE]
            smoothed = (self.knuckle_filters[0](knuckle.x, t), self.knuckle_filters[1](knuckle.y, t))
        else:
            # Start fresh when the hand reappears instead of gliding from where it was last seen.
            for f in self.knuckle_filters:
                f.reset()
            if t - self.last_hand_time > HAND_LOST_GRACE_S:
                self.gestures.reset()
                self.cursor.release_all()

        # Read once, since the Tk thread can replace it mid-frame while the user drags it.
        input_rect = self.input_rect
        # Don't fight the user's mouse while they're adjusting the rectangle.
        active = self.control_cursor and self.drag is None
        if not active:
            self.cursor.release_all()
        elif smoothed is not None:
            self.cursor.update(rect_to_screen(input_rect, *smoothed), left, right,
                               smoothed[1] if scrolling else None)
        stage_start = self.record_timing("logic", stage_start)

        now = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.last_frame_time, 1e-6)
        self.last_frame_time = now

        if not self.preview:
            # Nothing is shown, so skip drawing and converting the frame. Dropping the last frame keeps
            # it from flashing up stale when the preview is turned back on.
            self.latest_frame = None
            self.record_timing("process", process_start)
            return

        h, w = frame.shape[:2]
        draw_input_rect(frame, input_rect, active)
        draw_hands(frame, result)
        # The gesture meters sit right above the debug text.
        draw_gesture_meters(frame, self.gestures, h - 8 - DEBUG_TEXT_LINES * TIMING_LINE_HEIGHT)
        if smoothed is not None:
            center = (int(smoothed[0] * w), int(smoothed[1] * h))
            cv2.drawMarker(frame, center, (255, 255, 0), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        stage_start = self.record_timing("draw", stage_start)

        cv2.putText(frame, f"{self.fps:.0f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        # Drawn before this frame's out and process times exist, so those lag one frame behind.
        draw_debug_text(frame, self.camera_mode, self.timings)

        self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.record_timing("out", stage_start)
        self.record_timing("process", process_start)

    def show_frame(self) -> None:
        """Tk thread: poll the toggle keys and camera results, and show the newest frame if there is one."""
        self.poll_toggle_keys()
        self.handle_camera_results()
        if self.preview:
            frame = self.latest_frame
        else:
            # Same size as the video, so the window doesn't jump when the preview is toggled.
            w, h = self.frame_size
            if self.preview_off_frame is None or self.preview_off_frame.shape[:2] != (h, w):
                self.preview_off_frame = preview_off_image(w, h)
            frame = self.preview_off_frame
        if frame is not None and frame is not self.shown_frame:
            show_start = time.perf_counter()
            photo = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video.config(image=photo)
            self.video.image = photo  # Keep a reference so Tk doesn't drop the image.
            self.shown_frame = frame
            self.record_timing("show", show_start)
        self.root.after(DISPLAY_INTERVAL_MS, self.show_frame)

    def close(self) -> None:
        # The capture thread releases the camera and any held buttons on its way out.
        self.stopping.set()
        self.worker.join()
        self.landmarker.close()
        self.root.destroy()


def main() -> None:
    win_input.enable_dpi_awareness()
    root = tk.Tk()
    PointClickApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
