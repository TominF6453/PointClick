"""Show the webcam feed with MediaPipe hand landmarks drawn on top, and move the cursor with the index knuckle.

Pick the camera from the dropdown. Drag the input rectangle to move it, or its corners to resize it.
The knuckle's position inside the rectangle maps to the cursor's position on the primary monitor,
after One Euro smoothing tuned with the sliders. Pinch thumb and index for the left button (hold to drag),
make a fist for the right button. Hold four fingers straight and together, then move the hand up or down to scroll.
Press F8 (anywhere) to toggle cursor control. Press Q or Esc in the window to quit.
"""

import json
import time
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import ttk

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from PIL import Image, ImageTk
from pygrabber.dshow_graph import FilterGraph

import gestures
import win_input
from cursor_control import CursorController
from one_euro import OneEuroFilter

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
SETTINGS_PATH = Path(__file__).parent / "settings.json"

DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
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

TOGGLE_KEY = 0x77  # VK_F8
# Held buttons are released once the hand has been out of view this long. The delay stops a dropped
# frame from ending a drag.
HAND_LOST_GRACE_S = 0.25

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
    # DirectShow device order matches OpenCV's CAP_DSHOW indices.
    return FilterGraph().get_input_devices()


def open_camera(index: int) -> cv2.VideoCapture | None:
    # DirectShow opens faster and with less latency than MSMF on Windows.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


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


def draw_gesture_meters(frame, detector: gestures.GestureDetector) -> None:
    """Bars for the pinch, fist and scroll measurements, with the press (red) and release (yellow) thresholds."""
    h = frame.shape[0]
    rows = (
        ("Left", detector.pinch, gestures.PINCH_PRESS, gestures.PINCH_RELEASE, detector.left.pressed),
        ("Right", detector.fist, gestures.FIST_PRESS, gestures.FIST_RELEASE, detector.right.pressed),
        ("Scroll", detector.flat, gestures.SCROLL_PRESS, gestures.SCROLL_RELEASE, detector.scroll.pressed),
    )
    x0, bar_w, bar_h, full_scale = 80, 200, 14, 1.4
    top = h - 2 - len(rows) * 24
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


def draw_input_rect(frame, rect: list[float], active: bool) -> None:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = int(rect[0] * w), int(rect[1] * h), int(rect[2] * w), int(rect[3] * h)
    color = (0, 255, 0) if active else (0, 165, 255)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    s = HANDLE_RADIUS_PX // 2
    for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        cv2.rectangle(frame, (cx - s, cy - s), (cx + s, cy + s), color, -1)


class HandViewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.cap: cv2.VideoCapture | None = None
        self.camera_names: list[str] = []
        self.settings = load_settings()
        self.input_rect = list(self.settings.get("input_rect", DEFAULT_INPUT_RECT))
        self.frame_size = (FRAME_WIDTH, FRAME_HEIGHT)
        self.drag: dict | None = None
        self.toggle_key_was_down = False
        min_cutoff = self.settings.get("min_cutoff", DEFAULT_MIN_CUTOFF)
        beta = self.settings.get("beta", DEFAULT_BETA)
        self.knuckle_filters = (OneEuroFilter(min_cutoff, beta, D_CUTOFF), OneEuroFilter(min_cutoff, beta, D_CUTOFF))
        self.gestures = gestures.GestureDetector()
        self.cursor = CursorController()
        self.last_hand_time = 0.0
        self.start = time.perf_counter()
        self.last_frame_time = self.start
        self.fps = 0.0

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
        self.cursor_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="Control cursor (F8)", variable=self.cursor_enabled).pack(side="left", padx=12)
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
        self.update_frame()

    def add_smoothing_scale(self, parent: ttk.Frame, label: str, value_range: tuple, value: float) -> tk.Scale:
        low, high, resolution = value_range
        scale = tk.Scale(parent, label=label, from_=low, to=high, resolution=resolution,
                         orient="horizontal", length=280)
        scale.set(value)
        scale.pack(side="left", padx=(0, 12))
        # Save once the user lets go, not on every step of the drag.
        scale.bind("<ButtonRelease-1>", lambda _: self.save_smoothing())
        return scale

    def apply_smoothing(self, click_proximity: float) -> None:
        # The sliders are read every frame, so changes apply immediately.
        factor = 1.0 + (CLICK_SMOOTHING - 1.0) * click_proximity
        for f in self.knuckle_filters:
            f.min_cutoff = self.min_cutoff_scale.get() * factor
            f.beta = self.beta_scale.get() * factor

    def save_smoothing(self) -> None:
        self.settings["min_cutoff"] = self.min_cutoff_scale.get()
        self.settings["beta"] = self.beta_scale.get()
        save_settings(self.settings)

    def refresh_cameras(self) -> None:
        self.camera_names = list_cameras()
        self.camera_box["values"] = [f"{i}: {name}" for i, name in enumerate(self.camera_names)]
        if not self.camera_names:
            self.status.config(text="No cameras found")

    def select_camera(self, index: int) -> None:
        self.switch_camera(index)
        if self.cap is not None:
            self.settings["camera"] = self.camera_names[index]
            save_settings(self.settings)

    def switch_camera(self, index: int) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = open_camera(index)
        for f in self.knuckle_filters:
            f.reset()
        self.status.config(text="" if self.cap else f"Could not open camera {index}")

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
        hit = None if self.cursor_enabled.get() else self.hit_test(event.x, event.y)
        if hit is None:
            cursor = ""
        elif hit[0] == "move":
            cursor = "fleur"
        else:
            # Top-left and bottom-right corners resize along one diagonal, the other two along the other.
            cursor = "size_nw_se" if (hit[1] == 0) == (hit[2] == 1) else "size_ne_sw"
        self.video.config(cursor=cursor)

    def on_press(self, event: tk.Event) -> None:
        # The rectangle is locked while the hand drives the cursor, so gesture clicks over the video
        # can't move it.
        if self.cursor_enabled.get():
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

    def on_release(self, _event: tk.Event) -> None:
        if self.drag is not None:
            self.drag = None
            self.settings["input_rect"] = self.input_rect
            save_settings(self.settings)

    def poll_toggle_key(self) -> None:
        # Polled instead of a Tk key binding so it works while another window has focus.
        down = win_input.is_key_down(TOGGLE_KEY)
        if down and not self.toggle_key_was_down:
            self.cursor_enabled.set(not self.cursor_enabled.get())
        self.toggle_key_was_down = down

    def update_frame(self) -> None:
        self.poll_toggle_key()
        ok, frame = self.cap.read() if self.cap else (False, None)
        if ok:
            # Mirror so moving your hand right moves it right on screen.
            frame = cv2.flip(frame, 1)
            self.frame_size = (frame.shape[1], frame.shape[0])

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Timestamps must keep increasing across camera switches, so they use one shared start time.
            t = time.perf_counter() - self.start
            result = self.landmarker.detect_for_video(mp_image, int(t * 1000))

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

            # Don't fight the user's mouse while they're adjusting the rectangle.
            active = self.cursor_enabled.get() and self.drag is None
            if not active:
                self.cursor.release_all()
            elif smoothed is not None:
                self.cursor.update(rect_to_screen(self.input_rect, *smoothed), left, right,
                                   smoothed[1] if scrolling else None)

            draw_input_rect(frame, self.input_rect, active)
            draw_hands(frame, result)
            draw_gesture_meters(frame, self.gestures)
            if smoothed is not None:
                h, w = frame.shape[:2]
                center = (int(smoothed[0] * w), int(smoothed[1] * h))
                cv2.drawMarker(frame, center, (255, 255, 0), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)

            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - self.last_frame_time, 1e-6)
            self.last_frame_time = now
            cv2.putText(frame, f"{self.fps:.0f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video.config(image=photo)
            self.video.image = photo  # Keep a reference so Tk doesn't drop the image.

        # cap.read() blocks until the next frame, so this only needs to yield to Tk briefly.
        self.root.after(1 if ok else 100, self.update_frame)

    def close(self) -> None:
        self.cursor.release_all()
        if self.cap is not None:
            self.cap.release()
        self.landmarker.close()
        self.root.destroy()


def main() -> None:
    win_input.enable_dpi_awareness()
    root = tk.Tk()
    HandViewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
