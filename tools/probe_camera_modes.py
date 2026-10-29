"""Try several ways of asking a webcam for 640x480 @ 60 FPS through Media Foundation and report what each gets.

Usage, from the project root: python -m tools.probe_camera_modes "camera name"

The name is matched against Media Foundation's device list (exactly, or else as a case-insensitive part
of a name), since MSMF indices don't follow the DirectShow order the app's dropdown shows.
"""

import os
import sys
import time

# Without this, MSMF can take many seconds to open a camera.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402

from msmf_cameras import list_msmf_cameras  # noqa: E402

FPS, W, H = cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT

# (name, properties passed when opening, properties set one by one afterwards)
ATTEMPTS = [
    ("MSMF: nothing requested", [], []),
    ("MSMF: set size, then fps", [], [(W, 640), (H, 480), (FPS, 60)]),
    ("MSMF: set fps, then size", [], [(FPS, 60), (W, 640), (H, 480)]),
    ("MSMF: size + fps passed at open", [W, 640, H, 480, FPS, 60], []),
    ("MSMF: size passed at open, no fps", [W, 640, H, 480], []),
]


def probe(index: int, name: str, open_params: list, steps: list) -> None:
    start = time.perf_counter()
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF, open_params)
    if not cap.isOpened():
        print(f"{name:36s} could not open")
        return
    for prop, value in steps:
        cap.set(prop, value)
    opened_s = time.perf_counter() - start
    for _ in range(15):  # Let exposure and the stream settle.
        cap.read()
    frames, read_s, shape, t0 = 0, 0.0, None, time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        r0 = time.perf_counter()
        ok, frame = cap.read()
        read_s += time.perf_counter() - r0
        if ok:
            frames += 1
            shape = frame.shape
    elapsed = time.perf_counter() - t0
    size = f"{shape[1]}x{shape[0]}" if shape else "no frames"
    print(f"{name:36s} reports {int(cap.get(W))}x{int(cap.get(H))} @ {cap.get(FPS):.1f}, frames {size}"
          f"  measured {frames / elapsed:5.1f} FPS  avg read {1000 * read_s / max(frames, 1):.1f} ms"
          f"  (open {opened_s:.1f}s)")
    cap.release()


def find_camera(names: list[str], wanted: str) -> int | None:
    if wanted in names:
        return names.index(wanted)
    matches = [i for i, name in enumerate(names) if wanted.lower() in name.lower()]
    return matches[0] if len(matches) == 1 else None


def main() -> None:
    names = list_msmf_cameras()
    print("Media Foundation cameras:")
    for i, name in enumerate(names):
        print(f"  {i}: {name}")
    if len(sys.argv) < 2:
        sys.exit('\nPass a camera name, e.g. "USB Video Device".')
    index = find_camera(names, sys.argv[1])
    if index is None:
        sys.exit(f"\nNo single camera matches {sys.argv[1]!r}.")
    print(f"\nOpenCV {cv2.__version__}, MSMF camera {index}: {names[index]}\n")
    for attempt in ATTEMPTS:
        probe(index, *attempt)


if __name__ == "__main__":
    main()
