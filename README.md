# PointClick

Webcam hand tracking with MediaPipe Hand Landmarker. It moves the mouse cursor with your hand and clicks with hand gestures, and shows the webcam feed with the 21 detected hand landmarks drawn on top.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Run

```bash
.venv\Scripts\python point_click.py
```

On first run the model (`hand_landmarker.task`, about 7.5 MB) is downloaded into `models/`. Pick the webcam from the **Camera** dropdown; **Refresh** rescans for devices you've plugged in since starting.

Cameras are opened through Media Foundation, matched by name, and asked for 640×480 at 60 FPS (`FRAME_RATE` in `point_click.py`); cameras that can't do that use their nearest supported rate. Cameras Media Foundation doesn't list, such as some virtual cameras, open through DirectShow instead. The line above the timings at the bottom of the video shows which backend and mode the camera ended up in. The last camera you picked is saved by name in `settings.json` and reopened on the next start. Press **Q** or **Esc**, or close the window, to quit.

Tick **Enable Preview** or press **F7** (works from any window) to show or hide the video, along with the landmarks, gesture meters, FPS and timings drawn on it. Hand tracking and cursor control keep running while it's hidden, and skipping the drawing saves a little CPU. The input rectangle can only be moved or resized while the preview is shown.

### Cursor control

The rectangle on the video is the input area. Drag inside it to move it, or drag a corner to resize it. It's locked while cursor control is on. Its position is saved in `settings.json`.

Tick **Control cursor** or press **F8** (works from any window) to turn cursor control on or off. While it's on, the position of your index-finger knuckle (circled in magenta) inside the rectangle sets the cursor's position on the primary monitor, and the rectangle turns green. When no hand is visible the cursor is left alone, so your real mouse still works.

### Clicking

- **Left button:** pinch your thumb tip against your index fingertip. Hold the pinch and move to drag; open it to release.
- **Right button:** make a fist, curling all four fingers in. Open your hand to release.

### Scrolling

Hold all four fingers straight and pressed together (the thumb doesn't matter), then move your hand up to scroll down or down to scroll up, like dragging the page on a touchscreen. The cursor stays where it was while you scroll, so the scroll goes to the window under it (or the focused window, depending on Windows' "Scroll inactive windows" setting). Spread or bend your fingers to stop. `SCROLL_GAIN` in `cursor_control.py` sets the speed, and `SCROLL_BEND` and `SCROLL_SPREAD` in `gestures.py` set how straight and how close together the fingers must be. Scrolling can't start while a button is held, and buttons can't be pressed while scrolling.

### Gesture meters

The gestures are measured from MediaPipe's 3D hand landmarks, so they don't depend on the angle of your hand. The three bars at the bottom left of the video show the pinch, fist and scroll-pose measurements. A bar turns green while its button is held; the red mark is where it presses and the yellow mark is where it releases. The thresholds are constants at the top of `gestures.py`.

Buttons press and release the moment their threshold is crossed. To keep clicks accurate, smoothing gets much stronger as a gesture nears its threshold, so the cursor steadies where the click lands without freezing. The **Click smoothing** readout above the bars shows how much is applied. While a button is held, it's measured against the release threshold instead, so a firm pinch drags smoothly. Smoothing reaches full strength slightly before each threshold (`PINCH_LEAD` and `FIST_LEAD` in `gestures.py`), so the cursor is already steady when the click fires. `PINCH_APPROACH` and `FIST_APPROACH` set how early it starts, and `CLICK_SMOOTHING` in `point_click.py` sets how strong it gets. Only one button can be held at a time. Held buttons are released if your hand leaves view, if you turn cursor control off, or if you close the app.

### Smoothing

The knuckle position is smoothed with a One Euro filter (`one_euro.py`) before it moves the cursor. The cyan cross on the video is the smoothed position; compare it with the raw magenta circle. Tune it live with the two sliders, which are saved in `settings.json`:

1. Set **Beta** to 0 and lower **Min cutoff** until the cross stays still while your hand is still.
2. Raise **Beta** until the cross keeps up with fast movements.

Settings such as the default camera, resolution and number of hands are constants at the top of `point_click.py`.

## Tools

`tools/probe_camera_modes.py` tries several ways of opening a camera at 640×480 @ 60 FPS through Media Foundation and reports the frame size and frame rate each one really gets. Close PointClick first so the camera is free, then run it from the project root with the camera's name (or a unique part of it):

```bash
.venv\Scripts\python -m tools.probe_camera_modes "USB Video Device"
```

`msmf_cameras.py` lists cameras in Media Foundation's order (`.venv\Scripts\python msmf_cameras.py`), which can differ from the DirectShow order shown in the app's dropdown.
