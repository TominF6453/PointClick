"""Click and scroll gesture detection from MediaPipe hand world landmarks.

Left button: thumb tip pinched against the index fingertip.
Right button: closed fist (all four fingertips curled in towards their knuckles).
Scroll: all four fingers held straight and together. The scrolling itself comes from moving the hand,
which world landmarks can't see, so it's handled by the cursor controller.

World landmarks are 3D positions in meters relative to the hand, so distances between them don't
change with hand angle the way 2D image distances do. All distances are divided by the hand's size
(wrist to middle-finger knuckle) so the thresholds work for any hand and camera distance.
"""

import math

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_KNUCKLE = 9
# (knuckle, fingertip) for the index, middle, ring and pinky fingers.
FINGERS = ((5, 8), (9, 12), (13, 16), (17, 20))

# Thumb tip to index tip, relative to hand size.
PINCH_PRESS = 0.3
PINCH_RELEASE = 0.33
# Furthest fingertip-to-knuckle distance of the four fingers, relative to hand size.
FIST_PRESS = 0.40
FIST_RELEASE = 0.43
# The scroll pose is measured as a single value that is 1 at its press threshold: the larger of the
# most bent finger's bend over SCROLL_BEND and the finger spread over SCROLL_SPREAD.
# Bend is 1 minus knuckle-to-tip distance over the summed finger segment lengths (0 when straight).
SCROLL_BEND = 0.10
# Spread is the index-to-pinky fingertip span over the index-to-pinky knuckle span (about 1 when together).
SCROLL_SPREAD = 1.3
SCROLL_PRESS = .8
SCROLL_RELEASE = .9
INDEX_KNUCKLE = 5
PINKY_KNUCKLE = 17
PINKY_TIP = 20
# Consecutive frames a condition must hold before the button changes, to ignore single-frame glitches.
DEBOUNCE_FRAMES = 1
# Click smoothing proximity is measured against the press threshold while a button is up, and the
# release threshold while it's held. It reaches 1 (full smoothing) LEAD before the threshold, so the
# cursor is already steady when the click fires, and falls to 0 a further APPROACH away.
PINCH_LEAD = 0.05
PINCH_APPROACH = 0.30
FIST_LEAD = 0.05
FIST_APPROACH = 0.35


def approach(value: float, threshold: float, width: float, lead: float, from_above: bool) -> float:
    """1 within lead of threshold (or past it), falling to 0 a further width away."""
    gap = (value - threshold if from_above else threshold - value) - lead
    return min(max(1.0 - gap / width, 0.0), 1.0)


def distance(a, b) -> float:
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


class HysteresisButton:
    """Presses when the value stays below one threshold and releases when it stays above a higher one."""

    def __init__(self, press_below: float, release_above: float, frames: int = DEBOUNCE_FRAMES) -> None:
        self.press_below = press_below
        self.release_above = release_above
        self.frames = frames
        self.reset()

    def reset(self) -> None:
        self.pressed = False
        self.count = 0

    def update(self, value: float, can_press: bool = True) -> bool:
        if self.pressed:
            changing = value > self.release_above
        else:
            changing = can_press and value < self.press_below
        self.count = self.count + 1 if changing else 0
        if self.count >= self.frames:
            self.pressed = not self.pressed
            self.count = 0
        return self.pressed


class GestureDetector:
    def __init__(self) -> None:
        self.left = HysteresisButton(PINCH_PRESS, PINCH_RELEASE)
        self.right = HysteresisButton(FIST_PRESS, FIST_RELEASE)
        self.scroll = HysteresisButton(SCROLL_PRESS, SCROLL_RELEASE)
        self.pinch = 0.0
        self.fist = 0.0
        self.flat = 0.0
        self.proximity = 0.0

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()
        self.scroll.reset()
        self.proximity = 0.0

    def update_proximity(self) -> None:
        """How close either button is to changing state, from 0 (far) to 1 (at the threshold)."""
        self.proximity = max(
            self.button_proximity(self.left, self.pinch, PINCH_APPROACH, PINCH_LEAD),
            self.button_proximity(self.right, self.fist, FIST_APPROACH, FIST_LEAD),
        )

    @staticmethod
    def button_proximity(button: HysteresisButton, value: float, width: float, lead: float) -> float:
        # A released button is approached from above its press threshold; a held one from below its
        # release threshold, so a firm pinch during a drag doesn't add smoothing.
        if button.pressed:
            return approach(value, button.release_above, width, lead, from_above=False)
        return approach(value, button.press_below, width, lead, from_above=True)

    @staticmethod
    def flatness(world_landmarks) -> float:
        """How far the four fingers are from being straight and together, 1 at the scroll threshold."""
        bends = []
        for k, t in FINGERS:
            # Landmarks run knuckle, two middle joints, fingertip.
            length = sum(distance(world_landmarks[i], world_landmarks[i + 1]) for i in range(k, t))
            bends.append(1.0 - distance(world_landmarks[k], world_landmarks[t]) / max(length, 1e-6))
        knuckle_span = max(distance(world_landmarks[INDEX_KNUCKLE], world_landmarks[PINKY_KNUCKLE]), 1e-6)
        spread = distance(world_landmarks[INDEX_TIP], world_landmarks[PINKY_TIP]) / knuckle_span
        return max(max(bends) / SCROLL_BEND, spread / SCROLL_SPREAD)

    def update(self, world_landmarks) -> tuple[bool, bool, bool]:
        """Return (left_pressed, right_pressed, scrolling) for one frame's world landmarks."""
        scale = max(distance(world_landmarks[WRIST], world_landmarks[MIDDLE_KNUCKLE]), 1e-6)
        curls = [distance(world_landmarks[k], world_landmarks[t]) / scale for k, t in FINGERS]
        self.pinch = distance(world_landmarks[THUMB_TIP], world_landmarks[INDEX_TIP]) / scale
        self.fist = max(curls)
        self.flat = self.flatness(world_landmarks)

        # The thumb lands near the index fingertip while making a fist, so a pinch only counts while the
        # middle, ring and pinky fingers aren't all curled. Only one of the three can be active at a time.
        others_curled = max(curls[1:]) < FIST_RELEASE
        scrolling = self.scroll.pressed
        left = self.left.update(self.pinch, can_press=not self.right.pressed and not scrolling and not others_curled)
        right = self.right.update(self.fist, can_press=not left and not scrolling)
        scrolling = self.scroll.update(self.flat, can_press=not left and not right)
        self.update_proximity()
        return left, right, scrolling
