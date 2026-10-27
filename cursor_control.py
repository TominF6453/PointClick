"""Turns a target cursor position, button states and scroll movement into Windows cursor moves, clicks and scrolls."""

import win_input

# Wheel units per frame height of vertical hand movement while scrolling (about 25 notches).
SCROLL_GAIN = 25 * win_input.WHEEL_DELTA


class CursorController:
    def __init__(self) -> None:
        self.buttons = {"left": False, "right": False}
        self.last_pos: tuple[int, int] | None = None
        self.last_scroll_y: float | None = None
        # Wheel units not sent yet, since each frame's movement rarely comes to a whole unit.
        self.scroll_remainder = 0.0

    def update(self, target: tuple[int, int], left: bool, right: bool, scroll_y: float | None = None) -> None:
        """Move the cursor to target, then apply the wanted button states there.

        While scroll_y (the hand's height in normalized frame coordinates) is given, the cursor stays put
        so the scroll lands on the window it's over, and the hand's vertical movement scrolls instead.
        """
        if scroll_y is not None:
            self.scroll(scroll_y)
            return
        self.stop_scroll()
        if target != self.last_pos:
            win_input.move_cursor(*target)
            self.last_pos = target
        for button, wanted in (("left", left), ("right", right)):
            if wanted != self.buttons[button]:
                win_input.mouse_button(button, wanted)
                self.buttons[button] = wanted

    def scroll(self, y: float) -> None:
        if self.last_scroll_y is not None:
            # Frame y grows downwards and a positive wheel scrolls up, so moving the hand up scrolls
            # down, dragging the content along with it like a touchscreen.
            self.scroll_remainder += (y - self.last_scroll_y) * SCROLL_GAIN
            amount = int(self.scroll_remainder)
            if amount:
                win_input.scroll_wheel(amount)
                self.scroll_remainder -= amount
        self.last_scroll_y = y

    def stop_scroll(self) -> None:
        self.last_scroll_y = None
        self.scroll_remainder = 0.0

    def release_all(self) -> None:
        """Release any held buttons, stop scrolling and stop tracking the cursor position."""
        for button, down in self.buttons.items():
            if down:
                win_input.mouse_button(button, False)
                self.buttons[button] = False
        self.stop_scroll()
        self.last_pos = None
