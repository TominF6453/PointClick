"""Turns a target cursor position and button states into Windows cursor moves and clicks."""

import win_input


class CursorController:
    def __init__(self) -> None:
        self.buttons = {"left": False, "right": False}
        self.last_pos: tuple[int, int] | None = None

    def update(self, target: tuple[int, int], left: bool, right: bool) -> None:
        """Move the cursor to target, then apply the wanted button states there."""
        if target != self.last_pos:
            win_input.move_cursor(*target)
            self.last_pos = target
        for button, wanted in (("left", left), ("right", right)):
            if wanted != self.buttons[button]:
                win_input.mouse_button(button, wanted)
                self.buttons[button] = wanted

    def release_all(self) -> None:
        """Release any held buttons and stop tracking the cursor position."""
        for button, down in self.buttons.items():
            if down:
                win_input.mouse_button(button, False)
                self.buttons[button] = False
        self.last_pos = None
