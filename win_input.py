"""Thin ctypes wrappers around the Win32 calls used for cursor control."""

import ctypes

user32 = ctypes.windll.user32

SM_CXSCREEN = 0
SM_CYSCREEN = 1
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)


def enable_dpi_awareness() -> None:
    # Without this, Windows scales coordinates on high-DPI displays and the cursor can't reach every pixel.
    try:
        user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
    except (AttributeError, OSError):
        pass


def primary_screen_size() -> tuple[int, int]:
    return user32.GetSystemMetrics(SM_CXSCREEN), user32.GetSystemMetrics(SM_CYSCREEN)


def move_cursor(x: int, y: int) -> None:
    # The primary monitor's top-left corner is always (0, 0) in screen coordinates.
    user32.SetCursorPos(x, y)


MOUSE_BUTTON_FLAGS = {
    # (down, up) flags for mouse_event.
    "left": (0x0002, 0x0004),
    "right": (0x0008, 0x0010),
}


def mouse_button(button: str, down: bool) -> None:
    """Press or release a mouse button at the current cursor position."""
    # mouse_event is superseded by SendInput but still supported, and much simpler to call.
    down_flag, up_flag = MOUSE_BUTTON_FLAGS[button]
    user32.mouse_event(down_flag if down else up_flag, 0, 0, 0, 0)


MOUSEEVENTF_WHEEL = 0x0800
# One notch of a standard mouse wheel.
WHEEL_DELTA = 120


def scroll_wheel(amount: int) -> None:
    """Turn the vertical wheel by amount (WHEEL_DELTA per notch, positive scrolls up)."""
    user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, amount, 0)


def is_key_down(virtual_key: int) -> bool:
    return bool(user32.GetAsyncKeyState(virtual_key) & 0x8000)
