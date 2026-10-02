"""Global hotkey registration (Win32 ``RegisterHotKey``) delivered through Qt's native filter."""

import ctypes
from collections.abc import Callable
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter

WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000
MODIFIERS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
VIRTUAL_KEYS = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B}
_FUNCTION_KEY_BASE = 0x70  # VK_F1


def parse_hotkey(spec: str) -> tuple[int, int]:
    """``'ctrl+alt+space'`` -> ``(modifier mask, virtual key)``."""
    modifiers, key = 0, None
    for part in spec.lower().replace(" ", "").split("+"):
        if part in MODIFIERS:
            modifiers |= MODIFIERS[part]
        elif part in VIRTUAL_KEYS:
            key = VIRTUAL_KEYS[part]
        elif len(part) == 1:
            key = ord(part.upper())
        elif part.startswith("f") and part[1:].isdigit():
            key = _FUNCTION_KEY_BASE + int(part[1:]) - 1
        else:
            raise ValueError(f"unknown key in hotkey: {part}")
    if key is None:
        raise ValueError("hotkey needs a non-modifier key")
    return modifiers, key


class HotkeyFilter(QAbstractNativeEventFilter):
    HOTKEY_ID = 0x5645  # "VE"

    def __init__(self, callback: Callable[[], None]) -> None:
        super().__init__()
        self.callback = callback
        self.registered = False

    def register(self, spec: str) -> bool:
        modifiers, key = parse_hotkey(spec)
        user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
        self.registered = bool(
            user32.RegisterHotKey(None, self.HOTKEY_ID, modifiers | MOD_NOREPEAT, key)
        )
        return self.registered

    def unregister(self) -> None:
        if self.registered:
            ctypes.windll.user32.UnregisterHotKey(None, self.HOTKEY_ID)  # type: ignore[attr-defined,unused-ignore]
            self.registered = False

    def nativeEventFilter(self, eventType: object, message: object) -> tuple[bool, int]:  # noqa: N802, N803
        if eventType == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))  # type: ignore[call-overload]
            if msg.message == WM_HOTKEY and msg.wParam == self.HOTKEY_ID:
                self.callback()
                return True, 0
        return False, 0
