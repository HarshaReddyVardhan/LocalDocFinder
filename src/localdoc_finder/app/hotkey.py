"""Global hotkey registration (Win32 ``RegisterHotKey``) delivered through Qt's native filter."""

import ctypes
from collections.abc import Callable
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter

WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000
MODIFIERS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
VIRTUAL_KEYS = {
    "space": 0x20,
    "enter": 0x0D,
    "tab": 0x09,
    "esc": 0x1B,
    "backspace": 0x08,
    "delete": 0x2E,
    "insert": 0x2D,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "pause": 0x13,
    # Names for the characters that cannot stand alone in a "+" separated spec.
    "plus": 0xBB,
    "minus": 0xBD,
    "comma": 0xBC,
    "period": 0xBE,
}
# Punctuation keys are not their ASCII code: Windows numbers them as "OEM" keys (US layout).
PUNCTUATION_KEYS = {
    ";": 0xBA,
    "=": 0xBB,
    ",": 0xBC,
    "-": 0xBD,
    ".": 0xBE,
    "/": 0xBF,
    "`": 0xC0,
    "[": 0xDB,
    "\\": 0xDC,
    "]": 0xDD,
    "'": 0xDE,
}
_FUNCTION_KEY_BASE = 0x70  # VK_F1
_FUNCTION_KEYS = range(1, 25)  # F1 to F24; Windows has no others


def parse_hotkey(spec: str) -> tuple[int, int]:
    """``'ctrl+alt+space'`` -> ``(modifier mask, virtual key)``.

    A key with no modifier (other than F1 to F24) is refused: it would take that key away from
    every program on the machine.
    """
    modifiers, key, function_key = 0, None, False
    for part in spec.lower().replace(" ", "").split("+"):
        if part in MODIFIERS:
            modifiers |= MODIFIERS[part]
        elif part in VIRTUAL_KEYS:
            key = VIRTUAL_KEYS[part]
        elif len(part) == 1 and part.isalnum() and part.isascii():
            key = ord(part.upper())  # letters and digits are their own virtual-key codes
        elif part in PUNCTUATION_KEYS:
            key = PUNCTUATION_KEYS[part]
        elif part.startswith("f") and part[1:].isdigit() and int(part[1:]) in _FUNCTION_KEYS:
            key = _FUNCTION_KEY_BASE + int(part[1:]) - 1
            function_key = True
        else:
            raise ValueError(f"unknown key in hotkey: {part}")
    if key is None:
        raise ValueError("hotkey needs a non-modifier key")
    if not modifiers and not function_key:
        raise ValueError("hotkey needs a modifier such as ctrl or alt")
    return modifiers, key


class HotkeyFilter(QAbstractNativeEventFilter):
    HOTKEY_IDS = (0x5645, 0x5646)  # "VE": two, so a new key is claimed before the old is freed

    def __init__(self, callback: Callable[[], None]) -> None:
        super().__init__()
        self.callback = callback
        self.registered = False
        self.spec: str | None = None  # the hotkey that is actually active
        self._active_id: int | None = None

    def register(self, spec: str) -> bool:
        """Claim ``spec``; the previous hotkey is released only once the new one is secured.

        If Windows refuses the new one (another program owns it), the old hotkey keeps working
        and ``False`` is returned. An invalid ``spec`` raises ``ValueError`` before anything
        changes.
        """
        modifiers, key = parse_hotkey(spec)
        user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
        spare = next(i for i in self.HOTKEY_IDS if i != self._active_id)
        if not user32.RegisterHotKey(None, spare, modifiers | MOD_NOREPEAT, key):
            return False
        self.unregister()  # now the old one can go
        self._active_id, self.registered, self.spec = spare, True, spec
        return True

    def unregister(self) -> None:
        if self._active_id is not None:
            ctypes.windll.user32.UnregisterHotKey(None, self._active_id)  # type: ignore[attr-defined,unused-ignore]
        self._active_id, self.registered, self.spec = None, False, None

    def nativeEventFilter(self, eventType: object, message: object) -> tuple[bool, int]:  # noqa: N802, N803
        if eventType == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))  # type: ignore[call-overload]
            if msg.message == WM_HOTKEY and msg.wParam == self._active_id:
                self.callback()
                return True, 0
        return False, 0
