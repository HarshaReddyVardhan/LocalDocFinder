"""The app's light and dark themes: Fusion plus an explicit palette, so every window agrees.

Qt follows Windows' dark mode only halfway (light text on parts that stay painted white), so
the app pins the Fusion style and its own palette. The user picks ``system`` (follow Windows),
``light`` or ``dark`` in Settings; the choice applies to every window at once.
"""

import enum
import tempfile
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QCheckBox, QRadioButton, QStyleFactory, QWidget

THEME_CHOICES = ("system", "light", "dark")


class Scheme(enum.Enum):
    LIGHT = "light"
    DARK = "dark"


_COLOURS: dict[Scheme, dict[QPalette.ColorRole, str]] = {
    Scheme.LIGHT: {
        QPalette.ColorRole.Window: "#f6f6f9",
        QPalette.ColorRole.WindowText: "#1b1b1f",
        QPalette.ColorRole.Base: "#ffffff",
        QPalette.ColorRole.AlternateBase: "#f3f3f5",
        QPalette.ColorRole.Text: "#1b1b1f",
        QPalette.ColorRole.Button: "#f0f0f3",
        QPalette.ColorRole.ButtonText: "#1b1b1f",
        QPalette.ColorRole.ToolTipBase: "#ffffff",
        QPalette.ColorRole.ToolTipText: "#1b1b1f",
        QPalette.ColorRole.PlaceholderText: "#6b6e78",
        QPalette.ColorRole.Highlight: "#4c7dff",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: "#2a5bd7",
    },
    Scheme.DARK: {
        QPalette.ColorRole.Window: "#1e1f24",
        QPalette.ColorRole.WindowText: "#e6e6e6",
        QPalette.ColorRole.Base: "#17181c",
        QPalette.ColorRole.AlternateBase: "#22242a",
        QPalette.ColorRole.Text: "#e6e6e6",
        QPalette.ColorRole.Button: "#2a2c33",
        QPalette.ColorRole.ButtonText: "#e6e6e6",
        QPalette.ColorRole.ToolTipBase: "#2a2c33",
        QPalette.ColorRole.ToolTipText: "#e6e6e6",
        QPalette.ColorRole.PlaceholderText: "#8a8f9c",
        QPalette.ColorRole.Highlight: "#4c7dff",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: "#7fa2ff",
    },
}
# Fusion draws frames, radio buttons and separators from these; the defaults vanish on dark.
_FRAME_SHADES: dict[Scheme, dict[QPalette.ColorRole, str]] = {
    Scheme.LIGHT: {},
    Scheme.DARK: {
        QPalette.ColorRole.Light: "#4a4d57",
        QPalette.ColorRole.Midlight: "#3b3e47",
        QPalette.ColorRole.Mid: "#4a4d57",
        QPalette.ColorRole.Dark: "#5b5f6b",
        QPalette.ColorRole.Shadow: "#0c0d10",
    },
}
_DISABLED_TEXT = {Scheme.LIGHT: "#9a9da8", Scheme.DARK: "#6b6e78"}
_TICK_SIZE = 15
_TICK_FILE = "vector_embed_tick.png"
# Fusion can paint a check box's frame the same colour as its background, so draw it ourselves.
# Applied to each check box itself: a style sheet on a window would also reset its palette.
_CHECKBOX_STYLE = """
QCheckBox {{ color:{text}; background:transparent; spacing:8px; }}
QCheckBox:disabled {{ color:{disabled}; }}
QCheckBox::indicator {{ width:{size}px; height:{size}px; border:1px solid {frame};
                        border-radius:3px; background:{base}; }}
QCheckBox::indicator:checked {{ background:#4c7dff; border-color:#4c7dff; image:url({tick}); }}
QCheckBox::indicator:disabled {{ border-color:{disabled_frame}; background:{button}; }}
QRadioButton {{ color:{text}; background:transparent; spacing:8px; }}
QRadioButton:disabled {{ color:{disabled}; }}
QRadioButton::indicator {{ width:{size}px; height:{size}px; border:1px solid {frame};
                           border-radius:{radius}px; background:{base}; }}
QRadioButton::indicator:checked {{ border-color:#4c7dff;
    background:qradialgradient(cx:.5, cy:.5, radius:.5, fx:.5, fy:.5,
                               stop:0 #4c7dff, stop:.5 #4c7dff, stop:.6 {base}, stop:1 {base}); }}
QRadioButton::indicator:disabled {{ border-color:{disabled_frame}; background:{button}; }}
"""
_CHECKBOX_FRAME = {Scheme.LIGHT: "#7a7d88", Scheme.DARK: "#8a8f9c"}
_CHECKBOX_DISABLED_FRAME = {Scheme.LIGHT: "#c4c6cc", Scheme.DARK: "#3b3e47"}

# The search popup's own sheet: big input, flat result list. Colours come from the scheme.
_POPUP_COLOURS = {
    Scheme.DARK: {
        "bg": "#1e1f24",
        "fg": "#e6e6e6",
        "input": "#2a2c33",
        "edge": "#3b3e47",
        "row": "#2a2c33",
        "selected": "#33405a",
        "answer": "#17181c",
        "muted": "#8a8f9c",
    },
    Scheme.LIGHT: {
        "bg": "#f6f6f9",
        "fg": "#1b1b1f",
        "input": "#ffffff",
        "edge": "#c9cbd3",
        "row": "#e4e5ea",
        "selected": "#d5e0ff",
        "answer": "#ffffff",
        "muted": "#5f6370",
    },
}
_POPUP_STYLE = """
QWidget {{ background:{bg}; color:{fg}; font-size:13px; }}
QLineEdit {{ background:{input}; border:1px solid {edge}; border-radius:6px;
            padding:9px 12px; font-size:16px; }}
QListWidget {{ background:{bg}; border:none; outline:0; }}
QListWidget::item {{ padding:6px 8px; border-bottom:1px solid {row}; }}
QListWidget::item:selected {{ background:{selected}; }}
QTextBrowser {{ background:{answer}; border:1px solid {row}; font-size:13px; }}
QLabel#status {{ color:{muted}; padding:2px 6px; }}
QLabel#mode {{ color:#4c7dff; font-weight:bold; padding:0 8px; }}
"""


def system_scheme() -> Scheme:
    """Windows' current light/dark choice (light when Qt cannot tell)."""
    hints = QGuiApplication.styleHints()
    if hints is not None and hints.colorScheme() == Qt.ColorScheme.Dark:
        return Scheme.DARK
    return Scheme.LIGHT


def resolve_scheme(choice: str) -> Scheme:
    """``system`` follows Windows; ``light`` and ``dark`` are fixed."""
    if choice == "light":
        return Scheme.LIGHT
    if choice == "dark":
        return Scheme.DARK
    return system_scheme()


def palette_for(scheme: Scheme) -> QPalette:
    colours = _COLOURS[scheme]
    base = colours[QPalette.ColorRole.Button], colours[QPalette.ColorRole.Base]
    palette = QPalette(QColor(base[0]), QColor(base[1]))  # derives the frame/shadow shades
    for role, colour in {**colours, **_FRAME_SHADES[scheme]}.items():
        palette.setColor(role, QColor(colour))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(_DISABLED_TEXT[scheme]))
    return palette


def scheme_in_use() -> Scheme:
    """The scheme the application palette currently shows."""
    window = QApplication.palette().color(QPalette.ColorRole.Window)
    return Scheme.DARK if window.lightness() < 128 else Scheme.LIGHT


def popup_style(scheme: Scheme) -> str:
    return _POPUP_STYLE.format(**_POPUP_COLOURS[scheme])


def secondary_text(palette: QPalette, strength: float) -> QColor:
    """Text colour faded toward the background: ``strength`` 1 is full text, 0 is invisible."""
    text = palette.color(QPalette.ColorRole.Text)
    base = palette.color(QPalette.ColorRole.Base)
    mix = [
        round(t * strength + b * (1 - strength))
        for t, b in zip(
            (text.red(), text.green(), text.blue()),
            (base.red(), base.green(), base.blue()),
            strict=True,
        )
    ]
    return QColor(*mix)


def _tick_image() -> str:
    """A white check mark drawn once into the temp folder (style sheets can only load files)."""
    path = Path(tempfile.gettempdir()) / _TICK_FILE
    pixmap = QPixmap(_TICK_SIZE, _TICK_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("#ffffff"), 2))
    painter.drawPolyline([QPoint(3, 8), QPoint(6, 11), QPoint(12, 4)])
    painter.end()
    pixmap.save(str(path))
    return path.as_posix()


def style_check_boxes(widget: QWidget, scheme: Scheme) -> None:
    """Give every check box and radio button inside ``widget`` an explicit frame for ``scheme``."""
    colours = _COLOURS[scheme]
    sheet = _CHECKBOX_STYLE.format(
        size=_TICK_SIZE,
        radius=_TICK_SIZE // 2,
        tick=_tick_image(),
        text=colours[QPalette.ColorRole.Text],
        base=colours[QPalette.ColorRole.Base],
        button=colours[QPalette.ColorRole.Button],
        disabled=_DISABLED_TEXT[scheme],
        frame=_CHECKBOX_FRAME[scheme],
        disabled_frame=_CHECKBOX_DISABLED_FRAME[scheme],
    )
    for box in (*widget.findChildren(QCheckBox), *widget.findChildren(QRadioButton)):
        box.setStyleSheet(sheet)


def apply_theme(app: QApplication, choice: str) -> Scheme:
    """Pin Fusion and the palette for ``choice`` on the whole application."""
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    scheme = resolve_scheme(choice)
    app.setPalette(palette_for(scheme))
    for widget in app.topLevelWidgets():
        style_check_boxes(widget, scheme)
    return scheme
