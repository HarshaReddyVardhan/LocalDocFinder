"""The app's light and dark themes: Fusion plus an explicit palette, so every window agrees.

Qt follows Windows' dark mode only halfway (light text on parts that stay painted white), so
the app pins the Fusion style and its own palette. The user picks ``system`` (follow Windows),
``light`` or ``dark`` in Settings; the choice applies to every window at once.
"""

import enum
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPalette, QPen, QPixmap
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
_CHEVRON_SIZE = 20  # drawn at 2x and shown at 10 px, so it stays sharp on high-DPI screens
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

# The search popup: a rounded, softly shadowed card (painted by the window, see ``CardColours``)
# whose controls are styled here. Launcher conventions: a large borderless field, rounded
# selection, pills for the mode, thin scroll bars, key hints in the footer.
_POPUP_COLOURS = {
    Scheme.LIGHT: {
        "fg": "#1c2030",
        "muted": "#687085",
        "accent": "#3b6cf6",
        "accent_hover": "#2f5be0",
        "field": "#eef1f6",
        "field_focus": "#ffffff",
        "focus_edge": "#b8c8fb",
        "edge": "#e3e7ef",
        "surface": "#fbfcfe",
        "hover": "#f1f3f8",
        "selected": "#e6edff",
        "button": "#f3f5f9",
        "button_hover": "#e9edf5",
        "button_down": "#dfe5f1",
        "scroll": "#c5cbd8",
        "menu": "#ffffff",
    },
    Scheme.DARK: {
        "fg": "#e8eaf0",
        "muted": "#9097a8",
        "accent": "#5b84ff",
        "accent_hover": "#7097ff",
        "field": "#1b1d24",
        "field_focus": "#16181e",
        "focus_edge": "#3f5596",
        "edge": "#363a46",
        "surface": "#1d1f26",
        "hover": "#2c2f39",
        "selected": "#2f3b5c",
        "button": "#2c2f39",
        "button_hover": "#353946",
        "button_down": "#3d4252",
        "scroll": "#4a4f5e",
        "menu": "#25272f",
    },
}
_POPUP_STYLE = """
* {{ font-family:"Segoe UI Variable Text","Segoe UI",sans-serif; font-size:13px; color:{fg}; }}
QFrame#card {{ background:transparent; }}
QLineEdit#query {{ background:{field}; border:1px solid {edge}; border-radius:12px;
    padding:10px 14px 10px 6px; font-size:17px; selection-background-color:{accent};
    selection-color:#ffffff; }}
QLineEdit#query:focus {{ background:{field_focus}; border-color:{focus_edge}; }}
QPushButton#modePill {{ background:transparent; border:1px solid transparent; border-radius:13px;
    padding:4px 13px; color:{muted}; font-size:12px; font-weight:600; }}
QPushButton#modePill:hover {{ background:{hover}; color:{fg}; }}
QPushButton#modePill:checked {{ background:{accent}; color:#ffffff; }}
QPushButton#close {{ background:transparent; border:1px solid transparent; border-radius:12px;
    color:{muted}; font-size:12px; padding:0; min-width:24px; min-height:24px; }}
QPushButton#close:hover {{ background:{hover}; color:{fg}; }}
QLabel#modeHint, QLabel#status, QLabel#hints {{ color:{muted}; font-size:12px; }}
QListWidget {{ background:transparent; border:none; outline:0; }}
QListWidget::item {{ padding:6px 8px; margin:1px 0; border-radius:10px; }}
QListWidget::item:hover {{ background:{hover}; }}
QListWidget::item:selected {{ background:{selected}; color:{fg}; }}
QTextBrowser {{ background:{surface}; border:1px solid {edge}; border-radius:12px;
    padding:8px 10px; font-size:13.5px; }}
QPushButton {{ background:{button}; border:1px solid {edge}; border-radius:9px;
    padding:5px 12px; }}
QPushButton:hover {{ background:{button_hover}; }}
QPushButton:pressed {{ background:{button_down}; }}
QPushButton:disabled {{ color:{muted}; }}
QComboBox {{ background:{button}; border:1px solid {edge}; border-radius:9px;
    padding:4px 8px 4px 10px; }}
QComboBox:hover {{ background:{button_hover}; }}
QComboBox::drop-down {{ border:none; width:22px; }}
QComboBox::down-arrow {{ image:url({chevron}); width:10px; height:10px; }}
QComboBox QAbstractItemView {{ background:{menu}; border:1px solid {edge};
    selection-background-color:{selected}; selection-color:{fg}; outline:0; }}
QTableWidget {{ background:{surface}; border:1px solid {edge}; border-radius:12px;
    gridline-color:{edge}; selection-background-color:{selected}; selection-color:{fg}; }}
QHeaderView {{ background:transparent; }}
QHeaderView::section {{ background:transparent; color:{muted}; border:none;
    border-bottom:1px solid {edge}; padding:6px 8px; font-weight:600; }}
QTableCornerButton::section {{ background:transparent; border:none; }}
QSplitter::handle {{ background:transparent; }}
QScrollBar:vertical {{ background:transparent; width:10px; margin:2px; }}
QScrollBar:horizontal {{ background:transparent; height:10px; margin:2px; }}
QScrollBar::handle {{ background:{scroll}; border-radius:3px; min-height:28px; min-width:28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background:transparent; }}
QMenu {{ background:{menu}; border:1px solid {edge}; padding:4px; }}
QMenu::item {{ padding:6px 14px; border-radius:6px; }}
QMenu::item:selected {{ background:{selected}; }}
QToolTip {{ background:{menu}; color:{fg}; border:1px solid {edge}; padding:4px 6px; }}
"""


@dataclass(frozen=True)
class CardColours:
    """What the popup paints itself: a glossy top-to-bottom gradient, a hairline and a shadow."""

    top: QColor
    bottom: QColor
    edge: QColor
    highlight: QColor  # the 1px sheen along the top edge
    shadow: QColor  # the darkest shadow ring; outer rings fade from it


_CARD = {
    Scheme.LIGHT: ("#ffffff", "#f5f7fb", (20, 28, 48, 30), (255, 255, 255, 255), (16, 24, 40, 34)),
    Scheme.DARK: ("#2b2e37", "#23252d", (255, 255, 255, 22), (255, 255, 255, 18), (0, 0, 0, 90)),
}


def card_colours(scheme: Scheme) -> CardColours:
    top, bottom, edge, highlight, shadow = _CARD[scheme]
    return CardColours(
        QColor(top), QColor(bottom), QColor(*edge), QColor(*highlight), QColor(*shadow)
    )


def search_icon(scheme: Scheme, size: int = 18) -> QIcon:
    """A magnifier drawn at the screen's resolution, so it stays crisp at any scaling."""
    screen = QGuiApplication.primaryScreen()
    ratio = screen.devicePixelRatio() if screen is not None else 1.0
    pixmap = QPixmap(round(size * ratio), round(size * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(_POPUP_COLOURS[scheme]["muted"]), 1.8)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawEllipse(QRectF(2.5, 2.5, size * 0.55, size * 0.55))
    end = size - 2.5
    start = 2.5 + size * 0.55 * 0.85
    painter.drawLine(QPointF(start, start), QPointF(end, end))
    painter.end()
    return QIcon(pixmap)


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
    colours = _POPUP_COLOURS[scheme]
    return _POPUP_STYLE.format(**colours, chevron=_chevron_image(scheme, colours["muted"]))


def _chevron_image(scheme: Scheme, colour: str) -> str:
    """A small down chevron for combo boxes (a styled combo box draws no arrow of its own)."""
    path = Path(tempfile.gettempdir()) / f"vector_embed_chevron_{scheme.value}.png"
    pixmap = QPixmap(_CHEVRON_SIZE, _CHEVRON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(colour), 2.4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    third = _CHEVRON_SIZE / 3
    painter.drawPolyline(
        [
            QPointF(third * 0.6, third * 1.1),
            QPointF(_CHEVRON_SIZE / 2, third * 2.1),
            QPointF(_CHEVRON_SIZE - third * 0.6, third * 1.1),
        ]
    )
    painter.end()
    pixmap.save(str(path))
    return path.as_posix()


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
