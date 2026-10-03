"""A fixed light theme for the setup wizard.

Qt follows Windows' dark mode (light text), but the wizard's page area is always painted
white, which left its text invisible. Pinning the Fusion style and a light palette makes the
wizard readable under either Windows theme.
"""

import tempfile
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QCheckBox, QStyleFactory, QWidget

_ROLES = {
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
}
_DISABLED_TEXT = "#9a9da8"
_TICK_SIZE = 15
_TICK_FILE = "vector_embed_tick.png"
# Fusion paints a check box's frame white-on-white when Windows is in dark mode, so draw it.
# Applied to each check box itself: a style sheet on the wizard would also reset its palette.
_CHECKBOX_STYLE = """
QCheckBox {{ color:#1b1b1f; background:transparent; spacing:8px; }}
QCheckBox:disabled {{ color:#9a9da8; }}
QCheckBox::indicator {{ width:{size}px; height:{size}px; border:1px solid #7a7d88;
                        border-radius:3px; background:#ffffff; }}
QCheckBox::indicator:checked {{ background:#4c7dff; border-color:#4c7dff; image:url({tick}); }}
QCheckBox::indicator:disabled {{ border-color:#c4c6cc; background:#f0f0f3; }}
"""


def light_palette() -> QPalette:
    palette = QPalette(QColor("#f0f0f3"), QColor("#ffffff"))  # derives the frame/shadow shades
    for role, colour in _ROLES.items():
        palette.setColor(role, QColor(colour))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(_DISABLED_TEXT))
    return palette


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


def apply_light_theme(widget: QWidget) -> None:
    """Style ``widget`` and every widget already inside it in the light theme.

    Call it after the child widgets exist: Qt does not hand a widget's style on to children.
    """
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        fusion.setParent(widget)  # a widget does not own its style; without this it is freed
        for target in (widget, *widget.findChildren(QWidget)):
            target.setStyle(fusion)
    widget.setPalette(light_palette())
    sheet = _CHECKBOX_STYLE.format(size=_TICK_SIZE, tick=_tick_image())
    for box in widget.findChildren(QCheckBox):
        box.setStyleSheet(sheet)
