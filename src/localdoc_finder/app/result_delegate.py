"""Draws a search hit the way Explorer does: file icon, bold name, full path underneath.

The selected row also shows its snippet. A thin bar under the icon shows the result's relevance;
less relevant results are drawn greyed, and the first of them carries a "Less relevant" heading.
Rows without a ``ResultRow`` (the source list in Ask and Chat mode) fall back to the default
painting.
"""

import time
from collections import OrderedDict
from pathlib import Path
from typing import Generic, TypeVar

from PySide6.QtCore import QFileInfo, QModelIndex, QPersistentModelIndex, QRect, QSize, Qt
from PySide6.QtGui import QFont, QFontMetrics, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileIconProvider,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from localdoc_finder.app.controller import ResultRow
from localdoc_finder.app.theme import secondary_text
from localdoc_finder.core.extractors.image import thumbnail_path

_V = TypeVar("_V")
_ICON_CACHE_SIZE = 512
_THUMB_CACHE_SIZE = 256
_RETRY_THUMBNAIL_SECONDS = 5.0  # a thumbnail may appear once the image has been indexed
ROW_ROLE = Qt.ItemDataRole.UserRole
ICON_SIZE = 32
PADDING = 8
LINE_GAP = 2
GAP = 10
META_MAX_WIDTH = 170
PATH_STRENGTH = 0.55  # how much of the text colour the path and file details keep
SNIPPET_STRENGTH = 0.75
WEAK_STRENGTH = 0.6  # a less relevant result's name
WEAK_ICON_OPACITY = 0.5
DIVIDER_TEXT = "Less relevant"
DIVIDER_STRENGTH = 0.5
BAR_HEIGHT = 3
BAR_GAP = 3  # between the icon and its relevance bar
# Their icon is part of the file (or differs per file), so one icon per extension would be wrong.
PER_FILE_ICON_EXTS = frozenset({"", ".exe", ".lnk", ".ico", ".msi", ".url", ".appx"})


class _BoundedCache(Generic[_V]):
    """A small least-recently-used cache: a list that is scrolled for hours must not grow."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._items: OrderedDict[str, _V] = OrderedDict()

    def get(self, key: str) -> _V | None:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: str, value: _V) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._limit:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


class ResultDelegate(QStyledItemDelegate):
    def __init__(
        self,
        thumbs_dir: Path,
        parent: QWidget | None = None,
        icons: QFileIconProvider | None = None,
    ) -> None:
        super().__init__(parent)
        self._thumbs_dir = thumbs_dir
        self._icons = icons or QFileIconProvider()
        self._cache: _BoundedCache[QIcon] = _BoundedCache(_ICON_CACHE_SIZE)
        self._thumbs: _BoundedCache[QIcon] = _BoundedCache(_THUMB_CACHE_SIZE)
        self._missing: _BoundedCache[float] = _BoundedCache(_THUMB_CACHE_SIZE)  # path -> when

    # ------------------------------------------------------------------ icons
    def icon_for(self, row: ResultRow) -> QIcon:
        thumb = self._thumbnail(row)
        if thumb is not None:
            return thumb
        suffix = Path(row.path).suffix.lower()
        key = row.path.lower() if suffix in PER_FILE_ICON_EXTS else suffix
        icon = self._cache.get(key)
        if icon is None:
            icon = self._icons.icon(QFileInfo(row.path))
            self._cache.put(key, icon)
        return icon

    def _thumbnail(self, row: ResultRow) -> QIcon | None:
        """The image's thumbnail, loaded once (painting asks again for every visible row)."""
        if not row.is_image:
            return None
        cached = self._thumbs.get(row.path)
        if cached is not None:
            return cached
        tried_at = self._missing.get(row.path)
        if tried_at is not None and time.monotonic() - tried_at < _RETRY_THUMBNAIL_SECONDS:
            return None  # no thumbnail a moment ago: do not hit the disk on every repaint
        try:
            pixmap = QPixmap(str(thumbnail_path(self._thumbs_dir, Path(row.path))))
        except OSError:
            pixmap = QPixmap()
        if pixmap.isNull():
            self._missing.put(row.path, time.monotonic())
            return None
        icon = QIcon(pixmap)
        self._thumbs.put(row.path, icon)
        return icon

    # ------------------------------------------------------------------ layout
    @staticmethod
    def _row(index: QModelIndex | QPersistentModelIndex) -> ResultRow | None:
        data = index.data(ROW_ROLE)
        return data if isinstance(data, ResultRow) else None

    def sizeHint(  # noqa: N802
        self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> QSize:
        row = self._row(index)
        if row is None:
            return super().sizeHint(option, index)
        line = QFontMetrics(option.font).height()
        lines = 2 + (1 if self._selected(option, index) and row.snippet else 0)
        height = max(ICON_SIZE + BAR_GAP + BAR_HEIGHT, lines * line + (lines - 1) * LINE_GAP)
        return QSize(0, height + 2 * PADDING + self._divider_height(option, row))

    @staticmethod
    def _divider_height(option: QStyleOptionViewItem, row: ResultRow) -> int:
        return QFontMetrics(option.font).height() + 2 * LINE_GAP if row.divider_above else 0

    @staticmethod
    def _selected(option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex) -> bool:
        """Views do not put the selection into the option when asking for sizes, so ask the view."""
        view = option.widget
        if isinstance(view, QAbstractItemView):
            return view.currentIndex() == index
        return bool(option.state & QStyle.StateFlag.State_Selected)

    # ------------------------------------------------------------------ painting
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        row = self._row(index)
        if row is None:
            super().paint(painter, option, index)
            return
        divider = self._divider_height(option, row)
        if divider:
            self._draw_divider(painter, option, divider)
        background = QStyleOptionViewItem(option)
        self.initStyleOption(background, index)
        background.text = ""
        background.icon = QIcon()
        background.rect = option.rect.adjusted(0, divider, 0, 0)  # the heading is not selectable
        style = option.widget.style() if option.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, background, painter, option.widget)

        painter.save()
        area = background.rect.adjusted(PADDING, PADDING, -PADDING, -PADDING)
        icon_box = QRect(area.left(), area.top(), ICON_SIZE, ICON_SIZE)
        if row.weak:
            painter.setOpacity(WEAK_ICON_OPACITY)
        self.icon_for(row).paint(painter, icon_box)
        painter.setOpacity(1.0)
        self._draw_relevance(painter, option, row, icon_box)
        self._draw_text(painter, option, row, area, self._selected(option, index))
        painter.restore()

    @staticmethod
    def _draw_divider(painter: QPainter, option: QStyleOptionViewItem, height: int) -> None:
        painter.save()
        box = QRect(option.rect.left() + PADDING, option.rect.top(), option.rect.width(), height)
        painter.setPen(secondary_text(option.palette, DIVIDER_STRENGTH))
        painter.drawText(box, Qt.AlignmentFlag.AlignVCenter, DIVIDER_TEXT)
        painter.restore()

    @staticmethod
    def _draw_relevance(
        painter: QPainter, option: QStyleOptionViewItem, row: ResultRow, icon_box: QRect
    ) -> None:
        """A thin bar under the icon, as long as the result is relevant."""
        width = round(ICON_SIZE * max(0, min(row.relevance, 100)) / 100)
        if not width:
            return
        colour = (
            secondary_text(option.palette, DIVIDER_STRENGTH)
            if row.weak
            else option.palette.color(option.palette.ColorRole.Highlight).lighter(170)
        )
        bar = QRect(icon_box.left(), icon_box.bottom() + BAR_GAP, width, BAR_HEIGHT)
        painter.fillRect(bar, colour)

    def _draw_text(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        row: ResultRow,
        area: QRect,
        selected: bool,
    ) -> None:
        left = area.left() + ICON_SIZE + GAP
        bold = QFont(option.font)
        bold.setBold(True)
        bold_metrics, metrics = QFontMetrics(bold), QFontMetrics(option.font)
        meta_width = min(META_MAX_WIDTH, metrics.horizontalAdvance(row.meta)) if row.meta else 0
        first = QRect(
            left, area.top(), area.right() - left - meta_width - GAP, bold_metrics.height()
        )

        painter.setFont(bold)
        text = option.palette.color(option.palette.ColorRole.Text)
        painter.setPen(secondary_text(option.palette, WEAK_STRENGTH) if row.weak else text)
        title = row.name + (f"  {row.detail}" if row.detail else "")
        painter.drawText(
            first,
            Qt.AlignmentFlag.AlignVCenter,
            bold_metrics.elidedText(title, Qt.TextElideMode.ElideRight, first.width()),
        )

        painter.setFont(option.font)
        if row.meta:
            meta_box = QRect(area.right() - meta_width, area.top(), meta_width, first.height())
            painter.setPen(secondary_text(option.palette, PATH_STRENGTH))
            painter.drawText(
                meta_box, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, row.meta
            )

        second = QRect(left, first.bottom() + LINE_GAP, area.right() - left, metrics.height())
        painter.setPen(secondary_text(option.palette, PATH_STRENGTH))
        place = f"{row.project}  {row.path}" if row.project else row.path
        painter.drawText(
            second,
            Qt.AlignmentFlag.AlignVCenter,
            metrics.elidedText(place, Qt.TextElideMode.ElideMiddle, second.width()),
        )

        if selected and row.snippet:
            third = QRect(left, second.bottom() + LINE_GAP, area.right() - left, metrics.height())
            painter.setPen(secondary_text(option.palette, SNIPPET_STRENGTH))
            painter.drawText(
                third,
                Qt.AlignmentFlag.AlignVCenter,
                metrics.elidedText(
                    row.snippet.replace("\n", " "), Qt.TextElideMode.ElideRight, third.width()
                ),
            )
