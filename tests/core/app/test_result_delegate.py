import time
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QFileInfo, Qt
from PySide6.QtGui import (
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPalette,
    QPixmap,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import QApplication, QFileIconProvider, QStyle, QStyleOptionViewItem

from localdoc_finder.app.controller import ResultRow
from localdoc_finder.app.result_delegate import ICON_SIZE, ROW_ROLE, ResultDelegate
from localdoc_finder.core.extractors.image import thumbnail_path


def make_row(**overrides: object) -> ResultRow:
    fields: dict[str, object] = {
        "name": "report.pdf",
        "detail": "(page 4)",
        "path": r"D:\docs\report.pdf",
        "project": "docs",
        "snippet": "quarterly revenue grew",
        "tag": "PDF",
        "modified": "2026-09-28",
        "size": "1.2 MB",
        "is_image": False,
    }
    fields.update(overrides)
    return ResultRow(**fields)  # type: ignore[arg-type]


class CountingIcons(QFileIconProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def icon(self, info: object) -> QIcon:  # type: ignore[override]
        assert isinstance(info, QFileInfo)
        self.calls.append(info.filePath())
        return super().icon(QFileIconProvider.IconType.File)


@pytest.fixture
def delegate(qapp: QApplication, tmp_path: Path) -> ResultDelegate:
    return ResultDelegate(tmp_path / "thumbs", icons=CountingIcons())


def index_for(row: object) -> tuple[QStandardItemModel, object]:
    model = QStandardItemModel()
    item = QStandardItem("text")
    item.setData(row, ROW_ROLE)
    item.setBackground(QColor("#1e1f24"))  # the app's dark list background
    model.appendRow(item)
    return model, model.index(0, 0)


def option(selected: bool = False) -> QStyleOptionViewItem:
    opt = QStyleOptionViewItem()
    opt.palette.setColor(QPalette.ColorRole.Text, QColor("#e6e6e6"))
    opt.palette.setColor(QPalette.ColorRole.Highlight, QColor("#33405a"))  # the app's selection
    opt.rect.setSize(opt.rect.size().expandedTo(opt.rect.size()))
    opt.rect.setWidth(700)
    opt.rect.setHeight(80)
    if selected:
        opt.state |= QStyle.StateFlag.State_Selected
    return opt


def test_icons_are_cached_per_extension(delegate: ResultDelegate) -> None:
    icons = delegate._icons
    assert isinstance(icons, CountingIcons)
    delegate.icon_for(make_row(path=r"D:\a\one.pdf"))
    delegate.icon_for(make_row(path=r"D:\b\two.PDF"))
    delegate.icon_for(make_row(path=r"D:\b\three.txt"))
    assert len(icons.calls) == 2


def test_executables_and_shortcuts_get_their_own_icon(delegate: ResultDelegate) -> None:
    icons = delegate._icons
    assert isinstance(icons, CountingIcons)
    delegate.icon_for(make_row(path=r"D:\a\one.exe"))
    delegate.icon_for(make_row(path=r"D:\a\two.exe"))
    delegate.icon_for(make_row(path=r"D:\a\one.exe"))
    assert len(icons.calls) == 2


def test_whole_images_use_their_thumbnail(delegate: ResultDelegate, tmp_path: Path) -> None:
    picture = tmp_path / "pic.png"
    Image.new("RGB", (200, 100), "red").save(picture)
    thumb = thumbnail_path(tmp_path / "thumbs", picture)
    thumb.parent.mkdir(parents=True)
    Image.new("RGB", (50, 25), "red").save(thumb, format="JPEG")
    icons = delegate._icons
    assert isinstance(icons, CountingIcons)
    icon = delegate.icon_for(make_row(path=str(picture), is_image=True))
    assert not icon.pixmap(ICON_SIZE).isNull()
    assert icons.calls == []  # the thumbnail replaced the file-type icon
    # no thumbnail yet: fall back to the file icon
    delegate.icon_for(make_row(path=str(tmp_path / "gone.png"), is_image=True))
    assert len(icons.calls) == 1


def test_selected_row_is_taller_because_it_shows_the_snippet(delegate: ResultDelegate) -> None:
    _, index = index_for(make_row())
    plain = delegate.sizeHint(option(False), index)  # type: ignore[arg-type]
    selected = delegate.sizeHint(option(True), index)  # type: ignore[arg-type]
    assert selected.height() > plain.height()
    _, quiet = index_for(make_row(snippet=""))
    assert delegate.sizeHint(option(True), quiet).height() == plain.height()  # type: ignore[arg-type]


def test_rows_without_data_use_default_sizing_and_painting(
    delegate: ResultDelegate, qapp: QApplication
) -> None:
    model = QStandardItemModel()
    model.appendRow(QStandardItem("[1] a.py"))
    index = model.index(0, 0)
    assert delegate.sizeHint(option(), index).height() > 0
    canvas = QPixmap(300, 60)
    painter = QPainter(canvas)
    delegate.paint(painter, option(), index)
    painter.end()


def painted(delegate: ResultDelegate, row: ResultRow, selected: bool) -> QImage:
    _, index = index_for(row)
    canvas = QPixmap(700, 80)
    canvas.fill(Qt.GlobalColor.black)
    painter = QPainter(canvas)
    opt = option(selected)
    opt.rect.moveTo(0, 0)
    delegate.paint(painter, opt, index)  # type: ignore[arg-type]
    painter.end()
    return canvas.toImage()


def text_bands(image: QImage) -> int:
    """Number of separate text lines drawn: runs of pixel rows that differ from the background."""
    bands, inside = 0, False
    for y in range(image.height()):
        background = image.pixelColor(46, y).lightness()  # the gap between icon and text
        marked = any(
            abs(image.pixelColor(x, y).lightness() - background) > 40 for x in range(50, 500)
        )
        bands += marked and not inside
        inside = marked
    return bands


def test_paint_draws_name_and_path_lines(delegate: ResultDelegate) -> None:
    assert text_bands(painted(delegate, make_row(), selected=False)) == 2


def test_snippet_line_only_appears_on_the_selected_row(delegate: ResultDelegate) -> None:
    assert text_bands(painted(delegate, make_row(), selected=False)) == 2
    assert text_bands(painted(delegate, make_row(), selected=True)) == 3
    assert text_bands(painted(delegate, make_row(snippet=""), selected=True)) == 2


def test_rows_without_metadata_or_project_still_paint(delegate: ResultDelegate) -> None:
    row = make_row(modified="", size="", project="", detail="")
    assert text_bands(painted(delegate, row, selected=False)) == 2


def test_size_hint_follows_the_views_current_row(qapp: QApplication, tmp_path: Path) -> None:
    from PySide6.QtWidgets import QListWidget, QListWidgetItem

    view = QListWidget()
    delegate = ResultDelegate(tmp_path, view)
    view.setItemDelegate(delegate)
    for name in ("a", "b"):
        item = QListWidgetItem(name)
        item.setData(ROW_ROLE, make_row(name=name))
        view.addItem(item)
    view.setCurrentRow(0)
    view.show()
    qapp.processEvents()
    first, second = view.sizeHintForRow(0), view.sizeHintForRow(1)
    assert first > second  # the current row makes room for its snippet


def test_a_thumbnail_is_read_from_disk_once_not_on_every_paint(
    delegate: ResultDelegate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    picture = tmp_path / "pic.png"
    Image.new("RGB", (200, 100), "red").save(picture)
    thumb = thumbnail_path(tmp_path / "thumbs", picture)
    thumb.parent.mkdir(parents=True)
    Image.new("RGB", (50, 25), "red").save(thumb, format="JPEG")
    loads: list[str] = []
    from localdoc_finder.app import result_delegate as module

    original = module.QPixmap
    monkeypatch.setattr(
        module, "QPixmap", lambda path="": loads.append(str(path)) or original(path)
    )
    row = make_row(path=str(picture), is_image=True)
    for _ in range(30):  # a repaint asks for the same rows again and again
        assert not delegate.icon_for(row).pixmap(ICON_SIZE).isNull()
    assert len(loads) == 1


def test_a_missing_thumbnail_is_not_looked_up_on_every_paint_but_is_retried_later(
    delegate: ResultDelegate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localdoc_finder.app import result_delegate as module

    picture = tmp_path / "later.png"
    Image.new("RGB", (200, 100), "blue").save(picture)
    lookups: list[int] = []
    original = module.thumbnail_path
    monkeypatch.setattr(module, "thumbnail_path", lambda *a: lookups.append(1) or original(*a))
    row = make_row(path=str(picture), is_image=True)
    for _ in range(10):
        delegate.icon_for(row)
    assert len(lookups) == 1  # remembered as missing
    clock = [time.monotonic()]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0] + 60)  # much later
    delegate.icon_for(row)
    assert len(lookups) == 2  # tried again: the image may have been indexed meanwhile


def test_the_icon_caches_are_bounded() -> None:
    from localdoc_finder.app.result_delegate import _BoundedCache

    cache: _BoundedCache[int] = _BoundedCache(3)
    for i in range(10):
        cache.put(f"k{i}", i)
    assert len(cache) == 3
    assert cache.get("k0") is None
    assert cache.get("k9") == 9
    cache.get("k7")  # touching an entry keeps it
    cache.put("k10", 10)
    assert cache.get("k7") == 7
    assert cache.get("k8") is None
