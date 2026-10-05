from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication
from tests.core.app.test_app import FakeService, result

from localdoc_finder.app.controller import Launcher, SearchOutcome
from localdoc_finder.app.window import CLEAR_HISTORY, SearchWindow
from localdoc_finder.core.result_view import SortOrder

HITS = [
    result(path=r"D:\p\zeta.pdf", ext=".pdf", indexed_at=30.0),
    result(path=r"D:\p\alpha.txt", ext=".txt", indexed_at=10.0),
    result(path=r"D:\p\beta.pdf", ext=".pdf", indexed_at=20.0),
]


@pytest.fixture
def window(qapp: QApplication, tmp_path: Path) -> tuple[SearchWindow, FakeService]:
    service = FakeService(HITS)
    win = SearchWindow(service, Launcher(startfile=lambda _path: None), tmp_path)  # type: ignore[arg-type]
    win.show()
    return win, service


def shown(win: SearchWindow) -> list[str]:
    return [r.path.rsplit("\\", 1)[1] for r in win._results]


def test_the_refine_bar_appears_only_once_there_are_results(
    window: tuple[SearchWindow, FakeService],
) -> None:
    win, _ = window
    assert not win.refine_bar.isVisible()
    win.show_results(SearchOutcome(HITS, 1.0))
    assert win.refine_bar.isVisible()
    win.show_results(SearchOutcome([], 1.0))
    assert not win.refine_bar.isVisible()
    win.input.setText("")
    win.show_results(SearchOutcome(HITS, 1.0))
    win.run_search()  # an emptied box clears the list and hides the bar again
    assert not win.refine_bar.isVisible()


def test_picking_a_type_filters_without_searching_again(
    window: tuple[SearchWindow, FakeService],
) -> None:
    win, service = window
    win.show_results(SearchOutcome(HITS, 1.0))
    labels = [win.refine_bar.type_box.itemText(i) for i in range(win.refine_bar.type_box.count())]
    assert [label.split()[0] for label in labels] == ["All", ".pdf", ".txt"]
    win.refine_bar.type_box.setCurrentIndex(1)
    assert shown(win) == ["zeta.pdf", "beta.pdf"]
    assert win.list.count() == 2
    assert win.status.text() == "2 of 3 results"
    assert service.queries == []  # filtered in memory


def test_sorting_changes_the_rows_and_the_selection_follows_them(
    window: tuple[SearchWindow, FakeService],
) -> None:
    win, _ = window
    win.show_results(SearchOutcome(HITS, 1.0))
    win.refine_bar.sort_box.setCurrentIndex(list(SortOrder).index(SortOrder.NAME))
    assert shown(win) == ["alpha.txt", "beta.pdf", "zeta.pdf"]
    assert win.selected() is win._results[0]
    win.refine_bar.sort_box.setCurrentIndex(list(SortOrder).index(SortOrder.DATE_INDEXED))
    assert shown(win) == ["zeta.pdf", "beta.pdf", "alpha.txt"]


def test_the_chosen_type_survives_a_new_search_only_if_it_is_still_there(
    window: tuple[SearchWindow, FakeService],
) -> None:
    win, _ = window
    win.show_results(SearchOutcome(HITS, 1.0))
    win.refine_bar.type_box.setCurrentIndex(2)  # .txt
    win.show_results(SearchOutcome(HITS, 1.0))
    assert win.refine_bar.ext == ".txt"
    win.show_results(SearchOutcome([HITS[0], HITS[2]], 1.0))  # no .txt left
    assert win.refine_bar.ext is None
    assert not win.refine_bar.type_box.isVisible()  # one type: nothing to filter
    assert len(shown(win)) == 2


def test_the_window_stays_open_when_a_filter_hides_every_result(
    window: tuple[SearchWindow, FakeService],
) -> None:
    win, _ = window
    win.show_results(SearchOutcome(HITS, 1.0))
    height = win.height()
    win.refine_bar.type_box.setCurrentIndex(2)
    win._found = [HITS[0]]  # the txt hit goes away; the pdf stays
    win._fill_list()
    assert win.height() == height


def test_search_history_menu_reruns_a_query(window: tuple[SearchWindow, FakeService]) -> None:
    win, service = window
    assert win.recent_button.isVisible()
    win.show_recent()
    assert win.status.text() == "no earlier searches yet"

    service.history = ["retry policy", "invoice"]
    offered: list[list[str]] = []
    win.choose_search = lambda queries: offered.append(queries) or "invoice"  # type: ignore[func-returns-value]
    win.recent_button.click()
    assert offered == [["retry policy", "invoice"]]
    assert win.input.text() == "invoice"
    assert service.history[0] == "invoice"  # moved to the top


def test_search_history_can_be_cleared_or_dismissed(
    window: tuple[SearchWindow, FakeService],
) -> None:
    win, service = window
    service.history = ["a"]
    win.choose_search = lambda _queries: None
    win.show_recent()
    assert service.history == ["a"]
    win.choose_search = lambda _queries: CLEAR_HISTORY
    win.show_recent()
    assert service.history == []


def test_enter_remembers_the_query(window: tuple[SearchWindow, FakeService]) -> None:
    win, service = window
    win.input.setText("  deploy notes ")
    win.show_results(SearchOutcome(HITS, 1.0))
    win._handle_enter(ctrl=False, shift=False)
    assert service.history == ["deploy notes"]
