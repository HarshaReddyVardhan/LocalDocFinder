import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "release_notes.py"
spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
assert spec and spec.loader
release_notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes)


def test_user_facing_commits_are_grouped_and_housekeeping_is_dropped() -> None:
    notes = release_notes.render(
        "0.4.0",
        [
            "feat(app): show-in-folder button on results",
            "fix(store): survive concurrent index changes",
            "chore: release 0.3.2",
            "docs: update the readme",
            "test: more coverage",
            "perf(search): cache the reranker",
        ],
    )
    assert notes.startswith("# LocalDoc Finder 0.4.0")
    assert "## New\n\n- Show-in-folder button on results" in notes
    assert "## Fixed\n\n- Survive concurrent index changes" in notes
    assert "## Faster" in notes
    assert "release 0.3.2" not in notes
    assert "readme" not in notes


def test_a_release_without_user_facing_commits_says_so() -> None:
    assert release_notes.FALLBACK in release_notes.render(
        "0.4.0", ["chore: bump", "not conventional"]
    )
