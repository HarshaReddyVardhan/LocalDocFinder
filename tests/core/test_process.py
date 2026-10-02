from pathlib import Path

from vector_embed.core.process import single_instance


def test_second_holder_is_refused_until_released(tmp_path: Path) -> None:
    with single_instance("worker", tmp_path) as first:
        assert first is True
        with single_instance("worker", tmp_path) as second:
            assert second is False
    with single_instance("worker", tmp_path) as again:
        assert again is True


def test_locks_are_per_name(tmp_path: Path) -> None:
    with single_instance("a", tmp_path) as a, single_instance("b", tmp_path) as b:
        assert a is True
        assert b is True


def test_creates_the_data_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir"
    with single_instance("x", target) as got:
        assert got is True
    assert target.is_dir()
