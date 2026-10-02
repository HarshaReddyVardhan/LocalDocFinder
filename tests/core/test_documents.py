from pathlib import Path

import pytest
from tests.core.conftest import Env

from vector_embed.core.documents import DocumentError, DocumentLoader

RESUME = (
    "Jane Doe\nSummary\nBackend engineer\nWork Experience\nBuilt payment systems in Python\n"
    "Education\nBSc Computer Science\nSkills\nPython SQL\nProjects\nSearch engine\n"
)


@pytest.fixture
def loader(env: Env) -> DocumentLoader:
    return DocumentLoader(env.store, env.scope, lambda: env.extractors)


def write(env: Env, rel: str, text: str) -> Path:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_indexed_documents_come_from_the_index(env: Env, loader: DocumentLoader) -> None:
    path = write(env, "Resume_v1.txt", RESUME)
    env.indexer.index_paths([str(path)])
    path.write_text("changed on disk after indexing", encoding="utf-8")
    doc = loader.load(path)
    assert doc.from_index
    assert doc.doc_type == "resume"
    assert doc.title == "Jane Doe"
    assert "payment systems" in doc.text


def test_unindexed_files_are_extracted_on_demand(env: Env, loader: DocumentLoader) -> None:
    path = write(env, "fresh.txt", "first line\nsecond line\n")
    doc = loader.load(str(path))
    assert not doc.from_index
    assert doc.text == "first line\nsecond line"
    assert doc.title == "first line"


def test_large_documents_are_reextracted_because_the_index_keeps_no_text(env: Env) -> None:
    env.settings = env.settings.model_copy(
        update={"doctypes": env.settings.doctypes.model_copy(update={"full_text_max_chars": 20})}
    )
    env.indexer.settings = env.settings
    path = write(env, "long.txt", ("a long line of words\n") * 10)
    env.indexer.index_paths([str(path)])
    doc = DocumentLoader(env.store, env.scope, lambda: env.extractors).load(path)
    assert not doc.from_index
    assert doc.text.count("a long line of words") >= 10


def test_secrets_are_never_loaded(env: Env, loader: DocumentLoader) -> None:
    secret = write(env, ".env", "TOKEN=abc")
    with pytest.raises(DocumentError, match="secret"):
        loader.load(secret)


def test_missing_and_unreadable_files(env: Env, loader: DocumentLoader) -> None:
    with pytest.raises(DocumentError, match="does not exist"):
        loader.load(env.root / "nope.txt")
    binary = env.root / "blob.txt"
    binary.write_bytes(b"a\x00b")
    with pytest.raises(DocumentError, match="cannot read"):
        loader.load(binary)


def test_blank_title_falls_back_to_the_file_name(env: Env, loader: DocumentLoader) -> None:
    path = write(env, "blank.txt", "\n\n")
    assert loader.load(path).title == "blank.txt"


def test_files_outside_the_scope_rules_are_refused_like_secrets(
    env: Env, loader: DocumentLoader
) -> None:
    unsupported = write(env, "tool.exe", "MZ binary")
    with pytest.raises(DocumentError, match="does not read"):
        loader.load(unsupported)
    noise = write(env, "bundle.min.js", "var a=1;" * 50)
    with pytest.raises(DocumentError, match="does not read"):
        loader.load(noise)


def test_a_file_in_a_blocked_directory_is_refused(env: Env) -> None:
    blocked_scope = type(env.scope)(env.scope._s, blocked_roots=[env.root / "off-limits"])
    path = write(env, "off-limits/notes.txt", "private notes")
    with pytest.raises(DocumentError, match="does not read"):
        DocumentLoader(env.store, blocked_scope, lambda: env.extractors).load(path)


def test_gitignored_files_can_still_be_chosen_explicitly(env: Env, loader: DocumentLoader) -> None:
    write(env, ".gitignore", "ignored.txt\n")
    path = write(env, "ignored.txt", "the user picked this file on purpose")
    assert loader.load(path).text.startswith("the user picked")


def lines(sentence: str, count: int = 30) -> str:
    """Ordinary multi-line text (one long line would be mistaken for a minified file)."""
    return chr(10).join([sentence] * count)


class TestExtractionCache:
    def count_extractions(self, env: Env, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        calls: list[str] = []
        original = env.extractors.extract

        def spy(path: str | Path) -> object:
            calls.append(Path(path).name)
            return original(path)

        monkeypatch.setattr(env.extractors, "extract", spy)
        return calls

    def test_a_pinned_file_is_extracted_once_across_chat_turns(
        self, env: Env, loader: DocumentLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = write(env, "pinned.txt", lines("pinned words"))
        calls = self.count_extractions(env, monkeypatch)
        for _turn in range(5):
            assert loader.load(path).text.startswith("pinned words")
        assert calls == ["pinned.txt"]  # not once per turn

    def test_editing_the_file_extracts_it_again(
        self, env: Env, loader: DocumentLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = write(env, "draft.txt", lines("first version"))
        calls = self.count_extractions(env, monkeypatch)
        assert "first version" in loader.load(path).text
        path.write_text(lines("second version, now longer than before"), encoding="utf-8")
        assert "second version" in loader.load(path).text
        assert calls == ["draft.txt", "draft.txt"]

    def test_the_cache_is_bounded(
        self, env: Env, loader: DocumentLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vector_embed.core import documents as documents_module

        monkeypatch.setattr(documents_module, "_RECENT_DOCUMENTS", 2)
        paths = [write(env, f"doc{i}.txt", lines(f"document {i}")) for i in range(4)]
        for path in paths:
            loader.load(path)
        assert len(loader._recent) == 2
        calls = self.count_extractions(env, monkeypatch)
        loader.load(paths[0])  # the oldest was forgotten
        loader.load(paths[3])  # the newest is still remembered
        assert calls == ["doc0.txt"]

    def test_indexed_documents_still_come_from_the_index(
        self, env: Env, loader: DocumentLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = write(env, "Resume_v1.txt", RESUME)
        env.indexer.index_paths([str(path)])
        calls = self.count_extractions(env, monkeypatch)
        assert loader.load(path).from_index
        assert calls == []
