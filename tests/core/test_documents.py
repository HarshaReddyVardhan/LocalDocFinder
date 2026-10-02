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
