import pytest

from localdoc_finder.core.model_names import model_family, normalise_model_name, same_model


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("qwen3-embedding:8b", "qwen3-embedding"),
        ("nomic-embed-text", "nomic-embed-text"),
        ("Nomic-Embed-Text:latest", "nomic-embed-text"),
        ("dengcao/Qwen3-Reranker-0.6B:Q8_0", "dengcao/qwen3-reranker-0.6b"),
    ],
)
def test_model_family_drops_the_tag(name: str, family: str) -> None:
    assert model_family(name) == family


def test_a_missing_tag_means_latest() -> None:
    assert normalise_model_name(" BGE-M3 ") == "bge-m3:latest"
    assert same_model("bge-m3", "bge-m3:latest")
    assert not same_model("qwen3-embedding:4b", "qwen3-embedding:8b")
