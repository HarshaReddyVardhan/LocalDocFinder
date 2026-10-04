"""Ollama model-name comparison; dependency-free so the watcher can use it."""

DEFAULT_TAG = "latest"


def normalise_model_name(name: str) -> str:
    """``qwen3`` and ``qwen3:latest`` are the same model; compare names in this form."""
    cleaned = name.strip().lower()
    return cleaned if ":" in cleaned.rsplit("/", 1)[-1] else f"{cleaned}:{DEFAULT_TAG}"


def same_model(first: str, second: str) -> bool:
    return normalise_model_name(first) == normalise_model_name(second)


def model_family(name: str) -> str:
    """The name without its tag: ``qwen3-embedding:4b`` and ``:8b`` are one family."""
    return normalise_model_name(name).rsplit(":", 1)[0]
