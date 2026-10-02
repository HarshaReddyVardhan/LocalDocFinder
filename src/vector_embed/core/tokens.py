"""Cheap token estimates: enough to keep prompts inside the model's context window."""

_CHARS_PER_TOKEN = 4  # conservative average for English prose and code


def estimate_tokens(text: str) -> int:
    return -(-len(text) // _CHARS_PER_TOKEN)


def fit_to_budget(text: str, budget_tokens: int) -> tuple[str, bool]:
    """``text`` cut to roughly ``budget_tokens``; the flag says whether anything was cut."""
    limit = budget_tokens * _CHARS_PER_TOKEN
    if len(text) <= limit:
        return text, False
    return text[:limit], True
