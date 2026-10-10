"""Small, deterministic text utilities for the LLM handbook."""


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace and trim surrounding whitespace."""
    return " ".join(text.split())


def chunk_text(text: str, max_words: int) -> list[str]:
    """Split text into chunks containing at most max_words words."""
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")

    words = normalize_text(text).split()

    return [
        " ".join(words[start : start + max_words])
        for start in range(0, len(words), max_words)
    ]
