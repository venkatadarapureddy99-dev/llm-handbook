import pytest

from lib.text_utils import chunk_text, normalize_text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("  hello   world  ", "hello world"),
        ("line one\nline two", "line one line two"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_text(source: str, expected: str) -> None:
    assert normalize_text(source) == expected


def test_chunk_text_respects_word_limit() -> None:
    assert chunk_text("one two three four five", 2) == [
        "one two",
        "three four",
        "five",
    ]


def test_chunk_text_handles_empty_input() -> None:
    assert chunk_text("", 3) == []


@pytest.mark.parametrize("max_words", [0, -1])
def test_chunk_text_rejects_invalid_word_limit(max_words: int) -> None:
    with pytest.raises(ValueError, match="max_words must be greater than zero"):
        chunk_text("one two", max_words)
