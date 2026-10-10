import pytest

from lib.bpe import BPETokenizer


def make_tokenizer() -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.train("low low low lower", num_merges=5)
    return tokenizer


def test_training_learns_merge_rules() -> None:
    tokenizer = make_tokenizer()

    assert tokenizer.merges
    assert all(len(left) > 0 and len(right) > 0 for left, right in tokenizer.merges)


def test_encode_decode_round_trip() -> None:
    tokenizer = make_tokenizer()
    text = "low lower low"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_token_ids_are_valid() -> None:
    tokenizer = make_tokenizer()

    ids = tokenizer.encode("low")

    assert all(isinstance(token_id, int) for token_id in ids)
    assert all(token_id in tokenizer.vocab.values() for token_id in ids)


def test_training_is_deterministic() -> None:
    first = make_tokenizer()
    second = make_tokenizer()

    assert first.merges == second.merges
    assert first.vocab == second.vocab


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="corpus must not be empty"):
        BPETokenizer().train("", num_merges=2)


def test_negative_merge_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="num_merges must be non-negative"):
        BPETokenizer().train("abc", num_merges=-1)


def test_unseen_character_is_rejected() -> None:
    tokenizer = make_tokenizer()

    with pytest.raises(ValueError, match="outside the training vocabulary"):
        tokenizer.encode("z")


def test_unknown_token_id_is_rejected() -> None:
    tokenizer = make_tokenizer()

    with pytest.raises(ValueError, match="Unknown token ID"):
        tokenizer.decode([999999])


def test_untrained_tokenizer_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="must be trained"):
        BPETokenizer().encode("hello")


def test_single_character_corpus_stops_without_merges() -> None:
    tokenizer = BPETokenizer()
    tokenizer.train("a", num_merges=3)

    assert tokenizer.merges == []
    assert tokenizer.decode(tokenizer.encode("a")) == "a"


def test_round_trip_preserves_unicode_and_whitespace() -> None:
    text = "café 🙂\n  café"

    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=10)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_overlapping_pairs_merge_left_to_right() -> None:
    symbols = ["a", "a", "a"]

    assert BPETokenizer._merge_pair(
        symbols,
        ("a", "a"),
    ) == ["aa", "a"]


def test_retraining_replaces_previous_vocabulary() -> None:
    tokenizer = BPETokenizer()
    tokenizer.train("banana", num_merges=3)

    tokenizer.train("cocoa", num_merges=3)

    assert tokenizer.decode(tokenizer.encode("cocoa")) == "cocoa"

    with pytest.raises(ValueError, match="outside the training vocabulary"):
        tokenizer.encode("banana")
