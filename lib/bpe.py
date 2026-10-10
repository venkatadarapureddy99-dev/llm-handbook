"""A small, deterministic character-level BPE tokenizer."""

from collections import Counter


class BPETokenizer:
    """Learn frequent adjacent-symbol merges from a text corpus."""

    def __init__(self) -> None:
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self._trained = False

    @staticmethod
    def _pairs(symbols: list[str]) -> Counter[tuple[str, str]]:
        """Count adjacent symbol pairs."""
        return Counter(zip(symbols, symbols[1:], strict=False))

    @staticmethod
    def _merge_pair(
        symbols: list[str],
        pair: tuple[str, str],
    ) -> list[str]:
        """Merge non-overlapping occurrences of a pair, left to right."""
        merged: list[str] = []
        i = 0

        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1

        return merged

    def train(self, corpus: str, num_merges: int) -> None:
        """Learn merge rules from a non-empty training corpus."""
        if not corpus:
            raise ValueError("corpus must not be empty")
        if num_merges < 0:
            raise ValueError("num_merges must be non-negative")

        symbols = list(corpus)
        self.merges = []

        for _ in range(num_merges):
            pair_counts = self._pairs(symbols)
            if not pair_counts:
                break

            # Counter preserves first-seen order for tied counts.
            best_pair = max(
                pair_counts,
                key=lambda pair: pair_counts[pair],
            )
            symbols = self._merge_pair(symbols, best_pair)
            self.merges.append(best_pair)

        tokens = set(corpus)
        for left, right in self.merges:
            tokens.add(left + right)

        # Stable IDs make results reproducible across runs.
        self.vocab = {token: index for index, token in enumerate(sorted(tokens))}
        self._trained = True

    def encode(self, text: str) -> list[int]:
        """Encode text using the learned merge rules."""
        self._require_trained()
        symbols = list(text)

        for pair in self.merges:
            symbols = self._merge_pair(symbols, pair)

        unknown = [symbol for symbol in symbols if symbol not in self.vocab]
        if unknown:
            raise ValueError(
                f"Input contains symbols outside the training vocabulary: {unknown!r}"
            )

        return [self.vocab[symbol] for symbol in symbols]

    def decode(self, token_ids: list[int]) -> str:
        """Reconstruct text from token IDs."""
        self._require_trained()
        inverse_vocab = {index: token for token, index in self.vocab.items()}

        try:
            return "".join(inverse_vocab[token_id] for token_id in token_ids)
        except KeyError as exc:
            raise ValueError(f"Unknown token ID: {exc.args[0]}") from exc

    def _require_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("Tokenizer must be trained before use")
