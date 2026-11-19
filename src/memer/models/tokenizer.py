"""Deterministic character tokenizer for the first Memer language model."""

from __future__ import annotations

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


class CharacterTokenizer:
    """Maps Unicode characters to stable integer token IDs."""

    def __init__(self, vocabulary: list[str]) -> None:
        if not isinstance(vocabulary, list) or len(vocabulary) < 4:
            raise ValueError("vocabulary must contain special tokens and at least one character")
        if vocabulary[:3] != [BOS, EOS, UNK]:
            raise ValueError("vocabulary must start with <bos>, <eos>, <unk>")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("vocabulary contains duplicate tokens")
        self.vocabulary = list(vocabulary)
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not isinstance(text, str) or not text:
            raise ValueError("training text must not be empty")
        characters = sorted(set(text))
        return cls([BOS, EOS, UNK, *characters])

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def encode(self, text: str, add_boundaries: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = [self.token_to_id.get(char, self.token_to_id[UNK]) for char in text]
        if add_boundaries:
            return [self.token_to_id[BOS], *ids, self.token_to_id[EOS]]
        return ids

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        output = []
        special = {BOS, EOS, UNK}
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError("token IDs must be integers")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token ID out of range: {token_id}")
            token = self.vocabulary[token_id]
            if skip_special and token in special:
                continue
            output.append(token)
        return "".join(output)

    def to_dict(self) -> dict:
        return {"type": "character", "version": 1, "vocabulary": self.vocabulary}

    @classmethod
    def from_dict(cls, payload: dict) -> "CharacterTokenizer":
        if not isinstance(payload, dict) or payload.get("type") != "character" or payload.get("version") != 1:
            raise ValueError("unsupported tokenizer format")
        return cls(payload.get("vocabulary"))
