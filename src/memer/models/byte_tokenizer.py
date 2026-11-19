"""Stable UTF-8 byte tokenizer for Memer Models 0.2.

The fixed vocabulary avoids corpus-dependent token IDs and supports every
Unicode string. It is a tokenizer foundation, not a learned BPE tokenizer.
"""

from __future__ import annotations

BOS = 256
EOS = 257
PAD = 258
BYTE_OFFSET = 0
VOCAB_SIZE = 259


class ByteTokenizer:
    """Encodes UTF-8 bytes as token IDs plus three special tokens."""

    vocab_size = VOCAB_SIZE

    def encode(self, text: str, add_boundaries: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        token_ids = list(text.encode("utf-8"))
        if add_boundaries:
            return [BOS, *token_ids, EOS]
        return token_ids

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        raw = bytearray()
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError("token IDs must be integers")
            if 0 <= token_id <= 255:
                raw.append(token_id)
            elif token_id in (BOS, EOS, PAD) and skip_special:
                continue
            else:
                raise ValueError(f"token ID out of range or unexpected special token: {token_id}")
        return raw.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"type": "utf8-byte", "version": 1, "vocab_size": VOCAB_SIZE}

    @classmethod
    def from_dict(cls, payload: dict) -> "ByteTokenizer":
        if (
            not isinstance(payload, dict)
            or payload.get("type") != "utf8-byte"
            or payload.get("version") != 1
            or payload.get("vocab_size") != VOCAB_SIZE
        ):
            raise ValueError("unsupported byte tokenizer format")
        return cls()
