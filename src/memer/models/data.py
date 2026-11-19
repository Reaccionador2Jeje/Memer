"""Dataset preparation and padded batches for decoder-only language models."""

from __future__ import annotations

import random
from collections.abc import Iterator

from memer.models.byte_tokenizer import BOS, EOS, PAD, ByteTokenizer

IGNORE_INDEX = -100


def encode_documents(text: str, tokenizer: ByteTokenizer) -> list[list[int]]:
    """Encode non-empty lines as independent BOS...EOS documents."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("tokenizer must be a ByteTokenizer")

    documents = []
    for line in text.splitlines():
        if line.strip():
            documents.append([BOS, *tokenizer.encode(line), EOS])
    if not documents and text:
        documents.append([BOS, *tokenizer.encode(text), EOS])
    return documents


def make_batches(
    documents: list[list[int]],
    block_size: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 17,
) -> Iterator[dict[str, list[list[int]]]]:
    """Yield next-token training batches with right-padding and loss masks.

    Each input position predicts the following token. Padded targets use
    IGNORE_INDEX so a future trainer can exclude them from cross-entropy.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(documents, list):
        raise TypeError("documents must be a list")

    samples: list[tuple[list[int], list[int]]] = []
    for document in documents:
        if not isinstance(document, list) or len(document) < 2:
            continue
        for start in range(0, len(document) - 1, block_size):
            chunk = document[start : start + block_size + 1]
            if len(chunk) >= 2:
                samples.append((chunk[:-1], chunk[1:]))

    if shuffle:
        random.Random(seed).shuffle(samples)

    for offset in range(0, len(samples), batch_size):
        group = samples[offset : offset + batch_size]
        input_rows = []
        target_rows = []
        masks = []
        for inputs, targets in group:
            padding = block_size - len(inputs)
            input_rows.append(inputs + [PAD] * padding)
            target_rows.append(targets + [IGNORE_INDEX] * padding)
            masks.append([1] * len(inputs) + [0] * padding)
        yield {
            "input_ids": input_rows,
            "target_ids": target_rows,
            "attention_mask": masks,
        }
