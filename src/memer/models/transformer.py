"""Small pure-Python decoder-only Transformer inference core.

This module implements causal multi-head self-attention, pre-layer
normalization, residual connections, and a position-wise feed-forward block.
It is an educational reference implementation; training/backpropagation is a
separate milestone.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from memer.models.byte_tokenizer import BOS, EOS, PAD, VOCAB_SIZE


class TransformerError(ValueError):
    """Raised for invalid Transformer configuration or input."""


def _linear(vector: list[float], weights: list[list[float]], bias: list[float]) -> list[float]:
    return [
        sum(value * weight for value, weight in zip(vector, row)) + bias[index]
        for index, row in enumerate(weights)
    ]


def _layer_norm(vector: list[float], scale: list[float], bias: list[float]) -> list[float]:
    mean = sum(vector) / len(vector)
    variance = sum((value - mean) ** 2 for value in vector) / len(vector)
    inverse = 1.0 / math.sqrt(variance + 1e-5)
    return [
        (value - mean) * inverse * scale[index] + bias[index]
        for index, value in enumerate(vector)
    ]


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


class CausalSelfAttention:
    """Multi-head scaled dot-product attention with a causal mask."""

    def __init__(self, dimension: int, heads: int, rng: random.Random) -> None:
        if dimension % heads:
            raise TransformerError("dimension must be divisible by heads")
        self.dimension = dimension
        self.heads = heads
        self.head_dim = dimension // heads
        scale = 1.0 / math.sqrt(dimension)

        def matrix(rows: int, columns: int) -> list[list[float]]:
            return [[rng.uniform(-scale, scale) for _ in range(columns)] for _ in range(rows)]

        self.query = matrix(dimension, dimension)
        self.key = matrix(dimension, dimension)
        self.value = matrix(dimension, dimension)
        self.output = matrix(dimension, dimension)
        self.query_bias = [0.0] * dimension
        self.key_bias = [0.0] * dimension
        self.value_bias = [0.0] * dimension
        self.output_bias = [0.0] * dimension

    def forward(self, sequence: list[list[float]], attention_mask: list[int] | None = None) -> list[list[float]]:
        length = len(sequence)
        if not length:
            return []
        if attention_mask is None:
            attention_mask = [1] * length
        if len(attention_mask) != length:
            raise TransformerError("attention mask length does not match sequence")

        queries = [_linear(row, self.query, self.query_bias) for row in sequence]
        keys = [_linear(row, self.key, self.key_bias) for row in sequence]
        values = [_linear(row, self.value, self.value_bias) for row in sequence]
        outputs: list[list[float]] = []

        for position in range(length):
            combined = [0.0] * self.dimension
            for head in range(self.heads):
                start = head * self.head_dim
                end = start + self.head_dim
                scores = []
                visible_positions = []
                for previous in range(position + 1):
                    if not attention_mask[previous]:
                        continue
                    score = sum(
                        queries[position][index] * keys[previous][index]
                        for index in range(start, end)
                    ) / math.sqrt(self.head_dim)
                    scores.append(score)
                    visible_positions.append(previous)

                if not scores:
                    continue
                probabilities = _softmax(scores)
                for offset, probability in enumerate(probabilities):
                    previous = visible_positions[offset]
                    for index in range(start, end):
                        combined[index] += probability * values[previous][index]

            outputs.append(_linear(combined, self.output, self.output_bias))
        return outputs


class TransformerBlock:
    """Pre-LN decoder block: causal attention and feed-forward residuals."""

    def __init__(self, dimension: int, heads: int, feed_forward_dim: int, rng: random.Random) -> None:
        self.attention_norm_scale = [1.0] * dimension
        self.attention_norm_bias = [0.0] * dimension
        self.feed_forward_norm_scale = [1.0] * dimension
        self.feed_forward_norm_bias = [0.0] * dimension
        self.attention = CausalSelfAttention(dimension, heads, rng)

        scale = 1.0 / math.sqrt(dimension)

        def matrix(rows: int, columns: int) -> list[list[float]]:
            return [[rng.uniform(-scale, scale) for _ in range(columns)] for _ in range(rows)]

        self.ff1 = matrix(feed_forward_dim, dimension)
        self.ff1_bias = [0.0] * feed_forward_dim
        self.ff2 = matrix(dimension, feed_forward_dim)
        self.ff2_bias = [0.0] * dimension

    def forward(self, sequence: list[list[float]], attention_mask: list[int] | None = None) -> list[list[float]]:
        normalized = [
            _layer_norm(row, self.attention_norm_scale, self.attention_norm_bias)
            for row in sequence
        ]
        attended = self.attention.forward(normalized, attention_mask)
        residual = [
            [value + update for value, update in zip(row, attended_row)]
            for row, attended_row in zip(sequence, attended)
        ]

        output = []
        for row in residual:
            normalized_row = _layer_norm(
                row, self.feed_forward_norm_scale, self.feed_forward_norm_bias
            )
            hidden = _linear(normalized_row, self.ff1, self.ff1_bias)
            hidden = [max(0.0, value) for value in hidden]
            update = _linear(hidden, self.ff2, self.ff2_bias)
            output.append([value + delta for value, delta in zip(row, update)])
        return output


class MemerTransformer:
    """Configurable, small decoder-only Transformer with tied output embeddings."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        dimension: int = 16,
        heads: int = 2,
        layers: int = 1,
        feed_forward_dim: int = 32,
        block_size: int = 32,
        seed: int = 17,
    ) -> None:
        integer_options = {
            "vocab_size": (vocab_size, 4, 4096),
            "dimension": (dimension, 4, 256),
            "heads": (heads, 1, 16),
            "layers": (layers, 1, 8),
            "feed_forward_dim": (feed_forward_dim, 4, 1024),
            "block_size": (block_size, 2, 512),
        }
        for name, (value, minimum, maximum) in integer_options.items():
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise TransformerError(f"{name} must be an integer between {minimum} and {maximum}")
        if dimension % heads:
            raise TransformerError("dimension must be divisible by heads")

        self.vocab_size = vocab_size
        self.dimension = dimension
        self.heads = heads
        self.layers = layers
        self.feed_forward_dim = feed_forward_dim
        self.block_size = block_size
        self.seed = seed
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(dimension)
        self.token_embeddings = [
            [rng.uniform(-scale, scale) for _ in range(dimension)]
            for _ in range(vocab_size)
        ]
        self.position_embeddings = [
            [rng.uniform(-scale, scale) for _ in range(dimension)]
            for _ in range(block_size)
        ]
        self.blocks = [
            TransformerBlock(dimension, heads, feed_forward_dim, rng)
            for _ in range(layers)
        ]
        self.final_norm_scale = [1.0] * dimension
        self.final_norm_bias = [0.0] * dimension
        self.output_bias = [0.0] * vocab_size

    def forward(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]] | None = None,
    ) -> list[list[list[float]]]:
        """Return logits shaped [batch][sequence][vocabulary]."""
        if not isinstance(input_ids, list) or not input_ids:
            raise TransformerError("input_ids must be a non-empty batch")
        if any(not isinstance(row, list) or not row for row in input_ids):
            raise TransformerError("each input row must be a non-empty list")
        sequence_length = len(input_ids[0])
        if sequence_length > self.block_size:
            raise TransformerError("sequence exceeds configured block_size")
        if any(len(row) != sequence_length for row in input_ids):
            raise TransformerError("all input rows must have equal length")
        if attention_mask is not None and (
            not isinstance(attention_mask, list)
            or len(attention_mask) != len(input_ids)
            or any(not isinstance(row, list) or len(row) != sequence_length for row in attention_mask)
        ):
            raise TransformerError("attention_mask shape does not match input_ids")

        batch_logits = []
        for batch_index, token_row in enumerate(input_ids):
            if any(
                isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < self.vocab_size
                for token in token_row
            ):
                raise TransformerError("input token ID is outside the vocabulary")
            mask = attention_mask[batch_index] if attention_mask is not None else [
                0 if token == PAD else 1 for token in token_row
            ]
            if any(value not in (0, 1) for value in mask):
                raise TransformerError("attention mask values must be 0 or 1")

            hidden = [
                [
                    self.token_embeddings[token][dimension]
                    + self.position_embeddings[position][dimension]
                    for dimension in range(self.dimension)
                ]
                for position, token in enumerate(token_row)
            ]
            for block in self.blocks:
                hidden = block.forward(hidden, mask)

            logits_row = []
            for vector in hidden:
                normalized = _layer_norm(vector, self.final_norm_scale, self.final_norm_bias)
                logits_row.append([
                    sum(
                        normalized[index] * self.token_embeddings[token][index]
                        for index in range(self.dimension)
                    ) + self.output_bias[token]
                    for token in range(self.vocab_size)
                ])
            batch_logits.append(logits_row)
        return batch_logits


    def _weights_payload(self) -> dict:
        def attention_payload(attention):
            return {
                "query": attention.query, "key": attention.key,
                "value": attention.value, "output": attention.output,
                "query_bias": attention.query_bias, "key_bias": attention.key_bias,
                "value_bias": attention.value_bias, "output_bias": attention.output_bias,
            }

        return {
            "token_embeddings": self.token_embeddings,
            "position_embeddings": self.position_embeddings,
            "final_norm_scale": self.final_norm_scale,
            "final_norm_bias": self.final_norm_bias,
            "output_bias": self.output_bias,
            "blocks": [
                {
                    "attention_norm_scale": block.attention_norm_scale,
                    "attention_norm_bias": block.attention_norm_bias,
                    "feed_forward_norm_scale": block.feed_forward_norm_scale,
                    "feed_forward_norm_bias": block.feed_forward_norm_bias,
                    "attention": attention_payload(block.attention),
                    "ff1": block.ff1, "ff1_bias": block.ff1_bias,
                    "ff2": block.ff2, "ff2_bias": block.ff2_bias,
                }
                for block in self.blocks
            ],
        }

    def save(self, path: str | Path) -> str:
        """Save architecture configuration and weights as a JSON checkpoint."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "MEMER-TRANSFORMER",
            "format_version": 1,
            "config": {
                "vocab_size": self.vocab_size,
                "dimension": self.dimension,
                "heads": self.heads,
                "layers": self.layers,
                "feed_forward_dim": self.feed_forward_dim,
                "block_size": self.block_size,
                "seed": self.seed,
            },
            "weights": self._weights_payload(),
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        temporary.replace(target)
        return str(target)

    @classmethod
    def load(cls, path: str | Path) -> "MemerTransformer":
        """Load a checkpoint, validating every tensor shape and numeric value."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TransformerError(f"could not load Transformer checkpoint: {error}") from error
        if not isinstance(payload, dict) or payload.get("format") != "MEMER-TRANSFORMER" or payload.get("format_version") != 1:
            raise TransformerError("unsupported Transformer checkpoint format")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise TransformerError("checkpoint configuration is invalid")
        try:
            model = cls(**config)
        except (TypeError, ValueError) as error:
            raise TransformerError(f"invalid checkpoint configuration: {error}") from error

        def restore(current, incoming, path_name):
            if isinstance(current, list):
                if not isinstance(incoming, list) or len(current) != len(incoming):
                    raise TransformerError(f"invalid checkpoint shape at {path_name}")
                if current and isinstance(current[0], list):
                    for index, (left, right) in enumerate(zip(current, incoming)):
                        restore(left, right, f"{path_name}[{index}]")
                    return incoming
                values = []
                for value in incoming:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise TransformerError(f"invalid checkpoint value at {path_name}")
                    values.append(float(value))
                return values
            if isinstance(current, dict):
                if not isinstance(incoming, dict) or set(current) != set(incoming):
                    raise TransformerError(f"invalid checkpoint fields at {path_name}")
                return {
                    key: restore(value, incoming[key], f"{path_name}.{key}")
                    for key, value in current.items()
                }
            raise TransformerError(f"unsupported checkpoint structure at {path_name}")

        # Restore top-level tensors and each block explicitly.
        for name in ("token_embeddings", "position_embeddings", "final_norm_scale", "final_norm_bias", "output_bias"):
            if name not in weights:
                raise TransformerError(f"missing checkpoint field weights.{name}")
            setattr(model, name, restore(getattr(model, name), weights[name], f"weights.{name}"))

        block_payloads = weights.get("blocks")
        if not isinstance(block_payloads, list) or len(block_payloads) != len(model.blocks):
            raise TransformerError("checkpoint block count is invalid")
        for index, (block, saved) in enumerate(zip(model.blocks, block_payloads)):
            if not isinstance(saved, dict):
                raise TransformerError(f"invalid block checkpoint at index {index}")
            for name in ("attention_norm_scale", "attention_norm_bias", "feed_forward_norm_scale",
                         "feed_forward_norm_bias", "ff1", "ff1_bias", "ff2", "ff2_bias"):
                if name not in saved:
                    raise TransformerError(f"missing block field {name}")
                setattr(block, name, restore(getattr(block, name), saved[name], f"blocks[{index}].{name}"))
            attention = saved.get("attention")
            if not isinstance(attention, dict):
                raise TransformerError("attention checkpoint is invalid")
            for name in ("query", "key", "value", "output", "query_bias", "key_bias", "value_bias", "output_bias"):
                if name not in attention:
                    raise TransformerError(f"missing attention field {name}")
                setattr(block.attention, name, restore(
                    getattr(block.attention, name), attention[name],
                    f"blocks[{index}].attention.{name}"
                ))
        return model

    def parameter_count(self) -> int:
        """Return the number of scalar trainable parameters."""
        total = self.vocab_size * self.dimension
        total += self.block_size * self.dimension
        total += self.vocab_size
        total += 2 * self.dimension
        for block in self.blocks:
            total += 4 * self.dimension * self.dimension + 4 * self.dimension
            total += 2 * self.dimension
            total += 2 * self.feed_forward_dim * self.dimension
            total += self.feed_forward_dim + self.dimension
        return total
