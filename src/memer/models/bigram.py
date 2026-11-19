"""Small pure-Python neural character language model.

Architecture: context embedding average -> linear projection -> softmax.
This is an educational, trainable next-character model, not a foundation model.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from memer.models.tokenizer import BOS, EOS, CharacterTokenizer


class ModelError(ValueError):
    """Raised when model configuration, data, or weights are invalid."""


class TinyLanguageModel:
    FORMAT = "MEMER-NLM"
    FORMAT_VERSION = 1

    def __init__(
        self,
        tokenizer: CharacterTokenizer,
        embedding_dim: int = 16,
        context_size: int = 8,
        seed: int = 17,
    ) -> None:
        if not isinstance(tokenizer, CharacterTokenizer):
            raise ModelError("tokenizer must be a CharacterTokenizer")
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or not 2 <= embedding_dim <= 128:
            raise ModelError("embedding_dim must be an integer between 2 and 128")
        if isinstance(context_size, bool) or not isinstance(context_size, int) or not 1 <= context_size <= 64:
            raise ModelError("context_size must be an integer between 1 and 64")
        if tokenizer.vocab_size > 256:
            raise ModelError("the first model supports at most 256 distinct tokens")

        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.seed = seed
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(embedding_dim)
        self.embeddings = [
            [rng.uniform(-scale, scale) for _ in range(embedding_dim)]
            for _ in range(tokenizer.vocab_size)
        ]
        self.output_weights = [
            [rng.uniform(-scale, scale) for _ in range(embedding_dim)]
            for _ in range(tokenizer.vocab_size)
        ]
        self.output_biases = [0.0] * tokenizer.vocab_size
        self.training_history: list[dict] = []

    def _context(self, token_ids: list[int]) -> list[int]:
        bos_id = self.tokenizer.token_to_id[BOS]
        recent = token_ids[-self.context_size:]
        return [bos_id] * (self.context_size - len(recent)) + recent

    def _forward(self, context_ids: list[int]) -> tuple[list[float], list[float]]:
        if len(context_ids) != self.context_size:
            raise ModelError("context has an incorrect length")
        vector = [
            sum(self.embeddings[token_id][dimension] for token_id in context_ids) / self.context_size
            for dimension in range(self.embedding_dim)
        ]
        logits = [
            sum(weight * value for weight, value in zip(row, vector)) + bias
            for row, bias in zip(self.output_weights, self.output_biases)
        ]
        peak = max(logits)
        exponents = [math.exp(value - peak) for value in logits]
        total = sum(exponents)
        probabilities = [value / total for value in exponents]
        return vector, probabilities

    def train(
        self,
        text: str,
        epochs: int = 10,
        learning_rate: float = 0.05,
        max_samples: int = 20000,
    ) -> dict:
        if not isinstance(text, str) or len(text) < 20:
            raise ModelError("training text must contain at least 20 characters")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 1000:
            raise ModelError("epochs must be between 1 and 1000")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or not 0.0001 <= learning_rate <= 1:
            raise ModelError("learning_rate must be between 0.0001 and 1")
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or not 1 <= max_samples <= 200000:
            raise ModelError("max_samples must be between 1 and 200000")

        sequence = self.tokenizer.encode(text, add_boundaries=True)
        samples = []
        for position in range(1, len(sequence)):
            samples.append((self._context(sequence[:position]), sequence[position]))
        if len(samples) > max_samples:
            samples = samples[:max_samples]

        rng = random.Random(self.seed + len(self.training_history))
        losses = []
        for _ in range(epochs):
            rng.shuffle(samples)
            total_loss = 0.0
            for context_ids, target_id in samples:
                vector, probabilities = self._forward(context_ids)
                probability = max(probabilities[target_id], 1e-12)
                total_loss -= math.log(probability)

                gradients = probabilities[:]
                gradients[target_id] -= 1.0
                context_gradient = [0.0] * self.embedding_dim

                for token_id, gradient in enumerate(gradients):
                    row = self.output_weights[token_id]
                    for dimension in range(self.embedding_dim):
                        context_gradient[dimension] += gradient * row[dimension]
                        row[dimension] -= learning_rate * gradient * vector[dimension]
                    self.output_biases[token_id] -= learning_rate * gradient

                for token_id in context_ids:
                    embedding = self.embeddings[token_id]
                    for dimension in range(self.embedding_dim):
                        embedding[dimension] -= (
                            learning_rate * context_gradient[dimension] / self.context_size
                        )

            average_loss = total_loss / len(samples)
            self.training_history.append({
                "epoch": len(self.training_history) + 1,
                "loss": average_loss,
                "samples": len(samples),
            })
            losses.append(average_loss)

        return {
            "epochs": epochs,
            "total_epochs": len(self.training_history),
            "samples": len(samples),
            "loss": losses[-1],
            "initial_loss": losses[0],
        }

    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> str:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 2000:
            raise ModelError("max_new_tokens must be between 1 and 2000")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0.1 <= temperature <= 2:
            raise ModelError("temperature must be between 0.1 and 2")

        generated = self.tokenizer.encode(prompt)
        result = list(generated)
        rng = random.Random(seed)
        eos_id = self.tokenizer.token_to_id[EOS]

        for _ in range(max_new_tokens):
            context_ids = self._context(generated)
            _, probabilities = self._forward(context_ids)
            scaled = [math.log(max(p, 1e-12)) / temperature for p in probabilities]
            peak = max(scaled)
            weights = [math.exp(value - peak) for value in scaled]
            total = sum(weights)
            threshold = rng.random() * total
            cumulative = 0.0
            next_id = 0
            for token_id, weight in enumerate(weights):
                cumulative += weight
                if cumulative >= threshold:
                    next_id = token_id
                    break
            if next_id == eos_id:
                break
            generated.append(next_id)
            result.append(next_id)

        return self.tokenizer.decode(result)

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.FORMAT,
            "format_version": self.FORMAT_VERSION,
            "tokenizer": self.tokenizer.to_dict(),
            "embedding_dim": self.embedding_dim,
            "context_size": self.context_size,
            "seed": self.seed,
            "embeddings": self.embeddings,
            "output_weights": self.output_weights,
            "output_biases": self.output_biases,
            "training_history": self.training_history,
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
        return str(target)

    @classmethod
    def load(cls, path: str | Path) -> "TinyLanguageModel":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ModelError(f"could not load model: {error}") from error
        if not isinstance(payload, dict) or payload.get("format") != cls.FORMAT or payload.get("format_version") != cls.FORMAT_VERSION:
            raise ModelError("unsupported model file format")

        tokenizer = CharacterTokenizer.from_dict(payload.get("tokenizer"))
        model = cls(
            tokenizer,
            embedding_dim=payload.get("embedding_dim"),
            context_size=payload.get("context_size"),
            seed=payload.get("seed", 17),
        )
        embeddings = payload.get("embeddings")
        weights = payload.get("output_weights")
        biases = payload.get("output_biases")
        vocab_size = tokenizer.vocab_size
        dim = model.embedding_dim

        def valid_matrix(matrix):
            return (
                isinstance(matrix, list)
                and len(matrix) == vocab_size
                and all(isinstance(row, list) and len(row) == dim for row in matrix)
                and all(
                    isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                    for row in matrix for value in row
                )
            )

        if not valid_matrix(embeddings) or not valid_matrix(weights):
            raise ModelError("model weight dimensions or values are invalid")
        if not isinstance(biases, list) or len(biases) != vocab_size or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in biases
        ):
            raise ModelError("model bias dimensions or values are invalid")

        model.embeddings = embeddings
        model.output_weights = weights
        model.output_biases = biases
        history = payload.get("training_history", [])
        if not isinstance(history, list):
            raise ModelError("training history is invalid")
        model.training_history = history
        return model
