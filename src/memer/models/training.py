"""Loss, optimizer, and end-to-end training utilities for Memer Models."""

from __future__ import annotations

import math
from memer.models.data import IGNORE_INDEX, make_batches


def cross_entropy_with_gradients(
    logits: list[list[list[float]]],
    targets: list[list[int]],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[float, list[list[list[float]]]]:
    """Return mean next-token cross-entropy and its gradient w.r.t. logits."""
    if not isinstance(logits, list) or not logits:
        raise ValueError("logits must be a non-empty batch")
    if not isinstance(targets, list) or len(targets) != len(logits):
        raise ValueError("targets batch shape does not match logits")
    vocabulary_size = None
    valid_count = 0
    probabilities = []
    for batch_index, (logit_row, target_row) in enumerate(zip(logits, targets)):
        if not isinstance(logit_row, list) or not logit_row:
            raise ValueError(f"logits row {batch_index} must be non-empty")
        if not isinstance(target_row, list) or len(target_row) != len(logit_row):
            raise ValueError(f"targets row {batch_index} shape does not match logits")
        probability_row = []
        for position, vector in enumerate(logit_row):
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"logits at [{batch_index}][{position}] must be non-empty")
            if vocabulary_size is None:
                vocabulary_size = len(vector)
            elif len(vector) != vocabulary_size:
                raise ValueError("all logits vectors must have the same vocabulary size")
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in vector):
                raise ValueError("logits must contain only finite numbers")
            target = target_row[position]
            if isinstance(target, bool) or not isinstance(target, int):
                raise ValueError("target IDs must be integers")
            if target != ignore_index and not 0 <= target < vocabulary_size:
                raise ValueError("target ID is outside the logits vocabulary")
            peak = max(vector)
            exps = [math.exp(value - peak) for value in vector]
            total = sum(exps)
            probability_row.append([value / total for value in exps])
            if target != ignore_index:
                valid_count += 1
        probabilities.append(probability_row)
    if valid_count == 0:
        raise ValueError("batch contains no non-ignored targets")
    loss = 0.0
    gradients = []
    for probability_row, target_row in zip(probabilities, targets):
        gradient_row = []
        for probs, target in zip(probability_row, target_row):
            if target == ignore_index:
                gradient_row.append([0.0] * len(probs))
                continue
            loss -= math.log(max(probs[target], 1e-300))
            grad = probs.copy()
            grad[target] -= 1.0
            gradient_row.append([value / valid_count for value in grad])
        gradients.append(gradient_row)
    return loss / valid_count, gradients


def _zeros_like(value):
    if isinstance(value, dict):
        return {key: _zeros_like(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_zeros_like(item) for item in value]
    return 0.0


class Adam:
    """Small dependency-free Adam optimizer over nested list parameter trees."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        gradient_clip_norm: float | None = 1.0,
    ) -> None:
        values = (learning_rate, beta1, beta2, epsilon, weight_decay)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Adam options must be finite")
        if learning_rate <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("invalid Adam learning rate or beta values")
        if epsilon <= 0 or weight_decay < 0:
            raise ValueError("Adam epsilon must be positive and weight decay non-negative")
        if gradient_clip_norm is not None and (
            not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be positive or None")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        self.step_count = 0
        self.first_moment = None
        self.second_moment = None

    def step(self, parameters, gradients) -> None:
        """Update parameters in place; trees must have identical list structure."""
        if self.first_moment is None:
            self.first_moment = _zeros_like(parameters)
            self.second_moment = _zeros_like(parameters)
        self.step_count += 1
        squared_norm = 0.0

        def inspect(param, grad):
            nonlocal squared_norm
            if isinstance(param, dict):
                if not isinstance(grad, dict) or set(param) != set(grad):
                    raise ValueError("Adam parameter and gradient fields differ")
                for key in param:
                    inspect(param[key], grad[key])
            elif isinstance(param, list):
                if not isinstance(grad, list) or len(param) != len(grad):
                    raise ValueError("Adam parameter and gradient shapes differ")
                for p, g in zip(param, grad):
                    inspect(p, g)
            else:
                if isinstance(grad, bool) or not isinstance(grad, (int, float)) or not math.isfinite(grad):
                    raise ValueError("Adam gradients must be finite numbers")
                squared_norm += float(grad) * float(grad)
        inspect(parameters, gradients)
        clip = 1.0
        if self.gradient_clip_norm is not None:
            norm = math.sqrt(squared_norm)
            if norm > self.gradient_clip_norm:
                clip = self.gradient_clip_norm / (norm + 1e-12)

        correction1 = 1.0 - self.beta1 ** self.step_count
        correction2 = 1.0 - self.beta2 ** self.step_count

        def update(param, grad, first, second):
            if isinstance(param, list):
                if not isinstance(grad, list) or len(param) != len(grad):
                    raise ValueError("Adam parameter and gradient shapes differ")
                for index in range(len(param)):
                    update(param[index], grad[index], first[index], second[index])
                return
            g = float(grad) * clip
            first_value = self.beta1 * first + (1.0 - self.beta1) * g
            second_value = self.beta2 * second + (1.0 - self.beta2) * g * g
            first_hat = first_value / correction1
            second_hat = second_value / correction2
            param -= self.learning_rate * (
                first_hat / (math.sqrt(second_hat) + self.epsilon)
                + self.weight_decay * param
            )
            return first_value, second_value

        def apply(param, grad, first, second):
            if isinstance(param, list):
                for index in range(len(param)):
                    apply(param[index], grad[index], first[index], second[index])
            else:
                first_value, second_value = update(param, grad, first, second)
                return first_value, second_value
            return first, second

        def update_states(param, grad, first, second):
            if isinstance(param, list):
                for index in range(len(param)):
                    update_states(param[index], grad[index], first[index], second[index])
            else:
                first_value, second_value = update(param, grad, first, second)
                return first_value, second_value
            return first, second

        # Recursively replace scalar moment leaves while mutating parameter leaves.
        def walk(param, grad, first, second):
            if isinstance(param, dict):
                if not isinstance(grad, dict) or set(param) != set(grad):
                    raise ValueError("Adam parameter and gradient fields differ")
                for key in param:
                    first[key], second[key] = walk(
                        param[key], grad[key], first[key], second[key]
                    )
                return first, second
            if isinstance(param, list):
                for index in range(len(param)):
                    first[index], second[index] = walk(
                        param[index], grad[index], first[index], second[index]
                    )
                return first, second
            g = float(grad) * clip
            first_value = self.beta1 * first + (1.0 - self.beta1) * g
            second_value = self.beta2 * second + (1.0 - self.beta2) * g * g
            param -= self.learning_rate * (
                (first_value / correction1) /
                (math.sqrt(second_value / correction2) + self.epsilon)
                + self.weight_decay * param
            )
            return first_value, second_value

        self.first_moment, self.second_moment = walk(
            parameters, gradients, self.first_moment, self.second_moment
        )

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "first_moment": self.first_moment,
            "second_moment": self.second_moment,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
        }


def train_batch(model, batch: dict[str, list[list[int]]], optimizer: Adam) -> float:
    """Run one complete forward/loss/backward/Adam update."""
    logits = model.forward(batch["input_ids"], batch["attention_mask"])
    loss, logits_gradients = cross_entropy_with_gradients(
        logits, batch["target_ids"]
    )
    gradients = model.backward(
        batch["input_ids"], logits_gradients, batch["attention_mask"]
    )
    optimizer.step(model._weights_payload(), gradients)
    return loss


def train_model(
    model,
    documents: list[list[int]],
    *,
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    seed: int = 17,
    optimizer: Adam | None = None,
) -> list[float]:
    """Train a MemerTransformer and return mean loss for each epoch."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if optimizer is None:
        optimizer = Adam(learning_rate)
    history = []
    for epoch in range(epochs):
        batches = list(make_batches(
            documents, model.block_size, batch_size,
            shuffle=True, seed=seed + epoch,
        ))
        if not batches:
            raise ValueError("training dataset produced no batches")
        total_loss = 0.0
        for batch in batches:
            total_loss += train_batch(model, batch, optimizer)
        history.append(total_loss / len(batches))
    return history
