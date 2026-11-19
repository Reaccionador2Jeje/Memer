"""Loss functions and output gradients for native Memer model training."""

from __future__ import annotations

import math

from memer.models.data import IGNORE_INDEX


def cross_entropy_with_gradients(
    logits: list[list[list[float]]],
    targets: list[list[int]],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[float, list[list[list[float]]]]:
    """Return mean next-token cross-entropy and its gradient w.r.t. logits.

    Shapes:
        logits:  [batch][sequence][vocabulary]
        targets: [batch][sequence]

    Ignored targets contribute neither to the loss nor to the gradient.
    The returned gradient is averaged over all non-ignored target positions.
    """
    if not isinstance(logits, list) or not logits:
        raise ValueError("logits must be a non-empty batch")
    if not isinstance(targets, list) or len(targets) != len(logits):
        raise ValueError("targets batch shape does not match logits")

    vocabulary_size = None
    valid_count = 0
    probabilities: list[list[list[float]]] = []

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

            clean = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("logits must contain only finite numbers")
                clean.append(float(value))

            target = target_row[position]
            if isinstance(target, bool) or not isinstance(target, int):
                raise ValueError("target IDs must be integers")
            if target != ignore_index and not 0 <= target < vocabulary_size:
                raise ValueError("target ID is outside the logits vocabulary")

            peak = max(clean)
            exponentials = [math.exp(value - peak) for value in clean]
            total = sum(exponentials)
            probability_row.append([value / total for value in exponentials])
            if target != ignore_index:
                valid_count += 1
        probabilities.append(probability_row)

    if valid_count == 0:
        raise ValueError("batch contains no non-ignored targets")

    loss = 0.0
    gradients = []
    for batch_index, (probability_row, target_row) in enumerate(zip(probabilities, targets)):
        gradient_row = []
        for position, (probs, target) in enumerate(zip(probability_row, target_row)):
            if target == ignore_index:
                gradient_row.append([0.0] * len(probs))
                continue
            loss -= math.log(max(probs[target], 1e-300))
            gradient = probs.copy()
            gradient[target] -= 1.0
            gradient_row.append([value / valid_count for value in gradient])
        gradients.append(gradient_row)

    return loss / valid_count, gradients
