"""Backward primitives used by the native Memer Transformer trainer."""

from __future__ import annotations

import math


def linear_backward(
    inputs: list[list[float]],
    weights: list[list[float]],
    output_gradients: list[list[float]],
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    """Compute gradients for a batch of independent linear-layer rows.

    The forward operation is output[b][o] = sum(inputs[b][i] * weights[o][i]).
    Returns gradients for inputs, weights, and the (shared) output bias.
    """
    if not inputs or not weights or not output_gradients:
        raise ValueError("linear backward inputs must be non-empty")
    input_size = len(inputs[0])
    output_size = len(weights)
    if input_size == 0 or any(len(row) != input_size for row in inputs):
        raise ValueError("input rows must have a consistent non-zero width")
    if any(len(row) != input_size for row in weights):
        raise ValueError("weight rows must match the input width")
    if len(output_gradients) != len(inputs) or any(
        len(row) != output_size for row in output_gradients
    ):
        raise ValueError("output gradient shape does not match linear output")

    grad_inputs = [[0.0] * input_size for _ in inputs]
    grad_weights = [[0.0] * input_size for _ in weights]
    grad_bias = [0.0] * output_size

    for batch_index, (input_row, grad_row) in enumerate(zip(inputs, output_gradients)):
        for output_index, grad_value in enumerate(grad_row):
            if not math.isfinite(grad_value):
                raise ValueError("gradients must contain finite numbers")
            grad_bias[output_index] += grad_value
            for input_index, input_value in enumerate(input_row):
                weight = weights[output_index][input_index]
                grad_weights[output_index][input_index] += grad_value * input_value
                grad_inputs[batch_index][input_index] += grad_value * weight

    return grad_inputs, grad_weights, grad_bias


def layer_norm_backward(
    inputs: list[float],
    scale: list[float],
    output_gradients: list[float],
    *,
    epsilon: float = 1e-5,
) -> tuple[list[float], list[float], list[float]]:
    """Compute input, scale, and bias gradients for one LayerNorm vector."""
    size = len(inputs)
    if size == 0 or len(scale) != size or len(output_gradients) != size:
        raise ValueError("LayerNorm vectors must have equal non-zero lengths")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number")
    if any(not math.isfinite(value) for value in inputs + scale + output_gradients):
        raise ValueError("LayerNorm values and gradients must be finite")

    mean = sum(inputs) / size
    centered = [value - mean for value in inputs]
    variance = sum(value * value for value in centered) / size
    inverse_std = 1.0 / math.sqrt(variance + epsilon)
    normalized = [value * inverse_std for value in centered]

    grad_normalized = [
        gradient * factor
        for gradient, factor in zip(output_gradients, scale)
    ]
    grad_scale = [
        gradient * value
        for gradient, value in zip(output_gradients, normalized)
    ]
    grad_bias = output_gradients.copy()

    mean_grad = sum(grad_normalized) / size
    mean_grad_normalized = sum(
        gradient * value for gradient, value in zip(grad_normalized, normalized)
    ) / size
    grad_inputs = [
        inverse_std * (
            gradient - mean_grad - value * mean_grad_normalized
        )
        for gradient, value in zip(grad_normalized, normalized)
    ]
    return grad_inputs, grad_scale, grad_bias


def relu_backward(
    inputs: list[list[float]],
    output_gradients: list[list[float]],
) -> list[list[float]]:
    """Pass gradients only through positive inputs of a ReLU operation."""
    if len(inputs) != len(output_gradients):
        raise ValueError("ReLU gradient batch shape does not match inputs")
    result = []
    for input_row, gradient_row in zip(inputs, output_gradients):
        if len(input_row) != len(gradient_row):
            raise ValueError("ReLU gradient row shape does not match inputs")
        if any(not math.isfinite(value) for value in input_row + gradient_row):
            raise ValueError("ReLU values and gradients must be finite")
        result.append([
            gradient if value > 0.0 else 0.0
            for value, gradient in zip(input_row, gradient_row)
        ])
    return result
