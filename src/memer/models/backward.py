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



def causal_attention_backward(
    sequence: list[list[float]],
    query_weights: list[list[float]],
    key_weights: list[list[float]],
    value_weights: list[list[float]],
    output_weights: list[list[float]],
    query_bias: list[float],
    key_bias: list[float],
    value_bias: list[float],
    output_bias: list[float],
    output_gradients: list[list[float]],
    *,
    heads: int,
    attention_mask: list[int] | None = None,
) -> tuple[list[list[float]], dict[str, list]]:
    """Backpropagate through the same causal multi-head attention as the model.

    Returns gradients for the input sequence and a dictionary containing
    gradients for all four projection matrices and their biases.
    """
    import math

    length = len(sequence)
    if length == 0:
        raise ValueError("attention sequence must be non-empty")
    dimension = len(sequence[0])
    if dimension == 0 or any(len(row) != dimension for row in sequence):
        raise ValueError("attention sequence rows must have equal non-zero width")
    if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1 or dimension % heads:
        raise ValueError("heads must be a positive divisor of the attention dimension")
    if attention_mask is None:
        attention_mask = [1] * length
    if len(attention_mask) != length or any(value not in (0, 1) for value in attention_mask):
        raise ValueError("attention mask must contain one binary value per position")
    if len(output_gradients) != length or any(len(row) != dimension for row in output_gradients):
        raise ValueError("attention output gradient shape is invalid")

    matrices = (query_weights, key_weights, value_weights, output_weights)
    biases = (query_bias, key_bias, value_bias, output_bias)
    for matrix in matrices:
        if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
            raise ValueError("attention projection matrix shape is invalid")
    for bias in biases:
        if len(bias) != dimension:
            raise ValueError("attention projection bias shape is invalid")
    all_values = [value for row in sequence + output_gradients for value in row]
    all_values += [value for matrix in matrices for row in matrix for value in row]
    all_values += [value for bias in biases for value in bias]
    if any(not math.isfinite(value) for value in all_values):
        raise ValueError("attention values and gradients must be finite")

    def project(rows, weights, bias):
        return [
            [
                sum(value * weights[out_index][in_index] for in_index, value in enumerate(row))
                + bias[out_index]
                for out_index in range(dimension)
            ]
            for row in rows
        ]

    queries = project(sequence, query_weights, query_bias)
    keys = project(sequence, key_weights, key_bias)
    values = project(sequence, value_weights, value_bias)
    head_dim = dimension // heads
    scale = 1.0 / math.sqrt(head_dim)
    contexts = [[0.0] * dimension for _ in range(length)]
    cached = [[[] for _ in range(heads)] for _ in range(length)]

    for position in range(length):
        for head in range(heads):
            start = head * head_dim
            end = start + head_dim
            visible = [
                previous for previous in range(position + 1)
                if attention_mask[previous]
            ]
            if not visible:
                continue
            scores = [
                sum(queries[position][index] * keys[previous][index] for index in range(start, end))
                * scale
                for previous in visible
            ]
            peak = max(scores)
            exponentials = [math.exp(score - peak) for score in scores]
            total = sum(exponentials)
            probabilities = [value / total for value in exponentials]
            cached[position][head] = list(zip(visible, probabilities))
            for previous, probability in zip(visible, probabilities):
                for index in range(start, end):
                    contexts[position][index] += probability * values[previous][index]

    # Output projection gradients.
    grad_context, grad_output_weights, grad_output_bias = linear_backward(
        contexts, output_weights, output_gradients
    )

    grad_queries = [[0.0] * dimension for _ in range(length)]
    grad_keys = [[0.0] * dimension for _ in range(length)]
    grad_values = [[0.0] * dimension for _ in range(length)]

    for position in range(length):
        for head in range(heads):
            start = head * head_dim
            end = start + head_dim
            entries = cached[position][head]
            if not entries:
                continue

            probability_gradients = []
            for previous, probability in entries:
                probability_gradients.append(sum(
                    grad_context[position][index] * values[previous][index]
                    for index in range(start, end)
                ))
                for index in range(start, end):
                    grad_values[previous][index] += probability * grad_context[position][index]

            weighted_mean = sum(
                probability * grad_probability
                for (_, probability), grad_probability in zip(entries, probability_gradients)
            )
            for (previous, probability), grad_probability in zip(entries, probability_gradients):
                grad_score = probability * (grad_probability - weighted_mean)
                for index in range(start, end):
                    grad_queries[position][index] += grad_score * keys[previous][index] * scale
                    grad_keys[previous][index] += grad_score * queries[position][index] * scale

    grad_input_q, grad_query_weights, grad_query_bias = linear_backward(
        sequence, query_weights, grad_queries
    )
    grad_input_k, grad_key_weights, grad_key_bias = linear_backward(
        sequence, key_weights, grad_keys
    )
    grad_input_v, grad_value_weights, grad_value_bias = linear_backward(
        sequence, value_weights, grad_values
    )
    grad_inputs = [
        [
            grad_input_q[position][index]
            + grad_input_k[position][index]
            + grad_input_v[position][index]
            for index in range(dimension)
        ]
        for position in range(length)
    ]
    parameter_gradients = {
        "query": grad_query_weights,
        "key": grad_key_weights,
        "value": grad_value_weights,
        "output": grad_output_weights,
        "query_bias": grad_query_bias,
        "key_bias": grad_key_bias,
        "value_bias": grad_value_bias,
        "output_bias": grad_output_bias,
    }
    return grad_inputs, parameter_gradients
