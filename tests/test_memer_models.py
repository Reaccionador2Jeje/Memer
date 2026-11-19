import math
import tempfile
import unittest
from pathlib import Path

from memer.models import TinyLanguageModel
from memer.models.tokenizer import CharacterTokenizer


CORPUS = (
    "Memer is an experimental language model from AtDev. "
    "Memer learns patterns from text and generates new text. "
    "AtDev builds open and accessible software. "
) * 3


class MemerModelsTests(unittest.TestCase):
    def make_model(self):
        tokenizer = CharacterTokenizer.from_text(CORPUS)
        return TinyLanguageModel(tokenizer, embedding_dim=8, context_size=5, seed=3)

    def test_training_updates_model(self):
        model = self.make_model()
        result = model.train(CORPUS, epochs=3, learning_rate=0.03)
        self.assertEqual(result["epochs"], 3)
        self.assertTrue(result["loss"] > 0)
        self.assertEqual(len(model.training_history), 3)

    def test_generation_returns_text(self):
        model = self.make_model()
        model.train(CORPUS, epochs=2, learning_rate=0.03)
        output = model.generate("Memer", max_new_tokens=20, seed=4)
        self.assertTrue(output.startswith("Memer"))
        self.assertLessEqual(len(output), 25)

    def test_save_and_load(self):
        model = self.make_model()
        model.train(CORPUS, epochs=2, learning_rate=0.03)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = TinyLanguageModel.load(path)
            self.assertEqual(restored.tokenizer.vocabulary, model.tokenizer.vocabulary)
            self.assertEqual(restored.training_history, model.training_history)
            self.assertEqual(
                restored.generate("At", max_new_tokens=12, seed=9),
                model.generate("At", max_new_tokens=12, seed=9),
            )

    def test_rejects_large_vocabulary(self):
        text = "".join(chr(0x1000 + index) for index in range(260))
        tokenizer = CharacterTokenizer.from_text(text)
        with self.assertRaises(ValueError):
            TinyLanguageModel(tokenizer)



class ByteTokenizerTests(unittest.TestCase):
    def test_round_trip_unicode(self):
        from memer.models.byte_tokenizer import ByteTokenizer, BOS, EOS

        tokenizer = ByteTokenizer()
        text = "Memer: ¡Hola, mundo! 🚀 中文"
        encoded = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(encoded), text)
        bounded = tokenizer.encode(text, add_boundaries=True)
        self.assertEqual(bounded[0], BOS)
        self.assertEqual(bounded[-1], EOS)
        self.assertEqual(tokenizer.decode(bounded), text)

    def test_fixed_vocabulary(self):
        from memer.models.byte_tokenizer import ByteTokenizer

        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.vocab_size, 259)
        self.assertEqual(ByteTokenizer.from_dict(tokenizer.to_dict()).vocab_size, 259)

    def test_rejects_invalid_token_ids(self):
        from memer.models.byte_tokenizer import ByteTokenizer

        tokenizer = ByteTokenizer()
        with self.assertRaises(ValueError):
            tokenizer.decode([999])



class CausalBatchTests(unittest.TestCase):
    def test_batches_shift_targets_and_pad(self):
        from memer.models.byte_tokenizer import ByteTokenizer, BOS, EOS, PAD
        from memer.models.data import IGNORE_INDEX, encode_documents, make_batches

        tokenizer = ByteTokenizer()
        documents = encode_documents("a\nxy", tokenizer)
        batches = list(make_batches(documents, block_size=3, batch_size=2))
        self.assertTrue(batches)
        batch = batches[0]
        self.assertEqual(batch["input_ids"][0][0], BOS)
        self.assertEqual(batch["target_ids"][0][0], ord("a"))
        self.assertEqual(batch["input_ids"][0][2], PAD)
        self.assertEqual(batch["target_ids"][0][2], IGNORE_INDEX)
        self.assertEqual(batch["attention_mask"][0], [1, 1, 0])

    def test_batching_is_deterministic_when_shuffled(self):
        from memer.models.byte_tokenizer import ByteTokenizer
        from memer.models.data import encode_documents, make_batches

        documents = encode_documents("first\nsecond\nthird", ByteTokenizer())
        first = list(make_batches(documents, 4, 2, shuffle=True, seed=9))
        second = list(make_batches(documents, 4, 2, shuffle=True, seed=9))
        self.assertEqual(first, second)

    def test_rejects_invalid_batch_sizes(self):
        from memer.models.data import make_batches

        with self.assertRaises(ValueError):
            list(make_batches([], 0, 2))
        with self.assertRaises(ValueError):
            list(make_batches([], 4, 0))



class MemerTransformerTests(unittest.TestCase):
    def make_model(self):
        from memer.models.transformer import MemerTransformer

        return MemerTransformer(
            vocab_size=259, dimension=8, heads=2, layers=1,
            feed_forward_dim=16, block_size=8, seed=11,
        )

    def test_forward_logits_shape(self):
        model = self.make_model()
        logits = model.forward([[256, ord("h"), ord("i")], [256, ord("o"), ord("k")]])
        self.assertEqual(len(logits), 2)
        self.assertEqual(len(logits[0]), 3)
        self.assertEqual(len(logits[0][0]), 259)
        self.assertTrue(all(isinstance(value, float) for value in logits[0][0]))

    def test_causal_attention_blocks_future_tokens(self):
        model = self.make_model()
        first = model.forward([[256, ord("a"), ord("b")]])
        changed_future = model.forward([[256, ord("a"), ord("z")]])
        self.assertEqual(first[0][0], changed_future[0][0])
        self.assertEqual(first[0][1], changed_future[0][1])
        self.assertNotEqual(first[0][2], changed_future[0][2])

    def test_padding_mask_and_parameter_count(self):
        model = self.make_model()
        logits = model.forward([[256, ord("a"), 258]], [[1, 1, 0]])
        self.assertEqual(len(logits[0]), 3)
        self.assertGreater(model.parameter_count(), 0)

    def test_rejects_invalid_transformer_inputs(self):
        from memer.models.transformer import TransformerError

        model = self.make_model()
        with self.assertRaises(TransformerError):
            model.forward([[256, 999]])
        with self.assertRaises(TransformerError):
            model.forward([[256] * 9])
        with self.assertRaises(TransformerError):
            model.forward([[256, 97], [256]])



class TransformerCheckpointTests(unittest.TestCase):
    def make_model(self):
        from memer.models.transformer import MemerTransformer

        return MemerTransformer(
            vocab_size=259, dimension=8, heads=2, layers=1,
            feed_forward_dim=16, block_size=8, seed=23,
        )

    def test_checkpoint_round_trip_preserves_logits(self):
        from memer.models.transformer import MemerTransformer

        model = self.make_model()
        sample = [[256, ord("M"), ord("e")]]
        expected = model.forward(sample)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memer-transformer.json"
            model.save(path)
            restored = MemerTransformer.load(path)
            self.assertEqual(restored.forward(sample), expected)
            self.assertEqual(restored.parameter_count(), model.parameter_count())

    def test_checkpoint_rejects_invalid_format(self):
        from memer.models.transformer import MemerTransformer, TransformerError

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"format":"OTHER","format_version":1}', encoding="utf-8")
            with self.assertRaises(TransformerError):
                MemerTransformer.load(path)

    def test_checkpoint_rejects_wrong_tensor_shape(self):
        from memer.models.transformer import MemerTransformer, TransformerError

        model = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            model.save(path)
            import json
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["weights"]["token_embeddings"] = [[0.0]]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(TransformerError):
                MemerTransformer.load(path)


if __name__ == "__main__":
    unittest.main()



class TransformerLossTests(unittest.TestCase):
    def test_cross_entropy_and_gradient(self):
        from memer.models.training import cross_entropy_with_gradients

        logits = [[[2.0, 1.0, 0.0]]]
        loss, gradients = cross_entropy_with_gradients(logits, [[0]])
        self.assertAlmostEqual(loss, 0.4076059644, places=7)
        self.assertAlmostEqual(sum(gradients[0][0]), 0.0, places=12)
        self.assertLess(gradients[0][0][0], 0.0)
        self.assertGreater(gradients[0][0][1], 0.0)

    def test_ignored_targets_have_zero_gradient(self):
        from memer.models.data import IGNORE_INDEX
        from memer.models.training import cross_entropy_with_gradients

        loss, gradients = cross_entropy_with_gradients(
            [[[1.0, 2.0], [100.0, -100.0]]],
            [[1, IGNORE_INDEX]],
        )
        self.assertAlmostEqual(loss, 0.3132616875, places=7)
        self.assertEqual(gradients[0][1], [0.0, 0.0])

    def test_gradient_matches_finite_difference(self):
        from memer.models.training import cross_entropy_with_gradients

        logits = [[[0.3, -0.2, 0.7]]]
        _, gradients = cross_entropy_with_gradients(logits, [[2]])
        epsilon = 1e-5
        plus = [[[0.3, -0.2, 0.7 + epsilon]]]
        minus = [[[0.3, -0.2, 0.7 - epsilon]]]
        loss_plus, _ = cross_entropy_with_gradients(plus, [[2]])
        loss_minus, _ = cross_entropy_with_gradients(minus, [[2]])
        numeric = (loss_plus - loss_minus) / (2 * epsilon)
        self.assertAlmostEqual(gradients[0][0][2], numeric, places=6)

    def test_rejects_batches_without_valid_targets(self):
        from memer.models.data import IGNORE_INDEX
        from memer.models.training import cross_entropy_with_gradients

        with self.assertRaises(ValueError):
            cross_entropy_with_gradients([[[0.0, 1.0]]], [[IGNORE_INDEX]])

    def test_rejects_invalid_target(self):
        from memer.models.training import cross_entropy_with_gradients

        with self.assertRaises(ValueError):
            cross_entropy_with_gradients([[[0.0, 1.0]]], [[3]])



class TransformerBackwardPrimitiveTests(unittest.TestCase):
    def test_linear_backward_shapes_and_values(self):
        from memer.models.backward import linear_backward

        inputs = [[2.0, -1.0], [1.0, 3.0]]
        weights = [[0.5, 2.0], [-1.0, 1.0]]
        output_gradients = [[1.0, 2.0], [-1.0, 0.5]]
        grad_inputs, grad_weights, grad_bias = linear_backward(
            inputs, weights, output_gradients
        )
        self.assertEqual(grad_inputs, [[-1.5, 4.0], [-1.0, -1.5]])
        self.assertEqual(grad_weights, [[1.0, -4.0], [4.5, -0.5]])
        self.assertEqual(grad_bias, [0.0, 2.5])

    def test_layer_norm_backward_matches_finite_difference(self):
        from memer.models.backward import layer_norm_backward

        values = [0.2, -0.4, 1.1]
        scale = [1.2, 0.8, -0.7]
        upstream = [0.3, -0.2, 0.5]
        grad_inputs, _, _ = layer_norm_backward(values, scale, upstream)

        def objective(candidate):
            mean = sum(candidate) / len(candidate)
            variance = sum((value - mean) ** 2 for value in candidate) / len(candidate)
            inv = 1.0 / (variance + 1e-5) ** 0.5
            normalized = [(value - mean) * inv for value in candidate]
            return sum(a * b * c for a, b, c in zip(normalized, scale, upstream))

        epsilon = 1e-5
        for index in range(len(values)):
            plus = values.copy()
            minus = values.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numeric = (objective(plus) - objective(minus)) / (2 * epsilon)
            self.assertAlmostEqual(grad_inputs[index], numeric, places=5)

    def test_relu_backward_blocks_non_positive_inputs(self):
        from memer.models.backward import relu_backward

        result = relu_backward([[-1.0, 0.0, 2.0]], [[3.0, 4.0, 5.0]])
        self.assertEqual(result, [[0.0, 0.0, 5.0]])

    def test_backward_primitives_reject_invalid_shapes(self):
        from memer.models.backward import layer_norm_backward, linear_backward, relu_backward

        with self.assertRaises(ValueError):
            linear_backward([[1.0]], [[1.0]], [[1.0, 2.0]])
        with self.assertRaises(ValueError):
            layer_norm_backward([1.0, 2.0], [1.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            relu_backward([[1.0]], [[1.0, 2.0]])



class CausalAttentionBackwardTests(unittest.TestCase):
    def make_attention(self):
        import random
        from memer.models.transformer import CausalSelfAttention

        attention = CausalSelfAttention(2, 1, random.Random(31))
        attention.query = [[0.2, -0.1], [0.05, 0.3]]
        attention.key = [[-0.2, 0.15], [0.25, 0.1]]
        attention.value = [[0.4, -0.3], [0.2, 0.35]]
        attention.output = [[0.3, 0.1], [-0.15, 0.25]]
        attention.query_bias = [0.01, -0.02]
        attention.key_bias = [0.03, 0.01]
        attention.value_bias = [-0.01, 0.02]
        attention.output_bias = [0.0, 0.04]
        return attention

    def test_attention_gradients_match_finite_differences(self):
        from memer.models.backward import causal_attention_backward

        attention = self.make_attention()
        sequence = [[0.7, -0.2], [0.1, 0.9]]
        upstream = [[0.3, -0.4], [0.2, 0.5]]
        grad_inputs, grads = causal_attention_backward(
            sequence,
            attention.query, attention.key, attention.value, attention.output,
            attention.query_bias, attention.key_bias, attention.value_bias,
            attention.output_bias, upstream, heads=1,
        )

        def objective():
            output = attention.forward(sequence)
            return sum(
                output[position][dimension] * upstream[position][dimension]
                for position in range(len(output))
                for dimension in range(len(output[position]))
            )

        epsilon = 1e-5
        checks = [
            (attention.query, 0, 0, grads["query"][0][0]),
            (attention.key, 1, 0, grads["key"][1][0]),
            (attention.value, 0, 1, grads["value"][0][1]),
            (attention.output, 1, 0, grads["output"][1][0]),
            (sequence, 1, 1, grad_inputs[1][1]),
        ]
        for matrix, row, column, analytic in checks:
            original = matrix[row][column]
            matrix[row][column] = original + epsilon
            plus = objective()
            matrix[row][column] = original - epsilon
            minus = objective()
            matrix[row][column] = original
            numeric = (plus - minus) / (2 * epsilon)
            self.assertAlmostEqual(analytic, numeric, places=5)

    def test_attention_mask_blocks_masked_key_and_value_gradients(self):
        from memer.models.backward import causal_attention_backward

        attention = self.make_attention()
        sequence = [[0.7, -0.2], [0.1, 0.9]]
        upstream = [[0.3, -0.4], [0.2, 0.5]]
        _, grads = causal_attention_backward(
            sequence,
            attention.query, attention.key, attention.value, attention.output,
            attention.query_bias, attention.key_bias, attention.value_bias,
            attention.output_bias, upstream, heads=1,
            attention_mask=[1, 0],
        )
        self.assertEqual(grads["key_activations"][1], [0.0, 0.0])
        self.assertEqual(grads["value_activations"][1], [0.0, 0.0])

    def test_attention_backward_rejects_invalid_dimensions(self):
        from memer.models.backward import causal_attention_backward

        attention = self.make_attention()
        with self.assertRaises(ValueError):
            causal_attention_backward(
                [[1.0, 2.0]], attention.query, attention.key,
                attention.value, attention.output,
                attention.query_bias, attention.key_bias,
                attention.value_bias, attention.output_bias,
                [[1.0, 1.0]], heads=3,
            )


class TransformerBlockBackwardTests(unittest.TestCase):
    def make_block(self):
        import random
        from memer.models.transformer import TransformerBlock

        return TransformerBlock(4, 2, 5, random.Random(47))

    def test_block_backward_shapes_and_finite_differences(self):
        block = self.make_block()
        sequence = [
            [0.2, -0.4, 0.7, 0.1],
            [-0.3, 0.8, 0.15, -0.6],
        ]
        upstream = [
            [0.3, -0.2, 0.1, 0.4],
            [-0.1, 0.25, 0.5, -0.3],
        ]
        grad_inputs, grads = block.backward(sequence, upstream)
        self.assertEqual(len(grad_inputs), 2)
        self.assertTrue(all(len(row) == 4 for row in grad_inputs))
        self.assertEqual(len(grads["ff1"]), 5)
        self.assertEqual(len(grads["ff2"]), 4)

        def objective():
            output = block.forward(sequence)
            return sum(
                output[row][column] * upstream[row][column]
                for row in range(2) for column in range(4)
            )

        epsilon = 1e-5
        checks = [
            (sequence, 1, 2, grad_inputs[1][2]),
            (block.ff2, 0, 1, grads["ff2"][0][1]),
            (block.attention.query, 1, 2, grads["attention"]["query"][1][2]),
        ]
        for tensor, row, column, analytic in checks:
            original = tensor[row][column]
            tensor[row][column] = original + epsilon
            plus = objective()
            tensor[row][column] = original - epsilon
            minus = objective()
            tensor[row][column] = original
            numeric = (plus - minus) / (2 * epsilon)
            self.assertAlmostEqual(analytic, numeric, places=5)

    def test_block_backward_respects_attention_padding_mask(self):
        block = self.make_block()
        sequence = [[0.1, 0.2, -0.3, 0.4], [0.5, -0.2, 0.1, 0.3]]
        upstream = [[0.2, 0.1, -0.4, 0.3], [0.1, -0.3, 0.2, 0.4]]
        grad_inputs, grads = block.backward(
            sequence, upstream, attention_mask=[1, 0]
        )
        self.assertEqual(len(grad_inputs), 2)
        self.assertEqual(
            grads["attention"]["key_activations"][1], [0.0, 0.0, 0.0, 0.0]
        )
        self.assertEqual(
            grads["attention"]["value_activations"][1], [0.0, 0.0, 0.0, 0.0]
        )

    def test_block_backward_rejects_invalid_shapes(self):
        block = self.make_block()
        with self.assertRaises(ValueError):
            block.backward([[0.1, 0.2]], [[0.1, 0.2]])


class FullTransformerTrainingTests(unittest.TestCase):
    def make_model(self):
        from memer.models.transformer import MemerTransformer

        return MemerTransformer(
            vocab_size=259, dimension=4, heads=2, layers=1,
            feed_forward_dim=5, block_size=4, seed=61,
        )

    def test_full_model_backward_matches_finite_differences(self):
        from memer.models.training import cross_entropy_with_gradients

        model = self.make_model()
        inputs = [[256, 97, 98]]
        targets = [[97, 98, 257]]
        mask = [[1, 1, 1]]
        logits = model.forward(inputs, mask)
        loss, upstream = cross_entropy_with_gradients(logits, targets)
        grads = model.backward(inputs, upstream, mask)

        def objective():
            values = model.forward(inputs, mask)
            return cross_entropy_with_gradients(values, targets)[0]

        epsilon = 1e-5
        checks = [
            (model.token_embeddings, 97, 0, grads["token_embeddings"][97][0]),
            (model.position_embeddings, 1, 2, grads["position_embeddings"][1][2]),
            (model.blocks[0].ff2, 1, 2, grads["blocks"][0]["ff2"][1][2]),
            (model.final_norm_scale, 0, None, grads["final_norm_scale"][0]),
        ]
        for tensor, row, column, analytic in checks:
            if column is None:
                original = tensor[row]
                tensor[row] = original + epsilon
                plus = objective()
                tensor[row] = original - epsilon
                minus = objective()
                tensor[row] = original
            else:
                original = tensor[row][column]
                tensor[row][column] = original + epsilon
                plus = objective()
                tensor[row][column] = original - epsilon
                minus = objective()
                tensor[row][column] = original
            numeric = (plus - minus) / (2 * epsilon)
            self.assertAlmostEqual(analytic, numeric, places=4)

    def test_train_batch_updates_model_parameters(self):
        from memer.models.byte_tokenizer import ByteTokenizer
        from memer.models.data import encode_documents, make_batches
        from memer.models.training import Adam, train_batch

        model = self.make_model()
        docs = encode_documents("hi", ByteTokenizer())
        batch = next(make_batches(docs, block_size=4, batch_size=1))
        before = model.token_embeddings[ord("h")][0]
        loss = train_batch(model, batch, Adam(learning_rate=1e-3))
        self.assertTrue(math.isfinite(loss))
        self.assertNotEqual(model.token_embeddings[ord("h")][0], before)

    def test_adam_reduces_simple_scalar_objective(self):
        from memer.models.training import Adam

        parameters = [[2.0, -1.0]]
        optimizer = Adam(learning_rate=0.05, gradient_clip_norm=None)
        for _ in range(20):
            gradients = [[2.0 * parameters[0][0], 2.0 * parameters[0][1]]]
            optimizer.step(parameters, gradients)
        self.assertLess(abs(parameters[0][0]), 2.0)
        self.assertLess(abs(parameters[0][1]), 1.0)
        self.assertEqual(optimizer.step_count, 20)

    def test_generation_is_seeded_and_returns_prompt(self):
        model = self.make_model()
        first = model.generate("A", max_new_tokens=4, seed=3)
        second = model.generate("A", max_new_tokens=4, seed=3)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("A"))
