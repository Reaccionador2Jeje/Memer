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
        documents = encode_documents("ab\nxy", tokenizer)
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


if __name__ == "__main__":
    unittest.main()
