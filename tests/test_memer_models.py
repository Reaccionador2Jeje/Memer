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


if __name__ == "__main__":
    unittest.main()
