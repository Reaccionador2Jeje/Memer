import tempfile
import unittest
from pathlib import Path

from memer.models.transformer import MemerTransformer
from memer.providers.native import NativeMemerProvider
from memer.schemas import ChatMessage


class NativeMemerProviderTests(unittest.TestCase):
    def make_checkpoint(self, directory: str) -> Path:
        model = MemerTransformer(
            dimension=8,
            heads=2,
            layers=1,
            feed_forward_dim=16,
            block_size=8,
            seed=5,
        )
        path = Path(directory) / "native-checkpoint.json"
        model.save(path)
        return path

    def test_loads_checkpoint_and_generates_text(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(directory)
            provider = NativeMemerProvider(
                "memer-test",
                checkpoint,
                max_new_tokens=5,
                temperature=0.8,
                seed=7,
            )
            result = provider.generate([
                ChatMessage(role="user", content="Hello Memer")
            ])
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            self.assertEqual(provider.name, "native")

    def test_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with self.assertRaisesRegex(ValueError, "checkpoint does not exist"):
                NativeMemerProvider("memer-test", missing)

    def test_rejects_invalid_generation_options(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(directory)
            with self.assertRaises(ValueError):
                NativeMemerProvider("memer-test", checkpoint, max_new_tokens=0)


if __name__ == "__main__":
    unittest.main()
