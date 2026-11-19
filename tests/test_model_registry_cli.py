"""CLI tests for Memer model registry commands."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from memer.cli import main
from memer.models.transformer import MemerTransformer


class ModelRegistryCliTests(unittest.TestCase):
    def test_add_list_and_inspect_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.json"
            registry = root / "models.json"
            MemerTransformer(dimension=4, heads=1, layers=1,
                             feed_forward_dim=4, block_size=4).save(checkpoint)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["models", "add", str(checkpoint), "--name", "memer-test",
                             "--base-url", "https://memer.example/v1",
                             "--api-platform", "Memer", "--registry", str(registry)])
            self.assertEqual(code, 0)
            self.assertIn("Registered: memer-test", output.getvalue())
            self.assertIn("API key secret: not stored", output.getvalue())

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["models", "list", "--registry", str(registry)])
            self.assertEqual(code, 0)
            self.assertIn("https://memer.example/v1", output.getvalue())

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["models", "inspect", "memer-test", "--registry", str(registry)])
            self.assertEqual(code, 0)
            self.assertIn("API key platform: Memer", output.getvalue())

    def test_use_and_remove_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.json"
            registry = root / "models.json"
            MemerTransformer(dimension=4, heads=1, layers=1,
                             feed_forward_dim=4, block_size=4).save(checkpoint)
            main(["models", "add", str(checkpoint), "--name", "memer-test",
                  "--registry", str(registry)])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["models", "use", "memer-test", "--registry", str(registry)])
            self.assertEqual(code, 0)
            self.assertIn("Active model: memer-test", output.getvalue())

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["models", "remove", "memer-test", "--registry", str(registry)])
            self.assertEqual(code, 0)
            self.assertTrue(checkpoint.is_file())


if __name__ == "__main__":
    unittest.main()
