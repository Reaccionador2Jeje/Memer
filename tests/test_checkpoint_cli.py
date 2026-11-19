"""CLI tests for checkpoint management commands."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from memer.cli import main
from memer.models.transformer import MemerTransformer


class CheckpointCliTests(unittest.TestCase):
    def test_checkpoint_list_displays_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            MemerTransformer(dimension=4, heads=1, layers=1,
                             feed_forward_dim=4, block_size=4).save(path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["checkpoint", "list", directory])
            self.assertEqual(code, 0)
            self.assertIn("valid", output.getvalue())
            self.assertIn("model.json", output.getvalue())

    def test_checkpoint_inspect_displays_model_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            MemerTransformer(dimension=4, heads=1, layers=1,
                             feed_forward_dim=4, block_size=4).save(path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["checkpoint", "inspect", str(path)])
            self.assertEqual(code, 0)
            self.assertIn("dimension: 4", output.getvalue())

    def test_checkpoint_validate_confirms_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            MemerTransformer(dimension=4, heads=1, layers=1,
                             feed_forward_dim=4, block_size=4).save(path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["checkpoint", "validate", str(path)])
            self.assertEqual(code, 0)
            self.assertIn("Validation: OK", output.getvalue())


if __name__ == "__main__":
    unittest.main()
