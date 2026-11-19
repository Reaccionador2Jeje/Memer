"""Tests for local checkpoint management."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memer.models.checkpoints import (
    CheckpointManagerError,
    inspect_checkpoint,
    list_checkpoints,
    validate_checkpoint,
)
from memer.models.transformer import MemerTransformer


class CheckpointManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "memer-small.json"
        MemerTransformer(dimension=4, heads=1, layers=1, feed_forward_dim=4,
                         block_size=4).save(self.checkpoint)

    def tearDown(self):
        self.temp.cleanup()

    def test_inspect_returns_architecture_and_parameter_count(self):
        info = inspect_checkpoint(self.checkpoint)
        self.assertEqual(info["format"], "MEMER-TRANSFORMER")
        self.assertEqual(info["config"]["dimension"], 4)
        model = MemerTransformer(dimension=4, heads=1, layers=1, feed_forward_dim=4,
                                 block_size=4)
        self.assertEqual(info["parameters"], model.parameter_count())

    def test_validate_checkpoint_returns_metadata(self):
        self.assertEqual(validate_checkpoint(self.checkpoint)["format_version"], 1)

    def test_rejects_missing_checkpoint(self):
        with self.assertRaises(CheckpointManagerError):
            inspect_checkpoint(self.root / "missing.json")

    def test_rejects_invalid_checkpoint(self):
        bad = self.root / "bad.json"
        bad.write_text("{broken", encoding="utf-8")
        with self.assertRaises(CheckpointManagerError):
            validate_checkpoint(bad)

    def test_rejects_oversized_checkpoint_before_loading(self):
        bad = self.root / "large.json"
        bad.write_bytes(b"x" * 17)
        with patch("memer.models.checkpoints.MAX_CHECKPOINT_BYTES", 16):
            with self.assertRaises(CheckpointManagerError):
                inspect_checkpoint(bad)

    def test_list_checkpoints_marks_invalid_files(self):
        bad = self.root / "bad.json"
        bad.write_text(json.dumps({"not": "a checkpoint"}), encoding="utf-8")
        results = list_checkpoints(self.root)
        self.assertEqual(len(results), 2)
        by_name = {Path(item["path"]).name: item for item in results}
        self.assertTrue(by_name["memer-small.json"]["valid"])
        self.assertFalse(by_name["bad.json"]["valid"])

    def test_missing_directory_returns_empty_list(self):
        self.assertEqual(list_checkpoints(self.root / "missing"), [])

    def test_rejects_file_as_directory(self):
        with self.assertRaises(CheckpointManagerError):
            list_checkpoints(self.checkpoint)


if __name__ == "__main__":
    unittest.main()
