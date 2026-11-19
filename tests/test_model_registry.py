"""Tests for the local Memer model registry."""

import json
import tempfile
import unittest
from pathlib import Path

from memer.models.registry import (
    ModelRegistryError, add_model, get_active_model, get_model,
    list_models, remove_model, use_model,
)
from memer.models.transformer import MemerTransformer


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry = self.root / "registry" / "models.json"
        self.checkpoint = self.root / "model.json"
        MemerTransformer(
            dimension=4, heads=1, layers=1, feed_forward_dim=4, block_size=4
        ).save(self.checkpoint)

    def tearDown(self):
        self.temp.cleanup()

    def test_add_model_sets_first_model_active_and_saves_connection_metadata(self):
        item = add_model(
            self.checkpoint, name="memer-small", version="0.4.0",
            base_url="https://memer.example/v1", api_platform="Memer",
            api_key_env="MEMER_API_KEYS", path=self.registry,
        )
        self.assertTrue(item["active"])
        self.assertEqual(item["base_url"], "https://memer.example/v1")
        self.assertEqual(item["api_platform"], "Memer")
        self.assertEqual(get_active_model(self.registry)["name"], "memer-small")

    def test_add_rejects_duplicate_name(self):
        add_model(self.checkpoint, name="memer-small", path=self.registry)
        with self.assertRaises(ModelRegistryError):
            add_model(self.checkpoint, name="memer-small", path=self.registry)

    def test_add_rejects_invalid_base_url(self):
        with self.assertRaises(ModelRegistryError):
            add_model(self.checkpoint, name="memer-small",
                      base_url="javascript:alert(1)", path=self.registry)

    def test_use_switches_active_model(self):
        add_model(self.checkpoint, name="model-a", path=self.registry)
        add_model(self.checkpoint, name="model-b", path=self.registry)
        selected = use_model("model-b", path=self.registry)
        self.assertTrue(selected["active"])
        self.assertFalse(get_model("model-a", self.registry)["active"])

    def test_remove_keeps_checkpoint_file_and_selects_another_model(self):
        add_model(self.checkpoint, name="model-a", path=self.registry)
        add_model(self.checkpoint, name="model-b", path=self.registry)
        remove_model("model-a", path=self.registry)
        self.assertTrue(self.checkpoint.is_file())
        self.assertEqual(get_active_model(self.registry)["name"], "model-b")

    def test_remove_last_model_clears_active_selection(self):
        add_model(self.checkpoint, name="model-a", path=self.registry)
        remove_model("model-a", path=self.registry)
        self.assertIsNone(get_active_model(self.registry))
        self.assertEqual(list_models(self.registry), [])

    def test_corrupt_registry_is_rejected(self):
        self.registry.parent.mkdir(parents=True)
        self.registry.write_text(json.dumps({"format": "wrong"}), encoding="utf-8")
        with self.assertRaises(ModelRegistryError):
            list_models(self.registry)


if __name__ == "__main__":
    unittest.main()
