import unittest
from dataclasses import replace
from unittest.mock import patch

from fastapi.testclient import TestClient

from memer.api import app, settings as default_settings
from memer.providers.mock import MockProvider
from memer.security import RuntimeSecurity


class MemerApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.provider_patch = patch("memer.api.provider", MockProvider())
        self.provider_patch.start()
        self.addCleanup(self.provider_patch.stop)

    def test_health_reports_active_provider(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider"], "mock")

    def test_chat_returns_completion_contract(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi Memer"}]},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["provider"], "mock")
        self.assertIn("Hi Memer", body["content"])

    def test_rejects_unconfigured_model_switch(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "some-other-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_empty_message_content(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": ""}]},
        )
        self.assertEqual(response.status_code, 422)

    def test_protected_runtime_rejects_missing_api_key(self):
        guard = RuntimeSecurity(("k" * 40,), require_api_key=True)
        with patch("memer.api.security", guard):
            response = self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )
        self.assertEqual(response.status_code, 401)

    def test_protected_runtime_accepts_valid_api_key(self):
        guard = RuntimeSecurity(("k" * 40,), require_api_key=True)
        with patch("memer.api.security", guard):
            response = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer " + "k" * 40},
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )
        self.assertEqual(response.status_code, 200)

    def test_rejects_oversized_prompt(self):
        limits = replace(default_settings, max_input_chars=5)
        with patch("memer.api.settings", limits):
            response = self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "This is too long"}]},
            )
        self.assertEqual(response.status_code, 413)

    def test_rejects_generation_when_runtime_is_at_capacity(self):
        guard = RuntimeSecurity(max_concurrent_generations=1)
        self.assertTrue(guard.acquire_generation_slot())
        try:
            with patch("memer.api.security", guard):
                response = self.client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "Hi"}]},
                )
        finally:
            guard.release_generation_slot()
        self.assertEqual(response.status_code, 429)


if __name__ == "__main__":
    unittest.main()
