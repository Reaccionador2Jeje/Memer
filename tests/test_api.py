import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from memer.api import app
from memer.providers.mock import MockProvider


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


if __name__ == "__main__":
    unittest.main()
