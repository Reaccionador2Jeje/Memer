import io
import json
import unittest
from unittest.mock import patch

from memer.providers.ollama import OllamaProvider
from memer.schemas import ChatMessage


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class OllamaProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = OllamaProvider(
            model_name="tiny-test-model",
            base_url="http://localhost:11434/",
            timeout=10,
        )

    @patch("memer.providers.ollama.urlopen")
    def test_sends_chat_and_returns_assistant_content(self, mocked_urlopen):
        mocked_urlopen.return_value = FakeResponse(
            {"message": {"role": "assistant", "content": "Hello from a local model"}}
        )
        messages = [
            ChatMessage(role="system", content="Be concise"),
            ChatMessage(role="user", content="Hello"),
        ]

        result = self.provider.generate(messages)

        self.assertEqual(result, "Hello from a local model")
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "tiny-test-model")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][1]["content"], "Hello")
        self.assertEqual(mocked_urlopen.call_args.kwargs["timeout"], 10)

    @patch("memer.providers.ollama.urlopen")
    def test_rejects_empty_model_response(self, mocked_urlopen):
        mocked_urlopen.return_value = FakeResponse({"message": {"content": ""}})
        with self.assertRaisesRegex(RuntimeError, "did not contain message content"):
            self.provider.generate([ChatMessage(role="user", content="Hi")])

    def test_rejects_invalid_url_scheme(self):
        with self.assertRaisesRegex(ValueError, "HTTP or HTTPS"):
            OllamaProvider("model", "file:///tmp/ollama")

    @patch("memer.providers.ollama.urlopen", side_effect=OSError("connection refused"))
    def test_reports_connection_problem(self, mocked_urlopen):
        with self.assertRaisesRegex(RuntimeError, "Could not reach Ollama"):
            self.provider.generate([ChatMessage(role="user", content="Hi")])


if __name__ == "__main__":
    unittest.main()
