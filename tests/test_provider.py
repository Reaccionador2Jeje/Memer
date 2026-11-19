import unittest

from memer.providers.mock import MockProvider
from memer.schemas import ChatMessage


class MockProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = MockProvider()

    def test_returns_clear_placeholder_response(self):
        messages = [ChatMessage(role="user", content="Hello Memer")]
        result = self.provider.generate(messages)
        self.assertIn("development mode", result)
        self.assertIn("Hello Memer", result)
        self.assertIn("not AI-generated", result)

    def test_handles_conversation_and_uses_latest_user_message(self):
        messages = [
            ChatMessage(role="user", content="First message"),
            ChatMessage(role="assistant", content="Placeholder"),
            ChatMessage(role="user", content="Latest message"),
        ]
        result = self.provider.generate(messages)
        self.assertIn("Latest message", result)
        self.assertNotIn("First message", result)

    def test_handles_no_user_message(self):
        messages = [ChatMessage(role="system", content="Be helpful")]
        result = self.provider.generate(messages)
        self.assertIn("needs a user message", result)


if __name__ == "__main__":
    unittest.main()
