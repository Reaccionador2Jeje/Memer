"""Development-only provider. This is not a language model."""

from typing import Sequence
from memer.schemas import ChatMessage


class MockProvider:
    name = "mock"

    def __init__(self, model_name: str = "memer-dev-mock") -> None:
        self.model_name = model_name

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        last_user_message = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            None,
        )
        if last_user_message is None:
            return (
                "Memer Core is running, but this development provider needs a user "
                "message. A real model provider has not been connected yet."
            )

        return (
            "[Memer development mode] The API received your message: "
            f"{last_user_message}\n\n"
            "This is a placeholder response, not AI-generated reasoning. "
            "Connect a model runtime to enable real inference."
        )
