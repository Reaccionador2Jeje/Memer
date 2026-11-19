"""Interface implemented by every Memer model provider."""

from typing import Protocol, Sequence
from memer.schemas import ChatMessage


class ModelProvider(Protocol):
    name: str
    model_name: str

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        """Generate a response from an ordered conversation."""
        ...
