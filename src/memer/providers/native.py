"""Native Memer Transformer provider for local checkpoint inference."""

from pathlib import Path
from typing import Sequence

from memer.models.byte_tokenizer import ByteTokenizer
from memer.models.transformer import MemerTransformer
from memer.schemas import ChatMessage


class NativeMemerProvider:
    """Run the experimental native Memer Transformer from a saved checkpoint."""

    name = "native"

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | Path,
        *,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        seed: int = 17,
    ) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Memer checkpoint does not exist: {path}")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be an integer between 1 and 4096")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")

        self.model_name = model_name
        self.checkpoint_path = path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self.tokenizer = ByteTokenizer()
        self.model = MemerTransformer.load(path)

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        role_names = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
        }
        prompt_parts = [
            f"{role_names[message.role]}: {message.content}"
            for message in messages
        ]
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        return self.model.generate(
            prompt,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            seed=self.seed,
        )
