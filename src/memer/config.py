"""Environment-based settings for Memer Core."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    provider: str = "mock"
    model_name: str = "memer-dev-mock"
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        port_value = os.getenv("MEMER_PORT", "8000")
        try:
            port = int(port_value)
        except ValueError as exc:
            raise ValueError("MEMER_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("MEMER_PORT must be between 1 and 65535")

        return cls(
            provider=os.getenv("MEMER_PROVIDER", "mock").strip().lower(),
            model_name=os.getenv("MEMER_MODEL_NAME", "memer-dev-mock").strip(),
            host=os.getenv("MEMER_HOST", "127.0.0.1").strip(),
            port=port,
        )
