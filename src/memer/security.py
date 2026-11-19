"""Authentication and bounded concurrency for hosted Memer inference."""

from hmac import compare_digest
from threading import BoundedSemaphore


class RuntimeSecurity:
    """Small single-process guard for a single Memer runtime instance."""

    def __init__(
        self,
        api_keys: tuple[str, ...] = (),
        *,
        require_api_key: bool = False,
        max_concurrent_generations: int = 1,
    ) -> None:
        if not 1 <= max_concurrent_generations <= 8:
            raise ValueError("max_concurrent_generations must be between 1 and 8")
        self.api_keys = tuple(key for key in api_keys if key)
        self.require_api_key = require_api_key
        self._generation_slots = BoundedSemaphore(max_concurrent_generations)

    def is_authorized(self, authorization: str | None) -> bool:
        if not self.require_api_key:
            return True
        if not authorization:
            return False
        scheme, separator, supplied_key = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer" or not supplied_key:
            return False
        return any(compare_digest(supplied_key, key) for key in self.api_keys)

    def acquire_generation_slot(self) -> bool:
        """Return immediately; callers should reject requests when capacity is full."""
        return self._generation_slots.acquire(blocking=False)

    def release_generation_slot(self) -> None:
        self._generation_slots.release()
