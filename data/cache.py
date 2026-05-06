import time
from typing import Any, Optional


class TTLCache:
    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str, ttl: float) -> Optional[Any]:
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if time.time() - ts > ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any):
        self._store[key] = (value, time.time())


cache = TTLCache()
