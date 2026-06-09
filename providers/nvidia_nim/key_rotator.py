"""Round-robin API key rotator for NVIDIA NIM provider."""

import itertools
import threading
from collections.abc import Sequence


class NimKeyRotator:
    """Thread-safe round-robin rotator for a list of NVIDIA NIM API keys.

    When only a single key is configured it behaves identically to a plain
    string credential — ``next_key()`` always returns that key.  When multiple
    keys are provided each call to ``next_key()`` advances to the next key in
    the list, cycling back to the first after the last.

    Usage::

        rotator = NimKeyRotator(["key1", "key2", "key3"])
        key = rotator.next_key()   # "key1"
        key = rotator.next_key()   # "key2"
        key = rotator.next_key()   # "key3"
        key = rotator.next_key()   # "key1"  (wraps around)
    """

    def __init__(self, keys: Sequence[str]) -> None:
        if not keys:
            raise ValueError("NimKeyRotator requires at least one API key.")
        self._keys = list(keys)
        self._cycle = itertools.cycle(self._keys)
        self._lock = threading.Lock()

    @property
    def key_count(self) -> int:
        """Number of keys in the rotation pool."""
        return len(self._keys)

    def next_key(self) -> str:
        """Return the next API key in round-robin order (thread-safe)."""
        with self._lock:
            return next(self._cycle)

    @classmethod
    def from_env_string(cls, env_value: str) -> "NimKeyRotator":
        """Parse a comma-separated env string into a rotator.

        Whitespace around individual keys is stripped; empty segments are
        discarded so trailing commas and accidental double-commas are safe.

        Example env value::

            NVIDIA_NIM_API_KEYS="nvapi-key1, nvapi-key2, nvapi-key3"
        """
        keys = [k.strip() for k in env_value.split(",") if k.strip()]
        if not keys:
            raise ValueError(
                "NVIDIA_NIM_API_KEYS is set but contains no valid keys. "
                "Provide at least one non-empty key."
            )
        return cls(keys)

    @classmethod
    def from_single_key(cls, key: str) -> "NimKeyRotator":
        """Wrap a single API key string in a rotator."""
        return cls([key])
