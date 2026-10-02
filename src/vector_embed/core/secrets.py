"""API keys: Windows Credential Manager (via ``keyring``), never config, logs or the index."""

import logging
from typing import Protocol

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

logger = logging.getLogger(__name__)

SERVICE_NAME = "VectorEmbed"
_MASK = "[API KEY]"


class KeyStoreError(RuntimeError):
    """The credential store could not be read or written."""


class KeyStore(Protocol):
    def get(self, provider: str) -> str | None: ...

    def set(self, provider: str, key: str) -> None: ...

    def delete(self, provider: str) -> None: ...


class KeyringStore:
    """The OS credential vault. On Windows this is Credential Manager."""

    def get(self, provider: str) -> str | None:
        try:
            return keyring.get_password(SERVICE_NAME, provider)
        except KeyringError as exc:
            raise KeyStoreError(f"cannot read the API key for {provider}: {exc}") from exc

    def set(self, provider: str, key: str) -> None:
        if not key.strip():
            raise KeyStoreError("refusing to store an empty API key")
        try:
            keyring.set_password(SERVICE_NAME, provider, key.strip())
        except KeyringError as exc:
            raise KeyStoreError(f"cannot store the API key for {provider}: {exc}") from exc

    def delete(self, provider: str) -> None:
        try:
            keyring.delete_password(SERVICE_NAME, provider)
        except PasswordDeleteError:
            logger.debug("no stored key to delete for %s", provider)
        except KeyringError as exc:
            raise KeyStoreError(f"cannot delete the API key for {provider}: {exc}") from exc


class MemoryKeyStore:
    """In-memory store for tests and for keys supplied by the environment."""

    def __init__(self, keys: dict[str, str] | None = None) -> None:
        self._keys = dict(keys or {})

    def get(self, provider: str) -> str | None:
        return self._keys.get(provider)

    def set(self, provider: str, key: str) -> None:
        if not key.strip():
            raise KeyStoreError("refusing to store an empty API key")
        self._keys[provider] = key.strip()

    def delete(self, provider: str) -> None:
        self._keys.pop(provider, None)


def scrub(text: str, key: str | None) -> str:
    """Remove ``key`` from ``text`` (used on any message that could end up in a log)."""
    if key and key in text:
        return text.replace(key, _MASK)
    return text
