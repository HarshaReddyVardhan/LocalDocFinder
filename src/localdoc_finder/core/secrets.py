"""API keys: Windows Credential Manager (via ``keyring``), never config, logs or the index."""

import logging
from typing import Protocol

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

logger = logging.getLogger(__name__)

SERVICE_NAME = "LocalDocFinder"
OLD_SERVICE_NAME = "VectorEmbed"  # keys saved before the rename move over when first read
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
            key = keyring.get_password(SERVICE_NAME, provider)
            if key is None:
                key = self._move_old_key(provider)
        except KeyringError as exc:
            raise KeyStoreError(f"cannot read the API key for {provider}: {exc}") from exc
        return key

    @staticmethod
    def _move_old_key(provider: str) -> str | None:
        """A key stored under the old app name, moved to the new one."""
        key = keyring.get_password(OLD_SERVICE_NAME, provider)
        if key is not None:
            keyring.set_password(SERVICE_NAME, provider, key)
            keyring.delete_password(OLD_SERVICE_NAME, provider)
        return key

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
