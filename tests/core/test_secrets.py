import keyring
import pytest
from keyring.backend import KeyringBackend
from keyring.errors import KeyringError, PasswordDeleteError

from vector_embed.core import secrets
from vector_embed.core.secrets import KeyringStore, KeyStoreError, MemoryKeyStore, scrub


class InMemoryBackend(KeyringBackend):
    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.data: dict[tuple[str, str], str] = {}
        self.broken = False

    def _check(self) -> None:
        if self.broken:
            raise KeyringError("vault locked")

    def get_password(self, service: str, username: str) -> str | None:
        self._check()
        return self.data.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._check()
        self.data[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self._check()
        if (service, username) not in self.data:
            raise PasswordDeleteError("not found")
        del self.data[(service, username)]


@pytest.fixture
def backend() -> InMemoryBackend:
    previous = keyring.get_keyring()
    backend = InMemoryBackend()
    keyring.set_keyring(backend)
    yield backend  # type: ignore[misc]
    keyring.set_keyring(previous)


def test_keys_roundtrip_through_the_credential_store(backend: InMemoryBackend) -> None:
    store = KeyringStore()
    assert store.get("openrouter") is None
    store.set("openrouter", "  sk-abc  ")
    assert store.get("openrouter") == "sk-abc"
    assert backend.data == {(secrets.SERVICE_NAME, "openrouter"): "sk-abc"}
    store.delete("openrouter")
    assert store.get("openrouter") is None
    store.delete("openrouter")  # deleting a missing key is fine


def test_empty_keys_are_refused(backend: InMemoryBackend) -> None:
    with pytest.raises(KeyStoreError, match="empty"):
        KeyringStore().set("p", "   ")
    with pytest.raises(KeyStoreError, match="empty"):
        MemoryKeyStore().set("p", "")


def test_vault_failures_become_key_store_errors(backend: InMemoryBackend) -> None:
    backend.broken = True
    store = KeyringStore()
    with pytest.raises(KeyStoreError, match="cannot read"):
        store.get("p")
    with pytest.raises(KeyStoreError, match="cannot store"):
        store.set("p", "k")
    with pytest.raises(KeyStoreError, match="cannot delete"):
        store.delete("p")


def test_memory_store() -> None:
    store = MemoryKeyStore({"a": "1"})
    store.set("b", " 2 ")
    assert (store.get("a"), store.get("b"), store.get("c")) == ("1", "2", None)
    store.delete("a")
    store.delete("zzz")
    assert store.get("a") is None


def test_scrub_removes_the_key_from_text() -> None:
    assert scrub("auth failed for sk-123", "sk-123") == "auth failed for [API KEY]"
    assert scrub("nothing here", "sk-123") == "nothing here"
    assert scrub("text", None) == "text"
