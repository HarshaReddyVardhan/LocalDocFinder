import pytest

from vector_embed.core.registry import Registry, RegistryError


def test_register_and_get() -> None:
    reg: Registry[type] = Registry("thing")

    @reg.register("a")
    class A:
        pass

    assert reg.get("a") is A
    assert "a" in reg
    assert reg.names() == ["a"]
    assert reg.items() == [("a", A)]
    assert list(reg) == [A]
    assert len(reg) == 1


def test_duplicate_requires_replace() -> None:
    reg: Registry[int] = Registry("thing")
    reg.add("x", 1)
    with pytest.raises(RegistryError, match="already registered"):
        reg.add("x", 2)
    reg.add("x", 3, replace=True)
    assert reg.get("x") == 3


def test_unknown_name_lists_known() -> None:
    reg: Registry[int] = Registry("thing")
    with pytest.raises(RegistryError, match="none"):
        reg.get("nope")
    reg.add("a", 1)
    with pytest.raises(RegistryError, match="known: a"):
        reg.get("nope")


def test_remove_is_idempotent() -> None:
    reg: Registry[int] = Registry("thing")
    reg.add("a", 1)
    reg.remove("a")
    reg.remove("a")
    assert "a" not in reg
