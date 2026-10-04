import argparse
import tomllib
from pathlib import Path

import pytest
from tests.core.conftest import Env

from localdoc_finder import cli, cli_cloud
from localdoc_finder.core.secrets import MemoryKeyStore
from localdoc_finder.core.settings import Settings, load_settings


@pytest.fixture(autouse=True)
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cli_cloud.add_cloud_parsers(sub)
    return parser.parse_args(argv)


class Out(list):  # type: ignore[type-arg]
    def __call__(self, text: str = "") -> None:
        self.append(text)


def settings_file(env: Env) -> dict[str, object]:
    with (env.data_dir / "settings.toml").open("rb") as handle:
        return tomllib.load(handle)


def reload(env: Env) -> Settings:
    """Settings as the file now stands, still pointing at the test data directory."""
    loaded = load_settings(env.data_dir / "settings.toml")
    return loaded.model_copy(update={"storage": env.settings.storage})


class TestKeys:
    def test_set_status_and_delete_never_print_the_key(self, env: Env) -> None:
        store = MemoryKeyStore()
        out = Out()
        args = parse("keys", "set", "openrouter")
        cli_cloud.run_keys(
            args, env.settings, store, out, read_secret=lambda _p: "sk-secret-value-123"
        )
        assert store.get("openrouter") == "sk-secret-value-123"
        assert out == ["stored the key for openrouter"]
        assert "sk-secret" not in "".join(out)
        out.clear()
        cli_cloud.run_keys(parse("keys", "delete", "openrouter"), env.settings, store, out)
        assert store.get("openrouter") is None

    def test_key_can_come_from_stdin(self, env: Env) -> None:
        store = MemoryKeyStore()
        cli_cloud.run_keys(
            parse("keys", "set", "p", "--stdin"),
            env.settings,
            store,
            Out(),
            read_stdin=lambda: "sk-from-stdin",
        )
        assert store.get("p") == "sk-from-stdin"

    def test_status_lists_providers_and_whether_a_key_exists(self, env: Env) -> None:
        out = Out()
        cli_cloud.run_keys(parse("keys", "status"), env.settings, MemoryKeyStore(), out)
        assert out == ["no cloud providers configured (see: ldf cloud add)"]
        cli_cloud.run_cloud(
            parse("cloud", "add", "openrouter", "--base-url", "https://x/v1", "--use"),
            env.settings,
            MemoryKeyStore(),
            Out(),
        )
        configured = reload(env)
        out.clear()
        cli_cloud.run_keys(
            parse("keys", "status"), configured, MemoryKeyStore({"openrouter": "k"}), out
        )
        assert out == ["openrouter (active): key stored"]
        out.clear()
        cli_cloud.run_keys(parse("keys", "status"), configured, MemoryKeyStore(), out)
        assert out == ["openrouter (active): NO KEY"]


class TestCloud:
    def test_add_writes_provider_models_and_activation(self, env: Env) -> None:
        out = Out()
        cli_cloud.run_cloud(
            parse(
                "cloud", "add", "openrouter", "--base-url", "https://openrouter.ai/api/v1",
                "--label", "OpenRouter", "--model", "chat=vendor/chat",
                "--model", "match_scorer=vendor/judge",
                "--use",
            ),
            env.settings,
            MemoryKeyStore(),
            out,
        )  # fmt: skip
        assert "ldf keys set openrouter" in out[0]
        cloud = reload(env).cloud
        provider = cloud.providers["openrouter"]
        assert provider.base_url == "https://openrouter.ai/api/v1"
        assert provider.label == "OpenRouter"
        assert provider.models == {"chat": "vendor/chat", "match_scorer": "vendor/judge"}
        assert cloud.active == "openrouter"

    def test_bad_roles_and_assignments_are_rejected(self, env: Env) -> None:
        store = MemoryKeyStore()
        with pytest.raises(cli_cloud.CloudCommandError, match="unknown role"):
            cli_cloud.run_cloud(
                parse("cloud", "add", "p", "--base-url", "u", "--model", "poetry=x"),
                env.settings,
                store,
                Out(),
            )
        with pytest.raises(cli_cloud.CloudCommandError, match="expected ROLE=VALUE"):
            cli_cloud.run_cloud(
                parse("cloud", "add", "p", "--base-url", "u", "--model", "chat"),
                env.settings,
                store,
                Out(),
            )
        with pytest.raises(cli_cloud.CloudCommandError, match="unknown role"):
            cli_cloud.run_cloud(parse("cloud", "route", "poetry=cloud"), env.settings, store, Out())
        with pytest.raises(cli_cloud.CloudCommandError, match="policy must be"):
            cli_cloud.run_cloud(
                parse("cloud", "route", "chat=sometimes"), env.settings, store, Out()
            )

    def test_route_sets_policies(self, env: Env) -> None:
        out = Out()
        cli_cloud.run_cloud(
            parse("cloud", "route", "chat=auto", "match_scorer=cloud"),
            env.settings,
            MemoryKeyStore(),
            out,
        )
        assert out == ["chat -> auto", "match_scorer -> cloud"]
        assert reload(env).cloud.routing == {"chat": "auto", "match_scorer": "cloud"}

    def test_budget_set_and_cleared(self, env: Env) -> None:
        out = Out()
        store = MemoryKeyStore()
        cli_cloud.run_cloud(parse("cloud", "budget", "$5"), env.settings, store, out)
        assert reload(env).cloud.monthly_budget_usd == 5.0
        assert out[-1] == "monthly budget set to $5.00"
        cli_cloud.run_cloud(parse("cloud", "budget", "off"), env.settings, store, out)
        assert reload(env).cloud.monthly_budget_usd is None
        with pytest.raises(cli_cloud.CloudCommandError, match="number or 'off'"):
            cli_cloud.run_cloud(parse("cloud", "budget", "lots"), env.settings, store, out)

    def test_status_reports_providers_routing_and_spend(self, env: Env) -> None:
        store = MemoryKeyStore({"openrouter": "k"})
        cli_cloud.run_cloud(
            parse(
                "cloud", "add", "openrouter", "--base-url", "https://x/v1",
                "--model", "chat=m", "--use",
            ),
            env.settings, store, Out(),
        )  # fmt: skip
        cli_cloud.run_cloud(parse("cloud", "route", "chat=auto"), env.settings, store, Out())
        cli_cloud.run_cloud(parse("cloud", "budget", "10"), env.settings, store, Out())
        env.state.record_usage("openrouter", "m", 10, 10, 0.5)
        out = Out()
        cli_cloud.run_cloud(parse("cloud", "status"), reload(env), store, out)
        text = "\n".join(out)
        assert "active provider : openrouter" in text
        assert "openrouter: https://x/v1 [key stored] chat=m" in text
        assert "routing         : chat=auto" in text
        assert "spend this month: $0.5000 of $10.00" in text

    def test_status_with_nothing_configured(self, env: Env) -> None:
        out = Out()
        cli_cloud.run_cloud(parse("cloud", "status"), env.settings, MemoryKeyStore(), out)
        assert "everything stays local" in out[0]
        assert "routing         : all local" in out


class TestThroughTheMainCommand:
    def test_ve_cloud_and_keys_are_wired_with_the_credential_store(
        self, env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = MemoryKeyStore()
        monkeypatch.setattr(cli, "load_settings", lambda: env.settings)
        monkeypatch.setattr(cli, "KeyringStore", lambda: store)
        assert cli.main(["cloud", "add", "p", "--base-url", "https://x/v1"]) == 0
        assert cli.main(["cloud", "route", "chat=wrong"]) == 2
        assert "policy must be" in capsys.readouterr().err
        assert cli.main(["keys", "status"]) == 0

    def test_a_credential_store_failure_is_an_error_exit(
        self, env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from localdoc_finder.core.secrets import KeyStoreError

        class Broken:
            def get(self, provider: str) -> str | None:
                raise KeyStoreError("vault locked")

            def set(self, provider: str, key: str) -> None:
                raise KeyStoreError("vault locked")

            def delete(self, provider: str) -> None:
                raise KeyStoreError("vault locked")

        monkeypatch.setattr(cli, "load_settings", lambda: env.settings)
        monkeypatch.setattr(cli, "KeyringStore", Broken)
        assert cli.main(["keys", "delete", "p"]) == 1
        assert "vault locked" in capsys.readouterr().err
