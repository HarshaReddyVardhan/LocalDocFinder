"""``ve keys`` and ``ve cloud``: configure cloud providers without editing files by hand.

Settings go to ``settings.toml``; API keys go to Windows Credential Manager and are never
echoed, logged or written anywhere else.
"""

import argparse
import getpass
from collections.abc import Callable
from pathlib import Path

from vector_embed.core.health import month_start
from vector_embed.core.models.catalog import ROLES
from vector_embed.core.secrets import KeyStore
from vector_embed.core.settings import Settings
from vector_embed.core.settings_io import set_setting
from vector_embed.core.store.sqlite import StateDb

POLICIES = ("local", "cloud", "auto")
Printer = Callable[[str], None]


class CloudCommandError(RuntimeError):
    """Bad arguments to a cloud command."""


def add_cloud_parsers(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    keys = sub.add_parser("keys", help="store API keys in Windows Credential Manager")
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)
    set_cmd = keys_sub.add_parser("set", help="store a key (prompted; never shown)")
    set_cmd.add_argument("provider")
    set_cmd.add_argument("--stdin", action="store_true", help="read the key from standard input")
    keys_sub.add_parser("status", help="which providers have a stored key")
    delete = keys_sub.add_parser("delete", help="remove a stored key")
    delete.add_argument("provider")

    cloud = sub.add_parser("cloud", help="configure cloud providers and routing")
    cloud_sub = cloud.add_subparsers(dest="cloud_command", required=True)
    add = cloud_sub.add_parser("add", help="add or update an OpenAI-compatible provider")
    add.add_argument("name")
    add.add_argument("--base-url", required=True, help="e.g. https://openrouter.ai/api/v1")
    add.add_argument("--label", help="display name")
    add.add_argument("--model", action="append", metavar="ROLE=MODEL", help="model for a role")
    add.add_argument("--use", action="store_true", help="make it the active provider")
    route = cloud_sub.add_parser("route", help="set the routing policy for roles")
    route.add_argument("assignment", nargs="+", metavar="ROLE=local|cloud|auto")
    budget = cloud_sub.add_parser("budget", help="monthly spend limit in USD (or 'off')")
    budget.add_argument("amount")
    cloud_sub.add_parser("status", help="providers, routing, budget and spend this month")


def _split(assignment: str) -> tuple[str, str]:
    key, separator, value = assignment.partition("=")
    if not separator or not key or not value:
        raise CloudCommandError(f"expected ROLE=VALUE, got {assignment!r}")
    return key, value


def run_keys(
    args: argparse.Namespace,
    settings: Settings,
    store: KeyStore,
    out: Printer,
    *,
    read_secret: Callable[[str], str] = getpass.getpass,
    read_stdin: Callable[[], str] = lambda: input(),  # noqa: PLW0108
) -> int:
    command: str = args.keys_command
    if command == "set":
        key = read_stdin() if args.stdin else read_secret(f"API key for {args.provider}: ")
        store.set(args.provider, key)
        out(f"stored the key for {args.provider}")
    elif command == "delete":
        store.delete(args.provider)
        out(f"removed the key for {args.provider}")
    else:
        names = sorted(settings.cloud.providers)
        if not names:
            out("no cloud providers configured (see: ve cloud add)")
        for name in names:
            marker = "key stored" if store.get(name) else "NO KEY"
            active = " (active)" if settings.cloud.active == name else ""
            out(f"{name}{active}: {marker}")
    return 0


def run_cloud(args: argparse.Namespace, settings: Settings, store: KeyStore, out: Printer) -> int:
    path = settings.settings_path()
    command: str = args.cloud_command
    if command == "add":
        _add(args, path, out)
    elif command == "route":
        _route(args.assignment, path, out)
    elif command == "budget":
        _budget(args.amount, path, out)
    else:
        _status(settings, store, out)
    return 0


def _add(args: argparse.Namespace, path: Path, out: Printer) -> None:
    base = ["cloud", "providers", args.name]
    set_setting(path, [*base, "base_url"], args.base_url)
    if args.label:
        set_setting(path, [*base, "label"], args.label)
    for assignment in args.model or []:
        role, model = _split(assignment)
        if role not in ROLES:
            raise CloudCommandError(f"unknown role {role!r}")
        set_setting(path, [*base, "models", role], model)
    if args.use:
        set_setting(path, ["cloud", "active"], args.name)
    out(f"saved provider {args.name}; store its key with: ve keys set {args.name}")


def _route(assignments: list[str], path: Path, out: Printer) -> None:
    for assignment in assignments:
        role, policy = _split(assignment)
        if role not in ROLES:
            raise CloudCommandError(f"unknown role {role!r}")
        if policy not in POLICIES:
            raise CloudCommandError(f"policy must be one of {', '.join(POLICIES)}")
        set_setting(path, ["cloud", "routing", role], policy)
        out(f"{role} -> {policy}")


def _budget(amount: str, path: Path, out: Printer) -> None:
    if amount.lower() == "off":
        set_setting(path, ["cloud", "monthly_budget_usd"], None)
        out("monthly budget removed")
        return
    try:
        value = float(amount.lstrip("$"))
    except ValueError as exc:
        raise CloudCommandError(f"budget must be a number or 'off', got {amount!r}") from exc
    set_setting(path, ["cloud", "monthly_budget_usd"], value)
    out(f"monthly budget set to ${value:.2f}")


def _status(settings: Settings, store: KeyStore, out: Printer) -> None:
    cloud = settings.cloud
    out(f"active provider : {cloud.active or '(none: everything stays local)'}")
    for name, provider in sorted(cloud.providers.items()):
        key = "key stored" if store.get(name) else "NO KEY"
        models = ", ".join(f"{r}={m}" for r, m in provider.models.items()) or "no models set"
        out(f"  {name}: {provider.base_url} [{key}] {models}")
    routing = ", ".join(f"{r}={p}" for r, p in sorted(cloud.routing.items())) or "all local"
    out(f"routing         : {routing}")
    with StateDb(settings.storage.data_dir) as state:
        spent = state.spend_since(month_start(_now()))
    budget = f" of ${cloud.monthly_budget_usd:.2f}" if cloud.monthly_budget_usd else ""
    out(f"spend this month: ${spent:.4f}{budget}")


def _now() -> float:
    import time

    return time.time()
