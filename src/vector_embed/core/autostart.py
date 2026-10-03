"""Start with Windows: Task Scheduler entries for the watcher and the tray app (per user).

The settings that matter are the battery flags: without them Task Scheduler kills the task when the
laptop is unplugged, and the watcher must keep recording changes on battery (it only *indexes* on
AC). Everything goes through a PowerShell script built here and run through an injectable runner.
"""

import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from vector_embed.core.process import self_command

logger = logging.getLogger(__name__)

_NO_WINDOW = 0x08000000
_RESTART_MINUTES = 1
_RESTART_COUNT = 5
_SCRIPT_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class AutostartTask:
    name: str
    entry: str  # a ``vector_embed.core.process.ENTRY_POINTS`` name


TASKS = (
    AutostartTask("VectorEmbed Watcher", "watcher"),
    AutostartTask("VectorEmbed Search", "app"),
)

CommandBuilder = Callable[[str], list[str]]
ScriptRunner = Callable[[str], int]  # PowerShell script text -> exit code


def _quote(text: str) -> str:
    """A PowerShell single-quoted literal."""
    return "'" + text.replace("'", "''") + "'"


def _argument_line(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv)


def register_script(tasks: tuple[AutostartTask, ...], command: CommandBuilder) -> str:
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) "
        f"-RestartCount {_RESTART_COUNT} "
        f"-RestartInterval (New-TimeSpan -Minutes {_RESTART_MINUTES}) -MultipleInstances IgnoreNew",
        # DOMAIN\user (or MACHINE\user): the bare user name fails to resolve on a domain or
        # Microsoft-account PC ("No mapping between account names and security IDs").
        "$me = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name",
        "$trigger = New-ScheduledTaskTrigger -AtLogOn -User $me",
        "$principal = New-ScheduledTaskPrincipal -UserId $me "
        "-LogonType Interactive -RunLevel Limited",
    ]
    for task in tasks:
        argv = command(task.entry)
        lines.append(
            f"$action = New-ScheduledTaskAction -Execute {_quote(argv[0])} "
            f"-Argument {_quote(_argument_line(argv[1:]))}"
        )
        lines.append(
            f"Register-ScheduledTask -TaskName {_quote(task.name)} -Action $action "
            "-Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null"
        )
    return "\n".join(lines)


def unregister_script(tasks: tuple[AutostartTask, ...]) -> str:
    return "\n".join(
        f"Unregister-ScheduledTask -TaskName {_quote(task.name)} -Confirm:$false "
        "-ErrorAction SilentlyContinue"
        for task in tasks
    )


def run_powershell(script: str) -> int:
    try:
        result = subprocess.run(  # noqa: S603  # fixed argv; the script is built from our constants
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=_SCRIPT_TIMEOUT_SECONDS,
            creationflags=_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("autostart: could not run PowerShell: %s", exc)
        return 1
    if result.returncode != 0:
        logger.warning("autostart: PowerShell failed: %s", result.stderr.strip())
    return result.returncode


class AutostartError(RuntimeError):
    """The scheduled tasks could not be changed."""


class Autostart:
    def __init__(
        self,
        run: ScriptRunner = run_powershell,
        command: CommandBuilder = lambda entry: self_command(entry, windowless=True),
        tasks: tuple[AutostartTask, ...] = TASKS,
    ) -> None:
        self._run = run
        self._command = command
        self._tasks = tasks

    def register(self) -> None:
        """Create (or replace) the logon tasks for the current user."""
        if self._run(register_script(self._tasks, self._command)) != 0:
            raise AutostartError("could not register the startup tasks")

    def unregister(self) -> None:
        """Remove the tasks; a task that is not there is fine."""
        self._run(unregister_script(self._tasks))

    def apply(self, enabled: bool) -> None:
        if enabled:
            self.register()
        else:
            self.unregister()
