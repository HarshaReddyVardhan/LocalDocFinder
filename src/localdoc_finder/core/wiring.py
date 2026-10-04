"""The few wiring helpers the always-on watcher needs.

Kept apart from ``runtime`` (which imports the embedding, chat and extraction stacks) so the
watcher stays a small process that never loads ML libraries: its idle memory is a design goal.
"""

from pathlib import Path

from localdoc_finder.core.projects import Projects
from localdoc_finder.core.scope import ScopePolicy
from localdoc_finder.core.scope_roots import resolve_roots
from localdoc_finder.core.settings import Settings

LOGS_DIRNAME = "logs"
THUMBS_DIRNAME = "thumbs"


def build_scope(settings: Settings) -> ScopePolicy:
    """The scope policy; the app's own data folder is never indexed."""
    return ScopePolicy(
        settings.scope,
        blocked_roots=[settings.storage.data_dir],
        scan_roots=resolve_roots(settings.scope),
    )


def build_projects(settings: Settings, scope: ScopePolicy) -> Projects:
    return Projects(scope, settings.scope, roots=resolve_roots(settings.scope))


def log_dir(settings: Settings) -> Path:
    return settings.storage.data_dir / LOGS_DIRNAME
