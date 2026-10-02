"""The few wiring helpers the always-on watcher needs.

Kept apart from ``runtime`` (which imports the embedding, chat and extraction stacks) so the
watcher stays a small process that never loads ML libraries: its idle memory is a design goal.
"""

from pathlib import Path

from vector_embed.core.projects import Projects
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import Settings

LOGS_DIRNAME = "logs"
THUMBS_DIRNAME = "thumbs"


def build_scope(settings: Settings) -> ScopePolicy:
    """The scope policy; the app's own data folder is never indexed."""
    return ScopePolicy(settings.scope, blocked_roots=[settings.storage.data_dir])


def build_projects(settings: Settings, scope: ScopePolicy) -> Projects:
    return Projects(scope, settings.scope)


def log_dir(settings: Settings) -> Path:
    return settings.storage.data_dir / LOGS_DIRNAME
