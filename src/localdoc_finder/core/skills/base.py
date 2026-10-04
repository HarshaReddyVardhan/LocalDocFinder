"""Skills: user-facing features (Search, Ask, Chat, Match, ...) behind one small contract.

A skill declares a name, a typed input model, the model *roles* it needs and a UI hint. The
CLI, the desktop UI and the MCP server are generated from these declarations, so a new feature
is one file with ``@register_skill`` and no front-end edits.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import numpy as np
from pydantic import BaseModel

from localdoc_finder.core.power import PowerGate
from localdoc_finder.core.providers.base import EmbedKind
from localdoc_finder.core.registry import Registry, discover_modules
from localdoc_finder.core.settings import Settings
from localdoc_finder.core.store.lance import LanceStore
from localdoc_finder.core.store.sqlite import CHAT_LOCK, StateDb

UI_LIST = "list"  # a ranked result list
UI_PANEL = "panel"  # a streaming text panel (answers, chat)
UI_TABLE = "table"  # a table (match results)


class QueryEmbedder(Protocol):
    def embed(self, texts: list[str], kind: EmbedKind = "doc", cpu: bool = False) -> np.ndarray: ...


@dataclass
class SkillContext:
    """Collaborators a skill may use. ``extras`` carries optional parts added by later skills."""

    settings: Settings
    state: StateDb
    store: LanceStore
    embedder: QueryEmbedder
    power: PowerGate
    extras: dict[str, Any] = field(default_factory=dict)

    def query_on_cpu(self) -> bool:
        """Embed queries on the CPU when unplugged or any process holds the chat lock."""
        return self.power.search_on_cpu() or self.state.lock_held(CHAT_LOCK)

    @property
    def similarity_floor(self) -> float:
        """The embedder's ``min_similarity``: below it a vector-only match is probably unrelated."""
        return self.settings.embedding.profile_for().min_similarity


class SkillInput(BaseModel):
    """Base for skill inputs; subclasses are rendered into CLI flags and MCP tool schemas."""


class Skill(ABC):
    name: ClassVar[str]
    title: ClassVar[str]
    description: ClassVar[str]
    Input: ClassVar[type[SkillInput]] = SkillInput
    roles: ClassVar[tuple[str, ...]] = ()  # model roles the skill needs (see models catalog)
    ui_hint: ClassVar[str] = UI_LIST
    cli_positional: ClassVar[str | None] = None  # input field taken as the CLI positional

    def __init__(self, ctx: SkillContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def run(self, params: SkillInput) -> object:
        """Execute the skill; the return type is skill-specific (see ``render``)."""
        ...

    def stream(self, params: SkillInput) -> Iterator[str] | None:
        """Text deltas for skills that answer incrementally (Ask, Chat); ``None`` otherwise."""
        return None

    def render(self, output: object) -> str:
        """Plain-text rendering of ``run`` output for the CLI."""
        return str(output)


SKILLS: Registry[type[Skill]] = Registry("skill")
register_skill = SKILLS.register

_BUILTIN_PACKAGE = "localdoc_finder.core.skills"


def load_skills() -> list[type[Skill]]:
    """Import built-in skill modules and return every registered skill class."""
    discover_modules(_BUILTIN_PACKAGE)
    return list(SKILLS)


def panel_skills() -> list[type[Skill]]:
    """Skills any front-end can run without special UI: one text input, a text answer."""
    return [s for s in load_skills() if s.ui_hint == UI_PANEL and s.cli_positional]


def create_skill(name: str, ctx: SkillContext) -> Skill:
    load_skills()
    return SKILLS.get(name)(ctx)
