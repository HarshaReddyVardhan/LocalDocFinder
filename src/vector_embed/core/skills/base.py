"""Skills: user-facing features (Search, Ask, Chat, Match, ...) behind one small contract.

A skill declares a name, a typed input model, the model *roles* it needs and a UI hint. The
CLI, the desktop UI and the MCP server are generated from these declarations, so a new feature
is one file with ``@register_skill`` and no front-end edits.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import numpy as np
from pydantic import BaseModel

from vector_embed.core.power import PowerGate
from vector_embed.core.providers.base import EmbedKind
from vector_embed.core.registry import Registry, discover_modules
from vector_embed.core.settings import Settings
from vector_embed.core.store.lance import LanceStore
from vector_embed.core.store.sqlite import StateDb

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

    def render(self, output: object) -> str:
        """Plain-text rendering of ``run`` output for the CLI."""
        return str(output)


SKILLS: Registry[type[Skill]] = Registry("skill")
register_skill = SKILLS.register

_BUILTIN_PACKAGE = "vector_embed.core.skills"


def load_skills() -> list[type[Skill]]:
    """Import built-in skill modules and return every registered skill class."""
    discover_modules(_BUILTIN_PACKAGE)
    return list(SKILLS)


def create_skill(name: str, ctx: SkillContext) -> Skill:
    load_skills()
    return SKILLS.get(name)(ctx)
