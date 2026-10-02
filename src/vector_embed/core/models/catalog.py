"""Curated model catalog (``models_catalog.toml``): role preference lists and model facts."""

import tomllib
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

CATALOG_FILENAME = "models_catalog.toml"

ROLE_EMBED = "embed"
ROLE_CHAT = "chat"
ROLE_MATCH_SCORER = "match_scorer"
ROLE_CODE_CHAT = "code_chat"
ROLE_CAPTION = "caption"
ROLE_SUMMARIZER = "summarizer"
ROLE_RERANKER = "reranker"
ROLES = (
    ROLE_EMBED,
    ROLE_CHAT,
    ROLE_MATCH_SCORER,
    ROLE_CODE_CHAT,
    ROLE_CAPTION,
    ROLE_SUMMARIZER,
    ROLE_RERANKER,
)


class CatalogError(ValueError):
    """The catalog file is unreadable or inconsistent."""


class CatalogModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    vram_mb: int = Field(gt=0)
    notes: str = ""
    cpu_ok: bool = False  # small enough to be a sensible choice on a machine without a GPU


class Catalog(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    roles: dict[str, list[str]]
    models: dict[str, CatalogModel] = {}
    warnings: dict[str, str] = {}

    def preferences(self, role: str) -> list[str]:
        return list(self.roles.get(role, ()))

    def vram_mb(self, name: str) -> int | None:
        entry = self.models.get(name)
        return entry.vram_mb if entry else None

    def known_names(self) -> set[str]:
        return {name for names in self.roles.values() for name in names} | set(self.models)


def parse_catalog(text: str) -> Catalog:
    try:
        catalog = Catalog.model_validate(tomllib.loads(text))
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise CatalogError(f"invalid model catalog: {exc}") from exc
    unknown_roles = sorted(set(catalog.roles) - set(ROLES))
    if unknown_roles:
        raise CatalogError(f"unknown roles in catalog: {', '.join(unknown_roles)}")
    return catalog


def load_catalog(override_dir: Path | None = None) -> Catalog:
    """The packaged catalog, or ``<override_dir>/models_catalog.toml`` when it exists."""
    if override_dir is not None:
        override = Path(override_dir) / CATALOG_FILENAME
        if override.is_file():
            return parse_catalog(override.read_text(encoding="utf-8"))
    text = resources.files(__package__).joinpath(CATALOG_FILENAME).read_text(encoding="utf-8")
    return parse_catalog(text)
