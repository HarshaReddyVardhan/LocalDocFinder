"""Typed, versioned application settings.

Precedence (highest first): environment variables (``VE_`` prefix, ``__`` for nesting),
a local ``.env`` file, ``settings.toml``, built-in defaults. The TOML file carries a
``schema_version`` and is migrated forward before validation, so old files keep working.
"""

import os
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from vector_embed.core.data_migration import migrate_legacy_data

SCHEMA_VERSION = 1
APP_DIR_NAME = "VectorEmbed"  # the Velopack install folder; uninstall deletes all of it
DATA_DIR_NAME = "VectorEmbedData"  # a sibling, so uninstalling never takes the index with it
SETTINGS_FILENAME = "settings.toml"

RawSettings = dict[str, Any]
Migration = Callable[[RawSettings], RawSettings]

# Maps "from version" -> function producing the next version's layout.
MIGRATIONS: dict[int, Migration] = {}


class SettingsError(ValueError):
    """Raised when settings cannot be read, migrated or validated."""


def default_data_dir() -> Path:
    """Where the index, queue and logs live; under LOCALAPPDATA, a directory scope never indexes."""
    return _local_base() / DATA_DIR_NAME


def legacy_data_dir() -> Path:
    """Where releases before the data/install split kept their data (the install folder)."""
    return _local_base() / APP_DIR_NAME


def _local_base() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) if base else Path.home()


class _Section(BaseModel):
    """Base for settings sections: immutable, and unknown keys are errors (typo protection)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- scope
class ScopeSettings(_Section):
    """What is indexed. Rule order is documented in ``vector_embed.core.scope``."""

    roots: tuple[str, ...] = (str(Path.home()), "D:\\")

    blocked_dirs: frozenset[str] = frozenset(
        {
            # system and program directories
            "windows", "$recycle.bin", "system volume information", "recovery",
            "program files", "program files (x86)", "programdata", "appdata",
            # build output, package caches, virtualenvs, tool state
            "node_modules", ".git", "target", "dist", "build", "out", "bin", "obj",
            "venv", ".venv", "env", ".env", "__pycache__", ".pytest_cache", ".mypy_cache",
            ".ruff_cache", ".idea", ".vscode", ".next", ".nuxt", ".cache", "vendor", "pkg",
        }
    )  # fmt: skip

    project_markers: frozenset[str] = frozenset(
        {".git", "package.json", "pyproject.toml", "Cargo.toml", "go.mod", "pom.xml"}
    )
    project_marker_globs: tuple[str, ...] = ("*.sln",)

    # Hidden dir (fnmatch, lowercase) -> include globs relative to it. Exempt from the
    # hidden-directory rule only: the secrets denylist and blocked dirs are still checked first.
    ai_note_dirs: dict[str, tuple[str, ...]] = {
        ".claude": ("**/*.md",),
        ".cursor": ("rules/**", "*.md", "*.mdc"),
        ".github": ("copilot-instructions.md", "instructions/**/*.md", "prompts/**"),
        ".gemini": ("**/*.md",),
        ".codex": ("**/*.md",),
        ".continue": ("**/*.md", "rules/**"),
        ".windsurf": ("**/*.md", "rules/**"),
        ".kiro": ("**/*.md",),
        ".aider*": ("**/*.md",),
    }
    ai_excluded_subdirs: frozenset[str] = frozenset(
        {"plugins", "shell-snapshots", "todos", "statsig", "ide", "cache"}
    )
    # Session transcripts are large, noisy and may contain pasted secrets.
    index_ai_transcripts: bool = False
    ai_transcript_glob: str = "projects/**/*.jsonl"
    agent_rule_files: frozenset[str] = frozenset(
        {"claude.md", "agents.md", "gemini.md", ".cursorrules", ".windsurfrules"}
    )

    # Secrets denylist: evaluated first, on every path, without exceptions.
    secret_name_patterns: tuple[str, ...] = (
        ".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12", "id_rsa*", "id_ed25519*",
        "*credential*", ".npmrc", ".pypirc", "*.kdbx",
        "client_secret*.json", "*service-account*.json", "*serviceaccount*.json",
        "*adminsdk*.json", "token.json", "*.ppk", "id_dsa*", "id_ecdsa*", ".netrc", "_netrc",
        "*.jks", "*.keystore", ".htpasswd", "secrets.json", "secrets.yaml", "secrets.yml",
        "secrets.toml",
    )  # fmt: skip
    secret_name_exceptions: frozenset[str] = frozenset(
        {".env.example", ".env.sample", ".env.template"}
    )
    # Slash-separated globs over the lowercase full path.
    secret_path_globs: tuple[str, ...] = ("**/.claude/settings*.json",)

    noise_name_patterns: tuple[str, ...] = (
        "*.min.js", "*.min.css", "*.map", "*.lock", "package-lock.json", "pnpm-lock.yaml",
        "yarn.lock", "npm-shrinkwrap.json", "go.sum", "composer.lock", "gemfile.lock",
        "poetry.lock", "uv.lock", "pipfile.lock", ".aider*history*", ".aider.tags.cache*",
    )  # fmt: skip

    # Extension-less or dot-prefixed files that are still plain text (lowercase).
    text_filenames: frozenset[str] = frozenset(
        {
            "dockerfile", "makefile", "readme", "license", "licence", "copying", ".gitignore",
            ".gitattributes", ".dockerignore", ".editorconfig", ".env.example", ".env.sample",
            ".env.template", ".cursorrules", ".windsurfrules",
        }
    )  # fmt: skip
    text_exts: frozenset[str] = frozenset(
        {
            ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".rs", ".go", ".dart", ".java", ".cs",
            ".kt", ".kts", ".scala", ".swift", ".gradle", ".py", ".ts", ".tsx", ".js", ".jsx",
            ".mjs", ".cjs", ".vue", ".svelte", ".php", ".rb", ".lua", ".r", ".sh", ".bash",
            ".ps1", ".bat", ".cmd", ".html", ".htm", ".css", ".scss", ".sass", ".less", ".svg",
            ".sql", ".json", ".yaml", ".yml", ".toml", ".xml", ".ini", ".cfg", ".ipynb", ".md",
            ".mdc", ".markdown", ".txt", ".rtf", ".tex",
        }
    )  # fmt: skip
    # Parsed into function/class chunks with tree-sitter; others are chunked by lines/paragraphs.
    code_exts: frozenset[str] = frozenset(
        {
            ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".rs", ".go", ".dart", ".java", ".cs",
            ".kt", ".kts", ".scala", ".swift", ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs",
            ".cjs", ".php", ".rb", ".lua", ".r", ".sh", ".bash", ".ps1",
        }
    )  # fmt: skip
    # Machine-written / data-ish formats: skipped above ``ChunkingSettings.max_data_file_kb``.
    data_exts: frozenset[str] = frozenset(
        {".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sql", ".html", ".htm",
         ".css", ".scss", ".sass", ".less"}
    )  # fmt: skip
    doc_exts: frozenset[str] = frozenset({".pdf", ".docx", ".pptx"})
    image_exts: frozenset[str] = frozenset(
        {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    )

    max_text_size_mb: float = Field(default=15, gt=0)
    max_doc_size_mb: float = Field(default=50, gt=0)
    max_image_size_mb: float = Field(default=100, gt=0)

    # Generated-file heuristics, applied by extractors once content is read.
    max_avg_line_length: int = Field(default=300, gt=0)
    generated_markers: tuple[str, ...] = (
        "@generated",
        "auto-generated",
        "autogenerated",
        "do not edit",
    )


# --------------------------------------------------------------------------- power / idle
class PowerSettings(_Section):
    """Hard power rules: indexing only on AC; chat on battery is opt-in."""

    require_ac_power: bool = True
    ac_settle_seconds: int = Field(default=120, ge=0)
    poll_seconds: int = Field(default=30, gt=0)
    search_on_battery: bool = True
    search_cpu_on_battery: bool = True  # embed the query with num_gpu=0 when unplugged
    chat_on_battery: bool = False


class IdleSettings(_Section):
    """Idle gate evaluated before every indexing batch."""

    cpu_percent: float = Field(default=15, ge=0, le=100)
    cpu_seconds: int = Field(default=60, gt=0)
    no_input_seconds: int = Field(default=60, ge=0)
    gpu_max_util_percent: float = Field(default=20, ge=0, le=100)
    # Checked before every batch while the embedder is loaded; utilisation is not, because the
    # worker's own embedding work would trip it.
    min_free_vram_mb: int = Field(default=300, ge=0)
    worker_yield_input_seconds: int = Field(default=3, ge=0)  # running worker yields on user input
    file_debounce_seconds: int = Field(default=30, ge=0)
    reconcile_interval_hours: float = Field(default=6, gt=0)


# --------------------------------------------------------------------------- models
class ModelPrefixes(_Section):
    """Per-model task prefixes; queries and indexed chunks are embedded differently."""

    query: str = ""
    document: str = ""


_QWEN3_QUERY_PREFIX = (
    "Instruct: Given a search query, retrieve relevant code and text passages "
    "that answer the query\nQuery: "
)


class EmbeddingSettings(_Section):
    """Embedding model. Ollama silently truncates to ``num_ctx``, so always pass it."""

    model: str = "qwen3-embedding:0.6b"
    dim: int = Field(default=1024, gt=0)
    num_ctx: int = Field(default=8192, gt=0)
    batch_size: int = Field(default=32, gt=0)
    keep_alive: str = "5m"
    prefixes: dict[str, ModelPrefixes] = {
        "qwen3-embedding:0.6b": ModelPrefixes(query=_QWEN3_QUERY_PREFIX),
        "qwen3-embedding:4b": ModelPrefixes(query=_QWEN3_QUERY_PREFIX),
        "nomic-embed-text": ModelPrefixes(query="search_query: ", document="search_document: "),
        "bge-m3": ModelPrefixes(),
        "mxbai-embed-large": ModelPrefixes(
            query="Represent this sentence for searching relevant passages: "
        ),
    }

    def prefixes_for(self, model: str | None = None) -> ModelPrefixes:
        """Prefixes for ``model`` (default: the configured one); unknown models get none."""
        return self.prefixes.get(model or self.model, ModelPrefixes())


class ChunkingSettings(_Section):
    max_chunk_chars: int = Field(default=4500, gt=0)  # ~1500 tokens of code, under num_ctx
    target_chunk_chars: int = Field(default=1800, gt=0)  # small neighbours merge up to this
    min_chunk_chars: int = Field(default=400, ge=0)
    line_overlap: int = Field(default=6, ge=0)  # overlap when an oversized unit is split by lines
    max_chunks_per_file: int = Field(default=300, gt=0)
    max_chunks_per_doc: int = Field(default=1500, gt=0)  # PDFs / Office documents (books)
    max_data_file_kb: int = Field(default=512, gt=0)  # .json/.xml/... above this are data dumps
    stored_text_chars: int = Field(default=3000, gt=0)  # raw text kept for snippets / FTS
    worker_batch_files: int = Field(default=16, gt=0)


class ImageSettings(_Section):
    """Images become text (OCR, optional caption) so they share the text embedding space."""

    enable_captions: bool = False  # off by default; runs only during idle indexing
    caption_model: str = "qwen2.5vl:3b"
    min_pixels: int = Field(default=150, gt=0)  # skip icons/logos below this on both sides
    max_per_doc: int = Field(default=50, gt=0)  # a 500-page scan must not stall the queue
    max_decode_pixels: int = Field(default=200_000_000, gt=0)  # decompression-bomb guard
    thumbnail_size: int = Field(default=480, gt=0)
    ocr_max_dimension: int = Field(default=4000, gt=0)


class DocTypeRule(_Section):
    """Regex evidence for one document type (all case-insensitive)."""

    filename: tuple[str, ...] = ()  # matched against the file name
    headings: tuple[str, ...] = ()  # matched against whole lines (section headings)
    keywords: tuple[str, ...] = ()  # matched anywhere in the text
    path: tuple[str, ...] = ()  # matched against the full path with forward slashes


def _default_doctype_rules() -> dict[str, DocTypeRule]:
    return {
        "resume": DocTypeRule(
            filename=(r"resume", r"(?<![a-z])cv(?![a-z])", r"curriculum"),
            headings=(r"(work )?experience", r"education", r"skills", r"projects", r"summary"),
            keywords=(r"references available", r"professional experience"),
        ),
        "cover_letter": DocTypeRule(
            filename=(r"cover[ _-]?letter",),
            headings=(),
            keywords=(
                r"dear (hiring|sir|madam|mr|ms)",
                r"sincerely",
                r"i am writing to (apply|express)",
            ),
        ),
        "jd": DocTypeRule(
            filename=(r"(?<![a-z])jd(?![a-z])", r"job[ _-]?(description|posting)"),
            headings=(
                r"responsibilities",
                r"requirements",
                r"qualifications",
                r"about the (role|job)",
                r"what you('| wi)ll do",
                r"nice to have",
                r"benefits",
            ),
            keywords=(
                r"we are (hiring|looking for)",
                r"years of experience",
                r"apply now",
                r"equal opportunity",
            ),
        ),
        "invoice": DocTypeRule(
            filename=(r"invoice", r"receipt"),
            headings=(r"bill to", r"amount due", r"subtotal"),
            keywords=(r"invoice (no|number|#)", r"amount due", r"\btotal\b", r"due date"),
        ),
        "paper": DocTypeRule(
            headings=(r"abstract", r"introduction", r"references", r"related work", r"conclusion"),
            keywords=(r"\bdoi\b", r"arxiv", r"et al\."),
        ),
        "plan": DocTypeRule(
            path=(r"/plans/", r"/\.claude/", r"plan\.md$"),
            filename=(r"plan",),
            headings=(r"context", r"verification", r"steps", r"build order", r"plan"),
        ),
        "notes": DocTypeRule(
            path=(r"/memory/", r"/notes?/"),
            filename=(r"notes?", r"todo", r"journal"),
        ),
    }  # fmt: skip


class DocTypeSettings(_Section):
    """Document classification and version grouping (see ``vector_embed.core.doctypes``)."""

    rules: dict[str, DocTypeRule] = Field(default_factory=_default_doctype_rules)
    threshold: float = Field(default=0.4, ge=0, le=1)  # below this a document is "other"
    prototype_weight: float = Field(default=0.4, ge=0, le=1)
    version_similarity: float = Field(default=0.9, gt=0, le=1)
    versioned_types: frozenset[str] = frozenset(
        {"resume", "cover_letter", "jd", "invoice", "paper"}
    )
    full_text_max_chars: int = Field(default=20_000, gt=0)  # larger documents keep no full text


class ModelSettings(_Section):
    """Role-based model selection (see ``vector_embed.core.models``)."""

    overrides: dict[str, str] = Field(default_factory=dict)  # role -> model name
    refresh_seconds: int = Field(default=3600, gt=0)  # re-discover installed models this often


class ChatSettings(_Section):
    """Chat / Ask / Match. The chat model is loaded on demand and unloaded afterwards."""

    num_ctx: int = Field(default=8192, gt=0)  # what fits next to the embedder-free VRAM budget
    keep_alive: str = "10m"  # how long the model stays loaded between follow-up questions
    idle_unload_seconds: int = Field(default=600, gt=0)
    temperature: float = Field(default=0.2, ge=0, le=2)
    context_token_budget: int = Field(default=5000, gt=0)  # retrieved or pinned text per prompt
    history_token_budget: int = Field(default=1500, gt=0)
    retrieve_chunks: int = Field(default=20, gt=0)
    code_routing: bool = True  # code-heavy context goes to the ``code_chat`` role


class MatchSettings(_Section):
    """Document matching, e.g. a job description against resumes."""

    default_doc_type: str = "resume"
    recall_k: int = Field(default=10, gt=0)  # candidates recalled before the user picks
    similarity_threshold: float = Field(default=0.3, ge=0, le=1)  # default ticked above this
    scoring_budget_tokens: int = Field(default=5000, gt=0)  # one document per scoring call
    reserved_output_tokens: int = Field(default=1200, gt=0)  # left free in the context window
    must_weight: float = Field(default=2.0, gt=0)  # a must-have counts double a nice-to-have
    nice_weight: float = Field(default=1.0, gt=0)
    max_requirements: int = Field(default=25, gt=0)
    evidence_threshold: float = Field(default=0.85, gt=0, le=1)  # fuzzy match of quoted evidence
    max_fts_terms: int = Field(default=200, gt=0)  # job descriptions are long; cap keyword terms


class PrivacySettings(_Section):
    """What may leave the machine. Local models are never filtered; cloud requests always are."""

    # Never sent to a cloud model: matched against the lowercase full path with "/" separators.
    never_send_globs: tuple[str, ...] = (
        "**/.claude/projects/*/memory/**",
        "**/.claude/memory/**",
    )
    never_send_doc_types: frozenset[str] = frozenset()
    redact_personal: bool = False  # name, email, phone, address, profile URLs -> placeholders
    known_names: tuple[str, ...] = ()  # extra names to redact when redact_personal is on
    mask_ids_locally: bool = False  # also mask IDs for local models (off: nothing leaves)


class CloudProviderSettings(_Section):
    """One OpenAI-compatible endpoint: OpenAI, OpenRouter, LM Studio, vLLM, Groq, ..."""

    base_url: str
    label: str = ""
    models: dict[str, str] = Field(default_factory=dict)  # role -> model id
    # USD per million tokens (input, output) for models whose catalog gives no pricing.
    pricing: dict[str, tuple[float, float]] = Field(default_factory=dict)


class CloudSettings(_Section):
    """Cloud routing. Empty by default: nothing is sent anywhere until a provider is added."""

    providers: dict[str, CloudProviderSettings] = Field(default_factory=dict)
    active: str | None = None  # key of the provider used for cloud calls
    routing: dict[str, Literal["local", "cloud", "auto"]] = Field(default_factory=dict)
    monthly_budget_usd: float | None = Field(default=None, gt=0)
    max_output_tokens: int = Field(default=2048, gt=0)  # cap on every cloud reply: it is billed

    def policy(self, role: str) -> str:
        """``local`` unless the user chose otherwise; the safe default."""
        return self.routing.get(role, "local")


class SearchSettings(_Section):
    rrf_k: int = Field(default=60, gt=0)
    candidates: int = Field(default=60, gt=0)
    results: int = Field(default=25, gt=0)
    current_project_boost: float = Field(default=1.15, ge=1)
    filename_boost: float = Field(default=1.3, ge=1)  # query word appears in the file name
    hotkey: str = "ctrl+alt+space"  # Alt+Space belongs to Windows and PowerToys Run
    vector_index_min_rows: int = Field(default=100_000, gt=0)  # flat search below, IVF_PQ above


class AppSettings(_Section):
    start_with_windows: bool = True  # register the watcher and tray app with Task Scheduler


class UpdateSettings(_Section):
    auto_check: bool = True  # look for a new release in the background (at start, then daily)
    repo_url: str = ""  # GitHub repository to update from; empty uses the one built in


class StorageSettings(_Section):
    data_dir: Path = Field(default_factory=default_data_dir)


# --------------------------------------------------------------------------- root
class Settings(BaseSettings):
    """Application settings; see the module docstring for precedence."""

    model_config = SettingsConfigDict(
        env_prefix="VE_",
        env_nested_delimiter="__",
        env_file=None,  # chosen per load: only the settings folder's .env, never the cwd's
        env_file_encoding="utf-8",
        extra="ignore",  # a shared .env may hold unrelated keys; TOML keys are checked in load
        frozen=True,
    )

    schema_version: int = SCHEMA_VERSION
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    ollama_host: str = "http://127.0.0.1:11434"
    scope: ScopeSettings = Field(default_factory=ScopeSettings)
    power: PowerSettings = Field(default_factory=PowerSettings)
    idle: IdleSettings = Field(default_factory=IdleSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    images: ImageSettings = Field(default_factory=ImageSettings)
    doctypes: DocTypeSettings = Field(default_factory=DocTypeSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    match: MatchSettings = Field(default_factory=MatchSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    cloud: CloudSettings = Field(default_factory=CloudSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    app: AppSettings = Field(default_factory=AppSettings)
    updates: UpdateSettings = Field(default_factory=UpdateSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    # Where these settings were read from; not a user setting (``load_settings`` fills it in).
    settings_file: Path | None = Field(default=None, exclude=True, repr=False)

    def settings_path(self) -> Path:
        """The file settings are saved to: the one they were loaded from.

        Never ``storage.data_dir / settings.toml`` for a loaded configuration: a ``[storage]
        data_dir`` entry moves the index, but ``settings.toml`` itself stays where it is read.
        """
        return self.settings_file or self.storage.data_dir / SETTINGS_FILENAME

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Environment beats ``.env`` beats the TOML file (passed as init kwargs)."""
        assert isinstance(env_settings, EnvSettingsSource)
        assert isinstance(dotenv_settings, DotEnvSettingsSource)
        return (env_settings, dotenv_settings, init_settings)


def migrate(
    raw: Mapping[str, Any], migrations: Mapping[int, Migration] | None = None
) -> RawSettings:
    """Upgrade a raw settings mapping to ``SCHEMA_VERSION``; a missing version means current."""
    steps = MIGRATIONS if migrations is None else migrations
    data: RawSettings = dict(raw)
    version = data.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise SettingsError(f"invalid schema_version: {version!r}")
    if version > SCHEMA_VERSION:
        raise SettingsError(
            f"settings schema_version {version} is newer than this app supports ({SCHEMA_VERSION})"
        )
    while version < SCHEMA_VERSION:
        step = steps.get(version)
        if step is None:
            raise SettingsError(f"no migration from settings schema_version {version}")
        data = step(data)
        version += 1
        data["schema_version"] = version
    return data


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from ``path`` (default ``<data dir>/settings.toml``); missing is fine."""
    toml_path = path if path is not None else _default_settings_path()
    raw: RawSettings = {}
    if toml_path.is_file():
        try:
            with toml_path.open("rb") as handle:
                raw = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise SettingsError(f"cannot read {toml_path}: {exc}") from exc
    try:
        data = migrate(raw)
        unknown = sorted(set(data) - set(Settings.model_fields))
        if unknown:
            raise SettingsError(f"unknown settings keys in {toml_path}: {', '.join(unknown)}")
        # A repo's own .env must not steer the app (an Ollama host or cloud URL, for instance),
        # so only the .env beside settings.toml counts.
        # ``_env_file`` is a runtime option of pydantic-settings that its stubs do not declare.
        data.pop("settings_file", None)  # the file cannot name itself; it is recorded below
        return Settings(
            **{**data, "_env_file": toml_path.parent / ".env", "settings_file": toml_path}
        )
    except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
        if isinstance(exc, SettingsError):
            raise
        raise SettingsError(f"invalid settings in {toml_path}: {exc}") from exc


def _default_settings_path() -> Path:
    override = os.environ.get("VE_STORAGE__DATA_DIR")
    if override:
        return Path(override) / SETTINGS_FILENAME
    migrate_legacy_data(legacy_data_dir(), default_data_dir())
    return default_data_dir() / SETTINGS_FILENAME
