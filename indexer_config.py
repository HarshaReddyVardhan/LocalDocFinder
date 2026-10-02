import fnmatch
import os
import re
import stat
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
# Target roots: user directory on C:, and the full D: workspace drive
USER_HOME = Path.home()  # Resolves automatically to C:\Users\<Username>
WATCH_ROOTS = [
    str(USER_HOME),
    "D:\\",
]

# ---------------------------------------------------------------------------
# Embedding model (Ollama is the only model runtime)
# ---------------------------------------------------------------------------
EMBED_MODEL = "qwen3-embedding:0.6b"   # fallbacks: "bge-m3", "nomic-embed-text"
EMBED_DIM = 1024                        # qwen3-embedding:0.6b native; Matryoshka-truncatable
NUM_CTX = 8192                          # Ollama truncates to num_ctx; never rely on the default
MAX_CHUNK_TOKENS = 1500                 # keep chunks well under NUM_CTX
EMBED_BATCH_SIZE = 32

# Prefixes are applied per role: queries get "query", indexed chunks get "document".
MODEL_PREFIXES = {
    "qwen3-embedding:0.6b": {
        "query": "Instruct: Given a search query, retrieve relevant code and text passages that answer the query\nQuery: ",
        "document": "",
    },
    "qwen3-embedding:4b": {
        "query": "Instruct: Given a search query, retrieve relevant code and text passages that answer the query\nQuery: ",
        "document": "",
    },
    "nomic-embed-text": {"query": "search_query: ", "document": "search_document: "},
    "bge-m3": {"query": "", "document": ""},
    "mxbai-embed-large": {
        "query": "Represent this sentence for searching relevant passages: ",
        "document": "",
    },
}

# Images become text (OCR + optional caption) and share the text embedding space.
ENABLE_IMAGE_CAPTIONS = False
CAPTION_MODEL = "qwen2.5vl:3b"
MIN_IMAGE_PIXELS = 150          # skip icons/logos smaller than this on either side
MAX_IMAGES_PER_DOC = 50         # a 500-page scan must not stall the queue

# ---------------------------------------------------------------------------
# Power policy: indexing only when plugged in
# ---------------------------------------------------------------------------
REQUIRE_AC_POWER = True
AC_SETTLE_SECONDS = 120
POWER_POLL_SECONDS = 30
SEARCH_ON_BATTERY = True
SEARCH_CPU_ON_BATTERY = True    # query embedding with num_gpu=0 on battery

# ---------------------------------------------------------------------------
# Idle gate / scheduling
# ---------------------------------------------------------------------------
IDLE_CPU_PERCENT = 15
IDLE_CPU_SECONDS = 60
IDLE_NO_INPUT_SECONDS = 60
IDLE_GPU_MAX_UTIL_PERCENT = 20
WORKER_YIELD_INPUT_SECONDS = 3   # a running worker stops when the user touches the machine
FILE_DEBOUNCE_SECONDS = 30
RECONCILE_INTERVAL_HOURS = 6
SEARCH_MODEL_KEEP_ALIVE = "5m"

# Alt+Space is the Windows window menu and is claimed by PowerToys Run.
HOTKEY = "ctrl+alt+space"

# ---------------------------------------------------------------------------
# Storage. Lives under %LOCALAPPDATA% (a blocked dir, so the indexer never indexes itself).
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("LOCALAPPDATA", str(USER_HOME))) / "VectorEmbed"
LANCE_TABLE = "chunks"
VECTOR_INDEX_MIN_ROWS = 100_000  # flat search below this, IVF_PQ above

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
MAX_CHUNK_CHARS = 4500          # ~1500 tokens of code, safely under NUM_CTX
TARGET_CHUNK_CHARS = 1800       # small neighbours are merged up to this size
MIN_CHUNK_CHARS = 400           # chunks below this are merge candidates
LINE_OVERLAP = 6                # overlap when an oversized unit is split by lines
MAX_CHUNKS_PER_FILE = 300       # safety valve for huge files
MAX_CHUNKS_PER_DOC = 1500       # PDFs / Office documents (books)
MAX_DATA_FILE_KB = 512          # .json/.xml/.yaml/... beyond this are skipped (data dumps)
STORED_TEXT_CHARS = 3000        # raw text kept in the table for snippets / FTS
WORKER_BATCH_FILES = 16

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
RRF_K = 60
SEARCH_CANDIDATES = 60
SEARCH_RESULTS = 25
CURRENT_PROJECT_BOOST = 1.15

# ---------------------------------------------------------------------------
# Directory rules
# ---------------------------------------------------------------------------
# Excluded system, build, package, and cache directories (lowercase)
BLOCKED_DIRS = {
    # System & Program directories
    "windows", "$recycle.bin", "system volume information", "recovery",
    "program files", "program files (x86)", "programdata", "appdata",
    # Language/framework build artifacts & package caches
    "node_modules", ".git", "target", "dist", "build", "out", "bin", "obj",
    "venv", ".venv", "env", ".env", "__pycache__", ".pytest_cache",
    ".idea", ".vscode", ".next", ".nuxt", ".cache", "vendor", "pkg",
}

# Markers that make a directory a project root (used by projects.py).
PROJECT_MARKERS = {
    ".git", "package.json", "pyproject.toml", "Cargo.toml", "go.mod", "pom.xml",
}
PROJECT_MARKER_GLOBS = ("*.sln",)

# ---------------------------------------------------------------------------
# AI-tool folders (plans, memory, agent rules) are indexed on purpose.
# Maps a hidden directory (fnmatch pattern, lowercase) to include globs that are
# relative to that directory. Allowlisted dirs are exempt from the hidden-directory
# rule ONLY: the secrets denylist and BLOCKED_DIRS still apply first.
# ---------------------------------------------------------------------------
AI_NOTE_DIRS = {
    ".claude": ["**/*.md"],  # plans/, projects/*/memory/, skills/, agents/, commands/, CLAUDE.md
    ".cursor": ["rules/**", "*.md", "*.mdc"],
    ".github": ["copilot-instructions.md", "instructions/**/*.md", "prompts/**"],
    ".gemini": ["**/*.md"],
    ".codex": ["**/*.md"],
    ".continue": ["**/*.md", "rules/**"],
    ".windsurf": ["**/*.md", "rules/**"],
    ".kiro": ["**/*.md"],
    ".aider*": ["**/*.md"],
}

# Subdirectories that are never indexed inside an AI-tool dir.
AI_EXCLUDED_SUBDIRS = {
    "plugins", "shell-snapshots", "todos", "statsig", "ide", "cache",
}

# Session transcripts are large, noisy and may contain pasted secrets.
INDEX_AI_TRANSCRIPTS = False
AI_TRANSCRIPT_GLOB = "projects/**/*.jsonl"

# Root-level agent rule files in any project (extension-less ones need TEXT_FILENAMES).
AGENT_RULE_FILES = {
    "claude.md", "agents.md", "gemini.md", ".cursorrules", ".windsurfrules",
}

# ---------------------------------------------------------------------------
# File rules
# ---------------------------------------------------------------------------
# Secrets denylist: evaluated FIRST, on every path, with no exceptions.
SECRET_NAME_PATTERNS = (
    ".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12", "id_rsa*", "id_ed25519*",
    "*credential*", ".npmrc", ".pypirc", "*.kdbx",
)
# Template env files are safe and useful.
SECRET_NAME_EXCEPTIONS = (".env.example", ".env.sample", ".env.template")

# Lockfiles, minified bundles, source maps and tool chatter.
NOISE_NAME_PATTERNS = (
    "*.min.js", "*.min.css", "*.map", "*.lock", "package-lock.json",
    "pnpm-lock.yaml", "yarn.lock", "npm-shrinkwrap.json", "go.sum",
    "composer.lock", "gemfile.lock", "poetry.lock", "uv.lock", "pipfile.lock",
    ".aider*history*", ".aider.tags.cache*",
)

# Extension-less / dot-prefixed files that are still plain text (lowercase).
TEXT_FILENAMES = {
    "dockerfile", "makefile", "readme", "license", "licence", "copying",
    ".gitignore", ".gitattributes", ".dockerignore", ".editorconfig",
    ".env.example", ".env.sample", ".env.template",
    ".cursorrules", ".windsurfrules",
}

# Exhaustive code and note extensions
TEXT_EXTS = {
    # Systems & Compiled
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".rs", ".go", ".dart",
    # Enterprise & Managed
    ".java", ".cs", ".kt", ".kts", ".scala", ".swift", ".gradle",
    # Scripting & Web backend/frontend
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte",
    ".php", ".rb", ".lua", ".r", ".sh", ".bash", ".ps1", ".bat", ".cmd",
    # Web markup & styles
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".svg",
    # Query & Data / Config
    ".sql", ".json", ".yaml", ".yml", ".toml", ".xml", ".ini", ".cfg", ".ipynb",
    # Notes & Markup
    ".md", ".mdc", ".markdown", ".txt", ".rtf", ".tex",
}
if INDEX_AI_TRANSCRIPTS:
    TEXT_EXTS.add(".jsonl")

# Parsed into function/class chunks with tree-sitter (others are chunked by lines/paragraphs).
CODE_EXTS = {
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".rs", ".go", ".dart",
    ".java", ".cs", ".kt", ".kts", ".scala", ".swift",
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".php", ".rb", ".lua", ".r", ".sh", ".bash", ".ps1",
}
# Machine-written / data-ish formats: skipped above MAX_DATA_FILE_KB.
DATA_EXTS = {".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sql", ".html", ".htm",
             ".css", ".scss", ".sass", ".less"}

# Binary documents (own size limit, own extractors)
DOC_EXTS = {".pdf", ".docx", ".pptx"}

# Images are OCR'd / captioned into text. SVG is XML text and lives in TEXT_EXTS.
IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif",
}

# Separate size limits: text parsing vs. documents vs. large image assets
MAX_TEXT_SIZE_MB = 15
MAX_DOC_SIZE_MB = 50
MAX_IMAGE_SIZE_MB = 100

# Generated-file heuristics, applied by the extractors after reading content.
MAX_AVG_LINE_LENGTH = 300
GENERATED_MARKERS = ("@generated", "auto-generated", "autogenerated", "do not edit")

# Windows file attributes
_ATTR_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_ATTR_OFFLINE = getattr(stat, "FILE_ATTRIBUTE_OFFLINE", 0x1000)
_ATTR_RECALL_ON_OPEN = 0x40000
_ATTR_RECALL_ON_DATA_ACCESS = 0x400000
_ATTR_CLOUD_PLACEHOLDER = _ATTR_OFFLINE | _ATTR_RECALL_ON_OPEN | _ATTR_RECALL_ON_DATA_ACCESS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _glob_to_regex(pattern: str) -> re.Pattern:
    """Translate a '/'-separated glob (supporting **) to a compiled regex."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
            continue
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("^" + "".join(out) + "$")


_AI_GLOBS = {k: [_glob_to_regex(g) for g in v] for k, v in AI_NOTE_DIRS.items()}
_TRANSCRIPT_RE = _glob_to_regex(AI_TRANSCRIPT_GLOB)


def _ai_dir_key(dir_name: str):
    """Return the AI_NOTE_DIRS key matching a (lowercase) directory name, or None."""
    for key in AI_NOTE_DIRS:
        if fnmatch.fnmatchcase(dir_name, key):
            return key
    return None


def _matches_any(name: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def is_secret_name(name: str) -> bool:
    name = name.lower()
    if name in SECRET_NAME_EXCEPTIONS:
        return False
    return _matches_any(name, SECRET_NAME_PATTERNS)


def is_noise_name(name: str) -> bool:
    return _matches_any(name.lower(), NOISE_NAME_PATTERNS)


def _is_cloud_placeholder(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & _ATTR_CLOUD_PLACEHOLDER)


def _is_reparse_point(path: str) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return True
    return bool(getattr(st, "st_file_attributes", 0) & _ATTR_REPARSE_POINT) or stat.S_ISLNK(st.st_mode)


def ai_note_source(file_path: str):
    """Return 'claude-plan' | 'claude-memory' | 'agent-rules' for AI-tool notes, else None."""
    parts = [p.lower() for p in Path(file_path).parts]
    name = parts[-1]
    for i, part in enumerate(parts[:-1]):
        if _ai_dir_key(part) is None:
            continue
        rel = parts[i + 1:]
        if part == ".claude":
            if "plans" in rel[:-1]:
                return "claude-plan"
            if "memory" in rel[:-1]:
                return "claude-memory"
        return "agent-rules"
    if name in AGENT_RULE_FILES:
        return "agent-rules"
    return None


def should_descend(dir_path: str, is_ignored=None) -> bool:
    """Decide whether a directory walk should enter dir_path.

    is_ignored: optional callable(dir_path) -> bool (e.g. a project's .gitignore matcher).
    """
    name = os.path.basename(os.path.normpath(dir_path)).lower()
    if not name:  # drive root
        return True
    if name in BLOCKED_DIRS:
        return False
    if name.startswith(".") and len(name) > 1 and _ai_dir_key(name) is None:
        return False
    if _is_reparse_point(dir_path):  # legacy junctions, symlink loops
        return False
    if is_ignored is not None and is_ignored(dir_path):
        return False
    return True


def is_valid_file(file_path: str, is_ignored=None) -> bool:
    """Decide whether file_path should be indexed. Cheap string checks run before any syscall."""
    try:
        path = Path(file_path)
        name = path.name.lower()
        ext = path.suffix.lower()

        # 1. Secrets denylist: first, unconditional.
        if is_secret_name(name):
            return False

        # 2. Noise: lockfiles, minified bundles, source maps.
        if is_noise_name(name):
            return False

        # 3. Extension / filename check.
        is_text = ext in TEXT_EXTS or name in TEXT_FILENAMES
        is_doc = ext in DOC_EXTS
        is_image = ext in IMAGE_EXTS
        if not (is_text or is_doc or is_image):
            return False

        # 4. Path inspection (still no syscalls).
        parts = [p.lower() for p in path.parts]
        dirs = parts[:-1]
        ai_dir_index = None
        for i, part in enumerate(dirs):
            if part in BLOCKED_DIRS:
                return False
            if part.startswith(".") and len(part) > 1:
                # Hidden dir: allowed only if it is an AI-tool dir.
                if _ai_dir_key(part) is None:
                    return False
                if ai_dir_index is None:
                    ai_dir_index = i

        if ai_dir_index is not None:
            key = _ai_dir_key(dirs[ai_dir_index])
            rel_parts = parts[ai_dir_index + 1:]
            if any(p in AI_EXCLUDED_SUBDIRS for p in rel_parts[:-1]):
                return False
            rel = "/".join(rel_parts)
            allowed = any(rx.match(rel) for rx in _AI_GLOBS[key])
            if not allowed and INDEX_AI_TRANSCRIPTS and key == ".claude":
                allowed = bool(_TRANSCRIPT_RE.match(rel))
            if not allowed:
                return False

        if is_ignored is not None and is_ignored(file_path):
            return False

        # 5. Stat once: size + cloud placeholders (reading those triggers a download).
        st = os.stat(file_path)
        if _is_cloud_placeholder(st):
            return False
        size_mb = st.st_size / (1024 * 1024)
        if st.st_size == 0:
            return False
        if is_text and size_mb > MAX_TEXT_SIZE_MB:
            return False
        if is_doc and size_mb > MAX_DOC_SIZE_MB:
            return False
        if is_image and size_mb > MAX_IMAGE_SIZE_MB:
            return False

        return True
    except (PermissionError, OSError):
        return False
