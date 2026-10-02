"""Code chunking: one chunk per function/class (tree-sitter) plus a file outline chunk.

Each chunk's symbol is its qualified name (``Class.method``), so search results can show
``file > Class.method``. Languages without a parser fall back to line-window chunks.
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeAlias

from vector_embed.core.extractors.base import (
    KIND_CODE,
    KIND_OUTLINE,
    Chunk,
    ExtractContext,
    Extractor,
    register_extractor,
)
from vector_embed.core.extractors.chunking import merge_small, read_source, split_text
from vector_embed.core.settings import ChunkingSettings

logger = logging.getLogger(__name__)

Node: TypeAlias = Any  # tree-sitter nodes are untyped
Parser: TypeAlias = Any

LANGS: dict[str, str] = {
    ".py": "python", ".ts": "typescript", ".tsx": "tsx", ".js": "javascript",
    ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".java": "java",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".swift": "swift", ".lua": "lua",
    ".sh": "bash", ".bash": "bash", ".ps1": "powershell", ".dart": "dart", ".r": "r",
}  # fmt: skip

FUNC_TYPES = {
    "function_definition", "function_declaration", "function_item", "method_definition",
    "method_declaration", "constructor_declaration", "singleton_method", "method",
    "generator_function_declaration", "local_function_statement", "function_statement",
    "init_declaration", "protocol_function_declaration", "function_signature",
    "constructor_definition", "destructor_declaration", "operator_declaration",
    "secondary_constructor", "arrow_function_declaration",
}  # fmt: skip
CLASS_TYPES = {
    "class_definition", "class_declaration", "interface_declaration", "struct_item",
    "impl_item", "trait_item", "enum_item", "enum_declaration", "struct_declaration",
    "record_declaration", "object_declaration", "module", "class", "namespace_declaration",
    "mod_item", "protocol_declaration", "trait_definition", "object_definition",
    "class_specifier", "struct_specifier", "namespace_definition", "interface_definition",
    "trait_declaration", "enum_specifier", "type_declaration", "extension_declaration",
    "class_body_declaration",
}  # fmt: skip
WRAPPER_TYPES = {"decorated_definition", "export_statement", "template_declaration"}
VAR_FUNC_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
IMPORT_HINTS = (
    "import", "use_declaration", "include", "using_directive", "package_clause",
    "namespace_use", "require",
)  # fmt: skip

_MAX_IMPORTS = 60
_OUTLINE_IMPORTS = 40
_OUTLINE_SYMBOLS = 120
_OUTLINE_CHARS = 3000
_SHORT_IMPORT_LINES = 3

_parsers: dict[str, Any] = {}


def _parser(lang: str) -> Parser | None:
    if lang not in _parsers:
        try:
            from tree_sitter_language_pack import get_parser

            _parsers[lang] = get_parser(lang)
        except Exception:  # unsupported language or missing grammar: use line chunks
            logger.debug("code: no parser for %s", lang, exc_info=True)
            _parsers[lang] = None
    return _parsers[lang]


def _text(node: Node, src: bytes) -> str:
    return str(src[node.start_byte : node.end_byte].decode("utf-8", "replace"))


def _name(node: Node, src: bytes) -> str:
    named = node.child_by_field_name("name")
    if named is not None:
        return _text(named, src)
    if node.type == "impl_item":
        target = node.child_by_field_name("type")
        return "impl " + _text(target, src) if target is not None else "impl"
    declarator = node.child_by_field_name("declarator")
    while declarator is not None:
        inner = declarator.child_by_field_name("declarator")
        if inner is None:
            break
        declarator = inner
    if declarator is not None:
        return _text(declarator, src)
    for child in node.named_children:  # go: type_declaration -> type_spec -> name
        if child.type in ("type_spec", "variable_declarator"):
            spec_name = child.child_by_field_name("name")
            if spec_name is not None:
                return _text(spec_name, src)
    return ""


def _classify(node: Node, src: bytes) -> tuple[str | None, str, Node]:
    """``(kind, symbol name, inner node)`` where kind is ``func``, ``class`` or ``None``."""
    node_type = node.type
    if node_type in WRAPPER_TYPES:
        for child in node.named_children:
            kind, name, inner = _classify(child, src)
            if kind:
                return kind, name, inner
        return None, "", node
    if node_type in FUNC_TYPES:
        return "func", _name(node, src), node
    if node_type in CLASS_TYPES:
        return "class", _name(node, src), node
    if node_type in ("lexical_declaration", "variable_declaration", "public_field_definition"):
        for child in node.named_children:
            if child.type == "variable_declarator":
                value = child.child_by_field_name("value")
                if value is not None and value.type in VAR_FUNC_VALUES:
                    return "func", _text(child.child_by_field_name("name"), src), node
    return None, "", node


class _Collector:
    def __init__(self, src: bytes, lines: list[str], cfg: ChunkingSettings) -> None:
        self.src = src
        self.lines = lines
        self.cfg = cfg
        self.units: list[Chunk] = []
        self.symbols: list[str] = []
        self.imports: list[str] = []

    def _emit(self, symbol: str, row0: int, row1: int) -> None:
        body = self.lines[row0 : row1 + 1]
        text = "\n".join(body)
        if len(text) <= self.cfg.max_chunk_chars:
            self.units.append(Chunk(text, KIND_CODE, symbol, row0 + 1, row1 + 1))
            return
        for part, (piece, start, end) in enumerate(split_text(self.cfg, body, row0 + 1), 1):
            label = f"{symbol} (part {part})" if symbol else f"(part {part})"
            self.units.append(Chunk(piece, KIND_CODE, label, start, end))

    def collect(self, parent: Node, scope: list[str]) -> None:
        loose: list[Node] = []  # consecutive non-definition nodes

        def flush() -> None:
            if not loose:
                return
            self._emit(
                ".".join(scope) if scope else "<module>",
                loose[0].start_point[0],
                loose[-1].end_point[0],
            )
            loose.clear()

        for node in parent.named_children:
            if (
                any(hint in node.type for hint in IMPORT_HINTS)
                and node.end_point[0] - node.start_point[0] < _SHORT_IMPORT_LINES
                and len(self.imports) < _MAX_IMPORTS
            ):
                self.imports.append(_text(node, self.src).strip().splitlines()[0])
            kind, name, inner = _classify(node, self.src)
            if kind is None:
                loose.append(node)
                continue
            first_row = node.start_point[0]
            # Comments directly above a definition travel with it (docstring-style).
            while loose and "comment" in loose[-1].type and loose[-1].end_point[0] >= first_row - 1:
                first_row = loose.pop().start_point[0]
            flush()
            last_row = node.end_point[0]
            full = ".".join([*scope, name]) if name else ".".join(scope) or "<anonymous>"
            self.symbols.append(("class " if kind == "class" else "") + full)
            size = sum(len(self.lines[r]) + 1 for r in range(first_row, last_row + 1))
            if kind == "class" and size > self.cfg.max_chunk_chars:
                self._split_container(
                    node, inner, first_row, [*scope, name] if name else scope, full
                )
            else:
                self._emit(full, first_row, last_row)
        flush()

    def _split_container(
        self, node: Node, inner: Node, first_row: int, scope: list[str], full: str
    ) -> None:
        body = inner.child_by_field_name("body") or inner
        first_def = next((m for m in body.named_children if _classify(m, self.src)[0]), None)
        if first_def is None:  # a big class without parseable members: plain line split
            self._emit(full, first_row, node.end_point[0])
            return
        head_end = max(first_def.start_point[0] - 1, first_row)
        if head_end >= first_row:
            self._emit(full + " (header)", first_row, head_end)
        self.collect(body, scope)


def _line_chunks(cfg: ChunkingSettings, content: str) -> list[Chunk]:
    return [
        Chunk(text, KIND_CODE, "", start, end)
        for text, start, end in split_text(cfg, content.splitlines())
    ]


def extract_code(ctx: ExtractContext, content: str, name: str, ext: str) -> list[Chunk]:
    cfg = ctx.chunking
    lang = LANGS.get(ext)
    parser = _parser(lang) if lang else None
    if parser is None:
        return _line_chunks(cfg, content)
    src = content.encode("utf-8", "replace")
    lines = content.splitlines()
    collector = _Collector(src, lines, cfg)
    try:
        collector.collect(parser.parse(src).root_node, [])
    except Exception:  # a grammar quirk must not lose the file: fall back to line chunks
        logger.debug("code: parse failed for %s", name, exc_info=True)
        return _line_chunks(cfg, content)
    if not collector.units:
        return _line_chunks(cfg, content)

    outline = [f"File: {name}", f"Language: {lang}"]
    if collector.imports:
        outline.append("Imports:\n" + "\n".join(collector.imports[:_OUTLINE_IMPORTS]))
    if collector.symbols:
        outline.append("Symbols: " + "; ".join(collector.symbols[:_OUTLINE_SYMBOLS]))
    header = Chunk("\n".join(outline)[:_OUTLINE_CHARS], KIND_OUTLINE, "<outline>", 1, len(lines))
    return [header, *merge_small(collector.units, cfg)]


@register_extractor("code")
class CodeExtractor(Extractor):
    name = "code"
    priority = 80

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.ctx.scope_settings.code_exts

    def extract(self, path: Path) -> Iterable[Chunk]:
        content = read_source(self.ctx, path)
        return extract_code(self.ctx, content, path.name, path.suffix.lower())
