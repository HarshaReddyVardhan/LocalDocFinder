"""Code chunking: one chunk per function/class (tree-sitter), plus a file outline chunk."""
import os
from typing import Dict, List, Optional

import indexer_config as cfg
from . import Chunk, merge_small, split_by_lines

LANGS: Dict[str, str] = {
    ".py": "python", ".ts": "typescript", ".tsx": "tsx", ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript", ".java": "java", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".cs": "csharp", ".go": "go",
    ".rs": "rust", ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala", ".swift": "swift", ".lua": "lua", ".sh": "bash", ".bash": "bash",
    ".ps1": "powershell", ".dart": "dart", ".r": "r",
}

FUNC_TYPES = {
    "function_definition", "function_declaration", "function_item", "method_definition",
    "method_declaration", "constructor_declaration", "singleton_method", "method",
    "generator_function_declaration", "local_function_statement", "function_statement",
    "init_declaration", "protocol_function_declaration", "function_signature",
    "constructor_definition", "destructor_declaration", "operator_declaration",
    "secondary_constructor", "arrow_function_declaration",
}
CLASS_TYPES = {
    "class_definition", "class_declaration", "interface_declaration", "struct_item", "impl_item",
    "trait_item", "enum_item", "enum_declaration", "struct_declaration", "record_declaration",
    "object_declaration", "module", "class", "namespace_declaration", "mod_item",
    "protocol_declaration", "trait_definition", "object_definition", "class_specifier",
    "struct_specifier", "namespace_definition", "interface_definition", "trait_declaration",
    "enum_specifier", "type_declaration", "extension_declaration", "class_body_declaration",
}
WRAPPER_TYPES = {"decorated_definition", "export_statement", "template_declaration"}
VAR_FUNC_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
IMPORT_HINTS = ("import", "use_declaration", "include", "using_directive", "package_clause",
                "namespace_use", "require")

_parsers: Dict[str, object] = {}


def _parser(lang: str):
    if lang not in _parsers:
        try:
            from tree_sitter_language_pack import get_parser
            _parsers[lang] = get_parser(lang)
        except Exception:
            _parsers[lang] = None
    return _parsers[lang]


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _name(node, src: bytes) -> str:
    n = node.child_by_field_name("name")
    if n is not None:
        return _text(n, src)
    if node.type == "impl_item":
        t = node.child_by_field_name("type")
        return "impl " + _text(t, src) if t is not None else "impl"
    d = node.child_by_field_name("declarator")
    while d is not None:
        nxt = d.child_by_field_name("declarator")
        if nxt is None:
            break
        d = nxt
    if d is not None:
        return _text(d, src)
    for c in node.named_children:  # go type_declaration -> type_spec -> name
        if c.type in ("type_spec", "variable_declarator"):
            nn = c.child_by_field_name("name")
            if nn is not None:
                return _text(nn, src)
    return ""


def _classify(node, src: bytes):
    """Return (kind, symbol_name, inner_node) where kind in 'func' | 'class' | None."""
    t = node.type
    if t in WRAPPER_TYPES:
        for c in node.named_children:
            k, nm, inner = _classify(c, src)
            if k:
                return k, nm, inner
        return None, "", node
    if t in FUNC_TYPES:
        return "func", _name(node, src), node
    if t in CLASS_TYPES:
        return "class", _name(node, src), node
    if t in ("lexical_declaration", "variable_declaration", "public_field_definition"):
        for c in node.named_children:
            if c.type == "variable_declarator":
                v = c.child_by_field_name("value")
                if v is not None and v.type in VAR_FUNC_VALUES:
                    return "func", _text(c.child_by_field_name("name"), src), node
    return None, "", node


class _Collector:
    def __init__(self, src: bytes, lines: List[str]):
        self.src = src
        self.lines = lines
        self.units: List[Chunk] = []
        self.symbols: List[str] = []
        self.imports: List[str] = []

    def _emit(self, symbol: str, row0: int, row1: int, split_ok: bool = True):
        body = self.lines[row0:row1 + 1]
        text = "\n".join(body)
        if len(text) <= cfg.MAX_CHUNK_CHARS:
            self.units.append(Chunk(text, "code", symbol, row0 + 1, row1 + 1))
            return
        for i, (t, s, e) in enumerate(split_by_lines(body, row0 + 1), 1):
            self.units.append(Chunk(t, "code", f"{symbol} (part {i})" if symbol else f"(part {i})", s, e))

    def collect(self, parent, scope: List[str]):
        loose: List = []  # consecutive non-definition nodes

        def flush():
            if not loose:
                return
            r0, r1 = loose[0].start_point[0], loose[-1].end_point[0]
            self._emit(".".join(scope) if scope else "<module>", r0, r1)
            loose.clear()

        for node in parent.named_children:
            if any(h in node.type for h in IMPORT_HINTS) and node.end_point[0] - node.start_point[0] < 3:
                if len(self.imports) < 60:
                    self.imports.append(_text(node, self.src).strip().splitlines()[0])
            kind, nm, inner = _classify(node, self.src)
            if kind is None:
                loose.append(node)
                continue
            # Attach directly preceding comments (docstring-style) to the definition.
            r0 = node.start_point[0]
            while loose and "comment" in loose[-1].type and loose[-1].end_point[0] >= r0 - 1:
                r0 = loose.pop().start_point[0]
            flush()
            r1 = node.end_point[0]
            full = ".".join(scope + [nm]) if nm else ".".join(scope) or "<anonymous>"
            self.symbols.append(("class " if kind == "class" else "") + full)
            size = sum(len(self.lines[r]) + 1 for r in range(r0, r1 + 1))
            if kind == "class" and size > cfg.MAX_CHUNK_CHARS:
                self._split_container(node, inner, r0, scope + [nm] if nm else scope, full)
            else:
                self._emit(full, r0, r1)
        flush()

    def _split_container(self, node, inner, r0: int, scope: List[str], full: str):
        body = inner.child_by_field_name("body") or inner
        members = list(body.named_children)
        first_def: Optional[object] = None
        for m in members:
            if _classify(m, self.src)[0]:
                first_def = m
                break
        if first_def is None:  # big class with no parseable members -> line split
            self._emit(full, r0, node.end_point[0])
            return
        head_end = max(first_def.start_point[0] - 1, r0)
        if head_end >= r0:
            self._emit(full + " (header)", r0, head_end)
        self.collect(body, scope)


def _line_chunks(content: str, kind: str = "doc") -> List[Chunk]:
    lines = content.splitlines()
    return [Chunk(t, kind, "", s, e) for t, s, e in split_by_lines(lines, 1)]


def extract_code(content: str, path: str, ext: str) -> List[Chunk]:
    lang = LANGS.get(ext)
    parser = _parser(lang) if lang else None
    name = os.path.basename(path)
    if parser is None:
        return _line_chunks(content, "code")
    src = content.encode("utf-8", "replace")
    try:
        tree = parser.parse(src)
        lines = content.splitlines()
        col = _Collector(src, lines)
        col.collect(tree.root_node, [])
    except Exception:
        return _line_chunks(content, "code")
    if not col.units:
        return _line_chunks(content, "code")

    units = merge_small(col.units)
    outline = [f"File: {name}", f"Language: {lang}"]
    if col.imports:
        outline.append("Imports:\n" + "\n".join(col.imports[:40]))
    if col.symbols:
        outline.append("Symbols: " + "; ".join(col.symbols[:120]))
    out_text = "\n".join(outline)[:3000]
    return [Chunk(out_text, "outline", "<outline>", 1, len(lines))] + units
