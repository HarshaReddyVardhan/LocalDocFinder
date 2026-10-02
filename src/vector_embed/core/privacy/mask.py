"""Masking: irreversible for government/financial IDs, reversible for personal details.

IDs are replaced with ``[SSN REMOVED]`` style markers and never restored: the cloud never
sees the value and no answer needs it. Personal details (name, email, phone, street address,
profile URLs) are optional: they become ``[EMAIL_1]`` placeholders that are swapped back into
the answer shown to the user. City and country are never touched; they matter for location fit.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial

from vector_embed.core.privacy.detectors import Finding, detect_sensitive


@dataclass(frozen=True)
class MaskResult:
    text: str
    findings: list[Finding]


def mask_sensitive(text: str) -> MaskResult:
    """Replace every detected ID or secret with ``[<KIND> REMOVED]``."""
    findings = detect_sensitive(text)
    out: list[str] = []
    cursor = 0
    for finding in findings:
        out.append(text[cursor : finding.start])
        out.append(f"[{finding.label} REMOVED]")
        cursor = finding.end
    out.append(text[cursor:])
    return MaskResult("".join(out), findings)


_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL = re.compile(
    r"(?:https?://)?(?:www\.)?(?:linkedin\.com/(?:in|pub)/|github\.com/|gitlab\.com/|"
    r"twitter\.com/|x\.com/|facebook\.com/|instagram\.com/)[\w./%-]+",
    re.I,
)
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{8,}\d(?![\w])")
_ADDRESS = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][\w.'-]*\s+){1,4}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b\.?",
)
_NAME_LABEL = re.compile(r"(?im)^\s*(?:full\s+)?name\s*[:\-]\s*(.+)$")
_NAME_LINE = re.compile(r"^[A-Z][a-z]+(?:[ '-][A-Z][a-z.]+){1,3}$")
_NAME_LINE_CAPS = re.compile(r"^[A-Z]{2,}(?:[ '-][A-Z][A-Z.]*){1,3}$")
_MIN_PHONE_DIGITS = 10
_NAME_SCAN_LINES = 5


_SECTION_WORDS = frozenset(
    [
        "experience",
        "education",
        "skills",
        "summary",
        "projects",
        "certifications",
        "objective",
        "profile",
        "contact",
        "languages",
        "references",
        "awards",
        "publications",
        "interests",
        "employment",
        "work",
        "professional",
        "technical",
        "technologies",
        "curriculum",
        "vitae",
        "resume",
    ]
)
# Job-title lines look like names ("Senior Software Engineer"); a person's name has none of these.
_TITLE_WORDS = frozenset(
    [
        "engineer",
        "developer",
        "manager",
        "director",
        "analyst",
        "consultant",
        "architect",
        "designer",
        "scientist",
        "administrator",
        "specialist",
        "lead",
        "intern",
        "officer",
        "president",
        "senior",
        "junior",
        "principal",
        "staff",
        "software",
        "data",
        "product",
        "project",
        "backend",
        "frontend",
        "fullstack",
        "devops",
        "cloud",
        "machine",
        "learning",
        "technical",
        "head",
        "founder",
        "associate",
        "assistant",
        "coordinator",
    ]
)
_BLOCK_START = re.compile(r"^(?:Resume \(|=== )")


def _looks_like_a_name(line: str) -> bool:
    words = {w.lower().strip(".") for w in line.split()}
    shaped = bool(_NAME_LINE.fullmatch(line) or _NAME_LINE_CAPS.fullmatch(line))
    return shaped and not words & (_SECTION_WORDS | _TITLE_WORDS)


def _heading_names(text: str) -> list[str]:
    """Names at the top of a document: its first lines, and the lines after each block header."""
    lines = [ln.strip() for ln in text.splitlines()]
    starts = [0] + [i + 1 for i, ln in enumerate(lines) if _BLOCK_START.match(ln)]
    found: list[str] = []
    for start in starts:
        window = [ln for ln in lines[start : start + _NAME_SCAN_LINES] if ln]
        found.extend(ln for ln in window if _looks_like_a_name(ln))
    return list(dict.fromkeys(found))


@dataclass
class Redaction:
    """Result of ``redact_personal``: the cloud-safe text and the way back."""

    text: str
    mapping: dict[str, str] = field(default_factory=dict)

    def restore(self, answer: str) -> str:
        """Put the original values back into text that came from the cloud."""
        for placeholder, original in self.mapping.items():
            answer = answer.replace(placeholder, original)
        return answer

    @property
    def count(self) -> int:
        return len(self.mapping)


class PersonalRedactor:
    """Reversible redaction; one instance per request so numbering stays consistent.

    Names found in any text it has seen (``learn``) are replaced everywhere, in every later call,
    so a name that appears in an early message is still masked when it recurs in a later one.
    """

    def __init__(self, known_names: Iterable[str] = ()) -> None:
        self._names_seen: dict[str, str] = {}  # lowercase -> as first written
        self._mapping: dict[str, str] = {}
        self._reverse: dict[str, str] = {}
        self._counts: dict[str, int] = {}
        for name in known_names:
            self._remember(name)

    def _remember(self, name: str) -> None:
        cleaned = " ".join(name.split())
        if cleaned:
            self._names_seen.setdefault(cleaned.lower(), cleaned)

    @property
    def known_names(self) -> list[str]:
        """Every name this redactor will replace."""
        return list(self._names_seen.values())

    @property
    def mapping(self) -> dict[str, str]:
        """Every placeholder handed out so far and the value it stands for."""
        return dict(self._mapping)

    def _placeholder(self, kind: str, original: str, key: str | None = None) -> str:
        lookup = key if key is not None else original
        existing = self._reverse.get(lookup)
        if existing is not None:
            return existing
        self._counts[kind] = self._counts.get(kind, 0) + 1
        placeholder = f"[{kind}_{self._counts[kind]}]"
        self._mapping[placeholder] = original
        self._reverse[lookup] = placeholder
        return placeholder

    def _sub(self, text: str, pattern: re.Pattern[str], kind: str) -> str:
        return pattern.sub(lambda m: self._placeholder(kind, m.group()), text)

    def _phones(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", match.group())
            if len(digits) < _MIN_PHONE_DIGITS or re.fullmatch(r"\d{4}\s*-\s*\d{4}", match.group()):
                return match.group()
            return self._placeholder("PHONE", match.group())

        return _PHONE.sub(replace, text)

    def _name_placeholder(self, name: str, _match: re.Match[str]) -> str:
        return self._placeholder("NAME", name, key=name.lower())

    def learn(self, text: str) -> None:
        """Note the names in ``text`` without changing it (call on every message first)."""
        for match in _NAME_LABEL.finditer(text):
            self._remember(match.group(1).strip())
        for name in _heading_names(text):
            self._remember(name)

    def _names(self, text: str) -> str:
        self.learn(text)
        for name in sorted(self._names_seen.values(), key=len, reverse=True):
            # Whole words only ("Ann" must not eat "Annual"), any capitalisation ("JANE DOE").
            pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
            text = pattern.sub(partial(self._name_placeholder, name), text)
        return text

    def redact(self, text: str) -> Redaction:
        text = self._sub(text, _EMAIL, "EMAIL")
        text = self._sub(text, _URL, "URL")
        text = self._phones(text)
        text = self._sub(text, _ADDRESS, "ADDRESS")
        text = self._names(text)
        return Redaction(text, dict(self._mapping))


def redact_personal(text: str, known_names: Sequence[str] = ()) -> Redaction:
    return PersonalRedactor(known_names).redact(text)
