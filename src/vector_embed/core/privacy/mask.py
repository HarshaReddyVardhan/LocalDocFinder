"""Masking: irreversible for government/financial IDs, reversible for personal details.

IDs are replaced with ``[SSN REMOVED]`` style markers and never restored: the cloud never
sees the value and no answer needs it. Personal details (name, email, phone, street address,
profile URLs) are optional: they become ``[EMAIL_1]`` placeholders that are swapped back into
the answer shown to the user. City and country are never touched; they matter for location fit.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

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
_MIN_PHONE_DIGITS = 10
_NAME_SCAN_LINES = 5


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
    """Reversible redaction; one instance per request so numbering stays consistent."""

    def __init__(self, known_names: Iterable[str] = ()) -> None:
        self._known = [n for n in known_names if n.strip()]
        self._mapping: dict[str, str] = {}
        self._reverse: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def _placeholder(self, kind: str, original: str) -> str:
        existing = self._reverse.get(original)
        if existing is not None:
            return existing
        self._counts[kind] = self._counts.get(kind, 0) + 1
        placeholder = f"[{kind}_{self._counts[kind]}]"
        self._mapping[placeholder] = original
        self._reverse[original] = placeholder
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

    def _names(self, text: str) -> str:
        names = list(self._known)
        for match in _NAME_LABEL.finditer(text):
            names.append(match.group(1).strip())
        lines = [ln.strip() for ln in text.splitlines()[:_NAME_SCAN_LINES] if ln.strip()]
        if lines and _NAME_LINE.fullmatch(lines[0]):
            names.append(lines[0])
        for name in sorted(set(names), key=len, reverse=True):
            placeholder = self._placeholder("NAME", name)
            text = text.replace(name, placeholder)
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
