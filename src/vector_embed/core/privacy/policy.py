"""The privacy filter every cloud request passes through.

Always on for the cloud: government/financial IDs are masked (irreversibly). Optional: personal
details become reversible placeholders. Files under the never-send rules are kept off cloud
requests entirely. The filter also produces the exact text that will be sent, so the user can
inspect it ("View what will be sent") before consenting.
"""

import re
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vector_embed.core.privacy.detectors import Finding
from vector_embed.core.privacy.mask import PersonalRedactor, mask_sensitive
from vector_embed.core.providers.base import Message
from vector_embed.core.scope import ScopePolicy, glob_to_regex
from vector_embed.core.settings import PrivacySettings
from vector_embed.core.tokens import estimate_tokens

_MAX_LEARNED_NAMES = 200
_SOURCE_MARKERS = (
    re.compile(r"Resume \(([^)]+)\)"),
    re.compile(r"=== (.+?) ==="),
    re.compile(r"^\[\d+\] (.+)$", re.M),
)


@dataclass(frozen=True)
class SourcedFinding:
    finding: Finding
    source: str  # document the match was found in ("" when unknown)


@dataclass
class Outbound:
    """A request after filtering: the messages to send and what was changed."""

    messages: list[Message]
    findings: list[SourcedFinding] = field(default_factory=list)
    placeholders: dict[str, str] = field(default_factory=dict)  # for restoring the answer
    personal_redacted: int = 0

    @property
    def tokens(self) -> int:
        return sum(estimate_tokens(m.content) for m in self.messages)

    def restore(self, answer: str) -> str:
        """Swap personal-detail placeholders back into the answer shown to the user."""
        for placeholder, original in self.placeholders.items():
            answer = answer.replace(placeholder, original)
        return answer


def _source_at(text: str, position: int) -> str:
    """The document a position belongs to: the nearest preceding header marker."""
    best_start, best_name = -1, ""
    for marker in _SOURCE_MARKERS:
        for match in marker.finditer(text, 0, position):
            if match.start() > best_start:
                best_start, best_name = match.start(), match.group(1).strip()
    return best_name


DocTypeLookup = Callable[[Sequence[str]], dict[str, str]]


class PrivacyFilter:
    def __init__(
        self,
        settings: PrivacySettings,
        scope: ScopePolicy,
        doc_types: DocTypeLookup | None = None,
    ) -> None:
        self._settings = settings
        self._scope = scope
        self._doc_types = doc_types  # path -> stored document type, for never_send_doc_types
        self._never_send = tuple(glob_to_regex(g) for g in settings.never_send_globs)
        # Names met in earlier requests: a name restored into an answer is real text in the next
        # turn's history, and must still be masked there.
        self._learned_names: dict[str, str] = {}
        self._names_lock = threading.Lock()

    # ------------------------------------------------------------------ never-send rules
    def is_never_send(self, path: str | Path, doc_type: str | None = None) -> bool:
        """True for secrets, files matching the never-send globs, and blocked document types."""
        if self._scope.is_secret(path):
            return True
        if doc_type is None:
            doc_type = self._stored_doc_type(path)
        if doc_type is not None and doc_type in self._settings.never_send_doc_types:
            return True
        posix = Path(path).as_posix().lower()
        return any(regex.match(posix) for regex in self._never_send)

    def _stored_doc_type(self, path: str | Path) -> str | None:
        """The indexed type of ``path``, looked up only when a type is actually blocked."""
        if not self._settings.never_send_doc_types or self._doc_types is None:
            return None
        key = str(path)
        return self._doc_types([key]).get(key)

    # ------------------------------------------------------------------ filtering
    def prepare(self, messages: list[Message]) -> Outbound:
        """Mask IDs everywhere; optionally redact personal details. Never alters the originals."""
        outbound = Outbound(messages=[])
        redactor = self._new_redactor(messages)
        for message in messages:
            masked = mask_sensitive(message.content)
            outbound.findings.extend(
                SourcedFinding(f, _source_at(message.content, f.start)) for f in masked.findings
            )
            content = masked.text
            if redactor is not None:
                content = redactor.redact(content).text
            outbound.messages.append(Message(message.role, content))
        if redactor is not None:
            outbound.placeholders = redactor.mapping
            outbound.personal_redacted = len(outbound.placeholders)
            self._remember_names(redactor)
        return outbound

    def _new_redactor(self, messages: list[Message]) -> PersonalRedactor | None:
        """A redactor that already knows every name in this request and in earlier ones."""
        if not self._settings.redact_personal:
            return None
        with self._names_lock:
            earlier = list(self._learned_names.values())
        redactor = PersonalRedactor([*self._settings.known_names, *earlier])
        for message in messages:  # first pass: a name in a late message still masks an early one
            redactor.learn(message.content)
        return redactor

    def _remember_names(self, redactor: PersonalRedactor) -> None:
        with self._names_lock:
            for name in redactor.known_names:
                if len(self._learned_names) >= _MAX_LEARNED_NAMES:
                    break
                self._learned_names.setdefault(name.lower(), name)

    def forget_names(self) -> None:
        """Drop the names remembered between requests (a new chat starts clean)."""
        with self._names_lock:
            self._learned_names.clear()

    # ------------------------------------------------------------------ transparency
    @staticmethod
    def preview(outbound: Outbound) -> str:
        """Exactly the text that will be sent, message by message."""
        return "\n\n".join(f"--- {m.role} ---\n{m.content}" for m in outbound.messages)

    @staticmethod
    def shield_note(outbound: Outbound) -> str:
        """``🛡 2 sensitive items will be masked: Passport (a.pdf), DL (b.docx)`` or ``""``."""
        if not outbound.findings:
            return ""
        parts = []
        for item in outbound.findings:
            where = f" ({item.source})" if item.source else ""
            parts.append(f"{item.finding.label.title()}{where}")
        counts = Counter(parts)
        listed = ", ".join(f"{name} x{n}" if n > 1 else name for name, n in counts.items())
        total = len(outbound.findings)
        noun = "item" if total == 1 else "items"
        return f"🛡 {total} sensitive {noun} will be masked: {listed}"

    @staticmethod
    def badge(outbound: Outbound, destination: str, excerpts: int | None = None) -> str:
        """``☁ Sending 6 excerpts (≈3.1k tokens) to OpenRouter / model``."""
        count = excerpts if excerpts is not None else len(outbound.messages)
        tokens = outbound.tokens
        size = f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)
        noun = "excerpt" if count == 1 else "excerpts"
        return f"☁ Sending {count} {noun} (≈{size} tokens) to {destination}"
