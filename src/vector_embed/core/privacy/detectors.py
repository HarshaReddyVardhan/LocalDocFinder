"""Detect government and financial identifiers and secrets in text.

Each detector combines a format regex with a checksum (Luhn, Verhoeff, IBAN mod 97, ABA) or a
nearby label ("SSN", "Passport No", "DL#") so ordinary numbers - years, phone numbers, GPAs,
zip codes - are not masked. Overlapping hits are resolved in favour of the longest match.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

_LABEL_WINDOW = 40  # characters before a number in which its label must appear


@dataclass(frozen=True)
class Finding:
    kind: str
    start: int
    end: int
    text: str

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)


KIND_LABELS = {
    "ssn": "SSN",
    "passport": "PASSPORT",
    "drivers_license": "DRIVERS LICENSE",
    "aadhaar": "AADHAAR",
    "pan": "PAN",
    "ni_number": "NI NUMBER",
    "sin": "SIN",
    "tax_id": "TAX ID",
    "card": "CARD",
    "iban": "IBAN",
    "bank_account": "BANK ACCOUNT",
    "routing": "ROUTING NUMBER",
    "secret": "SECRET",
}

# --------------------------------------------------------------------------- checksums
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5), (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7), (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3), (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)  # fmt: skip
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4), (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7), (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)  # fmt: skip


def luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0 and len(digits) > 1


def verhoeff_valid(digits: str) -> bool:
    check = 0
    for index, char in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[index % 8][int(char)]]
    return check == 0


def iban_valid(value: str) -> bool:
    compact = value.replace(" ", "").upper()
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    number = "".join(str(int(c, 36)) for c in rearranged)
    return int(number) % 97 == 1


def aba_valid(digits: str) -> bool:
    if len(digits) != 9:
        return False
    weights = (3, 7, 1) * 3
    return sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 10 == 0


# --------------------------------------------------------------------------- detectors
def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _has_label(text: str, start: int, pattern: re.Pattern[str]) -> bool:
    return bool(pattern.search(text[max(0, start - _LABEL_WINDOW) : start]))


_SSN_LABEL = re.compile(r"\b(ssn|social\s+security)\b", re.I)
_SIN_LABEL = re.compile(r"\b(sin|social\s+insurance)\b", re.I)
_PASSPORT_LABEL = re.compile(r"passport(?:\s*(?:no\.?|number|num|#))?\s*[:#-]?\s*$", re.I)
_DL_LABEL = re.compile(
    r"(driver'?s?\s+licen[cs]e|driving\s+licen[cs]e|licen[cs]e\s*(?:no\.?|number|#)|"
    r"\bdl\b\s*(?:no\.?|number|#)?)(?:\s*(?:no\.?|number|num|#))?"
    r"\s*[:#-]?\s*$",
    re.I,
)
_AADHAAR_LABEL = re.compile(r"\b(aadhaar|aadhar|uidai|uid)\b[^\n]{0,15}$", re.I)
_PAN_LABEL = re.compile(r"\b(pan|permanent\s+account(?:\s+number)?)\b[^\n]{0,15}$", re.I)
_NI_LABEL = re.compile(
    r"\b(ni|nino|national\s+insurance)\b(?:\s+(?:no\.?|number|num|#))?[^\n]{0,10}$", re.I
)
_TAX_LABEL = re.compile(r"\b(tin|ein|itin|tax\s+id|tax\s+identification)\b[^\n]{0,15}$", re.I)
_ACCOUNT_LABEL = re.compile(
    r"(account\s*(?:no\.?|number|num|#)|acct\.?\s*(?:no\.?|#)|a/c)[^\n]{0,10}$", re.I
)
_ROUTING_LABEL = re.compile(r"\b(routing|aba)\b[^\n]{0,15}$", re.I)

_SSN = re.compile(r"\b(\d{3})\s?[-. ]\s?(\d{2})\s?[-. ]\s?(\d{4})\b")  # dashes, dots or spaces
_NINE_DIGITS = re.compile(r"\b\d{9}\b")
_CARD = re.compile(
    r"\b(?:\d{4}[ -]?){3}\d{1,7}\b"  # 4-4-4-4 style, up to 19 digits
    r"|\b3[47]\d{2}[ -]?\d{6}[ -]?\d{5}\b"  # American Express 4-6-5
    r"|\b\d{13,19}\b"  # unseparated
)
# Issuer prefixes: Visa, Mastercard, Amex, Discover, JCB, Diners. Keeps ISBNs and order ids out.
_CARD_PREFIX = re.compile(r"^(4|5[1-5]|2[2-7]|3[0-9]|6011|65|64[4-9])")
_AADHAAR = re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b")
_PAN = re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b")
_NI = re.compile(r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b")
_SIN_NUMBER = re.compile(r"\b\d{3}[ -]?\d{3}[ -]?\d{3}\b")
# "K1234567" or "K 1234567" (a letter prefix may be spaced off); "no 123..." is the label's "no".
_PASSPORT_VALUE = re.compile(r"\b(?:(?!no\b)[A-Z]{1,2} \d{6,9}|[A-Z0-9]{6,9})\b", re.I)
_DL_VALUE = re.compile(r"\b[A-Z0-9][A-Z0-9 -]{4,18}[A-Z0-9]\b")
_TAX_VALUE = re.compile(r"\b(?:\d{2}-\d{7}|\d{3}-\d{2}-\d{4}|\d{9})\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
_ACCOUNT_VALUE = re.compile(r"\b\d{8,17}\b")
_SECRET = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    r"|\bsk-[A-Za-z0-9_-]{20,}\b|\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bAKIA[0-9A-Z]{16}\b"
    r"|\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b|\bAIza[0-9A-Za-z_-]{30,}\b"
    r"|\bgithub_pat_[A-Za-z0-9_]{20,}\b|\bglpat-[A-Za-z0-9_-]{20,}\b|\bhf_[A-Za-z0-9]{30,}\b"
    r"|\bAccountKey=[A-Za-z0-9+/=]{20,}"
    r"|\bxox[baprs]-[A-Za-z0-9-]{10,}\b|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    r"|(?i:\b(?:api[_-]?key|secret|token|password|passwd)\b)\s*[:=]\s*['\"]?[^\s'\"]{8,}"
)


def _after_label(
    text: str, label: re.Pattern[str], value: re.Pattern[str], kind: str, needs_digit: bool = True
) -> list[Finding]:
    """Values that directly follow a label (``label`` must end at the value's start)."""
    found: list[Finding] = []
    for match in value.finditer(text):
        if not _has_label(text, match.start(), label):
            continue
        if needs_digit and not any(c.isdigit() for c in match.group()):
            continue
        found.append(Finding(kind, match.start(), match.end(), match.group()))
    return found


def _ssn(text: str) -> list[Finding]:
    found: list[Finding] = []
    for m in _SSN.finditer(text):
        area, group, serial = m.groups()
        if area in {"000", "666"} or area.startswith("9") or group == "00" or serial == "0000":
            continue
        found.append(Finding("ssn", m.start(), m.end(), m.group()))
    for m in _NINE_DIGITS.finditer(text):
        if _has_label(text, m.start(), _SSN_LABEL):
            found.append(Finding("ssn", m.start(), m.end(), m.group()))
    return found


def _cards(text: str) -> list[Finding]:
    found: list[Finding] = []
    for m in _CARD.finditer(text):
        digits = _digits(m.group())
        plausible = 13 <= len(digits) <= 19 and _CARD_PREFIX.match(digits) is not None
        if plausible and luhn_valid(digits) and len(set(digits)) > 1:
            found.append(Finding("card", m.start(), m.end(), m.group()))
    return found


def _aadhaar(text: str) -> list[Finding]:
    return [
        Finding("aadhaar", m.start(), m.end(), m.group())
        for m in _AADHAAR.finditer(text)
        if verhoeff_valid(_digits(m.group())) and _has_label(text, m.start(), _AADHAAR_LABEL)
    ]


def _regex(kind: str, pattern: re.Pattern[str]) -> Callable[[str], list[Finding]]:
    def detect(text: str) -> list[Finding]:
        return [Finding(kind, m.start(), m.end(), m.group()) for m in pattern.finditer(text)]

    return detect


def _sin(text: str) -> list[Finding]:
    found: list[Finding] = []
    for m in _SIN_NUMBER.finditer(text):
        digits = _digits(m.group())
        if _has_label(text, m.start(), _SIN_LABEL) and luhn_valid(digits):
            found.append(Finding("sin", m.start(), m.end(), m.group()))
    return found


def _iban(text: str) -> list[Finding]:
    return [
        Finding("iban", m.start(), m.end(), m.group())
        for m in _IBAN.finditer(text)
        if iban_valid(m.group())
    ]


def _routing(text: str) -> list[Finding]:
    return [
        f for f in _after_label(text, _ROUTING_LABEL, _NINE_DIGITS, "routing") if aba_valid(f.text)
    ]


_DETECTORS: tuple[Callable[[str], list[Finding]], ...] = (
    _ssn,
    _cards,
    _aadhaar,
    lambda t: _after_label(t, _PAN_LABEL, _PAN, "pan", needs_digit=False),
    lambda t: _after_label(t, _NI_LABEL, _NI, "ni_number", needs_digit=False),
    _sin,
    lambda t: _after_label(t, _PASSPORT_LABEL, _PASSPORT_VALUE, "passport"),
    lambda t: _after_label(t, _DL_LABEL, _DL_VALUE, "drivers_license"),
    lambda t: _after_label(t, _TAX_LABEL, _TAX_VALUE, "tax_id"),
    _iban,
    lambda t: _after_label(t, _ACCOUNT_LABEL, _ACCOUNT_VALUE, "bank_account"),
    _routing,
    _regex("secret", _SECRET),
)


def detect_sensitive(text: str) -> list[Finding]:
    """Non-overlapping findings in order of position.

    Findings that overlap are merged into one span covering all of them, so no digit of either
    is left behind; the merged span keeps the kind of its longest member.
    """
    candidates = [f for detector in _DETECTORS for f in detector(text)]
    candidates.sort(key=lambda f: (f.start, -(f.end - f.start)))
    merged: list[Finding] = []
    for finding in candidates:
        if not merged or finding.start >= merged[-1].end:
            merged.append(finding)
            continue
        last = merged[-1]
        end = max(last.end, finding.end)
        kind = finding.kind if finding.end - finding.start > last.end - last.start else last.kind
        merged[-1] = Finding(kind, last.start, end, text[last.start : end])
    return merged
