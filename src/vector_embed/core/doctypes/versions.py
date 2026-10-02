"""Version groups: ``Resume_v1.pdf``, ``Resume_final.docx`` and ``resume (2).pdf`` are one document.

Documents of a versioned type whose text is more than ``similarity`` alike (Jaccard over word
shingles) are grouped; matching then shows the newest member of each group by default.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import xxhash

_SHINGLE = 5
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class VersionCandidate:
    path: str
    doc_type: str
    text: str
    modified_at: int


def shingles(text: str, size: int = _SHINGLE) -> frozenset[int]:
    """Hashed word n-grams; texts shorter than ``size`` words use their words as shingles."""
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < size:
        grams = [" ".join(words)] if words else []
    else:
        grams = [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]
    return frozenset(xxhash.xxh32_intdigest(g.encode()) for g in grams)


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def group_versions(
    items: Sequence[VersionCandidate],
    versioned_types: frozenset[str],
    similarity: float = 0.9,
) -> dict[str, str]:
    """Map each grouped path to a stable group id; ungrouped paths are absent from the result."""
    result: dict[str, str] = {}
    for doc_type in sorted({i.doc_type for i in items} & versioned_types):
        members = [i for i in items if i.doc_type == doc_type]
        sets = [shingles(m.text) for m in members]
        order = sorted(range(len(members)), key=lambda i: len(sets[i]))
        union = _UnionFind(len(members))
        for pos, i in enumerate(order):
            for j in order[pos + 1 :]:
                # Jaccard >= s implies the smaller set is at least s times the larger one,
                # so once sizes diverge further than that no later pair can match.
                if len(sets[i]) < similarity * len(sets[j]):
                    break
                if jaccard(sets[i], sets[j]) >= similarity:
                    union.union(i, j)
        clusters: dict[int, list[int]] = {}
        for index in range(len(members)):
            clusters.setdefault(union.find(index), []).append(index)
        for indices in clusters.values():
            if len(indices) < 2:
                continue
            anchor = min(members[i].path.lower() for i in indices)
            group = "vg_" + xxhash.xxh3_64_hexdigest(anchor.encode())[:12]
            for i in indices:
                result[members[i].path] = group
    return result


def newest_per_group(items: Sequence[VersionCandidate], groups: dict[str, str]) -> list[str]:
    """Paths to show by default: every ungrouped path plus the newest member of each group."""
    newest: dict[str, VersionCandidate] = {}
    shown: list[str] = []
    for item in items:
        group = groups.get(item.path)
        if group is None:
            shown.append(item.path)
        elif group not in newest or item.modified_at > newest[group].modified_at:
            newest[group] = item
    return shown + [c.path for c in newest.values()]
