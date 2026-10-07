"""Write the release notes (Markdown) for a version from the commits since the last tag.

The notes are packed into the release by ``vpk pack --releaseNotes``, which makes them the body of
the GitHub Release and what the app shows in "What's new" when it downloads the update.

    python scripts/release_notes.py 0.4.0 notes.md
"""

import re
import subprocess
import sys
from pathlib import Path

# commit type -> heading; types not listed (chore, test, ci, build, style) are not user-visible
SECTIONS = {"feat": "New", "fix": "Fixed", "perf": "Faster", "refactor": "Improved"}
SUBJECT = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?(?P<breaking>!)?:\s*(?P<text>.+)$")
FALLBACK = "Maintenance release: no user-facing changes."


def parse(subjects: list[str]) -> dict[str, list[str]]:
    """Group commit subjects by section heading, in the order of ``SECTIONS``."""
    grouped: dict[str, list[str]] = {heading: [] for heading in SECTIONS.values()}
    for subject in subjects:
        match = SUBJECT.match(subject.strip())
        if not match or match["type"] not in SECTIONS:
            continue
        text = match["text"].strip()
        grouped[SECTIONS[match["type"]]].append(text[:1].upper() + text[1:])
    return {heading: items for heading, items in grouped.items() if items}


def render(version: str, subjects: list[str]) -> str:
    grouped = parse(subjects)
    lines = [f"# LocalDoc Finder {version}", ""]
    if not grouped:
        return "\n".join([*lines, FALLBACK, ""])
    for heading, items in grouped.items():
        lines += [f"## {heading}", "", *(f"- {item}" for item in items), ""]
    return "\n".join(lines)


def previous_tag(current: str) -> str | None:
    """The newest tag other than ``current`` that is reachable from HEAD."""
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--exclude", current, "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def commit_subjects(since: str | None) -> list[str]:
    rev = f"{since}..HEAD" if since else "HEAD"
    result = subprocess.run(
        ["git", "log", "--no-merges", "--format=%s", rev],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write(__doc__ or "")
        return 2
    version, output = argv[1], Path(argv[2])
    notes = render(version, commit_subjects(previous_tag(f"v{version}")))
    output.write_text(notes, encoding="utf-8")
    sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
