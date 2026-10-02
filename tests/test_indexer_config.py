import os
from pathlib import Path

import pytest

import indexer_config as cfg



ACCEPT = [
    ".claude/plans/plan.md",
    ".claude/projects/x/memory/foo.md",
    ".claude/CLAUDE.md",
    "proj/CLAUDE.md",
    "proj/AGENTS.md",
    "proj/.cursorrules",
    "proj/.cursor/rules/a.mdc",
    "proj/.github/copilot-instructions.md",
    "proj/Dockerfile",
    "proj/Makefile",
    "proj/.gitignore",
    "proj/.env.example",
    "proj/nb.ipynb",
    "proj/app.vue",
    "proj/logo.svg",
    "proj/src/main.py",
]

REJECT = [
    ".claude/.credentials.json",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".claude/projects/x/abc.jsonl",
    ".claude/plugins/foo/README.md",
    ".claude/todos/a.md",
    "proj/.env",
    "proj/.env.local",
    "proj/server.pem",
    "proj/id_rsa",
    "proj/package-lock.json",
    "proj/yarn.lock",
    "proj/dist/app.min.js",
    "proj/app.js.map",
    "proj/node_modules/pkg/.claude/notes.md",
    "proj/node_modules/pkg/index.js",
    "proj/.git/config",
    "proj/.vscode/settings.json",
    "proj/.github/workflows/ci.yml",
    "proj/notes.exe",
    "proj/.ssh/notes.md",
]


@pytest.mark.parametrize("rel", ACCEPT)
def test_accept(make, rel):
    assert cfg.is_valid_file(make(rel)), rel


@pytest.mark.parametrize("rel", REJECT)
def test_reject(make, rel):
    assert not cfg.is_valid_file(make(rel)), rel


def test_empty_file_rejected(make):
    assert not cfg.is_valid_file(make("proj/empty.py", ""))


def test_gitignore_callback(make):
    path = make("proj/gen/out.py")
    assert cfg.is_valid_file(path)
    assert not cfg.is_valid_file(path, is_ignored=lambda p: "gen" in p)


def test_doc_size_limit(make, monkeypatch):
    monkeypatch.setattr(cfg, "MAX_DOC_SIZE_MB", 0.0001)
    monkeypatch.setattr(cfg, "MAX_TEXT_SIZE_MB", 1)
    assert not cfg.is_valid_file(make("proj/big.pdf", "x" * 1000))
    assert cfg.is_valid_file(make("proj/big.py", "x" * 1000))


def test_should_descend(tmp_path):
    for d in ("src", "node_modules", ".git", ".claude", ".cursor", ".random"):
        (tmp_path / d).mkdir()
    assert cfg.should_descend(str(tmp_path / "src"))
    assert not cfg.should_descend(str(tmp_path / "node_modules"))
    assert not cfg.should_descend(str(tmp_path / ".git"))
    assert not cfg.should_descend(str(tmp_path / ".random"))
    assert cfg.should_descend(str(tmp_path / ".claude"))
    assert cfg.should_descend(str(tmp_path / ".cursor"))
    assert not cfg.should_descend(str(tmp_path / "src"), is_ignored=lambda p: True)
    assert cfg.should_descend("D:\\")


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows-only")
def test_should_descend_skips_junction(tmp_path):
    import subprocess
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                   check=True, capture_output=True)
    assert cfg.should_descend(str(target))
    assert not cfg.should_descend(str(link))


def test_ai_note_source():
    h = Path.home()
    assert cfg.ai_note_source(str(h / ".claude/plans/a.md")) == "claude-plan"
    assert cfg.ai_note_source(str(h / ".claude/projects/x/memory/a.md")) == "claude-memory"
    assert cfg.ai_note_source(str(h / ".cursor/rules/a.mdc")) == "agent-rules"
    assert cfg.ai_note_source("D:/p/CLAUDE.md") == "agent-rules"
    assert cfg.ai_note_source("D:/p/src/main.py") is None


def test_watch_root_has_single_backslash():
    assert "D:\\" in cfg.WATCH_ROOTS
    assert "D:\\\\" not in cfg.WATCH_ROOTS
