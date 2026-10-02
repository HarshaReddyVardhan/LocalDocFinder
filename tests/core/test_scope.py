import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from vector_embed.core import scope
from vector_embed.core.scope import AiNoteSource, ScopePolicy, glob_to_regex
from vector_embed.core.settings import ScopeSettings

Make = Callable[..., str]

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
    "proj/data.jsonl",
]


@pytest.mark.parametrize("rel", ACCEPT)
def test_accepts(policy: ScopePolicy, make: Make, rel: str) -> None:
    assert policy.is_valid_file(make(rel)), rel


@pytest.mark.parametrize("rel", REJECT)
def test_rejects(policy: ScopePolicy, make: Make, rel: str) -> None:
    assert not policy.is_valid_file(make(rel)), rel


def test_nested_ai_dir_uses_the_outermost_allowlist(policy: ScopePolicy, make: Make) -> None:
    assert policy.is_valid_file(make(".claude/.cursor/notes.md"))
    assert not policy.is_valid_file(make(".claude/.random/notes.md"))


def test_empty_file_rejected(policy: ScopePolicy, make: Make) -> None:
    assert not policy.is_valid_file(make("proj/empty.py", ""))


def test_missing_file_is_rejected_not_raised(policy: ScopePolicy, tmp_path: Path) -> None:
    assert not policy.is_valid_file(tmp_path / "ghost.py")


def test_gitignore_callback(policy: ScopePolicy, make: Make) -> None:
    path = make("proj/gen/out.py")
    assert policy.is_valid_file(path)
    assert not policy.is_valid_file(path, is_ignored=lambda p: "gen" in p)


def test_size_limits_are_per_kind(scope_settings: ScopeSettings, make: Make) -> None:
    tiny_docs = ScopePolicy(scope_settings.model_copy(update={"max_doc_size_mb": 0.0001}))
    assert not tiny_docs.is_valid_file(make("proj/big.pdf", "x" * 1000))
    assert tiny_docs.is_valid_file(make("proj/big.py", "x" * 1000))
    tiny_images = ScopePolicy(scope_settings.model_copy(update={"max_image_size_mb": 0.0001}))
    assert not tiny_images.is_valid_file(make("proj/big.png", "x" * 1000))


def test_cloud_placeholder_is_skipped(
    policy: ScopePolicy, make: Make, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make("proj/onedrive.py")
    monkeypatch.setattr(scope, "is_cloud_placeholder", lambda _info: True)
    assert not policy.is_valid_file(path)


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [(0, False), (0x1000, True), (0x40000, True), (0x400000, True), (0x20, False)],
)
def test_is_cloud_placeholder(attributes: int, expected: bool) -> None:
    info = SimpleNamespace(st_file_attributes=attributes)
    assert scope.is_cloud_placeholder(info) is expected  # type: ignore[arg-type]


def test_app_data_dir_is_never_indexed(scope_settings: ScopeSettings, make: Make) -> None:
    path = make("appdata_clone/notes.md")
    blocked = ScopePolicy(scope_settings, blocked_roots=[Path(path).parent])
    assert not blocked.is_valid_file(path)
    assert not blocked.should_descend(Path(path).parent)
    assert blocked.is_valid_file(make("elsewhere/notes.md"))


def test_blocked_root_does_not_match_sibling_prefix(
    scope_settings: ScopeSettings, make: Make
) -> None:
    root = Path(make("data/a.md")).parent
    sibling = make("data-other/a.md")
    assert ScopePolicy(scope_settings, blocked_roots=[root]).is_valid_file(sibling)


class TestSecrets:
    @pytest.mark.parametrize(
        "name", [".env", ".env.local", "a.pem", "k.key", "id_rsa", "id_rsa.pub", ".npmrc", "x.kdbx"]
    )
    def test_secret_names(self, policy: ScopePolicy, name: str) -> None:
        assert policy.is_secret(Path("proj") / name)

    @pytest.mark.parametrize(
        "name",
        [
            "client_secret_123.apps.googleusercontent.com.json",
            "client_secret.json",
            "my-service-account.json",
            "project-firebase-adminsdk-abc12.json",
            "token.json",
            "putty.ppk",
            "id_dsa",
            "id_ecdsa",
            "id_ecdsa.pub",
            ".netrc",
            "_netrc",
            "release.jks",
            "debug.keystore",
            ".htpasswd",
            "secrets.json",
            "secrets.yaml",
            "secrets.yml",
            "secrets.toml",
        ],
    )
    def test_more_credential_files(self, policy: ScopePolicy, name: str) -> None:
        assert policy.is_secret(Path("proj") / name)

    @pytest.mark.parametrize(
        "name", ["secrets_handling.md", "tokens.py", "token_utils.json", "keystore_docs.md"]
    )
    def test_similar_names_that_are_not_credentials(self, policy: ScopePolicy, name: str) -> None:
        assert not policy.is_secret(Path("proj") / name)

    @pytest.mark.parametrize("name", [".env.example", ".env.sample", ".env.template", "main.py"])
    def test_non_secrets(self, policy: ScopePolicy, name: str) -> None:
        assert not policy.is_secret(Path("proj") / name)

    def test_claude_settings_path_glob(self, policy: ScopePolicy) -> None:
        assert policy.is_secret(Path.home() / ".claude" / "settings.json")
        assert policy.is_secret(Path.home() / ".claude" / "settings.local.json")
        assert not policy.is_secret(Path.home() / "proj" / "settings.json")

    def test_secret_beats_ai_note_allowlist(self, policy: ScopePolicy, make: Make) -> None:
        assert not policy.is_valid_file(make(".claude/plans/credentials.md"))


class TestTranscripts:
    def test_off_by_default(self, policy: ScopePolicy, make: Make) -> None:
        assert not policy.is_valid_file(make(".claude/projects/x/s.jsonl"))

    def test_opt_in_allows_only_claude_transcripts(
        self, scope_settings: ScopeSettings, make: Make
    ) -> None:
        on = ScopePolicy(scope_settings.model_copy(update={"index_ai_transcripts": True}))
        assert on.is_valid_file(make(".claude/projects/x/s.jsonl"))
        assert not on.is_valid_file(make("proj/data.jsonl"))
        assert not on.is_valid_file(make(".claude/plans/s.jsonl"))
        assert not on.is_valid_file(make(".cursor/rules/s.jsonl"))


class TestShouldDescend:
    def test_directory_rules(self, policy: ScopePolicy, tmp_path: Path) -> None:
        for name in ("src", "node_modules", ".git", ".claude", ".cursor", ".random"):
            (tmp_path / name).mkdir()
        assert policy.should_descend(tmp_path / "src")
        assert not policy.should_descend(tmp_path / "node_modules")
        assert not policy.should_descend(tmp_path / ".git")
        assert not policy.should_descend(tmp_path / ".random")
        assert policy.should_descend(tmp_path / ".claude")
        assert policy.should_descend(tmp_path / ".cursor")
        assert not policy.should_descend(tmp_path / "src", is_ignored=lambda _p: True)

    def test_drive_root_is_entered(self, policy: ScopePolicy) -> None:
        assert policy.should_descend("D:\\")

    def test_ai_excluded_subdirs_are_pruned_only_inside_ai_dirs(
        self, policy: ScopePolicy, tmp_path: Path
    ) -> None:
        (tmp_path / ".claude" / "plugins").mkdir(parents=True)
        (tmp_path / "src" / "plugins").mkdir(parents=True)
        assert not policy.should_descend(tmp_path / ".claude" / "plugins")
        assert policy.should_descend(tmp_path / "src" / "plugins")

    @pytest.mark.skipif(os.name != "nt", reason="junctions are Windows-only")
    def test_junction_is_skipped(self, policy: ScopePolicy, tmp_path: Path) -> None:
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "link"
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True
        )
        assert policy.should_descend(target)
        assert not policy.should_descend(link)


def test_unreadable_path_counts_as_reparse_point(tmp_path: Path) -> None:
    assert scope.is_reparse_point(tmp_path / "does-not-exist")


class TestAiNoteSource:
    def test_classification(self, policy: ScopePolicy) -> None:
        home = Path.home()
        assert policy.ai_note_source(home / ".claude/plans/a.md") is AiNoteSource.CLAUDE_PLAN
        assert (
            policy.ai_note_source(home / ".claude/projects/x/memory/a.md")
            is AiNoteSource.CLAUDE_MEMORY
        )
        assert policy.ai_note_source(home / ".cursor/rules/a.mdc") is AiNoteSource.AGENT_RULES
        assert policy.ai_note_source(Path("D:/p/CLAUDE.md")) is AiNoteSource.AGENT_RULES
        assert policy.ai_note_source(Path("D:/p/src/main.py")) is None

    def test_claude_dir_without_plans_or_memory_is_agent_rules(self, policy: ScopePolicy) -> None:
        assert policy.ai_note_source(Path("D:/p/.claude/skills/s.md")) is AiNoteSource.AGENT_RULES


class TestLooksGenerated:
    def test_marker_in_header(self, policy: ScopePolicy) -> None:
        assert policy.looks_generated("// @generated by protoc\nint x;\n")
        assert policy.looks_generated("# DO NOT EDIT\nx = 1\n")

    def test_marker_deep_in_file_is_ignored(self, policy: ScopePolicy) -> None:
        body = "x = 1\n" * 20 + "# do not edit this later\n"
        assert not policy.looks_generated(body)

    def test_very_long_lines(self, policy: ScopePolicy) -> None:
        assert policy.looks_generated("a" * 5000)
        assert not policy.looks_generated("short line\n" * 50)

    def test_empty_text(self, policy: ScopePolicy) -> None:
        assert not policy.looks_generated("")


@pytest.mark.parametrize(
    ("pattern", "target", "matches"),
    [
        ("**/*.md", "a.md", True),
        ("**/*.md", "x/y/a.md", True),
        ("**/*.md", "a.txt", False),
        ("rules/**", "rules/deep/a.mdc", True),
        ("*.md", "sub/a.md", False),
        ("a?c", "abc", True),
        ("a?c", "a/c", False),
    ],
)
def test_glob_to_regex(pattern: str, target: str, matches: bool) -> None:
    assert bool(glob_to_regex(pattern).match(target)) is matches


def test_default_roots_use_single_backslash() -> None:
    roots = ScopeSettings().roots
    assert "D:\\" in roots
    assert "D:\\\\" not in roots
