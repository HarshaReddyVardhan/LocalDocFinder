import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from vector_embed.core import projects as projects_module
from vector_embed.core.projects import Projects
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import ScopeSettings

Make = Callable[..., str]

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture
def projects(policy: ScopePolicy, scope_settings: ScopeSettings, tmp_path: Path) -> Projects:
    return Projects(policy, scope_settings, roots=[tmp_path])


def names(paths: object) -> set[str]:
    return {p.name for p in paths}  # type: ignore[attr-defined]


class TestProjectRoot:
    def test_git_dir_marks_a_root(self, projects: Projects, make: Make, tmp_path: Path) -> None:
        (tmp_path / "app" / ".git").mkdir(parents=True)
        file = make("app/src/main.py")
        assert projects.project_root(file) == tmp_path / "app"
        assert projects.project_name(file) == "app"

    def test_marker_file_marks_a_root(self, projects: Projects, make: Make, tmp_path: Path) -> None:
        make("lib/pyproject.toml", "[project]\n")
        assert projects.project_root(make("lib/pkg/mod.py")) == tmp_path / "lib"

    def test_sln_glob_marks_a_root(self, projects: Projects, make: Make, tmp_path: Path) -> None:
        make("dotnet/App.sln", "sln")
        assert projects.project_root(make("dotnet/Program.cs")) == tmp_path / "dotnet"

    def test_nested_marker_inside_git_repo_belongs_to_the_repo(
        self, projects: Projects, make: Make, tmp_path: Path
    ) -> None:
        (tmp_path / "mono" / ".git").mkdir(parents=True)
        make("mono/web/package.json", "{}")
        assert projects.project_root(make("mono/web/a.py")) == tmp_path / "mono"

    def test_watch_root_is_not_a_project(
        self, projects: Projects, make: Make, tmp_path: Path
    ) -> None:
        make("pyproject.toml", "[project]\n")  # marker directly in the watched root
        file = make("loose/a.py")
        assert projects.project_root(file) is None
        assert projects.project_name(file) == ""

    def test_worktree_style_git_file_counts(
        self, projects: Projects, make: Make, tmp_path: Path
    ) -> None:
        make("wt/.git", "gitdir: elsewhere")
        assert projects.project_root(make("wt/a.py")) == tmp_path / "wt"
        assert projects.is_git_root(tmp_path / "wt")


class TestGitignore:
    def test_root_gitignore_matches_files_and_dirs(
        self, projects: Projects, make: Make, tmp_path: Path
    ) -> None:
        make("p/pyproject.toml", "[project]\n")
        make("p/.gitignore", "gen/\n*.log\n")
        assert projects.is_ignored(make("p/gen/out.py"))
        assert projects.is_ignored(make("p/run.log"))
        assert projects.is_ignored(tmp_path / "p" / "gen", is_dir=True)
        assert not projects.is_ignored(make("p/src/ok.py"))

    def test_nested_gitignore_is_scoped_to_its_directory(
        self, projects: Projects, make: Make
    ) -> None:
        make("p/pyproject.toml", "[project]\n")
        make("p/sub/.gitignore", "secret.py\n")
        assert projects.is_ignored(make("p/sub/secret.py"))
        assert not projects.is_ignored(make("p/other/secret.py"))

    def test_outside_a_project_nothing_is_ignored(self, projects: Projects, make: Make) -> None:
        make(".gitignore", "*.py\n")
        assert not projects.is_ignored(make("loose/a.py"))

    def test_gitignore_edits_are_picked_up_after_ttl(
        self, policy: ScopePolicy, scope_settings: ScopeSettings, make: Make, tmp_path: Path
    ) -> None:
        now = [0.0]
        proj = Projects(policy, scope_settings, roots=[tmp_path], clock=lambda: now[0])
        make("p/pyproject.toml", "[project]\n")
        ignore = Path(make("p/.gitignore", "a.py\n"))
        target = make("p/b.py")
        assert not proj.is_ignored(target)
        ignore.write_text("b.py\n", encoding="utf-8")
        assert not proj.is_ignored(target)  # still inside the TTL: cached spec
        now[0] += projects_module._GITIGNORE_TTL_SECONDS + 1
        assert proj.is_ignored(target)

    def test_unchanged_gitignore_is_not_recompiled_after_ttl(
        self, policy: ScopePolicy, scope_settings: ScopeSettings, make: Make, tmp_path: Path
    ) -> None:
        now = [0.0]
        proj = Projects(policy, scope_settings, roots=[tmp_path], clock=lambda: now[0])
        make("p/pyproject.toml", "[project]\n")
        make("p/.gitignore", "a.py\n")
        target = make("p/a.py")
        assert proj.is_ignored(target)
        first = proj._spec_for(tmp_path / "p")
        now[0] += projects_module._GITIGNORE_TTL_SECONDS + 1
        assert proj.is_ignored(target)
        assert proj._spec_for(tmp_path / "p") is first


class TestIterFiles:
    def test_plain_directory_walk_prunes_and_filters(
        self, projects: Projects, make: Make, tmp_path: Path
    ) -> None:
        make("a/main.py")
        make("a/node_modules/dep/index.js")
        make("a/.env", "SECRET=1")
        make("a/README.md", "# hi")
        make("a/pyproject.toml", "[project]\n")
        make("a/.gitignore", "build_out.py\n")
        make("a/build_out.py")
        make("notes/.claude/plans/p.md", "# plan")
        found = names(projects.iter_files())
        assert found == {"main.py", "README.md", "pyproject.toml", ".gitignore", "p.md"}

    def test_explicit_roots_override_defaults(
        self, projects: Projects, make: Make, tmp_path: Path
    ) -> None:
        make("one/a.py")
        make("two/b.py")
        assert names(projects.iter_files([tmp_path / "one"])) == {"a.py"}

    def test_unreadable_root_yields_nothing(self, projects: Projects, tmp_path: Path) -> None:
        assert list(projects.iter_files([tmp_path / "missing"])) == []

    @needs_git
    def test_git_repo_uses_git_view(self, projects: Projects, make: Make, tmp_path: Path) -> None:
        make("repo/tracked.py")
        make("repo/ignored.py")
        make("repo/.gitignore", "ignored.py\n")
        make("repo/untracked.py")
        subprocess.run(["git", "init", "-q", str(tmp_path / "repo")], check=True)
        found = names(projects.iter_files([tmp_path / "repo"]))
        assert found == {"tracked.py", "untracked.py", ".gitignore"}

    @needs_git
    def test_nested_repo_is_descended(self, projects: Projects, make: Make, tmp_path: Path) -> None:
        make("outer/a.py")
        make("outer/inner/b.py")
        subprocess.run(["git", "init", "-q", str(tmp_path / "outer")], check=True)
        subprocess.run(["git", "init", "-q", str(tmp_path / "outer" / "inner")], check=True)
        found = names(projects.iter_files([tmp_path / "outer"]))
        assert found == {"a.py", "b.py"}

    def test_falls_back_to_scandir_when_git_is_missing(
        self,
        projects: Projects,
        make: Make,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        make("repo/.git/HEAD", "ref")
        make("repo/a.py")
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        assert names(projects.iter_files([tmp_path / "repo"])) == {"a.py"}

    def test_falls_back_when_git_errors(
        self,
        projects: Projects,
        make: Make,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        make("repo/.git/HEAD", "ref")  # not a real repository, so git exits non-zero
        make("repo/a.py")
        monkeypatch.setattr(shutil, "which", lambda _name: "git")
        assert names(projects.iter_files([tmp_path / "repo"])) == {"a.py"}

    def test_falls_back_when_git_cannot_start(
        self,
        projects: Projects,
        make: Make,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        make("repo/.git/HEAD", "ref")
        make("repo/a.py")

        def boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("no git")

        monkeypatch.setattr(subprocess, "run", boom)
        assert names(projects.iter_files([tmp_path / "repo"])) == {"a.py"}


def test_unreadable_directory_is_not_a_marker(
    projects: Projects, make: Make, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: Path, pattern: str) -> None:
        raise OSError("denied")

    monkeypatch.setattr(Path, "glob", boom)
    assert projects.project_root(make("loose/a.py")) is None


def test_unreadable_gitignore_is_treated_as_absent(
    projects: Projects, make: Make, monkeypatch: pytest.MonkeyPatch
) -> None:
    make("p/pyproject.toml", "[project]\n")
    make("p/.gitignore", "a.py\n")
    target = make("p/a.py")

    def boom(self: Path, *args: object, **kwargs: object) -> str:
        raise OSError("denied")

    monkeypatch.setattr(Path, "read_text", boom)
    assert not projects.is_ignored(target)
