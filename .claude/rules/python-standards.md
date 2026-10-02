# Python standards (team baseline)

## Environment and dependencies
- One manager (`uv` preferred; `pip` + venv acceptable until `uv` is installed). Commit the lockfile (`uv.lock`) when using `uv`.
- Runtime pinned in `.python-version`; `requires-python = ">=3.11"` in `pyproject.toml`.
- Never install into the global interpreter. Work inside `.venv`.
- Dependencies are declared in `pyproject.toml` only: runtime in `[project.dependencies]`, tooling in a `dev` group.

## Formatting, linting, typing
- `ruff format` (line length 100) and `ruff check` replace black/isort/flake8. Config lives in `pyproject.toml`.
- Type hints on every function signature; `mypy --strict` on `src/`. Do not use `Any` or `# type: ignore` without a one-line reason.
- `pre-commit` runs ruff (lint+format), mypy and whitespace/EOF fixers on every commit.

## Layout
- `src/vector_embed/` package (src layout). Tests in `tests/` mirror the package tree.
- All tool config (ruff, pytest, mypy, coverage) is in `pyproject.toml`. No stray dotfiles for tool config.
- Absolute imports for project code. No wildcard imports. No import-time side effects (no I/O, no model loads, no network).

## Testing
- `pytest` only. Prefer function tests and fixtures over classes; use `tmp_path`, never the real home directory or `%LOCALAPPDATA%`.
- Coverage floor: 80% overall (`--cov-fail-under=80`); security/privacy/power/scope modules aim for ~100%.
- Mock external boundaries (Ollama, OpenAI-compatible HTTP, Windows APIs, `psutil`, `pynvml`). Tests are deterministic and offline by default; real-Ollama tests carry a marker and skip when the model is absent.
- Every bug fix adds a regression test first.

## Configuration and secrets
- 12-factor: environment-dependent values come from env vars or `settings.toml`, validated at startup with `pydantic` / `pydantic-settings`.
- API keys go to Windows Credential Manager via `keyring`; never in files, logs or the index.
- `.env.example` lists every variable with a non-sensitive default.

## Logging
- No `print()` in library code (CLI output is the only exception, via a dedicated output helper).
- `logging.getLogger(__name__)`; pass context as structured fields (`extra=`), not string interpolation; use `%s`-style lazy args.
- Never log secrets, document contents or masked values.

## Git
- Conventional Commits. Small focused commits; each leaves the tree green.
- No `Co-Authored-By` or AI-attribution trailers.
- CI (when a remote exists) runs ruff, mypy, pytest with coverage, and `pip-audit`.
