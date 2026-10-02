# Vector_Embed

Local semantic search + chat-with-documents engine for Windows (RTX 2070 8 GB), Ollama-backed.
The full design and build order live in @.claude/PLAN.md. Follow its build order (§10a), one step at a time.

## Working agreement
- **Commit after every task**, once its tests and checks pass. One logical change per commit.
- **Never add `Co-Authored-By` lines or any AI-agent attribution** to commits, PR bodies, code or docs.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `test:`, `refactor:`, `chore:`, `docs:`).
- Work on `main` for the initial build; use short-lived `feat/`, `fix/`, `chore/` branches once there is a CI-gated remote.
- Never commit `.env`, keys, tokens or local data. `.env.example` documents every variable.
- Before saying a task is done: `ruff format`, `ruff check`, `mypy`, `pytest` must all pass.

## Rules (read before writing code)
- @.claude/rules/python-standards.md — tooling, layout, typing, testing, config, logging, git
- @.claude/rules/design-principles.md — SOLID, KISS/DRY/YAGNI, error handling, project-specific invariants

## Commands
```powershell
.venv\Scripts\python -m pytest            # tests + coverage
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check . --fix
.venv\Scripts\mypy
.venv\Scripts\pre-commit run --all-files
```

## Layout and migration status
- New code lives in `src/vector_embed/` (src layout, per the standards). The plan's `core/`, `app/` and `cli.py` map to `src/vector_embed/core/`, `.../app/`, `.../cli.py`.
- The flat prototype at the repo root (`indexer_config.py`, `store.py`, `worker.py`, `watcher.py`, `search.py`, `power.py`, `projects.py`, `embedder.py`, `extract/`, `ui/`, `eval/` and their tests) still runs and is excluded from ruff/mypy. Delete each piece when its rebuilt replacement lands (plan steps 3-5); do not edit it beyond what is needed to keep its tests green.
- Done: step 1 (`core/settings.py`, `core/scope.py`, `core/projects.py`). Next: step 2 (`core/store/`, `core/providers/ollama.py`, `core/models/registry.py`).
- Tests that walk real directories use `tmp_path`; the legacy root `conftest.py` relocates it outside `%TEMP%` because `appdata` is a blocked dir. New scope tests use the `policy` fixture in `tests/core/conftest.py` instead.

## Project invariants (never break; they come from the plan)
- Indexing never runs on battery. VRAM returns to 0 after work. The embedder and chat LLM are never on the GPU together.
- Secrets are never indexed and never sent to a cloud provider; government/financial IDs are always masked before any cloud call.
- `core/` has no UI imports. Front-ends (app, cli, mcp) are thin.
- Extend by registry (`@register`), not by editing core switch statements.
