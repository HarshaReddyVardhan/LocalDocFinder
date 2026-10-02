# Vector_Embed

Local semantic search + chat-with-documents engine for Windows (RTX 2070 8 GB), Ollama-backed.
The full design and build order live in @.claude/PLAN.md. Follow its build order (§10a), one step at a time.
The installer work (setup wizard, minimal popup, Setup.exe with auto-update) is planned in @.claude/INSTALL_PLAN.md; follow its 10 steps in order, one commit each.

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

## Layout and status
- Code lives in `src/vector_embed/` (src layout): `core/` (no UI), `worker.py`, `watcher.py`, `cli.py`, `app/` (PySide6).
- The prototype that predated the plan has been removed. Tests mirror the package under `tests/core/`; shared fixtures (`env`, `skill_ctx`, `FakeEmbedder`) are in `tests/core/conftest.py` and `tests/core/fakes.py`.
- Done: plan steps 1-10 (core, search, ask/chat/match, models tab and health, cloud providers with privacy masking, `ve eval`), `ve mcp` (§11.1).
- In progress: INSTALL_PLAN.md (installable app). Steps done: 1 (starter picker), 2 (Ollama install; signer "Ollama Inc."). Update this line after each step. Then: choose the embedder from `ve eval`, then the §11 feature ideas as new skills.
- pytest's `tmp_path` lives under `AppData` (a blocked dir). Scope-sensitive tests use the `scope_settings` fixture, which unblocks it.
- Avoid backslashes in Bash heredocs (the tool mangles them); use the Write/Edit tools for files containing regexes or Windows paths.

## Project invariants (never break; they come from the plan)
- Indexing never runs on battery. VRAM returns to 0 after work. The embedder and chat LLM are never on the GPU together.
- Secrets are never indexed and never sent to a cloud provider; government/financial IDs are always masked before any cloud call.
- `core/` has no UI imports. Front-ends (app, cli, mcp) are thin.
- Extend by registry (`@register`), not by editing core switch statements.
