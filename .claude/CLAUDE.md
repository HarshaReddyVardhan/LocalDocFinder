# LocalDoc Finder

Local semantic search + chat-with-documents app for Windows, Ollama-backed. Runs on any PC: an NVIDIA GPU is used when present, CPU-sized models otherwise.
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
- @.claude/rules/release-process.md — tags, generated release notes, the in-app "What's new" dialog

## Commands
```powershell
.venv\Scripts\python -m pytest            # tests + coverage
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check . --fix
.venv\Scripts\mypy
.venv\Scripts\pre-commit run --all-files
```

## Layout and status
- Code lives in `src/localdoc_finder/` (src layout): `core/` (no UI), `worker.py`, `watcher.py`, `cli.py`, `app/` (PySide6).
- The prototype that predated the plan has been removed. Tests mirror the package under `tests/core/`; shared fixtures (`env`, `skill_ctx`, `FakeEmbedder`) are in `tests/core/conftest.py` and `tests/core/fakes.py`.
- Done: plan steps 1-10 (core, search, ask/chat/match, models tab and health, cloud providers with privacy masking, `ldf eval`), `ldf mcp` (§11.1).
- Done: INSTALL_PLAN.md steps 1-10 (installable app). Setup logic is in `core/setup/` (`plan.py` pure picks, `flow.py` resumable `SetupFlow`, `ollama_install.py`, `wiring.py`), `core/models/{starter,benchmark,fit}.py`, `core/{autostart,lifecycle,updates}.py`; UI is the setup wizard (`app/setup_wizard.py`), Settings window (`app/settings_*.py`), minimal popup (`app/window.py` + `mode_bar.py` + `result_delegate.py`) and `app/update_scheduler.py`. `python -m localdoc_finder <app|watcher|worker|setup>` is the one dispatcher (frozen exe too); spawn our own processes with `core.process.self_command`.
- Packaging: `packaging/localdoc_finder.spec` (PyInstaller, `LocalDocFinder.exe` + `ldf.exe`), `scripts/build.ps1` (build + smoke test + `vpk pack`; needs the .NET SDK and `vpk`, installed locally with `dotnet tool install vpk --tool-path .tools`). Built and packed locally with delta packages verified; **Setup.exe has not been run on a clean machine** (Windows Sandbox check in INSTALL_PLAN.md "Verification" is still open), and the update flow has not been exercised against a real GitHub release.
- Done: the code-audit fix plan (phases 0-6) and its phase 7 features: event hooks (`core/hooks.py`), `core/sources/` registry, versioned LanceDB schema, generated Advanced settings tab (`core/settings_schema.py`), window modes for any registered panel skill, chat history/reopen, drag-and-drop pinning, clickable citations, Match session ticks and personal-details box, chat `code_chat` routing with relevance cuts (`core/relevance.py`), prompt-injection fences (`core/prompt_safety.py`).
- Done: renamed to LocalDoc Finder (package `localdoc_finder`, CLI `ldf`, env `LDF_`, pack ID `LocalDocFinder`; old-name data, keys and tasks migrate in `core/data_migration.py`, `core/secrets.py`, `core/autostart.py`). File kinds with a custom selection (`core/file_kinds.py`, `ScopeSettings.kind_exts`), RTF reader with picture OCR, whole scanned PDFs and multi-page TIFFs.
- Done: cloud providers in the GUI (`core/providers/presets.py` presets for OpenRouter, OpenAI, Gemini, Anthropic, Groq, Mistral, DeepSeek and Custom; `ldf cloud add --preset` and `ldf cloud models`; Settings → Cloud & Privacy in `app/cloud_tab.py` with `cloud_provider_dialog.py`, `cloud_model_picker.py`, `cloud_routing_box.py`, Qt-free logic in `SettingsController`; session-wide "don't ask again" consent (`CloudConsent.grant_session`, one object kept by `ContextFactory`) and per-feature routing; routed requests preview before sending (`AssistantService.needs_cloud_consent`, `ask_routed`/`chat_routed`); fallback to the local model for routed calls that fail (`LlmGateway`, `ChatTarget.fallback`, `ChatChunk.notice`); the popup's **Model ▾** switcher (`app/cloud_switcher.py`).
- Docs: user-facing README.md; developer docs in CONTRIBUTING.md (GitHub's Contributing tab) and docs/DEVELOPER_GUIDE.md. No LICENSE by the owner's choice.
- Next: the reranker role (needs `dengcao/Qwen3-Reranker-0.6B:Q8_0` pulled to build and verify), choose the embedder from `ldf eval` on your own queries, then the §11 feature ideas as new skills.
- pytest's `tmp_path` lives under `AppData` (a blocked dir). Scope-sensitive tests use the `scope_settings` fixture, which unblocks it.
- Avoid backslashes in Bash heredocs (the tool mangles them); use the Write/Edit tools for files containing regexes or Windows paths.

## Project invariants (never break; they come from the plan)
- Indexing never runs on battery. VRAM returns to 0 after work. The embedder and chat LLM are never on the GPU together.
- Secrets are never indexed and never sent to a cloud provider; government/financial IDs are always masked before any cloud call.
- `core/` has no UI imports. Front-ends (app, cli, mcp) are thin.
- Extend by registry (`@register`), not by editing core switch statements.
