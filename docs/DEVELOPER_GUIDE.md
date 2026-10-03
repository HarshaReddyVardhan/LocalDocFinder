# Vector_Embed — Developer Guide

Local semantic search and chat-with-documents for Windows, backed by [Ollama](https://ollama.com).
This guide takes a new developer from a clean machine to a running app, a green test suite and a built installer.

For design background see [.claude/PLAN.md](../.claude/PLAN.md) and [.claude/INSTALL_PLAN.md](../.claude/INSTALL_PLAN.md).
Coding rules: [.claude/rules/](../.claude/rules/).

---

## 1. Prerequisites

| Tool | Why | Notes |
|---|---|---|
| Windows 10/11 | the app uses Windows APIs (OCR, hotkey, Task Scheduler, Credential Manager) | |
| Python 3.11+ | `requires-python = ">=3.11"`; `.python-version` pins 3.14 | `uv` will fetch it for you |
| [uv](https://docs.astral.sh/uv/) | dependency manager, reads `uv.lock` | `pip` + venv also works |
| [Ollama](https://ollama.com/download) | serves the embedding and chat models | must be running on `127.0.0.1:11434` |
| Git | | |
| NVIDIA GPU (optional) | designed for an 8 GB RTX 2070; CPU-only works with small models | |
| .NET SDK (build only) | needed for `vpk` when building the installer | end users need nothing |

## 2. Get the code and install dependencies

```powershell
git clone <repo-url> LocalDocFinder
cd LocalDocFinder
uv sync                      # creates .venv from uv.lock (runtime + dev group)
.venv\Scripts\pre-commit install
```

Never install into the global interpreter; always use `.venv`.

## 3. Pull the models

```powershell
ollama pull qwen3-embedding:0.6b     # embeddings (required for indexing/search)
ollama pull qwen3.5:9b               # chat model, only for ask/chat/match (8 GB GPU pick)
```

`ve setup --dry-run` shows what first-run setup would pick for your hardware.
`ve models` lists installed models and role choices.

## 4. Verify the environment

```powershell
.venv\Scripts\ve doctor
```

It checks Python, Ollama, models, extractors and settings, and explains any failure. Fix every `[fail]` before continuing.

## 5. Run the project

### 5a. Command line (`ve`)

```powershell
.venv\Scripts\ve index --now --path D:\Projects\myapp     # index a folder now
.venv\Scripts\ve status                                    # index + queue state
.venv\Scripts\ve search "where do we retry failed payments"
.venv\Scripts\ve search "charge_card type:code proj:billing after:2026-01"
.venv\Scripts\ve ask "how does the retry logic work?"
.venv\Scripts\ve health                                    # queue, VRAM, loaded models, cloud spend
```

Search filters: `type:img|code|doc|plan|memory|note|pdf`, `ext:py`, `proj:name`, `in:D:\path`, `after:2026-01`, `before:2026-06`.
`ve --help` and `ve <command> --help` list everything; skill commands (`search`, `ask`, `chat`, `match`) are generated from the skill registry.

Global flags: `--data-dir <dir>` (override the data folder), `--cloud-ok` (allow this command to send *masked* text to the configured cloud provider).

### 5b. Desktop app (tray + hotkey popup)

```powershell
.venv\Scripts\python -m vector_embed app         # same as: python -m vector_embed.app
```

- **Ctrl+Alt+Space** opens the popup. Enter opens a file, Ctrl+Enter reveals it in Explorer, Shift+Enter opens it in VS Code, `?`/Tab switch to Ask/Chat/Match, Esc hides.
- The tray icon opens **Settings** (models, health, cloud, updates, Advanced).
- First launch with no settings runs the **setup wizard**; force it with `python -m vector_embed setup`.

### 5c. Background processes

```powershell
.venv\Scripts\python -m vector_embed watcher     # watches folders, queues changes
.venv\Scripts\python -m vector_embed worker      # drains the queue: extract -> embed -> LanceDB
.venv\Scripts\ve autostart on                    # watcher + tray at logon (off to remove)
```

`python -m vector_embed <app|watcher|worker|setup>` is the single dispatcher (the frozen `VectorEmbed.exe` takes the same arguments).
To spawn our own processes from code use `core.process.self_command`.

**Power rules:** indexing never runs on battery (the worker only starts on AC power, settled and idle). Set `VE_POWER__REQUIRE_AC_POWER=false` only for local debugging.

### 5d. MCP server (Claude Code, Cursor, …)

```powershell
claude mcp add vector-embed -- D:\Projects\LocalDocFinder\.venv\Scripts\ve.exe mcp
```

Serves read-only `search`, `ask`, `match` over stdio. Outputs are treated as outbound: secrets are excluded, government/financial IDs masked, and `ask`/`match` use local models only.

## 6. Configuration

- Data folder: `%LOCALAPPDATA%\VectorEmbedData` (index, SQLite queue, logs, `settings.toml`). The install folder is separate, so uninstalling keeps data.
- `settings.toml` is validated at startup (pydantic-settings). Env vars override it: prefix `VE_`, nested keys joined with `__`.
- To use local overrides, copy [.env.example](../.env.example) to `.env` **in the data folder** (a `.env` elsewhere is ignored).

| Variable | Meaning |
|---|---|
| `VE_LOG_LEVEL` | logging level |
| `VE_OLLAMA_HOST` | Ollama URL |
| `VE_STORAGE__DATA_DIR` | move the data folder |
| `VE_POWER__REQUIRE_AC_POWER` | gate indexing on AC power |
| `VE_UPDATES__AUTO_CHECK` / `VE_UPDATES__REPO_URL` | auto-update behaviour and source |
| `VE_APP__START_WITH_WINDOWS` | autostart |

API keys for cloud providers go in Windows Credential Manager (`keyring`) — never in files. Manage them with the cloud commands (`ve cloud --help`).

## 7. Project layout

```
src/vector_embed/
  __main__.py      dispatcher (app | watcher | worker | setup)
  cli.py           `ve` command (thin)       mcp_server.py   `ve mcp` (thin)
  watcher.py       watchdog -> SQLite queue  worker.py       queue -> embeddings -> LanceDB
  app/             PySide6: popup, Settings window, setup wizard, update scheduler
  core/            all logic, NO UI imports
    skills/        search, ask, chat, match  (registered with @register)
    extractors/ doctypes/ sources/ providers/ privacy/ match/ models/ setup/ store/
    settings.py, power.py, scope.py, secrets.py, rag.py, retrieval.py, hooks.py, ...
tests/             mirrors the package (tests/core/...); fixtures in tests/core/conftest.py, fakes.py
eval/              queries.yaml for `ve eval` (compare embedding models)
packaging/         PyInstaller spec, icon generator
scripts/           build.ps1, install_task.ps1, sandbox/
.claude/           PLAN.md, INSTALL_PLAN.md, rules/
```

Request flow: **watcher** (file events) → **queue** → **worker** (hash diff → extract → chunk → embed → LanceDB, unload model) → **skills** (search/ask/chat/match over the store) → front-ends (app, cli, mcp).

## 8. Development workflow

```powershell
.venv\Scripts\python -m pytest                              # tests + coverage (floor 80%)
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check . --fix
.venv\Scripts\mypy                                          # strict, on src/
.venv\Scripts\pre-commit run --all-files                    # everything above + whitespace fixers
```

All four must pass before a task counts as done.

Rules to remember:
1. **Tests are offline and deterministic.** External boundaries (Ollama, HTTP, Windows APIs, psutil, pynvml) are faked. Real-Ollama tests carry `@pytest.mark.ollama` and skip when the model is absent.
2. Use `tmp_path`, never the real home or `%LOCALAPPDATA%`. pytest's `tmp_path` sits under a blocked dir (`AppData`), so scope-sensitive tests use the `scope_settings` fixture.
3. Every bug fix starts with a failing regression test.
4. Type hints everywhere; no `Any`/`# type: ignore` without a one-line reason; no `print()` in library code; no import-time side effects (heavy imports are lazy on purpose).
5. Conventional Commits (`feat:`, `fix:`, `test:`, `refactor:`, `chore:`, `docs:`), one logical change per commit, no AI-attribution trailers. Never commit `.env`, keys or local data.

### Invariants you must not break
- Indexing never runs on battery; VRAM returns to 0 after work; embedder and chat LLM are never on the GPU together.
- Secrets are never indexed or sent to a cloud provider; government/financial IDs are always masked before any cloud call.
- `core/` has no UI imports; front-ends stay thin.
- Extend via registry (`@register`), not by editing core switch statements.

### Adding a feature (example: a new skill)
1. Create `src/vector_embed/core/skills/<name>.py` with a class deriving from the skill base (`skills/base.py`), a pydantic `Input`, `name`, `description`, and decorate it with `@register`.
2. It appears automatically in `ve <name>` (CLI is generated from the registry) and in app panels that list skills.
3. Depend on the `ChatProvider` / `EmbedProvider` protocols, never on `ollama`/`openai` directly.
4. Add `tests/core/skills/test_<name>.py` using `FakeEmbedder`/`skill_ctx`; run the four checks; commit.

New extractors, doctypes, providers and sources work the same way: a new file plus `@register`.

## 9. Evaluating embedding models

```powershell
.venv\Scripts\ve eval --spec eval/queries.yaml --models qwen3-embedding:0.6b <other-model>
```

Add `--corpus <dirs>` to index your own folders and `--reuse` to skip re-indexing. Switching the embedder for real is `ve models --embedder <model> --yes` (triggers a re-index).

## 10. Build the Windows installer

One-time setup:

```powershell
dotnet tool install vpk --tool-path .tools
```

Build:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
# -> Releases\VectorEmbed-win-Setup.exe (+ full and delta update packages)
```

The script syncs the `build` dependency group, runs PyInstaller (`packaging/vector_embed.spec` → `VectorEmbed.exe` windowed, `ve.exe` console), smoke-tests `ve.exe doctor`, then packs with Velopack.

Publish an update:
1. Bump `version` in `pyproject.toml`.
2. `gh auth login` (or set `GITHUB_TOKEN`).
3. ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -RepoUrl https://github.com/OWNER/REPO -Upload
   ```

Builds are unsigned, so Windows shows a SmartScreen warning ("More info" → "Run anyway").
**Open item:** Setup.exe has not been verified on a clean machine (use Windows Sandbox, see `scripts/sandbox/`), and the update flow has not been tried against a real GitHub release.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `ve doctor` says Ollama unreachable | start Ollama; check `VE_OLLAMA_HOST` |
| Search returns nothing | run `ve index --now --path <folder>`, then `ve status` |
| Indexing never starts | you are on battery; plug in or set `VE_POWER__REQUIRE_AC_POWER=false` |
| Search "disabled on battery" error | same power gate; see above |
| Model slow / out of VRAM | pick a smaller model: `ve models --set chat=<model>` |
| Tests fail on `tmp_path` scope checks | use the `scope_settings` fixture |
| Hotkey does nothing | another app owns Ctrl+Alt+Space; change it in Settings |
| Stale state while developing | point `--data-dir` (or `VE_STORAGE__DATA_DIR`) at a scratch folder |

## 12. Roadmap (from project status)

Done: core, search, ask/chat/match, models tab, cloud providers with masking, `ve eval`, `ve mcp`, installer (steps 1–10), audit fixes and phase‑7 features.
Next: reranker role (`dengcao/Qwen3-Reranker-0.6B:Q8_0`), pick the embedder from `ve eval`, then new skills from PLAN §11.
