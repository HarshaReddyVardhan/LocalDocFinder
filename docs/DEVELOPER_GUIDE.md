# LocalDoc Finder — Developer Guide

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

`ldf setup --dry-run` shows what first-run setup would pick for your hardware.
`ldf models` lists installed models and role choices.

## 4. Verify the environment

```powershell
.venv\Scripts\ldf doctor
```

It checks Python, Ollama, models, extractors and settings, and explains any failure. Fix every `[fail]` before continuing.

## 5. Run the project

### 5a. Command line (`ldf`)

```powershell
.venv\Scripts\ldf index --now --path D:\Projects\myapp     # index a folder now
.venv\Scripts\ldf status                                    # index + queue state
.venv\Scripts\ldf search "where do we retry failed payments"
.venv\Scripts\ldf search "charge_card type:code proj:billing after:2026-01"
.venv\Scripts\ldf ask "how does the retry logic work?"
.venv\Scripts\ldf health                                    # queue, VRAM, loaded models, cloud spend
```

Search filters: `type:img|code|doc|plan|memory|note|pdf`, `ext:py`, `proj:name`, `in:D:\path`, `after:2026-01`, `before:2026-06`.
`ldf --help` and `ldf <command> --help` list everything; skill commands (`search`, `ask`, `chat`, `match`) are generated from the skill registry.

Global flags: `--data-dir <dir>` (override the data folder), `--cloud-ok` (allow this command to send *masked* text to the configured cloud provider).

### 5b. Desktop app in dev mode (UI)

Run these from the repo root, with Ollama running (section 3). No build or install is needed; the app runs straight from `src/` in `.venv`.

```powershell
# Start the UI and open the search window immediately (recommended while developing)
.venv\Scripts\python -m localdoc_finder app --show

# Other ways to start it
.venv\Scripts\python -m localdoc_finder app           # tray icon only; press the hotkey to open the popup
.venv\Scripts\python -m localdoc_finder.app --show    # equivalent module form
.venv\Scripts\python -m localdoc_finder setup         # open the first-run setup wizard now
.venv\Scripts\python -m localdoc_finder app --setup   # same as above
.venv\Scripts\pythonw -m localdoc_finder app          # no console window (like a real launch)
```

Flags of the app entry: `--show` (show the window at start), `--setup` (run the wizard at start).

What you get:
- **Tray icon** (blue "S"): left-click opens the search popup; right-click menu has Search, Settings…, Run setup again…, Restart to update, Quit.
- **Ctrl+Alt+Space** opens the popup from anywhere. If another app owns that key the tray tooltip says "hotkey unavailable"; use the tray icon or change the key in Settings.
- **Popup keys:** type to search; Enter opens the file; Ctrl+Enter reveals it in Explorer; Shift+Enter opens it in VS Code; `?` or Tab switch to Ask / Chat / Match; Esc hides.
- **Settings window** (tray > Settings…): general settings, Models & Health, cloud and privacy, Updates, a generated Advanced tab for every setting.
- **First launch** (no completed setup recorded in the data folder) runs the setup wizard automatically.

Dev-mode things to know:
1. **Quit with the tray menu, not Ctrl+C.** Closing the window only hides it (`setQuitOnLastWindowClosed(False)`); Quit unloads the models. Use `Stop-Process -Name pythonw,python` only as a last resort.
2. **Only one copy runs per data folder** (file lock `app.lock`). A second launch logs "another copy of the app is already running" and exits silently. Quit the tray copy first, including an installed `LocalDocFinder.exe` that uses the same data folder.
3. **Python changes need a restart.** There is no hot reload: quit from the tray and start it again. Settings changes made in the Settings window apply without restart.
4. **The watcher is not started for you from source.** The app only auto-starts the watcher in the packaged build. To index files while the UI runs, start it yourself (5c), or index on demand with `ldf index --now --path <folder>`.
5. **Use a scratch data folder to avoid touching your real index and settings:**
   ```powershell
   $env:LDF_STORAGE__DATA_DIR = "D:\scratch\ve-data"
   .venv\Scripts\python -m localdoc_finder app --show
   ```
   Set the same variable in every terminal that runs `ldf`, the watcher or the worker so they share that folder (or pass `ldf --data-dir D:\scratch\ve-data ...` for CLI commands).
6. **Debug output:** run with `python` (not `pythonw`) to see logs in the console, set `LDF_LOG_LEVEL=DEBUG`, and read the JSON log files in `<data folder>\logs` (default `%LOCALAPPDATA%\LocalDocFinderData\logs`; `app` writes the app's log). Unhandled exceptions are logged by `install_excepthooks`.
7. **Invalid settings:** the app shows an error dialog and exits with code 2. Fix `settings.toml` in the data folder or the `LDF_*` variable named in the message.
8. **Qt needs a desktop session.** It cannot run over SSH or in a headless service.

### 5b-1. Full dev stack, step by step

1. Start Ollama (tray app or `ollama serve`) and confirm: `ldf doctor`.
2. Terminal 1, UI: `.venv\Scripts\python -m localdoc_finder app --show`.
3. Terminal 2, background indexing: `.venv\Scripts\python -m localdoc_finder watcher` (starts the worker on AC power when idle).
4. Terminal 3, tools: `ldf status`, `ldf health`, `ldf search "..."`, and `pytest`.
5. In the popup, search; open Settings from the tray to add folders to index and check models.
6. Edit code, quit from the tray, restart step 2. Run the four checks (section 8) before committing.

### 5c. Background processes

```powershell
.venv\Scripts\python -m localdoc_finder watcher     # watches folders, queues changes
.venv\Scripts\python -m localdoc_finder worker      # drains the queue: extract -> embed -> LanceDB
.venv\Scripts\ldf autostart on                    # watcher + tray at logon (off to remove)
```

`python -m localdoc_finder <app|watcher|worker|setup>` is the single dispatcher (the frozen `LocalDocFinder.exe` takes the same arguments).
To spawn our own processes from code use `core.process.self_command`.

**Power rules:** indexing never runs on battery (the worker only starts on AC power, settled and idle). Set `LDF_POWER__REQUIRE_AC_POWER=false` only for local debugging.

### 5d. MCP server (Claude Code, Cursor, …)

```powershell
claude mcp add localdoc-finder -- D:\Projects\LocalDocFinder\.venv\Scripts\ldf.exe mcp
```

Serves read-only `search`, `ask`, `match` over stdio. Outputs are treated as outbound: secrets are excluded, government/financial IDs masked, and `ask`/`match` use local models only.

## 6. Configuration

- Data folder: `%LOCALAPPDATA%\LocalDocFinderData` (index, SQLite queue, logs, `settings.toml`). The install folder is separate, so uninstalling keeps data.
- `settings.toml` is validated at startup (pydantic-settings). Env vars override it: prefix `LDF_`, nested keys joined with `__`.
- To use local overrides, copy [.env.example](../.env.example) to `.env` **in the data folder** (a `.env` elsewhere is ignored).

| Variable | Meaning |
|---|---|
| `LDF_LOG_LEVEL` | logging level |
| `LDF_OLLAMA_HOST` | Ollama URL |
| `LDF_STORAGE__DATA_DIR` | move the data folder |
| `LDF_POWER__REQUIRE_AC_POWER` | gate indexing on AC power |
| `LDF_UPDATES__AUTO_CHECK` / `LDF_UPDATES__REPO_URL` | auto-update behaviour and source |
| `LDF_APP__START_WITH_WINDOWS` | autostart |

API keys for cloud providers go in Windows Credential Manager (`keyring`) — never in files. Manage them with the cloud commands (`ldf cloud --help`).

## 7. Project layout

```
src/localdoc_finder/
  __main__.py      dispatcher (app | watcher | worker | setup)
  cli.py           `ldf` command (thin)       mcp_server.py   `ldf mcp` (thin)
  watcher.py       watchdog -> SQLite queue  worker.py       queue -> embeddings -> LanceDB
  app/             PySide6: popup, Settings window, setup wizard, update scheduler
  core/            all logic, NO UI imports
    skills/        search, ask, chat, match  (registered with @register)
    extractors/ doctypes/ sources/ providers/ privacy/ match/ models/ setup/ store/
    settings.py, power.py, scope.py, secrets.py, rag.py, retrieval.py, hooks.py, ...
tests/             mirrors the package (tests/core/...); fixtures in tests/core/conftest.py, fakes.py
eval/              queries.yaml for `ldf eval` (compare embedding models)
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
1. Create `src/localdoc_finder/core/skills/<name>.py` with a class deriving from the skill base (`skills/base.py`), a pydantic `Input`, `name`, `description`, and decorate it with `@register`.
2. It appears automatically in `ldf <name>` (CLI is generated from the registry) and in app panels that list skills.
3. Depend on the `ChatProvider` / `EmbedProvider` protocols, never on `ollama`/`openai` directly.
4. Add `tests/core/skills/test_<name>.py` using `FakeEmbedder`/`skill_ctx`; run the four checks; commit.

New extractors, doctypes, providers and sources work the same way: a new file plus `@register`.

## 9. Evaluating embedding models

```powershell
.venv\Scripts\ldf eval --spec eval/queries.yaml --models qwen3-embedding:0.6b <other-model>
```

Add `--corpus <dirs>` to index your own folders and `--reuse` to skip re-indexing. Switching the embedder for real is `ldf models --embedder <model> --yes` (triggers a re-index).

## 10. Build the Windows installer

One-time setup:

```powershell
dotnet tool install vpk --tool-path .tools
```

Build:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
# -> Releases\LocalDocFinder-win-Setup.exe (+ full and delta update packages)
```

The script syncs the `build` dependency group, runs PyInstaller (`packaging/localdoc_finder.spec` → `LocalDocFinder.exe` windowed, `ldf.exe` console), smoke-tests `ldf.exe doctor`, then packs with Velopack.

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
| `ldf doctor` says Ollama unreachable | start Ollama; check `LDF_OLLAMA_HOST` |
| Search returns nothing | run `ldf index --now --path <folder>`, then `ldf status` |
| Indexing never starts | you are on battery; plug in or set `LDF_POWER__REQUIRE_AC_POWER=false` |
| Search "disabled on battery" error | same power gate; see above |
| Model slow / out of VRAM | pick a smaller model: `ldf models --set chat=<model>` |
| Tests fail on `tmp_path` scope checks | use the `scope_settings` fixture |
| Hotkey does nothing | another app owns Ctrl+Alt+Space; change it in Settings |
| Stale state while developing | point `--data-dir` (or `LDF_STORAGE__DATA_DIR`) at a scratch folder |

## 12. Roadmap (from project status)

Done: core, search, ask/chat/match, models tab, cloud providers with masking, `ldf eval`, `ldf mcp`, installer (steps 1–10), audit fixes and phase‑7 features.
Next: reranker role (`dengcao/Qwen3-Reranker-0.6B:Q8_0`), pick the embedder from `ldf eval`, then new skills from PLAN §11.
