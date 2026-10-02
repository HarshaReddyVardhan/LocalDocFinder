# Vector_Embed

Local semantic search and chat-with-documents engine for Windows, backed by Ollama.
Design and build order: [.claude/PLAN.md](.claude/PLAN.md). Working rules: [.claude/CLAUDE.md](.claude/CLAUDE.md).

## Install (Windows, no Python needed)
Download `VectorEmbed-win-Setup.exe` from the project's GitHub Releases and run it. It installs for
the current user only (no admin rights), bundles its own Python, and updates itself. Windows may
show a SmartScreen warning because the build is not code-signed: choose "More info" > "Run anyway".

On first start a **setup wizard** runs:
1. It checks your hardware and **Ollama**. If Ollama is missing it offers to download and install
   it from ollama.com (about 1.2 GB); nothing is downloaded until you tick the box, and the
   installer's signature (Ollama Inc.) is checked before it runs.
2. It **picks an embedding model and a chat model that fit your GPU** (8 GB card: `qwen3.5:9b` +
   `qwen3-embedding:0.6b`; 4 GB: `llama3.2`; no GPU: small CPU models). You can change both, and
   tick extras (image captions, reranker, ...). It shows download sizes and checks free disk space.
3. It downloads the models (resumable) and **speed-tests** them one at a time; a model that is too
   slow on your machine gets a "switch to a smaller one?" offer.
4. It shows all your settings once (hotkey, folders to index, start with Windows, cloud and
   privacy, updates), then you are done. Settings are per Windows user, in
   `%LOCALAPPDATA%\VectorEmbed\settings.toml`.

Press **Ctrl+Alt+Space** anywhere: the popup is just a search bar, and results appear below it like
Explorer's (file icon, name, full path, date and size; the selected row also shows its snippet).
Enter opens, Ctrl+Enter reveals in Explorer, Shift+Enter opens in VS Code, `?` or Tab switch to Ask,
Chat and Match. Models, health and every setting live in the tray icon's **Settings** window.

Choosing models yourself: pick them in the wizard, later in Settings > Models & Health, or from the
command line: `ve setup --embed qwen3-embedding:0.6b --chat llama3.2` (add `--dry-run` to preview,
`--yes` to skip prompts, `--no-install-ollama`, `--skip-bench`, `--extras caption reranker`).
Switching the embedding model later re-indexes your files (the app tells you before it does).

Updates: the app checks GitHub Releases at start and then once a day (Settings > Updates, or
`updates.auto_check = false`), downloads only the changed parts, and offers "Restart to update" in
the tray menu. Uninstalling removes the app and its startup tasks but keeps your index and
settings; Settings > About > "Delete my data" removes those too. Ollama and its models are left alone.

## Setup (from source)
```powershell
uv sync                                   # creates .venv from uv.lock
ollama pull qwen3-embedding:0.6b          # embeddings
.venv\Scripts\pre-commit install
```

## Use
```powershell
.venv\Scripts\ve doctor                   # check the environment
.venv\Scripts\ve index --now --path D:\Projects\myapp
.venv\Scripts\ve search "where do we retry failed payments"
.venv\Scripts\ve search "charge_card type:code proj:billing after:2026-01"
.venv\Scripts\ve models                   # installed models, role choices, recommendations
.venv\Scripts\ve setup --dry-run         # what first-run setup would download for this PC
.venv\Scripts\python -m vector_embed.app  # hotkey window (Ctrl+Alt+Space) + tray icon
.venv\Scripts\ve autostart on            # watcher + UI at logon (or scripts\install_task.ps1)
```

Search filters: `type:img|code|doc|plan|memory|note|pdf` `ext:py` `proj:name` `in:D:\path` `after:2026-01` `before:2026-06`.
UI keys: Enter open · Ctrl+Enter reveal in Explorer · Shift+Enter `code -g file:line` · Esc hide.

## Use from Claude Code, Cursor and other MCP clients
`ve mcp` serves `search`, `ask` and `match` over stdio. Register it once:
```powershell
claude mcp add vector-embed -- D:\Projects\Vector_Embed\.venv\Scripts\ve.exe mcp
```
Other clients take the same command in their MCP config. The caller is usually a cloud model, so the server treats everything it returns as outbound: files under the never-send rules (secrets, `.claude` memory) are left out, government/financial IDs are masked, and `ask`/`match` use local models only (never a cloud provider). Tools are read-only, `match` takes its text inline (it cannot read arbitrary files), and nothing is loaded until the first call. `ask` and `match` take the chat lock, so they wait while the desktop app is chatting and unload their model when done.

## How it runs
| piece | role |
|---|---|
| `vector_embed.watcher` | always on, light. watchdog events -> SQLite queue (debounced). Starts the worker only on AC power, settled, idle. |
| `vector_embed.worker` | drains the queue: hash diff -> extract -> embed only new chunks -> LanceDB -> unload the model. Checks power before every batch. |
| `vector_embed.cli` | `ve` command; skill commands are generated from the skill registry. |
| `vector_embed.mcp_server` | `ve mcp`: MCP front-end for search, ask and match. |
| `vector_embed.app` | PySide6 tray app: the hotkey popup, the Settings window and the setup wizard. |
| `python -m vector_embed <app\|watcher\|worker\|setup>` | one dispatcher for every entry point; the installed `VectorEmbed.exe` takes the same arguments. |

Data lives in `%LOCALAPPDATA%\VectorEmbed` (index, queue, logs). Settings: `settings.toml` there, overridable with `VE_*` environment variables (see `.env.example`).

## Development
```powershell
.venv\Scripts\python -m pytest            # tests + coverage (>=80%)
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check . --fix
.venv\Scripts\mypy
```

## Build and release the installer
Needs the .NET SDK on the build machine (end users need nothing):
```powershell
dotnet tool install vpk --tool-path .tools                 # once; or: dotnet tool install -g vpk
powershell -ExecutionPolicy Bypass -File scripts\build.ps1  # -> Releases\VectorEmbed-win-Setup.exe
```
`scripts\build.ps1` syncs the build dependencies, builds the one-folder app with PyInstaller
(`packaging\vector_embed.spec`: `VectorEmbed.exe` windowed, `ve.exe` console), smoke-tests the
packaged `ve.exe doctor`, and packs it with Velopack (full package, plus a small delta package when
an earlier release can be downloaded).

To publish an update: bump `version` in `pyproject.toml`, then
```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -RepoUrl https://github.com/OWNER/REPO -Upload
```
(`gh auth login` or `GITHUB_TOKEN` first). `-RepoUrl` is baked into the build as the update source;
users can override it with `updates.repo_url`. Installed apps find the release, download the delta,
and offer the restart. Unsigned builds trigger SmartScreen; code signing is an optional later step.

Verify on a clean machine (Windows Sandbox, no Python, no Ollama): run Setup.exe, go through the
wizard, search; publish a newer version and check the app updates itself; uninstall and check that
the app and its scheduled tasks are gone.
