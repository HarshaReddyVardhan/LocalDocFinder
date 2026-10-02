# Vector_Embed

Local semantic search and chat-with-documents engine for Windows, backed by Ollama.
Design and build order: [.claude/PLAN.md](.claude/PLAN.md). Working rules: [.claude/CLAUDE.md](.claude/CLAUDE.md).

## Setup
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
.venv\Scripts\python -m vector_embed.app  # hotkey window (Ctrl+Alt+Space) + tray icon
powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1   # watcher + UI at logon
```

Search filters: `type:img|code|doc|plan|memory|note|pdf` `ext:py` `proj:name` `in:D:\path` `after:2026-01` `before:2026-06`.
UI keys: Enter open · Ctrl+Enter reveal in Explorer · Shift+Enter `code -g file:line` · Esc hide.

## How it runs
| piece | role |
|---|---|
| `vector_embed.watcher` | always on, light. watchdog events -> SQLite queue (debounced). Starts the worker only on AC power, settled, idle. |
| `vector_embed.worker` | drains the queue: hash diff -> extract -> embed only new chunks -> LanceDB -> unload the model. Checks power before every batch. |
| `vector_embed.cli` | `ve` command; skill commands are generated from the skill registry. |
| `vector_embed.app` | PySide6 hotkey window. |

Data lives in `%LOCALAPPDATA%\VectorEmbed` (index, queue, logs). Settings: `settings.toml` there, overridable with `VE_*` environment variables (see `.env.example`).

## Development
```powershell
.venv\Scripts\python -m pytest            # tests + coverage (>=80%)
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check . --fix
.venv\Scripts\mypy
```
