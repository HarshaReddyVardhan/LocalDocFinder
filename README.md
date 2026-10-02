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
| `vector_embed.app` | PySide6 hotkey window. |

Data lives in `%LOCALAPPDATA%\VectorEmbed` (index, queue, logs). Settings: `settings.toml` there, overridable with `VE_*` environment variables (see `.env.example`).

## Development
```powershell
.venv\Scripts\python -m pytest            # tests + coverage (>=80%)
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check . --fix
.venv\Scripts\mypy
```
