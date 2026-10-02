# Vector_Embed — local semantic desktop search

Ollama-only (no torch). Code, notes, PDFs, DOCX/PPTX and images (via OCR) are chunked, embedded with
`qwen3-embedding:0.6b`, and stored in LanceDB. Search is vector + BM25 fused with RRF.

## Quick start
```powershell
pip install -r requirements.txt
ollama pull qwen3-embedding:0.6b

# index something now (bypasses the idle gate; add --allow-battery to run unplugged)
python worker.py --now --path D:\Projects\myapp

# search from the terminal
python search.py "where do we retry failed payments" 
python search.py "charge_card type:code proj:billing after:2026-01"

# popup search window (Ctrl+Alt+Space) + background watcher
python -m ui.app
python watcher.py
powershell -ExecutionPolicy Bypass -File install_task.ps1   # start both at logon
```

## How it runs
| piece | role |
|---|---|
| `watcher.py` | always on, light. watchdog events → SQLite queue (30 s debounce). Starts `worker.py` only when on AC for 120 s, CPU/GPU idle, no input, nothing fullscreen. On battery it only records. |
| `worker.py` | drains the queue: hash diff → extract → embed only new chunks → LanceDB → `keep_alive=0`. Checks power before every file and between embedding batches; unplug → commit, unload, exit. `--reconcile` rescans disk for missed changes (also automatic every 6 h). |
| `search.py` / `ui/app.py` | hybrid search + filters; the UI pre-warms the model on the hotkey; on battery the query embeds on CPU. |
| `eval/run.py` | recall@10 / MRR per model on your own queries (`eval/queries.yaml`). |

Search filters: `type:img|code|doc|plan|memory|note|pdf` `ext:py` `proj:name` `in:D:\path` `after:2026-01` `before:2026-06`.
UI keys: Enter open · Ctrl+Enter reveal in Explorer · Shift+Enter `code -g file:line` · Esc hide.

Everything lives in `%LOCALAPPDATA%\VectorEmbed` (index, queue, logs). `python watcher.py --status` shows the state.
All knobs are in `indexer_config.py` (model, power policy, idle thresholds, hotkey, AI-notes allowlist, ...).

## Tests
`python -m pytest tests` (the real-Ollama test is skipped if the model isn't pulled).
