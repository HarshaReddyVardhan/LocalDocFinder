# Plan: Local Search + Chat-with-Documents Engine (rebuild from scratch, Ollama-based)

## Context
The user wants a local desktop engine on a Windows laptop (RTX 2070 8 GB). It has two features:
1. **Semantic search** over code, documents, PDFs, images and AI-tool notes (`.claude/plans`, memory files), opened with a hotkey.
2. **New: chat with documents.** Example: paste a job description, ask "which resume on my system best matches this JD?", then keep chatting ("what's missing?", "rewrite my bullets for it"). A **regular chat LLM is loaded on demand only when a question is asked** and unloaded afterwards.

Only `d:\Projects\Vector_Embed\indexer_config.py` exists so far, and the user has said everything can be rebuilt. Requirements that must stay:
- No heavy background process when idle.
- VRAM goes back to 0 after work.
- **Indexing only on AC power.**
- AI-note folders get indexed; secrets never do.
- Incremental updates.
- Large code files and many projects.
- PDFs and DOCX files with images.

**Is it possible? Yes.** The user's Ollama is already the backend. Search uses an embedding model. Chat and matching use a chat LLM through the same Ollama server, loaded only while a chat is active.

---

## 1. Models (all through Ollama; nothing else loads on the GPU)

| Role | Model | Notes |
|---|---|---|
| Embeddings (default) | **`qwen3-embedding:0.6b`** (pull) | Long context, strong on code |
| Embeddings (candidates for the eval) | `embeddinggemma` (pull), `bge-m3`, `nomic-embed-text` (installed) | Decided by the eval in Step 10. Drop `mxbai-embed-large`, which only reads 512 tokens. |
| **Chat / RAG / matching** | **`qwen3.5:9b`** (pull, about 5.5–6.6 GB at Q4) | Current top pick for 8 GB cards. About 8K usable context on 8 GB, and the design below works within that. |
| Chat fallback (fast, smaller) | `qwen2.5:7b` (pull) or `llama3.2` (installed) | Used when VRAM is tight or for quick answers |
| Code questions (optional routing) | `qwen2.5-coder:7b` (installed) | Used when the retrieved context is mostly code |
| Image captions (optional, off by default) | `qwen2.5vl:3b` (pull) | Runs only during idle indexing |

`deepseek-r1:8b` isn't used: its long "thinking" output makes interactive chat slow. All model names live in config, so they can be swapped without code changes.

**VRAM rules:**
- The embedder and the chat LLM are **never on the GPU at the same time.** In chat mode the query embedding runs on the CPU (`num_gpu: 0`, about 100–200 ms), so the GPU belongs to the LLM.
- Chat LLM: `num_ctx = 8192`. Enable flash attention and a q8 KV cache (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`) to save VRAM.
- Lifecycle: the model **pre-warms** when the user switches to chat mode (an empty request starts the load while they type). It stays loaded with `keep_alive = "10m"` for follow-up questions and is **explicitly unloaded** (`keep_alive=0`) when the window closes, after 10 minutes idle, on unplugging, or when a fullscreen game starts.
- The indexing worker **never starts while a chat session is active** (a lock row in SQLite).

**Ollama pitfalls:**
- Ollama silently truncates input to `num_ctx`, so always pass it explicitly.
- Embed in batches with `ollama.embed(input=[...])`.
- Each model needs its own query/document prefixes (qwen3 `Instruct:...Query:`, nomic `search_query:`), kept in a table in config.

---

## 1a. Model registry: tracks which models exist and which would help
The app knows which models are available and recommends better ones, for this user or anyone else who installs it.
- **Discovery** at startup, every hour and on demand:
  - Ollama: `ollama.list()` lists installed models, and `ollama.show(name)` gives each one's capabilities (`embedding`, `completion`, `vision`, `tools`, `thinking`), context length, parameter count, quantization and size.
  - Cloud providers: their `/models` endpoint, when a key is configured.
- **Hardware probe:** total and free VRAM and GPU name (`pynvml`), RAM and CPU (`psutil`), and AC or battery. A machine with no NVIDIA GPU falls back to CPU-friendly model choices.
- **Roles, not hard-coded model names:** `embed`, `chat`, `match_scorer`, `code_chat`, `caption`, `summarizer`, `reranker`. Each role has a **curated preference list** in `models_catalog.toml` that includes size and VRAM requirements, for example `chat: [qwen3.5:9b, qwen3:8b, qwen2.5:7b, llama3.2]`.
- **Resolver:** for each role it picks the best **installed** model that **fits the free VRAM**. The user can override any choice in settings.
- **Recommendations panel ("Models" tab):**
  - Lists installed models per role with ✓ / ⚠ ("mxbai-embed-large: 512-token limit, not recommended").
  - Shows "**Better option available:** `ollama pull qwen3.5:9b`" with a one-click pull and a progress bar.
  - Flags models that are installed but unused (for example "deepseek-r1:8b, 5.2 GB, unused, consider removing").
  - The catalog file can be updated without code changes.
- **Embedding safety:** the `embed` role is pinned. Changing it shows "re-index required (~N files, est. time)" and runs only after confirmation, because vectors from different models can't be mixed.
- Stored in the SQLite tables `models` and `model_capabilities`, and shown in a health dashboard.

## 1b. Providers: local by default, ready for OpenAI and OpenRouter
- **Interfaces** in `core/providers/base.py`:
  - `ChatProvider`: `stream_chat()`, `chat_json(schema)`, `list_models()`, `capabilities()`, `estimate_cost()`.
  - `EmbedProvider`: `embed(texts, kind=query|doc)`.
- **Implementations:**
  - `OllamaProvider` (default). Handles load and unload, `keep_alive` and the GPU/CPU switch.
  - `OpenAICompatibleProvider`, built on the `openai` SDK with a `base_url`. **One adapter covers OpenAI, OpenRouter, LM Studio, llama.cpp server, vLLM and Groq.** OpenRouter only needs `base_url=https://openrouter.ai/api/v1` and a key.
  - Adding another provider means one new file plus a registry entry.
- **Keys** are stored in Windows Credential Manager via `keyring`, never in config files and **never indexed**. The secrets denylist also blocks the app's own data folder.
- **Routing policy per role** (in settings):
  - `local`: always local.
  - `cloud`: always the chosen cloud model.
  - `auto`: local first. Cloud is used when the user clicks "Answer better ☁", when the local model doesn't fit in VRAM, or **when on battery**, since cloud uses no local GPU or battery.
- **Privacy guardrails before anything leaves the machine:**
  - A visible badge: "☁ Sending 6 excerpts (≈3.1k tokens) to OpenRouter / gpt-x".
  - A per-session opt-in.
  - `NEVER_SEND_TO_CLOUD` path and doc_type rules (default: `.claude` memory, anything matching the secrets list).
  - **Always-on masking of government and financial IDs** (SSN, passport, DL, national ID, cards, bank numbers, secrets), plus an **optional "Remove personal details" checkbox** for name, email, phone and address, off by default. Details in §3.
- Token use and estimated **cost per provider** are recorded in SQLite and shown in the dashboard. Optional monthly budget cap.
- Embeddings stay local by default. A cloud embedding provider is supported but means a separate index.

## 1c. Designed to be extended: plugin registries in a core library
- **`core/`** is a plain Python library with no UI. The PySide6 app, the CLI, and an optional local API and MCP server are thin front-ends on top of it.
- **Registries** use `@register(...)` decorators, so new things are added by dropping in one file:
  - `extractors/`: one class per file type, `supports(path) → bool` and `extract(path) → Iterable[Chunk]`. A new format (pptx, epub, xlsx) is one file.
  - `doctypes/`: classifiers that tag documents (resume, jd, invoice and so on).
  - `providers/`: as in §1b.
  - **`skills/`**: user-facing features with a name, input schema, the model roles they need and a UI hint (panel or table). **Search, Ask, Chat and Match are all skills.** Future features are new skills, not rewrites.
  - `sources/`: where content comes from. Filesystem first; later browser history, git history, email, clipboard.
- **Event hooks:** `on_file_indexed`, `on_document_classified`, `on_query`, `on_answer`. These let skills react, for example "auto-match new JDs in Downloads".
- **Settings:**
  - One versioned `settings.toml` with a typed schema (pydantic), defaults, and migrations between versions.
  - The settings UI is generated from the schema, so new options appear automatically.
- **Data versioning:** LanceDB and SQLite schemas carry a `schema_version`, and migrations run at startup.

---

## 2. What the user sees: one hotkey window, three modes (Ctrl+Alt+Space)
Alt+Space is avoided because Windows and PowerToys already use it.

| Mode | How to enter | What happens |
|---|---|---|
| **Search** (default) | type | Instant hybrid results, filters such as `type:pdf ext:py proj:x type:plan` |
| **Ask** | `Tab` or start with `?` | Answers from the whole index, streamed, with clickable citations `[1] file:line` |
| **Chat with docs** | "Chat" button on any result(s), or drag in or paste text | A conversation pinned to the chosen documents and/or pasted text |
| **Match** | "Match…" button, or paste text and choose "Find best match in: [Resumes ▾]" | Ranks a group of documents against the pasted text (the JD example), then continues as a chat |

- The chat panel streams markdown. Citations open the file (`code -g file:line` for code, the default app for PDFs) or reveal it in Explorer.
- Pasted text such as a JD becomes a temporary **"scratch document"**, held in memory for the session and never written to the index.
- Chat history for a session is kept in SQLite so a chat can be reopened.

---

## 3. How "best resume for this JD" works (document-level matching)
Searching chunks isn't enough for this task, so the index needs to understand whole **documents**.

**At indexing time:**
- A `documents` table holds one row per file: path, `doc_type`, title, full text (when under about 20k characters), a **document-level vector** and `modified_at`.
- `doc_type` classification needs no LLM:
  - Filename rules: `resume|cv|curriculum`.
  - Section headings: Experience / Education / Skills / Projects.
  - Similarity to a "resume" prototype vector.
  - Types: `resume`, `cover_letter`, `jd`, `invoice`, `paper`, `notes`, `plan`, `code`, `other`.
  - The rule list is easy to extend in config.
- **Version groups:** files like `Resume_v1.pdf`, `Resume_final.docx` and `resume (2).pdf` that are more than 90% similar are grouped. Matching shows the **newest version per group** by default, with a toggle to show all.

**At query time (Match). What is sent to the LLM at each step:**

| Step | Runs where | What is sent | What comes back |
|---|---|---|---|
| 1. Recall | **Always local**, no chat LLM | The JD text goes only to the local embedding model (CPU) | Top 10 candidate resumes from vector search plus BM25, fused with RRF |
| 2. Requirements checklist (**once**) | Chat LLM (local or cloud) | **JD only** | JSON `requirements[]`: `{id, text, type: must/nice, category: skill/experience/education/cert, years?}` |
| 3. Score each resume (**one call per resume**) | Chat LLM | **The checklist + the full text of ONE resume** (with the JD summary) | JSON per requirement: `{id, status: met/partial/missing, evidence_quote}` plus `seniority_fit` and `summary` |
| 4. Verdict | Chat LLM | **Only the JSON results from step 3** (resumes are not resent) | A short ranked explanation |
| 5. Follow-up chat | Chat LLM | Pinned context: **JD + the selected resume(s) in full + their step-3 JSON**, plus the chat history | Answers ("what's missing?", "rewrite bullets", "cover letter") |

**Design choices behind this:**
- **Whole resume, not chunks, in step 3.** Deciding that a skill is *missing* requires seeing the entire document. If only matching chunks were sent, the model would wrongly report skills as "missing" because they sat in a chunk it never saw. Resumes are short (about 1–2k tokens), so the whole document fits. If a document is longer than the budget (about 5k tokens locally), it's cut down to the sections most relevant to the JD, and that cut is **reported in the UI** as reduced reliability.
- **One fixed checklist for every resume (step 2).** Every resume is judged against the same requirements, so results are **consistent and comparable**. The **score is calculated in code**, not invented by the LLM: must-have counts double, partial counts half, then normalized to 0–100. That makes it deterministic and explainable ("7/9 must-haves met").
- **Evidence is checked.** Each `evidence_quote` is matched against the resume text, using whitespace and case-insensitive fuzzy matching. A "met" with no real quote behind it is downgraded to "unverified ⚠", which catches hallucinated matches.
- **The checklist can be edited.** Before scoring, the user can tick, untick or reweight requirements in the UI ("ignore the cert requirement").
- **Cost and limits:** locally there are 1 + N + 1 calls, each about 2–5k tokens, all within the 8K context. On cloud the N scoring calls run **in parallel**. The pinned context goes **first** in each prompt, so providers that cache a repeated prompt start (OpenAI, and OpenRouter models that support it) charge less for follow-up questions.
- **Candidate checklist (between steps 1 and 2).** The recalled resumes (usually 5–10) appear in a **checkbox list** before anything goes to a chat LLM:
  - Each row shows: file name, folder, last modified, version-group badge ("3 versions, showing newest"), recall similarity, estimated tokens, and the destination (🖥 local / ☁ cloud).
  - **Default selection:** every candidate above the similarity threshold is ticked. Older versions in a group are unticked. Buttons: "Select all", "Select none", "Top 3".
  - "**+ Add file…**" lets the user include a resume that recall missed (file picker or search box).
  - Only **ticked** resumes are scored (step 3) and can be pinned in the follow-up chat. The selection is remembered for the session, so re-running with the same JD keeps it.
  - The footer shows a live total: "☁ JD + 6 resumes ≈ 11k tokens → OpenRouter/<model> · est. $0.01". The "Score" button sends.
- **Sensitive-ID masking: always on for anything sent to the cloud** (not a checkbox; only changeable in settings):
  - Masked as `[SSN REMOVED]`, `[PASSPORT REMOVED]` and so on. This is **not reversible**: the cloud never sees the value and the output never needs it.
  - Detected types (default list `ALWAYS_REDACT`): SSN; passport numbers; driver's licence numbers; national ID numbers (for example Aadhaar, PAN, NI and SIN formats); tax IDs; credit and debit cards (Luhn-validated); bank account, IBAN and routing numbers; and API keys or secrets that turn up in documents.
  - Detection combines format regexes and checksums (Luhn, Verhoeff for Aadhaar) with **label keywords nearby** ("SSN", "Passport No", "DL#", "License No", "Aadhaar", "PAN"). That catches formats without a fixed pattern, such as DL numbers, and avoids masking ordinary numbers.
  - Microsoft Presidio can be added later as an optional detector plugin for higher recall.
  - The UI shows what was found before sending: "🛡 2 sensitive items will be masked: Passport (resume_v2.pdf), DL (resume_old.docx)". Each can be expanded to see where it is.
- **"Remove personal details" checkbox: unticked by default** (per user preference):
  - Ticked: name, email, phone, street address and profile URLs are replaced with **reversible** placeholders (`[NAME_1]`, `[EMAIL_1]`). The cloud answer is shown with the real values restored.
  - Unticked (default): these are sent as they are. City and country are never removed, because they matter for location fit.
- **"👁 View what will be sent" button:** shows the exact text going out after masking, per resume, so nothing is hidden.
- **Local runs (Ollama) send nothing off the machine.** No masking is needed; it can optionally be applied via settings.
- Resumes under `NEVER_SEND_TO_CLOUD` paths show a 🔒 icon. They can still be ticked, and are then **scored locally** even in cloud mode.
- The same masking pipeline (`core/privacy/`) applies to **every** cloud request: Ask, Chat and future skills, not only Match.
- **Embeddings never leave the machine**, so step 1 is always private. Only steps 2–5 can go to the cloud, and only with consent.

The same mechanism generalizes to **Match any text against any doc_type**: a JD against resumes, an error log against code, a spec against plans. One feature, many uses.

**General Ask/RAG flow:**
1. Hybrid retrieval of the top ~20 chunks.
2. Merge neighbouring chunks and group by document.
3. Pack about 5k tokens of context.
4. The LLM answers with `[n]` citations, under the instruction "only answer from the sources; say if not found".
5. Code-heavy context goes to `qwen2.5-coder:7b` when routing is on.

---

## 4. Power policy (hard rule)
- **Indexing never runs on battery:** no worker, OCR, captions or reconcile scan. On battery the watcher only appends to the SQLite queue, which costs almost nothing.
- **If power is lost mid-run,** the worker checks before every batch, commits what it has, unloads the model and exits. It resumes once AC has been stable for 120 s **and** the system is idle.
- **Search on battery:** allowed, on the CPU (`num_gpu: 0`).
- **Local chat on battery:** `CHAT_ON_BATTERY = False` by default. If a cloud provider is configured and routing is `auto`, chat goes to the **cloud** on battery, which uses no local GPU. Otherwise the UI shows "plug in to chat", with an option to allow the small fallback model.
- **Detection:** `psutil.sensors_battery()` (a result of `None` means a desktop, treated as plugged in), polled every 30 s.
- **Manual override:** `worker.py --now --allow-battery`.

## 5. Idle gate (on top of power)
- CPU below 15% for 60 s.
- No user input for 60 s (`GetLastInputInfo`).
- GPU utilization low and enough free VRAM (`pynvml`).
- No fullscreen app or game (`SHQueryUserNotificationState`).
- No active chat session.
- Checked again before every batch; the worker gives way immediately if anything fails.

## 6. Change detection (yes, it picks up updates)
1. **Live:** `watchdog` events, debounced for 30 s, are written to a persistent SQLite queue.
2. **Missed events:** a reconciliation scan at startup and every 6 h (on AC only) compares `(path, mtime, size)` against the manifest. This catches reboots, crashes and watcher buffer overflows.
3. **Cheap updates:** file `xxhash`, then per-chunk hashes. A no-op `git checkout` re-embeds nothing, and editing one function re-embeds only that chunk.
4. **Deletes and renames** remove rows, then `table.optimize()` runs after each worker run.
5. **Model changes:** every row stores `model_id`, so changing the model triggers a clean re-index.

## 7. Scope rules (rewritten `indexer_config.py`)
**Roots:** the home folder and `"D:\\"`. The current `r"D:\\"` is a bug: it is literally two backslashes.

**Order of checks** (cheap first; the final step allows):
1. Secrets denylist: `.env*`, `*.pem`, `*.key`, `*.pfx`, `id_rsa*`, `*credential*`, `.npmrc`, `.pypirc`, `*.kdbx`, `~/.claude/settings*.json`, `.credentials.json`.
2. Blocked directories, through `should_descend()` pruning: system folders, AppData, build output, `node_modules`, `.git`, venvs and caches.
3. The project's `.gitignore` (`pathspec`); inside git repos, files are listed with `git ls-files -co --exclude-standard`.
4. Hidden-directory rule, with the **`AI_NOTE_DIRS` allowlist**:
   - `.claude`: `plans/**/*.md`, `projects/*/memory/**/*.md`, `CLAUDE.md`, `skills|agents|commands/**/*.md`, and project-level `.claude/**/*.md`.
   - Also `.cursor` rules, `.github/copilot-instructions.md` and `instructions/`, and `.gemini`, `.codex`, `.windsurf`, `.continue` markdown.
   - Root-level `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.cursorrules`.
   - `.jsonl` transcripts are off by default (`INDEX_AI_TRANSCRIPTS = False`).
   - Excluded: the `plugins/`, `shell-snapshots/`, `todos/`, `statsig/`, `ide/` and cache folders.
5. Extension or exact-filename match:
   - `TEXT_FILENAMES` covers `Dockerfile`, `Makefile`, `README`, `LICENSE` and `.env.example`. The current `.env.example` extension entry never matches.
   - New types: `.ipynb`, `.vue`, `.svelte`, `.dart`, `.r`, `.ini`, `.cfg`, `.gradle`, `.tex`, `.pptx`.
   - `.svg` moves to text.
   - Noise blocked: `*.min.js`, `*.map`, lockfiles.
6. Attributes: skip reparse points or junctions and OneDrive placeholder files (`RECALL_ON_DATA_ACCESS` / `OFFLINE`), so reading them doesn't trigger downloads.
7. Size limits (the only syscall): text 15 MB, documents 50 MB, images 100 MB.
8. Generated-file heuristic: average line length over 300, or an `@generated` header.

## 8. Extraction and chunking
- **Code:** functions and classes via `tree-sitter-language-pack`, with oversized ones split by lines with overlap. Each chunk gets the header `project › path › Class.method`, and start and end lines are stored. Each file also gets **one outline chunk** (imports and top-level symbols). Embeddings are reused by content hash across projects.
- **Markdown and plans:** split by heading sections, with heading breadcrumbs.
- **PDF (PyMuPDF):** text by page and paragraph. Embedded images over 150 px are OCR'd and stored with their page number. Scanned pages are rendered and OCR'd. At most 50 images per document.
- **DOCX (`python-docx`):** text by heading. `word/media/*` images are OCR'd.
- **Images:** downscaled with draft mode or thumbnails, deduplicated with pHash, and given a thumbnail for the UI. OCR uses the built-in Windows OCR, plus an optional caption. All of it becomes **text in the same embedding space**, so CLIP isn't needed.
- **Projects:** a root is detected by `.git`, `package.json`, `pyproject.toml`, `Cargo.toml`, `go.mod`, `*.sln` or `pom.xml`. Projects modified most recently are indexed first.

## 9. Architecture and storage
```
watcher.py (always on, ~30 MB, no ML)      worker.py (spawned; exits when done)
 watchdog + reconcile → SQLite queue        queue → hash diff → extract/OCR → classify doc_type
 power gate → idle gate → spawn worker      → ollama.embed (GPU) → LanceDB upsert/delete
                                            → optimize() → keep_alive=0 → exit
app/ (resident PySide6, hotkey)
 search.py   hybrid: vector + LanceDB FTS (BM25 on text/path/symbol) → RRF + filename boost
 rag.py      retrieve → pack context → stream answer w/ citations
 match.py    recall (documents table) → per-doc JSON scoring → ranked table → chat
 llm.py      Ollama chat client: prewarm, keep_alive, unload, model routing, power/VRAM checks
 ui/         search list, chat panel (streaming markdown), match table, previews

LanceDB: chunks(vector, text, path, project, kind, symbol, start_line, end_line, page,
                chunk_hash, model_id)  + FTS index
         documents(doc_vector, path, doc_type, title, full_text, version_group,
                   modified_at, model_id) + FTS index
SQLite:  manifest, queue, chat_sessions/messages, locks
```
Search and query latency target: **under 150 ms**. Index maintenance: flat search below about 100k rows, IVF_PQ index above that.

## 10. Project layout (fresh, in `d:\Projects\Vector_Embed\`)
```
core/
  settings.py          pydantic schema + settings.toml + migrations
  scope.py             is_valid_file, should_descend, gitignore, AI_NOTE_DIRS, projects
  power.py, idle.py    AC/battery, CPU/input/GPU/fullscreen gates
  store/               lance.py (chunks, documents, FTS), sqlite.py (manifest, queue, sessions, models, usage)
  models/              registry.py (discovery + hardware probe + resolver), models_catalog.toml
  providers/           base.py, ollama.py, openai_compat.py  (+ registry)
  privacy/             detectors.py (ID regex + checksums + label keywords), mask.py (irreversible IDs,
                       reversible personal placeholders), payload_preview.py; wraps every cloud call
  extractors/          code.py, markdown.py, pdf.py, docx.py, image.py, ocr.py  (+ registry)
  doctypes/            resume.py, jd.py, plan.py, ...  (+ registry, version groups)
  skills/              search.py, ask.py, chat.py, match.py  (+ registry)
  hooks.py             event bus
watcher.py             always-on gatekeeper
worker.py              short-lived indexer
app/                   PySide6 hotkey UI: search list, chat panel, match table, Models tab, dashboard
cli.py                 `ve search|ask|match|index|models|doctor`
eval/                  queries.yaml, match_cases/, run.py
tests/
```

## 10a. Build order
1. `core/settings.py` and `core/scope.py` (§7) with unit tests. `indexer_config.py` is replaced.
2. `core/store/`, `core/providers/ollama.py`, and `core/models/registry.py` (discovery, hardware probe, resolver, catalog).
3. `core/extractors/*` and `core/doctypes/*`.
4. `worker.py` (with power and idle checks before every batch) and `watcher.py` (plus Task Scheduler registration).
5. `skills/search.py`, `cli.py` and the basic hotkey UI. **First usable version.**
6. `skills/ask.py` and `skills/chat.py`: streaming answers with citations, model pre-warm and unload.
7. `skills/match.py` and the Match UI: JD-to-resume ranking, then follow-up chat.
8. The Models tab and health dashboard: installed models, recommendations, one-click pull, VRAM, queue size, cost.
9. `core/providers/openai_compat.py`: OpenAI and OpenRouter keys via `keyring`, routing policy, privacy badge and rules, usage tracking.
10. `eval/` and model selection. Then features from §11 as new skills.

**Extra pip:** `pymupdf python-docx tree-sitter-language-pack pathspec psutil nvidia-ml-py xxhash pillow imagehash winrt-Windows.Media.Ocr winrt-Windows.Graphics.Imaging winrt-Windows.Storage PySide6 keyboard`.
**Ollama pulls:** `qwen3-embedding:0.6b`, `qwen3.5:9b`, plus optionally `embeddinggemma`, `qwen2.5:7b` and `qwen2.5vl:3b`.

## 11. Feature ideas (each is a new skill or source; no core rewrite needed)
**High value, low effort:**
1. **MCP server** (`ve mcp`) exposing `search`, `ask` and `match` to Claude Code, Cursor and similar tools. Your AI coding tools can then search your whole drive, your old projects and your plans.
2. **Collections and tags:** for example "Job hunt" = resumes + JDs + cover letters, so Match and Ask can be limited to a collection.
3. **Context pack:** select results and choose "Copy as context", which produces clean markdown with paths and snippets to paste into any chatbot.
4. **"Similar to this file" and duplicate finder:** near-duplicate documents, images and code across projects, to reclaim disk space.
5. **Recent-work timeline:** "what was I working on last Tuesday?" Time-filtered search across plans, code and documents.

**Job hunt workflow (builds on Match):**
6. **Auto-match inbox:** a new JD saved to Downloads is automatically scored against your resumes, with a toast showing the result.
7. **Tailored resume generator:** master resume + JD → a tailored DOCX (`python-docx`) and cover letter, with a diff showing the changes.
8. **Application tracker:** company, role, JD, which resume was sent, date, status. A table in SQLite with reminders.

**Developer power features:**
9. **Git history source:** index commit messages and diffs, to answer "when did I fix the auth refresh bug, and in which repo?"
10. **Project cards:** an LLM-written summary per project (stack, purpose, entry points), generated on AC and idle, and searchable.
11. **Cross-project code answers:** "how did I implement JWT refresh before?" Retrieves functions from all projects and explains them.
12. **Reranker role:** an optional `qwen3-reranker` (or cloud) model reorders the top 30 results for sharper ranking.

**More sources (opt-in, privacy-gated):**
13. Browser bookmarks and history (Chrome/Edge SQLite), so you can "find that article about X".
14. Screenshot folder auto-OCR, and clipboard history (opt-in).
15. Email exports (Outlook PST or Gmail Takeout), and Obsidian/Notion exports.

**UX:**
16. Voice query via local `faster-whisper` (push-to-talk).
17. Saved searches that notify when new matching files appear.
18. A `graphify` integration that exports indexed projects into your knowledge-graph tool.

## Verification
- **Scope tests:**
  - Reject: `.credentials.json`, `settings.json`, `.env`, lockfiles, `node_modules/**/.claude/**`, gitignored files, OneDrive placeholders.
  - Accept: `~/.claude/plans/*.md`, memory `.md` files, `CLAUDE.md`, `.cursor/rules/*.mdc`, `Dockerfile`, `.ipynb`.
- **Power:**
  - On battery the queue grows but nothing is embedded, and `nvidia-smi` shows no Ollama runner.
  - Unplugging mid-run makes the worker exit within one batch, and plugging back in resumes from where it stopped.
  - Chat on battery shows "plug in" by default.
- **VRAM:** after indexing exits and after a chat session closes or times out, `nvidia-smi` shows 0 MB for Ollama. The embedder and the LLM never appear loaded together (`ollama ps`).
- **Incremental:** editing one function re-embeds one chunk; a no-op `git checkout` re-embeds 0; a deleted file's rows disappear; changes made while the watcher was stopped are picked up by the reconcile scan.
- **PDF with images:** a query for text that appears only in a PDF diagram returns that PDF and page.
- **Match:**
  - Put 4 resumes (two of them versions of the same one) and 1 JD on disk. Match ranks the expected resume first, shows only the newest of the two versions, the JSON scores parse, and follow-up chat answers "what's missing" using the pinned context.
  - Each scoring call stays under `num_ctx` (logged token counts).
  - The checklist is generated once and reused for every resume.
  - Scores are reproducible across two runs (same checklist → same calculated score, within ±1 partial).
  - A resume that has the skill only on its last page is still marked "met", which proves the whole document is sent.
  - A fabricated `evidence_quote` (unit test with a mocked LLM) is downgraded to "unverified".
  - **Candidate checkboxes:** untick 2 of 6 candidates. The captured requests contain exactly 4 resumes, and an unticked resume's text never appears in any request (test with a mocked provider). "+ Add file" adds a resume that recall missed.
  - **ID masking (always on for cloud):** fixture resumes containing an SSN, passport, DL, Aadhaar/PAN and a card number come out of the captured cloud request with all of them replaced by `[… REMOVED]`.
    - Plain numbers (years, phone numbers, GPA, zip codes) are **not** masked (false-positive test).
    - The "🛡 N items will be masked" preview matches what was actually masked.
  - **Personal details:** with the checkbox off (default), name, email and phone are present in the request. With it on, they are replaced with placeholders and restored in the displayed answer.
  - **"View what will be sent"** shows text identical to the captured request body.
- **Ask:** every answer cites real files and lines, and a question with no answer in the index returns "not found" instead of making something up.
- **Latency:** search under 150 ms; first chat token under about 3 s with the model pre-warmed.
- **Quality:** `python eval/run.py` prints the comparison table, which is used to choose the final models.
- **Model registry:** `ve models` lists the installed Ollama models with roles and capabilities, and flags `mxbai-embed-large` (512-token context) and an unused `deepseek-r1`.
  - Remove `qwen3.5:9b`: the chat role falls back to `qwen2.5-coder:7b` or `llama3.2`, and the UI suggests the pull.
  - Simulate low free VRAM: the resolver picks a smaller model.
- **Providers:**
  - With an OpenRouter key set, the "Answer better ☁" button shows the privacy badge before sending.
  - Paths on the `NEVER_SEND_TO_CLOUD` list are stripped from the cloud context (unit test).
  - The key never appears in `settings.toml`, the logs or the index.
  - On battery with `auto` routing, chat goes to the cloud and `nvidia-smi` shows no local load.
- **Extensibility:** adding a dummy `extractors/epub.py` or `skills/hello.py` with only a decorator makes it appear in the CLI and UI without changing core code (test).
