# Plan: Installable LocalDoc Finder (setup wizard, minimal popup, Setup.exe with auto-update)

## Context
Today LocalDoc Finder only runs from a dev checkout: you need Python and a `.venv`, Ollama installed and models pulled by hand, and `scripts/install_task.ps1` pointing at `.venv\Scripts\pythonw.exe`. Goals:
1. **A Setup.exe for any Windows PC.** It bundles its own Python and libraries, never touches a global Python, needs no admin rights, and **updates itself**.
2. **A first-run setup wizard.** It installs Ollama if missing (with consent), **auto-picks an embedder and a chat model that fit the hardware** (the user can change them), downloads them, **speed-tests them**, then shows all user settings once. Settings are saved per Windows user in `%LOCALAPPDATA%\LocalDocFinder\settings.toml`.
3. **A minimal hotkey popup.** Just a search bar and an Explorer-style result list (icon, file name, full path underneath). Ask/Chat/Match stay reachable (`?`/Tab) but hidden. Models, health and all settings move to a separate **Settings window** (opened from the tray).

Decisions: Windows only · official OllamaSetup.exe downloaded on first run · NVIDIA or CPU-only tiers · PyInstaller one-folder build · **Velopack** for the installer and updates (it replaces Inno Setup; Inno Setup was only a "Setup.exe wizard maker" and cannot update) · updates published to **GitHub Releases**.

**Reused, not rebuilt:**
- `models_catalog.toml` + `catalog.py` (the supported-model list).
- `hardware.probe_hardware()`.
- `ModelRegistry._fits/_budget_mb`.
- `OllamaProvider.pull()/prewarm()/unload()`.
- `ModelManager`, `settings_io.set_setting`, `ModelsPanel`.
- `doctor.py` checks.
- `controller.result_label` data (name, path, project, snippet).

Before Step 1: commit or stash the uncommitted changes (`runtime.py`, `skills/ask.py`, `pyproject.toml`, `uv.lock`).

## Steps (one commit each; ruff/mypy/pytest green before each)

### 1. `feat(models): pick a starter model set for a machine with nothing installed`
- Catalog: add `download_mb` (to show sizes and check disk space) and `min_ram_mb` for models marked `cpu_ok`. Extend `CatalogModel`.
- New pure module `core/models/starter.py`: `pick_starter(catalog, hardware, roles) -> StarterPlan`.
  - For each role it takes the first model in `preferences(role)` that fits **total** VRAM. With no GPU it uses a fraction of total RAM and only `cpu_ok` models.
  - It returns the total download size and a reason per role, plus the next smaller fitting model per role (used by the speed test in Step 3).
- Pull the fit check out of `ModelRegistry._fits` into a shared pure helper.
- Expected picks: 8 GB → `qwen3.5:9b` + `qwen3-embedding:0.6b`; 4 GB → `llama3.2` + `qwen3-embedding:0.6b`; CPU → `llama3.2` + a small embedder.
- Tests: table-driven over fake `Hardware` profiles.

### 2. `feat(setup): detect, download and start Ollama`
`core/setup/ollama_install.py`, with adapters injected so tests can fake them:
- `detect()` → RUNNING / INSTALLED_NOT_RUNNING / MISSING. It pings the host, then looks on PATH and in `%LOCALAPPDATA%\Programs\Ollama`.
- `download_installer()` (progress callback).
- `verify_signature()`: Authenticode must be Valid with the Ollama publisher. Confirm the exact signer name during implementation.
- `run_installer()` (`/VERYSILENT /SUPPRESSMSGBOXES /NORESTART`, per-user), then wait until the server is up.
- `start_server()`, `free_disk_mb()` (`OLLAMA_MODELS` or `~/.ollama/models`).
- Tests with fakes only. No network.

### 3. `feat(models): speed test after download`
`core/models/benchmark.py`:
- `bench_chat(provider, model)`: a short fixed prompt of about 64 tokens. Tokens/sec comes from Ollama's `eval_count/eval_duration`, and load time is measured too.
- `bench_embed(provider, model)`: a fixed batch of texts, giving embeds/sec.
- Runs **one model at a time and unloads in `finally`**, so the embedder and the chat LLM are never on the GPU together and VRAM goes back to 0.
- Pure `judge(result) -> ok | slow`, with thresholds as module constants (e.g. chat < 8 tok/s = slow). When a model is slow, the flow offers the next smaller fitting model from `StarterPlan`.
- Results are stored in `StateDb` meta and shown in the Settings → Models tab.
- Tests: fake provider responses, judge thresholds, and unload on error.

### 4. `feat(setup): resumable setup flow and ldf setup`
- `core/setup/flow.py`, `SetupFlow`: *Ollama → pick → disk check → pull (skipping models already installed; Ollama resumes partial pulls) → speed test (with downgrade offer) → write settings → mark done*. The plan is pure; execution goes through adapters and reports progress through a callback.
- It writes `embedding.model` and the chat override via `set_setting`, and stores `setup_completed_at` in the `StateDb` meta.
- `ldf setup [--yes] [--embed X] [--chat Y] [--extras ...] [--dry-run] [--no-install-ollama] [--skip-bench]`.
- Add a "setup not run" check to `doctor.py`.
- Tests: `tests/core/setup/test_flow.py`, `tests/core/test_cli_setup.py`.

### 5. `feat(app): settings window`
- `app/settings_window.py`: tabbed window opened from the tray ("Settings…").
  - **General**: hotkey, start with Windows, folders to index (scope roots).
  - **Models & Health**: the existing `ModelsPanel` moves here unchanged, plus the speed-test results.
  - **Cloud & Privacy**: the existing cloud and key controls.
  - **Updates**: check now, auto-check on/off.
  - **About**: version, data folder, "Delete my data".
- Every change goes through `set_setting` into the user's `settings.toml`.
- Each tab is a small widget; one settings controller keeps the window thin.
- Tests: in the style of `tests/core/app/`.

### 6. `feat(app): first-run setup wizard`
- `app/setup_wizard.py` (`QWizard`) + `app/setup_controller.py`, which runs `SetupFlow` on a worker thread and reports progress with signals, following the `models_panel.py` pattern. Pages:
  1. Welcome/hardware.
  2. Ollama: found / "Install (~X MB from ollama.com)" consent / start.
  3. Models: embedding and chat dropdowns pre-filled with the auto-picks, listing every catalog model with its size and a "won't fit" warning; extras as checkboxes; disk check.
  4. Download progress.
  5. Speed test, with the downgrade offer.
  6. **Your settings**: the Settings-window tabs reused in the wizard.
  7. Done: shows the hotkey.
- `app/main.py` runs the wizard when setup is not complete. The tray gets "Settings…" and "Run setup again…".

### 7. `feat(app): minimal search popup with Explorer-style results`
- `window.py`: the popup opens as just a search bar and results. The mode label is hidden in Search mode. `Mode.MODELS` and `ModelsPanel` are removed from the popup (they now live in Settings).
  - `?`/Tab still reach Ask/Chat/Match. The mode label appears only while one of them is active.
- New `app/result_delegate.py` (`QStyledItemDelegate`) draws each row:
  - the real Windows file icon (`QFileIconProvider`, cached per extension);
  - the file name in bold, with symbol/page;
  - the full path underneath in grey, as in Explorer;
  - modified date/size on the right;
  - the snippet only on the selected row.
- The keys stay the same: Enter opens, Ctrl+Enter reveals in Explorer, Shift+Enter opens in VS Code.
- `controller.result_label` turns into a small data object for the delegate; existing tests are updated.

### 8. `refactor: make entry points work when frozen`
- `core/process.py` gets `self_command(entry)`: `[sys.executable, "-m", "localdoc_finder.<entry>"]` when unfrozen, `[...\LocalDocFinder.exe, entry]` when frozen. `watcher.py` and `app/controller.py` use it.
- `localdoc_finder/__main__.py` dispatches `app|watcher|worker|setup`, with `app` as the default.
- Move the logic in `install_task.ps1` into `core/autostart.py`, which registers the Task Scheduler tasks through `schtasks`/PowerShell with the same battery flags. The installer hooks and the "start with Windows" setting can then call it. The script stays as a thin wrapper.
- Tests: `self_command` with `sys.frozen` patched, and the autostart command line it builds.

### 9. `build: PyInstaller build and Velopack installer with auto-update`
- `packaging/localdoc_finder.spec`: one folder `dist\LocalDocFinder\` containing `LocalDocFinder.exe` (windowed) and `ldf.exe` (console).
  - Collect `models_catalog.toml`, `tree_sitter_language_pack`, `lancedb`, `pyarrow` and the `winrt.*` packages.
  - Exclude the dev packages.
- `velopack` (PyPI) is a runtime dependency. `LocalDocFinder.exe` runs `velopack.App()` first, with hooks:
  - **after install**: register autostart;
  - **before uninstall**: remove the scheduled tasks and stop the processes;
  - **after update**: restart the watcher.
  - Confirm the exact Python hook API names against the Velopack docs during implementation.
  - Uninstall keeps `%LOCALAPPDATA%\LocalDocFinder` (index and settings) and Ollama; "Delete my data" in Settings removes the data.
- `core/updates.py`: checks GitHub Releases in the background at startup and then once a day (setting `updates.auto_check`). It downloads only the changed parts (delta updates), then a tray message offers "Restart to update". New model lists arrive with app updates.
- `scripts/build.ps1`:
  1. `uv sync --group build`
  2. `pyinstaller packaging\localdoc_finder.spec`
  3. smoke test `ldf.exe doctor`
  4. `vpk pack --packId LocalDocFinder --packVersion <pyproject version> --packDir dist\LocalDocFinder --mainExe LocalDocFinder.exe`
  5. optional `vpk upload github`

  Output: `LocalDocFinder-win-Setup.exe` + update packages. The build machine needs the .NET SDK for `vpk`; end users don't.
- `.gitignore` gets `build/`, `dist/` and `Releases/`. Unsigned builds show a SmartScreen warning; code signing is optional later.

### 10. `docs: install, build and release instructions`
- README: Install, first-run setup, choosing models manually, Settings window, building and releasing.
- `.env.example`: `LDF_UPDATES__AUTO_CHECK`.
- `.claude/CLAUDE.md` status.

## Invariants kept
- Nothing is downloaded or installed without explicit consent.
- The Ollama installer is signature-checked before it runs.
- The embedder stays pinned: setup writes `embedding.model`, and later changes still go through `change_embedder`.
- The benchmark never puts the embedder and the chat model on the GPU together, and unloads in `finally`.
- `core/` has no UI imports: wizard, settings window and CLI are thin layers over `SetupFlow` and the settings.

## Verification
- After each step: `ruff format`, `ruff check`, `mypy`, `pytest` (coverage ≥ 80%; setup, starter and benchmark modules close to 100%).
- On this RTX 2070: `ldf setup --dry-run` → `qwen3.5:9b` + `qwen3-embedding:0.6b` with the correct size. `ldf setup --skip-bench=false` prints tok/s, and `ldf health` shows VRAM back to 0 afterwards.
- Popup: the hotkey shows only the search bar; results show the icon, the name and the path underneath; `?` still asks; Models is gone from Tab and present in Settings.
- Build: `scripts\build.ps1`; `dist\LocalDocFinder\ldf.exe doctor` passes.
- Clean machine (**Windows Sandbox**, no Python, no Ollama):
  1. Run Setup.exe → wizard → Ollama installs → models download → speed test → settings page → hotkey search works.
  2. Bump the version, rebuild, `vpk upload` → the installed app finds the update, restarts and is on the new version.
  3. Uninstall removes the app and its tasks.
