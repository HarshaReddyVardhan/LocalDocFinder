<div align="center">

<img src="docs/assets/logo.png" alt="LocalDoc Finder logo" width="112" height="112">

# LocalDoc Finder

**Find anything on your PC by what it says, not what it's called.**

Private, on-device search for your documents, scans, images and code, with optional answers and chat about your files. Nothing leaves your computer.

<a href="https://github.com/HarshaReddyVardhan/LocalDocFinder/releases/latest"><img src="https://img.shields.io/badge/Download%20for%20Windows-2563EB?style=for-the-badge&logo=windows&logoColor=white" alt="Download for Windows" height="44"></a>

[![Latest release](https://img.shields.io/github/v/release/HarshaReddyVardhan/LocalDocFinder?label=release&color=2563EB)](https://github.com/HarshaReddyVardhan/LocalDocFinder/releases/latest)
[![CI](https://github.com/HarshaReddyVardhan/LocalDocFinder/actions/workflows/ci.yml/badge.svg)](https://github.com/HarshaReddyVardhan/LocalDocFinder/actions/workflows/ci.yml)
![Platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-0078D4)
![Runs offline](https://img.shields.io/badge/runs-100%25%20on%20device-16A34A)

[Features](#features) · [What it reads](#what-it-reads) · [Requirements](#requirements) · [Install](#install) · [Using it](#using-it) · [Privacy](#privacy-and-security) · [For developers](#for-developers)

</div>

---

## Why LocalDoc Finder

Windows search matches file names. LocalDoc Finder understands content. Ask for *"the lease clause about late rent"* or *"where we retry failed payments"* and it finds the right page of a scanned PDF, the paragraph in a Word file, or the function in your code, even when none of those words is in the file name.

Everything runs on your own PC with open models through [Ollama](https://ollama.com). There is no account, no subscription and no telemetry.

## Features

- **Search by meaning.** Press **Ctrl+Alt+Space** anywhere and type. Results appear instantly, with the matching snippet, page or function.
- **Reads what other tools skip.** Scanned PDFs, photos of documents, screenshots and pictures inside Word, PowerPoint and RTF files are read with the OCR built into Windows.
- **You choose what is indexed.** The whole PC or chosen folders, and which kinds of file: documents, notes, images and scans, code, or data.
- **Ask, Chat and Match (optional).** Get answers with citations to your own files, hold a conversation about a folder, or rank documents against a job description. Switch them on when you want them.
- **Gentle on your PC.** Indexing waits until the PC is idle and plugged in, never runs on battery, and unloads models as soon as it is done.
- **Works on any modern PC.** An NVIDIA GPU is used automatically when present; without one, setup picks models sized for your CPU and memory.
- **Keeps itself up to date.** Updates download in the background and apply on the next restart.
- **Works with AI coding tools.** A built-in MCP server lets Claude Code, Cursor and similar tools search your files locally.

## What it reads

| Kind | Formats | Notes |
| --- | --- | --- |
| Documents | PDF, Word (`.docx`), PowerPoint (`.pptx`), RTF | Text, tables, speaker notes, and the text inside pictures and scanned pages |
| Text and notes | `.txt`, Markdown, LaTeX | Split by heading so results point to the right section |
| Images and scans | JPG, PNG, TIFF (multi-page), WebP, BMP, SVG | Text in the image is read with Windows OCR |
| Source code | Python, JavaScript/TypeScript, C#, Java, Go, Rust, C/C++ and 20+ more | Indexed per function and class |
| Data and config | JSON, YAML, XML, CSV, SQL, INI, TOML | Large data dumps are skipped |

Pick any combination during setup or later in **Settings → General → File types**. Documents, and text and notes, are on by default.

Some things are never indexed, whatever you choose: Windows and program folders, other users' profiles, build and cache folders, and files that look like passwords or keys.

## Requirements

| | Minimum | Recommended |
| --- | --- | --- |
| Operating system | Windows 10 or 11, 64-bit | Windows 11 |
| Memory | 4 GB RAM (search only) | 8 GB RAM or more (search plus Ask, Chat and Match) |
| Graphics | Not required | Any NVIDIA GPU with 4 GB+ of video memory, for faster answers |
| Disk space | About 5 GB for Ollama and a search model | 10 GB+ if you add a larger chat model |

You don't need to install Python or anything else. The installer brings everything it needs, and setup offers to install Ollama for you.

## Install

1. **[Download the latest installer](https://github.com/HarshaReddyVardhan/LocalDocFinder/releases/latest)** (`LocalDocFinder-win-Setup.exe`) and run it.
   It installs for your Windows account only and needs no administrator rights.
2. The setup wizard checks your PC, installs Ollama if it is missing (you are asked first), downloads models that suit your hardware, and asks which folders and kinds of file to index.
3. Press **Ctrl+Alt+Space** and start searching. The first index builds in the background while your PC is idle.

Prefer the command line? This downloads and runs the latest installer:

```powershell
irm https://raw.githubusercontent.com/HarshaReddyVardhan/LocalDocFinder/main/scripts/install.ps1 | iex
```

> **Windows SmartScreen:** the installer is not code-signed yet, so Windows may warn you the first time. Choose **More info → Run anyway**.

## Using it

| Key | Action |
| --- | --- |
| **Ctrl+Alt+Space** | Open the search popup from anywhere |
| **Enter** | Open the selected file |
| **Ctrl+Enter** | Show it in File Explorer |
| **Shift+Enter** | Open it in VS Code at the matching line |
| **Tab** | Switch between Search, Ask, Chat and Match |
| **Esc** | Hide the popup |

Narrow a search with filters: `type:pdf`, `type:img`, `type:code`, `ext:docx`, `in:D:\Contracts`, `after:2026-01`, `before:2026-06`.

Everything else, including folders, file types, models, the hotkey, features, updates and privacy, is in **Settings**: click the gear in the popup, or right-click the tray icon.

## Cloud models (optional)

Everything works with local models. If you also want a hosted model for harder questions, add one in **Settings → Cloud & Privacy**:

1. Press **+ Add provider…**, pick the service (OpenRouter, OpenAI, Google Gemini, Anthropic, Groq, Mistral, DeepSeek, or any OpenAI-compatible address under *Custom*) and paste your API key. **Get a key ↗** opens the service's key page. **Test & save** checks the key by listing the models, and saves nothing if it fails.
2. Under **Models**, pick the model for **Ask & Chat** and, if you like, a different one for **Match**. The list is searchable (OpenRouter has hundreds of models), shows context size and price per million tokens when the service publishes one, and a ★ keeps favourites at the top. **Refresh list** reloads it.
3. If you set a monthly spend limit and a model has no published price, enter its input and output price there so the limit can be enforced. Without a price the limit blocks the call.

Keys go to Windows Credential Manager. With several providers, the **Active** radio chooses which one is used.

**When the cloud is used.** Nothing leaves your PC unless you agree. By default only the **Answer better ☁** button in the popup uses the cloud, and it first shows exactly what would be sent (with IDs masked). In Settings you can change that per feature, for Ask & Chat and for Match:

| Setting | Meaning |
| --- | --- |
| Only when I press Answer better | The default. Local models do the work. |
| When the local model can't | Local first; the cloud when no local model fits or the PC is on battery. |
| Always use the cloud | Every request goes to the cloud. |

Routed requests show the same "what will be sent" dialog. Tick **Don't ask again until LocalDoc Finder restarts** to skip it for the rest of the session; **Forget "don't ask again"** in Settings takes that back, and restarting the app does too.

**If the cloud fails** (rate limit, outage, rejected key, unknown model or a used-up budget) a routed request is answered by the local model instead, with a note saying so. You can turn this off in Settings. A request you made with **Answer better** shows the error instead.

The **Model ▾** button next to **Answer better ☁** switches between your providers' models, favourites included, without opening Settings; **Manage…** opens the Cloud tab.

The same setup from the command line:

```powershell
ldf cloud add or --preset openrouter --use   # --preset fills in the address; also: openai, gemini, anthropic, groq, mistral, deepseek
ldf keys set or                              # prompts for the key; it is never shown
ldf cloud models or                          # the chat models the service offers, with prices when known
ldf cloud add or --model chat=vendor/model   # choose the model for a role
ldf cloud route chat=auto match_scorer=local # when each feature uses the cloud
ldf cloud budget 10                          # monthly limit in USD, or "off"
ldf cloud status
```

## Privacy and security

- **Local by default.** Files, the search index and every model stay on your PC.
- **Cloud is opt-in.** If you connect a cloud model for answers, passwords, keys and other secrets are never sent, and government and financial ID numbers are masked first. Each request is shown to you before it is sent, unless you chose "don't ask again" for the session.
- **API keys** are stored in Windows Credential Manager, never in files or logs.
- **Your data is yours.** The index lives in `%LOCALAPPDATA%\LocalDocFinderData`. Uninstalling keeps it, and **Settings → About → Delete my data** removes it.

## Updating and uninstalling

LocalDoc Finder checks for updates at start and once a day, downloads only what changed, and offers **Restart to update** from the tray. To remove it, open Windows **Settings → Apps → Installed apps** and uninstall LocalDoc Finder. Ollama and its models are left in place for other apps.

## For developers

LocalDoc Finder is written in Python (PySide6, LanceDB, Ollama) with a strict, fully tested core.

- **[Contributing guide](CONTRIBUTING.md)**: set up the project, the workflow and how to submit changes.
- **[Developer guide](docs/DEVELOPER_GUIDE.md)**: architecture, running from source, configuration, testing, building the installer and releasing.

## License

No open-source license has been granted yet; all rights are reserved by the author. Use of the app is covered by the terms shown during setup.
