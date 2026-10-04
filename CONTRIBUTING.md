# Contributing to LocalDoc Finder

Thank you for your interest in improving LocalDoc Finder. This page covers what you need to make your first change. The **[Developer guide](docs/DEVELOPER_GUIDE.md)** has the full reference: architecture, configuration, testing, packaging and releases.

## Contents

- [Ways to contribute](#ways-to-contribute)
- [Set up in five minutes](#set-up-in-five-minutes)
- [Make a change](#make-a-change)
- [Quality bar](#quality-bar)
- [Commit messages](#commit-messages)
- [Pull requests](#pull-requests)
- [Design rules that are not negotiable](#design-rules-that-are-not-negotiable)
- [Reporting bugs and security issues](#reporting-bugs-and-security-issues)

## Ways to contribute

- **Report a bug** with the steps to reproduce it, what you expected, and the output of `ldf doctor`.
- **Suggest a feature** by opening an issue that describes the problem first, then your idea.
- **Add support for a file format, model provider or skill.** These are plug-ins: a new file with a `@register` decorator, with no edits to core code.
- **Improve the docs**, including this page.

## Set up in five minutes

You need Windows 10 or 11, [Git](https://git-scm.com), [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/download). A GPU is optional.

```powershell
git clone https://github.com/HarshaReddyVardhan/LocalDocFinder.git
cd LocalDocFinder
uv sync                                  # creates .venv with the pinned dependencies
.venv\Scripts\pre-commit install         # runs the checks on every commit
ollama pull qwen3-embedding:0.6b         # the search model
.venv\Scripts\ldf doctor                 # confirms everything is ready
.venv\Scripts\python -m localdoc_finder app --show
```

The last command starts the desktop app from source. See [Running from source](docs/DEVELOPER_GUIDE.md#4-running-from-source) for the watcher, the worker and a scratch data folder that keeps your real index untouched.

## Make a change

1. Create a short-lived branch from `main`: `feat/…`, `fix/…`, `docs/…` or `chore/…`.
2. For a bug, write a failing test that reproduces it **before** fixing it.
3. Keep each commit to one logical change that leaves the tree green.
4. Run the checks below, then open a pull request.

## Quality bar

All four must pass before a change is ready for review:

```powershell
.venv\Scripts\ruff format . ; .venv\Scripts\ruff check . --fix   # formatting and lint
.venv\Scripts\mypy                                                # strict typing on src/
.venv\Scripts\python -m pytest                                    # tests, coverage floor 80%
.venv\Scripts\pre-commit run --all-files                          # all of the above, plus whitespace fixers
```

Tests run offline and deterministically. Ollama, HTTP, Windows APIs and GPU probes are faked. Tests that need a real model carry `@pytest.mark.ollama` and skip when it is absent.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, with an optional scope such as `feat(extractors): …`. Describe the change from the user's point of view where you can. Please don't add `Co-Authored-By` or AI-attribution trailers.

## Pull requests

A good pull request:

- explains **why** the change is needed and links the issue it addresses
- includes tests for new behaviour and regression tests for fixes
- updates the README or the developer guide when behaviour or setup changes
- passes CI (lint, types, tests and a dependency audit run on every pull request)

Small, focused pull requests are reviewed fastest.

## Design rules that are not negotiable

These protect users' privacy and their PCs; a change that breaks one is not merged.

- Indexing never runs on battery, and models are unloaded after work.
- The embedding model and the chat model are never on the GPU at the same time.
- Secrets are never indexed or sent to a cloud provider. Government and financial ID numbers are always masked before any cloud call.
- `core/` has no UI imports; the app, CLI and MCP server stay thin.
- New behaviour is added through registries (`@register`), not by editing core switch statements.

The reasoning behind each is in the [Developer guide](docs/DEVELOPER_GUIDE.md#7-design-principles).

## Reporting bugs and security issues

Open a [GitHub issue](https://github.com/HarshaReddyVardhan/LocalDocFinder/issues) for bugs and ideas. Please include your Windows version, whether you have an NVIDIA GPU, and the relevant lines from the log in `%LOCALAPPDATA%\LocalDocFinderData\logs`. Remove anything personal first.

For a security problem, such as a way to make the app read, index or send something it should not, please **don't open a public issue**. Contact the maintainer privately through their [GitHub profile](https://github.com/HarshaReddyVardhan) instead.
