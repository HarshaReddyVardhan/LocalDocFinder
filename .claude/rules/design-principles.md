# Design principles

## General
- **Single responsibility.** One module/class/function does one thing. Split when a name needs "and".
- **Open/closed via registries.** New extractors, doctypes, providers, skills and sources are added by a new file with a `@register` decorator; core code is not edited.
- **Depend on abstractions.** Skills and workers use `ChatProvider` / `EmbedProvider` protocols, never `ollama` or `openai` directly. Inject collaborators (clock, power probe, provider, store) so tests can fake them.
- **Liskov / interface segregation.** Keep protocols small (`typing.Protocol`); every implementation must honour the full contract.
- **KISS, DRY, YAGNI.** Build the simplest thing the plan's current step needs. Extract duplication on the third use, not the second. No speculative options.
- **Pure core, impure edges.** Decision logic (scope rules, scoring, masking, routing) is pure and unit-tested; I/O (disk, network, OS APIs) sits behind thin adapters.
- **Make illegal states unrepresentable.** Use frozen dataclasses, enums and pydantic models over loose dicts and magic strings.

## Errors and safety
- Fail fast on invalid config at startup; fail soft per-file during indexing (log, record, continue).
- Catch specific exceptions only; never bare `except:`; never swallow silently. A broad `except Exception` is allowed only at a worker/UI boundary and must log.
- Validate at boundaries (settings, files, LLM JSON output, user input); trust internal calls.
- Secrets/privacy checks run first and cannot be bypassed by later rules.
- Treat all LLM output as untrusted: parse into schemas, verify evidence quotes, never execute it.

## Code style
- Small functions (aim < 40 lines), shallow nesting, early returns, descriptive names, no abbreviations except well-known ones.
- Comments explain *why*, not *what*. Public modules/classes/functions get a short docstring.
- Constants live in settings or a module-level `UPPER_CASE` with a unit; no magic numbers inline.
- Paths via `pathlib.Path`; never hand-build separators. Windows path semantics are case-insensitive: normalise before comparing.
- Resources (files, DB connections, model sessions) use context managers and are always released; GPU models are unloaded in `finally`.

## Performance and resources
- Idle cost must be near zero: no polling loops faster than needed, no model loads at import time.
- Batch I/O and embeddings; hash before re-embedding; check power/idle gates before every batch.
