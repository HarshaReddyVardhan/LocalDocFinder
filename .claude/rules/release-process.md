# Release process

Every release must tell users what changed, in the GitHub Release and in the app.

- **Cut a release** by bumping `version` in `pyproject.toml` (commit `chore: release X.Y.Z`), then pushing the tag `vX.Y.Z`. The `release` workflow checks that the two match.
- **Release notes are generated, not optional.** `scripts/release_notes.py` turns the Conventional Commits since the previous tag into Markdown (`feat` → New, `fix` → Fixed, `perf` → Faster, `refactor` → Improved; `chore`, `docs`, `test` and `ci` are left out). `scripts/build.ps1` passes the file to `vpk pack --releaseNotes`, so it becomes the GitHub Release body and the update's `NotesMarkdown`.
- **So commit subjects are the changelog.** Write each `feat:`/`fix:` subject as a short sentence a user understands ("Show-in-folder button on results"), not an internal detail. Put internal work under `chore:`/`refactor:`/`test:`.
- **The app shows them.** `Updater` copies `NotesMarkdown` into `UpdateOutcome.notes`; when an update is downloaded, `app/whats_new.py` opens a "What's new in X" dialog (Restart now / Later) and Settings → Updates shows the same notes. Never restart without the user's choice.
- The release workflow checks out with `fetch-depth: 0`; the notes need the tag history.
- After tagging, open the GitHub Release and confirm its body is not empty.
