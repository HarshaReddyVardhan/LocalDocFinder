"""PyInstaller entry for LocalDocFinder.exe: the dispatcher (app, watcher, worker, setup)."""

import sys

from localdoc_finder.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
