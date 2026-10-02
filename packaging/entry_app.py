"""PyInstaller entry for VectorEmbed.exe: the dispatcher (app, watcher, worker, setup)."""

import sys

from vector_embed.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
