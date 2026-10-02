"""PyInstaller entry for ve.exe: the command line."""

import sys

from vector_embed.cli import main

if __name__ == "__main__":
    sys.exit(main())
