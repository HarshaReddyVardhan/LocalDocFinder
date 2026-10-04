"""PyInstaller entry for ldf.exe: the command line."""

import sys

from localdoc_finder.cli import main

if __name__ == "__main__":
    sys.exit(main())
