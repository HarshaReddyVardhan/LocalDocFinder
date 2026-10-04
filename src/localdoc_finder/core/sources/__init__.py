"""Where indexed content comes from. One file per source, registered with ``@SOURCES.register``.

The filesystem is the only source today; browser history, git history, email or the clipboard
would each be one new file here. Reconciliation scans every registered source.
"""
