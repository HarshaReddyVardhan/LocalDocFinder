"""The fixed (internal) drives of this PC; removable and network drives are never included."""

import logging

import psutil

logger = logging.getLogger(__name__)


def fixed_drives() -> list[str]:
    r"""Drive roots such as ``C:\`` for every fixed local disk, system drive first."""
    drives: list[str] = []
    try:
        partitions = psutil.disk_partitions(all=False)
    except OSError:
        logger.warning("could not list the drives", exc_info=True)
        return drives
    for partition in partitions:
        options = partition.opts.split(",")
        if "fixed" in options and "cdrom" not in options:
            drives.append(partition.mountpoint)
    return sorted(drives)
