"""Pure "does this model fit this machine" rules, shared by the registry and the starter picker."""

from vector_embed.core.models.catalog import CatalogModel
from vector_embed.core.models.hardware import Hardware

CPU_RAM_FRACTION = 0.5  # share of RAM a CPU-only machine may devote to a model


def budget_mb(hardware: Hardware, *, total: bool = False) -> int:
    """Memory a model may use: VRAM on a GPU machine, a fraction of RAM otherwise.

    ``total=False`` uses what is free right now (the runtime resolver); ``total=True`` uses the
    machine's capacity (choosing models for a machine that has nothing installed yet).
    """
    if hardware.has_gpu:
        return hardware.vram_total_mb if total else hardware.vram_free_mb
    ram = hardware.ram_total_mb if total else hardware.ram_free_mb
    return int(ram * CPU_RAM_FRACTION)


def fits(
    entry: CatalogModel | None, needed_mb: int | None, hardware: Hardware, budget: int
) -> bool:
    """True if a model needing ``needed_mb`` may run within ``budget`` on ``hardware``.

    Without a GPU only catalog models flagged ``cpu_ok`` qualify, and only when the machine has
    the RAM the entry asks for. An unknown size is not held against a model.
    """
    if not hardware.has_gpu and entry is not None:
        if not entry.cpu_ok:
            return False
        if entry.min_ram_mb is not None and hardware.ram_total_mb < entry.min_ram_mb:
            return False
    return needed_mb is None or needed_mb <= budget
