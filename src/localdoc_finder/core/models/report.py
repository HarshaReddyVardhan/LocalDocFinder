"""Plain-text rendering of the model registry report (CLI and doctor output)."""

from localdoc_finder.core.models.registry import FLAG_OK, FLAG_UNUSED, FLAG_WARNING, Report

_GB = 1024**3
_MARK = {FLAG_OK: "ok  ", FLAG_WARNING: "warn", FLAG_UNUSED: "unused"}


def format_report(report: Report) -> str:
    hw = report.hardware
    gpu = (
        f"{hw.gpu_name}, {hw.vram_free_mb}/{hw.vram_total_mb} MB VRAM free"
        if hw.has_gpu
        else "no NVIDIA GPU (CPU-friendly models only)"
    )
    lines = [
        f"hardware: {gpu}; RAM {hw.ram_free_mb} MB free; {'AC' if hw.on_ac else 'battery'}",
        "",
        "roles:",
    ]
    for role, resolution in report.resolutions.items():
        chosen = resolution.model or "(none)"
        lines.append(f"  {role:<13} {chosen:<28} {resolution.reason}")
    lines += ["", "installed models:"]
    if not report.rows:
        lines.append("  (none found; is Ollama running?)")
    for row in report.rows:
        size = f"{(row.info.size_bytes or 0) / _GB:.1f} GB"
        roles = ",".join(row.roles) or "-"
        caps = ",".join(sorted(row.info.capabilities)) or "?"
        lines.append(f"  {row.info.name:<30} {size:>8}  roles: {roles:<14} caps: {caps}")
        for flag in row.flags:
            if flag.kind != FLAG_OK:
                lines.append(f"      {_MARK[flag.kind]}: {flag.message}")
    if report.recommendations:
        lines += ["", "recommendations:"]
        for rec in report.recommendations:
            lines.append(
                f"  {rec.role}: better option available: `{rec.pull_command}` ({rec.reason})"
            )
    return "\n".join(lines)
