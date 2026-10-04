from pathlib import Path

import pytest

from localdoc_finder.core.models.benchmark import BenchKind, BenchResult
from localdoc_finder.core.models.hardware import Hardware
from localdoc_finder.core.settings import Settings
from localdoc_finder.core.setup import wiring
from localdoc_finder.core.setup.flow import SetupFlow
from localdoc_finder.core.store.sqlite import StateDb


def test_build_flow_wires_the_real_collaborators_without_touching_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        wiring, "probe_hardware", lambda: Hardware("RTX 2070", 8192, 7000, 32000, 16000, 8, True)
    )
    settings = Settings(storage={"data_dir": tmp_path})  # type: ignore[arg-type]
    with StateDb(tmp_path) as state:
        flow = wiring.build_flow(settings, state, lambda _event: None, lambda _offer: False)
    assert isinstance(flow, SetupFlow)


def test_the_embed_benchmark_uses_the_requested_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_bench_embed(provider: object) -> BenchResult:
        seen.append(provider.embed_model)  # type: ignore[attr-defined]
        return BenchResult(seen[-1], BenchKind.EMBED, 50.0, 0.1)

    monkeypatch.setattr(wiring, "bench_embed", fake_bench_embed)
    monkeypatch.setattr(
        wiring, "probe_hardware", lambda: Hardware(None, 0, 0, 16000, 8000, 8, True)
    )
    settings = Settings(storage={"data_dir": tmp_path})  # type: ignore[arg-type]
    with StateDb(tmp_path) as state:
        flow = wiring.build_flow(settings, state, lambda _event: None, lambda _offer: False)
        result = flow._bench_fns["embed"]("nomic-embed-text")  # type: ignore[misc]
    assert seen == ["nomic-embed-text"]
    assert result.model == "nomic-embed-text"
