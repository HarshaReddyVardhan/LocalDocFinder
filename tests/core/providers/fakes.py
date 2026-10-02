"""Test doubles that return real ollama response types."""

from collections.abc import Callable, Iterator
from typing import Any

import ollama
from ollama._types import ModelDetails


class FakeOllamaClient:
    """Records calls; ``embed_fn`` maps input texts to vectors."""

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        models: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.embed_fn = embed_fn or (lambda texts: [[3.0, 4.0, 0.0, 0.0] for _ in texts])
        self.models = models or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.failures: list[BaseException] = []  # raised (in order) before a call succeeds
        self.chat_reply = ["Hel", "lo"]
        self.chat_json_reply = '{"ok": true}'
        self.chat_json_fn: Callable[[dict[str, Any]], str] | None = None  # reply per request
        self.loaded: list[str] = []
        self.loaded_on_cpu: set[str] = set()  # resident models with no VRAM

    def _maybe_fail(self) -> None:
        if self.failures:
            raise self.failures.pop(0)

    def embed(self, **kwargs: Any) -> ollama.EmbedResponse:
        self.calls.append(("embed", kwargs))
        self._maybe_fail()
        return ollama.EmbedResponse(
            embeddings=self.embed_fn(kwargs["input"]), prompt_eval_count=len(kwargs["input"])
        )

    def chat(self, **kwargs: Any) -> Any:
        self.calls.append(("chat", kwargs))
        self._maybe_fail()
        if kwargs.get("stream"):
            return self._stream()
        reply = self.chat_json_fn(kwargs) if self.chat_json_fn else self.chat_json_reply
        return ollama.ChatResponse(
            message=ollama.Message(role="assistant", content=reply),
            done=True,
            prompt_eval_count=7,
            eval_count=3,
        )

    def _stream(self) -> Iterator[ollama.ChatResponse]:
        for piece in self.chat_reply:
            yield ollama.ChatResponse(message=ollama.Message(role="assistant", content=piece))
        yield ollama.ChatResponse(
            message=ollama.Message(role="assistant", content=""),
            done=True,
            prompt_eval_count=11,
            eval_count=5,
        )

    def generate(self, **kwargs: Any) -> ollama.GenerateResponse:
        self.calls.append(("generate", kwargs))
        self._maybe_fail()
        return ollama.GenerateResponse(response="")

    def ps(self) -> ollama.ProcessResponse:
        self.calls.append(("ps", {}))
        return ollama.ProcessResponse(
            models=[
                ollama.ProcessResponse.Model(
                    model=name, size_vram=0 if name in self.loaded_on_cpu else 1_000_000
                )
                for name in self.loaded
            ]
        )

    def list(self) -> ollama.ListResponse:
        self.calls.append(("list", {}))
        self._maybe_fail()
        return ollama.ListResponse(
            models=[
                ollama.ListResponse.Model(
                    model=name,
                    size=meta.get("size", 1),
                    details=ModelDetails(
                        parameter_size=meta.get("params"), quantization_level=meta.get("quant")
                    ),
                )
                for name, meta in self.models.items()
            ]
        )

    def show(self, name: str) -> ollama.ShowResponse:
        self.calls.append(("show", {"name": name}))
        meta = self.models[name]
        if meta.get("show_fails"):
            raise ollama.ResponseError("boom", 500)
        return ollama.ShowResponse(
            capabilities=meta.get("caps", ["completion"]),
            model_info=meta.get("info", {}),
        )

    def pull(self, name: str, stream: bool = False) -> Iterator[ollama.ProgressResponse]:
        self.calls.append(("pull", {"name": name, "stream": stream}))
        yield ollama.ProgressResponse(status="pulling", completed=50, total=100)
        yield ollama.ProgressResponse(status="success")
