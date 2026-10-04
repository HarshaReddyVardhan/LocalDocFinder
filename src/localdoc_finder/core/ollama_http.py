"""Talk to Ollama over plain HTTP for the few calls the light processes need.

The watcher and the install hooks must not import the ``ollama`` package (it pulls in httpx and
pydantic); a model unload only needs ``/api/ps`` and ``/api/embed``.
"""

import json
import logging
import urllib.request

from localdoc_finder.core.model_names import same_model

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_SECONDS = 10
_UNLOAD_NOTE = "unload is best effort"


def _ollama_json(host: str, path: str, body: dict[str, object] | None = None) -> object:
    request = urllib.request.Request(  # noqa: S310  # host comes from our own settings
        f"{host.rstrip('/')}{path}",
        method="GET" if body is None else "POST",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    raw = urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS).read()  # noqa: S310
    return json.loads(raw or b"{}")


def _resident_entry(host: str, model: str) -> dict[str, object] | None:
    """The ``/api/ps`` entry for ``model``; ``None`` when it is not loaded."""
    listing = _ollama_json(host, "/api/ps")
    models = listing.get("models") if isinstance(listing, dict) else None
    for entry in models if isinstance(models, list) else []:
        if isinstance(entry, dict) and same_model(str(entry.get("model", "")), model):
            return entry
    return None


def unload_model(host: str, model: str) -> None:
    """``keep_alive=0`` over HTTP: the watcher must not import the ollama package.

    A request that frees a model loads it first, so nothing is sent unless ``/api/ps`` lists it,
    and the same GPU/CPU placement is requested so no second runner is started.
    """
    try:
        resident = _resident_entry(host, model)
        if resident is None:
            return
        body: dict[str, object] = {"model": model, "input": "x", "keep_alive": 0}
        if not resident.get("size_vram"):
            body["options"] = {"num_gpu": 0}
        _ollama_json(host, "/api/embed", body)
    except (OSError, ValueError):
        logger.debug("model unload request failed (%s)", _UNLOAD_NOTE)
