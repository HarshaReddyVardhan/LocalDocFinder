"""Make sure the Ollama server is up before the app, the worker or the CLI needs a model.

If it already answers it is used as it is; if it is installed but stopped it is started; if it
is not installed nothing is downloaded (that needs the user's consent in setup) and the caller
reports it. Never raises: callers carry on and fail soft with their own, clearer error.
"""

import logging
import os

from localdoc_finder.core.setup.ollama_install import OllamaSetup, OllamaSetupError, OllamaState
from localdoc_finder.core.setup.windows_ollama import WindowsOllamaSystem

logger = logging.getLogger(__name__)

AUTOSTART_DISABLE_ENV = "LDF_NO_OLLAMA_AUTOSTART"  # set to 1 to never start Ollama (tests, CI)


def ensure_ollama_running(host: str, *, setup: OllamaSetup | None = None) -> OllamaState:
    """Return the state after trying to get Ollama running: ``RUNNING`` means it is ready."""
    if os.environ.get(AUTOSTART_DISABLE_ENV) == "1":
        return OllamaState.RUNNING
    setup = setup or OllamaSetup(WindowsOllamaSystem(host))
    state = setup.detect()
    if state is OllamaState.RUNNING:
        return state
    if state is OllamaState.MISSING:
        logger.warning("ollama: not installed; models are unavailable until it is")
        return state
    logger.info("ollama: installed but not running; starting it")
    try:
        setup.start_server()
    except OllamaSetupError as exc:
        logger.warning("ollama: could not start it: %s", exc)
        return state
    return OllamaState.RUNNING
