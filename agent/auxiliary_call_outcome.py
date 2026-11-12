"""How one auxiliary logical call ended, for its Relay ``hermes.logical_llm_call`` scope.

The scope's ``outcome`` + ``error_class`` become the shared-metrics ``model_route`` row of an
auxiliary call, under the same contract as a primary call: a failure carries the classifier's
``FailoverReason`` for the error that ended it, a Hermes-side abort (``/stop``, Ctrl+C, an
interrupt, shutdown) is ``cancelled`` rather than ``failed``, and a success keeps the last
attempt error it recovered from (a retried timeout, a fallback past a 429).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Attribute an aux "invalid response" error carries when the HTTP-200 body held a provider
# ``error`` object instead of choices (aggregators relay upstream failures this way).
PROVIDER_ERROR_ATTR = "provider_error"


class _EmbeddedProviderError(Exception):
    """An HTTP-200 ``error`` object reshaped so ``classify_api_error`` reads it as an SDK body."""

    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(str(error.get("message") or ""))
        self.body = {"error": error}


def _error_fields(error: Any) -> dict[str, Any] | None:
    """Normalize a provider error represented by a dict or SDK model."""
    if isinstance(error, dict):
        return error or None
    if error is None:
        return None
    fields = {
        key: getattr(error, key, None)
        for key in ("message", "code", "type", "metadata", "status")
    }
    extra = getattr(error, "model_extra", None)
    if isinstance(extra, dict):
        fields.update(extra)
    normalized = {key: value for key, value in fields.items() if value is not None}
    return normalized or None


def embedded_choice_provider_error(response: Any) -> dict[str, Any] | None:
    """First choices[].error, including error finish frames with no error object."""
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)):
        return None
    for choice in choices:
        error = choice.get("error") if isinstance(choice, dict) else getattr(choice, "error", None)
        if error is None:
            extra = choice.get("model_extra") if isinstance(choice, dict) else getattr(choice, "model_extra", None)
            if isinstance(extra, dict):
                error = extra.get("error")
        normalized = _error_fields(error)
        if normalized:
            return normalized
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else getattr(choice, "finish_reason", None)
        if str(finish_reason or "").strip().lower() == "error":
            return {"message": "Chat completion choice finished with error", "type": "choice_error"}
    return None


def embedded_provider_error(response: Any) -> dict[str, Any] | None:
    """Provider error from an HTTP-200 envelope or a partial-choice response."""
    error = response.get("error") if isinstance(response, dict) else getattr(response, "error", None)
    normalized = _error_fields(error)
    return normalized or embedded_choice_provider_error(response)


def is_cancellation(error: BaseException) -> bool:
    """Hermes aborted the call (explicit cancel, interrupt, Ctrl+C, task cancel, shutdown):
    everything that is not an ``Exception``, plus the ``InterruptedError`` aux streams raise."""
    return not isinstance(error, Exception) or isinstance(error, InterruptedError)


def error_class(error: BaseException, *, provider: str = "", model: str = "") -> str:
    """The classifier's reason for one aux attempt error; ``unknown`` when it cannot say.
    Never raises: telemetry must not replace the caller's exception."""
    if not isinstance(error, Exception):
        return "unknown"
    try:
        from agent.error_classifier import classify_api_error

        embedded = getattr(error, PROVIDER_ERROR_ATTR, None)
        target = _EmbeddedProviderError(embedded) if isinstance(embedded, dict) else error
        return classify_api_error(target, provider=provider, model=model).reason.value
    except Exception:
        logger.debug("Auxiliary error classification failed", exc_info=True)
        return "unknown"
