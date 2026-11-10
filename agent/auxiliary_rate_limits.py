"""HTTP error-shape classification for auxiliary provider recovery."""

from __future__ import annotations

from agent.auxiliary_error_shapes import _botocore_error, _exc_http_status
from agent.error_classifier import _OVERLOADED_PATTERNS


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    """True when any needle is a substring of the text."""
    return any(kw in text for kw in needles)


_RATE_LIMIT_KEYWORDS = (
    "rate limit", "rate_limit", "too many requests", "try again", "retry after", "resets in"
)
_RATE_LIMIT_BILLING_KEYWORDS = (
    "credits", "insufficient funds", "billing", "payment required", "can only afford",
    "out of funds", "run out of funds", "balance_depleted", "no usable credits",
    "model_not_supported_on_free_tier", "not available on the free tier", "isn't available on the free tier",
)


def _openrouter_error_payload(body: dict) -> dict | None:
    """Return one structured error object from supported Chat Completions shapes.

    OpenRouter errors may be top-level, flat SDK-compatible bodies, or attached
    to an individual choice when non-streaming generation returns partial
    content. Ignore malformed choices and never infer an error from message text.
    """
    nested = body.get("error")
    if isinstance(nested, dict):
        return nested

    if "error" not in body and ("metadata" in body or "code" in body):
        return body

    choices = body.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_error = choice.get("error")
            if isinstance(choice_error, dict):
                return choice_error
    return None


def _openrouter_error_type(exc: Exception) -> str | None:
    """Read Chat Completions' typed OpenRouter error discriminator, when present.

    Supports top-level errors, flat SDK-compatible bodies, and per-choice
    errors in non-streaming partial-content responses. Metadata presence alone
    is deliberately insufficient.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            body = response
        else:
            json_method = getattr(response, "json", None)
            if callable(json_method):
                try:
                    body = json_method()
                except Exception:
                    body = None
    if not isinstance(body, dict):
        return None

    error = _openrouter_error_payload(body)
    metadata = error.get("metadata") if error is not None else None
    if not isinstance(metadata, dict):
        return None
    error_type = metadata.get("error_type")
    if not isinstance(error_type, str):
        return None
    error_type = error_type.strip()
    return error_type or None


def _is_overloaded_error(exc: Exception) -> bool:
    """Server busy, credential fine ("at capacity upstream … not your API key's rate limit"): back off,
    never bench the credential (#108349). Same phrase table as the main classifier."""
    return _contains_any(str(exc).lower(), _OVERLOADED_PATTERNS)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Generic HTTP 429 classifier shared by all auxiliary providers.

    OpenRouter's typed provider discriminator is intentionally applied only
    inside the OpenRouter-specific recovery branch. Other providers retain this
    established class/status/message fallback behavior.
    """
    # OpenAI's RateLimitError may omit .status_code; botocore exposes throttling
    # through its structured response instead of an HTTP status attribute.
    if type(exc).__name__ == "RateLimitError" or _botocore_error(exc)[1] == "ThrottlingException":
        return True
    if _exc_http_status(exc) != 429:
        return False
    err_lower = str(exc).lower()
    return _contains_any(err_lower, _RATE_LIMIT_KEYWORDS) or not _contains_any(err_lower, _RATE_LIMIT_BILLING_KEYWORDS)


def _openrouter_error_code(exc: Exception) -> int | None:
    """Read the HTTP-like code from the typed OpenRouter error envelope."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            body = response
        else:
            json_method = getattr(response, "json", None)
            if callable(json_method):
                try:
                    body = json_method()
                except Exception:
                    body = None
    if not isinstance(body, dict):
        return None
    error = _openrouter_error_payload(body)
    code = error.get("code") if error is not None else None
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _is_openrouter_upstream_rate_limit(exc: Exception) -> bool:
    """Require both canonical type and code 429, including mid-stream SSE.

    A mid-stream SSE error can have outer HTTP status 200, so inspect the
    embedded error object's code rather than the exception status alone.
    """
    return (
        _openrouter_error_type(exc) == "rate_limit_exceeded"
        and _openrouter_error_code(exc) == 429
    )
