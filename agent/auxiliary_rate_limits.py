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


def _openrouter_error_type(exc: Exception) -> str | None:
    """Read Chat Completions' typed OpenRouter error discriminator, when present.

    The OpenAI SDK normally exposes the response as body["error"]; a flat
    {message, code, metadata} body is also accepted for SDK compatibility.
    Metadata presence alone is deliberately insufficient.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        response = getattr(exc, "response", None)
        try:
            body = response.json() if response is not None else None
        except (ValueError, TypeError):
            body = None
    if not isinstance(body, dict):
        return None

    nested = body.get("error")
    if isinstance(nested, dict):
        metadata = nested.get("metadata")
    elif "error" not in body:
        metadata = body.get("metadata")
    else:
        return None

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
    """Classify HTTP 429 errors without overriding a typed OpenRouter error.

    OpenRouter's Chat Completions metadata.error_type is authoritative when
    supplied: only rate_limit_exceeded with HTTP 429 is a recoverable 429.
    Other typed errors must not become rate limits merely because their HTTP
    status or message looks like 429. Untyped account-level 429s retain the
    existing fallback behavior.
    """
    error_type = _openrouter_error_type(exc)
    if error_type is not None:
        return error_type == "rate_limit_exceeded" and _exc_http_status(exc) == 429

    # OpenAI's RateLimitError may omit .status_code; botocore exposes throttling
    # through its structured response instead of an HTTP status attribute.
    if type(exc).__name__ == "RateLimitError" or _botocore_error(exc)[1] == "ThrottlingException":
        return True
    if _exc_http_status(exc) != 429:
        return False
    err_lower = str(exc).lower()
    return _contains_any(err_lower, _RATE_LIMIT_KEYWORDS) or not _contains_any(err_lower, _RATE_LIMIT_BILLING_KEYWORDS)


def _is_openrouter_upstream_rate_limit(exc: Exception) -> bool:
    """Identify only a typed OpenRouter upstream 429, not an account-level 429."""
    return (
        _exc_http_status(exc) == 429
        and _openrouter_error_type(exc) == "rate_limit_exceeded"
    )
