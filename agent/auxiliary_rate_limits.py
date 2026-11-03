"""HTTP error-shape classification for auxiliary provider recovery."""

from __future__ import annotations

from agent.auxiliary_error_shapes import _botocore_error, _exc_http_status
from agent.error_classifier import _OVERLOADED_PATTERNS

def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    """True when any needle is a substring of ``text``."""
    return any(kw in text for kw in needles)

_RATE_LIMIT_KEYWORDS = (
    "rate limit", "rate_limit", "too many requests", "try again", "retry after", "resets in"
)
_RATE_LIMIT_BILLING_KEYWORDS = (
    "credits", "insufficient funds", "billing", "payment required", "can only afford",
    "out of funds", "run out of funds", "balance_depleted", "no usable credits",
    "model_not_supported_on_free_tier", "not available on the free tier", "isn't available on the free tier",
)


def _is_overloaded_error(exc: Exception) -> bool:
    """Server busy, credential fine ("at capacity upstream … not your API key's rate limit"): back off,
    never bench the credential (#108349). Same phrase table as the main classifier."""
    return _contains_any(str(exc).lower(), _OVERLOADED_PATTERNS)


def _is_rate_limit_error(exc: Exception) -> bool:
    """429 rate limit (not billing/quota, which _is_payment_error owns).

    OpenAI's RateLimitError may omit .status_code — matched by class name. A generic 429 without
    billing keywords counts as a rate limit.
    """
    # (PR #8023 pattern)
    if type(exc).__name__ == "RateLimitError" or _botocore_error(exc)[1] == "ThrottlingException":
        return True
    if _exc_http_status(exc) != 429:
        return False
    err_lower = str(exc).lower()
    return _contains_any(err_lower, _RATE_LIMIT_KEYWORDS) or not _contains_any(err_lower, _RATE_LIMIT_BILLING_KEYWORDS)

def _is_openrouter_upstream_rate_limit(exc: Exception) -> bool:
    """Identify an OpenRouter-wrapped upstream error, not an account-level 429.

    OpenAI SDK versions may expose OpenRouter's inner error as either a nested
    body["error"] envelope or a flat {message, code, metadata} body. Require the
    exact wrapper message plus structured OpenRouter metadata before suppressing
    credential recovery; ordinary account-level rate limits still rotate.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        response = getattr(exc, "response", None)
        try:
            body = response.json() if response is not None else None
        except (ValueError, TypeError):
            body = None
    if not isinstance(body, dict):
        return False

    nested = body.get("error")
    if isinstance(nested, dict):
        error_obj = nested
    elif "error" not in body and isinstance(body.get("metadata"), dict):
        error_obj = body
    else:
        return False

    if str(error_obj.get("message") or "").strip().lower() != "provider returned error":
        return False
    metadata = error_obj.get("metadata")
    if not isinstance(metadata, dict):
        return False
    raw = metadata.get("raw")
    provider_name = metadata.get("provider_name")
    return (
        isinstance(raw, str) and bool(raw.strip())
    ) or (
        isinstance(provider_name, str) and bool(provider_name.strip())
    )
