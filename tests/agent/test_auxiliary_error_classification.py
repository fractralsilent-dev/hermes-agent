"""Auxiliary error classification shares the main classifier's billing table and labels quarantines truthfully.

Invariants for #107166 (OpenRouter org "Budget limit exceeded" 403 must be a payment error for the aux
ladder, like the main loop), #64144 (an unhealthy mark for absent credentials must not claim a payment
error), and #108349 (a vision fallback never lands on a text-only main model; an upstream-capacity 429
never benches the credential pool).
"""
import logging

import pytest

from agent import auxiliary_client as ac
from agent.error_classifier import FailoverReason, classify_api_error


class _Err(Exception):
    def __init__(self, msg, status):
        super().__init__(msg)
        self.status_code = status


@pytest.mark.parametrize("body", [
    "Error code: 403 - {'error': {'message': 'Budget limit exceeded (monthly limit). Contact your org admin.', 'code': 403}}",
    "Error code: 403 - {'error': {'message': 'Key limit exceeded (total limit). Contact your org admin.', 'code': 403}}",
])
def test_openrouter_limit_403_is_billing_for_main_loop_and_aux_ladder(body):
    exc = _Err(body, 403)
    assert classify_api_error(exc).reason is FailoverReason.billing
    assert ac._is_payment_error(exc)
    # Every main-classifier billing phrase is an aux payment phrase (one table, no drift).
    assert all(p in ac._PAYMENT_KEYWORDS for p in classify_api_error.__globals__["_BILLING_PATTERNS"])


def test_absent_credentials_quarantine_is_debug_and_names_the_real_reason(caplog):
    ac._reset_aux_unhealthy_cache()
    with caplog.at_level(logging.DEBUG, logger="agent.auxiliary_client"):
        ac._mark_provider_unhealthy("openrouter", ttl=60, reason="OPENROUTER_API_KEY not set", level=logging.DEBUG)
        ac._log_skip_unhealthy("openrouter", "title_generation")
        ac._mark_provider_unhealthy("nous")  # confirmed 402 path keeps the payment wording at WARNING
    marks = [r for r in caplog.records if "marking" in r.getMessage()]
    assert [(r.levelno, "payment" in r.getMessage()) for r in marks] == [(logging.DEBUG, False), (logging.WARNING, True)]
    assert "OPENROUTER_API_KEY not set" in marks[0].getMessage()
    skip = next(r for r in caplog.records if "skipping" in r.getMessage())
    assert "OPENROUTER_API_KEY not set" in skip.getMessage() and "payment" not in skip.getMessage()
    ac._reset_aux_unhealthy_cache()


def test_vision_fallback_skips_text_only_main_model(monkeypatch):
    monkeypatch.setattr(ac, "_read_main_provider", lambda: "zai")
    monkeypatch.setattr(ac, "_read_main_model", lambda: "glm-5.3")
    monkeypatch.setattr(ac, "_main_model_supports_vision", lambda provider, model: False)
    monkeypatch.setattr(ac, "resolve_provider_client", lambda **kw: pytest.fail("must not build a client for a text-only model"))
    assert ac._try_main_agent_model_fallback("nous", "vision", reason="rate limit") == (None, None, "")


@pytest.mark.parametrize("body, benches_pool", [
    ("Error code: 429 - {'status': 429, 'message': \"The requested model is temporarily at capacity upstream. "
     "This is not your API key's rate limit — please retry shortly.\"}", False),
    ("Error code: 429 - {'error': {'message': 'Rate limit exceeded for this API key', 'code': 429}}", True),
])
def test_upstream_capacity_429_is_not_a_credential_to_bench(monkeypatch, body, benches_pool):
    exc = _Err(body, 429)
    assert ac._is_rate_limit_error(exc)
    assert ac._is_overloaded_error(exc) is (not benches_pool)
    assert classify_api_error(exc).reason is (
        FailoverReason.rate_limit if benches_pool else FailoverReason.overloaded)

    # The predicate is only half the contract: _recover_provider_pool must not bench a credential
    # for an upstream-capacity 429, while a real per-key rate limit still rotates the pool.
    class _StubPool:
        def __init__(self):
            self.rotate_calls = []

        def has_credentials(self):
            return True

        def try_refresh_current(self):
            return None

        def mark_exhausted_and_rotate(self, **kwargs):
            self.rotate_calls.append(kwargs)
            return object()  # a next entry exists

    pool = _StubPool()
    monkeypatch.setattr(ac, "load_pool", lambda provider: pool)
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda provider: None)
    recovered = ac._recover_provider_pool("openrouter", exc, failed_api_key="sk-failed")
    assert recovered is benches_pool
    assert len(pool.rotate_calls) == (1 if benches_pool else 0)
    if benches_pool:
        assert pool.rotate_calls[0]["status_code"] == 429
        assert pool.rotate_calls[0]["api_key_hint"] == "sk-failed"

@pytest.mark.parametrize(
    "body",
    [
        {
            "error": {
                "message": "Provider returned error",
                "metadata": {
                    "raw": '{"error": {"message": "rate limit exceeded for this model"}}',
                    "provider_name": "mock-upstream",
                },
            },
        },
        {
            # The OpenAI SDK may expose the inner OpenRouter error object directly.
            "message": "Provider returned error",
            "code": 429,
            "metadata": {
                "raw": '{"error": {"message": "rate limit exceeded for this model"}}',
                "provider_name": "mock-upstream",
            },
        },
    ],
    ids=["nested-http-envelope", "flat-openai-sdk-body"],
)
def test_openrouter_wrapped_upstream_429_preserves_pool(monkeypatch, body):
    """Nested and flat OpenRouter upstream 429s must not quarantine a healthy credential."""

    class Wrapped429(Exception):
        status_code = 429

        def __init__(self, body):
            super().__init__("Error code: 429 - Provider returned error")
            self.body = body

    exc = Wrapped429(body)
    assert ac._is_rate_limit_error(exc)
    assert not ac._is_overloaded_error(exc)

    # Confirm the shared classifier recognizes the equivalent nested envelope.
    from agent.error_classifier import _is_openrouter_upstream_error

    classifier_body = body if isinstance(body.get("error"), dict) else {"error": body}
    assert _is_openrouter_upstream_error(classifier_body, "openrouter")

    class _StubPool:
        def __init__(self):
            self.rotate_calls = []

        def has_credentials(self):
            return True

        def try_refresh_current(self):
            return None

        def mark_exhausted_and_rotate(self, **kwargs):
            self.rotate_calls.append(kwargs)
            return object()

    pool = _StubPool()
    pool_requests = []
    monkeypatch.setattr(
        ac,
        "pool_for_aux_provider",
        lambda provider, normalized, loader: pool_requests.append((provider, normalized)) or pool,
    )
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda provider: None)

    assert ac._recover_provider_pool("openrouter", exc) is False
    assert pool_requests == [], "upstream 429 must return before requesting the credential pool"
    assert pool.rotate_calls == [], "upstream 429 must not rotate a healthy credential"


def test_openrouter_account_level_429_still_rotates_pool(monkeypatch):
    """A genuine account-level OpenRouter 429 must still rotate credentials."""

    class Account429(Exception):
        status_code = 429
        body = {
            "error": {
                "message": "Rate limit exceeded for this API key",
                "code": 429,
            },
        }

    pool_requests = []
    class _StubPool:
        def __init__(self):
            self.rotate_calls = []

        def has_credentials(self):
            return True

        def try_refresh_current(self):
            return None

        def mark_exhausted_and_rotate(self, **kwargs):
            self.rotate_calls.append(kwargs)
            return object()

    pool = _StubPool()
    monkeypatch.setattr(
        ac,
        "pool_for_aux_provider",
        lambda provider, normalized, loader: pool_requests.append((provider, normalized)) or pool,
    )
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda provider: None)

    exc = Account429("Rate limit exceeded for this API key")
    assert ac._is_rate_limit_error(exc)
    assert ac._recover_provider_pool("openrouter", exc, failed_api_key="sk-failed") is True
    assert pool_requests == [("openrouter", "openrouter")]
    assert len(pool.rotate_calls) == 1
    assert pool.rotate_calls[0]["status_code"] == 429
    assert pool.rotate_calls[0]["api_key_hint"] == "sk-failed"
