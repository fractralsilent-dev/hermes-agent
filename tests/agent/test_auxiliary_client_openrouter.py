"""FRA-11 regressions for typed OpenRouter errors and credential-pool recovery."""

from types import SimpleNamespace

import pytest

from agent.auxiliary_client import (
    _is_overloaded_error,
    _is_rate_limit_error,
    _recover_provider_pool,
)
from agent.auxiliary_rate_limits import _is_openrouter_upstream_rate_limit


def _install_pool(monkeypatch):
    class StubPool:
        def __init__(self):
            self.rotate_calls = []

        def has_credentials(self):
            return True

        def try_refresh_current(self):
            return None

        def mark_exhausted_and_rotate(self, **kwargs):
            self.rotate_calls.append(kwargs)
            return object()

    pool = StubPool()
    monkeypatch.setattr(
        "agent.auxiliary_client.pool_for_aux_provider",
        lambda provider, normalized, loader: pool,
    )
    monkeypatch.setattr("agent.auxiliary_client._evict_cached_clients", lambda provider: None)
    return pool


@pytest.mark.parametrize(
    "body",
    [
        {
            "error": {
                "code": 429,
                "message": "Rate limit exceeded",
                "metadata": {
                    "error_type": "rate_limit_exceeded",
                    "provider_code": "rate_limited",
                },
            },
        },
        {
            # Preserve compatibility with SDKs exposing a flat error body.
            "code": 429,
            "message": "Rate limit exceeded",
            "metadata": {
                "error_type": "rate_limit_exceeded",
                "provider_code": "rate_limited",
            },
        },
    ],
    ids=["documented-nested-chat-completions-envelope", "flat-sdk-body"],
)
def test_typed_openrouter_upstream_rate_limit_preserves_credential_pool(monkeypatch, body):
    class Wrapped429(Exception):
        status_code = 429

        def __init__(self, error_body):
            super().__init__(f"Error code: 429 - {error_body}")
            self.body = error_body

    exc = Wrapped429(body)
    assert _is_rate_limit_error(exc) is True
    assert _is_openrouter_upstream_rate_limit(exc) is True

    pool_requests = []
    monkeypatch.setattr(
        "agent.auxiliary_client.pool_for_aux_provider",
        lambda *args, **kwargs: pool_requests.append((args, kwargs)),
    )
    assert _recover_provider_pool("openrouter", exc, failed_api_key="sk-failed") is False
    assert pool_requests == [], "typed upstream 429 must short-circuit before credential-pool lookup"


def test_account_level_openrouter_429_without_provider_error_type_still_rotates(monkeypatch):
    class Account429(Exception):
        status_code = 429
        body = {
            "error": {
                "code": 429,
                "message": "Rate limit exceeded for this API key",
                "metadata": {
                    "limit_source": "openrouter_key_limit",
                    "scope": "api_key",
                },
            },
        }

    exc = Account429()
    assert _is_rate_limit_error(exc) is True
    assert _is_openrouter_upstream_rate_limit(exc) is False

    pool = _install_pool(monkeypatch)
    assert _recover_provider_pool("openrouter", exc, failed_api_key="sk-failed") is True
    assert len(pool.rotate_calls) == 1
    assert pool.rotate_calls[0]["status_code"] == 429
    assert pool.rotate_calls[0]["api_key_hint"] == "sk-failed"


def test_typed_non_rate_limit_provider_error_overrides_429_message_and_sdk_class(monkeypatch):
    class RateLimitError(Exception):
        status_code = 429

        def __init__(self):
            super().__init__("Error code: 429 - Rate limit exceeded")
            self.body = {
                "error": {
                    "code": 429,
                    "message": "Rate limit exceeded",
                    "metadata": {
                        "error_type": "provider_unavailable",
                        "provider_code": "upstream_unavailable",
                        "provider_name": "mock-upstream",
                    },
                },
            }

    exc = RateLimitError()
    assert _is_rate_limit_error(exc) is False
    assert _is_openrouter_upstream_rate_limit(exc) is False

    pool = _install_pool(monkeypatch)
    assert _recover_provider_pool("openrouter", exc, failed_api_key="sk-failed") is False
    assert pool.rotate_calls == [], "a typed non-rate-limit provider error must not bench the API key"


def test_openrouter_metadata_without_typed_error_type_is_not_an_upstream_rate_limit():
    class MetadataOnly429(Exception):
        status_code = 429
        body = {
            "error": {
                "code": 429,
                "message": "Provider returned error",
                "metadata": {
                    "provider_name": "mock-upstream",
                    "raw": '{"error":{"message":"some upstream response"}}',
                },
            },
        }

    assert _is_openrouter_upstream_rate_limit(MetadataOnly429()) is False


def test_openrouter_provider_overload_does_not_rotate_credentials(monkeypatch):
    class ProviderOverloaded(Exception):
        status_code = 503
        body = {
            "error": {
                "code": 503,
                "message": "Provider is temporarily overloaded",
                "metadata": {
                    "error_type": "provider_overloaded",
                },
            },
        }

    exc = ProviderOverloaded("Error code: 503 - Provider is temporarily overloaded")
    assert _is_rate_limit_error(exc) is False
    assert _is_overloaded_error(exc) is True
    assert _is_openrouter_upstream_rate_limit(exc) is False

    pool = _install_pool(monkeypatch)
    assert _recover_provider_pool("openrouter", exc, failed_api_key="sk-failed") is False
    assert pool.rotate_calls == []


@pytest.mark.parametrize(
    "body_state",
    ["absent", "non-dict"],
    ids=["response-json-fallback", "response-json-fallback-for-non-dict-body"],
)
def test_typed_openrouter_rate_limit_can_be_read_from_response_json(body_state):
    response_body = {
        "error": {
            "code": 429,
            "message": "Rate limit exceeded",
            "metadata": {"error_type": "rate_limit_exceeded"},
        },
    }

    class ResponseWrapped429(Exception):
        status_code = 429

        def __init__(self):
            super().__init__("Error code: 429 - Rate limit exceeded")
            self.response = SimpleNamespace(json=lambda: response_body)
            if body_state == "non-dict":
                self.body = "not-a-dict"

    assert _is_openrouter_upstream_rate_limit(ResponseWrapped429()) is True


def test_other_provider_does_not_use_openrouter_short_circuit(monkeypatch):
    class OtherProvider429(Exception):
        status_code = 429
        body = {
            "error": {
                "code": 429,
                "message": "Rate limit exceeded",
                "metadata": {"error_type": "rate_limit_exceeded"},
            },
        }

    pool = _install_pool(monkeypatch)
    assert _recover_provider_pool("other-provider", OtherProvider429(), failed_api_key="sk-failed") is True
    assert len(pool.rotate_calls) == 1
    assert pool.rotate_calls[0]["status_code"] == 429
