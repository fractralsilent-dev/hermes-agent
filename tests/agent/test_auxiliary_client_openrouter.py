"""FRA-11 OpenRouter upstream-429 credential-pool regression tests."""

from types import SimpleNamespace

import pytest

from agent.auxiliary_client import _is_overloaded_error, _is_rate_limit_error, _recover_provider_pool
from agent.auxiliary_rate_limits import _is_openrouter_upstream_rate_limit


# FRA-11 regression: distinguish OpenRouter upstream 429 from account-level 429.
@pytest.mark.parametrize(
    "body",
    [
        {
            "error": {
                "message": "Provider returned error",
                "metadata": {
                    "raw": '{"error":{"message":"rate limit exceeded for this model"}}',
                    "provider_name": "mock-upstream",
                },
            },
        },
        {
            # Some OpenAI SDK versions expose the inner OpenRouter error directly.
            "message": "Provider returned error",
            "code": 429,
            "metadata": {
                "raw": '{"error":{"message":"rate limit exceeded for this model"}}',
                "provider_name": "mock-upstream",
            },
        },
    ],
    ids=["nested-openrouter-envelope", "flat-openai-sdk-body"],
)
def test_openrouter_upstream_429_preserves_pool_before_lookup(monkeypatch, body):
    class Wrapped429(Exception):
        status_code = 429

        def __init__(self, error_body):
            super().__init__("Error code: 429 - Provider returned error")
            self.body = error_body

    exc = Wrapped429(body)
    assert _is_rate_limit_error(exc) is True
    pool_requests = []
    monkeypatch.setattr(
        "agent.auxiliary_client.pool_for_aux_provider",
        lambda *args, **kwargs: pool_requests.append((args, kwargs)),
    )

    assert _recover_provider_pool("openrouter", exc) is False
    assert pool_requests == [], "upstream 429 must return before credential-pool lookup"


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": "Provider returned error"}},
        {"error": {"message": "Provider returned error", "metadata": {}}},
        {
            "error": {
                "message": "Provider returned error",
                "metadata": {"provider_name": "   ", "raw": ""},
            }
        },
    ],
)
def test_openrouter_upstream_detector_requires_nonempty_structured_metadata(body):
    class Wrapped429(Exception):
        status_code = 429

        def __init__(self, error_body):
            super().__init__("Error code: 429 - Provider returned error")
            self.body = error_body

    assert _is_openrouter_upstream_rate_limit(Wrapped429(body)) is False


def test_openrouter_account_level_429_still_rotates_pool(monkeypatch):
    class Account429(Exception):
        status_code = 429
        body = {
            "error": {
                "message": "Rate limit exceeded for this API key",
                "code": 429,
            },
        }

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

    assert _recover_provider_pool("openrouter", Account429(), failed_api_key="sk-failed") is True
    assert len(pool.rotate_calls) == 1
    assert pool.rotate_calls[0]["status_code"] == 429
    assert pool.rotate_calls[0]["api_key_hint"] == "sk-failed"


def test_other_provider_does_not_use_openrouter_429_guard(monkeypatch):
    class Wrapped429(Exception):
        status_code = 429
        body = {
            "error": {
                "message": "Provider returned error",
                "metadata": {"provider_name": "mock-upstream"},
            },
        }

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

    assert _recover_provider_pool("other-provider", Wrapped429(), failed_api_key="sk-failed") is True
    assert len(pool.rotate_calls) == 1
    assert pool.rotate_calls[0]["status_code"] == 429

@pytest.mark.parametrize(
    "body_state",
    ["absent", "non-dict"],
    ids=["response-json-when-body-absent", "response-json-when-body-non-dict"],
)
def test_openrouter_upstream_429_uses_response_json_fallback(monkeypatch, body_state):
    response_body = {
        "error": {
            "message": "Provider returned error",
            "metadata": {
                "raw": '{"error":{"message":"rate limit exceeded for this model"}}',
                "provider_name": "mock-upstream",
            },
        },
    }

    class ResponseWrapped429(Exception):
        status_code = 429

        def __init__(self):
            super().__init__("Error code: 429 - Provider returned error")
            self.response = SimpleNamespace(json=lambda: response_body)
            if body_state == "non-dict":
                self.body = "not-a-dict"

    exc = ResponseWrapped429()
    assert _is_rate_limit_error(exc) is True
    assert _is_openrouter_upstream_rate_limit(exc) is True

    pool_requests = []
    monkeypatch.setattr(
        "agent.auxiliary_client.pool_for_aux_provider",
        lambda *args, **kwargs: pool_requests.append((args, kwargs)),
    )

    assert _recover_provider_pool("openrouter", exc) is False
    assert pool_requests == [], "response.json fallback must preserve upstream-429 pool short-circuit"


def test_openrouter_overloaded_429_keeps_existing_overload_policy(monkeypatch):
    class OverloadedWrapped429(Exception):
        status_code = 429
        body = {
            "error": {
                "message": "Provider returned error",
                "metadata": {
                    "raw": '{"error":{"message":"upstream overloaded at capacity"}}',
                    "provider_name": "mock-upstream",
                },
            },
        }

    exc = OverloadedWrapped429("Error code: 429 - upstream overloaded at capacity")
    assert _is_rate_limit_error(exc) is True
    assert _is_overloaded_error(exc) is True
    assert _is_openrouter_upstream_rate_limit(exc) is True

    pool_requests = []

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
        lambda provider, normalized, loader: pool_requests.append((provider, normalized)) or pool,
    )
    monkeypatch.setattr("agent.auxiliary_client._evict_cached_clients", lambda provider: None)

    # Overloaded is excluded from both the new upstream-429 short-circuit and
    # the ordinary 429 rotation path: preserve the existing overload/backoff behavior.
    assert _recover_provider_pool("openrouter", exc, failed_api_key="sk-failed") is False
    assert pool_requests == [("openrouter", "openrouter")]
    assert pool.rotate_calls == []

def test_openrouter_sdk_rate_limit_error_uses_nested_raw_overload_for_classification(monkeypatch):
    import httpx
    from openai import RateLimitError

    body = {
        "error": {
            "message": "Provider returned error",
            "metadata": {
                "raw": '{"error":{"message":"upstream overloaded at capacity"}}',
                "provider_name": "mock-upstream",
            },
        },
    }
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(429, json=body, request=request)
    # Match openai-python 2.54.0's _make_status_error_from_response message construction.
    exc = RateLimitError(
        f"Error code: {response.status_code} - {body}",
        response=response,
        body=body,
    )

    assert "upstream overloaded at capacity" in str(exc)
    assert _is_rate_limit_error(exc) is True
    assert _is_openrouter_upstream_rate_limit(exc) is True
    assert _is_overloaded_error(exc) is True

    pool_requests = []

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
        lambda provider, normalized, loader: pool_requests.append((provider, normalized)) or pool,
    )
    monkeypatch.setattr("agent.auxiliary_client._evict_cached_clients", lambda provider: None)

    assert _recover_provider_pool("openrouter", exc, failed_api_key="sk-failed") is False
    assert pool_requests == [("openrouter", "openrouter")]
    assert pool.rotate_calls == []
