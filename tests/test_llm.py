"""Tests for planner/llm.py. None of these make a real API call."""

import pytest

from planner import llm


class FakeApiError(Exception):
    """Stands in for the SDK's API errors, which carry a `status_code`."""

    def __init__(self, status_code: int):
        super().__init__(f"fake error {status_code}")
        self.status_code = status_code


@pytest.fixture(autouse=True)
def isolated_llm(tmp_path, monkeypatch):
    """Give every test its own cache folder, a fake key, and no real sleeping."""
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "_last_call_time", None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    return sleeps


def fake_send(monkeypatch, outcomes: list):
    """Replace _send_request. Each call uses the next outcome: an Exception is
    raised, anything else is returned. Returns the list of prompts it received."""
    calls: list[str] = []
    remaining = list(outcomes)

    def send(prompt: str) -> str:
        calls.append(prompt)
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm, "_send_request", send)
    return calls


# --- cache ---


def test_second_identical_call_uses_the_cache(monkeypatch):
    calls = fake_send(monkeypatch, ['{"a": 1}'])
    assert llm.call_llm("prompt") == '{"a": 1}'
    assert llm.call_llm("prompt") == '{"a": 1}'
    assert len(calls) == 1  # the second call never reached the API


def test_successful_reply_is_saved_to_disk(monkeypatch):
    fake_send(monkeypatch, ["reply"])
    llm.call_llm("prompt")
    assert [p.read_text(encoding="utf-8") for p in llm.CACHE_DIR.iterdir()] == ["reply"]


def test_different_prompts_are_cached_separately(monkeypatch):
    calls = fake_send(monkeypatch, ["one", "two"])
    assert llm.call_llm("first") == "one"
    assert llm.call_llm("second") == "two"
    assert len(calls) == 2


def test_changing_the_model_does_not_reuse_the_cache(monkeypatch):
    calls = fake_send(monkeypatch, ["from model A", "from model B"])
    monkeypatch.setenv("GEMINI_MODEL", "model-a")
    assert llm.call_llm("prompt") == "from model A"
    monkeypatch.setenv("GEMINI_MODEL", "model-b")
    assert llm.call_llm("prompt") == "from model B"
    assert len(calls) == 2


def test_cache_hit_works_without_an_api_key(monkeypatch):
    fake_send(monkeypatch, ["reply"])
    llm.call_llm("prompt")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert llm.call_llm("prompt") == "reply"


# --- retries ---


def test_rate_limit_is_retried_then_succeeds(monkeypatch, isolated_llm):
    calls = fake_send(monkeypatch, [FakeApiError(429), "reply"])
    assert llm.call_llm("prompt") == "reply"
    assert len(calls) == 2
    # We waited at least the base delay (2s) before the retry.
    assert any(delay >= llm.BASE_DELAY_SECONDS for delay in isolated_llm)


def test_server_errors_are_retried(monkeypatch):
    calls = fake_send(monkeypatch, [FakeApiError(503), FakeApiError(500), "reply"])
    assert llm.call_llm("prompt") == "reply"
    assert len(calls) == 3


def test_waits_get_longer_each_retry():
    delays = [llm._backoff_delay(attempt) for attempt in (1, 2, 3, 4)]
    # Jitter adds under 1s, so each delay is at least the doubled base and below the next.
    assert [int(d) for d in delays] == [2, 4, 8, 16]


def test_gives_up_after_max_attempts(monkeypatch):
    calls = fake_send(monkeypatch, [FakeApiError(429)] * llm.MAX_ATTEMPTS)
    with pytest.raises(llm.LLMError, match="Gemini request failed"):
        llm.call_llm("prompt")
    assert len(calls) == llm.MAX_ATTEMPTS


def test_non_retryable_error_fails_immediately(monkeypatch):
    calls = fake_send(monkeypatch, [FakeApiError(400), "never reached"])
    with pytest.raises(llm.LLMError):
        llm.call_llm("prompt")
    assert len(calls) == 1


def test_failed_call_is_not_cached(monkeypatch):
    fake_send(monkeypatch, [FakeApiError(400)])
    with pytest.raises(llm.LLMError):
        llm.call_llm("prompt")
    assert not llm.CACHE_DIR.exists() or not list(llm.CACHE_DIR.iterdir())


# --- pacing ---


def test_real_calls_are_spaced_out(monkeypatch, isolated_llm):
    fake_send(monkeypatch, ["one", "two"])
    llm.call_llm("first")
    assert isolated_llm == []  # nothing to wait for on the first call
    llm.call_llm("second")
    assert len(isolated_llm) == 1
    assert 0 < isolated_llm[0] <= llm.MIN_SECONDS_BETWEEN_CALLS


# --- the real _send_request, with the SDK client faked ---


class FakeClient:
    """Records how it was built and called, and replies or raises as told."""

    created_with: dict = {}
    create_args: dict = {}
    error: Exception | None = None

    def __init__(self, **kwargs):
        FakeClient.created_with = kwargs
        self.interactions = self

    def create(self, **kwargs):
        FakeClient.create_args = kwargs
        if FakeClient.error:
            raise FakeClient.error

        class Reply:
            output_text = '{"ok": true}'

        return Reply()


def test_request_is_built_safely(monkeypatch):
    FakeClient.error = None
    monkeypatch.setattr(llm.genai, "Client", FakeClient)
    assert llm.call_llm("my prompt") == '{"ok": true}'
    # SDK retries are off so they can't stack with ours.
    assert FakeClient.created_with["http_options"] == {"retry_options": {"attempts": 1}}
    assert FakeClient.create_args["input"] == "my prompt"
    assert FakeClient.create_args["model"] == llm.DEFAULT_MODEL
    assert FakeClient.create_args["store"] is False  # don't store syllabus text server-side
    assert FakeClient.create_args["response_format"]["mime_type"] == "application/json"


def test_missing_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.call_llm("prompt")


def test_api_key_never_appears_in_error_messages(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY-123")
    FakeClient.error = FakeApiError(403)
    monkeypatch.setattr(llm.genai, "Client", FakeClient)
    with pytest.raises(llm.LLMError) as caught:
        llm.call_llm("prompt")
    assert "SECRET-KEY-123" not in str(caught.value)
    FakeClient.error = None
