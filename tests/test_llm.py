"""Tests for planner/llm.py. None of these make a real API call."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from planner import llm


class FakeApiError(Exception):
    """Stands in for the SDK's API errors, which carry a `status_code`."""

    def __init__(self, status_code: int):
        super().__init__(f"fake error {status_code}")
        self.status_code = status_code


class APIConnectionError(Exception):
    """Same name as the SDK's connection error, which is what llm.py looks for."""


class APITimeoutError(APIConnectionError):
    """Same name and parent as the SDK's timeout error."""


@pytest.fixture(autouse=True)
def isolated_llm(tmp_path, monkeypatch):
    """Give every test its own cache folder, a fake key, and no real sleeping."""
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "_last_call_time", None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    # Fallback off by default so each test sees one model; fallback tests turn it on.
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    return sleeps


def fake_send(monkeypatch, outcomes: list):
    """Replace _send_request. Each call uses the next outcome: an Exception is
    raised, anything else is returned. Returns the model name of every call made."""
    calls: list[str] = []
    remaining = list(outcomes)

    def send(prompt: str, model: str) -> str:
        calls.append(model)
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


# --- fallback models ---


@pytest.fixture
def two_models(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "main-model")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "backup-model")


@pytest.fixture
def three_models(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "main-model")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "second-model,third-model")


def busy(times: int) -> list:
    """`times` temporary 503 errors in a row."""
    return [FakeApiError(503)] * times


def test_a_model_with_a_fallback_gets_fewer_tries_than_the_last_one():
    assert llm.ATTEMPTS_BEFORE_SWITCHING < llm.MAX_ATTEMPTS


def test_switches_to_the_backup_after_only_a_few_tries(monkeypatch, two_models):
    calls = fake_send(monkeypatch, busy(llm.ATTEMPTS_BEFORE_SWITCHING) + ["backup reply"])
    assert llm.call_llm("prompt") == "backup reply"
    assert calls == ["main-model"] * llm.ATTEMPTS_BEFORE_SWITCHING + ["backup-model"]


def test_main_model_is_used_when_it_works(monkeypatch, two_models):
    calls = fake_send(monkeypatch, ["main reply"])
    assert llm.call_llm("prompt") == "main reply"
    assert calls == ["main-model"]


def test_a_reply_from_the_backup_is_cached_and_reused(monkeypatch, two_models):
    calls = fake_send(monkeypatch, busy(llm.ATTEMPTS_BEFORE_SWITCHING) + ["backup reply"])
    llm.call_llm("prompt")
    calls_so_far = len(calls)
    assert llm.call_llm("prompt") == "backup reply"  # no new API calls, no waiting
    assert len(calls) == calls_so_far


def test_the_last_model_gets_the_full_number_of_tries(monkeypatch, two_models):
    calls = fake_send(monkeypatch, busy(llm.ATTEMPTS_BEFORE_SWITCHING + llm.MAX_ATTEMPTS))
    with pytest.raises(llm.LLMUnavailableError, match="backup-model"):
        llm.call_llm("prompt")
    assert calls == ["main-model"] * llm.ATTEMPTS_BEFORE_SWITCHING + ["backup-model"] * llm.MAX_ATTEMPTS


def test_non_retryable_error_does_not_trigger_the_fallback(monkeypatch, two_models):
    calls = fake_send(monkeypatch, [FakeApiError(400), "never reached"])
    with pytest.raises(llm.LLMError) as caught:
        llm.call_llm("prompt")
    assert not isinstance(caught.value, llm.LLMUnavailableError)
    assert calls == ["main-model"]  # switching models can't fix a bad request


def test_fallbacks_can_be_turned_off(monkeypatch, two_models):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    calls = fake_send(monkeypatch, busy(llm.MAX_ATTEMPTS))
    with pytest.raises(llm.LLMUnavailableError):
        llm.call_llm("prompt")
    assert calls == ["main-model"] * llm.MAX_ATTEMPTS  # no one to switch to, so full tries


def test_a_fallback_equal_to_the_main_model_is_not_tried_twice(monkeypatch, two_models):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "main-model")
    calls = fake_send(monkeypatch, busy(llm.MAX_ATTEMPTS))
    with pytest.raises(llm.LLMUnavailableError):
        llm.call_llm("prompt")
    assert len(calls) == llm.MAX_ATTEMPTS


# --- a chain of several fallbacks ---


def test_the_second_model_answers_if_the_main_one_is_busy(monkeypatch, three_models):
    calls = fake_send(monkeypatch, busy(llm.ATTEMPTS_BEFORE_SWITCHING) + ["second reply"])
    assert llm.call_llm("prompt") == "second reply"
    assert calls == ["main-model"] * llm.ATTEMPTS_BEFORE_SWITCHING + ["second-model"]


def test_the_third_model_answers_if_the_first_two_are_busy(monkeypatch, three_models):
    calls = fake_send(monkeypatch, busy(2 * llm.ATTEMPTS_BEFORE_SWITCHING) + ["third reply"])
    assert llm.call_llm("prompt") == "third reply"
    assert calls == (
        ["main-model"] * llm.ATTEMPTS_BEFORE_SWITCHING
        + ["second-model"] * llm.ATTEMPTS_BEFORE_SWITCHING
        + ["third-model"]
    )


def test_only_the_last_model_in_the_chain_gets_the_full_tries(monkeypatch, three_models):
    calls = fake_send(monkeypatch, busy(2 * llm.ATTEMPTS_BEFORE_SWITCHING + llm.MAX_ATTEMPTS))
    with pytest.raises(llm.LLMUnavailableError, match="third-model"):
        llm.call_llm("prompt")
    assert calls.count("main-model") == llm.ATTEMPTS_BEFORE_SWITCHING
    assert calls.count("second-model") == llm.ATTEMPTS_BEFORE_SWITCHING
    assert calls.count("third-model") == llm.MAX_ATTEMPTS


def test_a_cached_reply_from_a_later_model_is_used_without_any_call(monkeypatch, three_models):
    llm._write_cache(llm._cache_path("prompt", "third-model"), "cached third reply")
    calls = fake_send(monkeypatch, [])
    assert llm.call_llm("prompt") == "cached third reply"
    assert calls == []


# --- the list of models, from the settings ---


def test_default_models_are_the_main_model_then_3_6_then_3_5_lite(monkeypatch):
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS")  # not set at all means "use the defaults"
    assert llm._models_to_try() == [llm.DEFAULT_MODEL, "gemini-3.6-flash", "gemini-3.5-flash-lite"]


def test_empty_setting_means_no_fallbacks(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    assert llm._models_to_try() == [llm.DEFAULT_MODEL]


def test_setting_is_trimmed_and_blanks_and_repeats_are_dropped(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "main-model")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", " a , ,b,a,main-model ")
    assert llm._models_to_try() == ["main-model", "a", "b"]


# --- timeouts ---


def test_a_timeout_moves_straight_on_to_the_next_model(monkeypatch, two_models):
    calls = fake_send(monkeypatch, [APITimeoutError("slow"), "backup reply"])
    assert llm.call_llm("prompt") == "backup reply"
    assert calls == ["main-model", "backup-model"]  # no second try on the slow model


def test_a_dropped_connection_is_treated_the_same_way(monkeypatch, two_models):
    calls = fake_send(monkeypatch, [APIConnectionError("no route"), "backup reply"])
    assert llm.call_llm("prompt") == "backup reply"
    assert calls == ["main-model", "backup-model"]


def test_a_timeout_on_the_last_model_fails_at_once_with_a_clear_message(monkeypatch):
    calls = fake_send(monkeypatch, [APITimeoutError("slow"), "never reached"])
    with pytest.raises(llm.LLMUnavailableError, match=r"did not respond in time") as caught:
        llm.call_llm("prompt")
    assert len(calls) == 1  # one try only, not MAX_ATTEMPTS
    assert "APITimeoutError" in str(caught.value)
    assert "timeout` argument" not in str(caught.value)  # the SDK's own advice isn't shown


def test_every_model_timing_out_names_the_last_one(monkeypatch, three_models):
    calls = fake_send(monkeypatch, [APITimeoutError("slow")] * 3)
    with pytest.raises(llm.LLMUnavailableError, match="third-model"):
        llm.call_llm("prompt")
    assert calls == ["main-model", "second-model", "third-model"]  # one try each


def test_a_timeout_is_not_cached(monkeypatch, two_models):
    fake_send(monkeypatch, [APITimeoutError("slow"), APITimeoutError("slow")])
    with pytest.raises(llm.LLMUnavailableError):
        llm.call_llm("prompt")
    assert not llm.CACHE_DIR.exists() or not list(llm.CACHE_DIR.iterdir())


@pytest.fixture
def slow_server():
    """A local server that takes 3 seconds to answer. Returns its address."""
    release = threading.Event()

    class Slow(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            release.wait(3)  # not time.sleep, which these tests replace with a no-op
            try:
                body = json.dumps({}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # the client has already given up

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    release.set()
    server.shutdown()


def test_the_real_sdk_times_out_and_is_recognised(monkeypatch, slow_server):
    """Uses the real SDK against a slow local server (no call to Google). If an SDK
    upgrade renames its timeout error, this test fails and says so."""
    monkeypatch.setattr(llm, "REQUEST_TIMEOUT_SECONDS", 0.5)
    real_client = llm.genai.Client

    def client_for_the_slow_server(**kwargs):
        kwargs["http_options"] = {**kwargs["http_options"], "base_url": slow_server}
        return real_client(**kwargs)

    monkeypatch.setattr(llm.genai, "Client", client_for_the_slow_server)

    started = time.monotonic()
    with pytest.raises(llm.LLMUnavailableError, match="did not respond in time"):
        llm.call_llm("prompt")
    assert time.monotonic() - started < 2.5  # gave up at about 0.5s, not after the 3s answer


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
    assert FakeClient.create_args["timeout"] == llm.REQUEST_TIMEOUT_SECONDS  # never wait forever
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
