"""The ONLY file that knows about Gemini.

Everything else calls call_llm(prompt) and gets text back. This file adds:
  - a disk cache, so the same prompt never uses free-tier quota twice
  - retries with growing waits when Google says "too many requests" (429)
  - a short pause between real calls
"""

import hashlib
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Read GEMINI_API_KEY (and optionally GEMINI_MODEL) from the .env file.
load_dotenv()

DEFAULT_MODEL = "gemini-3.8-flash"
# Tried in this order if the main model keeps failing with a temporary error.
# Set GEMINI_FALLBACK_MODELS in .env to change them (comma-separated),
# or set it to nothing (GEMINI_FALLBACK_MODELS=) to turn the fallbacks off.
DEFAULT_FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
CACHE_DIR = Path(".cache") / "llm"  # ".cache" is already in .gitignore

# A model that has another model after it gets only a few tries before we move on,
# so a busy model doesn't hold things up. The LAST model has nothing to fall back
# to, so it gets the full number of tries.
ATTEMPTS_BEFORE_SWITCHING = 2
# How long to wait for ONE request before giving up on it. The biggest syllabus we
# have (about 50 events) took 16-32 seconds on a normal day, so 60 leaves room
# without letting a stuck model hold things up for minutes.
REQUEST_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 5  # tries for the last model, including the first
BASE_DELAY_SECONDS = 2.0  # waits are 2s, 4s, 8s, 16s (plus a little jitter)
MIN_SECONDS_BETWEEN_CALLS = 4.0  # pause between real API calls

# HTTP status codes that are worth retrying: timeouts, rate limits, server trouble.
# Anything else (400 bad request, 403 bad key, ...) won't fix itself, so we fail fast.
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# When the last real API call started (None = no call yet). Used for pacing.
_last_call_time: float | None = None


class LLMError(Exception):
    """Raised when we can't get a reply from the model. The message is user-safe."""


class LLMUnavailableError(LLMError):
    """The model kept failing with a temporary error (429/5xx) through every retry.
    This is the one failure where trying a different model can help."""


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def _models_to_try() -> list[str]:
    """The main model first, then each fallback in order (skipping blanks and repeats)."""
    models = [_model_name()]
    setting = os.environ.get("GEMINI_FALLBACK_MODELS")
    if setting is None:  # not set at all: use the defaults ("" means "no fallbacks")
        fallbacks = DEFAULT_FALLBACK_MODELS
    else:
        fallbacks = [name.strip() for name in setting.split(",")]
    for name in fallbacks:
        if name and name not in models:
            models.append(name)
    return models


# --- cache ---


def _cache_path(prompt: str, model: str) -> Path:
    """One file per (model, prompt). Including the model means a reply is
    always filed under the model that actually wrote it."""
    key = hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.txt"


def _read_cache(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _write_cache(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- talking to Gemini ---


def _send_request(prompt: str, model: str) -> str:
    """One call to Gemini, no retries. The only function that uses the SDK."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")

    # The SDK retries 429s by itself by default (5 requests per call). We do our
    # own retrying below, so turn the SDK's off; otherwise the two would multiply.
    client = genai.Client(api_key=api_key, http_options={"retry_options": {"attempts": 1}})

    interaction = client.interactions.create(
        model=model,
        input=prompt,
        store=False,  # don't keep a copy of the syllabus text on Google's servers
        response_format={"type": "text", "mime_type": "application/json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return interaction.output_text


def _is_retryable(error: Exception) -> bool:
    # The SDK's API errors carry the HTTP status as `status_code`. Other errors
    # (like our own LLMError) don't have one, so they are never retried.
    return getattr(error, "status_code", None) in RETRYABLE_STATUS_CODES


def _is_connection_problem(error: Exception) -> bool:
    """True if the request timed out or never got through (as opposed to Google
    answering with an error). The SDK's timeout error is a kind of connection
    error, so checking the class names up the family tree covers both. We check
    names because the SDK keeps these classes in a private module we shouldn't import."""
    return any(cls.__name__ == "APIConnectionError" for cls in type(error).__mro__)


def _backoff_delay(attempt: int) -> float:
    """Seconds to wait after failed attempt number `attempt` (1, 2, 3, ...)."""
    # Jitter (a random extra bit) stops many clients retrying at the same instant.
    return BASE_DELAY_SECONDS * 2 ** (attempt - 1) + random.uniform(0, 1)


def _wait_for_pacing() -> None:
    """Sleep if the last real call was less than MIN_SECONDS_BETWEEN_CALLS ago."""
    global _last_call_time
    if _last_call_time is not None:
        wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_time)
        if wait > 0:
            time.sleep(wait)
    _last_call_time = time.monotonic()


def _send_with_retries(prompt: str, model: str, attempts: int) -> str:
    for attempt in range(1, attempts + 1):
        _wait_for_pacing()
        try:
            return _send_request(prompt, model)
        except LLMError:
            raise  # already has a clear message (e.g. missing key)
        except Exception as error:
            if _is_connection_problem(error):
                # A timeout means this model is too slow right now, and trying it
                # again would just wait out another full timeout. Report it as
                # unavailable so the next model in the list gets a turn.
                raise LLMUnavailableError(
                    f"Gemini ({model}) did not respond in time or could not be reached "
                    f"({type(error).__name__})."
                ) from error
            message = f"Gemini request failed ({model}): {type(error).__name__}: {error}"
            if not _is_retryable(error):
                raise LLMError(message) from error  # retrying (or switching models) won't help
            if attempt == attempts:
                raise LLMUnavailableError(message) from error
            time.sleep(_backoff_delay(attempt))
    raise LLMError("Gemini request failed.")  # unreachable; keeps type checkers happy


def _ask_model(prompt: str, model: str, attempts: int) -> str:
    """Send the prompt to one model (up to `attempts` tries) and cache a successful reply."""
    text = _send_with_retries(prompt, model, attempts)
    _write_cache(_cache_path(prompt, model), text)  # only successful replies are cached
    return text


def call_llm(prompt: str) -> str:
    """Return the model's reply (JSON text) for `prompt`.

    Order: (1) the cache, checking every model in the list; (2) each model in
    turn, giving up on it after ATTEMPTS_BEFORE_SWITCHING tries if it keeps
    failing with a temporary error; (3) the last model gets the full MAX_ATTEMPTS.
    Any other kind of error (bad key, bad request) is raised straight away.
    """
    models = _models_to_try()

    for model in models:
        cached = _read_cache(_cache_path(prompt, model))
        if cached is not None:
            return cached

    for model in models[:-1]:
        try:
            return _ask_model(prompt, model, ATTEMPTS_BEFORE_SWITCHING)
        except LLMUnavailableError:
            pass  # this model is overloaded; move on to the next one
    return _ask_model(prompt, models[-1], MAX_ATTEMPTS)  # the last model's errors are raised
