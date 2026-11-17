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
CACHE_DIR = Path(".cache") / "llm"  # ".cache" is already in .gitignore

MAX_ATTEMPTS = 5  # total tries per prompt, including the first
BASE_DELAY_SECONDS = 2.0  # waits are 2s, 4s, 8s, 16s (plus a little jitter)
MIN_SECONDS_BETWEEN_CALLS = 4.0  # pause between real API calls

# HTTP status codes that are worth retrying: timeouts, rate limits, server trouble.
# Anything else (400 bad request, 403 bad key, ...) won't fix itself, so we fail fast.
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# When the last real API call started (None = no call yet). Used for pacing.
_last_call_time: float | None = None


class LLMError(Exception):
    """Raised when we can't get a reply from the model. The message is user-safe."""


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


# --- cache ---


def _cache_path(prompt: str) -> Path:
    """One file per (model, prompt). Including the model means switching
    models never returns an answer that a different model gave."""
    key = hashlib.sha256(f"{_model_name()}\n{prompt}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.txt"


def _read_cache(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _write_cache(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- talking to Gemini ---


def _send_request(prompt: str) -> str:
    """One call to Gemini, no retries. The only function that uses the SDK."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")

    # The SDK retries 429s by itself by default (5 requests per call). We do our
    # own retrying below, so turn the SDK's off; otherwise the two would multiply.
    client = genai.Client(api_key=api_key, http_options={"retry_options": {"attempts": 1}})

    interaction = client.interactions.create(
        model=_model_name(),
        input=prompt,
        store=False,  # don't keep a copy of the syllabus text on Google's servers
        response_format={"type": "text", "mime_type": "application/json"},
    )
    return interaction.output_text


def _is_retryable(error: Exception) -> bool:
    # The SDK's API errors carry the HTTP status as `status_code`. Other errors
    # (like our own LLMError) don't have one, so they are never retried.
    return getattr(error, "status_code", None) in RETRYABLE_STATUS_CODES


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


def _send_with_retries(prompt: str) -> str:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        _wait_for_pacing()
        try:
            return _send_request(prompt)
        except LLMError:
            raise  # already has a clear message (e.g. missing key)
        except Exception as error:
            if not _is_retryable(error) or attempt == MAX_ATTEMPTS:
                raise LLMError(f"Gemini request failed: {type(error).__name__}: {error}") from error
            time.sleep(_backoff_delay(attempt))
    raise LLMError("Gemini request failed.")  # unreachable; keeps type checkers happy


def call_llm(prompt: str) -> str:
    """Return the model's reply (JSON text) for `prompt`, using the cache first."""
    path = _cache_path(prompt)
    cached = _read_cache(path)
    if cached is not None:
        return cached

    text = _send_with_retries(prompt)
    _write_cache(path, text)  # only successful replies are cached
    return text
