"""Two more real syllabi (winter 2026), checked against answer keys I read off the PDFs by hand.

GGR124H1: three dated assignments, an undated final exam ("TBA") and weekly tutorial engagement.
MAT223H1: almost everything is recurring (weekly packets, weekly homework, a quiz every other
week), which should come out as undated "ongoing" items. Only the term test has a date:
"Tuesday or Wednesday in Week 8 (March 3 or March 4)".

Offline tests replay a recorded real reply; opt-in live tests (`pytest -m live`) ask the real model.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from planner import extractor
from planner.extractor import build_prompt, extract_events
from planner.llm import LLMUnavailableError
from planner.models import Event
from planner.pdf_reader import read_pdf

FIXTURES = Path(__file__).parent / "fixtures"
TERM_YEAR = 2026

# Each answer: a word that must be in the title, the allowed dates (None = no date), the weight
# and the type. Items that repeat ("weekly", "every other week") are "ongoing" with no date.
# The syllabus gives the term test as March 3 or 4; the prompt rule is "Monday of the week when
# unsure", so March 2 is accepted as well.
ANSWER_KEYS = {
    "GGR124H1": [
        {"word": "assignment #1", "dates": [date(2026, 1, 30)], "weight": 10, "type": "assignment"},
        {"word": "assignment #2", "dates": [date(2026, 2, 27)], "weight": 20, "type": "assignment"},
        {"word": "assignment #3", "dates": [date(2026, 3, 27)], "weight": 25, "type": "assignment"},
        {"word": "tutorial", "dates": [None], "weight": 10, "type": "ongoing"},
        {"word": "final exam", "dates": [None], "weight": 35, "type": "exam"},
    ],
    "MAT223H1": [
        {"word": "term test", "dates": [date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)], "weight": 25, "type": "exam"},
        {"word": "packet", "dates": [None], "weight": 10, "type": "ongoing"},
        {"word": "homework", "dates": [None], "weight": 15, "type": "ongoing"},
        {"word": "quiz", "dates": [None], "weight": 15, "type": "ongoing"},
        {"word": "final exam", "dates": [None], "weight": 35, "type": "exam"},
    ],
}


def check_against_key(events: list[Event], key: list[dict]) -> list[str]:
    """Return a list of problems. An empty list means the events match the answer key."""
    problems = []
    for answer in key:
        matches = [e for e in events if answer["word"] in e.title.lower()]
        if len(matches) != 1:
            problems.append(f"'{answer['word']}': expected 1 event, found {len(matches)}")
            continue
        (found,) = matches
        if found.due_date not in answer["dates"]:
            problems.append(f"'{answer['word']}': date is {found.due_date}, expected one of {answer['dates']}")
        if found.type.value != answer["type"]:
            problems.append(f"'{answer['word']}': type is {found.type.value}, expected {answer['type']}")
        if found.weight_percent is None or abs(found.weight_percent - answer["weight"]) > 0.01:
            problems.append(f"'{answer['word']}': weight is {found.weight_percent}, expected {answer['weight']}")
    if len(events) != len(key):
        problems.append(f"found {len(events)} events, expected {len(key)}")
    return problems


# --- tests of the checker itself ---


def make_event(title: str, due_date, weight, type="exam") -> Event:
    return Event(course="X", title=title, type=type, due_date=due_date, weight_percent=weight, confidence=1)


KEY = [{"word": "test", "dates": [date(2026, 3, 3)], "weight": 25, "type": "exam"}]


def test_checker_accepts_a_matching_event():
    assert check_against_key([make_event("Term Test", date(2026, 3, 3), 25)], KEY) == []


def test_checker_catches_a_wrong_date_a_wrong_weight_and_extra_events():
    events = [make_event("Term Test", date(2026, 3, 9), 20), make_event("Extra", None, None)]
    problems = check_against_key(events, KEY)
    assert any("date is 2026-03-09" in p for p in problems)
    assert any("weight is 20" in p for p in problems)
    assert any("found 2 events" in p for p in problems)


def test_checker_catches_a_wrong_type():
    problems = check_against_key([make_event("Term Test", date(2026, 3, 3), 25, type="other")], KEY)
    assert any("type is other, expected exam" in p for p in problems)


def test_checker_catches_a_missing_event_and_a_missing_weight():
    assert any("found 0" in p for p in check_against_key([], KEY))
    assert any("weight is None" in p for p in check_against_key([make_event("Test", date(2026, 3, 3), None)], KEY))


def test_answer_keys_add_up_to_100():
    for name, key in ANSWER_KEYS.items():
        assert sum(answer["weight"] for answer in key) == 100, name


# --- offline: replay the recorded real replies ---


def load_recording(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}_recorded_reply.json").read_text(encoding="utf-8"))


def syllabus_text(name: str) -> str:
    return read_pdf(FIXTURES / f"{name}_Syllabus.pdf")


@pytest.mark.parametrize("name", ANSWER_KEYS)
def test_recording_matches_the_current_prompt(name):
    """Fails if the prompt or the PDF text changed after the reply was recorded.
    The fix is: python -m tests.record_reply, then check the tests again."""
    current = hashlib.sha256(build_prompt(syllabus_text(name), TERM_YEAR).encode("utf-8")).hexdigest()
    assert current == load_recording(name)["prompt_sha256"], "Prompt or PDF reader changed: re-record the reply."


@pytest.mark.parametrize("name", ANSWER_KEYS)
def test_recorded_reply_matches_the_answer_key(monkeypatch, name):
    monkeypatch.setattr(extractor, "call_llm", lambda prompt: load_recording(name)["reply"])
    result = extract_events(syllabus_text(name), course=name, default_year=TERM_YEAR)
    assert result.skipped == []
    assert check_against_key(result.events, ANSWER_KEYS[name]) == []


# --- live: the real model (opt-in, uses free-tier quota unless the reply is cached) ---


@pytest.mark.live
@pytest.mark.parametrize("name", ANSWER_KEYS)
def test_live_model_matches_the_answer_key(name):
    try:
        result = extract_events(syllabus_text(name), course=name, default_year=TERM_YEAR)
    except LLMUnavailableError as error:
        pytest.skip(f"Gemini is overloaded, can't check this right now: {error}")
    assert result.skipped == []
    assert check_against_key(result.events, ANSWER_KEYS[name]) == []
