"""Does the whole pipeline find the right events in a real syllabus (MAT235)?

Three layers:
  - tests of the scorer itself (a scorer that can't fail proves nothing)
  - offline tests: replay a recorded real model reply and score it, plus a check
    that the recording still matches the current prompt
  - one live test (opt-in): `pytest -m live` calls the real model
"""

import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

from planner import extractor
from planner.extractor import build_prompt, extract_events
from planner.llm import LLMUnavailableError
from planner.models import Event
from planner.pdf_reader import read_pdf

FIXTURES = Path(__file__).parent / "fixtures"
PDF_PATH = FIXTURES / "MAT235H1_Syllabus.pdf"
KEY = json.loads((FIXTURES / "MAT235H1_expected.json").read_text(encoding="utf-8"))
RECORDING = json.loads((FIXTURES / "MAT235H1_recorded_reply.json").read_text(encoding="utf-8"))


# --- the scorer: compares extracted events with the hand-labelled answer key ---


def find_match(events: list[Event], label: str) -> Event | None:
    """Find the event the answer-key entry `label` refers to ("Quiz 3", "Term Test 1", ...).
    Matching is loose on purpose: the model may title things "Tutorial Quiz 3" or "TT1"."""
    number = re.search(r"\d", label)
    for event in events:
        title = event.title.lower()
        if label.startswith("Quiz"):
            if event.type.value == "quiz" and re.search(rf"quiz\s*{number.group()}\b", title):
                return event
        elif label.startswith("Term Test"):
            n = number.group()
            if event.type.value == "exam" and any(s in title for s in (f"test {n}", f"tt{n}", f"midterm {n}")):
                return event
        elif label == "Final Exam" and "final" in title:
            return event
    return None


def score(events: list[Event], key: dict) -> list[str]:
    """Return a list of problems. An empty list means a perfect match."""
    problems = []
    matched_ids = set()

    for want in key["events"]:
        label = want["label"]
        event = find_match(events, label)
        if event is None:
            problems.append(f"{label}: missing")
            continue
        matched_ids.add(id(event))

        if event.type.value != want["type"]:
            problems.append(f"{label}: type is {event.type.value}, expected {want['type']}")

        if "due_week_of" in want:  # only the week is known, so any day in that week is fine
            monday = date.fromisoformat(want["due_week_of"])
            in_week = event.due_date is not None and monday <= event.due_date <= monday + timedelta(days=6)
            if not in_week:
                problems.append(f"{label}: date {event.due_date} is not in the week of {monday}")
        else:
            expected = date.fromisoformat(want["due_date"]) if want["due_date"] else None
            if event.due_date != expected:
                problems.append(f"{label}: date is {event.due_date}, expected {expected}")

        if event.weight_percent not in want["weight_percent"]:
            problems.append(f"{label}: weight is {event.weight_percent}, expected one of {want['weight_percent']}")

    extras = [e.title for e in events if id(e) not in matched_ids]
    if extras:
        problems.append(f"unexpected extra events: {extras}")
    return problems


def events_matching_the_key() -> list[Event]:
    """Build events that follow the answer key exactly, to feed the scorer."""
    events = []
    for want in KEY["events"]:
        due = want.get("due_date") or want.get("due_week_of")
        events.append(
            Event(
                course=KEY["course"],
                title=want["label"],
                type=want["type"],
                due_date=due,
                weight_percent=want["weight_percent"][0],
                confidence=1,
            )
        )
    return events


def test_scorer_accepts_events_that_match_the_key():
    assert score(events_matching_the_key(), KEY) == []


def test_scorer_accepts_any_day_in_the_right_week():
    events = events_matching_the_key()
    quiz1 = find_match(events, "Quiz 1")
    quiz1.due_date = quiz1.due_date + timedelta(days=4)  # Monday -> Friday, same week
    assert score(events, KEY) == []


def test_scorer_accepts_either_allowed_quiz_weight():
    events = events_matching_the_key()
    find_match(events, "Quiz 2").weight_percent = None  # the key allows 2.5 or null
    assert score(events, KEY) == []


def test_scorer_reports_a_missing_event():
    events = [e for e in events_matching_the_key() if e.title != "Quiz 4"]
    assert score(events, KEY) == ["Quiz 4: missing"]


def test_scorer_reports_a_wrong_exam_date():
    events = events_matching_the_key()
    find_match(events, "Term Test 1").due_date = date(2026, 10, 20)
    problems = score(events, KEY)
    assert len(problems) == 1 and problems[0].startswith("Term Test 1: date")


def test_scorer_reports_a_quiz_date_in_the_wrong_week():
    events = events_matching_the_key()
    quiz = find_match(events, "Quiz 1")
    quiz.due_date = quiz.due_date + timedelta(days=7)
    problems = score(events, KEY)
    assert len(problems) == 1 and "not in the week" in problems[0]


def test_scorer_reports_a_wrong_weight():
    events = events_matching_the_key()
    find_match(events, "Quiz 3").weight_percent = 1.875  # the mistake the model once made
    problems = score(events, KEY)
    assert len(problems) == 1 and problems[0].startswith("Quiz 3: weight")


def test_scorer_reports_an_invented_final_exam_date():
    events = events_matching_the_key()
    find_match(events, "Final Exam").due_date = date(2026, 12, 15)
    problems = score(events, KEY)
    assert len(problems) == 1 and problems[0].startswith("Final Exam: date")


def test_scorer_reports_extra_events():
    extra = Event(course="MAT235H1", title="TT1 review", type="other", confidence=0.9)
    problems = score(events_matching_the_key() + [extra], KEY)
    assert problems == ["unexpected extra events: ['TT1 review']"]


# --- offline: replay the recorded model reply ---


@pytest.fixture(scope="module")
def syllabus_text() -> str:
    return read_pdf(PDF_PATH)


def test_recording_matches_the_current_prompt(syllabus_text):
    """Fails if the prompt or the PDF text changed after the reply was recorded.
    The fix is: python -m tests.record_reply, then check the accuracy tests again."""
    prompt = build_prompt(syllabus_text, KEY["default_year"])
    current = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert current == RECORDING["prompt_sha256"], "Prompt or PDF reader changed: re-record the reply."


def test_recorded_reply_is_extracted_correctly(monkeypatch, syllabus_text):
    monkeypatch.setattr(extractor, "call_llm", lambda prompt: RECORDING["reply"])
    result = extract_events(syllabus_text, course=KEY["course"], default_year=KEY["default_year"])
    assert result.skipped == []
    assert score(result.events, KEY) == []


# --- live: the real model (opt-in, uses free-tier quota unless the reply is cached) ---


@pytest.mark.live
def test_live_model_extracts_the_syllabus_correctly(syllabus_text):
    try:
        result = extract_events(syllabus_text, course=KEY["course"], default_year=KEY["default_year"])
    except LLMUnavailableError as error:
        pytest.skip(f"Gemini is overloaded, can't judge accuracy right now: {error}")
    assert result.skipped == []
    assert score(result.events, KEY) == []
