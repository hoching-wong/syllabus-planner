"""Group weights: a total stated for a group of events must be split between them.

The EAS110 syllabus's marking scheme is all group totals ("Quizzes 15%" for 31
quizzes, "Midterm (Oral & Written) 20%" for two parts). The model once left the
quiz and homework weights empty and repeated the midterm, final-exam and project
totals on every part, so the weights added up to over 100%.

Offline tests replay a recorded real reply; one opt-in live test (`pytest -m live`)
asks the real model.
"""

import hashlib
import json
from pathlib import Path

import pytest

from planner import extractor
from planner.extractor import build_prompt, extract_events
from planner.llm import LLMUnavailableError
from planner.models import Event
from planner.pdf_reader import read_pdf

FIXTURES = Path(__file__).parent / "fixtures"
PDF_PATH = FIXTURES / "EAS110Y1_Syllabus.pdf"
RECORDING = json.loads((FIXTURES / "EAS110Y1_recorded_reply.json").read_text(encoding="utf-8"))
TERM_YEAR = 2026

# Straight from the marking scheme. `total` is the stated weight of the whole group.
# `count` is how many events the schedule lists (checked only where it's clear-cut).
# `tolerance` allows for the model rounding each share to two decimals.
GROUPS = [
    {"name": "Quizzes", "word": "quiz", "total": 15, "count": 31, "tolerance": 0.3},
    {"name": "Homework", "word": "homework", "total": 10, "count": 16, "tolerance": 0.3},
    {"name": "Midterm test", "word": "midterm", "total": 20, "tolerance": 0.01},
    {"name": "Final exam", "word": "final exam", "total": 30, "tolerance": 0.01},
    {"name": "Final oral project", "word": "project", "total": 5, "tolerance": 0.01},
]


def members_of(group: dict, events: list[Event]) -> list[Event]:
    return [event for event in events if group["word"] in event.title.lower()]


def check_group_weights(events: list[Event]) -> list[str]:
    """Return a list of problems. An empty list means the weights are consistent."""
    problems = []
    for group in GROUPS:
        members = members_of(group, events)
        if not members:
            problems.append(f"{group['name']}: no events found")
            continue
        if "count" in group and len(members) != group["count"]:
            problems.append(f"{group['name']}: found {len(members)} events, the syllabus lists {group['count']}")
        total = sum(e.weight_percent for e in members if e.weight_percent is not None)
        if abs(total - group["total"]) > group["tolerance"]:
            problems.append(f"{group['name']}: weights add up to {total:.2f}, the syllabus says {group['total']}")

    everything = sum(e.weight_percent or 0 for e in events)
    if everything > 100.5:
        problems.append(f"all weights add up to {everything:.2f}, which is over 100")
    return problems


# --- tests of the checker itself ---


def event(title: str, weight: float | None) -> Event:
    return Event(course="EAS110", title=title, type="other", weight_percent=weight, confidence=1)


def consistent_events() -> list[Event]:
    """A set of events that follows the marking scheme exactly."""
    titles = {
        "Quizzes": "Quiz {}",
        "Homework": "Homework {}",
        "Midterm test": "Midterm Test part {}",
        "Final exam": "Final Exam part {}",
        "Final oral project": "Final Oral Project part {}",
    }
    events = []
    for group in GROUPS:
        parts = group.get("count", 2)
        for number in range(1, parts + 1):
            events.append(event(titles[group["name"]].format(number), round(group["total"] / parts, 2)))
    return events


def test_checker_accepts_consistent_weights():
    assert check_group_weights(consistent_events()) == []


def test_checker_catches_a_total_repeated_on_every_part():
    events = [e for e in consistent_events() if "Midterm" not in e.title]
    events += [event("Midterm Test (Oral)", 20), event("Midterm Test (Written)", 20)]  # the old mistake
    problems = check_group_weights(events)
    assert any(p.startswith("Midterm test: weights add up to 40") for p in problems)


def test_checker_catches_missing_group_weights():
    events = [event(e.title, None) if "Quiz" in e.title else e for e in consistent_events()]
    problems = check_group_weights(events)
    assert any(p.startswith("Quizzes: weights add up to 0.00") for p in problems)


def test_checker_catches_a_missing_group_and_a_wrong_count():
    events = [e for e in consistent_events() if "Homework" not in e.title and e.title != "Quiz 31"]
    problems = check_group_weights(events)
    assert "Homework: no events found" in problems
    assert any(p.startswith("Quizzes: found 30 events") for p in problems)


def test_checker_catches_a_total_over_100():
    problems = check_group_weights(consistent_events() + [event("Participation", 50)])
    assert any("over 100" in p for p in problems)


# --- offline: replay the recorded real reply ---


@pytest.fixture(scope="module")
def syllabus_text() -> str:
    return read_pdf(PDF_PATH)


def test_recording_matches_the_current_prompt(syllabus_text):
    """Fails if the prompt or the PDF text changed after the reply was recorded.
    The fix is: python -m tests.record_reply, then check the tests again."""
    current = hashlib.sha256(build_prompt(syllabus_text, TERM_YEAR).encode("utf-8")).hexdigest()
    assert current == RECORDING["prompt_sha256"], "Prompt or PDF reader changed: re-record the reply."


def test_recorded_reply_has_consistent_group_weights(monkeypatch, syllabus_text):
    monkeypatch.setattr(extractor, "call_llm", lambda prompt: RECORDING["reply"])
    result = extract_events(syllabus_text, course="EAS110", default_year=TERM_YEAR)
    assert result.skipped == []
    assert check_group_weights(result.events) == []


# --- live: the real model (opt-in, uses free-tier quota unless the reply is cached) ---


@pytest.mark.live
def test_live_model_splits_group_weights_correctly(syllabus_text):
    try:
        result = extract_events(syllabus_text, course="EAS110", default_year=TERM_YEAR)
    except LLMUnavailableError as error:
        pytest.skip(f"Gemini is overloaded, can't check this right now: {error}")
    assert result.skipped == []
    assert check_group_weights(result.events) == []
