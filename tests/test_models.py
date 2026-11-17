from datetime import date

import pytest
from pydantic import ValidationError

from planner.models import Event, EventType


def make_event(**overrides) -> Event:
    """Build a valid Event, replacing any fields passed in."""
    fields = {
        "course": "CS101",
        "title": "Midterm",
        "type": "exam",
        "due_date": "2026-10-20",
        "weight_percent": 25,
        "confidence": 0.9,
    }
    fields.update(overrides)
    return Event(**fields)


def test_valid_event_is_accepted():
    event = make_event()
    assert event.course == "CS101"
    assert event.type == EventType.EXAM
    assert event.due_date == date(2026, 10, 20)
    assert event.weight_percent == 25.0
    assert event.confidence == 0.9


# --- due_date: a bad or missing date becomes None, the event is kept ---


@pytest.mark.parametrize("bad_date", [None, "", "TBA", "2026-13-45", "next week", 12345])
def test_bad_or_missing_date_becomes_none(bad_date):
    assert make_event(due_date=bad_date).due_date is None


def test_date_object_is_accepted():
    assert make_event(due_date=date(2026, 12, 1)).due_date == date(2026, 12, 1)


# --- weight_percent: clean up what we can, otherwise None ---


@pytest.mark.parametrize(
    "raw, expected",
    [(25, 25.0), ("25", 25.0), ("30%", 30.0), (" 12.5 % ", 12.5), (0, 0.0), (100, 100.0)],
)
def test_weight_is_cleaned_into_a_number(raw, expected):
    assert make_event(weight_percent=raw).weight_percent == expected


@pytest.mark.parametrize("bad_weight", [None, "", "lots", "TBA", 150, -5, "150%"])
def test_bad_or_out_of_range_weight_becomes_none(bad_weight):
    assert make_event(weight_percent=bad_weight).weight_percent is None


# --- type: unknown labels fall back to "other" ---


def test_type_is_case_insensitive():
    assert make_event(type="EXAM").type == EventType.EXAM


def test_unknown_type_becomes_other():
    assert make_event(type="midterm thingy").type == EventType.OTHER


# --- required fields and confidence are strict: these raise ---


def test_empty_title_is_rejected():
    with pytest.raises(ValidationError):
        make_event(title="")


def test_missing_course_is_rejected():
    with pytest.raises(ValidationError):
        Event(title="Midterm", type="exam", confidence=0.9)


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.5, "high"])
def test_bad_confidence_is_rejected(bad_confidence):
    with pytest.raises(ValidationError):
        make_event(confidence=bad_confidence)
