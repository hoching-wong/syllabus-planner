from datetime import date

from ics import Calendar

from planner.calendar_export import (
    dated_events,
    entry_description,
    entry_title,
    entry_uid,
    events_to_ics,
)
from planner.models import Event


def make_event(**overrides) -> Event:
    fields = {
        "course": "CS101",
        "title": "Midterm",
        "type": "exam",
        "due_date": date(2026, 10, 20),
        "weight_percent": 25.0,
        "confidence": 0.3,
    }
    fields.update(overrides)
    return Event(**fields)


def parse(events: list[Event]):
    """Read the generated text back with the same library, as a calendar app would."""
    return sorted(Calendar(events_to_ics(events)).events, key=lambda e: e.begin)


# --- text of an entry ---


def test_title_has_course_name_and_weight():
    assert entry_title(make_event()) == "CS101 – Midterm (25%)"


def test_title_keeps_decimals_in_the_weight():
    assert entry_title(make_event(weight_percent=22.5)) == "CS101 – Midterm (22.5%)"


def test_title_has_no_brackets_without_a_weight():
    assert entry_title(make_event(weight_percent=None)) == "CS101 – Midterm"


def test_description_has_type_and_weight():
    assert entry_description(make_event()) == "Type: exam\nWeight: 25%"


def test_description_without_weight_has_only_the_type():
    assert entry_description(make_event(weight_percent=None)) == "Type: exam"


def test_low_confidence_is_not_mentioned():
    # The user reviews the table before exporting, so the calendar doesn't repeat the doubt.
    assert "confidence" not in entry_description(make_event(confidence=0.1)).lower()


# --- ids ---


def test_same_event_always_gets_the_same_id():
    assert entry_uid(make_event()) == entry_uid(make_event())


def test_id_changes_with_course_title_or_date():
    base = entry_uid(make_event())
    assert entry_uid(make_event(course="BIO200")) != base
    assert entry_uid(make_event(title="Final")) != base
    assert entry_uid(make_event(due_date=date(2026, 10, 21))) != base


def test_id_ignores_weight_so_a_corrected_weight_updates_the_entry():
    assert entry_uid(make_event(weight_percent=10.0)) == entry_uid(make_event(weight_percent=30.0))


# --- the whole file ---


def test_no_events_gives_a_valid_empty_calendar():
    assert parse([]) == []
    assert "BEGIN:VCALENDAR" in events_to_ics([])


def test_each_dated_event_becomes_one_entry():
    entries = parse([make_event(title="A"), make_event(title="B", due_date=date(2026, 11, 3))])
    assert [e.name for e in entries] == ["CS101 – A (25%)", "CS101 – B (25%)"]


def test_entries_are_all_day_on_the_due_date():
    (entry,) = parse([make_event()])
    assert entry.all_day
    assert entry.begin.date() == date(2026, 10, 20)
    assert entry.end.date() == date(2026, 10, 21)  # all-day entries end at the start of the next day


def test_description_and_id_are_in_the_file():
    event = make_event()
    (entry,) = parse([event])
    assert entry.description == "Type: exam\nWeight: 25%"
    assert entry.uid == entry_uid(event)


def test_undated_events_are_left_out():
    entries = parse([make_event(title="Dated"), make_event(title="Final", due_date=None)])
    assert [e.name for e in entries] == ["CS101 – Dated (25%)"]


def test_dated_events_filters_without_changing_the_input():
    events = [make_event(), make_event(due_date=None)]
    assert dated_events(events) == [events[0]]
    assert len(events) == 2


def test_same_input_gives_the_same_file_text_ids():
    first = Calendar(events_to_ics([make_event()])).events
    second = Calendar(events_to_ics([make_event()])).events
    assert [e.uid for e in first] == [e.uid for e in second]


def test_special_characters_survive_the_round_trip():
    (entry,) = parse([make_event(title="Essay, part 1; draft")])
    assert entry.name == "CS101 – Essay, part 1; draft (25%)"
