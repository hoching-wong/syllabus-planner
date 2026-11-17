from datetime import date

import pandas as pd
import pytest

from planner.editing import COLUMNS, TYPE_OPTIONS, events_to_rows, rows_to_events, sort_events
from planner.models import Event


def make_event(**overrides) -> Event:
    fields = {
        "course": "CS101",
        "title": "Midterm",
        "type": "exam",
        "due_date": date(2026, 10, 20),
        "weight_percent": 25.0,
        "confidence": 0.9,
    }
    fields.update(overrides)
    return Event(**fields)


def make_row(**overrides) -> dict:
    row = {
        "Date": date(2026, 10, 20),
        "Course": "CS101",
        "Title": "Midterm",
        "Type": "exam",
        "Weight": 25.0,
        "Confidence": 0.9,
    }
    row.update(overrides)
    return row


# --- sorting ---


def test_events_are_sorted_by_date_then_course_then_title():
    events = [
        make_event(title="B", due_date=date(2026, 11, 1)),
        make_event(title="Z", due_date=date(2026, 10, 1), course="CS101"),
        make_event(title="A", due_date=date(2026, 10, 1), course="BIO200"),
        make_event(title="C", due_date=date(2026, 10, 1), course="CS101"),
    ]
    assert [(e.course, e.title) for e in sort_events(events)] == [
        ("BIO200", "A"),
        ("CS101", "C"),
        ("CS101", "Z"),
        ("CS101", "B"),
    ]


def test_events_without_a_date_go_last():
    events = [make_event(title="Final", due_date=None), make_event(title="Midterm")]
    assert [e.title for e in sort_events(events)] == ["Midterm", "Final"]


# --- events -> rows ---


def test_row_holds_raw_editable_values():
    (row,) = events_to_rows([make_event()])
    assert row == make_row()
    assert list(row) == COLUMNS


def test_missing_date_and_weight_stay_empty():
    (row,) = events_to_rows([make_event(due_date=None, weight_percent=None)])
    assert row["Date"] is None and row["Weight"] is None


def test_rows_follow_the_sorted_order():
    rows = events_to_rows([make_event(title="Final", due_date=None), make_event(title="Midterm")])
    assert [r["Title"] for r in rows] == ["Midterm", "Final"]


def test_type_options_are_the_event_types():
    assert TYPE_OPTIONS == ["assignment", "exam", "quiz", "project", "ongoing", "other"]


# --- rows -> events ---


def test_unedited_rows_turn_back_into_the_same_events():
    events = [
        make_event(),
        make_event(title="Final", due_date=None, weight_percent=None),
        make_event(title="Quiz 1", type="quiz", weight_percent=2.5, confidence=0.4),
    ]
    rebuilt, problems = rows_to_events(events_to_rows(events))
    assert problems == []
    assert rebuilt == sort_events(events)


def test_edited_values_are_picked_up():
    row = make_row(Date=date(2026, 10, 27), Title="Midterm (moved)", Weight=30.0, Type="quiz")
    (event,), problems = rows_to_events([row])
    assert problems == []
    assert (event.due_date, event.title, event.weight_percent, event.type.value) == (
        date(2026, 10, 27),
        "Midterm (moved)",
        30.0,
        "quiz",
    )


def test_blank_cells_from_the_table_become_none():
    # The table hands back NaN for an empty number cell and NaT for an empty date cell.
    row = make_row(Date=pd.NaT, Weight=float("nan"))
    (event,), _ = rows_to_events([row])
    assert event.due_date is None and event.weight_percent is None


def test_none_cells_become_none():
    (event,), _ = rows_to_events([make_row(Date=None, Weight=None)])
    assert event.due_date is None and event.weight_percent is None


def test_timestamps_become_plain_dates():
    (event,), _ = rows_to_events([make_row(Date=pd.Timestamp("2026-10-20"))])
    assert event.due_date == date(2026, 10, 20)
    assert type(event.due_date) is date


def test_weight_out_of_range_becomes_not_stated():
    (event,), problems = rows_to_events([make_row(Weight=150.0)])
    assert problems == [] and event.weight_percent is None


def test_new_row_with_only_a_title_gets_sensible_defaults():
    row = {"Date": None, "Course": None, "Title": "Study group", "Type": None, "Weight": float("nan"), "Confidence": float("nan")}
    (event,), problems = rows_to_events([row], default_course="CS101")
    assert problems == []
    assert event.course == "CS101"  # filled from the default
    assert event.type.value == "other"
    assert event.confidence == 1.0  # the user typed it, so they're sure


def test_typed_course_beats_the_default():
    (event,), _ = rows_to_events([make_row(Course="BIO200")], default_course="CS101")
    assert event.course == "BIO200"


def test_row_with_an_empty_title_is_ignored_and_explained():
    events, problems = rows_to_events([make_row(Title="Keep"), make_row(Title=""), make_row(Title=None)])
    assert [e.title for e in events] == ["Keep"]
    assert len(problems) == 2
    assert "Row 2" in problems[0] and "title" in problems[0]
    assert "Row 3" in problems[1]


def test_row_with_no_course_and_no_default_is_ignored():
    events, problems = rows_to_events([make_row(Course="")])
    assert events == []
    assert "Row 1" in problems[0] and "course" in problems[0]


def test_row_order_is_kept():
    rows = [make_row(Title="Second", Date=date(2026, 12, 1)), make_row(Title="First", Date=date(2026, 9, 1))]
    events, _ = rows_to_events(rows)
    assert [e.title for e in events] == ["Second", "First"]  # the user's order, not re-sorted


def test_empty_table_gives_no_events():
    assert rows_to_events([]) == ([], [])


@pytest.mark.parametrize("bad_confidence", [5, -1])
def test_out_of_range_confidence_is_reported(bad_confidence):
    events, problems = rows_to_events([make_row(Confidence=bad_confidence)])
    assert events == [] and "confidence" in problems[0]
