"""Events <-> rows of the editable table. Pure logic, no Streamlit.

The user can edit the table, so what comes back from it is untrusted, just like
the model's output: every row goes through the Event model again.
"""

from datetime import date, datetime
from typing import Any

from pydantic import ValidationError

from planner.models import Event, EventType, first_problem

# The table's columns, in order. These names are also what the table shows.
COLUMNS = ["Date", "Course", "Title", "Type", "Weight", "Confidence"]
TYPE_OPTIONS = [event_type.value for event_type in EventType]


def _sort_key(event: Event) -> tuple:
    # Python can't compare a date with None, so sort on (has no date, date).
    # Events without a date get True, which sorts after False, so they go last.
    return (event.due_date is None, event.due_date or date.max, event.course, event.title)


def sort_events(events: list[Event]) -> list[Event]:
    """Sort by date (undated last), then course, then title."""
    return sorted(events, key=_sort_key)


def events_to_rows(events: list[Event]) -> list[dict]:
    """One dict per event, in sorted order, holding the raw values the table edits."""
    return [
        {
            "Date": event.due_date,
            "Course": event.course,
            "Title": event.title,
            "Type": event.type.value,
            "Weight": event.weight_percent,
            "Confidence": event.confidence,
        }
        for event in sort_events(events)
    ]


def _clean(value: Any) -> Any:
    """Make empty cells uniform: None, NaN (empty number) and NaT (empty date) all become None.

    A timestamp becomes a plain date.
    """
    # NaN and NaT are the only values that are not equal to themselves, which
    # gives a way to spot them without importing pandas here.
    if value is None or value != value:
        return None
    if isinstance(value, datetime):
        return value.date()
    return value


def rows_to_events(rows: list[dict], default_course: str = "") -> tuple[list[Event], list[str]]:
    """Turn table rows back into events.

    Returns (events, problems). A row that can't become a valid Event (for
    example, an empty title) is left out and explained in `problems`.
    `default_course` fills the course of rows the user added without one.
    """
    events = []
    problems = []
    for number, row in enumerate(rows, start=1):
        confidence = _clean(row.get("Confidence"))
        try:
            events.append(
                Event(
                    course=_clean(row.get("Course")) or default_course,
                    title=_clean(row.get("Title")) or "",
                    type=_clean(row.get("Type")),  # empty -> "other" (see Event)
                    due_date=_clean(row.get("Date")),
                    weight_percent=_clean(row.get("Weight")),
                    # The model's confidence column can't be edited, so a row with
                    # none was typed by the user, who is sure of it.
                    confidence=1.0 if confidence is None else confidence,
                )
            )
        except ValidationError as error:
            problems.append(f"Row {number} ignored: {first_problem(error)}")
    return events, problems
