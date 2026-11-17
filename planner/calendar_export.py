"""Events -> .ics calendar text (all-day entries on each due date).

Events with no date can't go on a calendar, so they are skipped. The app tells the user.
"""

import hashlib

from ics import Calendar
from ics import Event as CalendarEntry  # renamed so it isn't confused with our own Event

from planner.models import Event


def _percent(value: float) -> str:
    """25.0 -> '25', 22.5 -> '22.5' (same style as the rest of the app)."""
    return f"{round(value, 2):g}"


def entry_title(event: Event) -> str:
    """'CS101 – Midterm (25%)'; the bracket is left out when the weight is unknown."""
    title = f"{event.course} – {event.title}"
    if event.weight_percent is not None:
        title += f" ({_percent(event.weight_percent)}%)"
    return title


def entry_description(event: Event) -> str:
    """The type, plus the weight when known."""
    description = f"Type: {event.type.value}"
    if event.weight_percent is not None:
        description += f"\nWeight: {_percent(event.weight_percent)}%"
    return description


def entry_uid(event: Event) -> str:
    """A fixed ID built from course, title and date.

    Calendar apps use the ID to recognise an event. Exporting twice gives the same IDs,
    so importing the second file updates the entries instead of duplicating them.
    """
    key = f"{event.course}|{event.title}|{event.due_date.isoformat()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16] + "@syllabus-planner"


def _to_entry(event: Event) -> CalendarEntry:
    entry = CalendarEntry(
        name=entry_title(event),
        begin=event.due_date,
        description=entry_description(event),
        uid=entry_uid(event),
    )
    entry.make_all_day()  # a date with no time of day
    return entry


def dated_events(events: list[Event]) -> list[Event]:
    """The events that can be exported (they have a date)."""
    return [event for event in events if event.due_date is not None]


def events_to_ics(events: list[Event]) -> str:
    """The text of an .ics file with one all-day entry per dated event."""
    calendar = Calendar()
    for event in dated_events(events):
        calendar.events.add(_to_entry(event))
    return calendar.serialize()
