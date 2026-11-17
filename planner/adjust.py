"""Display-time tweaks to events. Pure logic, no UI.

A syllabus shows quizzes on one nominal day, but tutorials run on different
days for different students. This moves quiz dates to the user's own day.
"""

from datetime import timedelta

from planner.models import Event, EventType


def courses_with_quizzes(events: list[Event]) -> list[str]:
    """Course names that have at least one quiz, each once, in alphabetical order.
    These are the courses that need a tutorial-day setting."""
    return sorted({event.course for event in events if event.type == EventType.QUIZ})


def move_to_weekday(events: list[Event], weekday: int, course: str | None = None) -> list[Event]:
    """Return new events with each dated quiz moved to `weekday` of its own week.

    `weekday` uses Python's numbering (Monday = 0 ... Sunday = 6) and weeks run
    Monday to Sunday. If `course` is given, only that course's quizzes move
    (each course can have a different tutorial day); otherwise all quizzes do.
    Other event types, and quizzes with no date, are untouched.
    The events passed in are never modified.
    """
    if not 0 <= weekday <= 6:
        raise ValueError("weekday must be between 0 (Monday) and 6 (Sunday)")

    moved = []
    for event in events:
        is_target = course is None or event.course == course
        if is_target and event.type == EventType.QUIZ and event.due_date is not None:
            monday = event.due_date - timedelta(days=event.due_date.weekday())
            # model_copy makes a new Event with just this field replaced.
            event = event.model_copy(update={"due_date": monday + timedelta(days=weekday)})
        moved.append(event)
    return moved
