"""Text shown to the user: course-name guesses and review notes. Pure logic."""

import re
from pathlib import Path

from planner.editing import sort_events
from planner.models import Event, EventType

LOW_CONFIDENCE = 0.7  # below this, tell the user to double-check the event


def default_course_name(filename: str) -> str:
    """Guess a course name from a file name: 'MAT235H1_Syllabus.pdf' -> 'MAT235H1'."""
    stem = Path(filename).stem
    # Cut from the word "syllabus" to the end, plus any separators just before it.
    name = re.sub(r"[ _-]*syllabus.*$", "", stem, flags=re.IGNORECASE)
    return name or stem  # if nothing is left (file is just "syllabus.pdf"), keep the stem


def _problems_with(event: Event) -> list[str]:
    problems = []
    if event.due_date is None:
        problems.append("no date given")
    if event.confidence < LOW_CONFIDENCE:
        problems.append("low confidence")
    return problems


def review_notes(events: list[Event], tutorial_day: str | None = None) -> list[str]:
    """A short list of things the user should double-check.

    One line per event that has a problem, then one general line about quiz
    dates (a single line, not one per quiz). `tutorial_day` is the day name the
    user picked, or None if they haven't picked one.
    """
    notes = []
    for event in sort_events(events):
        problems = _problems_with(event)
        if problems:
            notes.append(f"{event.course} – {event.title}: {', '.join(problems)}")

    if any(e.type == EventType.QUIZ and e.due_date is not None for e in events):
        if tutorial_day:
            notes.append(f"Quiz dates were moved to your tutorial day ({tutorial_day}).")
        else:
            notes.append(
                "Quiz dates are the syllabus's nominal day. "
                "Set your tutorial day in the sidebar if yours is different."
            )
    return notes
