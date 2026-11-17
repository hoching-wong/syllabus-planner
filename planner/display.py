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


def duplicate_names(names: list[str]) -> list[str]:
    """File names that appear more than once, each listed once (case matters)."""
    seen = set()
    duplicates = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def _problems_with(event: Event) -> list[str]:
    problems = []
    if event.due_date is None:
        problems.append("no date given")
    if event.confidence < LOW_CONFIDENCE:
        problems.append("low confidence")
    return problems


def review_notes(events: list[Event], tutorial_days: dict[str, str] | None = None) -> list[str]:
    """A short list of things the user should double-check.

    One line per event that has a problem, then one line per course about its
    quiz dates (not one per quiz). `tutorial_days` maps a course name to the day
    name the user picked for it; courses missing from it have no day picked.
    """
    tutorial_days = tutorial_days or {}
    notes = []
    for event in sort_events(events):
        problems = _problems_with(event)
        if problems:
            notes.append(f"{event.course} – {event.title}: {', '.join(problems)}")

    courses = sorted({e.course for e in events if e.type == EventType.QUIZ and e.due_date is not None})
    for course in courses:
        if course in tutorial_days:
            notes.append(f"{course}: quiz dates were moved to your tutorial day ({tutorial_days[course]}).")
        else:
            notes.append(
                f"{course}: quiz dates are the syllabus's nominal day. "
                "Set your tutorial day in the sidebar if yours is different."
            )
    return notes
