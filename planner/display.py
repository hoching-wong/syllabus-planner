"""Text shown to the user: course-name guesses and review notes. Pure logic."""

import re
from datetime import date
from pathlib import Path

from planner.editing import sort_events
from planner.models import Event, EventType
from planner.workload import Workload

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


# --- the weekly workload view ---


def _percent(value: float) -> str:
    # round() hides float noise (30.299999999999997 -> 30.3); :g drops a trailing .0 (25.0 -> 25).
    return f"{round(value, 2):g}%"


def _day(day: date) -> str:
    return f"{day:%b} {day.day}, {day.year}"  # e.g. "Oct 19, 2026" (no zero-padded day)


def crunch_warnings(workload: Workload) -> list[str]:
    """One line per crunch week, in date order, naming each course's share."""
    lines = []
    for week in workload.weeks:
        if not week.is_crunch:
            continue
        # Biggest share first; courses with nothing due (weight 0) aren't worth listing.
        shares = sorted(
            ((course, weight) for course, weight in week.weight_by_course.items() if weight > 0),
            key=lambda share: (-share[1], share[0]),
        )
        detail = ", ".join(f"{course} {_percent(weight)}" for course, weight in shares)
        lines.append(f"Week of {_day(week.week_start)}: {_percent(week.total_weight)} of your grades are due ({detail}).")
    return lines


def unscheduled_note(workload: Workload) -> str | None:
    """Say how much grade weight isn't in the weekly view because it has no date."""
    if workload.unscheduled_weight <= 0:
        return None
    count = workload.unscheduled_count
    events = "event" if count == 1 else "events"
    return (
        f"{_percent(workload.unscheduled_weight)} of your grades ({count} {events}) "
        "has no date, so it isn't in the weekly view."
    )


def chart_rows(workload: Workload) -> list[dict]:
    """One row per week for the bar chart. Crunch weeks go in their own column so
    the chart can colour them differently."""
    rows = []
    for week in workload.weeks:
        weight = round(week.total_weight, 2)
        rows.append(
            {
                "Week": week.week_start.isoformat(),  # ISO dates sort correctly as text
                "Crunch weeks": weight if week.is_crunch else 0.0,
                "Other weeks": 0.0 if week.is_crunch else weight,
            }
        )
    return rows
