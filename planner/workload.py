"""Weekly workload and crunch weeks. Pure logic, no UI.

Adds up the grade weight due in each week, across all courses, and flags the
weeks that are too heavy. Weeks run Monday to Sunday, like in adjust.py.
"""

from dataclasses import dataclass
from datetime import date, timedelta

from planner.editing import sort_events
from planner.models import Event

DEFAULT_THRESHOLD = 25.0  # percent of the final grades due in one week


@dataclass
class WeekLoad:
    week_start: date  # the Monday
    events: list[Event]  # sorted by date, course, title
    total_weight: float  # all courses together; events with no weight add 0
    weight_by_course: dict[str, float]
    is_crunch: bool


@dataclass
class Workload:
    weeks: list[WeekLoad]  # from the first busy week to the last, quiet weeks included
    unscheduled_weight: float  # weight of events with no date (not in any week)
    unscheduled_count: int


def monday_of(day: date) -> date:
    """The Monday of the week containing `day` (weekday() is 0 for Monday)."""
    return day - timedelta(days=day.weekday())


def compute_workload(events: list[Event], threshold: float = DEFAULT_THRESHOLD) -> Workload:
    """Group events into weeks and flag the weeks whose total weight is at or above `threshold`."""
    if threshold <= 0:
        raise ValueError("threshold must be greater than 0")

    dated = [event for event in events if event.due_date is not None]
    undated = [event for event in events if event.due_date is None]
    unscheduled_weight = sum(event.weight_percent or 0 for event in undated)  # None counts as 0
    if not dated:
        return Workload([], unscheduled_weight, len(undated))

    # Put each dated event in the bucket for its week.
    by_week: dict[date, list[Event]] = {}
    for event in sort_events(dated):
        by_week.setdefault(monday_of(event.due_date), []).append(event)

    # Walk every week from the first to the last, so quiet weeks show up as zero.
    weeks = []
    week_start = min(by_week)
    last_week = max(by_week)
    while week_start <= last_week:
        week_events = by_week.get(week_start, [])
        weight_by_course: dict[str, float] = {}
        for event in week_events:
            weight_by_course[event.course] = weight_by_course.get(event.course, 0) + (event.weight_percent or 0)
        total = sum(weight_by_course.values())
        weeks.append(WeekLoad(week_start, week_events, total, weight_by_course, total >= threshold))
        week_start += timedelta(days=7)

    return Workload(weeks, unscheduled_weight, len(undated))
