from datetime import date

import pytest

from planner.models import Event
from planner.workload import compute_workload

# Oct 19, 2026 is a Monday, so the week is Oct 19 (Mon) to Oct 25 (Sun).
MONDAY = date(2026, 10, 19)
SUNDAY = date(2026, 10, 25)
NEXT_MONDAY = date(2026, 10, 26)


def make_event(**overrides) -> Event:
    fields = {
        "course": "CS101",
        "title": "Midterm",
        "type": "exam",
        "due_date": MONDAY,
        "weight_percent": 10.0,
        "confidence": 1.0,
    }
    fields.update(overrides)
    return Event(**fields)


# --- grouping into weeks ---


def test_no_events_gives_an_empty_workload():
    workload = compute_workload([])
    assert workload.weeks == []
    assert workload.unscheduled_weight == 0
    assert workload.unscheduled_count == 0


def test_a_week_starts_on_monday():
    (week,) = compute_workload([make_event(due_date=date(2026, 10, 22))]).weeks  # a Thursday
    assert week.week_start == MONDAY


def test_monday_and_sunday_are_the_same_week_but_the_next_monday_is_not():
    events = [make_event(due_date=MONDAY), make_event(due_date=SUNDAY), make_event(due_date=NEXT_MONDAY)]
    weeks = compute_workload(events).weeks
    assert [w.week_start for w in weeks] == [MONDAY, NEXT_MONDAY]
    assert [len(w.events) for w in weeks] == [2, 1]


def test_weeks_are_in_date_order_whatever_the_input_order():
    events = [make_event(due_date=date(2026, 11, 9)), make_event(due_date=MONDAY)]
    assert [w.week_start for w in compute_workload(events).weeks] == [
        MONDAY,
        NEXT_MONDAY,  # a quiet week in between
        date(2026, 11, 2),
        date(2026, 11, 9),
    ]


def test_quiet_weeks_between_busy_ones_are_included_with_zero_weight():
    events = [make_event(due_date=MONDAY), make_event(due_date=date(2026, 11, 9))]
    weeks = compute_workload(events).weeks
    assert [w.week_start for w in weeks] == [MONDAY, date(2026, 10, 26), date(2026, 11, 2), date(2026, 11, 9)]
    quiet = weeks[1]
    assert quiet.total_weight == 0 and quiet.events == [] and not quiet.is_crunch


def test_weeks_across_new_year_work():
    events = [make_event(due_date=date(2026, 12, 29)), make_event(due_date=date(2027, 1, 5))]
    assert [w.week_start for w in compute_workload(events).weeks] == [date(2026, 12, 28), date(2027, 1, 4)]


# --- adding up weights ---


def test_weights_are_added_up_across_courses():
    events = [
        make_event(course="CS101", weight_percent=20.0),
        make_event(course="BIO200", weight_percent=15.0, title="Test"),
    ]
    (week,) = compute_workload(events).weeks
    assert week.total_weight == 35.0
    assert week.weight_by_course == {"CS101": 20.0, "BIO200": 15.0}


def test_several_events_from_one_course_add_up_in_its_total():
    events = [make_event(title="A", weight_percent=5.0), make_event(title="B", weight_percent=7.5)]
    (week,) = compute_workload(events).weeks
    assert week.weight_by_course == {"CS101": 12.5}


def test_an_event_with_no_weight_counts_as_zero_but_is_still_listed():
    events = [make_event(title="Known", weight_percent=10.0), make_event(title="Unknown", weight_percent=None)]
    (week,) = compute_workload(events).weeks
    assert week.total_weight == 10.0
    assert [e.title for e in week.events] == ["Known", "Unknown"]


def test_events_in_a_week_are_sorted_by_date_then_course_then_title():
    events = [
        make_event(title="Z", due_date=date(2026, 10, 21), course="CS101"),
        make_event(title="B", due_date=MONDAY, course="CS101"),
        make_event(title="A", due_date=MONDAY, course="BIO200"),
    ]
    (week,) = compute_workload(events).weeks
    assert [(e.course, e.title) for e in week.events] == [("BIO200", "A"), ("CS101", "B"), ("CS101", "Z")]


# --- events with no date ---


def test_undated_events_are_left_out_of_the_weeks_and_counted_separately():
    events = [
        make_event(title="Dated", weight_percent=10.0),
        make_event(title="Final", due_date=None, weight_percent=40.0),
        make_event(title="Participation", due_date=None, weight_percent=None),
    ]
    workload = compute_workload(events)
    assert [e.title for w in workload.weeks for e in w.events] == ["Dated"]
    assert workload.unscheduled_weight == 40.0  # a missing weight adds nothing
    assert workload.unscheduled_count == 2


def test_only_undated_events_give_no_weeks():
    workload = compute_workload([make_event(due_date=None, weight_percent=30.0)])
    assert workload.weeks == []
    assert workload.unscheduled_weight == 30.0


# --- the crunch flag ---


def test_a_week_below_the_threshold_is_not_a_crunch():
    (week,) = compute_workload([make_event(weight_percent=24.9)], threshold=25).weeks
    assert not week.is_crunch


def test_a_week_exactly_at_the_threshold_is_a_crunch():
    (week,) = compute_workload([make_event(weight_percent=25.0)], threshold=25).weeks
    assert week.is_crunch


def test_a_week_above_the_threshold_is_a_crunch():
    (week,) = compute_workload([make_event(weight_percent=40.0)], threshold=25).weeks
    assert week.is_crunch


def test_weights_from_different_courses_can_add_up_to_a_crunch():
    events = [make_event(course="CS101", weight_percent=15.0), make_event(course="BIO200", weight_percent=12.0)]
    (week,) = compute_workload(events, threshold=25).weeks
    assert week.is_crunch  # neither course alone reaches 25


def test_the_threshold_can_be_changed():
    events = [make_event(weight_percent=15.0)]
    assert not compute_workload(events, threshold=25).weeks[0].is_crunch
    assert compute_workload(events, threshold=10).weeks[0].is_crunch


def test_default_threshold_is_25():
    assert compute_workload([make_event(weight_percent=25.0)]).weeks[0].is_crunch
    assert not compute_workload([make_event(weight_percent=24.0)]).weeks[0].is_crunch


def test_a_week_of_unweighted_events_is_never_a_crunch():
    events = [make_event(title=str(n), weight_percent=None) for n in range(10)]
    assert not compute_workload(events).weeks[0].is_crunch


@pytest.mark.parametrize("bad_threshold", [0, -5])
def test_a_threshold_that_is_not_positive_is_rejected(bad_threshold):
    with pytest.raises(ValueError):
        compute_workload([make_event()], threshold=bad_threshold)


def test_the_input_list_is_not_changed():
    events = [make_event(due_date=date(2026, 11, 9)), make_event(due_date=MONDAY)]
    before = list(events)
    compute_workload(events)
    assert events == before
