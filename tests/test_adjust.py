from datetime import date

import pytest

from planner.adjust import courses_with_quizzes, move_to_weekday
from planner.models import Event

MONDAY, THURSDAY, FRIDAY = 0, 3, 4  # Python's weekday numbers: Monday is 0


def make_event(title="Quiz 1", type="quiz", due=date(2026, 9, 17), course="MAT235H1") -> Event:
    """Sep 17, 2026 is a Thursday."""
    return Event(
        course=course, title=title, type=type, due_date=due, weight_percent=2.5, confidence=1
    )


def test_quiz_moves_to_the_chosen_weekday_of_its_week():
    (moved,) = move_to_weekday([make_event()], FRIDAY)
    assert moved.due_date == date(2026, 9, 18)  # Thursday Sep 17 -> Friday Sep 18


def test_quiz_can_move_earlier_in_the_week():
    (moved,) = move_to_weekday([make_event()], MONDAY)
    assert moved.due_date == date(2026, 9, 14)


def test_quiz_already_on_that_weekday_stays_put():
    (moved,) = move_to_weekday([make_event()], THURSDAY)
    assert moved.due_date == date(2026, 9, 17)


def test_weeks_run_monday_to_sunday():
    sunday_quiz = make_event(due=date(2026, 9, 20))  # Sunday of the Sep 14-20 week
    (moved,) = move_to_weekday([sunday_quiz], MONDAY)
    assert moved.due_date == date(2026, 9, 14)  # not the following Monday


def test_only_quizzes_are_moved():
    exam = make_event(title="Term Test 1", type="exam", due=date(2026, 10, 19))
    assignment = make_event(title="Essay", type="assignment")
    moved = move_to_weekday([exam, assignment], FRIDAY)
    assert [e.due_date for e in moved] == [date(2026, 10, 19), date(2026, 9, 17)]


def test_quiz_without_a_date_is_left_alone():
    (moved,) = move_to_weekday([make_event(due=None)], FRIDAY)
    assert moved.due_date is None


def test_other_fields_are_kept_and_the_original_is_not_changed():
    original = make_event()
    (moved,) = move_to_weekday([original], FRIDAY)
    assert (moved.course, moved.title, moved.weight_percent) == ("MAT235H1", "Quiz 1", 2.5)
    assert original.due_date == date(2026, 9, 17)  # untouched


def test_order_is_preserved():
    events = [make_event(title="Quiz 2"), make_event(title="Quiz 1")]
    assert [e.title for e in move_to_weekday(events, FRIDAY)] == ["Quiz 2", "Quiz 1"]


def test_only_the_named_courses_quizzes_are_moved():
    events = [make_event(course="CS101"), make_event(course="BIO200")]
    moved = move_to_weekday(events, FRIDAY, course="CS101")
    assert [(e.course, e.due_date) for e in moved] == [
        ("CS101", date(2026, 9, 18)),  # moved
        ("BIO200", date(2026, 9, 17)),  # left alone
    ]


def test_no_course_given_means_every_course():
    events = [make_event(course="CS101"), make_event(course="BIO200")]
    assert [e.due_date for e in move_to_weekday(events, FRIDAY)] == [date(2026, 9, 18)] * 2


def test_a_course_with_no_matching_quizzes_changes_nothing():
    events = [make_event(course="CS101")]
    assert move_to_weekday(events, FRIDAY, course="NOPE") == events


# --- courses_with_quizzes: which courses need a tutorial-day picker ---


def test_courses_with_quizzes_are_listed_once_in_order():
    events = [
        make_event(course="CS101"),
        make_event(course="BIO200", title="Quiz 2"),
        make_event(course="CS101", title="Quiz 3"),
        make_event(course="MAT135", title="Midterm", type="exam"),  # no quiz
        make_event(course="PHY100", due=None),  # an undated quiz still counts
    ]
    assert courses_with_quizzes(events) == ["BIO200", "CS101", "PHY100"]


def test_no_quizzes_means_no_courses():
    assert courses_with_quizzes([make_event(type="exam")]) == []
    assert courses_with_quizzes([]) == []


@pytest.mark.parametrize("bad_weekday", [-1, 7, 10])
def test_invalid_weekday_is_rejected(bad_weekday):
    with pytest.raises(ValueError):
        move_to_weekday([make_event()], bad_weekday)
