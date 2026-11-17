from datetime import date

import pytest

from planner.adjust import move_to_weekday
from planner.models import Event

MONDAY, THURSDAY, FRIDAY = 0, 3, 4  # Python's weekday numbers: Monday is 0


def make_event(title="Quiz 1", type="quiz", due=date(2026, 9, 17)) -> Event:
    """Sep 17, 2026 is a Thursday."""
    return Event(
        course="MAT235H1", title=title, type=type, due_date=due, weight_percent=2.5, confidence=1
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


@pytest.mark.parametrize("bad_weekday", [-1, 7, 10])
def test_invalid_weekday_is_rejected(bad_weekday):
    with pytest.raises(ValueError):
        move_to_weekday([make_event()], bad_weekday)
