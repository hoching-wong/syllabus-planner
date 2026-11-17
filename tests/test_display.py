from datetime import date

import pytest

from planner.display import default_course_name, review_notes
from planner.models import Event


def make_event(**overrides) -> Event:
    fields = {
        "course": "CS101",
        "title": "Midterm",
        "type": "exam",
        "due_date": date(2026, 10, 20),
        "weight_percent": 25.0,
        "confidence": 0.9,
    }
    fields.update(overrides)
    return Event(**fields)


# --- default_course_name: guess a course name from the file name ---


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("MAT235H1_Syllabus.pdf", "MAT235H1"),
        ("cs101 syllabus fall 2026.pdf", "cs101"),
        ("CSC-108-SYLLABUS.pdf", "CSC-108"),
        ("Biology.pdf", "Biology"),
        ("syllabus.pdf", "syllabus"),  # nothing left after cutting, so keep the name
    ],
)
def test_default_course_name(filename, expected):
    assert default_course_name(filename) == expected


# --- review_notes: a short list of things the user should double-check ---


def test_clean_events_need_no_notes():
    assert review_notes([make_event()]) == []


def test_no_events_need_no_notes():
    assert review_notes([]) == []


def test_missing_date_is_noted():
    (note,) = review_notes([make_event(title="Final", due_date=None)])
    assert note == "CS101 – Final: no date given"


def test_low_confidence_is_noted_below_the_threshold_only():
    notes = review_notes([make_event(title="A", confidence=0.69), make_event(title="B", confidence=0.7)])
    assert notes == ["CS101 – A: low confidence"]


def test_several_problems_on_one_event_share_a_line():
    (note,) = review_notes([make_event(due_date=None, confidence=0.2)])
    assert note == "CS101 – Midterm: no date given, low confidence"


def test_notes_follow_date_order():
    notes = review_notes(
        [
            make_event(title="Late", due_date=None),
            make_event(title="Early", confidence=0.1, due_date=date(2026, 9, 1)),
        ]
    )
    assert [n.split(":")[0] for n in notes] == ["CS101 – Early", "CS101 – Late"]


def test_dated_quizzes_get_one_general_note_not_one_each():
    quizzes = [make_event(title=f"Quiz {i}", type="quiz") for i in (1, 2, 3)]
    (note,) = review_notes(quizzes)
    assert "tutorial day" in note and "sidebar" in note


def test_quiz_note_names_the_chosen_day():
    (note,) = review_notes([make_event(type="quiz")], tutorial_day="Friday")
    assert "moved to your tutorial day (Friday)" in note
    assert "sidebar" not in note


def test_undated_quiz_gets_no_tutorial_note():
    (note,) = review_notes([make_event(type="quiz", due_date=None)])
    assert "tutorial" not in note  # only the "no date" note


def test_events_that_are_not_quizzes_get_no_tutorial_note():
    assert review_notes([make_event(type="assignment")]) == []
