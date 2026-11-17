from datetime import date

import pytest

from planner.display import (
    chart_rows,
    crunch_warnings,
    default_course_name,
    duplicate_names,
    review_notes,
    unscheduled_note,
)
from planner.models import Event
from planner.workload import compute_workload


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


def test_dated_quizzes_get_one_note_per_course_not_one_each():
    quizzes = [make_event(title=f"Quiz {i}", type="quiz") for i in (1, 2, 3)]
    (note,) = review_notes(quizzes)
    assert note.startswith("CS101:") and "tutorial day" in note and "sidebar" in note


def test_quiz_note_names_the_chosen_day():
    (note,) = review_notes([make_event(type="quiz")], tutorial_days={"CS101": "Friday"})
    assert note == "CS101: quiz dates were moved to your tutorial day (Friday)."


def test_each_course_gets_its_own_quiz_note():
    events = [make_event(course="CS101", type="quiz"), make_event(course="BIO200", type="quiz")]
    notes = review_notes(events, tutorial_days={"CS101": "Monday"})
    assert notes == [
        "BIO200: quiz dates are the syllabus's nominal day. "
        "Set your tutorial day in the sidebar if yours is different.",
        "CS101: quiz dates were moved to your tutorial day (Monday).",
    ]


# --- the workload view: warnings, a note about undated weight, and chart data ---
# Oct 19, 2026 is a Monday; Nov 2 is the Monday two weeks later.


def load_of(*events: Event, threshold: float = 25):
    return compute_workload(list(events), threshold)


def test_no_crunch_weeks_means_no_warnings():
    assert crunch_warnings(load_of(make_event(weight_percent=10.0))) == []


def test_a_crunch_week_is_described_with_its_courses_biggest_first():
    workload = load_of(
        make_event(course="MAT235H1", weight_percent=22.5, due_date=date(2026, 10, 19)),
        make_event(course="GGR124", weight_percent=25.0, due_date=date(2026, 10, 21), title="Test"),
    )
    assert crunch_warnings(workload) == [
        "Week of Oct 19, 2026: 47.5% of your grades are due (GGR124 25%, MAT235H1 22.5%)."
    ]


def test_only_crunch_weeks_are_listed_in_date_order():
    workload = load_of(
        make_event(weight_percent=30.0, due_date=date(2026, 11, 2)),
        make_event(weight_percent=5.0, due_date=date(2026, 10, 26)),
        make_event(weight_percent=26.0, due_date=date(2026, 10, 19)),
    )
    warnings = crunch_warnings(workload)
    assert [w.split(":")[0] for w in warnings] == ["Week of Oct 19, 2026", "Week of Nov 2, 2026"]


def test_float_noise_is_hidden_in_the_totals():
    # 10.1 + 20.2 is 30.299999999999997 in floating point; it should read as 30.3%.
    workload = load_of(make_event(weight_percent=10.1), make_event(weight_percent=20.2, title="B"))
    assert crunch_warnings(workload) == ["Week of Oct 19, 2026: 30.3% of your grades are due (CS101 30.3%)."]


def test_courses_with_no_weight_due_are_not_listed():
    workload = load_of(
        make_event(course="CS101", weight_percent=30.0),
        make_event(course="BIO200", weight_percent=None, title="Unweighted"),
    )
    assert crunch_warnings(workload) == ["Week of Oct 19, 2026: 30% of your grades are due (CS101 30%)."]


def test_no_note_when_every_event_has_a_date():
    assert unscheduled_note(load_of(make_event())) is None


def test_note_says_how_much_weight_has_no_date():
    workload = load_of(make_event(title="Final", due_date=None, weight_percent=40.0))
    assert unscheduled_note(workload) == (
        "40% of your grades (1 event) has no date or is ongoing, so it isn't in the weekly view."
    )


def test_note_counts_several_undated_events():
    workload = load_of(
        make_event(title="A", due_date=None, weight_percent=20.0),
        make_event(title="B", due_date=None, weight_percent=12.5),
    )
    assert "32.5% of your grades (2 events)" in unscheduled_note(workload)


def test_undated_events_without_weights_give_no_note():
    assert unscheduled_note(load_of(make_event(due_date=None, weight_percent=None))) is None


def test_chart_rows_split_crunch_weeks_from_the_others():
    workload = load_of(
        make_event(weight_percent=30.0, due_date=date(2026, 10, 19)),
        make_event(weight_percent=10.0, due_date=date(2026, 11, 2)),
    )
    assert chart_rows(workload) == [
        {"Week": "2026-10-19", "Crunch weeks": 30.0, "Other weeks": 0.0},
        {"Week": "2026-10-26", "Crunch weeks": 0.0, "Other weeks": 0.0},  # a quiet week in between
        {"Week": "2026-11-02", "Crunch weeks": 0.0, "Other weeks": 10.0},
    ]


def test_chart_rows_are_empty_without_weeks():
    assert chart_rows(load_of()) == []


# --- duplicate_names: stop two uploads from having the same file name ---


def test_unique_names_have_no_duplicates():
    assert duplicate_names(["a.pdf", "b.pdf"]) == []
    assert duplicate_names([]) == []


def test_repeated_name_is_reported_once():
    assert duplicate_names(["a.pdf", "b.pdf", "a.pdf", "a.pdf"]) == ["a.pdf"]


def test_several_duplicates_are_all_reported():
    assert duplicate_names(["a.pdf", "b.pdf", "b.pdf", "a.pdf"]) == ["b.pdf", "a.pdf"]


def test_names_that_differ_only_by_case_are_different():
    assert duplicate_names(["a.pdf", "A.pdf"]) == []


def test_undated_quiz_gets_no_tutorial_note():
    (note,) = review_notes([make_event(type="quiz", due_date=None)])
    assert "tutorial" not in note  # only the "no date" note


def test_events_that_are_not_quizzes_get_no_tutorial_note():
    assert review_notes([make_event(type="assignment")]) == []
