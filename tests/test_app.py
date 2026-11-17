"""Drive the real Streamlit page with Streamlit's test runner.

PDF reading is real (we upload the MAT235 fixture). Only the model is faked,
so nothing here spends API quota.

The test runner can't click table cells, but it can set the table's saved edits
(Streamlit keeps them in session state), so edit_table() simulates a user editing,
adding or deleting rows.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from planner import extractor
from planner.llm import LLMError

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
PDF_BYTES = (FIXTURES / "MAT235H1_Syllabus.pdf").read_bytes()

# A fixed fake model reply, written out here so these UI tests never change when
# the real model's wording changes (the accuracy tests cover the real model's output).
QUIZ_DATES = ["2026-09-17", "2026-09-24", "2026-10-01", "2026-10-08", "2026-11-05", "2026-11-12", "2026-11-19", "2026-12-03"]
CANNED_REPLY = json.dumps(
    {
        "events": [
            {"title": f"Quiz {n}", "type": "quiz", "due_date": day, "weight_percent": 2.5, "confidence": 1}
            for n, day in enumerate(QUIZ_DATES, start=1)
        ]
        + [
            {"title": "Term Test 1", "type": "exam", "due_date": "2026-10-19", "weight_percent": 22.5, "confidence": 1},
            {"title": "Term Test 2", "type": "exam", "due_date": "2026-11-27", "weight_percent": 22.5, "confidence": 1},
            {"title": "Final Exam", "type": "exam", "due_date": None, "weight_percent": 40, "confidence": 1},
        ]
    }
)


@pytest.fixture
def model_calls(monkeypatch) -> list[str]:
    """Fake the model with the canned reply. Returns the prompts it was sent."""
    prompts: list[str] = []

    def fake_call_llm(prompt: str) -> str:
        prompts.append(prompt)
        return CANNED_REPLY

    monkeypatch.setattr(extractor, "call_llm", fake_call_llm)
    return prompts


def new_app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()


def upload(at: AppTest, *files: tuple[str, bytes]) -> AppTest:
    return at.file_uploader[0].set_value([(name, data, "application/pdf") for name, data in files]).run()


def click(at: AppTest, label: str) -> AppTest:
    (button,) = [b for b in at.button if b.label == label]
    return button.click().run()


def extract(at: AppTest) -> AppTest:
    return click(at, "Extract events")


def extracted_app(model_calls) -> AppTest:
    return extract(upload(new_app(), ("MAT235H1_Syllabus.pdf", PDF_BYTES)))


def table(at: AppTest) -> pd.DataFrame:
    """The table as it is on screen (after any edits)."""
    return at.data_editor[0].value


def row_of(at: AppTest, title: str) -> int:
    return table(at)["Title"].tolist().index(title)


def edit_table(at: AppTest, edited_rows=None, added_rows=None, deleted_rows=None) -> AppTest:
    """Simulate the user changing the table. Row numbers are positions on screen."""
    key = f"editor_{at.session_state['editor_version']}"
    at.session_state[key] = {
        "edited_rows": edited_rows or {},
        "added_rows": added_rows or [],
        "deleted_rows": deleted_rows or [],
    }
    return at.run()


def event(at: AppTest, title: str):
    """The user's current (edited) event with this title."""
    (found,) = [e for e in at.session_state["current_events"] if e.title == title]
    return found


def page_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def picker(at: AppTest, course: str):
    """The sidebar's tutorial-day dropdown for one course."""
    return at.sidebar.selectbox(key=f"tutorial_day_{course}")


def two_course_app(model_calls) -> AppTest:
    """Two syllabi that produce the same events, under different course names."""
    at = upload(new_app(), ("MAT235H1_Syllabus.pdf", PDF_BYTES), ("OTHER_Syllabus.pdf", PDF_BYTES))
    return extract(at)


def quiz_date(at: AppTest, course: str, title: str = "Quiz 1") -> date:
    (found,) = [e for e in at.session_state["current_events"] if e.course == course and e.title == title]
    return found.due_date


# --- loading and extracting ---


def test_page_loads_with_nothing_to_do(model_calls):
    at = new_app()
    assert not at.exception
    assert at.title[0].value == "Syllabus Planner"
    assert at.button[0].disabled  # nothing uploaded yet
    assert not at.data_editor
    assert not at.sidebar.selectbox  # tutorial-day pickers only appear once quizzes exist


def test_page_warns_that_the_model_can_make_mistakes(model_calls):
    at = new_app()
    text = " ".join(c.value for c in at.caption)
    assert "Gemini" in text and "mistakes" in text
    assert "double-check" in text


def test_two_files_with_the_same_name_are_rejected(model_calls):
    at = upload(new_app(), ("syllabus.pdf", PDF_BYTES), ("syllabus.pdf", PDF_BYTES))
    assert not at.exception
    assert any("same name" in e.value and "syllabus.pdf" in e.value for e in at.error)
    assert at.button[0].disabled  # can't extract until the duplicate is removed
    assert model_calls == []


def test_removing_the_duplicate_clears_the_error(model_calls):
    at = upload(new_app(), ("syllabus.pdf", PDF_BYTES), ("syllabus.pdf", PDF_BYTES))
    at = upload(at, ("syllabus.pdf", PDF_BYTES))
    assert not at.error
    assert not at.button[0].disabled


def test_files_with_different_names_are_fine(model_calls):
    at = upload(new_app(), ("a_syllabus.pdf", PDF_BYTES), ("b_syllabus.pdf", PDF_BYTES))
    assert not at.error
    assert not at.button[0].disabled


def test_uploading_alone_does_not_call_the_model(model_calls):
    at = upload(new_app(), ("MAT235H1_Syllabus.pdf", PDF_BYTES))
    assert not at.exception
    assert not at.button[0].disabled
    assert model_calls == []  # no quota used until the button is pressed
    assert not at.data_editor


def test_course_name_is_guessed_from_the_file_name(model_calls):
    at = upload(new_app(), ("MAT235H1_Syllabus.pdf", PDF_BYTES))
    assert at.text_input[0].value == "MAT235H1"


def test_extracting_shows_a_sorted_timeline(model_calls):
    at = extracted_app(model_calls)
    assert not at.exception
    assert len(model_calls) == 1
    assert "found 11 events" in at.success[0].value

    rows = table(at)
    assert len(rows) == 11
    assert rows["Date"].iloc[0] == date(2026, 9, 17)  # Quiz 1 comes first
    assert rows["Title"].iloc[-1] == "Final Exam"  # no date, so it goes last
    assert pd.isna(rows["Date"].iloc[-1])
    assert set(rows["Course"]) == {"MAT235H1"}


def test_edited_course_name_is_used(model_calls):
    at = upload(new_app(), ("MAT235H1_Syllabus.pdf", PDF_BYTES))
    at.text_input[0].set_value("MAT 235").run()
    at = extract(at)
    assert set(table(at)["Course"]) == {"MAT 235"}


def test_two_syllabi_are_merged_into_one_timeline(model_calls):
    at = upload(new_app(), ("MAT235H1_Syllabus.pdf", PDF_BYTES), ("OTHER_Syllabus.pdf", PDF_BYTES))
    at = extract(at)
    assert len(model_calls) == 2
    assert len(table(at)) == 22
    assert set(table(at)["Course"]) == {"MAT235H1", "OTHER"}


def test_one_bad_file_does_not_stop_the_others(model_calls):
    at = upload(new_app(), ("notes.pdf", b"not a pdf at all"), ("MAT235H1_Syllabus.pdf", PDF_BYTES))
    at = extract(at)
    assert not at.exception
    assert "notes.pdf" in at.error[0].value and "Could not open" in at.error[0].value
    assert len(table(at)) == 11  # the good file still produced its events


def test_model_failure_is_shown_as_an_error_not_a_crash(monkeypatch):
    def broken(prompt: str) -> str:
        raise LLMError("Gemini is down")

    monkeypatch.setattr(extractor, "call_llm", broken)
    at = extract(upload(new_app(), ("MAT235H1_Syllabus.pdf", PDF_BYTES)))
    assert not at.exception
    assert "Gemini is down" in at.error[0].value
    # An empty table is still shown, so the user can type the events in by hand.
    assert len(table(at)) == 0


def test_skipped_items_are_shown_as_warnings(monkeypatch):
    reply = json.dumps({"events": [
        {"title": "Quiz 1", "type": "quiz", "due_date": "2026-09-17", "weight_percent": 2.5, "confidence": 1},
        {"title": "", "type": "exam", "due_date": "2026-10-19", "weight_percent": 20, "confidence": 1},
    ]})
    monkeypatch.setattr(extractor, "call_llm", lambda prompt: reply)
    at = extract(upload(new_app(), ("MAT235H1_Syllabus.pdf", PDF_BYTES)))
    assert not at.exception
    assert "Item 2 skipped" in at.warning[0].value
    assert len(table(at)) == 1


# --- review notes ---


def test_notes_say_what_to_check(model_calls):
    text = page_text(extracted_app(model_calls))
    assert "Final Exam: no date given" in text
    assert "nominal day" in text  # the quiz-date note


# --- editing the table ---


def test_unedited_table_gives_the_extracted_events(model_calls):
    at = extracted_app(model_calls)
    assert len(at.session_state["current_events"]) == 11


def test_edited_date_and_weight_are_used(model_calls):
    at = extracted_app(model_calls)
    row = row_of(at, "Term Test 1")
    at = edit_table(at, edited_rows={row: {"Date": "2026-10-26", "Weight": 30.0}})
    assert not at.exception
    test1 = event(at, "Term Test 1")
    assert (test1.due_date, test1.weight_percent) == (date(2026, 10, 26), 30.0)
    assert event(at, "Term Test 2").weight_percent == 22.5  # other rows untouched


def test_a_missing_date_can_be_filled_in(model_calls):
    at = extracted_app(model_calls)
    at = edit_table(at, edited_rows={row_of(at, "Final Exam"): {"Date": "2026-12-15"}})
    assert event(at, "Final Exam").due_date == date(2026, 12, 15)
    assert "Final Exam: no date given" not in page_text(at)  # the note is gone


def test_a_row_can_be_deleted(model_calls):
    at = extracted_app(model_calls)
    at = edit_table(at, deleted_rows=[row_of(at, "Quiz 8")])
    titles = [e.title for e in at.session_state["current_events"]]
    assert len(titles) == 10 and "Quiz 8" not in titles


def test_a_row_can_be_added_and_gets_defaults(model_calls):
    at = extracted_app(model_calls)
    at = edit_table(at, added_rows=[{"Title": "Study group", "Date": "2026-10-30"}])
    assert not at.exception
    added = event(at, "Study group")
    assert added.course == "MAT235H1"  # taken from the uploaded course
    assert added.type.value == "other"
    assert added.confidence == 1.0
    assert added.due_date == date(2026, 10, 30)
    assert len(at.session_state["current_events"]) == 12


def test_an_unusable_row_is_ignored_with_a_warning(model_calls):
    at = extracted_app(model_calls)
    at = edit_table(at, added_rows=[{"Date": "2026-10-30"}])  # no title
    assert not at.exception
    assert len(at.session_state["current_events"]) == 11
    assert any("Row 12 ignored" in w.value and "title" in w.value for w in at.warning)


# --- tutorial day, and "the newest action wins" ---


def test_tutorial_day_moves_quizzes_without_calling_the_model_again(model_calls):
    at = extracted_app(model_calls)
    assert table(at)["Date"].iloc[0] == date(2026, 9, 17)

    picker(at, "MAT235H1").select("Friday").run()
    assert not at.exception
    assert table(at)["Date"].iloc[0] == date(2026, 9, 18)  # Thursday Sep 17 -> Friday Sep 18
    assert "moved to your tutorial day (Friday)" in page_text(at)
    assert len(model_calls) == 1  # the sidebar change reused the stored results


def test_exams_do_not_move_with_the_tutorial_day(model_calls):
    at = extracted_app(model_calls)
    picker(at, "MAT235H1").select("Monday").run()
    dates = table(at)["Date"].tolist()
    assert date(2026, 10, 19) in dates and date(2026, 11, 27) in dates


def test_picking_a_tutorial_day_overwrites_earlier_quiz_edits_but_keeps_other_edits(model_calls):
    at = extracted_app(model_calls)
    at = edit_table(
        at,
        edited_rows={
            row_of(at, "Quiz 1"): {"Date": "2026-09-14"},  # a hand edit to a quiz date
            row_of(at, "Term Test 1"): {"Weight": 30.0},  # a hand edit to an exam
        },
    )
    assert event(at, "Quiz 1").due_date == date(2026, 9, 14)

    picker(at, "MAT235H1").select("Friday").run()  # newer action: the tutorial day wins
    assert event(at, "Quiz 1").due_date == date(2026, 9, 18)
    assert event(at, "Term Test 1").weight_percent == 30.0  # not a quiz, so the edit survives


def test_a_quiz_edit_after_picking_a_tutorial_day_is_kept(model_calls):
    at = extracted_app(model_calls)
    picker(at, "MAT235H1").select("Friday").run()
    at = edit_table(at, edited_rows={row_of(at, "Quiz 1"): {"Date": "2026-09-16"}})  # newer action
    assert event(at, "Quiz 1").due_date == date(2026, 9, 16)
    assert event(at, "Quiz 2").due_date == date(2026, 9, 25)  # the others are still on Friday


def test_choosing_not_set_does_not_change_dates(model_calls):
    at = extracted_app(model_calls)
    picker(at, "MAT235H1").select("Friday").run()
    picker(at, "MAT235H1").select("Not set").run()
    assert event(at, "Quiz 1").due_date == date(2026, 9, 18)  # only Reset undoes the move


# --- the weekly workload and crunch weeks ---
# With the canned reply, Term Test 1 (22.5%) is due the week of Mon Oct 19 and
# Term Test 2 (22.5%) the week of Mon Nov 23: busy, but under the default 25%.


def crunch_lines(at: AppTest) -> list[str]:
    return [w.value for w in at.warning if w.value.startswith("Week of")]


def charts(at: AppTest):
    return at.get("vega_lite_chart")


def captions(at: AppTest) -> str:
    return " ".join(c.value for c in at.caption)


def test_no_workload_before_extracting(model_calls):
    at = new_app()
    assert not charts(at)
    assert at.sidebar.slider[0].value == 25  # the default threshold is already there


def test_workload_chart_appears_after_extracting(model_calls):
    at = extracted_app(model_calls)
    assert not at.exception
    assert len(charts(at)) == 1
    assert "Workload by week" in [s.value for s in at.subheader]


def test_no_warning_when_every_week_is_under_the_threshold(model_calls):
    assert crunch_lines(extracted_app(model_calls)) == []


def test_lowering_the_threshold_flags_the_busy_weeks(model_calls):
    at = extracted_app(model_calls)
    at.sidebar.slider[0].set_value(20).run()
    assert not at.exception
    assert crunch_lines(at) == [
        "Week of Oct 19, 2026: 22.5% of your grades are due (MAT235H1 22.5%).",
        "Week of Nov 23, 2026: 22.5% of your grades are due (MAT235H1 22.5%).",
    ]
    assert len(model_calls) == 1  # moving the slider doesn't re-extract


def test_undated_weight_is_mentioned(model_calls):
    # The final exam (40%) has no date, so it can't be placed in a week.
    assert "40% of your grades (1 event) has no date" in captions(extracted_app(model_calls))


def test_adding_an_event_to_a_busy_week_creates_a_crunch_week(model_calls):
    at = extracted_app(model_calls)
    at = edit_table(at, added_rows=[{"Title": "Essay", "Date": "2026-10-21", "Weight": 10.0}])
    assert crunch_lines(at) == ["Week of Oct 19, 2026: 32.5% of your grades are due (MAT235H1 32.5%)."]


def test_editing_a_weight_changes_the_crunch_weeks(model_calls):
    at = extracted_app(model_calls)
    at = edit_table(at, edited_rows={row_of(at, "Term Test 2"): {"Weight": 30.0}})
    assert crunch_lines(at) == ["Week of Nov 23, 2026: 30% of your grades are due (MAT235H1 30%)."]


def test_moving_an_event_to_another_week_updates_the_warning(model_calls):
    at = extracted_app(model_calls)
    at.sidebar.slider[0].set_value(20).run()
    at = edit_table(at, edited_rows={row_of(at, "Term Test 1"): {"Date": "2026-10-27"}})  # a Tuesday
    assert crunch_lines(at)[0].startswith("Week of Oct 26, 2026")


def test_deleting_an_event_removes_its_warning(model_calls):
    at = extracted_app(model_calls)
    at.sidebar.slider[0].set_value(20).run()
    at = edit_table(at, deleted_rows=[row_of(at, "Term Test 1")])
    assert all("Oct 19" not in line for line in crunch_lines(at))
    assert any("Nov 23" in line for line in crunch_lines(at))


def test_two_courses_in_the_same_week_add_up(model_calls):
    at = two_course_app(model_calls)  # both courses have a 22.5% test that week
    assert crunch_lines(at) == [
        "Week of Oct 19, 2026: 45% of your grades are due (MAT235H1 22.5%, OTHER 22.5%).",
        "Week of Nov 23, 2026: 45% of your grades are due (MAT235H1 22.5%, OTHER 22.5%).",
    ]


def test_only_undated_events_still_mention_the_missing_dates(monkeypatch):
    reply = json.dumps({"events": [
        {"title": "Participation", "type": "other", "due_date": None, "weight_percent": 20, "confidence": 1},
    ]})
    monkeypatch.setattr(extractor, "call_llm", lambda prompt: reply)
    at = extract(upload(new_app(), ("MAT235H1_Syllabus.pdf", PDF_BYTES)))
    assert not at.exception
    assert not charts(at)  # no week to draw
    assert "20% of your grades (1 event) has no date" in captions(at)


# --- a separate tutorial day for each course ---


def test_each_course_has_its_own_picker(model_calls):
    at = two_course_app(model_calls)
    assert [s.label for s in at.sidebar.selectbox] == ["Tutorial day for MAT235H1", "Tutorial day for OTHER"]


def test_a_course_without_quizzes_gets_no_picker(monkeypatch):
    reply = json.dumps({"events": [
        {"title": "Midterm", "type": "exam", "due_date": "2026-10-19", "weight_percent": 30, "confidence": 1},
    ]})
    monkeypatch.setattr(extractor, "call_llm", lambda prompt: reply)
    at = extract(upload(new_app(), ("MAT235H1_Syllabus.pdf", PDF_BYTES)))
    assert not at.exception
    assert not at.sidebar.selectbox


def test_a_tutorial_day_only_moves_that_courses_quizzes(model_calls):
    at = two_course_app(model_calls)
    picker(at, "MAT235H1").select("Friday").run()
    assert not at.exception
    assert quiz_date(at, "MAT235H1") == date(2026, 9, 18)
    assert quiz_date(at, "OTHER") == date(2026, 9, 17)  # the other course is untouched


def test_different_courses_can_have_different_days(model_calls):
    at = two_course_app(model_calls)
    picker(at, "MAT235H1").select("Friday").run()
    picker(at, "OTHER").select("Monday").run()
    assert quiz_date(at, "MAT235H1") == date(2026, 9, 18)
    assert quiz_date(at, "OTHER") == date(2026, 9, 14)


def test_picking_a_day_for_one_course_keeps_edits_to_another(model_calls):
    at = two_course_app(model_calls)
    rows = table(at)
    other_quiz = int(rows.index[(rows["Course"] == "OTHER") & (rows["Title"] == "Quiz 1")][0])
    at = edit_table(at, edited_rows={other_quiz: {"Date": "2026-09-15"}})

    picker(at, "MAT235H1").select("Friday").run()  # newest action, but for a different course
    assert quiz_date(at, "OTHER") == date(2026, 9, 15)  # the edit survives
    assert quiz_date(at, "MAT235H1") == date(2026, 9, 18)


def test_quiz_notes_are_per_course(model_calls):
    at = two_course_app(model_calls)
    picker(at, "MAT235H1").select("Friday").run()
    text = page_text(at)
    assert "MAT235H1: quiz dates were moved to your tutorial day (Friday)." in text
    assert "OTHER: quiz dates are the syllabus's nominal day" in text


def test_chosen_days_still_apply_when_extracting_again(model_calls):
    at = two_course_app(model_calls)
    picker(at, "MAT235H1").select("Friday").run()
    at = extract(at)  # extract a second time with the pickers already set
    assert quiz_date(at, "MAT235H1") == date(2026, 9, 18)
    assert quiz_date(at, "OTHER") == date(2026, 9, 17)


def test_reset_clears_every_picker(model_calls):
    at = two_course_app(model_calls)
    picker(at, "MAT235H1").select("Friday").run()
    picker(at, "OTHER").select("Monday").run()
    at = click(at, "Reset to extracted values")
    assert picker(at, "MAT235H1").value == "Not set"
    assert picker(at, "OTHER").value == "Not set"
    assert quiz_date(at, "MAT235H1") == date(2026, 9, 17)
    assert quiz_date(at, "OTHER") == date(2026, 9, 17)


# --- reset ---


def test_reset_restores_the_extracted_events_and_clears_the_tutorial_day(model_calls):
    at = extracted_app(model_calls)
    at = edit_table(
        at,
        edited_rows={row_of(at, "Term Test 1"): {"Weight": 30.0}},
        deleted_rows=[row_of(at, "Quiz 8")],
    )
    picker(at, "MAT235H1").select("Friday").run()
    assert len(at.session_state["current_events"]) == 10

    at = click(at, "Reset to extracted values")
    assert not at.exception
    assert picker(at, "MAT235H1").value == "Not set"
    assert len(at.session_state["current_events"]) == 11
    assert event(at, "Term Test 1").weight_percent == 22.5
    assert event(at, "Quiz 1").due_date == date(2026, 9, 17)  # the syllabus's own date again
    assert len(model_calls) == 1  # no new model call


# --- calendar download ---


def download_buttons(at: AppTest):
    return at.get("download_button")


def test_no_download_button_before_extracting(model_calls):
    assert not download_buttons(new_app())


def test_download_button_appears_after_extracting(model_calls):
    at = extracted_app(model_calls)
    (button,) = download_buttons(at)
    assert button.proto.label == "Download calendar (.ics)"
    assert not button.proto.disabled
    assert button.proto.url.endswith(".ics")


def test_undated_events_are_reported_as_not_exported(model_calls):
    at = extracted_app(model_calls)  # the Final Exam has no date
    assert "1 event(s) have no date, so they won't be in the calendar." in captions(at)


def test_giving_the_undated_event_a_date_removes_the_note(model_calls):
    at = extracted_app(model_calls)
    at = edit_table(at, edited_rows={row_of(at, "Final Exam"): {"Date": "2026-12-15"}})
    assert "won't be in the calendar" not in captions(at)


def test_download_is_disabled_when_no_event_has_a_date(model_calls):
    at = extracted_app(model_calls)
    dates = {row: {"Date": None} for row in range(len(table(at)))}
    at = edit_table(at, edited_rows=dates)
    (button,) = download_buttons(at)
    assert button.proto.disabled


def test_checklist_heading_says_to_check_before_exporting(model_calls):
    at = extracted_app(model_calls)
    assert "Things to check before exporting" in [e.label for e in at.expander]
