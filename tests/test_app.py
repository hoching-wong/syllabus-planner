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
RECORDED_REPLY = json.loads((FIXTURES / "MAT235H1_recorded_reply.json").read_text(encoding="utf-8"))["reply"]


@pytest.fixture
def model_calls(monkeypatch) -> list[str]:
    """Fake the model with the recorded real reply. Returns the prompts it was sent."""
    prompts: list[str] = []

    def fake_call_llm(prompt: str) -> str:
        prompts.append(prompt)
        return RECORDED_REPLY

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


# --- loading and extracting ---


def test_page_loads_with_nothing_to_do(model_calls):
    at = new_app()
    assert not at.exception
    assert at.title[0].value == "Syllabus Planner"
    assert at.button[0].disabled  # nothing uploaded yet
    assert not at.data_editor


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

    at.sidebar.selectbox[0].select("Friday").run()
    assert not at.exception
    assert table(at)["Date"].iloc[0] == date(2026, 9, 18)  # Thursday Sep 17 -> Friday Sep 18
    assert "moved to your tutorial day (Friday)" in page_text(at)
    assert len(model_calls) == 1  # the sidebar change reused the stored results


def test_exams_do_not_move_with_the_tutorial_day(model_calls):
    at = extracted_app(model_calls)
    at.sidebar.selectbox[0].select("Monday").run()
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

    at.sidebar.selectbox[0].select("Friday").run()  # newer action: the tutorial day wins
    assert event(at, "Quiz 1").due_date == date(2026, 9, 18)
    assert event(at, "Term Test 1").weight_percent == 30.0  # not a quiz, so the edit survives


def test_a_quiz_edit_after_picking_a_tutorial_day_is_kept(model_calls):
    at = extracted_app(model_calls)
    at.sidebar.selectbox[0].select("Friday").run()
    at = edit_table(at, edited_rows={row_of(at, "Quiz 1"): {"Date": "2026-09-16"}})  # newer action
    assert event(at, "Quiz 1").due_date == date(2026, 9, 16)
    assert event(at, "Quiz 2").due_date == date(2026, 9, 25)  # the others are still on Friday


def test_choosing_not_set_does_not_change_dates(model_calls):
    at = extracted_app(model_calls)
    at.sidebar.selectbox[0].select("Friday").run()
    at.sidebar.selectbox[0].select("Not set").run()
    assert event(at, "Quiz 1").due_date == date(2026, 9, 18)  # only Reset undoes the move


# --- reset ---


def test_reset_restores_the_extracted_events_and_clears_the_tutorial_day(model_calls):
    at = extracted_app(model_calls)
    at = edit_table(
        at,
        edited_rows={row_of(at, "Term Test 1"): {"Weight": 30.0}},
        deleted_rows=[row_of(at, "Quiz 8")],
    )
    at.sidebar.selectbox[0].select("Friday").run()
    assert len(at.session_state["current_events"]) == 10

    at = click(at, "Reset to extracted values")
    assert not at.exception
    assert at.sidebar.selectbox[0].value == "Not set"
    assert len(at.session_state["current_events"]) == 11
    assert event(at, "Term Test 1").weight_percent == 22.5
    assert event(at, "Quiz 1").due_date == date(2026, 9, 17)  # the syllabus's own date again
    assert len(model_calls) == 1  # no new model call
