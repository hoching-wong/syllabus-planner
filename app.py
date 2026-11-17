"""Streamlit entry point. Run with: streamlit run app.py

This file only holds the screen. The real work lives in the planner/ modules.
"""

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import streamlit as st

from planner.adjust import courses_with_quizzes, move_to_weekday
from planner.display import default_course_name, duplicate_names, review_notes
from planner.editing import COLUMNS, TYPE_OPTIONS, events_to_rows, rows_to_events
from planner.extractor import ExtractionError, extract_events
from planner.llm import LLMError
from planner.models import Event
from planner.pdf_reader import PdfReadError, read_pdf

# Index 0 means "no preference"; Monday is index 1, so weekday number = index - 1.
TUTORIAL_DAYS = ["Not set", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass
class FileResult:
    """What happened with one uploaded syllabus: its events, or why it failed."""

    filename: str
    course: str
    events: list[Event] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # reasons items were dropped
    error: str | None = None


def process_file(uploaded_file, course: str, year: int) -> FileResult:
    """Read one syllabus and extract its events.

    Known failures become an error message on the result instead of crashing,
    so one bad file doesn't stop the other courses.
    """
    try:
        text = read_pdf(uploaded_file)
        result = extract_events(text, course=course, default_year=year)
    except (PdfReadError, ExtractionError, LLMError) as error:
        return FileResult(uploaded_file.name, course, error=str(error))
    return FileResult(uploaded_file.name, course, result.events, result.skipped)


# --- what the editable table starts from, kept in st.session_state ---
#
# "table_events"   the events the table is built from
# "current_events" the events as the user has edited them so far (what later steps use)
# "editor_version" part of the table's key; changing it makes Streamlit rebuild the table
#                  from "table_events", which discards the edits it was holding.
# "extracted"      the events exactly as extracted, so Reset can restore them
# "applied_days"   the tutorial day last applied to the table, per course (course -> day name)


def weekday_number(day_name: str) -> int | None:
    """'Monday' -> 0 ... 'Friday' -> 4, and 'Not set' -> None."""
    return None if day_name == "Not set" else TUTORIAL_DAYS.index(day_name) - 1


def apply_tutorial_days(events: list[Event], days: dict[str, str]) -> list[Event]:
    """Move each course's quizzes to the day picked for that course (course -> day name)."""
    for course, day_name in days.items():
        weekday = weekday_number(day_name)
        if weekday is not None:
            events = move_to_weekday(events, weekday, course)
    return events


def show_in_table(events: list[Event]) -> None:
    """Make `events` the table's contents. Used when the program, not the user, changes the data."""
    st.session_state["table_events"] = events
    st.session_state["current_events"] = events
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1


def reset_table() -> None:
    """Callback for the Reset button: back to what was extracted, every tutorial day cleared."""
    for key in list(st.session_state.keys()):
        if key.startswith("tutorial_day_"):
            st.session_state[key] = "Not set"
    st.session_state["applied_days"] = {}
    show_in_table(st.session_state["extracted"])


st.title("Syllabus Planner")
st.write("Upload your course syllabi (PDF) to build one combined timeline.")

# --- inputs ---

uploaded_files = st.file_uploader("Syllabus PDFs", type="pdf", accept_multiple_files=True)

# The uploader can't refuse a file, so we explain the problem and block Extract until it's fixed.
duplicates = duplicate_names([uploaded_file.name for uploaded_file in uploaded_files])
if duplicates:
    st.error(
        "Two uploaded files have the same name: "
        + ", ".join(duplicates)
        + ". Remove one of them, or rename it and upload it again."
    )

year = st.number_input(
    "Term year (used when a date in a syllabus has no year)",
    min_value=2000,
    max_value=2100,
    value=date.today().year,
    step=1,
)

# One editable course name per file, guessed from the file name.
# The key uses the position (i), not the file name, so two files with the same name can't clash.
course_names = []
for i, uploaded_file in enumerate(uploaded_files):
    typed = st.text_input(
        f"Course name for {uploaded_file.name}",
        value=default_course_name(uploaded_file.name),
        key=f"course_{i}",
    )
    # An empty name would make every event invalid, so fall back to the guess.
    course_names.append(typed.strip() or default_course_name(uploaded_file.name))

# --- extraction: only when the button is pressed, so uploading never uses quota ---

if st.button("Extract events", disabled=not uploaded_files or bool(duplicates)):
    results = []
    for uploaded_file, course in zip(uploaded_files, course_names):
        with st.spinner(f"Reading {uploaded_file.name} (this can take a minute if Gemini is busy)..."):
            results.append(process_file(uploaded_file, course, int(year)))
    extracted = [event for result in results for event in result.events]

    # Pickers left over from an earlier extraction keep their values, so apply them to the new events too.
    days = {
        course: st.session_state.get(f"tutorial_day_{course}", "Not set")
        for course in {event.course for event in extracted}
    }

    st.session_state["results"] = results
    st.session_state["extracted"] = extracted
    st.session_state["applied_days"] = days
    show_in_table(apply_tutorial_days(extracted, days))

# --- tutorial days: one picker per course that has quizzes ---
# This comes after extraction so that a newly extracted course gets its picker straight away.

chosen_days = {}
for course in courses_with_quizzes(st.session_state.get("current_events", [])):
    chosen_days[course] = st.sidebar.selectbox(
        f"Tutorial day for {course}",
        TUTORIAL_DAYS,
        key=f"tutorial_day_{course}",
        help="Syllabi list quizzes on one day, but your tutorial may be on another. "
        "Picking a day moves this course's quizzes to that day of their week, "
        "replacing any quiz dates you edited before. Edits you make afterwards are kept. "
        "To get the syllabus dates back, use 'Reset to extracted values'.",
    )
if not chosen_days:
    st.sidebar.caption("A tutorial-day setting appears here for each course that has quizzes.")

# The newest action wins, so a changed picker is applied to the table as it is right now.
applied_days = st.session_state.setdefault("applied_days", {})
for course, picked in chosen_days.items():
    if picked != applied_days.get(course, "Not set"):
        # Choosing "Not set" changes no dates: the originals can't be recovered
        # from the current ones. The Reset button does that.
        show_in_table(apply_tutorial_days(st.session_state["current_events"], {course: picked}))
        applied_days[course] = picked

# --- output ---

results = st.session_state.get("results", [])
for result in results:
    if result.error:
        st.error(f"{result.filename}: {result.error}")
        continue
    st.success(f"{result.course}: found {len(result.events)} events.")
    for reason in result.skipped:
        st.warning(f"{result.course}: {reason}")

if "table_events" in st.session_state:
    st.subheader("Timeline")
    st.caption(
        "Click a cell to edit it. Add a row at the bottom of the table, "
        "or select a row and press Delete to remove it."
    )
    table = pd.DataFrame(events_to_rows(st.session_state["table_events"]), columns=COLUMNS)
    edited = st.data_editor(
        table,
        key=f"editor_{st.session_state['editor_version']}",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        disabled=["Confidence"],  # the model's confidence isn't the user's to change
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Type": st.column_config.SelectboxColumn("Type", options=TYPE_OPTIONS),
            "Weight": st.column_config.NumberColumn("Weight (%)", min_value=0, max_value=100),
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
        },
    )

    # Whatever is in the table now, checked through Event again, is the user's final list.
    default_course = next((r.course for r in results if not r.error), "")
    events, problems = rows_to_events(edited.to_dict("records"), default_course)
    st.session_state["current_events"] = events

    for problem in problems:
        st.warning(problem)

    picked_days = {course: day for course, day in chosen_days.items() if day != "Not set"}
    notes = review_notes(events, picked_days)
    if notes:
        with st.expander("Things to check", expanded=True):
            st.markdown("\n".join(f"- {note}" for note in notes))

    st.button("Reset to extracted values", on_click=reset_table)
