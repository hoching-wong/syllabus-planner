"""Dates with no year: a fall term runs into the next year.

These call the real model, so they are opt-in: `pytest -m live`. The syllabus
text is made up. Each case is also a check that the rule isn't over-applied:
a winter term must not shift, and a year written in the syllabus must be kept.

Before the rule existed, case "fall term" gave 2026-01-12 for the final exam.
"""

from datetime import date

import pytest

from planner.extractor import extract_events
from planner.llm import LLMUnavailableError

CASES = {
    "fall term with a January final": (
        2026,
        """TEST101 - Introduction to Testing - Fall 2026
Grading
Assignment 1 - due October 15 - 10%
Midterm exam - November 12 - 30%
Assignment 2 - due December 3 - 10%
Final exam - January 12 - 50%""",
        {
            "assignment 1": date(2026, 10, 15),
            "midterm": date(2026, 11, 12),
            "assignment 2": date(2026, 12, 3),
            "final": date(2027, 1, 12),  # the term runs into the new year
        },
    ),
    "winter term does not shift": (
        2027,
        """TEST102 - Testing Seminar - Winter 2027
Grading
Assignment 1 - due January 28 - 20%
Midterm exam - February 25 - 30%
Final exam - April 20 - 50%""",
        {
            "assignment 1": date(2027, 1, 28),
            "midterm": date(2027, 2, 25),
            "final": date(2027, 4, 20),
        },
    ),
    "a year written in the syllabus is kept": (
        2026,
        """TEST103 - Testing Lab - Fall 2026
Grading
Assignment 1 - due September 30 - 20%
Project - due December 8 - 30%
Final exam - January 12, 2027 - 50%""",
        {
            "assignment 1": date(2026, 9, 30),
            "project": date(2026, 12, 8),
            "final": date(2027, 1, 12),
        },
    ),
}


@pytest.mark.live
@pytest.mark.parametrize("term_year, text, expected", CASES.values(), ids=CASES.keys())
def test_dates_without_a_year_land_in_the_right_year(term_year, text, expected):
    try:
        result = extract_events(text, course="TEST", default_year=term_year)
    except LLMUnavailableError as error:
        pytest.skip(f"Gemini is overloaded, can't check this right now: {error}")

    assert result.skipped == []
    for keyword, expected_date in expected.items():
        matches = [e for e in result.events if keyword in e.title.lower()]
        assert len(matches) == 1, f"expected exactly one event matching {keyword!r}"
        assert matches[0].due_date == expected_date, keyword
