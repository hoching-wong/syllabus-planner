"""The Event model: the one data shape shared by every other module.

Design rule: the LLM's output is never trusted. Optional details (date, weight)
that are bad become None so we keep the event. Things we cannot do without
(course, title, confidence) raise a ValidationError so the caller can skip it.
"""

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(str, Enum):
    ASSIGNMENT = "assignment"
    EXAM = "exam"
    QUIZ = "quiz"
    PROJECT = "project"
    OTHER = "other"


class Event(BaseModel):
    # Strip spaces around strings, so a title of "   " counts as empty.
    model_config = ConfigDict(str_strip_whitespace=True)

    course: str = Field(min_length=1)
    title: str = Field(min_length=1)
    type: EventType
    due_date: date | None = None
    weight_percent: float | None = None
    confidence: float = Field(ge=0, le=1)

    # mode="before" means these run on the raw input, before pydantic's own
    # type conversion. That lets us turn messy values into clean ones first.

    @field_validator("type", mode="before")
    @classmethod
    def unknown_type_becomes_other(cls, value: Any) -> str:
        """'EXAM' -> 'exam'; anything we don't recognise -> 'other'."""
        label = str(value).strip().lower()
        try:
            return EventType(label).value  # raises ValueError if not a known type
        except ValueError:
            return EventType.OTHER.value

    @field_validator("due_date", mode="before")
    @classmethod
    def bad_date_becomes_none(cls, value: Any) -> date | None:
        """Accept a date or an ISO string like '2026-10-20'; otherwise None."""
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value.strip())
            except ValueError:  # "TBA", "", "2026-13-45", ...
                return None
        return None

    @field_validator("weight_percent", mode="before")
    @classmethod
    def clean_weight(cls, value: Any) -> float | None:
        """Turn '30%' or 30 into 30.0. Unreadable or outside 0-100 -> None."""
        if isinstance(value, str):
            value = value.replace("%", "").strip()
        try:
            weight = float(value)
        except (TypeError, ValueError):  # None, "", "lots", ...
            return None
        # A wrong weight is worse than an unknown one, so out-of-range -> None.
        # (This comparison is also False for NaN, so NaN becomes None too.)
        if 0 <= weight <= 100:
            return weight
        return None
