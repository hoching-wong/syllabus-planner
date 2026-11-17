"""PDF -> plain text. This module knows nothing about the LLM or the UI."""

from pathlib import Path
from typing import BinaryIO

import pdfplumber
from pdfplumber.page import Page


class PdfReadError(Exception):
    """Raised when a PDF can't be opened or has no readable text.

    The message is written for the user, so the UI can show it as-is.
    """


# How far apart two letters can be before pdfplumber inserts a space. The
# default (3) glued words together on a real syllabus ("willcounttoward").
TEXT_TOLERANCE = 1.5


def _table_to_text(rows: list[list[str | None]]) -> str:
    """One line per table row, cells separated by ' | '.

    A line break inside a cell is shown as ' / ', so a calendar cell becomes
    'Sep 17 / Tutorial / Quiz 1': each entry stays next to its own date.
    """
    lines = []
    for row in rows:
        cells = [(cell or "").replace("\n", " / ") for cell in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _read_page(page: Page) -> str:
    """Text of one page. Tables are read as tables, the rest as plain text.

    Reading a calendar grid as plain text flattens it, so you can't tell which
    column an entry was in. Reading it as a table keeps each cell together.
    """
    tables = page.find_tables()

    # Cut the tables out so their text isn't included twice.
    outside_tables = page
    for table in tables:
        outside_tables = outside_tables.outside_bbox(table.bbox)

    # extract_text() can return None for a page with no text, so use "".
    parts = [outside_tables.extract_text(x_tolerance=TEXT_TOLERANCE) or ""]
    parts += [_table_to_text(table.extract(x_tolerance=TEXT_TOLERANCE)) for table in tables]
    return "\n".join(part for part in parts if part.strip())


def read_pdf(source: str | Path | BinaryIO) -> str:
    """Return all the text in a PDF, one page after another.

    `source` can be a file path or an open binary file. Streamlit's uploaded
    files count as the second kind, so this works for both the app and tests.
    """
    try:
        with pdfplumber.open(source) as pdf:
            pages = [_read_page(page) for page in pdf.pages]
    except Exception as error:
        # pdfplumber/pdfminer raise several different errors for bad files.
        # We don't care which, only that the file isn't a usable PDF.
        raise PdfReadError("Could not open this file as a PDF.") from error

    # Skip blank pages so they don't leave gaps in the text.
    text = "\n".join(page for page in pages if page.strip())

    if not text:
        raise PdfReadError(
            "No readable text found. This may be a scanned PDF (images only)."
        )
    return text
