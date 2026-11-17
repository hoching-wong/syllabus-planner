"""PDF -> plain text. This module knows nothing about the LLM or the UI."""

from pathlib import Path
from typing import BinaryIO

import pdfplumber


class PdfReadError(Exception):
    """Raised when a PDF can't be opened or has no readable text.

    The message is written for the user, so the UI can show it as-is.
    """


def read_pdf(source: str | Path | BinaryIO) -> str:
    """Return all the text in a PDF, one page after another.

    `source` can be a file path or an open binary file. Streamlit's uploaded
    files count as the second kind, so this works for both the app and tests.
    """
    try:
        with pdfplumber.open(source) as pdf:
            # extract_text() can return None for a page with no text, so use "".
            pages = [page.extract_text() or "" for page in pdf.pages]
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
