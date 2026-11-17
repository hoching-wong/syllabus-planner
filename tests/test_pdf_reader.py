import io

import pytest

from planner.pdf_reader import PdfReadError, read_pdf


def make_pdf(page_texts: list[str]) -> bytes:
    """Build a tiny valid PDF with one line of text per page ("" = blank page).

    We build it by hand so the tests need no extra PDF-writing library and no
    fixture file. Layout: object 1 = catalog, 2 = page list, 3 = font, then
    each page uses two objects (the page, then its content stream).
    """
    count = len(page_texts)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(page_texts):
        contents_number = 5 + 2 * i
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {contents_number} 0 R >>"
            ).encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )

    # Write the objects, remembering where each starts for the xref table.
    pdf = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_start = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode()
    return pdf


def test_reads_text_from_a_file_object():
    pdf = io.BytesIO(make_pdf(["Midterm on October 20"]))
    assert read_pdf(pdf) == "Midterm on October 20"


def test_reads_text_from_a_file_path(tmp_path):
    path = tmp_path / "syllabus.pdf"
    path.write_bytes(make_pdf(["Final exam in December"]))
    assert read_pdf(path) == "Final exam in December"


def test_pages_are_joined_in_order():
    pdf = io.BytesIO(make_pdf(["Page one", "Page two", "Page three"]))
    assert read_pdf(pdf) == "Page one\nPage two\nPage three"


def test_blank_pages_are_skipped():
    pdf = io.BytesIO(make_pdf(["First", "", "Third"]))
    assert read_pdf(pdf) == "First\nThird"


def test_pdf_with_no_text_raises_a_helpful_error():
    # Same situation as a scanned syllabus: pages exist but contain no text.
    pdf = io.BytesIO(make_pdf(["", ""]))
    with pytest.raises(PdfReadError, match="scanned"):
        read_pdf(pdf)


def test_file_that_is_not_a_pdf_raises():
    with pytest.raises(PdfReadError, match="Could not open"):
        read_pdf(io.BytesIO(b"this is just plain text, not a pdf"))
