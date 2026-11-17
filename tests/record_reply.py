"""Save the model's reply for the fixture syllabi so tests can replay them offline.

Run from the project root:
    python -m tests.record_reply              (records every fixture)
    python -m tests.record_reply EAS110Y1     (records just one)

It goes through call_llm, so it only spends API quota if the prompt (or the
PDF text) has changed since the last real call; otherwise the disk cache answers.
Re-run it after changing the prompt or the PDF reader, then check the accuracy tests.
"""

import hashlib
import json
import sys
from pathlib import Path

from planner.extractor import build_prompt
from planner.llm import call_llm
from planner.pdf_reader import read_pdf

FIXTURES = Path(__file__).parent / "fixtures"

# fixture name -> (PDF file name, the year the term starts)
SYLLABI = {
    "MAT235H1": ("MAT235H1_Syllabus.pdf", 2026),
    "EAS110Y1": ("EAS110Y1_Syllabus.pdf", 2026),
    "GGR124H1": ("GGR124H1_Syllabus.pdf", 2026),  # winter term, so Jan-Apr dates are in 2026
    "MAT223H1": ("MAT223H1_Syllabus.pdf", 2026),
}


def record(name: str) -> None:
    pdf_name, term_year = SYLLABI[name]
    prompt = build_prompt(read_pdf(FIXTURES / pdf_name), term_year)

    recording = {
        # The hash lets a test notice when the prompt has changed since recording.
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "reply": call_llm(prompt),
    }
    path = FIXTURES / f"{name}_recorded_reply.json"
    path.write_text(json.dumps(recording, indent=2), encoding="utf-8")
    print(f"Saved {path}")


def main() -> None:
    for name in sys.argv[1:] or list(SYLLABI):
        record(name)


if __name__ == "__main__":
    main()
