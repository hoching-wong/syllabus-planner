"""Save the model's reply for the MAT235 fixture so tests can replay it offline.

Run from the project root:   python -m tests.record_reply

It goes through call_llm, so it only spends API quota if the prompt (or the
PDF text) has changed since the last real call; otherwise the disk cache answers.
Re-run it after changing the prompt or the PDF reader, then check test_accuracy.py.
"""

import hashlib
import json
from pathlib import Path

from planner.extractor import build_prompt
from planner.llm import call_llm
from planner.pdf_reader import read_pdf

FIXTURES = Path(__file__).parent / "fixtures"


def main() -> None:
    key = json.loads((FIXTURES / "MAT235H1_expected.json").read_text(encoding="utf-8"))
    prompt = build_prompt(read_pdf(FIXTURES / "MAT235H1_Syllabus.pdf"), key["default_year"])

    recording = {
        # The hash lets a test notice when the prompt has changed since recording.
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "reply": call_llm(prompt),
    }
    path = FIXTURES / "MAT235H1_recorded_reply.json"
    path.write_text(json.dumps(recording, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
