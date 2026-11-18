"""
Record golden fixtures from a successful Report.pdf parse.
Captures the raw JSON responses from _call_gemini_multimodal_vision for each page,
so future tests can replay them without network access.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

server_dir = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, str(server_dir))

from modules.lab_parser import parse_lab_report_inputs

REPORT_PDF = Path(__file__).resolve().parent.parent / "Report.pdf"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

# We'll monkey-patch _call_gemini_multimodal_vision to record responses
recorded_calls = []
original_fn = None

def recording_wrapper(image_bytes, prompt, api_key, mime_type="image/jpeg"):
    """Call the real function and record the response."""
    result = original_fn(image_bytes, prompt, api_key, mime_type)
    recorded_calls.append({
        "prompt_snippet": prompt[:200],
        "response": result,
    })
    return result


def main():
    global original_fn

    import modules.lab_parser as lp
    original_fn = lp._call_gemini_multimodal_vision

    with patch.object(lp, "_call_gemini_multimodal_vision", side_effect=recording_wrapper):
        report_data, summary = parse_lab_report_inputs(
            inputs=[(REPORT_PDF, "Report.pdf")],
            report_id="fixture_recording",
            original_filename="Report.pdf",
        )

    # Save vision call responses as fixtures
    for idx, call in enumerate(recorded_calls):
        fixture_path = FIXTURE_DIR / f"vision_call_{idx}.json"
        with open(fixture_path, "w") as f:
            json.dump(call["response"], f, indent=2, ensure_ascii=False)
        print(f"Saved fixture {fixture_path} ({len(json.dumps(call['response']))} bytes)")

    # Also save the full parsed report data as a reference
    report_path = FIXTURE_DIR / "report_pdf_parsed.json"
    with open(report_path, "w") as f:
        json.dump(report_data.model_dump(), f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved parsed report fixture to {report_path}")
    print(f"Total tests: {len(report_data.tests)}")
    print(f"Abnormal: {summary.abnormal_count}")
    print(f"Total vision calls recorded: {len(recorded_calls)}")


if __name__ == "__main__":
    main()
