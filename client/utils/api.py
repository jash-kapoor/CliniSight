from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
import requests

try:
    from config import API_URL
except (ImportError, AttributeError):
    try:
        from client.config import API_URL
    except (ImportError, ModuleNotFoundError):
        API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

DEFAULT_TIMEOUT = 300  # Gemini parsing/insight generation may take some seconds


def upload_report_api(file_bytes_or_list: Any, filename: str = "report.pdf") -> Tuple[bool, Any]:
    """
    Upload a single lab report PDF or multiple images to /upload_report/.
    Accepts either (bytes, filename) or a list of (filename, bytes) tuples.
    Returns (True, parsed_response_dict) or (False, error_message).
    """
    try:
        if isinstance(file_bytes_or_list, list):
            files = [("files", (fname, fbytes, "application/octet-stream")) for fname, fbytes in file_bytes_or_list]
        else:
            files = {"file": (filename, file_bytes_or_list, "application/octet-stream")}

        resp = requests.post(
            f"{API_URL}/upload_report/",
            files=files,
            timeout=DEFAULT_TIMEOUT,
        )
        if resp.status_code == 200:
            return True, resp.json()
        
        detail = "Failed to upload report"
        try:
            err_data = resp.json()
            detail = err_data.get("detail") or err_data.get("error") or resp.text
        except Exception:
            detail = resp.text
        return False, detail
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to backend server. Please verify the FastAPI server is running on " + API_URL
    except Exception as exc:
        return False, str(exc)


def get_report_api(report_id: str) -> Tuple[bool, Any]:
    """Retrieve full parsed lab report data by report ID."""
    try:
        resp = requests.get(f"{API_URL}/report/{report_id}", timeout=30)
        if resp.status_code == 200:
            return True, resp.json()
        return False, resp.text
    except Exception as exc:
        return False, str(exc)


def get_report_summary_api(report_id: str) -> Tuple[bool, Any]:
    """
    Retrieve clinical insights summary (plain-language overview, priority findings,
    precautions, Indian diet, and doctor questions).
    """
    try:
        resp = requests.get(f"{API_URL}/report/{report_id}/summary", timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            return True, resp.json()
        return False, resp.text
    except Exception as exc:
        return False, str(exc)


def delete_report_api(report_id: str) -> Tuple[bool, Any]:
    """Delete all vector embeddings, parsed JSON, and PDF files for privacy."""
    try:
        resp = requests.delete(f"{API_URL}/report/{report_id}", timeout=30)
        if resp.status_code == 200:
            return True, resp.json()
        return False, resp.text
    except Exception as exc:
        return False, str(exc)


def ask_question_api(
    question: str,
    report_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[bool, Any]:
    """Send question grounded in report to /ask/ endpoint."""
    try:
        payload = {
            "question": question,
            "report_id": report_id,
            "session_id": session_id,
        }
        resp = requests.post(
            f"{API_URL}/ask/",
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        if resp.status_code == 200:
            return True, resp.json()
        return False, resp.text
    except Exception as exc:
        return False, str(exc)


def clear_chat_history_api(session_id: str) -> Tuple[bool, Any]:
    """Clear conversational memory for a given session."""
    try:
        resp = requests.delete(f"{API_URL}/chat/history/{session_id}", timeout=20)
        if resp.status_code == 200:
            return True, resp.json()
        return False, resp.text
    except Exception as exc:
        return False, str(exc)


def update_report_tests_api(report_id: str, tests: list) -> Tuple[bool, Any]:
    """
    Stage E: Submit updated or newly added tests with deterministic flag recomputation.
    """
    try:
        resp = requests.put(
            f"{API_URL}/report/{report_id}/tests",
            json={"tests": tests},
            timeout=DEFAULT_TIMEOUT,
        )
        if resp.status_code == 200:
            return True, resp.json()
        return False, resp.text
    except Exception as exc:
        return False, str(exc)


def confirm_report_api(
    report_id: str,
    tests: Optional[list] = None,
    acknowledged_incomplete: bool = False,
) -> Tuple[bool, Any]:
    """
    Stage E: Confirm verified lab report data, index vector store, and generate insights.
    """
    try:
        payload: Dict[str, Any] = {"acknowledged_incomplete": acknowledged_incomplete}
        if tests:
            payload["tests"] = tests
        resp = requests.post(
            f"{API_URL}/report/{report_id}/confirm",
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        if resp.status_code == 200:
            return True, resp.json()
        return False, resp.text
    except Exception as exc:
        return False, str(exc)


def get_report_page_image_url(report_id: str, page_number: int, bbox: Optional[str] = None) -> str:
    """Return backend URL for rendered report page image with optional bbox highlight."""
    url = f"{API_URL}/report/{report_id}/page/{page_number}/image"
    if bbox:
        url += f"?bbox={bbox}"
    return url


# Backward compatibility helpers
def upload_pdfs_api(files):
    files_payload = [("files", (f.name, f.read(), "application/pdf")) for f in files]
    return requests.post(f"{API_URL}/upload_pdfs/", files=files_payload, timeout=DEFAULT_TIMEOUT)


def ask_question(question: str):
    return requests.post(f"{API_URL}/ask/", data={"question": question}, timeout=DEFAULT_TIMEOUT)