"""Pure display helper functions for test values and references.

These are intentionally free of streamlit or any UI dependency so they can be
shared by both the client dashboard and the server-side test suite.
"""

from typing import Any, Dict


def format_test_display_value(test: Dict[str, Any]) -> str:
    """Format numeric or qualitative test results without generic fallbacks."""
    val_num = test.get("result_value_numeric") if test.get("result_value_numeric") is not None else test.get("value")
    val_txt = test.get("result_text")
    unit = test.get("unit") or ""

    if val_num is not None:
        return f"{val_num} {unit}".strip()
    if val_txt is not None and str(val_txt).strip():
        return str(val_txt).strip()
    return "-"


def format_test_display_reference(test: Dict[str, Any]) -> str:
    """Format reference interval, showing printed expected text or '-' instead of 'N/A'."""
    ref = test.get("reference_text")
    if ref is not None and str(ref).strip() and str(ref).strip().upper() != "N/A":
        return str(ref).strip()
    low = test.get("reference_range_low") if test.get("reference_range_low") is not None else test.get("reference_low")
    high = test.get("reference_range_high") if test.get("reference_range_high") is not None else test.get("reference_high")
    if low is not None and high is not None:
        return f"{low} - {high}"
    return "-"
