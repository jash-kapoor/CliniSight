"""Streamlit version compatibility layer.

Dynamically adapts full-width and container-width parameters across Streamlit versions
(e.g., Streamlit 1.37.x in base Anaconda using use_container_width / use_column_width vs
Streamlit 1.65+ in client/.venv using width='stretch').
"""

import inspect
from typing import Any, Callable, Dict


def get_stretch_kwarg(fn: Callable[..., Any]) -> Dict[str, Any]:
    """Return the correct keyword argument to make a Streamlit widget stretch full width."""
    try:
        params = inspect.signature(fn).parameters
        # In Streamlit 1.50+ / 1.65+, 'width' accepts 'stretch' (string)
        if "width" in params:
            default_val = params["width"].default
            if default_val != inspect._empty and not isinstance(default_val, (int, type(None))):
                return {"width": "stretch"}
        if "use_container_width" in params:
            return {"use_container_width": True}
        if "use_column_width" in params:
            return {"use_column_width": True}
    except Exception:
        pass
    return {}
