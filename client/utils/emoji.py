"""Utility to remove emojis and decorative unicode symbols from text and UI data."""

from __future__ import annotations

import re
from typing import Any

# Comprehensive regex matching emojis, emoticons, pictographs, transport symbols,
# geometric shapes used as color badges, variation selectors, zero-width joiners, dingbats
_EMOJI_REGEX = re.compile(
    r"["
    r"\U0001F000-\U0001FAFF"  # Emoticons, Pictographs, Symbols, Transport, Alchemical
    r"\U00002700-\U000027BF"  # Dingbats
    r"\U00002600-\U000026FF"  # Miscellaneous Symbols
    r"\U00002300-\U000023FF"  # Miscellaneous Technical (watch, hourglass, etc.)
    r"\U00002B00-\U00002BFF"  # Miscellaneous Symbols and Arrows
    r"\U000025A0-\U000025FF"  # Geometric shapes
    r"\U0000FE00-\U0000FE0F"  # Variation Selectors
    r"\U0000200D"              # Zero-width joiner
    r"\U000020E3"              # Combining Enclosing Keycap
    r"]+",
    flags=re.UNICODE,
)


def strip_emoji(text: Any) -> str:
    """
    Remove all emojis, decorative symbols, and variation selectors from text.
    Handles None and non-string inputs safely. Collapses redundant spaces.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    cleaned = _EMOJI_REGEX.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def sanitize_ui_data(data: Any) -> Any:
    """
    Recursively strip emojis from nested data structures (dicts, lists, strings)
    coming from the backend before display.
    """
    if isinstance(data, str):
        return strip_emoji(data)
    elif isinstance(data, dict):
        return {k: sanitize_ui_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_ui_data(item) for item in data]
    return data
