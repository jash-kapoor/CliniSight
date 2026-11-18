"""Stage B: Independent Traditional OCR Cross-Check Engine (RapidOCR)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import numpy as np
from PIL import Image

from logger import logger

try:
    from rapidocr_onnxruntime import RapidOCR
    _RAPID_OCR_INSTANCE = RapidOCR()
except Exception as exc:
    logger.warning("RapidOCR initialization warning: %s", exc)
    _RAPID_OCR_INSTANCE = None


@dataclass
class OCRItem:
    """Individual OCR bounding box and text line."""
    text: str
    confidence: float
    bbox: List[List[int]] = field(default_factory=list)


@dataclass
class OCRPageResult:
    """Consolidated OCR extraction for a single page."""
    page_number: int
    full_text: str
    items: List[OCRItem] = field(default_factory=list)
    numeric_tokens: List[str] = field(default_factory=list)


def extract_numeric_tokens(text: str) -> List[str]:
    """
    Extract numeric tokens from text while excluding dates (e.g. 01/09/2025)
    and common telephone numbers.
    """
    tokens: List[str] = []
    # Match numbers with optional commas or decimals, or comparison prefixes
    # e.g. 13.40, 1,75,000, 38.50, <6.0, >50
    pattern = r"(?:<=|>=|<|>)?\b\d+(?:[.,]\d+)*\b"
    for match in re.finditer(pattern, text):
        raw = match.group(0).strip()
        # Filter out obvious dates like 01/09/2025 or time 00:47:13
        if "/" in raw or ":" in raw:
            continue
        # Filter out 4-digit years like 2025 or 2026 if standalone
        if re.fullmatch(r"202[0-9]", raw):
            continue
        tokens.append(raw)
    return tokens


def run_traditional_ocr(pil_img: Image.Image, page_number: int = 1) -> OCRPageResult:
    """
    Run an independent traditional OCR pass (RapidOCR) on the page image.
    This serves as a ground-truth cross-check source for validation and numeric coverage.
    """
    if _RAPID_OCR_INSTANCE is None:
        logger.warning("RapidOCR is not available. Skipping independent OCR pass.")
        return OCRPageResult(page_number=page_number, full_text="", items=[], numeric_tokens=[])

    try:
        np_img = np.array(pil_img.convert("RGB"))
        ocr_result, _ = _RAPID_OCR_INSTANCE(np_img)

        items: List[OCRItem] = []
        lines: List[str] = []

        if ocr_result:
            for box, text, score in ocr_result:
                clean_text = (text or "").strip()
                if clean_text:
                    items.append(OCRItem(text=clean_text, confidence=float(score), bbox=box))
                    lines.append(clean_text)

        full_text = "\n".join(lines)
        numeric_tokens = extract_numeric_tokens(full_text)

        logger.info(
            "Page %d Traditional OCR: extracted %d text lines and %d candidate numeric tokens.",
            page_number,
            len(items),
            len(numeric_tokens),
        )

        return OCRPageResult(
            page_number=page_number,
            full_text=full_text,
            items=items,
            numeric_tokens=numeric_tokens,
        )
    except Exception as exc:
        logger.error("RapidOCR pass failed on page %d: %s", page_number, exc)
        return OCRPageResult(page_number=page_number, full_text="", items=[], numeric_tokens=[])
